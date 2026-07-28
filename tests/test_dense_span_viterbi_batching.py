"""The Viterbi walk is now shared across a batch; pin what sharing could break.

The DP over `end` is sequential and cannot be parallelised - a segment ending
at a frame may start anywhere before it - so the speedup came from removing the
two loops that multiplied it (one per lattice, one per label) and from moving
the remaining walk to the host, where a step costs what its ~8K values weigh
instead of a device launch.

Every one of those changes is a chance for lattices to contaminate each other:

  * one walk now serves items of different lengths, so a long item takes steps
    a short one should not see;
  * `structured_hinge` pads supervised runs to a common width and chunks them
    by a memory budget, so a run's result must not depend on which runs it was
    batched with;
  * the per-step mask that used to forbid a start at or beyond the current end
    is gone, replaced by the lattice's own sub-diagonal `-inf`.

These tests compare batched results against the same lattices decoded alone,
which is the property that makes the optimisation invisible to callers.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

torch = pytest.importorskip("torch")

from boundary import dense_span as dense_span_module  # noqa: E402
from boundary.dense_span import BinaryDenseSpanDecoder  # noqa: E402


def build(seed: int = 0, hidden: int = 16):
    torch.manual_seed(seed)
    decoder = BinaryDenseSpanDecoder(
        hidden_size=hidden,
        rank=4,
        duration_hidden_size=8,
        context_window_frames=256,
    )
    with torch.no_grad():
        # A zero-initialised decoder makes start/end/transition scores vanish,
        # which is exactly the case where batching bugs stay invisible.
        decoder.transitions.normal_(0.0, 0.7)
        decoder.start_scores.normal_(0.0, 0.7)
        decoder.end_scores.normal_(0.0, 0.7)
        decoder.span_residual_gate.fill_(0.8)
    return decoder


def lattice(decoder, batch: int, frames: int, seed: int = 1, mask=None):
    torch.manual_seed(seed)
    encoded = torch.randn(batch, frames, decoder.hidden_size)
    frame_logits = torch.randn(batch, frames, 2)
    if mask is None:
        mask = torch.ones(batch, frames, dtype=torch.bool)
    return decoder.span_scores(encoded, frame_logits, mask), mask


def test_batched_decode_matches_decoding_each_item_alone() -> None:
    """The property the whole rewrite rests on."""
    decoder = build()
    scores, mask = lattice(decoder, batch=5, frames=37)
    together = decoder.decode(scores, mask)
    for index in range(5):
        alone = decoder.decode(scores[index : index + 1], mask[index : index + 1])
        assert torch.equal(together[index], alone[0])


def test_a_short_item_is_unaffected_by_a_long_one() -> None:
    """Steps taken for the longest item must not leak into shorter ones."""
    decoder = build(seed=3)
    frames = 40
    mask = torch.ones(3, frames, dtype=torch.bool)
    for index, length in enumerate((40, 13, 2)):
        mask[index, length:] = False
    scores, _ = lattice(decoder, batch=3, frames=frames, mask=mask)
    batched = decoder.decode(scores, mask)
    for index, length in enumerate((40, 13, 2)):
        alone = decoder.decode(scores[index : index + 1], mask[index : index + 1])
        assert torch.equal(batched[index], alone[0])
        # nothing may be predicted past the prefix
        assert not batched[index, length:].any()


def test_single_lattice_helper_agrees_with_decode() -> None:
    decoder = build(seed=5)
    scores, mask = lattice(decoder, batch=1, frames=29)
    path = decoder._viterbi_path(scores[0])
    expected = torch.zeros(29, dtype=torch.long)
    for start, end, label in path.segments:
        expected[start:end] = label
    assert torch.equal(decoder.decode(scores, mask)[0], expected)


def test_decode_survives_a_lattice_without_the_triangular_mask() -> None:
    """The per-step mask is gone, so the DP must impose the constraint itself.

    A start at or beyond the current end is not a span. `span_scores` already
    fills those cells with -inf, but the DP no longer depends on that.
    """
    decoder = build(seed=7)
    scores, mask = lattice(decoder, batch=2, frames=24)
    finite = torch.where(torch.isinf(scores), torch.full_like(scores, 5.0), scores)
    assert torch.isfinite(finite).all()
    assert torch.equal(decoder.decode(finite, mask), decoder.decode(scores, mask))


@pytest.mark.parametrize("budget", [1024, 8192, 64 * 1024 * 1024])
def test_hinge_is_independent_of_the_chunking_budget(
    budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs are padded and chunked for memory; that must not move the loss."""
    decoder = build(seed=11)
    frames = 32
    scores, _ = lattice(decoder, batch=2, frames=frames)
    labels = (torch.rand(2, frames) > 0.5).long()
    supervision = torch.zeros(2, frames, dtype=torch.bool)
    supervision[0, 0:20] = True
    supervision[0, 24:32] = True
    supervision[1, 3:9] = True
    monkeypatch.setattr(dense_span_module, "_LATTICE_BUDGET", budget)
    value = decoder.structured_hinge(scores, labels, supervision)
    assert torch.isfinite(value)
    monkeypatch.setattr(dense_span_module, "_LATTICE_BUDGET", 64 * 1024 * 1024)
    assert torch.allclose(
        value, decoder.structured_hinge(scores, labels, supervision)
    )


def test_hinge_still_flows_gradient_to_the_decoder() -> None:
    """The DP runs under no_grad; the score recomputation is what must carry it."""
    decoder = build(seed=13)
    frames = 24
    scores, _ = lattice(decoder, batch=1, frames=frames)
    labels = (torch.rand(1, frames) > 0.5).long()
    supervision = torch.ones(1, frames, dtype=torch.bool)
    decoder.zero_grad()
    decoder.structured_hinge(scores, labels, supervision).backward()
    assert decoder.transitions.grad is not None
    assert torch.isfinite(decoder.transitions.grad).all()


def test_loss_augmented_path_counts_its_own_hamming_distance() -> None:
    """`hamming_count` is the margin, so it must describe the returned path."""
    decoder = build(seed=17)
    frames = 26
    scores, _ = lattice(decoder, batch=1, frames=frames)
    labels = (torch.rand(frames) > 0.5).long()
    path = decoder._viterbi_path(scores[0], labels=labels)
    expected = sum(
        int((labels[start:end] != label).sum())
        for start, end, label in path.segments
    )
    assert path.hamming_count == expected


def test_gold_labels_are_recovered_when_the_lattice_says_so() -> None:
    """A sanity anchor that survives any amount of vectorising."""
    decoder = build(seed=19)
    frames = 20
    truth = torch.zeros(1, frames, dtype=torch.long)
    truth[0, 6:14] = 1
    mask = torch.ones(1, frames, dtype=torch.bool)
    encoded = torch.zeros(1, frames, decoder.hidden_size)
    frame_logits = torch.zeros(1, frames, 2)
    frame_logits[0, torch.arange(frames), truth[0]] = 20.0
    with torch.no_grad():
        decoder.span_residual_gate.fill_(0.0)
    scores = decoder.span_scores(encoded, frame_logits, mask)
    assert torch.equal(decoder.decode(scores, mask)[0], truth[0])


def test_decode_of_an_empty_prefix_predicts_nothing() -> None:
    decoder = build(seed=23)
    scores, mask = lattice(decoder, batch=2, frames=12)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    scores, _ = lattice(decoder, batch=2, frames=12, mask=mask)
    assert not decoder.decode(scores, mask).any()
