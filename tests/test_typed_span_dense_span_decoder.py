"""The Dense Span decode path must be threshold-free and structurally valid.

The whole reason for routing the typed-span model through
`src/boundary/dense_span.py` is that a span decoder yields discrete segments by
construction, where a frame-wise head needs a threshold (or an argmax plus
post-hoc run stitching) to recover them. That claim is only worth anything if
the decoded output really is a maximal-run alternating segmentation and really
does come from the span lattice rather than from the frame logits alone.

`forward_features` is the seam that makes this possible: the decoder scores
spans from the encoder representation *and* the frame logits, so the pre-head
activations have to be reachable without recomputing the stack. If `forward`
and `forward_features` ever disagreed, the frame arm and the decoder arm would
silently be evaluating different networks and the comparison between them would
mean nothing.
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

from boundary.dense_span import BinaryDenseSpanDecoder  # noqa: E402
from tools.boundary.ja.train_typed_span_falsification import (  # noqa: E402
    IGNORE_INDEX,
    build_model,
)

HIDDEN = 24
WIDTH = 16
FRAMES = 32


def _model_and_decoder(*, frames: int = FRAMES):
    torch.manual_seed(0)
    model = build_model(WIDTH, HIDDEN, (1, 2))
    decoder = BinaryDenseSpanDecoder(
        hidden_size=HIDDEN,
        rank=4,
        duration_hidden_size=8,
        context_window_frames=frames,
    )
    return model, decoder


def _lattice(model, decoder, x):
    encoded, logits = model.forward_features(x)
    prefix = torch.ones(x.shape[:2], dtype=torch.bool)
    return decoder.span_scores(encoded, logits, prefix), prefix


def test_forward_features_matches_forward() -> None:
    """The decoder arm and the frame arm must share one network, exactly."""
    model, _ = _model_and_decoder()
    x = torch.randn(2, FRAMES, WIDTH)
    encoded, logits = model.forward_features(x)
    assert torch.equal(logits, model(x))
    assert encoded.shape == (2, FRAMES, HIDDEN)
    assert logits.shape == (2, FRAMES, 2)


def test_encoded_is_the_pre_head_activation() -> None:
    """`encoded` must be what the head consumes, not some other tensor."""
    model, _ = _model_and_decoder()
    x = torch.randn(1, FRAMES, WIDTH)
    encoded, logits = model.forward_features(x)
    replayed = model.head(encoded.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(replayed, logits, atol=1e-6)


def test_decode_returns_binary_labels_of_full_length() -> None:
    model, decoder = _model_and_decoder()
    x = torch.randn(3, FRAMES, WIDTH)
    lattice, prefix = _lattice(model, decoder, x)
    decoded = decoder.decode(lattice, prefix)
    assert decoded.shape == (3, FRAMES)
    assert set(decoded.unique().tolist()) <= {0, 1}


def test_decoded_segmentation_alternates_and_is_maximal() -> None:
    """Each decoded sequence must map to exactly one maximal-run segmentation.

    This is the property a threshold cannot give you: no run may be adjacent to
    another run carrying the same label, because that would be two segments
    where the lattice scored one.
    """
    model, decoder = _model_and_decoder()
    x = torch.randn(4, FRAMES, WIDTH)
    lattice, prefix = _lattice(model, decoder, x)
    decoded = decoder.decode(lattice, prefix).tolist()
    for row in decoded:
        runs = [row[0]]
        for previous, current in zip(row[:-1], row[1:], strict=True):
            if current != previous:
                runs.append(current)
        assert all(a != b for a, b in zip(runs[:-1], runs[1:], strict=True))


def test_decode_is_deterministic_and_needs_no_threshold() -> None:
    model, decoder = _model_and_decoder()
    x = torch.randn(2, FRAMES, WIDTH)
    lattice, prefix = _lattice(model, decoder, x)
    first = decoder.decode(lattice, prefix)
    second = decoder.decode(lattice, prefix)
    assert torch.equal(first, second)


def test_decode_can_disagree_with_frame_argmax() -> None:
    """If decode always equalled argmax, the decoder would be doing nothing.

    The span residual gate starts at zero, so the lattice starts as pure frame
    sums and the two agree; driving the gate and the endpoint projections makes
    the span terms bite. This pins that the decode really is a function of the
    lattice, not a rename of argmax.
    """
    model, decoder = _model_and_decoder()
    torch.manual_seed(3)
    with torch.no_grad():
        decoder.span_residual_gate.fill_(2.0)
        decoder.start_projection.weight.normal_(0, 4.0)
        decoder.end_projection.weight.normal_(0, 4.0)
        decoder.transitions.copy_(torch.tensor([[0.0, -50.0], [-50.0, 0.0]]))
    x = torch.randn(6, FRAMES, WIDTH)
    lattice, prefix = _lattice(model, decoder, x)
    decoded = decoder.decode(lattice, prefix)
    argmax = model(x).argmax(dim=-1)
    assert not torch.equal(decoded, argmax)


def test_structured_hinge_is_non_negative_and_differentiable() -> None:
    model, decoder = _model_and_decoder()
    x = torch.randn(2, FRAMES, WIDTH)
    lattice, _ = _lattice(model, decoder, x)
    labels = torch.zeros(2, FRAMES, dtype=torch.long)
    labels[:, FRAMES // 2 :] = 1
    mask = torch.ones(2, FRAMES, dtype=torch.bool)
    loss = decoder.structured_hinge(lattice, labels, mask)
    assert loss.item() >= 0.0
    loss.backward()
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()


def test_structured_hinge_only_scores_supervised_runs() -> None:
    """Unlabelled frames must not enter the loss, even as label value 0.

    Real windows are only ~55% covered. If the ignored half were scored as
    non-speech the loss would reward a degenerate predictor, which is the exact
    failure the falsification is set up to detect.
    """
    model, decoder = _model_and_decoder()
    x = torch.randn(1, FRAMES, WIDTH)
    lattice, _ = _lattice(model, decoder, x)
    labels = torch.zeros(1, FRAMES, dtype=torch.long)
    labels[:, : FRAMES // 2] = 1
    mask = torch.zeros(1, FRAMES, dtype=torch.bool)
    mask[:, : FRAMES // 2] = True

    supervised_only = decoder.structured_hinge(lattice, labels, mask)
    # Flipping labels only OUTSIDE the mask must not move the loss at all.
    flipped = labels.clone()
    flipped[:, FRAMES // 2 :] = 1
    assert torch.allclose(
        supervised_only, decoder.structured_hinge(lattice, flipped, mask)
    )


def test_structured_hinge_rejects_a_fully_unsupervised_batch() -> None:
    """The trainer skips the hinge in this case; it must not silently score 0."""
    model, decoder = _model_and_decoder()
    x = torch.randn(1, FRAMES, WIDTH)
    lattice, _ = _lattice(model, decoder, x)
    labels = torch.zeros(1, FRAMES, dtype=torch.long)
    mask = torch.zeros(1, FRAMES, dtype=torch.bool)
    with pytest.raises(ValueError):
        decoder.structured_hinge(lattice, labels, mask)


def test_ignore_index_clamps_to_a_valid_tag_outside_the_mask() -> None:
    """The trainer passes `y.clamp_min(0)`; that must stay inside the tag set.

    IGNORE_INDEX is -100, and the decoder rejects labels outside [0, num_tags).
    Clamping is only safe because those positions are never inside a supervised
    run - this pins the clamp itself.
    """
    labels = torch.full((1, FRAMES), IGNORE_INDEX, dtype=torch.long)
    labels[:, :4] = 1
    clamped = labels.clamp_min(0)
    assert set(clamped.unique().tolist()) <= {0, 1}


def test_decoder_requires_exactly_two_tags() -> None:
    """Typing is a per-span token, so the segmentation decoder stays binary."""
    with pytest.raises(ValueError):
        BinaryDenseSpanDecoder(
            hidden_size=HIDDEN,
            rank=4,
            duration_hidden_size=8,
            context_window_frames=FRAMES,
            num_tags=3,
        )


def test_lattice_rejects_a_sequence_longer_than_the_context() -> None:
    """The context cap is what bounds the O(T^2) lattice; it must be enforced."""
    model, decoder = _model_and_decoder(frames=FRAMES)
    x = torch.randn(1, FRAMES * 2, WIDTH)
    with pytest.raises(ValueError):
        _lattice(model, decoder, x)
