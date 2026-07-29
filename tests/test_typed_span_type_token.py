"""One type token per span is the core of the typed-span model; pin its rules.

Two properties decide whether the span head is measuring what it claims:

  * a span must be cut at the edge of the supervised region, never bridged
    across it. Real windows are only ~55% covered, so a span that silently
    spanned an unlabelled gap would be assigned a type from evidence that was
    never annotated.
  * `unsure` must stay unrepresentable. It is data-level only (-100), so a span
    with no known type is dropped rather than being folded into whichever class
    happens to be nearest - forcing it is exactly the failure the pre-ASR
    label audit found in the teacher itself.

Pooling is pinned separately because it is done by cumsum for speed, and a
cumsum mean that disagreed with a plain mean would corrupt every span feature
without ever raising.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

torch = pytest.importorskip("torch")

from tools.boundary.ja.train_typed_span_falsification import (  # noqa: E402
    IGNORE_INDEX,
    NONSPEECH_TYPE_CLASSES,
    TYPE_CLASSES,
    TYPE_NAMES,
    build_model,
    constant_runs,
    gold_type_spans,
    inverse_frequency_weights,
    pool_spans,
)


def test_constant_runs_are_maximal_and_cover_everything() -> None:
    values = np.array([1, 1, 0, 0, 0, 1, 2, 2])
    runs = constant_runs(values)
    assert runs == [(0, 2), (2, 5), (5, 6), (6, 8)]
    assert runs[0][0] == 0 and runs[-1][1] == values.size
    for (_, end), (start, _) in zip(runs[:-1], runs[1:], strict=True):
        assert end == start


def test_constant_runs_handles_empty() -> None:
    assert constant_runs(np.array([], dtype=np.int64)) == []


def test_gold_spans_split_on_the_speech_boundary() -> None:
    speech = np.array([1, 1, 1, 0, 0])
    types = np.array([0, 0, 0, 1, 1])
    assert gold_type_spans(speech, types) == [(0, 3, 0), (3, 5, 1)]


def test_gold_spans_never_bridge_an_unsupervised_gap() -> None:
    """Two speech runs separated by unlabelled frames must stay two spans."""
    speech = np.array([1, 1, IGNORE_INDEX, IGNORE_INDEX, 1, 1])
    types = np.array([0, 0, IGNORE_INDEX, IGNORE_INDEX, 0, 0])
    spans = gold_type_spans(speech, types)
    assert [(s, e) for s, e, _ in spans] == [(0, 2), (4, 6)]


def test_gold_spans_drop_a_span_with_no_known_type() -> None:
    """`unsure` is not a class - a fully unknown span is dropped, not guessed."""
    speech = np.array([1, 1, 0, 0])
    types = np.array([0, 0, IGNORE_INDEX, IGNORE_INDEX])
    assert gold_type_spans(speech, types) == [(0, 2, 0)]


def test_gold_span_type_is_the_majority_of_known_frames() -> None:
    speech = np.array([0, 0, 0, 0, 0])
    types = np.array([2, 1, 1, IGNORE_INDEX, 1])
    assert gold_type_spans(speech, types) == [(0, 5, 1)]


def test_gold_spans_ignore_unknown_frames_when_voting() -> None:
    speech = np.zeros(6, dtype=np.int64)
    types = np.array([IGNORE_INDEX] * 5 + [2])
    assert gold_type_spans(speech, types) == [(0, 6, 2)]


def test_every_gold_type_is_a_real_class() -> None:
    rng = np.random.default_rng(7)
    speech = rng.integers(0, 2, size=200)
    types = rng.integers(-1, TYPE_CLASSES, size=200)
    types[types < 0] = IGNORE_INDEX
    for _, _, value in gold_type_spans(speech, types):
        assert 0 <= value < TYPE_CLASSES
        assert TYPE_NAMES[value]


def test_pool_spans_mean_matches_a_plain_mean() -> None:
    """The cumsum shortcut must equal the obvious computation, exactly enough."""
    torch.manual_seed(0)
    encoded = torch.randn(40, 8, dtype=torch.float64)
    spans = [(0, 5), (5, 6), (7, 30), (30, 40)]
    pooled = pool_spans(encoded, spans)
    assert pooled.shape == (len(spans), 3 * 8)
    for row, (start, end) in enumerate(spans):
        assert torch.allclose(pooled[row, :8], encoded[start:end].mean(dim=0))
        assert torch.allclose(pooled[row, 8:16], encoded[start])
        assert torch.allclose(pooled[row, 16:], encoded[end - 1])


def test_pool_spans_handles_a_single_frame_span() -> None:
    encoded = torch.randn(5, 4)
    pooled = pool_spans(encoded, [(2, 3)])
    assert torch.allclose(pooled[0, :4], encoded[2])
    assert torch.allclose(pooled[0, 4:8], encoded[2])
    assert torch.allclose(pooled[0, 8:], encoded[2])


def test_pool_spans_empty_is_well_shaped() -> None:
    pooled = pool_spans(torch.randn(5, 4), [])
    assert pooled.shape == (0, 12)


def test_span_head_consumes_pooled_width() -> None:
    model = build_model(16, 24, (1, 2), type_head="span")
    x = torch.randn(1, 32, 16)
    encoded, _ = model.forward_features(x)
    pooled = pool_spans(encoded[0], [(0, 10), (10, 32)])
    assert model.type_span(pooled).shape == (2, TYPE_CLASSES)


def test_frame_head_emits_three_classes_per_frame() -> None:
    model = build_model(16, 24, (1, 2), type_head="frame")
    x = torch.randn(2, 32, 16)
    encoded, _ = model.forward_features(x)
    logits = model.type_frame(encoded.transpose(1, 2)).transpose(1, 2)
    assert logits.shape == (2, 32, TYPE_CLASSES)


def test_type_head_none_adds_no_parameters() -> None:
    """The binary falsification arms must be unchanged by this feature."""
    plain = build_model(16, 24, (1, 2))
    assert not hasattr(plain, "type_frame")
    assert not hasattr(plain, "type_span")
    assert plain.type_head == "none"
    typed = build_model(16, 24, (1, 2), type_head="span")
    assert sum(p.numel() for p in typed.parameters()) > sum(
        p.numel() for p in plain.parameters()
    )


def test_unknown_type_head_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_model(16, 24, (1, 2), type_head="perframe")


def test_inverse_weights_are_inverse_and_mean_one() -> None:
    """Mean 1 is what keeps `--type-weight` meaning the same thing across arms."""
    weights = inverse_frequency_weights(np.array([800, 150, 50]))
    assert weights[0] < weights[1] < weights[2]
    assert weights.mean() == pytest.approx(1.0)
    # inverse-frequency: weight ratio must equal the inverse count ratio
    assert weights[2] / weights[0] == pytest.approx(800 / 50)


def test_inverse_weights_survive_an_absent_class() -> None:
    """A class with zero support must not produce inf or nan."""
    weights = inverse_frequency_weights(np.array([100, 0, 10]))
    assert np.isfinite(weights).all()
    assert weights.mean() == pytest.approx(1.0)


def test_uniform_support_gives_uniform_weights() -> None:
    weights = inverse_frequency_weights(np.array([7, 7, 7]))
    assert weights == pytest.approx(np.ones(3))


def test_conditional_head_emits_only_the_two_non_speech_kinds() -> None:
    """`span_cond` must not be able to predict speech at all.

    Speech is already decided by the segmentation, and the unconditional span
    head collapsed onto it (non_semantic_vocal recall 2.3%). Removing the class
    from the head's output space is what makes that collapse unrepresentable
    rather than merely unlikely.
    """
    model = build_model(16, 24, (1, 2), type_head="span_cond")
    assert model.type_span.out_features == NONSPEECH_TYPE_CLASSES
    encoded, _ = model.forward_features(torch.randn(1, 32, 16))
    logits = model.type_span(pool_spans(encoded[0], [(0, 12), (12, 32)]))
    assert logits.shape == (2, NONSPEECH_TYPE_CLASSES)
    assert set((1 + logits.argmax(dim=-1)).tolist()) <= {1, 2}


def test_conditional_targets_shift_off_the_speech_class() -> None:
    """Gold types {1,2} must map onto head indices {0,1} with no collisions."""
    speech = np.array([0, 0, 1, 1, 0, 0])
    types = np.array([1, 1, 0, 0, 2, 2])
    spans = gold_type_spans(speech, types)
    non_speech = [(s, e, t) for s, e, t in spans if t != 0]
    assert [t for _, _, t in non_speech] == [1, 2]
    shifted = [t - 1 for _, _, t in non_speech]
    assert shifted == [0, 1]
    assert all(0 <= v < NONSPEECH_TYPE_CLASSES for v in shifted)


def test_conditional_projection_types_speech_spans_by_construction() -> None:
    """A predicted-speech span must come out as speech without asking the head."""
    predicted = np.array([1] * 10 + [0] * 6 + [1] * 4)
    projected = np.full(predicted.size, IGNORE_INDEX, dtype=np.int64)
    spans = constant_runs(predicted)
    for start, end in spans:
        if predicted[start] == 1:
            projected[start:end] = 0
    non_speech = [(s, e) for s, e in spans if predicted[s] == 0]
    assert non_speech == [(10, 16)]
    for start, end in non_speech:
        projected[start:end] = 1  # stand-in for the head's decision
    assert projected[:10].tolist() == [0] * 10
    assert projected[10:16].tolist() == [1] * 6
    assert projected[16:].tolist() == [0] * 4
    assert (projected != IGNORE_INDEX).all()


def test_span_typing_is_constant_within_a_span() -> None:
    """Projecting a span token back to frames must not vary inside the span.

    That constancy is the whole difference from a frame-wise head, and it is
    what makes the two comparable on one frame-level axis.
    """
    model = build_model(16, 24, (1, 2), type_head="span")
    x = torch.randn(1, 40, 16)
    encoded, _ = model.forward_features(x)
    predicted = np.array([1] * 12 + [0] * 8 + [1] * 20)
    spans = constant_runs(predicted)
    assigned = model.type_span(pool_spans(encoded[0], spans)).argmax(dim=-1)
    projected = np.full(40, IGNORE_INDEX, dtype=np.int64)
    for (start, end), value in zip(spans, assigned.tolist(), strict=True):
        projected[start:end] = int(value)
    assert (projected != IGNORE_INDEX).all()
    for start, end in spans:
        assert len(set(projected[start:end].tolist())) == 1


def test_auxiliary_targets_mask_speech_and_unknown_frames() -> None:
    """The aux head answers `span_cond`'s question, so it must share its blindness.

    Speech is already settled by the segmentation. Letting the majority class
    into a per-frame target is exactly what collapsed the unconditional span
    head, and the aux head supervises far more frames, so it would collapse
    harder.
    """
    from tools.boundary.ja.train_typed_span_falsification import (
        nonspeech_frame_targets,
    )

    types = np.array([0, 1, 1, 2, IGNORE_INDEX, 2, 0])
    assert nonspeech_frame_targets(types).tolist() == [
        IGNORE_INDEX, 0, 0, 1, IGNORE_INDEX, 1, IGNORE_INDEX
    ]


def test_auxiliary_targets_stay_inside_the_head_output_space() -> None:
    from tools.boundary.ja.train_typed_span_falsification import (
        nonspeech_frame_targets,
    )

    rng = np.random.default_rng(3)
    types = rng.integers(-1, TYPE_CLASSES, size=500)
    types[types < 0] = IGNORE_INDEX
    targets = nonspeech_frame_targets(types)
    known = targets != IGNORE_INDEX
    assert ((targets[known] >= 0) & (targets[known] < NONSPEECH_TYPE_CLASSES)).all()
    # supervised exactly where the truth is one of the two non-speech kinds
    assert np.array_equal(known, (types == 1) | (types == 2))


def test_auxiliary_head_is_absent_unless_asked_for() -> None:
    """It is training-only scaffolding; the default arms must be unchanged."""
    plain = build_model(16, 24, (1, 2), type_head="span_cond")
    assert not hasattr(plain, "type_frame_aux")
    assert plain.aux_frame_type is False


def test_auxiliary_head_emits_two_classes_per_frame() -> None:
    model = build_model(16, 24, (1, 2), type_head="span_cond", aux_frame_type=True)
    encoded, _ = model.forward_features(torch.randn(2, 40, 16))
    logits = model.type_frame_aux(encoded.transpose(1, 2))
    assert logits.shape == (2, NONSPEECH_TYPE_CLASSES, 40)


def test_auxiliary_head_refuses_to_supplement_a_frame_head() -> None:
    """Supplementing the frame head with a frame head would measure nothing."""
    with pytest.raises(ValueError):
        build_model(16, 24, (1, 2), type_head="frame", aux_frame_type=True)
    with pytest.raises(ValueError):
        build_model(16, 24, (1, 2), type_head="none", aux_frame_type=True)


def test_modality_mask_zeroes_only_the_other_block() -> None:
    """Ablation must remove information without changing anything else.

    Zeroing rather than slicing keeps width, parameter count and normalisation
    identical across arms, so a difference between them can only come from
    which information reached the stem.
    """
    from tools.boundary.ja.train_typed_span_falsification import apply_modality_mask

    features = np.arange(2 * 3 * 6, dtype=np.float32).reshape(2, 3, 6) + 1
    ptm_only = apply_modality_mask(features, 4, "ptm")
    mfcc_only = apply_modality_mask(features, 4, "mfcc")
    assert np.array_equal(ptm_only[..., :4], features[..., :4])
    assert not ptm_only[..., 4:].any()
    assert not mfcc_only[..., :4].any()
    assert np.array_equal(mfcc_only[..., 4:], features[..., 4:])


def test_modality_mask_leaves_both_untouched_and_does_not_alias() -> None:
    from tools.boundary.ja.train_typed_span_falsification import apply_modality_mask

    features = np.ones((2, 4), dtype=np.float32)
    assert apply_modality_mask(features, 2, "both") is features
    masked = apply_modality_mask(features, 2, "ptm")
    assert features.all(), "the caller's array must not be modified in place"
    assert not masked[..., 2:].any()


@pytest.mark.parametrize("modality", ["ptm_only", "", "PTM", "none"])
def test_unknown_modality_is_rejected(modality: str) -> None:
    from tools.boundary.ja.train_typed_span_falsification import apply_modality_mask

    with pytest.raises(ValueError):
        apply_modality_mask(np.ones((2, 4), dtype=np.float32), 2, modality)


def test_modality_mask_rejects_a_split_it_cannot_make() -> None:
    """A ptm_dim covering the whole width would silently ablate nothing."""
    from tools.boundary.ja.train_typed_span_falsification import apply_modality_mask

    features = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_modality_mask(features, 4, "ptm")
    with pytest.raises(ValueError):
        apply_modality_mask(features, 0, "mfcc")


class _TypeTrackStore:
    """Minimal FrameStore stand-in: only the type track and a constant read."""

    width = 4
    ptm_dim = 2

    def __init__(self, type_track: np.ndarray) -> None:
        self.type = type_track
        self.speech = np.zeros_like(type_track)

    def read(self, start: int, count: int) -> np.ndarray:
        return np.ones((count, self.width), dtype=np.float32)


def test_rare_windows_are_exactly_those_holding_the_class() -> None:
    from tools.boundary.ja.train_typed_span_falsification import (
        RARE_TYPE_INDEX,
        windows_containing_type,
    )

    store = _TypeTrackStore(np.array([2, 1, 0, 0, 1, 1, 0, 0, 0, 0, 2, 0]))
    rows = [
        {"frame_offset": 0, "frame_count": 4},   # holds a non_vocal frame
        {"frame_offset": 4, "frame_count": 4},   # does not
        {"frame_offset": 8, "frame_count": 4},   # holds one
    ]
    selected = windows_containing_type(store, rows, RARE_TYPE_INDEX)
    assert [r["frame_offset"] for r in selected] == [0, 8]


def test_oversampling_changes_which_windows_are_drawn() -> None:
    """The whole point: at fraction 1.0 every slot must carry the class.

    Class weighting was already tried and hurt. It scales the gradient of
    whatever was drawn, which does nothing on the ~80% of steps that contain no
    non_vocal frame at all - this lever is what changes that.
    """
    from tools.boundary.ja.train_typed_span_falsification import (
        RARE_TYPE_INDEX,
        sample_batch,
    )

    store = _TypeTrackStore(np.array([2, 1, 1, 0, 1, 1, 0, 0]))
    rows = [
        {"frame_offset": 0, "frame_count": 4},
        {"frame_offset": 4, "frame_count": 4},
    ]
    pools = [(rows, 1.0)]
    rare = [[rows[0]]]
    mean = np.zeros(4, dtype=np.float32)
    std = np.ones(4, dtype=np.float32)

    def rate(fraction: float) -> float:
        rng = np.random.default_rng(0)
        hits = 0
        for _ in range(200):
            _, _, types = sample_batch(
                store, pools, batch_size=1, window=4, rng=rng, mean=mean, std=std,
                rare_pools=rare, rare_fraction=fraction,
            )
            hits += int((types == RARE_TYPE_INDEX).any())
        return hits / 200

    assert rate(1.0) == 1.0
    assert 0.3 < rate(0.0) < 0.7


def test_oversampling_falls_back_when_no_window_holds_the_class() -> None:
    """Synthetic pools carry no non-speech type at all, so this must not divide
    by zero or raise - it has to degrade to ordinary sampling."""
    from tools.boundary.ja.train_typed_span_falsification import sample_batch

    store = _TypeTrackStore(np.zeros(8, dtype=np.int64))
    rows = [
        {"frame_offset": 0, "frame_count": 4},
        {"frame_offset": 4, "frame_count": 4},
    ]
    features, labels, types = sample_batch(
        store, [(rows, 1.0)], batch_size=3, window=4,
        rng=np.random.default_rng(0),
        mean=np.zeros(4, dtype=np.float32), std=np.ones(4, dtype=np.float32),
        rare_pools=[[]], rare_fraction=1.0,
    )
    assert features.shape == (3, 4, 4)
    assert labels.shape == types.shape == (3, 4)
