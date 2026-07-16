from __future__ import annotations

from types import SimpleNamespace

from boundary.ja.model import SPEECH_ISLAND_SCORER_LABELS
from tools.boundary.ja.build_inner_edge_refiner_v1_outer_distilled_dataset import (
    labels_from_prediction,
)


def test_outer_teacher_span_becomes_argmax_frame_targets() -> None:
    labels, decision = labels_from_prediction(
        SimpleNamespace(
            start_action="refined",
            end_action="refined",
            start_s=0.04,
            end_s=0.10,
        ),
        total_frames=8,
        frame_hop_s=0.02,
    )

    discardable = SPEECH_ISLAND_SCORER_LABELS.index("discardable")
    target = SPEECH_ISLAND_SCORER_LABELS.index("semantic_target")
    assert decision == "teacher_refined"
    assert labels.tolist() == [
        discardable,
        discardable,
        target,
        target,
        target,
        discardable,
        discardable,
        discardable,
    ]


def test_outer_teacher_abstain_stays_unsure_without_fallback() -> None:
    labels, decision = labels_from_prediction(
        SimpleNamespace(
            start_action="abstain",
            end_action="refined",
            start_s=0.0,
            end_s=0.10,
        ),
        total_frames=5,
        frame_hop_s=0.02,
    )

    unsure = SPEECH_ISLAND_SCORER_LABELS.index("unsure")
    assert decision == "teacher_abstain"
    assert labels.tolist() == [unsure] * 5
