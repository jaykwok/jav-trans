from __future__ import annotations

import pytest

from boundary.features import make_feature_bundle
from boundary.planner import BoundaryPlannerConfig, plan_boundary_chunks
from boundary.refiner import BoundaryDecision
from boundary.base import SpeechSegment


def test_boundary_planner_uses_candidate_split_before_packing():
    features = make_feature_bundle(
        frame_hop_s=1.0,
        cut_scores=[0.05] * 5 + [0.98] * 2 + [0.05] * 5,
    )

    chunks = plan_boundary_chunks(
        [SpeechSegment(0.0, 12.0)],
        features=features,
        config=BoundaryPlannerConfig(
            frame_hop_s=1.0,
            max_chunk_s=30.0,
            target_chunk_s=5.0,
            min_chunk_s=3.0,
            target_padding_s=1.0,
        ),
    )

    assert len(chunks) == 2
    assert chunks[0].islands[0].end == pytest.approx(6.0)
    assert chunks[1].islands[0].start == pytest.approx(6.0)
    assert chunks[0].split_reason == "cut_candidate"


def test_boundary_planner_validates_config():
    features = make_feature_bundle(frame_hop_s=1.0)

    with pytest.raises(ValueError, match="max_chunk_s"):
        plan_boundary_chunks(
            [SpeechSegment(0.0, 1.0)],
            features=features,
            config=BoundaryPlannerConfig(max_chunk_s=0.0),
        )


class _StaticSequenceRefiner:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[list[list[float]]] = []
        self.cursor = 0

    def decide_sequence(self, features: list[list[float]]) -> list[BoundaryDecision]:
        self.calls.append(features)
        scores = self.scores[self.cursor : self.cursor + len(features)]
        self.cursor += len(features)
        return [
            BoundaryDecision(
                score >= 0.5,
                score,
                "learned_sequence_merge" if score >= 0.5 else "learned_sequence_split",
                source="frame_sequence_refiner",
            )
            for score in scores
        ]

    def signature(self) -> dict:
        return {"type": "static_sequence_refiner"}


class _IndexFeatureProvider:
    def features_for_gap(
        self,
        *,
        left_start_s: float,
        left_end_s: float,
        right_start_s: float,
        right_end_s: float,
    ) -> list[float]:
        return [right_start_s - left_end_s]


def test_sequence_planner_dp_can_split_high_merge_gap_for_duration_target():
    features = make_feature_bundle(frame_hop_s=1.0)
    refiner = _StaticSequenceRefiner([0.9, 0.9, 0.9])

    chunks = plan_boundary_chunks(
        [
            SpeechSegment(0.0, 3.0),
            SpeechSegment(3.1, 6.0),
            SpeechSegment(6.1, 9.0),
            SpeechSegment(9.1, 12.0),
        ],
        features=features,
        config=BoundaryPlannerConfig(
            frame_hop_s=1.0,
            max_chunk_s=30.0,
            target_chunk_s=6.0,
            min_chunk_s=0.4,
            target_padding_s=1.0,
            start_weight=1.5,
        ),
        sequence_refiner=refiner,
        sequence_feature_provider=_IndexFeatureProvider(),
    )

    assert len(refiner.calls) == 1
    assert [(chunk.islands[0].start, chunk.islands[-1].end) for chunk in chunks] == [
        (0.0, 6.0),
        (6.1, 12.0),
    ]
    assert chunks[0].split_reason == "planner_dp"
    assert chunks[0].boundary_decision is not None
    assert chunks[0].boundary_decision.merge is False
    assert chunks[0].boundary_decision.score == pytest.approx(0.9)
    assert chunks[0].boundary_decision.source == "boundary_planner"


def test_sequence_planner_dp_preserves_learned_split_reason():
    features = make_feature_bundle(frame_hop_s=1.0)
    refiner = _StaticSequenceRefiner([0.95, 0.05])

    chunks = plan_boundary_chunks(
        [
            SpeechSegment(0.0, 1.0),
            SpeechSegment(1.1, 2.0),
            SpeechSegment(2.1, 3.0),
        ],
        features=features,
        config=BoundaryPlannerConfig(
            frame_hop_s=1.0,
            max_chunk_s=30.0,
            target_chunk_s=9.0,
            target_padding_s=1.0,
        ),
        sequence_refiner=refiner,
        sequence_feature_provider=_IndexFeatureProvider(),
    )

    assert [(chunk.islands[0].start, chunk.islands[-1].end) for chunk in chunks] == [
        (0.0, 2.0),
        (2.1, 3.0),
    ]
    assert chunks[0].split_reason == "boundary_refiner:learned_sequence_split"
    assert chunks[0].boundary_decision is not None
    assert chunks[0].boundary_decision.source == "frame_sequence_refiner"


def test_sequence_planner_batches_long_sequences():
    features = make_feature_bundle(frame_hop_s=1.0)
    refiner = _StaticSequenceRefiner([0.05, 0.05, 0.05])

    chunks = plan_boundary_chunks(
        [
            SpeechSegment(0.0, 1.0),
            SpeechSegment(1.1, 2.0),
            SpeechSegment(2.1, 3.0),
            SpeechSegment(3.1, 4.0),
        ],
        features=features,
        config=BoundaryPlannerConfig(
            frame_hop_s=1.0,
            max_chunk_s=30.0,
            target_chunk_s=9.0,
            target_padding_s=1.0,
            sequence_batch_size=2,
        ),
        sequence_refiner=refiner,
        sequence_feature_provider=_IndexFeatureProvider(),
    )

    assert [len(call) for call in refiner.calls] == [2, 1]
    assert [(chunk.islands[0].start, chunk.islands[-1].end) for chunk in chunks] == [
        (0.0, 1.0),
        (1.1, 2.0),
        (2.1, 3.0),
        (3.1, 4.0),
    ]
