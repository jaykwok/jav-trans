from __future__ import annotations

import pytest

from boundary.planner import BoundaryPlannerConfig, plan_boundary_chunks
from boundary.refiner import BoundaryDecision
from boundary.base import SpeechSegment


class _StaticSequenceRefiner:
    def __init__(self, deltas: list[tuple[float, float]]) -> None:
        self.deltas = deltas
        self.calls: list[list[list[float]]] = []
        self.cursor = 0

    def decide_sequence(self, features: list[list[float]]) -> list[BoundaryDecision]:
        self.calls.append(features)
        deltas = self.deltas[self.cursor : self.cursor + len(features)]
        self.cursor += len(features)
        return [
            BoundaryDecision(
                source="edge_sequence_refiner_v8",
                start_refine_delta_s=start_delta,
                end_refine_delta_s=end_delta,
            )
            for start_delta, end_delta in deltas
        ]

    def signature(self) -> dict:
        return {"type": "static_sequence_refiner"}


class _IndexFeatureProvider:
    def features_for_boundary(
        self,
        *,
        left_start_s: float,
        left_end_s: float,
        right_start_s: float,
        right_end_s: float,
    ) -> list[float]:
        return [right_start_s - left_end_s]


def test_boundary_planner_emits_one_chunk_per_scorer_island():
    chunks = plan_boundary_chunks(
        [SpeechSegment(0.0, 12.0), SpeechSegment(12.2, 13.0)],
        config=BoundaryPlannerConfig(frame_hop_s=1.0),
    )

    assert [(chunk.islands[0].start, chunk.islands[0].end) for chunk in chunks] == [
        (0.0, 12.0),
        (12.2, 13.0),
    ]
    assert [chunk.split_reason for chunk in chunks] == ["speech_island", "speech_island"]


def test_boundary_planner_validates_edge_only_config():
    with pytest.raises(ValueError, match="frame_hop_s"):
        plan_boundary_chunks([SpeechSegment(0.0, 1.0)], config=BoundaryPlannerConfig(frame_hop_s=0.0))
    with pytest.raises(ValueError, match="sequence_batch_size"):
        plan_boundary_chunks([SpeechSegment(0.0, 1.0)], config=BoundaryPlannerConfig(sequence_batch_size=0))


def test_sequence_refiner_scores_gaps_without_merging_or_splitting():
    refiner = _StaticSequenceRefiner([(0.0, 0.0), (0.1, -0.1), (0.0, 0.0)])

    chunks = plan_boundary_chunks(
        [
            SpeechSegment(0.0, 3.0),
            SpeechSegment(3.1, 6.0),
            SpeechSegment(6.1, 9.0),
            SpeechSegment(9.1, 12.0),
        ],
        config=BoundaryPlannerConfig(frame_hop_s=1.0),
        sequence_refiner=refiner,
        sequence_feature_provider=_IndexFeatureProvider(),
    )

    assert len(refiner.calls) == 1
    assert [(chunk.islands[0].start, chunk.islands[-1].end) for chunk in chunks] == [
        (0.0, 3.0),
        (3.1, 6.0),
        (6.1, 9.0),
        (9.1, 12.0),
    ]
    assert all(len(chunk.islands) == 1 for chunk in chunks)
    assert chunks[1].boundary_decision is not None
    assert chunks[1].boundary_decision.source == "edge_sequence_refiner_v8"


def test_sequence_planner_batches_long_sequences():
    refiner = _StaticSequenceRefiner([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])

    chunks = plan_boundary_chunks(
        [
            SpeechSegment(0.0, 1.0),
            SpeechSegment(1.1, 2.0),
            SpeechSegment(2.1, 3.0),
            SpeechSegment(3.1, 4.0),
        ],
        config=BoundaryPlannerConfig(frame_hop_s=1.0, sequence_batch_size=2),
        sequence_refiner=refiner,
        sequence_feature_provider=_IndexFeatureProvider(),
    )

    assert [len(call) for call in refiner.calls] == [2, 1]
    assert len(chunks) == 4
