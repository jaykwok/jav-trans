from __future__ import annotations

import pytest

from audio.chunk_packer import pack_speech_segments
from boundary.refiner import BoundaryDecision
from boundary.base import SpeechSegment


def _seg(start: float, end: float) -> SpeechSegment:
    return SpeechSegment(start=start, end=end)


class _SequenceDenyRefiner:
    def decide_sequence(self, features: list[list[float]]) -> list[BoundaryDecision]:
        assert features == [[42.0]]
        return [
            BoundaryDecision(
                False,
                0.08,
                "learned_sequence_split",
                source="frame_sequence_refiner",
            )
        ]

    def signature(self) -> dict:
        return {"type": "sequence_deny"}


class _StaticSequenceFeatureProvider:
    def features_for_gap(
        self,
        *,
        left_start_s: float,
        left_end_s: float,
        right_start_s: float,
        right_end_s: float,
    ) -> list[float]:
        assert (left_start_s, left_end_s, right_start_s, right_end_s) == (0.0, 2.0, 3.0, 5.0)
        return [42.0]


class _BatchSequenceRefiner:
    def __init__(self) -> None:
        self.calls: list[list[list[float]]] = []

    def decide_sequence(self, features: list[list[float]]) -> list[BoundaryDecision]:
        self.calls.append(features)
        assert features == [[1.0], [2.0]]
        return [
            BoundaryDecision(True, 0.91, "learned_sequence_merge", source="frame_sequence_refiner"),
            BoundaryDecision(
                False,
                0.07,
                "learned_sequence_split",
                source="frame_sequence_refiner",
            ),
        ]

    def signature(self) -> dict:
        return {"type": "batch_sequence"}


class _DeltaSequenceRefiner:
    def decide_sequence(self, features: list[list[float]]) -> list[BoundaryDecision]:
        assert features == [[1.0], [2.0]]
        return [
            BoundaryDecision(
                False,
                0.10,
                "learned_sequence_split",
                source="frame_sequence_refiner",
                start_refine_delta_s=-0.2,
                end_refine_delta_s=-0.3,
            ),
            BoundaryDecision(
                False,
                0.20,
                "learned_sequence_split",
                source="frame_sequence_refiner",
                start_refine_delta_s=0.15,
                end_refine_delta_s=0.25,
            ),
        ]

    def signature(self) -> dict:
        return {"type": "delta_sequence"}


class _IndexSequenceFeatureProvider:
    def features_for_gap(
        self,
        *,
        left_start_s: float,
        left_end_s: float,
        right_start_s: float,
        right_end_s: float,
    ) -> list[float]:
        if (left_start_s, left_end_s, right_start_s, right_end_s) == (0.0, 1.0, 1.2, 2.0):
            return [1.0]
        if (left_start_s, left_end_s, right_start_s, right_end_s) == (1.2, 2.0, 2.4, 3.0):
            return [2.0]
        if (left_start_s, left_end_s, right_start_s, right_end_s) == (0.0, 1.0, 1.4, 2.0):
            return [1.0]
        if (left_start_s, left_end_s, right_start_s, right_end_s) == (1.4, 2.0, 2.5, 3.0):
            return [2.0]
        raise AssertionError((left_start_s, left_end_s, right_start_s, right_end_s))


def test_dense_segments_stay_as_speech_core_chunks():
    chunks = pack_speech_segments(
        [_seg(5.0, 7.0), _seg(8.0, 9.0), _seg(10.0, 11.0)],
        frame_hop_s=1.0,
        max_core_chunk_s=30.0,
        target_chunk_s=9.0,
    )

    assert [(chunk.core_start, chunk.core_end) for chunk in chunks] == [
        (5.0, 7.0),
        (8.0, 9.0),
        (10.0, 11.0),
    ]
    assert [(chunk.start, chunk.end) for chunk in chunks] == [
        (5.0, 7.0),
        (8.0, 9.0),
        (10.0, 11.0),
    ]


def test_sequence_refiner_can_block_mergeable_gap():
    chunks = pack_speech_segments(
        [_seg(0.0, 2.0), _seg(3.0, 5.0)],
        frame_hop_s=1.0,
        max_core_chunk_s=30.0,
        target_chunk_s=9.0,
        sequence_boundary_refiner=_SequenceDenyRefiner(),
        sequence_feature_provider=_StaticSequenceFeatureProvider(),
    )

    assert [(chunk.core_start, chunk.core_end) for chunk in chunks] == [(0.0, 2.0), (3.0, 5.0)]
    assert chunks[0].split_reason == "boundary_refiner:learned_sequence_split"
    assert chunks[0].boundary_score == pytest.approx(0.08)
    assert chunks[0].boundary_decision_source == "frame_sequence_refiner"
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0.0, 2.0), (3.0, 5.0)]


def test_sequence_refiner_scores_all_gaps_in_one_batch():
    refiner = _BatchSequenceRefiner()

    chunks = pack_speech_segments(
        [_seg(0.0, 1.0), _seg(1.2, 2.0), _seg(2.4, 3.0)],
        frame_hop_s=1.0,
        max_core_chunk_s=30.0,
        target_chunk_s=9.0,
        sequence_boundary_refiner=refiner,
        sequence_feature_provider=_IndexSequenceFeatureProvider(),
    )

    assert len(refiner.calls) == 1
    assert [(chunk.core_start, chunk.core_end) for chunk in chunks] == [
        (0.0, 1.0),
        (1.2, 2.0),
        (2.4, 3.0),
    ]
    assert chunks[1].boundary_decision_merge is False
    assert chunks[1].boundary_decision_source == "frame_sequence_refiner"


def test_sequence_refiner_boundary_delta_adjusts_current_chunk_core():
    chunks = pack_speech_segments(
        [_seg(0.0, 1.0), _seg(1.4, 2.0), _seg(2.5, 3.0)],
        frame_hop_s=1.0,
        max_core_chunk_s=30.0,
        target_chunk_s=9.0,
        sequence_boundary_refiner=_DeltaSequenceRefiner(),
        sequence_feature_provider=_IndexSequenceFeatureProvider(),
    )

    assert [(chunk.core_start, chunk.core_end) for chunk in chunks] == [
        (0.0, pytest.approx(0.7)),
        (pytest.approx(1.2), pytest.approx(2.25)),
        (pytest.approx(2.65), 3.0),
    ]
    assert chunks[0].boundary_start_refine_delta_s is None
    assert chunks[0].boundary_end_refine_delta_s == pytest.approx(-0.3)
    assert chunks[1].boundary_start_refine_delta_s == pytest.approx(-0.2)
    assert chunks[1].boundary_end_refine_delta_s == pytest.approx(0.25)
    assert chunks[2].boundary_start_refine_delta_s == pytest.approx(0.15)
    assert chunks[2].boundary_end_refine_delta_s is None


def test_planner_max_splits_keep_islands_separate_without_sequence_refiner():
    chunks = pack_speech_segments(
        [_seg(0.0, 5.0), _seg(6.0, 11.0)],
        frame_hop_s=1.0,
        max_core_chunk_s=8.0,
        target_chunk_s=9.0,
    )

    assert [(chunk.core_start, chunk.core_end) for chunk in chunks] == [(0.0, 5.0), (6.0, 11.0)]
    assert chunks[0].split_reason == "speech_island"


def test_boundary_candidate_split_metadata_is_materialized():
    cut_scores = [0.05] * 5 + [0.98] * 2 + [0.05] * 5

    chunks = pack_speech_segments(
        [_seg(0.0, 12.0)],
        frame_hop_s=1.0,
        max_core_chunk_s=30.0,
        target_chunk_s=5.0,
        min_chunk_s=3.0,
        cut_frame_scores=cut_scores,
    )

    assert [(chunk.core_start, chunk.core_end) for chunk in chunks] == [
        (0.0, pytest.approx(6.0)),
        (pytest.approx(6.0), 12.0),
    ]
    assert chunks[0].split_reason == "cut_candidate"
    assert chunks[1].split_reason == "cut_candidate"
    assert {chunk.boundary_source for chunk in chunks} == {"cut"}
    assert [chunk.boundary_score for chunk in chunks] == [pytest.approx(0.98), pytest.approx(0.98)]


def test_candidate_split_accepts_numpy_score_arrays():
    np = pytest.importorskip("numpy")
    cut_scores = np.asarray([0.05] * 5 + [0.98] * 2 + [0.05] * 5, dtype=np.float32)

    chunks = pack_speech_segments(
        [_seg(0.0, 12.0)],
        frame_hop_s=1.0,
        max_core_chunk_s=30.0,
        target_chunk_s=5.0,
        min_chunk_s=3.0,
        cut_frame_scores=cut_scores,
    )

    assert len(chunks) == 2
    assert chunks[0].core_end == pytest.approx(6.0)


def test_overlong_segment_is_split_by_planner_capacity():
    chunks = pack_speech_segments(
        [_seg(10.0, 75.0)],
        frame_hop_s=1.0,
        max_core_chunk_s=20.0,
        target_chunk_s=9.0,
    )

    assert [(chunk.start, chunk.end, chunk.duration) for chunk in chunks] == [
        (10.0, 30.0, 20.0),
        (30.0, 50.0, 20.0),
        (50.0, 70.0, 20.0),
        (70.0, 75.0, 5.0),
    ]
    assert [[(seg.start, seg.end) for seg in chunk.speech_segments] for chunk in chunks] == [
        [(10.0, 30.0)],
        [(30.0, 50.0)],
        [(50.0, 70.0)],
        [(70.0, 75.0)],
    ]


def test_empty_segments_return_empty_list():
    assert pack_speech_segments([]) == []


def test_single_segment_uses_core_window_without_padding():
    chunks = pack_speech_segments(
        [_seg(0.4, 1.0)],
        frame_hop_s=1.0,
        max_core_chunk_s=30.0,
        target_chunk_s=9.0,
    )

    assert chunks[0].start == pytest.approx(0.4)
    assert chunks[0].end == pytest.approx(1.0)
