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
                source="edge_sequence_refiner_v8",
                start_refine_delta_s=0.0,
                end_refine_delta_s=0.0,
            )
        ]

    def signature(self) -> dict:
        return {"type": "sequence_deny"}


class _StaticSequenceFeatureProvider:
    def features_for_boundary(
        self,
        *,
        left_start_s: float,
        left_end_s: float,
        right_start_s: float,
        right_end_s: float,
    ) -> list[float]:
        assert (left_start_s, left_end_s, right_start_s, right_end_s) == (0.0, 2.0, 3.0, 5.0)
        return [42.0]


class _DeltaSequenceRefiner:
    def decide_sequence(self, features: list[list[float]]) -> list[BoundaryDecision]:
        assert features == [[1.0], [2.0]]
        return [
            BoundaryDecision(
                source="edge_sequence_refiner_v8",
                start_refine_delta_s=-0.2,
                end_refine_delta_s=-0.3,
            ),
            BoundaryDecision(
                source="edge_sequence_refiner_v8",
                start_refine_delta_s=0.15,
                end_refine_delta_s=0.25,
            ),
        ]

    def signature(self) -> dict:
        return {"type": "delta_sequence"}


class _IndexSequenceFeatureProvider:
    def features_for_boundary(
        self,
        *,
        left_start_s: float,
        left_end_s: float,
        right_start_s: float,
        right_end_s: float,
    ) -> list[float]:
        if (left_start_s, left_end_s, right_start_s, right_end_s) == (0.0, 1.0, 1.4, 2.0):
            return [1.0]
        if (left_start_s, left_end_s, right_start_s, right_end_s) == (1.4, 2.0, 2.5, 3.0):
            return [2.0]
        raise AssertionError((left_start_s, left_end_s, right_start_s, right_end_s))


def test_scorer_islands_stay_as_asr_chunks():
    chunks = pack_speech_segments(
        [_seg(5.0, 7.0), _seg(8.0, 9.0), _seg(10.0, 11.0)],
        frame_hop_s=1.0,
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


def test_sequence_refiner_scores_gap_without_merging():
    chunks = pack_speech_segments(
        [_seg(0.0, 2.0), _seg(3.0, 5.0)],
        frame_hop_s=1.0,
        sequence_boundary_refiner=_SequenceDenyRefiner(),
        sequence_feature_provider=_StaticSequenceFeatureProvider(),
    )

    assert [(chunk.core_start, chunk.core_end) for chunk in chunks] == [(0.0, 2.0), (3.0, 5.0)]
    assert chunks[0].split_reason == "speech_island"
    assert chunks[0].boundary_score is None
    assert chunks[0].boundary_decision_source == "edge_sequence_refiner_v8"
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0.0, 2.0), (3.0, 5.0)]


def test_sequence_refiner_boundary_delta_adjusts_only_chunk_edges():
    chunks = pack_speech_segments(
        [_seg(0.0, 1.0), _seg(1.4, 2.0), _seg(2.5, 3.0)],
        frame_hop_s=1.0,
        sequence_boundary_refiner=_DeltaSequenceRefiner(),
        sequence_feature_provider=_IndexSequenceFeatureProvider(),
    )

    assert [(chunk.core_start, chunk.core_end) for chunk in chunks] == [
        (0.0, pytest.approx(0.7)),
        (pytest.approx(1.2), pytest.approx(2.25)),
        (pytest.approx(2.65), 3.0),
    ]
    assert len(chunks) == 3
    assert chunks[0].boundary_start_refine_delta_s is None
    assert chunks[0].boundary_end_refine_delta_s == pytest.approx(-0.3)
    assert chunks[1].boundary_start_refine_delta_s == pytest.approx(-0.2)
    assert chunks[1].boundary_end_refine_delta_s == pytest.approx(0.25)
    assert chunks[2].boundary_start_refine_delta_s == pytest.approx(0.15)
    assert chunks[2].boundary_end_refine_delta_s is None


def test_overlong_scorer_island_is_not_duration_split_by_packer():
    chunks = pack_speech_segments([_seg(10.0, 75.0)], frame_hop_s=1.0)

    assert [(chunk.start, chunk.end, chunk.duration) for chunk in chunks] == [(10.0, 75.0, 65.0)]
    assert [[(seg.start, seg.end) for seg in chunk.speech_segments] for chunk in chunks] == [[(10.0, 75.0)]]


def test_empty_segments_return_empty_list():
    assert pack_speech_segments([]) == []


def test_single_segment_uses_core_window_without_padding():
    chunks = pack_speech_segments([_seg(0.4, 1.0)], frame_hop_s=1.0)

    assert chunks[0].start == pytest.approx(0.4)
    assert chunks[0].end == pytest.approx(1.0)
