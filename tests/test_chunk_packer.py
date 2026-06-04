from __future__ import annotations

import pytest

from audio.chunk_packer import pack_speech_segments
from boundary.refiner import BoundaryDecision, RefinerInput
from boundary.base import SpeechSegment


def _seg(start: float, end: float) -> SpeechSegment:
    return SpeechSegment(start=start, end=end)


class _DenyGapRefiner:
    def decide_gap(self, item: RefinerInput) -> BoundaryDecision:
        return BoundaryDecision(False, 0.12, "speaker_change")

    def signature(self) -> dict:
        return {"type": "deny_gap"}


class _AllowGapRefiner:
    def decide_gap(self, item: RefinerInput) -> BoundaryDecision:
        return BoundaryDecision(True, 0.92, "semantic_continuation")

    def signature(self) -> dict:
        return {"type": "allow_gap"}


def test_dense_segments_pack_into_one_chunk_with_dynamic_padding():
    chunks = pack_speech_segments(
        [_seg(5.0, 7.0), _seg(8.0, 9.0), _seg(10.0, 11.0)],
        frame_hop_s=1.0,
        max_chunk_s=30.0,
        target_chunk_s=9.0,
        target_padding_s=4.0,
    )

    assert len(chunks) == 1
    assert chunks[0].start == 1.0
    assert chunks[0].end == 15.0
    assert chunks[0].duration == 14.0
    assert chunks[0].left_padding_s == 4.0
    assert chunks[0].right_padding_s == 4.0
    assert chunks[0].speech_segments == [_seg(5.0, 7.0), _seg(8.0, 9.0), _seg(10.0, 11.0)]


def test_gap_refiner_can_block_mergeable_gap():
    chunks = pack_speech_segments(
        [_seg(0.0, 2.0), _seg(3.0, 5.0)],
        frame_hop_s=1.0,
        max_chunk_s=30.0,
        target_chunk_s=9.0,
        target_padding_s=1.0,
        boundary_refiner=_DenyGapRefiner(),
    )

    assert [(chunk.core_start, chunk.core_end) for chunk in chunks] == [(0.0, 2.0), (3.0, 5.0)]
    assert chunks[0].split_reason == "boundary_refiner:speaker_change"
    assert chunks[0].boundary_score == pytest.approx(0.12)
    assert chunks[0].boundary_reason == "speaker_change"


def test_planner_max_splits_even_when_refiner_allows_gap():
    chunks = pack_speech_segments(
        [_seg(0.0, 5.0), _seg(6.0, 11.0)],
        frame_hop_s=1.0,
        max_chunk_s=8.0,
        target_chunk_s=9.0,
        target_padding_s=1.0,
        boundary_refiner=_AllowGapRefiner(),
    )

    assert [(chunk.core_start, chunk.core_end) for chunk in chunks] == [(0.0, 5.0), (6.0, 11.0)]
    assert chunks[0].split_reason == "planner_max"


def test_boundary_candidate_split_metadata_is_materialized():
    cut_scores = [0.05] * 5 + [0.98] * 2 + [0.05] * 5

    chunks = pack_speech_segments(
        [_seg(0.0, 12.0)],
        frame_hop_s=1.0,
        max_chunk_s=30.0,
        target_chunk_s=5.0,
        min_chunk_s=3.0,
        target_padding_s=1.0,
        cut_frame_scores=cut_scores,
    )

    assert [(chunk.core_start, chunk.core_end) for chunk in chunks] == [
        (0.0, pytest.approx(6.0)),
        (pytest.approx(6.0), 12.0),
    ]
    assert chunks[0].split_reason == "cut_candidate"
    assert chunks[1].split_reason == "boundary_candidate"
    assert {chunk.boundary_source for chunk in chunks} == {"cut"}
    assert [chunk.boundary_score for chunk in chunks] == [pytest.approx(0.98), pytest.approx(0.98)]


def test_candidate_split_accepts_numpy_score_arrays():
    np = pytest.importorskip("numpy")
    cut_scores = np.asarray([0.05] * 5 + [0.98] * 2 + [0.05] * 5, dtype=np.float32)

    chunks = pack_speech_segments(
        [_seg(0.0, 12.0)],
        frame_hop_s=1.0,
        max_chunk_s=30.0,
        target_chunk_s=5.0,
        min_chunk_s=3.0,
        target_padding_s=1.0,
        cut_frame_scores=cut_scores,
    )

    assert len(chunks) == 2
    assert chunks[0].core_end == pytest.approx(6.0)


def test_overlong_segment_is_split_by_planner_capacity():
    chunks = pack_speech_segments(
        [_seg(10.0, 75.0)],
        frame_hop_s=1.0,
        max_chunk_s=28.0,
        target_chunk_s=9.0,
        target_padding_s=4.0,
    )

    assert [(chunk.start, chunk.end, chunk.duration) for chunk in chunks] == [
        (6.0, 34.0, 28.0),
        (26.0, 54.0, 28.0),
        (46.0, 74.0, 28.0),
        (66.0, 79.0, 13.0),
    ]
    assert [[(seg.start, seg.end) for seg in chunk.speech_segments] for chunk in chunks] == [
        [(10.0, 30.0)],
        [(30.0, 50.0)],
        [(50.0, 70.0)],
        [(70.0, 75.0)],
    ]
    assert all(chunk.duration <= 28.0 for chunk in chunks)


def test_empty_segments_return_empty_list():
    assert pack_speech_segments([]) == []


def test_single_segment_at_audio_start_clamps_left_padding_to_zero():
    chunks = pack_speech_segments(
        [_seg(0.4, 1.0)],
        frame_hop_s=1.0,
        max_chunk_s=30.0,
        target_chunk_s=9.0,
        target_padding_s=4.0,
    )

    assert chunks[0].start == 0.0
    assert chunks[0].left_padding_s == pytest.approx(0.4)
