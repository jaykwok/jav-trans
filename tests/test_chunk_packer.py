from __future__ import annotations

import pytest

from audio.chunk_packer import pack_vad_segments
from vad.base import SpeechSegment


def _seg(start: float, end: float) -> SpeechSegment:
    return SpeechSegment(start=start, end=end)


def _pack(segments):
    return pack_vad_segments(
        segments,
        frame_hop_s=1.0,
        window_frames=30,
        reserve_frames=2,
        target_padding_frames=4,
        gap_merge_frames=3,
    )


def test_dense_segments_pack_into_one_chunk_with_dynamic_padding():
    chunks = _pack([_seg(5.0, 7.0), _seg(8.0, 9.0), _seg(10.0, 11.0)])

    assert len(chunks) == 1
    assert chunks[0].start == 1.0
    assert chunks[0].end == 15.0
    assert chunks[0].duration == 14.0
    assert chunks[0].left_padding_s == 4.0
    assert chunks[0].right_padding_s == 4.0
    assert chunks[0].vad_segments == [_seg(5.0, 7.0), _seg(8.0, 9.0), _seg(10.0, 11.0)]


def test_gap_larger_than_frame_threshold_starts_new_chunk_and_splits_gap_padding():
    chunks = _pack([_seg(10.0, 11.0), _seg(16.0, 17.0)])

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(6.0, 13.5), (13.5, 21.0)]
    assert [(chunk.left_padding_s, chunk.right_padding_s) for chunk in chunks] == [
        (4.0, 2.5),
        (2.5, 4.0),
    ]


def _pack_capped(segments, max_core_frames):
    return pack_vad_segments(
        segments,
        frame_hop_s=1.0,
        window_frames=30,
        reserve_frames=2,
        target_padding_frames=4,
        gap_merge_frames=3,
        max_core_frames=max_core_frames,
    )


def test_max_core_cap_splits_accumulated_islands_at_mergeable_gap():
    # gaps (2s) are below gap_merge (3s) so without a cap all three islands merge
    # into one core=13s chunk; the soft cap forces a split at the island gap.
    segments = [_seg(0.0, 3.0), _seg(5.0, 8.0), _seg(10.0, 13.0)]

    assert len(_pack_capped(segments, 0)) == 1  # disabled == legacy behaviour

    capped = _pack_capped(segments, 8)
    assert [(c.vad_segments[0].start, c.vad_segments[-1].end) for c in capped] == [
        (0.0, 8.0),
        (10.0, 13.0),
    ]
    assert capped[0].split_reason == "soft_cap"
    assert capped[1].split_reason == "tail"


def test_max_core_cap_does_not_split_single_island_without_gap():
    # A single long VAD segment has no internal gap to split on, so the soft cap
    # leaves it intact (handled by overlong splitting / left as-is by design).
    capped = _pack_capped([_seg(2.0, 12.0)], 8)
    assert len(capped) == 1
    assert capped[0].vad_segments == [_seg(2.0, 12.0)]


def test_near_capacity_segment_reduces_padding_to_fit_window_reserve():
    chunks = _pack([_seg(2.0, 28.0)])

    assert len(chunks) == 1
    assert chunks[0].duration == 28.0
    assert chunks[0].start == 1.0
    assert chunks[0].end == 29.0
    assert chunks[0].left_padding_s == 1.0
    assert chunks[0].right_padding_s == 1.0


def test_single_overlong_segment_is_split_by_frame_capacity_before_packing():
    chunks = _pack([_seg(10.0, 75.0)])

    assert [(chunk.start, chunk.end, chunk.duration) for chunk in chunks] == [
        (6.0, 34.0, 28.0),
        (26.0, 54.0, 28.0),
        (48.5, 76.5, 28.0),
    ]
    assert [[(seg.start, seg.end) for seg in chunk.vad_segments] for chunk in chunks] == [
        [(10.0, 30.0)],
        [(30.0, 50.0)],
        [(50.0, 70.0), (70.0, 75.0)],
    ]


def test_overlong_segment_split_preserves_score():
    segment = SpeechSegment(start=0.0, end=30.0, score=0.75)
    chunks = pack_vad_segments(
        [segment],
        frame_hop_s=1.0,
        window_frames=12,
        reserve_frames=2,
        target_padding_frames=1,
        gap_merge_frames=0,
    )

    assert [(chunk.start, chunk.end, chunk.duration) for chunk in chunks] == [
        (0.0, 9.0, 9.0),
        (7.0, 17.0, 10.0),
        (15.0, 25.0, 10.0),
        (23.0, 31.0, 8.0),
    ]
    assert [chunk.vad_segments[0].score for chunk in chunks] == [0.75, 0.75, 0.75, 0.75]


def test_adjacent_overlong_tail_can_pack_with_following_segment_when_gap_allowed():
    chunks = pack_vad_segments(
        [_seg(10.0, 45.0), _seg(46.0, 47.0)],
        frame_hop_s=1.0,
        window_frames=30,
        reserve_frames=2,
        target_padding_frames=1,
        gap_merge_frames=1,
    )

    assert [[(seg.start, seg.end) for seg in chunk.vad_segments] for chunk in chunks] == [
        [(10.0, 36.0)],
        [(36.0, 45.0), (46.0, 47.0)],
    ]
    assert all(chunk.duration <= 28.0 for chunk in chunks)


def test_empty_segments_return_empty_list():
    assert pack_vad_segments([]) == []


def test_single_segment_at_audio_start_clamps_left_padding_to_zero():
    chunks = _pack([_seg(0.4, 1.0)])

    assert chunks[0].start == 0.0
    assert chunks[0].end == 5.0
    assert chunks[0].left_padding_s == 0.4
    assert chunks[0].right_padding_s == 4.0


def test_invalid_frame_config_raises():
    with pytest.raises(ValueError, match="reserve_frames must be smaller"):
        pack_vad_segments([_seg(0.0, 1.0)], window_frames=10, reserve_frames=10)
