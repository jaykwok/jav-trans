from __future__ import annotations

from audio.chunk_packer import pack_vad_segments
from vad.base import SpeechSegment


def _seg(start: float, end: float) -> SpeechSegment:
    return SpeechSegment(start=start, end=end)


def test_dense_segments_pack_into_one_chunk():
    chunks = pack_vad_segments(
        [_seg(1.0, 3.0), _seg(3.5, 5.0), _seg(5.4, 6.0)],
        max_s=10.0,
        gap_merge_s=1.5,
        padding_s=1.0,
    )

    assert len(chunks) == 1
    assert chunks[0].start == 0.0
    assert chunks[0].end == 7.0
    assert chunks[0].duration == 7.0
    assert chunks[0].vad_segments == [_seg(1.0, 3.0), _seg(3.5, 5.0), _seg(5.4, 6.0)]


def test_large_gap_starts_new_chunk():
    chunks = pack_vad_segments(
        [_seg(5.0, 6.0), _seg(8.0, 9.0)],
        max_s=20.0,
        gap_merge_s=1.5,
        padding_s=1.0,
    )

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(4.0, 7.0), (7.0, 10.0)]
    assert [len(chunk.vad_segments) for chunk in chunks] == [1, 1]


def test_near_max_duration_keeps_chunk_within_limit():
    chunks = pack_vad_segments(
        [_seg(2.0, 12.0), _seg(13.0, 28.0)],
        max_s=28.0,
        gap_merge_s=1.5,
        padding_s=1.0,
    )

    assert len(chunks) == 1
    assert chunks[0].start == 1.0
    assert chunks[0].end == 29.0
    assert chunks[0].duration == 28.0


def test_single_overlong_segment_is_independent_chunk():
    chunks = pack_vad_segments(
        [_seg(10.0, 45.0), _seg(46.0, 47.0)],
        max_s=28.0,
        gap_merge_s=1.5,
        padding_s=1.0,
    )

    assert [(chunk.start, chunk.end, chunk.duration) for chunk in chunks] == [
        (9.0, 46.0, 37.0),
        (45.0, 48.0, 3.0),
    ]
    assert [len(chunk.vad_segments) for chunk in chunks] == [1, 1]


def test_empty_segments_return_empty_list():
    assert pack_vad_segments([]) == []


def test_single_segment_gets_padding():
    chunks = pack_vad_segments([_seg(5.0, 8.0)], padding_s=2.0)

    assert len(chunks) == 1
    assert chunks[0].start == 3.0
    assert chunks[0].end == 10.0
    assert chunks[0].duration == 7.0
    assert chunks[0].vad_segments == [_seg(5.0, 8.0)]


def test_padding_start_is_clamped_to_zero():
    chunks = pack_vad_segments([_seg(0.4, 1.0)], padding_s=2.0)

    assert chunks[0].start == 0.0
    assert chunks[0].end == 3.0
    assert chunks[0].duration == 3.0
