from asr.subtitle_timing import build_boundary_word_timestamps


def test_boundary_word_timestamps_fill_chunk_window():
    words, mode, meta = build_boundary_word_timestamps("ABC", 2.0, 5.0)

    assert mode == "boundary_proportional"
    assert meta == {"timing_source": "boundary_chunk"}
    assert words[0]["start"] == 2.0
    assert words[-1]["end"] == 5.0


def test_boundary_word_timestamps_empty_text():
    words, mode, meta = build_boundary_word_timestamps("", 2.0, 5.0)

    assert words == []
    assert mode == "empty"
    assert meta == {"timing_source": "boundary_chunk"}
