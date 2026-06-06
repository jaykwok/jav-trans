from asr import timestamp_fallback


def test_vad_fallback_clips_spans_to_nonzero_window(monkeypatch):
    monkeypatch.setattr(
        timestamp_fallback,
        "detect_speech_spans",
        lambda _audio_path: ([(0.5, 1.5), (3.0, 4.0), (6.0, 7.0)], ""),
    )

    words, mode, meta = timestamp_fallback.build_word_timestamps_fallback(
        "ABC",
        2.0,
        5.0,
        audio_path="missing.wav",
    )

    assert mode == "aligner_vad_fallback"
    assert meta["speech_span_count"] == 3
    assert words[0]["start"] == 3.0
    assert words[-1]["end"] == 4.0
