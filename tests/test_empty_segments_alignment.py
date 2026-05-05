from whisper import pipeline as asr
from whisper import local_backend
from whisper.local_backend import align_text_to_words


class _NoFinalizeBackend:
    is_subprocess = False
    accepts_contexts = True
    timestamp_mode = "forced"
    request_batch_size = 1
    align_batch_size = 1

    def unload_model(self, on_stage=None):
        raise AssertionError("empty segment placeholders should not unload model")

    def finalize_text_results(self, text_results, on_stage=None):
        raise AssertionError("empty segment placeholders should not reach finalize")


class _EmptyAligner:
    def align(self, *, audio, text, language):
        return []


def test_align_TRANSCRIPTION_results_empty_input_returns_quarantine_placeholder():
    prepared, timings = asr._align_TRANSCRIPTION_results(_NoFinalizeBackend(), [])

    assert timings == {"alignment_s": 0.0}
    assert len(prepared) == 1
    chunk_result, chunk_log = prepared[0]
    assert chunk_result["words"] == []
    assert chunk_result["alignment_mode"] == "empty"
    assert any("QUARANTINED: kind=empty_segment" in entry for entry in chunk_log)


def test_align_TRANSCRIPTION_results_empty_segments_placeholder():
    prepared, timings = asr._align_TRANSCRIPTION_results(
        _NoFinalizeBackend(),
        [
            {
                "text": "",
                "raw_text": "",
                "duration": 1.25,
                "language": "Japanese",
                "segments": [],
                "log": ["QUARANTINED: kind=timeout, respawn_count=1"],
            }
        ],
    )

    assert timings["alignment_s"] >= 0.0
    assert len(prepared) == 1
    chunk_result, chunk_log = prepared[0]
    assert chunk_result["words"] == []
    assert chunk_result["alignment_mode"] == "empty"
    assert chunk_result["duration"] == 1.25
    assert any(entry.startswith("QUARANTINED:") for entry in chunk_log)


def test_align_text_to_words_with_empty_aligner_result():
    words, mode = align_text_to_words(
        "missing.wav",
        "テスト",
        "Japanese",
        aligner_handle=_EmptyAligner(),
    )

    assert words == []
    assert mode == "forced_aligner"


def test_qwen_finalize_empty_forced_words_falls_back(monkeypatch):
    backend = local_backend.LocalAsrBackend("cpu")
    backend.align_batch_size = 1

    def fake_forced_batch(items, on_stage=None):
        return [[] for _item in items]

    def fake_fallback(text, start, end, audio_path=None):
        return (
            [{"word": text, "start": 0.0, "end": end}],
            "aligner_vad_fallback",
            {"speech_span_count": 1},
        )

    monkeypatch.setattr(backend, "_forced_align_words_batch", fake_forced_batch)
    monkeypatch.setattr(
        local_backend,
        "build_word_timestamps_fallback",
        fake_fallback,
    )

    prepared = backend.finalize_text_results(
        [
            {
                "text": "テスト",
                "raw_text": "テスト",
                "duration": 2.0,
                "language": "Japanese",
                "normalized_path": "missing.wav",
                "log": [],
            }
        ]
    )

    chunk_result, chunk_log = prepared[0]
    assert chunk_result["words"] == [{"word": "テスト", "start": 0.0, "end": 2.0}]
    assert chunk_result["alignment_mode"] == "aligner_vad_fallback"
    assert any("forced aligner returned empty words" in entry for entry in chunk_log)


