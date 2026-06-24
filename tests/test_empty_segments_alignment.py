from asr import local_backend
from asr import pipeline as asr


class _NoFinalizeBackend:
    is_subprocess = False
    accepts_contexts = True
    request_batch_size = 1

    def unload_model(self, on_stage=None):
        raise AssertionError("empty segment placeholders should not unload model")

    def finalize_text_results(self, text_results, on_stage=None):
        raise AssertionError("empty segment placeholders should not reach finalize")


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


def test_qwen_finalize_uses_boundary_chunk_timeline(monkeypatch):
    backend = local_backend.LocalAsrBackend("cpu")
    calls = []

    def fake_boundary_timing(text, start, end, audio_path=None):
        calls.append((text, start, end, audio_path))
        return (
            [{"word": text, "start": start, "end": end}],
            "boundary_proportional",
            {"timing_source": "boundary_chunk"},
        )

    monkeypatch.setattr(local_backend, "build_boundary_word_timestamps", fake_boundary_timing)

    prepared = backend.finalize_text_results(
        [
            {
                "text": "テスト",
                "raw_text": "テスト",
                "duration": 10.0,
                "language": "Japanese",
                "normalized_path": "missing.wav",
                "alignment_window_start_s": 2.0,
                "alignment_window_end_s": 8.0,
                "alignment_window_source": "speech_core",
                "log": [],
            }
        ]
    )

    chunk_result, chunk_log = prepared[0]
    assert chunk_result["words"] == [{"word": "テスト", "start": 2.0, "end": 8.0}]
    assert chunk_result["alignment_mode"] == "boundary_proportional"
    assert calls == [("テスト", 2.0, 8.0, "missing.wav")]
    assert any("Subtitle timing: boundary_chunk_timeline" in entry for entry in chunk_log)
    assert any("Subtitle timing window: speech_core" in entry for entry in chunk_log)
