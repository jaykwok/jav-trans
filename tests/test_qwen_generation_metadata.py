from __future__ import annotations

from types import SimpleNamespace

from whisper import local_backend


def test_qwen_text_result_includes_generation_metadata(monkeypatch, tmp_path):
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"not used")
    monkeypatch.setattr(local_backend, "_get_wav_duration", lambda _path: 1.25)

    backend = local_backend.LocalAsrBackend("cpu")
    result, log = backend._build_text_result(
        str(audio_path),
        SimpleNamespace(language="Japanese", text="テスト"),
        "Japanese",
    )

    assert result["text"] == "テスト"
    assert result["asr_generation"]["backend"] == "qwen3-asr-1.7b"
    assert result["asr_generation"]["configured_max_new_tokens"] == local_backend.TRANSCRIPTION_MAX_NEW_TOKENS
    assert result["asr_generation"]["model_max_target_positions"] is None
    assert result["asr_generation"]["error_kind"] is None
    assert "ASR 输出模式: text_only" in log


def test_qwen_generation_metadata_records_timeout_kind():
    metadata = local_backend._qwen_generation_metadata(
        error_kind="timeout",
        error_detail="worker timeout",
        worker_mode="subprocess",
    )

    assert metadata["backend"] == "qwen3-asr-1.7b"
    assert metadata["worker_mode"] == "subprocess"
    assert metadata["error_kind"] == "timeout"
    assert metadata["error_detail"] == "worker timeout"
