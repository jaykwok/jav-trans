"""Regression tests for ASR single-chunk non-CUDA error isolation and cache safety."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from asr import local_backend, result_cache


def _write_wav(path: Path, *, duration_s: float = 0.2, rate: int = 16000) -> Path:
    frames = int(round(duration_s * rate))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * frames)
    return path


def test_local_asr_isolates_single_bad_chunk_instead_of_aborting_batch(
    monkeypatch, tmp_path
):
    wav_a = _write_wav(tmp_path / "a.wav")
    wav_bad = _write_wav(tmp_path / "bad.wav")
    wav_c = _write_wav(tmp_path / "c.wav")

    backend = local_backend.LocalAsrBackend(device="cpu")
    backend.model = object()
    backend.processor = object()

    def _fake_native(paths, _lang):
        if any(Path(p).name == "bad.wav" for p in paths):
            raise RuntimeError("corrupt chunk parse failure")
        return [
            local_backend.NativeAsrTranscription(
                language="Japanese", text=f"text:{Path(p).stem}"
            )
            for p in paths
        ]

    monkeypatch.setattr(backend, "_transcribe_native", _fake_native)
    stages: list[str] = []
    try:
        results = backend.transcribe_texts(
            [str(wav_a), str(wav_bad), str(wav_c)], on_stage=stages.append
        )
    finally:
        backend.model = None
        backend.close()

    assert [r["text"] for r in results] == ["text:a", "", "text:c"]
    assert results[1]["asr_generation"]["error_kind"] == "chunk_error"
    assert any("单块转写异常" in msg for msg in stages)


def test_local_asr_still_propagates_fatal_cuda_oom(monkeypatch, tmp_path):
    wav_a = _write_wav(tmp_path / "a.wav")
    backend = local_backend.LocalAsrBackend(device="cuda")
    backend.model = object()
    backend.processor = object()

    def _oom(_paths, _lang):
        raise RuntimeError("CUDA out of memory while allocating tensor")

    monkeypatch.setattr(backend, "_transcribe_native", _oom)
    try:
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            backend.transcribe_texts([str(wav_a)])
    finally:
        backend.model = None
        backend.close()


def test_asr_result_cache_does_not_store_chunk_error_results(monkeypatch, tmp_path):
    wav = _write_wav(tmp_path / "chunk.wav")
    monkeypatch.setenv("ASR_RESULT_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("ASR_RESULT_CACHE_ENABLED", "1")
    sig = {"schema": result_cache.CACHE_SCHEMA, "stage": "text", "test": True}

    failed_payload = {
        "text": "",
        "raw_text": "",
        "duration": 0.2,
        "language": "Japanese",
        "asr_generation": {"error_kind": "chunk_error", "error_detail": "boom"},
        "log": ["[WARN] ASR chunk skipped after error: boom"],
    }
    result_cache.store(wav, failed_payload, signature=sig)
    assert result_cache.lookup(wav, signature=sig) is None
