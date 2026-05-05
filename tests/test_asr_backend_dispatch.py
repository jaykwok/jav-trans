from __future__ import annotations

import importlib
import sys
import types

from whisper.backends.base import BaseAsrBackend


class FakeWhisperModelBackend:
    is_subprocess = False
    accepts_contexts = False
    timestamp_mode = "forced"
    request_batch_size = 1
    align_batch_size = 1

    def load(self, on_stage=None):
        return None

    def unload_model(self, on_stage=None):
        return None

    def unload_forced_aligner(self, on_stage=None):
        return None

    def transcribe_texts(self, audio_paths, contexts=None, on_stage=None):
        return []

    def finalize_text_results(self, text_results, on_stage=None):
        return []


def _reload_asr(monkeypatch, *, backend: str, worker_mode: str):
    monkeypatch.setenv("ASR_BACKEND", backend)
    monkeypatch.setenv("ASR_WORKER_MODE", worker_mode)
    from whisper import pipeline as asr
    return importlib.reload(asr)


def _install_fake_whisper_module(monkeypatch):
    calls = []
    fake_module = types.ModuleType("whisper.model_backend")

    def create_whisper_model_backend(preset_name, device):
        calls.append((preset_name, device))
        return FakeWhisperModelBackend()

    fake_module.create_whisper_model_backend = create_whisper_model_backend
    monkeypatch.setitem(sys.modules, "whisper.model_backend", fake_module)
    return calls


def _assert_whisper_dispatch(monkeypatch, backend_name: str):
    calls = _install_fake_whisper_module(monkeypatch)

    asr = _reload_asr(
        monkeypatch,
        backend=backend_name,
        worker_mode="inproc",
    )
    backend = asr._resolve_asr_backend("cuda:0")

    assert isinstance(backend, BaseAsrBackend)
    assert backend.is_subprocess is False
    assert backend.accepts_contexts is False
    assert asr.get_backend_label() == backend_name
    assert calls == [(backend_name, "cuda:0")]


def test_anime_backend_dispatch(monkeypatch):
    _assert_whisper_dispatch(monkeypatch, "anime-whisper")


def test_whisper_anime_backend_dispatch(monkeypatch):
    _assert_whisper_dispatch(monkeypatch, "whisper-ja-anime-v0.3")


def test_japanese_whisper_15b_backend_dispatch(monkeypatch):
    _assert_whisper_dispatch(monkeypatch, "whisper-ja-1.5b")


def test_qwen3_asr_dispatch_uses_local_backend(monkeypatch):
    asr = _reload_asr(monkeypatch, backend="qwen3-asr-1.7b", worker_mode="inproc")
    backend = asr._resolve_asr_backend("cpu")

    assert type(backend).__name__ == "LocalAsrBackend"
    assert isinstance(backend, BaseAsrBackend)
    assert backend.is_subprocess is False
    assert asr.get_backend_label() == "qwen3-asr-1.7b (inproc)"


def test_whisper_anime_backend_uses_whisper_model_backend(monkeypatch):
    calls = []
    fake_module = types.ModuleType("whisper.model_backend")

    def create_whisper_model_backend(preset_name, device):
        calls.append((preset_name, device))
        return FakeWhisperModelBackend()

    fake_module.create_whisper_model_backend = create_whisper_model_backend
    monkeypatch.setitem(sys.modules, "whisper.model_backend", fake_module)

    asr = _reload_asr(
        monkeypatch,
        backend="whisper-ja-anime-v0.3",
        worker_mode="inproc",
    )
    backend = asr._resolve_asr_backend("cuda:0")

    assert isinstance(backend, BaseAsrBackend)
    assert backend.is_subprocess is False
    assert backend.accepts_contexts is False
    assert calls == [("whisper-ja-anime-v0.3", "cuda:0")]
    assert asr.get_backend_label() == "whisper-ja-anime-v0.3"


def test_japanese_whisper_15b_backend_uses_whisper_model_backend(monkeypatch):
    calls = []
    fake_module = types.ModuleType("whisper.model_backend")

    def create_whisper_model_backend(preset_name, device):
        calls.append((preset_name, device))
        return FakeWhisperModelBackend()

    fake_module.create_whisper_model_backend = create_whisper_model_backend
    monkeypatch.setitem(sys.modules, "whisper.model_backend", fake_module)

    asr = _reload_asr(
        monkeypatch,
        backend="whisper-ja-1.5b",
        worker_mode="inproc",
    )
    backend = asr._resolve_asr_backend("cuda:0")

    assert isinstance(backend, BaseAsrBackend)
    assert backend.is_subprocess is False
    assert backend.accepts_contexts is False
    assert calls == [("whisper-ja-1.5b", "cuda:0")]
    assert asr.get_backend_label() == "whisper-ja-1.5b"


def test_whisper_backend_ignores_subprocess_worker_mode(monkeypatch):
    calls = _install_fake_whisper_module(monkeypatch)
    asr = _reload_asr(monkeypatch, backend="anime-whisper", worker_mode="subprocess")
    backend = asr._resolve_asr_backend("cuda:0")

    assert isinstance(backend, BaseAsrBackend)
    assert backend.is_subprocess is False
    assert calls == [("anime-whisper", "cuda:0")]


def test_invalid_worker_mode_is_rejected(monkeypatch):
    asr = _reload_asr(monkeypatch, backend="qwen3-asr-1.7b", worker_mode="invalid")

    try:
        asr._resolve_asr_backend("cpu")
    except ValueError as exc:
        assert "Unsupported ASR_WORKER_MODE" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_invalid_asr_backend_is_rejected(monkeypatch):
    asr = _reload_asr(monkeypatch, backend="unknown_backend", worker_mode="inproc")

    try:
        asr._resolve_asr_backend("cpu")
    except ValueError as exc:
        assert "Unsupported ASR_BACKEND" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_internal_asr_backend_names_are_rejected(monkeypatch):
    invalid_name = "local" + "_asr"
    asr = _reload_asr(monkeypatch, backend=invalid_name, worker_mode="inproc")

    try:
        asr._resolve_asr_backend("cpu")
    except ValueError as exc:
        assert "qwen3-asr-1.7b" in str(exc)
        assert "anime-whisper" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


