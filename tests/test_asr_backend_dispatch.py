from __future__ import annotations

import importlib

from asr.backends.base import BaseAsrBackend
from helpers import ASR_06B_BACKEND, ASR_17B_BACKEND


def _reload_asr(monkeypatch, *, backend: str, worker_mode: str):
    monkeypatch.setenv("ASR_BACKEND", backend)
    monkeypatch.setenv("ASR_WORKER_MODE", worker_mode)
    from asr import pipeline as asr
    return importlib.reload(asr)


def test_qwen3_asr_repo_backend_dispatch_uses_local_backend(monkeypatch):
    asr = _reload_asr(monkeypatch, backend=ASR_17B_BACKEND, worker_mode="inproc")
    backend = asr._resolve_asr_backend("cpu")

    assert type(backend).__name__ == "LocalAsrBackend"
    assert isinstance(backend, BaseAsrBackend)
    assert backend.is_subprocess is False
    assert asr.get_backend_label() == f"{ASR_17B_BACKEND} (inproc)"


def test_qwen3_asr_repo_backend_dispatch_uses_subprocess_backend(monkeypatch):
    asr = _reload_asr(monkeypatch, backend=ASR_06B_BACKEND, worker_mode="subprocess")
    backend = asr._resolve_asr_backend("cpu")

    assert isinstance(backend, BaseAsrBackend)
    assert type(backend).__name__ == "SubprocessAsrBackend"
    assert backend.is_subprocess is True
    assert asr.get_backend_label() == f"{ASR_06B_BACKEND} (subprocess worker)"


def test_invalid_worker_mode_is_rejected(monkeypatch):
    asr = _reload_asr(monkeypatch, backend=ASR_17B_BACKEND, worker_mode="invalid")

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
        assert ASR_06B_BACKEND in str(exc)
        assert ASR_17B_BACKEND in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_short_qwen_backend_aliases_are_rejected(monkeypatch):
    asr = _reload_asr(monkeypatch, backend="qwen3-asr-0.6b", worker_mode="inproc")

    try:
        asr._resolve_asr_backend("cpu")
    except ValueError as exc:
        assert "Unsupported ASR_BACKEND" in str(exc)
        assert ASR_06B_BACKEND in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_qwen_asr_batch_size_auto_uses_repo_table(monkeypatch):
    from asr.backends import qwen

    monkeypatch.setenv(
        "ASR_BATCH_SIZE_BY_REPO",
        f"{ASR_06B_BACKEND}=64,{ASR_17B_BACKEND}=32",
    )

    assert qwen.qwen_asr_default_batch_size(ASR_06B_BACKEND) == 64
    assert qwen.qwen_asr_default_batch_size(ASR_17B_BACKEND) == 32


def test_local_backend_asr_batch_size_auto_and_numeric_override(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", ASR_17B_BACKEND)
    monkeypatch.setenv(
        "ASR_BATCH_SIZE_BY_REPO",
        f"{ASR_06B_BACKEND}=64,{ASR_17B_BACKEND}=32",
    )
    monkeypatch.setenv("ASR_BATCH_SIZE", "auto")

    from asr import local_backend

    reloaded = importlib.reload(local_backend)
    assert reloaded.ASR_BATCH_SIZE == 32

    monkeypatch.setenv("ASR_BATCH_SIZE", "7")
    reloaded = importlib.reload(local_backend)
    assert reloaded.ASR_BATCH_SIZE == 7


