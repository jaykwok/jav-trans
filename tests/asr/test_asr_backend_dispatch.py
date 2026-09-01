from __future__ import annotations

from pathlib import Path

import pytest

from asr.backends.base import BaseAsrBackend
from helpers import ASR_17B_BACKEND, RETIRED_06B_BACKEND


def _asr(monkeypatch, *, backend: str):
    """Select the backend and hand back the stage.

    No reload: `ASR_BACKEND` is read inside `_resolve_asr_backend`, which is the
    property that lets one persistent worker serve jobs with different backends.
    Reloading here would have hidden a regression in exactly that property.
    """
    monkeypatch.setenv("ASR_BACKEND", backend)
    from asr import pipeline as asr

    return asr


def test_qwen3_asr_repo_backend_dispatch_uses_gpu_worker_local_backend(monkeypatch):
    asr = _asr(monkeypatch, backend=ASR_17B_BACKEND)
    backend = asr._resolve_asr_backend("cpu")

    assert type(backend).__name__ == "LocalAsrBackend"
    assert isinstance(backend, BaseAsrBackend)
    assert backend.is_subprocess is False
    assert asr.get_backend_label() == ASR_17B_BACKEND


def test_legacy_asr_worker_mode_env_is_ignored(monkeypatch):
    monkeypatch.setenv("ASR_WORKER_MODE", "subprocess")
    asr = _asr(monkeypatch, backend=ASR_17B_BACKEND)
    backend = asr._resolve_asr_backend("cpu")

    assert isinstance(backend, BaseAsrBackend)
    assert type(backend).__name__ == "LocalAsrBackend"
    assert backend.is_subprocess is False
    assert asr.get_backend_label() == ASR_17B_BACKEND


def test_qwen3_asr_default_runtime_mode_is_gpu_worker(monkeypatch):
    monkeypatch.delenv("ASR_WORKER_MODE", raising=False)
    monkeypatch.delenv("ASR_WORKER_MODE_BY_REPO", raising=False)
    asr = _asr(monkeypatch, backend=ASR_17B_BACKEND)
    backend_17b = asr._resolve_asr_backend("cpu")
    assert backend_17b.is_subprocess is False
    assert asr.get_backend_label() == ASR_17B_BACKEND


def test_retired_06b_backend_is_rejected(monkeypatch):
    # A stale `.env` from before 2026-07-31 still names the 0.6B repo.
    # Failing loudly beats silently transcribing with a model the user
    # did not pick, and beats a KeyError deep inside a batch-size lookup.
    asr = _asr(monkeypatch, backend=RETIRED_06B_BACKEND)

    with pytest.raises(ValueError, match="Unsupported ASR_BACKEND"):
        asr._resolve_asr_backend("cpu")


def test_invalid_asr_backend_is_rejected(monkeypatch):
    asr = _asr(monkeypatch, backend="unknown_backend")

    try:
        asr._resolve_asr_backend("cpu")
    except ValueError as exc:
        assert "Unsupported ASR_BACKEND" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_internal_asr_backend_names_are_rejected(monkeypatch):
    invalid_name = "local" + "_asr"
    asr = _asr(monkeypatch, backend=invalid_name)

    try:
        asr._resolve_asr_backend("cpu")
    except ValueError as exc:
        assert ASR_17B_BACKEND in str(exc)
        assert RETIRED_06B_BACKEND not in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_short_qwen_backend_aliases_are_rejected(monkeypatch):
    asr = _asr(monkeypatch, backend="qwen3-asr-1.7b")

    try:
        asr._resolve_asr_backend("cpu")
    except ValueError as exc:
        assert "Unsupported ASR_BACKEND" in str(exc)
        assert ASR_17B_BACKEND in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_legacy_non_hf_repo_id_is_rejected(monkeypatch):
    legacy_repo = ASR_17B_BACKEND.removesuffix("-hf")
    asr = _asr(monkeypatch, backend=legacy_repo)

    with pytest.raises(ValueError, match="Unsupported ASR_BACKEND"):
        asr._resolve_asr_backend("cpu")


def test_qwen_asr_batch_size_auto_uses_repo_table(monkeypatch):
    from asr.backends import qwen

    monkeypatch.setenv("ASR_BATCH_SIZE_BY_REPO", f"{ASR_17B_BACKEND}=32")
    assert qwen.qwen_asr_default_batch_size(ASR_17B_BACKEND) == 32

    # A stale table naming the dropped repo must fail, not be ignored.
    monkeypatch.setenv(
        "ASR_BATCH_SIZE_BY_REPO",
        f"{RETIRED_06B_BACKEND}=64,{ASR_17B_BACKEND}=32",
    )
    with pytest.raises(ValueError, match="Invalid ASR_BATCH_SIZE_BY_REPO repo"):
        qwen.qwen_asr_default_batch_size(ASR_17B_BACKEND)


def test_qwen_asr_minimum_physical_vram_uses_repo_table(monkeypatch):
    from asr.backends import qwen

    monkeypatch.setenv("ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO", f"{ASR_17B_BACKEND}=6144")
    assert qwen.qwen_asr_min_physical_vram_mb(ASR_17B_BACKEND) == 6144

    monkeypatch.setenv(
        "ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO",
        f"{RETIRED_06B_BACKEND}=4096,{ASR_17B_BACKEND}=6144",
    )
    with pytest.raises(
        ValueError, match="Invalid ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO repo"
    ):
        qwen.qwen_asr_min_physical_vram_mb(ASR_17B_BACKEND)


def test_retired_checkpoint_mapping_machinery_is_gone():
    # The per-repo checkpoint registries and their env resolution went with the
    # boundary chain on 2026-07-31. The alignment head binds through
    # ASR_ALIGNMENT_HEAD_PATH instead; nothing may quietly resurrect the
    # <repo_id>=<path> mapping surface.
    from asr.backends import qwen

    for name in (
        "checkpoint_path_for_repo_env",
        "repo_path_mapping",
        "repo_checkpoint_path",
        "validate_checkpoint_repo_id",
        "SMALL_MODEL_CHECKPOINT_ROOT",
        "BOUNDARY_PIPELINE_STATUS_BY_REPO",
        "DEFAULT_OUTER_EDGE_REFINER_CHECKPOINT_BY_REPO",
        "DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO",
        "DEFAULT_INNER_EDGE_REFINER_CHECKPOINT_BY_REPO",
        "DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO",
        "DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO",
        "DEFAULT_SPEECH_BOUNDARY_PROPOSAL_CHECKPOINT_BY_REPO",
    ):
        assert not hasattr(qwen, name), name


def test_local_backend_asr_batch_size_auto_and_numeric_override(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", ASR_17B_BACKEND)
    monkeypatch.setenv("ASR_BATCH_SIZE_BY_REPO", f"{ASR_17B_BACKEND}=32")
    monkeypatch.setenv("ASR_BATCH_SIZE", "auto")

    from asr import local_backend

    # Batch size is resolved at call time (reads env on each call) so a
    # persistent worker honors per-job / per-retry batch sizes without reload.
    assert local_backend._resolve_asr_batch_size() == 32

    monkeypatch.setenv("ASR_BATCH_SIZE", "7")
    assert local_backend._resolve_asr_batch_size() == 7
