from __future__ import annotations

import importlib

import pytest

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


def test_qwen_checkpoint_path_mapping_uses_repo_id_keys(monkeypatch, tmp_path):
    from asr.backends import qwen

    checkpoint_06b = tmp_path / "06b.pt"
    checkpoint_17b = tmp_path / "17b.pt"
    checkpoint_06b.write_bytes(b"06b")
    checkpoint_17b.write_bytes(b"17b")
    monkeypatch.setenv("ASR_BACKEND", ASR_17B_BACKEND)
    monkeypatch.setenv(
        "BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        f"{ASR_06B_BACKEND}={checkpoint_06b},{ASR_17B_BACKEND}={checkpoint_17b}",
    )

    assert qwen.qwen_asr_repo_tag(ASR_17B_BACKEND) == "jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame"
    assert (
        qwen.checkpoint_path_for_repo_env(
            repo_id=ASR_17B_BACKEND,
            mapping_env="BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        )
        == str(checkpoint_17b.resolve())
    )


def test_qwen_checkpoint_path_defaults_to_registry_when_env_is_absent(monkeypatch, tmp_path):
    from asr.backends import qwen

    checkpoint = tmp_path / "boundary_edge_refiner_v7.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt"
    checkpoint.write_bytes(b"v6")
    monkeypatch.delenv("BOUNDARY_REFINER_MODEL_PATH_BY_REPO", raising=False)

    path = qwen.checkpoint_path_for_repo_env(
        repo_id=ASR_17B_BACKEND,
        mapping_env="BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        default_mapping={ASR_17B_BACKEND: str(checkpoint)},
    )

    assert path.endswith("boundary_edge_refiner_v7.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt")


def test_qwen_checkpoint_path_auto_uses_registered_scorer(monkeypatch, tmp_path):
    from asr.backends import qwen

    checkpoint_06b = tmp_path / "speech_boundary_ja_frame_boundary_scorer_v7.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt"
    checkpoint_17b = tmp_path / "speech_boundary_ja_frame_boundary_scorer_v7.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt"
    checkpoint_06b.write_bytes(b"v6-06b")
    checkpoint_17b.write_bytes(b"v6-17b")
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO", "auto")

    default_mapping = {
        ASR_06B_BACKEND: str(checkpoint_06b),
        ASR_17B_BACKEND: str(checkpoint_17b),
    }
    path_06b = qwen.checkpoint_path_for_repo_env(
        repo_id=ASR_06B_BACKEND,
        mapping_env="SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO",
        default_mapping=default_mapping,
    )
    path_17b = qwen.checkpoint_path_for_repo_env(
        repo_id=ASR_17B_BACKEND,
        mapping_env="SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO",
        default_mapping=default_mapping,
    )

    assert path_06b.endswith(
        "speech_boundary_ja_frame_boundary_scorer_v7.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt"
    )
    assert path_17b.endswith(
        "speech_boundary_ja_frame_boundary_scorer_v7.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt"
    )


def test_qwen_checkpoint_path_mapping_requires_env(monkeypatch):
    from asr.backends import qwen

    monkeypatch.delenv("BOUNDARY_REFINER_MODEL_PATH_BY_REPO", raising=False)

    with pytest.raises(RuntimeError, match="BOUNDARY_REFINER_MODEL_PATH_BY_REPO is required"):
        qwen.checkpoint_path_for_repo_env(
            repo_id=ASR_17B_BACKEND,
            mapping_env="BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        )


def test_qwen_checkpoint_path_mapping_requires_selected_repo(monkeypatch, tmp_path):
    from asr.backends import qwen

    checkpoint_06b = tmp_path / "06b.pt"
    checkpoint_06b.write_bytes(b"06b")
    monkeypatch.setenv(
        "BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        f"{ASR_06B_BACKEND}={checkpoint_06b}",
    )

    with pytest.raises(RuntimeError, match="has no checkpoint for ASR repo"):
        qwen.checkpoint_path_for_repo_env(
            repo_id=ASR_17B_BACKEND,
            mapping_env="BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        )


def test_qwen_checkpoint_path_mapping_requires_existing_file(monkeypatch, tmp_path):
    from asr.backends import qwen

    missing = tmp_path / "missing.pt"
    monkeypatch.setenv(
        "BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        f"{ASR_17B_BACKEND}={missing}",
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        qwen.checkpoint_path_for_repo_env(
            repo_id=ASR_17B_BACKEND,
            mapping_env="BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        )


def test_qwen_checkpoint_path_mapping_rejects_malformed_env(monkeypatch):
    from asr.backends import qwen

    monkeypatch.setenv("BOUNDARY_REFINER_MODEL_PATH_BY_REPO", "not-a-mapping-entry")

    with pytest.raises(RuntimeError, match="is malformed"):
        qwen.checkpoint_path_for_repo_env(
            repo_id=ASR_17B_BACKEND,
            mapping_env="BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        )


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
