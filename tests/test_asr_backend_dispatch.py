from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from asr.backends.base import BaseAsrBackend
from helpers import ASR_06B_BACKEND, ASR_17B_BACKEND


def _reload_asr(monkeypatch, *, backend: str):
    monkeypatch.setenv("ASR_BACKEND", backend)
    from asr import pipeline as asr
    return importlib.reload(asr)


def test_qwen3_asr_repo_backend_dispatch_uses_gpu_worker_local_backend(monkeypatch):
    asr = _reload_asr(monkeypatch, backend=ASR_17B_BACKEND)
    backend = asr._resolve_asr_backend("cpu")

    assert type(backend).__name__ == "LocalAsrBackend"
    assert isinstance(backend, BaseAsrBackend)
    assert backend.is_subprocess is False
    assert asr.get_backend_label() == ASR_17B_BACKEND


def test_legacy_asr_worker_mode_env_is_ignored(monkeypatch):
    monkeypatch.setenv("ASR_WORKER_MODE", "subprocess")
    asr = _reload_asr(monkeypatch, backend=ASR_06B_BACKEND)
    backend = asr._resolve_asr_backend("cpu")

    assert isinstance(backend, BaseAsrBackend)
    assert type(backend).__name__ == "LocalAsrBackend"
    assert backend.is_subprocess is False
    assert asr.get_backend_label() == ASR_06B_BACKEND


def test_qwen3_asr_default_runtime_mode_is_gpu_worker(monkeypatch):
    monkeypatch.delenv("ASR_WORKER_MODE", raising=False)
    monkeypatch.delenv("ASR_WORKER_MODE_BY_REPO", raising=False)
    from asr import pipeline as asr

    monkeypatch.setenv("ASR_BACKEND", ASR_06B_BACKEND)
    asr = importlib.reload(asr)
    backend_06b = asr._resolve_asr_backend("cpu")
    assert backend_06b.is_subprocess is False
    assert asr.get_backend_label() == ASR_06B_BACKEND

    monkeypatch.setenv("ASR_BACKEND", ASR_17B_BACKEND)
    asr = importlib.reload(asr)
    backend_17b = asr._resolve_asr_backend("cpu")
    assert backend_17b.is_subprocess is False
    assert asr.get_backend_label() == ASR_17B_BACKEND


def test_invalid_asr_backend_is_rejected(monkeypatch):
    asr = _reload_asr(monkeypatch, backend="unknown_backend")

    try:
        asr._resolve_asr_backend("cpu")
    except ValueError as exc:
        assert "Unsupported ASR_BACKEND" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_internal_asr_backend_names_are_rejected(monkeypatch):
    invalid_name = "local" + "_asr"
    asr = _reload_asr(monkeypatch, backend=invalid_name)

    try:
        asr._resolve_asr_backend("cpu")
    except ValueError as exc:
        assert ASR_06B_BACKEND in str(exc)
        assert ASR_17B_BACKEND in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_short_qwen_backend_aliases_are_rejected(monkeypatch):
    asr = _reload_asr(monkeypatch, backend="qwen3-asr-0.6b")

    try:
        asr._resolve_asr_backend("cpu")
    except ValueError as exc:
        assert "Unsupported ASR_BACKEND" in str(exc)
        assert ASR_06B_BACKEND in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_legacy_non_hf_repo_id_is_rejected(monkeypatch):
    legacy_repo = ASR_17B_BACKEND.removesuffix("-hf")
    asr = _reload_asr(monkeypatch, backend=legacy_repo)

    with pytest.raises(ValueError, match="Unsupported ASR_BACKEND"):
        asr._resolve_asr_backend("cpu")


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
        "OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        f"{ASR_06B_BACKEND}={checkpoint_06b},{ASR_17B_BACKEND}={checkpoint_17b}",
    )

    assert qwen.qwen_asr_repo_tag(ASR_17B_BACKEND) == "jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    assert (
        qwen.checkpoint_path_for_repo_env(
            repo_id=ASR_17B_BACKEND,
            mapping_env="OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        )
        == str(checkpoint_17b.resolve())
    )


def test_qwen_checkpoint_path_defaults_to_registry_when_env_is_absent(monkeypatch, tmp_path):
    from asr.backends import qwen

    checkpoint = tmp_path / "outer_edge_refiner_v1.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt"
    checkpoint.write_bytes(b"v1")
    monkeypatch.delenv("OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO", raising=False)

    path = qwen.checkpoint_path_for_repo_env(
        repo_id=ASR_17B_BACKEND,
        mapping_env="OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        default_mapping={ASR_17B_BACKEND: str(checkpoint)},
    )

    assert path.endswith("outer_edge_refiner_v1.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt")


def test_qwen_checkpoint_path_defaults_fall_back_to_resource_root(monkeypatch, tmp_path):
    from asr.backends import qwen

    runtime_root = tmp_path / "runtime"
    resource_root = tmp_path / "resource"
    relative = Path("src/boundary/checkpoints/outer_edge_refiner_v1.test.pt")
    checkpoint = resource_root / relative
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"bundled")

    def fake_runtime_path(path: Path) -> Path:
        candidate = Path(path)
        return candidate.resolve() if candidate.is_absolute() else (runtime_root / candidate).resolve()

    def fake_resource_path(path: Path) -> Path:
        candidate = Path(path)
        return candidate.resolve() if candidate.is_absolute() else (resource_root / candidate).resolve()

    monkeypatch.delenv("OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO", raising=False)
    monkeypatch.setattr(qwen, "runtime_path", fake_runtime_path)
    monkeypatch.setattr(qwen, "resource_path", fake_resource_path)

    path = qwen.checkpoint_path_for_repo_env(
        repo_id=ASR_17B_BACKEND,
        mapping_env="OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        default_mapping={ASR_17B_BACKEND: relative.as_posix()},
    )

    assert path == str(checkpoint.resolve())


def test_qwen_checkpoint_path_auto_uses_registered_scorer(monkeypatch, tmp_path):
    from asr.backends import qwen

    checkpoint_06b = tmp_path / "speech_island_scorer_v8.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf.pt"
    checkpoint_17b = tmp_path / "speech_island_scorer_v8.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt"
    checkpoint_06b.write_bytes(b"v8-06b")
    checkpoint_17b.write_bytes(b"v8-17b")
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
        "speech_island_scorer_v8.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf.pt"
    )
    assert path_17b.endswith(
        "speech_island_scorer_v8.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt"
    )


def test_qwen_checkpoint_path_mapping_requires_env(monkeypatch):
    from asr.backends import qwen

    monkeypatch.delenv("OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO", raising=False)

    with pytest.raises(RuntimeError, match="OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO is required"):
        qwen.checkpoint_path_for_repo_env(
            repo_id=ASR_17B_BACKEND,
            mapping_env="OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        )


def test_qwen_checkpoint_path_mapping_requires_selected_repo(monkeypatch, tmp_path):
    from asr.backends import qwen

    checkpoint_06b = tmp_path / "06b.pt"
    checkpoint_06b.write_bytes(b"06b")
    monkeypatch.setenv(
        "OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        f"{ASR_06B_BACKEND}={checkpoint_06b}",
    )

    with pytest.raises(RuntimeError, match="has no checkpoint for ASR repo"):
        qwen.checkpoint_path_for_repo_env(
            repo_id=ASR_17B_BACKEND,
            mapping_env="OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        )


def test_qwen_checkpoint_path_mapping_requires_existing_file(monkeypatch, tmp_path):
    from asr.backends import qwen

    missing = tmp_path / "missing.pt"
    monkeypatch.setenv(
        "OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        f"{ASR_17B_BACKEND}={missing}",
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        qwen.checkpoint_path_for_repo_env(
            repo_id=ASR_17B_BACKEND,
            mapping_env="OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        )


def test_qwen_checkpoint_path_mapping_rejects_malformed_env(monkeypatch):
    from asr.backends import qwen

    monkeypatch.setenv("OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO", "not-a-mapping-entry")

    with pytest.raises(RuntimeError, match="is malformed"):
        qwen.checkpoint_path_for_repo_env(
            repo_id=ASR_17B_BACKEND,
            mapping_env="OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        )


def test_checkpoint_metadata_rejects_legacy_non_hf_repo_id():
    from asr.backends import qwen

    with pytest.raises(ValueError, match="does not match"):
        qwen.validate_checkpoint_repo_id(
            ASR_17B_BACKEND.removesuffix("-hf"),
            ASR_17B_BACKEND,
            checkpoint_kind="test",
            metadata_key="metadata.ptm_repo_id",
        )


def test_local_backend_asr_batch_size_auto_and_numeric_override(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", ASR_17B_BACKEND)
    monkeypatch.setenv(
        "ASR_BATCH_SIZE_BY_REPO",
        f"{ASR_06B_BACKEND}=64,{ASR_17B_BACKEND}=32",
    )
    monkeypatch.setenv("ASR_BATCH_SIZE", "auto")

    from asr import local_backend

    # Batch size is resolved at call time (reads env on each call) so a
    # persistent worker honors per-job / per-retry batch sizes without reload.
    assert local_backend._resolve_asr_batch_size() == 32

    monkeypatch.setenv("ASR_BATCH_SIZE", "7")
    assert local_backend._resolve_asr_batch_size() == 7
