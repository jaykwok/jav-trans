from __future__ import annotations

import sys
import types

import pytest

from utils import hf_progress
from utils import model_paths


def test_model_dir_name_uses_huggingface_repo_name():
    assert model_paths.model_dir_name("Qwen/Qwen3-ASR-1.7B") == "Qwen-Qwen3-ASR-1.7B"
    assert (
        model_paths.model_dir_name("jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf")
        == "jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    )
    assert (
        model_paths.model_dir_name("jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf")
        == "jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"
    )


def test_resolve_model_spec_prefers_models_repo_name(monkeypatch, tmp_path):
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")

    local_model = tmp_path / "models" / "jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"
    local_model.mkdir(parents=True)
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    (local_model / "model.safetensors").write_bytes(b"weights")

    assert (
        model_paths.resolve_model_spec(None, "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf")
        == str(local_model.resolve())
    )


def test_resolve_model_spec_uses_bundled_model_when_runtime_model_missing(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime"
    resource_root = tmp_path / "resource"
    monkeypatch.setattr(model_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", runtime_root)
    monkeypatch.setattr(model_paths, "MODELS_ROOT", runtime_root / "models")
    monkeypatch.setattr(model_paths, "RESOURCE_ROOT", resource_root)
    monkeypatch.setattr(model_paths, "BUNDLED_MODELS_ROOT", resource_root / "models")

    bundled_model = resource_root / "models" / "jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"
    bundled_model.mkdir(parents=True)
    (bundled_model / "config.json").write_text("{}", encoding="utf-8")
    (bundled_model / "model.safetensors").write_bytes(b"weights")

    assert (
        model_paths.resolve_model_spec(None, "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf")
        == str(bundled_model.resolve())
    )


def test_resolve_model_spec_uses_bundled_model_for_default_explicit_path(
    monkeypatch,
    tmp_path,
):
    runtime_root = tmp_path / "runtime"
    resource_root = tmp_path / "resource"
    repo_id = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"
    monkeypatch.setattr(model_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", runtime_root)
    monkeypatch.setattr(model_paths, "MODELS_ROOT", runtime_root / "models")
    monkeypatch.setattr(model_paths, "RESOURCE_ROOT", resource_root)
    monkeypatch.setattr(model_paths, "BUNDLED_MODELS_ROOT", resource_root / "models")

    bundled_model = resource_root / "models" / "jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"
    bundled_model.mkdir(parents=True)
    (bundled_model / "config.json").write_text("{}", encoding="utf-8")
    (bundled_model / "model.safetensors").write_bytes(b"weights")

    assert (
        model_paths.resolve_model_spec(
            "models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf",
            repo_id,
            download=True,
        )
        == str(bundled_model.resolve())
    )

    status = model_paths.model_spec_status(
        "models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf",
        repo_id,
    )
    assert status["present"] is True
    assert status["path"] == str(bundled_model.resolve())


def test_resolve_model_spec_downloads_to_models_repo_name(monkeypatch, tmp_path):
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")

    calls = []

    def fake_download(repo_id, target_dir, **kwargs):
        calls.append((repo_id, target_dir, kwargs))
        return str(target_dir)

    monkeypatch.setattr(model_paths, "_download_snapshot", fake_download)

    resolved = model_paths.resolve_model_spec(
        None,
        "Qwen/Qwen3-ASR-1.7B",
        download=True,
    )

    assert resolved == str(tmp_path / "models" / "Qwen-Qwen3-ASR-1.7B")
    assert calls == [
        (
            "Qwen/Qwen3-ASR-1.7B",
            tmp_path / "models" / "Qwen-Qwen3-ASR-1.7B",
            {"revision": None, "allow_patterns": None, "ignore_patterns": None},
        )
    ]


def test_model_spec_status_reports_missing_explicit_download_target(monkeypatch, tmp_path):
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")

    status = model_paths.model_spec_status(
        "models/custom-asr",
        "owner/repo",
    )

    assert status["present"] is False
    assert status["path"] == str((tmp_path / "models" / "custom-asr").resolve())
    assert status["checked_paths"][0] == str((tmp_path / "models" / "custom-asr").resolve())


def test_model_spec_status_treats_config_only_directory_as_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")

    partial_model = tmp_path / "models" / "owner-repo"
    partial_model.mkdir(parents=True)
    (partial_model / "config.json").write_text("{}", encoding="utf-8")

    status = model_paths.model_spec_status(None, "owner/repo")

    assert status["present"] is False
    assert status["path"] == str(partial_model.resolve())


def test_model_spec_status_finds_canonical_model_after_missing_explicit_path(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")

    local_model = tmp_path / "models" / "owner-repo"
    local_model.mkdir(parents=True)
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    (local_model / "model.safetensors").write_bytes(b"weights")

    status = model_paths.model_spec_status(
        "models/missing",
        "owner/repo",
        download=False,
    )

    assert status["present"] is True
    assert status["path"] == str(local_model.resolve())


def test_incomplete_indexed_safetensors_triggers_redownload(monkeypatch, tmp_path):
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")

    local_model = tmp_path / "models" / "Qwen-Qwen3-ASR-1.7B"
    local_model.mkdir(parents=True)
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    (local_model / "model.safetensors.index.json").write_text(
        """
        {
          "weight_map": {
            "layer_a": "model-00001-of-00002.safetensors",
            "layer_b": "model-00002-of-00002.safetensors"
          }
        }
        """,
        encoding="utf-8",
    )
    (local_model / "model-00002-of-00002.safetensors").write_bytes(b"partial")
    calls = []

    def fake_download(repo_id, target_dir, **kwargs):
        calls.append((repo_id, target_dir, kwargs))
        (target_dir / "model-00001-of-00002.safetensors").write_bytes(b"rest")
        return str(target_dir)

    monkeypatch.setattr(model_paths, "_download_snapshot", fake_download)

    resolved = model_paths.resolve_model_spec(
        None,
        "Qwen/Qwen3-ASR-1.7B",
        download=True,
    )

    assert resolved == str(local_model.resolve())
    assert calls


def test_sharded_safetensors_without_index_triggers_redownload(monkeypatch, tmp_path):
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")

    local_model = tmp_path / "models" / "Qwen-Qwen3-ASR-1.7B"
    local_model.mkdir(parents=True)
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    (local_model / "model-00001-of-00002.safetensors").write_bytes(b"partial")
    calls = []

    def fake_download(repo_id, target_dir, **kwargs):
        calls.append((repo_id, target_dir, kwargs))
        (target_dir / "model.safetensors.index.json").write_text(
            """
            {
              "weight_map": {
                "layer_a": "model-00001-of-00002.safetensors",
                "layer_b": "model-00002-of-00002.safetensors"
              }
            }
            """,
            encoding="utf-8",
        )
        (target_dir / "model-00002-of-00002.safetensors").write_bytes(b"rest")
        return str(target_dir)

    monkeypatch.setattr(model_paths, "_download_snapshot", fake_download)

    resolved = model_paths.resolve_model_spec(
        None,
        "Qwen/Qwen3-ASR-1.7B",
        download=True,
    )

    assert resolved == str(local_model.resolve())
    assert calls


def test_resolve_model_spec_redownloads_config_only_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")

    partial_model = tmp_path / "models" / "Qwen-Qwen3-ASR-1.7B"
    partial_model.mkdir(parents=True)
    (partial_model / "config.json").write_text("{}", encoding="utf-8")
    calls = []

    def fake_download(repo_id, target_dir, **kwargs):
        calls.append((repo_id, target_dir, kwargs))
        (target_dir / "model.safetensors").write_bytes(b"weights")
        return str(target_dir)

    monkeypatch.setattr(model_paths, "_download_snapshot", fake_download)

    resolved = model_paths.resolve_model_spec(
        None,
        "Qwen/Qwen3-ASR-1.7B",
        download=True,
    )

    assert resolved == str(partial_model.resolve())
    assert calls


def test_download_snapshot_uses_default_endpoint_when_env_is_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_ENDPOINT", "")
    monkeypatch.setattr(hf_progress, "snapshot_download_kwargs", lambda _fn: {})
    monkeypatch.setattr(hf_progress, "fallback_start", lambda _repo_id: "token")
    monkeypatch.setattr(hf_progress, "fallback_done", lambda _token: None)

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = kwargs["local_dir"]
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "model.safetensors").write_bytes(b"weights")

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    result = model_paths._download_snapshot("owner/repo", tmp_path / "model")

    assert result == str(tmp_path / "model")
    assert calls[0]["endpoint"] == model_paths.DEFAULT_HF_ENDPOINT
    assert "optimizer.pt" in calls[0]["ignore_patterns"]
    assert "**/optimizer.pt" in calls[0]["ignore_patterns"]
    assert "HF_ENDPOINT" not in model_paths.os.environ


def test_download_snapshot_merges_inference_ignore_patterns(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(hf_progress, "snapshot_download_kwargs", lambda _fn: {})
    monkeypatch.setattr(hf_progress, "fallback_start", lambda _repo_id: "token")
    monkeypatch.setattr(hf_progress, "fallback_done", lambda _token: None)

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = kwargs["local_dir"]
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "model.safetensors").write_bytes(b"weights")

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    model_paths._download_snapshot(
        "owner/repo",
        tmp_path / "model",
        ignore_patterns=["*.ckpt"],
    )

    assert "optimizer.pt" in calls[0]["ignore_patterns"]
    assert "*.ckpt" in calls[0]["ignore_patterns"]


def test_download_snapshot_keeps_existing_partial_dir_on_failure(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(hf_progress, "snapshot_download_kwargs", lambda _fn: {})
    monkeypatch.setattr(hf_progress, "fallback_start", lambda _repo_id: "token")
    monkeypatch.setattr(hf_progress, "fallback_error", lambda _token, _exc: None)

    target = tmp_path / "model"
    target.mkdir()
    keep = target / "keep.txt"
    keep.write_text("user file", encoding="utf-8")

    def fake_snapshot_download(**kwargs):
        local_dir = kwargs["local_dir"]
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "model-00001-of-00002.safetensors").write_bytes(b"partial")
        raise RuntimeError("boom")

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    with pytest.raises(RuntimeError, match="boom"):
        model_paths._download_snapshot("owner/repo", target)

    assert target.exists()
    assert keep.read_text(encoding="utf-8") == "user file"
    assert (target / "model-00001-of-00002.safetensors").exists()


def test_download_snapshot_rejects_incomplete_success(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(hf_progress, "snapshot_download_kwargs", lambda _fn: {})
    monkeypatch.setattr(hf_progress, "fallback_start", lambda _repo_id: "token")
    monkeypatch.setattr(hf_progress, "fallback_done", lambda _token: None)

    target = tmp_path / "model"

    def fake_snapshot_download(**kwargs):
        local_dir = kwargs["local_dir"]
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "model-00001-of-00002.safetensors").write_bytes(b"partial")

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    with pytest.raises(RuntimeError, match="did not produce a complete local model"):
        model_paths._download_snapshot("owner/repo", target)

    assert (target / "model-00001-of-00002.safetensors").exists()


def test_download_snapshot_rejects_hf_endpoint_without_protocol(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_ENDPOINT", "hf-mirror.com")
    monkeypatch.setattr(hf_progress, "snapshot_download_kwargs", lambda _fn: {})
    monkeypatch.setattr(hf_progress, "fallback_start", lambda _repo_id: "token")
    monkeypatch.setattr(hf_progress, "fallback_error", lambda _token, _exc: None)

    fake_module = types.SimpleNamespace(snapshot_download=lambda **_kwargs: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    with pytest.raises(ValueError, match="HF_ENDPOINT must be empty or a full URL"):
        model_paths._download_snapshot("owner/repo", tmp_path / "model")


def test_huggingface_runtime_cache_is_not_models_top_level():
    assert model_paths.HF_RUNTIME_CACHE_ROOT == model_paths.PROJECT_ROOT / "tmp" / "cache" / "hf"
    assert model_paths.HF_RUNTIME_CACHE_ROOT.parent != model_paths.MODELS_ROOT
