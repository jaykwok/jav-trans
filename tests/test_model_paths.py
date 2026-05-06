from __future__ import annotations

import sys
import types

import pytest

from utils import hf_progress
from utils import model_paths


def test_model_dir_name_uses_huggingface_repo_name():
    assert model_paths.model_dir_name("Qwen/Qwen3-ASR-1.7B") == "Qwen-Qwen3-ASR-1.7B"
    assert model_paths.model_dir_name("litagin/anime-whisper") == "litagin-anime-whisper"
    assert model_paths.model_dir_name("efwkjn/whisper-ja-1.5B") == "efwkjn-whisper-ja-1.5B"
    assert (
        model_paths.model_dir_name("efwkjn/whisper-ja-anime-v0.3")
        == "efwkjn-whisper-ja-anime-v0.3"
    )


def test_resolve_model_spec_prefers_models_repo_name(monkeypatch, tmp_path):
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")

    local_model = tmp_path / "models" / "litagin-anime-whisper"
    local_model.mkdir(parents=True)
    (local_model / "config.json").write_text("{}", encoding="utf-8")

    assert (
        model_paths.resolve_model_spec(None, "litagin/anime-whisper")
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

    bundled_model = resource_root / "models" / "efwkjn-whisper-ja-anime-v0.3"
    bundled_model.mkdir(parents=True)
    (bundled_model / "config.json").write_text("{}", encoding="utf-8")

    assert (
        model_paths.resolve_model_spec(None, "efwkjn/whisper-ja-anime-v0.3")
        == str(bundled_model.resolve())
    )


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


def test_incomplete_indexed_safetensors_triggers_redownload(monkeypatch, tmp_path):
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_paths, "MODELS_ROOT", tmp_path / "models")

    local_model = tmp_path / "models" / "Qwen-Qwen3-ASR-1.7B"
    local_model.mkdir(parents=True)
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


def test_download_snapshot_uses_default_endpoint_when_env_is_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_ENDPOINT", "")
    monkeypatch.setattr(hf_progress, "snapshot_download_kwargs", lambda _fn: {})
    monkeypatch.setattr(hf_progress, "fallback_start", lambda _repo_id: "token")
    monkeypatch.setattr(hf_progress, "fallback_done", lambda _token: None)

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    result = model_paths._download_snapshot("owner/repo", tmp_path / "model")

    assert result == str(tmp_path / "model")
    assert calls[0]["endpoint"] == model_paths.DEFAULT_HF_ENDPOINT
    assert "HF_ENDPOINT" not in model_paths.os.environ


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
    assert model_paths.HF_RUNTIME_CACHE_ROOT == model_paths.PROJECT_ROOT / "temp" / "hf-cache"
    assert model_paths.HF_RUNTIME_CACHE_ROOT.parent != model_paths.MODELS_ROOT
