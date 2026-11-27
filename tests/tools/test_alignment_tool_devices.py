"""Alignment sessions and feature/head loaders must validate CUDA before loading models."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asr import alignment, encoder_features, pipeline
from tools.align.qwen_asr_session import QwenAsrSession


@pytest.fixture
def model_loaders(monkeypatch):
    import torch
    from asr.backends import qwen
    from utils import gpu_safety, model_paths

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = {"resolve": [], "processor": [], "model": [], "cap": []}

    def resolve(*args, **kwargs):
        calls["resolve"].append((args, kwargs))
        return "sample-model"

    def processor(spec):
        calls["processor"].append(spec)
        return object()

    def model(spec, **kwargs):
        calls["model"].append((spec, kwargs))
        return SimpleNamespace(eval=lambda: None)

    monkeypatch.setattr(qwen, "active_qwen_asr_model_path", lambda: "")
    monkeypatch.setattr(qwen, "active_qwen_asr_model_id", lambda: "sample/model")
    monkeypatch.setattr(model_paths, "resolve_model_spec", resolve)
    monkeypatch.setattr(encoder_features, "resolve_model_spec", resolve)
    monkeypatch.setattr(gpu_safety, "apply_vram_safety_cap", calls["cap"].append)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=processor),
            AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=model),
        ),
    )
    return calls


def test_shared_session_checks_cuda_before_model_lookup(model_loaders):
    with pytest.raises(SystemExit, match="CUDA"):
        QwenAsrSession.load()
    assert not any(model_loaders.values())


def test_shared_session_preserves_offline_lookup_and_vram_cap(monkeypatch, model_loaders):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    session = QwenAsrSession.load(download=False, vram_cap=0.8)
    assert session.device.type == "cuda"
    assert session.dtype == torch.bfloat16
    assert model_loaders["model"] == [("sample-model", {"dtype": torch.bfloat16, "device_map": "cuda"})]
    assert model_loaders["cap"] == [0.8]
    assert model_loaders["resolve"][0][1]["download"] is False


def test_shared_session_cuda_load_failure_propagates_without_cpu_retry(monkeypatch, model_loaders):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def fail(spec, **kwargs):
        model_loaders["model"].append((spec, kwargs))
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(sys.modules["transformers"].AutoModelForMultimodalLM, "from_pretrained", fail)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        QwenAsrSession.load()
    assert model_loaders["model"] == [("sample-model", {"dtype": torch.bfloat16, "device_map": "cuda"})]


@pytest.mark.parametrize(
    "requested, available",
    [
        ("auto", False),
        ("cuda", False),
        ("cuda:0", False),
        ("cpu", False),
        ("cpu", True),
    ],
)
def test_feature_encoder_requires_cuda_before_model_lookup(model_loaders, monkeypatch, requested, available):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    with pytest.raises(RuntimeError, match="CUDA"):
        encoder_features.Qwen3AsrEncoder(encoder_features.EncoderFeatureConfig(device=requested))
    assert not any(model_loaders.values())


@pytest.mark.parametrize("requested", ["cuda", "cuda:1"])
def test_feature_encoder_keeps_the_requested_cuda_device(model_loaders, monkeypatch, requested):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    encoder = encoder_features.Qwen3AsrEncoder(
        encoder_features.EncoderFeatureConfig(device=requested, download=False)
    )
    assert str(encoder.device) == requested
    assert model_loaders["model"] == [("sample-model", {"dtype": torch.float16, "device_map": requested})]
    assert model_loaders["resolve"][0][1]["download"] is False


def test_runtime_feature_loader_requires_cuda_before_model_lookup(model_loaders):
    with pytest.raises(RuntimeError, match="CUDA"):
        pipeline._load_asr_model_for_features()
    assert not any(model_loaders.values())


def test_default_head_requires_cuda_before_checkpoint_lookup(model_loaders, monkeypatch):
    def unexpected_lookup(_path):
        pytest.fail("checkpoint lookup ran without CUDA")

    monkeypatch.setattr(alignment, "resolve_alignment_head_path", unexpected_lookup)
    with pytest.raises(RuntimeError, match="CUDA"):
        alignment.AlignmentHead.load("sample-head.pt")
