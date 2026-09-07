"""Real batch-tool CLIs must select their device before loading a model."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
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
    calls = {"resolve": [], "processor": [], "model": [], "cap": [], "heads": []}

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
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=processor),
        AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=model),
    ))
    return calls


_CLI_CASES = [
    ("build_real_alignment_lines", ("--manifest",), ("--checkpoint",), "--output"),
    ("measure_pregate_dropped_audio", ("--dataset",), ("--checkpoint",), "--output"),
    ("build_ctc_ab_jav_predictions", ("--answers", "--manifest"), ("--checkpoint-a", "--checkpoint-b"), "--output"),
    ("build_real_alignment_manifest", ("--dataset",), (), "--output-dir"),
    ("evaluate_pregate_loss", ("--dataset",), ("--checkpoint",), "--output"),
]


@pytest.fixture(params=_CLI_CASES, ids=[case[0] for case in _CLI_CASES])
def batch_cli(request, tmp_path, monkeypatch, model_loaders):
    name, input_flags, checkpoint_flags, output_flag = request.param
    module = importlib.import_module(f"tools.align.{name}")
    if hasattr(module, "apply_vram_safety_cap"):
        monkeypatch.setattr(module, "apply_vram_safety_cap", model_loaders["cap"].append)
    if hasattr(module, "AlignmentHead"):
        def load_head(_path, *, device):
            model_loaders["heads"].append(str(device))
            return object()

        monkeypatch.setattr(module, "AlignmentHead", SimpleNamespace(load=load_head))
    rows = tmp_path / "sample-input.jsonl"
    rows.write_text(json.dumps({
        "source_id": "sample-a", "source_partition": "test", "sample_id": "sample-a-0",
        "example_id": "sample-a-0", "row_id": "sample-a-0", "audio": "sample-a.wav",
        "text": "あ", "line_start_s": 0.0, "line_duration_s": 1.0,
        "provenance": "real_omni_joint",
    }) + "\n", "utf-8")
    checkpoint = tmp_path / "head.pt"
    checkpoint.write_bytes(b"mocked checkpoint")
    # Exercise the CLI and device handoff without GPU allocation or media.
    # Model/head constructors are mocked; unsupported-rate input is skipped.
    monkeypatch.setattr(module, "load_audio_16k_mono", lambda *_args: (np.zeros(0, dtype=np.float32), 8000))
    output = tmp_path / ("result" if output_flag == "--output-dir" else "result.jsonl")
    argv = [name, output_flag, str(output)]
    for flag in input_flags:
        argv.extend([flag, str(rows)])
    for flag in checkpoint_flags:
        argv.extend([flag, str(checkpoint)])
    return module, argv, output, len(checkpoint_flags)


def test_batch_cli_rejects_unavailable_cuda_before_model_lookup(batch_cli, model_loaders, monkeypatch):
    module, argv, output, _ = batch_cli
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="CUDA"):
        module.main()
    assert not any(model_loaders.values())
    assert not output.exists()


def test_batch_cli_places_models_and_heads_on_cuda(batch_cli, model_loaders, monkeypatch):
    import torch

    module, argv, output, head_count = batch_cli
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert model_loaders["model"] == [("sample-model", {"dtype": torch.bfloat16, "device_map": "cuda"})]
    assert model_loaders["cap"] == [0.95]
    assert model_loaders["heads"] == ["cuda"] * head_count
    assert output.exists()


def test_batch_cli_cuda_load_failure_propagates_without_cpu_retry(batch_cli, model_loaders, monkeypatch):
    import torch

    module, argv, output, _ = batch_cli
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", argv)

    def fail(spec, **kwargs):
        model_loaders["model"].append((spec, kwargs))
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(sys.modules["transformers"].AutoModelForMultimodalLM, "from_pretrained", fail)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        module.main()
    assert model_loaders["model"] == [("sample-model", {"dtype": torch.bfloat16, "device_map": "cuda"})]
    assert model_loaders["heads"] == []
    assert not output.exists()


def test_shared_session_checks_cuda_before_model_lookup(model_loaders):
    with pytest.raises(SystemExit, match="CUDA"):
        QwenAsrSession.load()
    assert not any(model_loaders.values())


def test_shared_session_preserves_offline_lookup_and_vram_cap(monkeypatch, model_loaders):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    session = QwenAsrSession.load(download=False, vram_cap=0.8)
    assert session.device.type == "cuda"
    assert session.dtype == torch.bfloat16
    assert model_loaders["model"] == [("sample-model", {"dtype": torch.bfloat16, "device_map": "cuda"})]
    assert model_loaders["cap"] == [0.8]
    assert model_loaders["resolve"][0][1]["download"] is False


@pytest.mark.parametrize("requested, available", [
    ("auto", False), ("cuda", False), ("cuda:0", False),
    ("cpu", False), ("cpu", True),
])
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
