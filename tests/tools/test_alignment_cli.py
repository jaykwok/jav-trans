"""CUDA is mandatory in the CLIs; computation tests use tiny tensor doubles."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asr.alignment import ALIGNMENT_MODEL_SCHEMA, AlignmentVocab, build_head
from tools.align import evaluate_alignment_geometry, evaluate_ctc_cache


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    vocab = AlignmentVocab(chars=("あ",))
    head = build_head(vocab_size=vocab.size, input_dim=4, hidden_dim=8, upsample=1, blocks=1, dropout=0.0)
    path = tmp_path / "head.pt"
    torch.save({
        "schema": ALIGNMENT_MODEL_SCHEMA, "vocab": vocab.to_payload(),
        "input_dim": 4, "hidden_dim": 8, "upsample": 1, "blocks": 1,
        "state_dict": head.state_dict(),
    }, path)
    return path


def _tiny_tensor_device(module, monkeypatch):
    import torch

    requested = []

    def resolve(choice):
        requested.append(choice)
        return torch.device("cpu")

    # Substitute only inside these computation tests. The real CLI boundary is
    # exercised separately below without replacing its device resolver.
    monkeypatch.setattr(module, "resolve_device", resolve)
    monkeypatch.setattr(module, "apply_vram_safety_cap", lambda *_args: None)
    return requested


@pytest.mark.parametrize("device_choice", ["auto", "cuda"])
def test_cache_cli_evaluates_tiny_synthetic_inputs(tmp_path, monkeypatch, checkpoint, device_choice):
    requested = _tiny_tensor_device(evaluate_ctc_cache, monkeypatch)
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "features.npy", np.zeros((8, 4), dtype=np.float32))
    (cache / "index.jsonl").write_text(json.dumps({
        "partition": "val", "audio_id": "sample-a", "text": "あ", "frames": 8,
        "shard": "features.npy", "offset": 0, "duration_s": 0.32,
    }) + "\n", "utf-8")
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_ctc_cache.py", "--checkpoint", str(checkpoint), "--cache-dir", str(cache),
        "--output", str(output), "--device", device_choice,
    ])
    evaluate_ctc_cache.main()
    result = json.loads(output.read_text("utf-8"))
    assert result["rows"] == 1
    assert result["rows_by_target_kind"] == {"text": 1}
    assert requested == [device_choice]


def test_geometry_resolved_device_reaches_encoder(tmp_path, monkeypatch, checkpoint):
    requested = _tiny_tensor_device(evaluate_alignment_geometry, monkeypatch)
    composites = tmp_path / "sample-a.jsonl"
    composites.write_text("", "utf-8")
    output = tmp_path / "geometry.json"
    configs = []
    monkeypatch.setattr(evaluate_alignment_geometry, "Qwen3AsrEncoder", lambda config: configs.append(config))
    monkeypatch.setattr(sys, "argv", [
        "evaluate_alignment_geometry.py", "--checkpoint", str(checkpoint),
        "--composites", str(composites), "--output", str(output), "--device", "cuda",
    ])
    evaluate_alignment_geometry.main()
    assert len(configs) == 1 and configs[0].device == "cpu"
    assert requested == ["cuda"]
    assert output.is_file()


_CUDA_CLI_CASES = [
    ("train_ctc_aligner", ("--cache-dir",), "--output-dir"),
    ("evaluate_ctc_cache", ("--checkpoint", "--cache-dir"), "--output"),
    ("evaluate_alignment_geometry", ("--checkpoint", "--composites"), "--output"),
    ("measure_blank_class_separation", (
        "--checkpoint", "--cache-dir", "--teacher-results", "--teacher-manifest",
    ), "--output"),
    ("sweep_blank_bias", ("--checkpoint", "--composites"), "--output"),
    ("sweep_edge_caps", ("--checkpoint", "--composites", "--edge-silence"), "--output"),
    ("build_alignment_features", ("--manifest",), "--output-dir"),
]


@pytest.fixture(params=_CUDA_CLI_CASES, ids=[case[0] for case in _CUDA_CLI_CASES])
def cuda_cli(request, tmp_path, monkeypatch):
    name, input_flags, output_flag = request.param
    module = importlib.import_module(f"tools.align.{name}")
    output = tmp_path / "result"
    argv = [name, output_flag, str(output)]
    for flag in input_flags:
        argv.extend([flag, str(tmp_path / "missing-input")])

    def premature_work(*_args, **_kwargs):
        pytest.fail("model/cache work started before the CUDA requirement was checked")

    for attribute in ("load_checkpoint", "build_head", "FeatureCache", "Qwen3AsrEncoder", "apply_vram_safety_cap"):
        if hasattr(module, attribute):
            monkeypatch.setattr(module, attribute, premature_work)
    return module, argv, output


def test_training_and_evaluation_fail_before_reading_inputs_without_cuda(cuda_cli, monkeypatch):
    import torch

    module, argv, output = cuda_cli
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="CUDA"):
        module.main()
    assert not output.exists()


@pytest.mark.parametrize("flags", [["--device", "cpu"], ["--allow-cpu"]])
def test_training_and_evaluation_reject_removed_cpu_options(cuda_cli, monkeypatch, capsys, flags):
    import torch

    module, argv, output = cuda_cli
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", [*argv, *flags])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert flags[0] in capsys.readouterr().err
    assert not output.exists()
