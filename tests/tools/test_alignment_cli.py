"""Exercise the actual CLI parsers with tiny synthetic CPU inputs."""
from __future__ import annotations

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


@pytest.mark.parametrize("device_args", [["--device", "cpu"], ["--allow-cpu"]])
def test_cache_cli_registers_checkpoint_and_device_options(tmp_path, monkeypatch, checkpoint, device_args):
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
        "--output", str(output), *device_args,
    ])
    evaluate_ctc_cache.main()
    result = json.loads(output.read_text("utf-8"))
    assert result["rows"] == 1
    assert result["rows_by_target_kind"] == {"text": 1}


def test_geometry_cpu_option_reaches_encoder(tmp_path, monkeypatch, checkpoint):
    composites = tmp_path / "sample-a.jsonl"
    composites.write_text("", "utf-8")
    output = tmp_path / "geometry.json"
    configs = []
    monkeypatch.setattr(evaluate_alignment_geometry, "Qwen3AsrEncoder", lambda config: configs.append(config))
    monkeypatch.setattr(sys, "argv", [
        "evaluate_alignment_geometry.py", "--checkpoint", str(checkpoint),
        "--composites", str(composites), "--output", str(output), "--device", "cpu",
    ])
    evaluate_alignment_geometry.main()
    assert len(configs) == 1 and configs[0].device == "cpu"
    assert output.is_file()


def test_cache_cli_missing_cuda_refuses_implicit_cpu(tmp_path, monkeypatch, checkpoint):
    monkeypatch.setattr(sys, "argv", [
        "evaluate_ctc_cache.py", "--checkpoint", str(checkpoint),
        "--cache-dir", str(tmp_path), "--output", str(tmp_path / "unused.json"),
    ])
    with pytest.raises(SystemExit, match="CUDA"):
        evaluate_ctc_cache.main()
