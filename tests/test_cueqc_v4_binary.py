from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

from asr import cueqc
from asr.backends.qwen import QWEN_ASR_17B_REPO_ID
from asr.cueqc_refiner import load_cueqc_mamba_checkpoint


class _StubCueQCMambaV4Binary:
    def __init__(self, **_kwargs):
        self.loaded = False

    def load_state_dict(self, _state_dict):
        self.loaded = True

    def to(self, _device):
        return self

    def eval(self):
        return self


def test_cueqc_runtime_signature_is_v4_binary(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "cueqc_mamba_v4_binary.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setenv("CUEQC_SHADOW_ENABLED", "1")
    monkeypatch.setenv("CUEQC_MODEL_PATH_BY_REPO", f"{QWEN_ASR_17B_REPO_ID}={checkpoint}")

    sig = cueqc.runtime_signature()

    assert sig["policy"] == "cueqc_mamba_v4_binary"
    assert sig["model_version"] == "cueqc_mamba_v4_binary"
    assert "threshold_profile" not in sig


def test_cueqc_loader_rejects_v3_schema(tmp_path: Path):
    checkpoint = tmp_path / "cueqc_v3.pt"
    torch.save({"schema": "cueqc_mamba_v3_fusion"}, checkpoint)

    with pytest.raises(ValueError, match="unsupported CueQC checkpoint schema"):
        load_cueqc_mamba_checkpoint(checkpoint, device="cpu")


def test_cueqc_loader_rejects_threshold_profile(monkeypatch, tmp_path: Path):
    module = types.ModuleType("asr.cueqc_model")
    module.CueQCMambaV4Binary = _StubCueQCMambaV4Binary
    monkeypatch.setitem(sys.modules, "asr.cueqc_model", module)
    checkpoint = tmp_path / "cueqc_v4_profile.pt"
    torch.save(
        {
            "schema": "cueqc_mamba_v4_binary",
            "model_config": {},
            "state_dict": {},
            "metadata": {},
            "feature_config": {},
            "normalization": {},
            "decision_config": {"drop_threshold": 0.85, "drop_threshold_profile": {"mode": "old"}},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="drop_threshold_profile"):
        load_cueqc_mamba_checkpoint(checkpoint, device="cpu")


def test_pending_cueqc_decision_has_no_profile_fields():
    decision = cueqc.pending_model_decision({"cluster_id": "cluster-a"})

    assert decision["model_version"] == "cueqc_mamba_v4_binary"
    assert decision["display_hint"] == "keep"
    assert "threshold_profile" not in decision
    assert "text_bucket" not in decision
