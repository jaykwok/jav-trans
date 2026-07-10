from __future__ import annotations

import json
from pathlib import Path

import torch

from tools.datasets.apply_joint_split_cueqc_calibration import apply_calibration


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_apply_calibration_updates_binding_without_changing_weights(tmp_path: Path) -> None:
    split_path = tmp_path / "split.pt"
    cueqc_path = tmp_path / "cueqc.pt"
    split_state = {"weight": torch.arange(4, dtype=torch.float32)}
    cueqc_state = {"weight": torch.arange(3, dtype=torch.float32)}
    torch.save(
        {
            "schema": "semantic_split_verifier_v2",
            "model_state_dict": split_state,
            "decision_config": {
                "normal_cut_threshold": 0.75,
                "short_core_cut_threshold": 0.8,
                "duration_pressure_enabled": False,
            },
            "metadata": {},
        },
        split_path,
    )
    torch.save(
        {
            "schema": "cueqc_pre_asr_semantic_chunk_v12_binary",
            "model_state_dict": cueqc_state,
            "decision_config": {"drop_threshold": 0.5, "model_only": True},
            "metadata": {"semantic_split_weights_sha256": "old"},
            "semantic_split_weights_sha256": "old",
        },
        cueqc_path,
    )
    calibration = _write_json(
        tmp_path / "calibration.json",
        {
            "schema": "joint_split_cueqc_calibration_v1",
            "selected": {"split_decision_change_ratio": 0.001},
            "recommendation": {
                "status": "ready_to_apply",
                "requires_de_prime": False,
                "requires_cueqc_reaudit": False,
                "weights_must_remain_unchanged": True,
                "split_decision_config": {
                    "normal_cut_threshold": 0.75,
                    "short_core_cut_threshold": 0.775,
                },
                "cueqc_decision_config": {"drop_threshold": 0.5},
            },
        },
    )
    operating = _write_json(
        tmp_path / "operating.json",
        {
            "semantic_split_duration_pressure": {
                "enabled": True,
                "log_median": -0.1,
                "log_mad": 1.0,
                "z": 1.7,
                "floor": 0.5,
            }
        },
    )

    summary = apply_calibration(
        calibration_summary_path=calibration,
        operating_point_summary_path=operating,
        split_checkpoint_path=split_path,
        cueqc_checkpoint_path=cueqc_path,
        output_summary_path=tmp_path / "summary.json",
        dry_run=False,
    )

    split = torch.load(split_path, map_location="cpu", weights_only=False)
    cueqc = torch.load(cueqc_path, map_location="cpu", weights_only=False)
    torch.testing.assert_close(split["model_state_dict"]["weight"], split_state["weight"])
    torch.testing.assert_close(cueqc["model_state_dict"]["weight"], cueqc_state["weight"])
    assert split["decision_config"]["short_core_cut_threshold"] == 0.775
    assert split["decision_config"]["duration_pressure_enabled"] is True
    assert split["decision_config"]["duration_pressure_z"] == 1.7
    assert cueqc["decision_config"]["drop_threshold"] == 0.5
    assert cueqc["metadata"]["semantic_split_weights_sha256"] == summary[
        "split_file_sha256_after"
    ]
    assert summary["weights_unchanged"] is True
