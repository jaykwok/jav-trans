from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.datasets.calibrate_joint_split_cueqc import (
    cueqc_metrics,
    load_cueqc_gate,
    select_operating_point,
    split_metrics,
)


def test_cueqc_gate_is_the_threshold_source_of_truth(tmp_path: Path) -> None:
    paired = tmp_path / "paired.jsonl"
    paired.write_text("", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema": "pre_asr_cueqc_v12_gate_summary_v1",
                "v12_threshold": 0.5,
                "outputs": {"paired_decisions": str(paired)},
            }
        ),
        encoding="utf-8",
    )
    summary, threshold = load_cueqc_gate(summary_path, paired)
    assert summary["schema"] == "pre_asr_cueqc_v12_gate_summary_v1"
    assert threshold == 0.5
    with pytest.raises(ValueError, match="do not belong"):
        load_cueqc_gate(summary_path, tmp_path / "other.jsonl")


def test_split_metrics_tracks_rectangular_threshold_churn() -> None:
    data = {
        "groups": {
            "short": np.asarray([0, 1], dtype=np.int64),
            "normal": np.asarray([2, 3], dtype=np.int64),
        },
        "scalars": np.zeros((4, 5), dtype=np.float32),
        "labels": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "pair_ids": np.asarray([-1, -1, -1, -1], dtype=np.int64),
    }
    data["scalars"][0, 4] = 4.0
    data["scalars"][2, 4] = 10.0
    gates = {0: 0.8, 1: 0.7, 2: 0.76, 3: 0.2}
    baseline = split_metrics(
        data,
        ["short", "normal"],
        gates,
        normal_threshold=0.75,
        short_threshold=0.8,
        short_core_max_s=6.0,
    )
    accepted = baseline.pop("_accepted")
    changed = split_metrics(
        data,
        ["short", "normal"],
        gates,
        normal_threshold=0.75,
        short_threshold=0.7,
        short_core_max_s=6.0,
        baseline_accept=accepted,
    )
    assert baseline["cut_recall"] == 1.0
    assert baseline["continue_false_cut"] == 0.0
    assert changed["decision_change_count"] == 1
    assert changed["decision_change_ratio"] == 0.25


def test_cueqc_metrics_are_duration_weighted() -> None:
    rows = [
        {"truth": "keep", "duration_s": 2.0, "v12_prob_drop": 0.6},
        {"truth": "keep", "duration_s": 8.0, "v12_prob_drop": 0.1},
        {"truth": "drop", "duration_s": 3.0, "v12_prob_drop": 0.7},
        {"truth": "drop", "duration_s": 1.0, "v12_prob_drop": 0.2},
    ]
    metrics = cueqc_metrics(rows, 0.5)
    assert metrics["drop_recall"] == 0.5
    assert metrics["semantic_keep_recall"] == 0.5
    assert metrics["speech_loss_duration_ratio"] == 0.2
    assert metrics["noise_residual_duration_ratio"] == 0.25


def test_selection_preserves_gates_and_prefers_lower_speech_loss() -> None:
    baseline_split = {
        "normal_threshold": 0.75,
        "short_threshold": 0.8,
        "cut_recall": 0.7,
        "continue_false_cut": 0.04,
    }
    baseline_cueqc = {"threshold": 0.5, "semantic_keep_recall": 0.99}
    split_grid = [
        {
            "normal_threshold": 0.75,
            "short_threshold": 0.8,
            "cut_precision": 0.75,
            "cut_recall": 0.7,
            "cut_f1": 0.72,
            "continue_false_cut": 0.04,
            "decision_change_count": 0,
            "decision_change_ratio": 0.0,
        },
        {
            "normal_threshold": 0.75,
            "short_threshold": 0.75,
            "cut_precision": 0.76,
            "cut_recall": 0.71,
            "cut_f1": 0.734,
            "continue_false_cut": 0.039,
            "decision_change_count": 2,
            "decision_change_ratio": 0.004,
        },
    ]
    cueqc_grid = [
        {
            "threshold": 0.5,
            "drop_recall": 0.99,
            "semantic_keep_recall": 0.99,
            "speech_loss_duration_ratio": 0.01,
            "noise_residual_duration_ratio": 0.02,
            "false_drop_count": 2,
            "false_keep_count": 3,
        },
        {
            "threshold": 0.55,
            "drop_recall": 0.985,
            "semantic_keep_recall": 1.0,
            "speech_loss_duration_ratio": 0.0,
            "noise_residual_duration_ratio": 0.03,
            "false_drop_count": 0,
            "false_keep_count": 4,
        },
    ]
    selected, grid = select_operating_point(
        split_grid,
        cueqc_grid,
        baseline_split=baseline_split,
        baseline_cueqc=baseline_cueqc,
        min_drop_recall=0.98,
    )
    assert len(grid) == 4
    assert selected["short_threshold"] == 0.75
    assert selected["cueqc_drop_threshold"] == 0.55


def test_selection_does_not_expand_audited_drop_set_by_default() -> None:
    split = {
        "normal_threshold": 0.75,
        "short_threshold": 0.8,
        "cut_precision": 0.75,
        "cut_recall": 0.7,
        "cut_f1": 0.72,
        "continue_false_cut": 0.04,
        "decision_change_count": 0,
        "decision_change_ratio": 0.0,
    }
    cue_baseline = {
        "threshold": 0.5,
        "drop_recall": 0.99,
        "semantic_keep_recall": 1.0,
        "speech_loss_duration_ratio": 0.0,
        "noise_residual_duration_ratio": 0.02,
        "false_drop_count": 0,
        "false_keep_count": 2,
    }
    lower = {**cue_baseline, "threshold": 0.45, "noise_residual_duration_ratio": 0.01}
    selected, _ = select_operating_point(
        [split],
        [lower, cue_baseline],
        baseline_split=split,
        baseline_cueqc=cue_baseline,
        min_drop_recall=0.98,
    )
    assert selected["cueqc_drop_threshold"] == 0.5
