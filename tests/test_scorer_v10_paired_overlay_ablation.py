from __future__ import annotations

import json
import numpy as np

from tools.audits.run_scorer_v10_paired_overlay_ablation import _mix, summarize
from tools.audits.run_scorer_v10_paired_overlay_ablation import _metrics


def test_paired_overlay_mix_hits_requested_snr_without_changing_length() -> None:
    clean = np.full(1600, 0.2, dtype=np.float32)
    overlay = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    mask = np.ones(1600, dtype=bool)
    mixed, detail = _mix(clean, overlay, mask, 14.0)
    assert mixed.shape == clean.shape
    assert abs(detail["achieved_snr_db"] - 14.0) < 1e-6


def test_summary_uses_clean_within_same_pair_as_only_baseline() -> None:
    common = {"schema": "speech_scorer_v10_paired_overlay_ablation_v1", "diagnostic_only": True,
              "training_manifest_allowed": False, "partition": "val", "pair_hash": "same"}
    rows = [
        {**common, "overlay_type": "clean", "overlay_snr_db": None, "continuity": 1.0,
         "internal_gap_frames": 0, "prediction_run_count_within_truth": 2, "speech_recall": 1.0},
        {**common, "overlay_type": "breathing", "overlay_snr_db": 10.0, "continuity": 0.5,
         "internal_gap_frames": 3, "prediction_run_count_within_truth": 4, "speech_recall": 0.8},
    ]
    summary = summarize(rows)
    assert summary["diagnostic_only"] is True
    assert summary["training_manifest_allowed"] is False
    assert summary["grouped_mean_deltas"]["breathing@10dB"]["continuity_delta"] == -0.5
    assert summary["grouped_mean_deltas"]["breathing@10dB"]["internal_gap_frames_delta"] == 3.0


def test_metrics_are_json_serializable_native_values() -> None:
    metrics = _metrics(
        np.asarray([1, 1, 1, 1], dtype=np.int64),
        np.asarray([1, 0, 1, 1], dtype=np.int64),
    )
    json.dumps(metrics)
    assert metrics["fragmented_truth_run_count"] == 1
