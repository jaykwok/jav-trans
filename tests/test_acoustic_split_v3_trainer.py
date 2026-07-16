from __future__ import annotations

import numpy as np

from tools.boundary.ja.train_acoustic_split_v3_model import (
    _classification_metrics,
    event_run_counts,
    gate_passes,
    partition_group_names,
)


def test_partition_groups_keep_test_out_of_training() -> None:
    data = {
        "groups": {
            "train-a": np.asarray([0, 1]),
            "val-a": np.asarray([2]),
            "test-a": np.asarray([3, 4]),
        },
        "partitions": np.asarray(["train", "train", "val", "test", "test"]),
    }
    assert partition_group_names(data) == {
        "test": ["test-a"],
        "train": ["train-a"],
        "val": ["val-a"],
    }


def test_numeric_gate_uses_direct_argmax_metrics() -> None:
    truth = np.asarray([0] * 20 + [1] * 20 + [2] * 4)
    predicted = truth.copy()
    metrics = _classification_metrics(truth, predicted)
    metrics["event_runs"] = {
        "basin_precision": 1.0,
        "basin_recall": 1.0,
    }
    assert gate_passes(metrics)

    predicted[0:3] = 1
    metrics = _classification_metrics(truth, predicted)
    metrics["event_runs"] = {
        "basin_precision": 1.0,
        "basin_recall": 0.80,
    }
    assert not gate_passes(metrics)


def test_adjacent_cut_candidates_are_one_event_run() -> None:
    truth = np.asarray([1, 2, 0, 0, 2, 1, 2, 0, 2, 1])
    predicted = np.asarray([1, 0, 0, 0, 2, 1, 2, 0, 0, 1])
    counts = event_run_counts(truth, predicted)
    assert counts == {
        "truth": 2,
        "predicted": 2,
        "exact": 0,
        "basin_matched": 2,
    }
