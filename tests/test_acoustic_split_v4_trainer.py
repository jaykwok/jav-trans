from __future__ import annotations

import numpy as np
from pathlib import Path

from tools.boundary.ja.train_acoustic_split_v4_model import (
    _classification_metrics,
    apply_manual_label_overrides,
    event_run_counts,
    gate_passes,
    partition_group_names,
)


def test_manual_overrides_keep_unsure_as_ignore_and_consume_holdout(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        '\n'.join([
            '{"audio_id":"a","time_s":1.0}',
            '{"audio_id":"a","time_s":2.0}',
            '{"audio_id":"b","time_s":3.0}',
        ]) + '\n',
        encoding="utf-8",
    )
    overrides = tmp_path / "overrides.jsonl"
    overrides.write_text(
        '\n'.join([
            '{"audio_id":"a","time_s":1.0,"training_label":"cut"}',
            '{"audio_id":"a","time_s":2.0,"training_label":"ignore"}',
        ]) + '\n',
        encoding="utf-8",
    )
    data = {
        "labels": np.asarray([1, 1, 0]),
        "partitions": np.asarray(["test", "test", "val"]),
        "group_ids": np.asarray(["a|island0000", "a|island0000", "b|island0000"]),
        "groups": {
            "a|island0000": np.asarray([0, 1]),
            "b|island0000": np.asarray([2]),
        },
    }
    summary = apply_manual_label_overrides(
        data, metadata_path=metadata, overrides_path=overrides
    )
    assert data["labels"].tolist() == [0, -100, 0]
    assert data["partitions"].tolist() == ["train", "train", "val"]
    assert summary["training_label_counts"] == {"cut": 1, "continue": 0, "ignore": 1}


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
