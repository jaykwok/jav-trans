from __future__ import annotations

import json
from pathlib import Path

from tools.asr.cueqc.evaluate_pre_asr_v13_binary_gate import evaluate


def test_v13_binary_gate_excludes_unsure_and_writes_every_false_drop(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {
            "row_id": "keep-ok",
            "source_partition": "val",
            "split_membership": "holdout",
            "truth_label": "keep",
            "prediction": "keep",
            "included_in_metrics": True,
        },
        {
            "row_id": "keep-missed",
            "source_partition": "test",
            "split_membership": "holdout",
            "truth_label": "keep",
            "prediction": "drop",
            "included_in_metrics": True,
        },
        {
            "row_id": "keep-test-ok",
            "source_partition": "test",
            "split_membership": "holdout",
            "truth_label": "keep",
            "prediction": "keep",
            "included_in_metrics": True,
        },
        {
            "row_id": "drop-val-ok",
            "source_partition": "val",
            "split_membership": "holdout",
            "truth_label": "drop",
            "prediction": "drop",
            "included_in_metrics": True,
        },
        {
            "row_id": "drop-test-ok",
            "source_partition": "test",
            "split_membership": "holdout",
            "truth_label": "drop",
            "prediction": "drop",
            "included_in_metrics": True,
        },
        {
            "row_id": "unsure",
            "source_partition": "test",
            "split_membership": "excluded",
            "truth_label": "unsure",
            "prediction": "drop",
            "included_in_metrics": False,
        },
    ]
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = evaluate(
        predictions=predictions,
        output_dir=tmp_path / "gate",
        min_keep_recall=0.5,
        min_drop_recall=1.0,
    )

    assert summary["gate"]["passed"] is True
    assert summary["holdout"]["keep_recall"] == 2 / 3
    assert summary["holdout"]["drop_recall"] == 1.0
    assert summary["excluded_truth_count"] == 1
    assert summary["all_false_drop_count"] == 1
    false_drops = [
        json.loads(line)
        for line in (tmp_path / "gate" / "false_drop_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["row_id"] for row in false_drops] == ["keep-missed"]
