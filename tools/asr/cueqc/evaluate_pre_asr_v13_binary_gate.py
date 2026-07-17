#!/usr/bin/env python3
"""Evaluate CueQC v13 binary argmax gates and emit every false-drop target."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    included = [row for row in rows if bool(row.get("included_in_metrics"))]
    keep = [row for row in included if row.get("truth_label") == "keep"]
    drop = [row for row in included if row.get("truth_label") == "drop"]
    keep_correct = sum(row.get("prediction") == "keep" for row in keep)
    drop_correct = sum(row.get("prediction") == "drop" for row in drop)
    return {
        "count": len(included),
        "keep_count": len(keep),
        "drop_count": len(drop),
        "keep_recall": keep_correct / len(keep) if keep else 0.0,
        "drop_recall": drop_correct / len(drop) if drop else 0.0,
        "false_drop_count": len(keep) - keep_correct,
        "false_keep_count": len(drop) - drop_correct,
    }


def evaluate(
    *,
    predictions: Path,
    output_dir: Path,
    min_keep_recall: float = 0.95,
    min_drop_recall: float = 0.95,
) -> dict[str, Any]:
    rows = _rows(predictions)
    invalid_predictions = sorted(
        {str(row.get("prediction") or "") for row in rows} - {"drop", "keep"}
    )
    if invalid_predictions:
        raise ValueError(f"runtime predictions must be binary: {invalid_predictions}")
    holdout = [row for row in rows if row.get("split_membership") == "holdout"]
    by_partition = {
        partition: _metrics(
            [row for row in rows if row.get("source_partition") == partition]
        )
        for partition in ("train", "val", "test")
    }
    holdout_metrics = _metrics(holdout)
    gate = {
        "min_keep_recall": float(min_keep_recall),
        "min_drop_recall": float(min_drop_recall),
        "holdout_keep_recall_pass": holdout_metrics["keep_recall"] >= min_keep_recall,
        "holdout_drop_recall_pass": holdout_metrics["drop_recall"] >= min_drop_recall,
        "partition_pass": {
            partition: {
                "keep_recall_pass": by_partition[partition]["keep_recall"]
                >= min_keep_recall,
                "drop_recall_pass": by_partition[partition]["drop_recall"]
                >= min_drop_recall,
            }
            for partition in ("val", "test")
        },
    }
    gate["passed"] = bool(
        gate["holdout_keep_recall_pass"] and gate["holdout_drop_recall_pass"]
    )
    false_drops = [
        row
        for row in rows
        if row.get("prediction") == "drop" and row.get("truth_label") == "keep"
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    false_drop_path = output_dir / "false_drop_manifest.jsonl"
    _write_jsonl(false_drop_path, false_drops)
    summary = {
        "schema": "cueqc_v13_binary_argmax_gate_summary_v1",
        "prediction_count": len(rows),
        "runtime_prediction_labels": dict(Counter(row["prediction"] for row in rows)),
        "excluded_truth_count": sum(not row.get("included_in_metrics") for row in rows),
        "holdout": holdout_metrics,
        "by_partition": by_partition,
        "gate": gate,
        "all_false_drop_count": len(false_drops),
        "holdout_false_drop_count": sum(
            row.get("split_membership") == "holdout" for row in false_drops
        ),
        "false_drop_manifest": str(false_drop_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-keep-recall", type=float, default=0.95)
    parser.add_argument("--min-drop-recall", type=float, default=0.95)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                predictions=Path(args.predictions),
                output_dir=Path(args.output_dir),
                min_keep_recall=args.min_keep_recall,
                min_drop_recall=args.min_drop_recall,
            ),
            ensure_ascii=False,
        )
    )
