#!/usr/bin/env python3
"""Evaluate a v12 paired replay against the fixed manual holdout."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.asr.cueqc.pre_asr_feature_compiler import project_path  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(dict(row))
    return rows


def evaluate(
    paired_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    candidate_ids: list[str],
) -> dict[str, Any]:
    wanted = {candidate_id.strip() for candidate_id in candidate_ids if candidate_id.strip()}
    paired = {str(row.get("id") or row.get("candidate_id") or ""): row for row in paired_rows}
    labels = {str(row.get("candidate_id") or row.get("sample_id") or ""): row for row in label_rows}
    missing_paired = sorted(wanted - paired.keys())
    missing_labels = sorted(wanted - labels.keys())
    if missing_paired or missing_labels:
        raise ValueError(
            f"holdout closure failed: missing_paired={missing_paired} missing_labels={missing_labels}"
        )

    decisions: list[dict[str, Any]] = []
    for candidate_id in sorted(wanted):
        label = str(labels[candidate_id]["label"])
        truth = "keep" if label == "definite_keep" else "drop" if label == "definite_drop" else "ignore"
        if truth == "ignore":
            continue
        prediction = str(paired[candidate_id]["v12_prediction"])
        decisions.append(
            {
                "candidate_id": candidate_id,
                "truth": truth,
                "prediction": prediction,
                "v12_prob_drop": float(paired[candidate_id]["v12_prob_drop"]),
            }
        )

    counts = Counter((row["truth"], row["prediction"]) for row in decisions)
    keep_count = sum(row["truth"] == "keep" for row in decisions)
    drop_count = sum(row["truth"] == "drop" for row in decisions)
    keep_false_drops = [
        row for row in decisions if row["truth"] == "keep" and row["prediction"] == "drop"
    ]
    drop_false_keeps = [
        row for row in decisions if row["truth"] == "drop" and row["prediction"] == "keep"
    ]
    return {
        "schema": "pre_asr_manual_holdout_gate_v1",
        "candidate_count": len(decisions),
        "keep_count": keep_count,
        "drop_count": drop_count,
        "confusion": {f"{truth}_to_{prediction}": count for (truth, prediction), count in sorted(counts.items())},
        "keep_false_drop_count": len(keep_false_drops),
        "drop_false_keep_count": len(drop_false_keeps),
        "keep_recall": 1.0 - len(keep_false_drops) / max(1, keep_count),
        "drop_recall": 1.0 - len(drop_false_keeps) / max(1, drop_count),
        "keep_false_drops": keep_false_drops,
        "drop_false_keeps": drop_false_keeps,
        "gate_pass": len(keep_false_drops) == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the fixed Pre-ASR manual holdout.")
    parser.add_argument("--paired-decisions", required=True)
    parser.add_argument("--manual-labels", required=True)
    parser.add_argument("--candidate-ids", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    candidate_ids = project_path(args.candidate_ids).read_text(encoding="utf-8").splitlines()
    summary = evaluate(
        _read_jsonl(project_path(args.paired_decisions)),
        _read_jsonl(project_path(args.manual_labels)),
        candidate_ids,
    )
    summary.update(
        {
            "paired_decisions": str(project_path(args.paired_decisions)),
            "manual_labels": str(project_path(args.manual_labels)),
            "candidate_ids": str(project_path(args.candidate_ids)),
        }
    )
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
