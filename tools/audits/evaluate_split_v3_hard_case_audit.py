#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID_LABELS = {"cut", "continue", "unsure"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    *,
    manifest: Path,
    verdicts: Path,
    output: Path,
    minimum_cut_precision: float = 0.90,
    minimum_cut_recall: float = 0.90,
    minimum_continue_recall: float = 0.90,
) -> dict[str, Any]:
    targets = {str(row["candidate_id"]): row for row in _read_jsonl(manifest)}
    reviewed: dict[str, dict[str, Any]] = {}
    duplicate_ids = []
    invalid = []
    for row in _read_jsonl(verdicts):
        candidate_id = str(row.get("candidate_id") or "")
        verdict = str(row.get("verdict") or "").strip().lower()
        if not candidate_id or verdict not in VALID_LABELS:
            invalid.append(row)
            continue
        if candidate_id in reviewed:
            duplicate_ids.append(candidate_id)
        reviewed[candidate_id] = row
    target_ids = set(targets)
    reviewed_ids = set(reviewed)
    missing_ids = sorted(target_ids - reviewed_ids)
    extra_ids = sorted(reviewed_ids - target_ids)
    confusion: Counter[tuple[str, str]] = Counter()
    heldout_failures = []
    for candidate_id in sorted(target_ids & reviewed_ids):
        target = targets[candidate_id]
        manual = str(reviewed[candidate_id]["verdict"])
        model = str(target["label"])
        confusion[(manual, model)] += 1
        expected = str(target.get("expected_gate_label") or "")
        if expected and manual != expected:
            heldout_failures.append(
                {
                    "candidate_id": candidate_id,
                    "expected": expected,
                    "manual": manual,
                }
            )
    cut_tp = confusion[("cut", "cut")]
    cut_fp = confusion[("continue", "cut")]
    cut_fn = confusion[("cut", "continue")]
    continue_tp = confusion[("continue", "continue")]
    continue_fn = confusion[("continue", "cut")]
    cut_precision = _ratio(cut_tp, cut_tp + cut_fp)
    cut_recall = _ratio(cut_tp, cut_tp + cut_fn)
    continue_recall = _ratio(continue_tp, continue_tp + continue_fn)
    pass_gate = (
        not missing_ids
        and not extra_ids
        and not duplicate_ids
        and not invalid
        and not heldout_failures
        and cut_precision >= minimum_cut_precision
        and cut_recall >= minimum_cut_recall
        and continue_recall >= minimum_continue_recall
    )
    summary = {
        "schema": "semantic_split_v3_hard_case_audit_evaluation_v1",
        "target_count": len(targets),
        "reviewed_count": len(target_ids & reviewed_ids),
        "missing_count": len(missing_ids),
        "extra_count": len(extra_ids),
        "duplicate_count": len(duplicate_ids),
        "invalid_count": len(invalid),
        "heldout_failure_count": len(heldout_failures),
        "cut_precision": cut_precision,
        "cut_recall": cut_recall,
        "continue_recall": continue_recall,
        "thresholds": {
            "minimum_cut_precision": minimum_cut_precision,
            "minimum_cut_recall": minimum_cut_recall,
            "minimum_continue_recall": minimum_continue_recall,
        },
        "pass": pass_gate,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "duplicate_ids": duplicate_ids,
        "heldout_failures": heldout_failures,
        "confusion": {
            f"manual_{manual}__model_{model}": count
            for (manual, model), count in sorted(confusion.items())
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-cut-precision", type=float, default=0.90)
    parser.add_argument("--minimum-cut-recall", type=float, default=0.90)
    parser.add_argument("--minimum-continue-recall", type=float, default=0.90)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                manifest=Path(args.manifest),
                verdicts=Path(args.verdicts),
                output=Path(args.output),
                minimum_cut_precision=args.minimum_cut_precision,
                minimum_cut_recall=args.minimum_cut_recall,
                minimum_continue_recall=args.minimum_continue_recall,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
