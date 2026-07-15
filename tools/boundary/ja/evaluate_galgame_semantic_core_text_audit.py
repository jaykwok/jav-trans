#!/usr/bin/env python3
"""Evaluate the fixed manual gate for the Galgame semantic-core text teacher."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ITEM_SCHEMA = "galgame_semantic_core_text_teacher_v1"
VERDICT_SCHEMA = "galgame_semantic_core_text_manual_verdict_v1"
SUMMARY_SCHEMA = "galgame_semantic_core_text_manual_gate_v1"
TEACHER_LABELS = {"all_semantic", "contains_nonsemantic", "unsure"}
MANUAL_VERDICTS = {"correct", "wrong", "unsure"}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(*, items: Path, verdicts: Path, output: Path) -> dict[str, Any]:
    item_rows = _rows(items)
    if not item_rows or any(row.get("schema") != ITEM_SCHEMA for row in item_rows):
        raise ValueError("semantic-core text audit items are empty or incompatible")
    item_ids = [str(row["audio_id"]) for row in item_rows]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("semantic-core text audit contains duplicate audio ids")

    verdict_rows = [
        row for row in _rows(verdicts) if row.get("schema") == VERDICT_SCHEMA
    ]
    verdict_by_id = {str(row["audio_id"]): row for row in verdict_rows}
    if len(verdict_by_id) != len(verdict_rows):
        raise ValueError("semantic-core text verdicts contain duplicate audio ids")

    verdict_counts: Counter[str] = Counter()
    teacher_label_counts: Counter[str] = Counter()
    correct_by_teacher_label: Counter[str] = Counter()
    reviewed_count = 0
    notes: list[dict[str, str]] = []
    for item in item_rows:
        audio_id = str(item["audio_id"])
        teacher_label = str(item.get("label") or "")
        if teacher_label not in TEACHER_LABELS:
            raise ValueError(f"unsupported teacher label: {teacher_label!r}")
        teacher_label_counts[teacher_label] += 1
        verdict = verdict_by_id.get(audio_id) or {}
        if verdict and str(verdict.get("teacher_label") or "") != teacher_label:
            raise ValueError(f"teacher label mismatch for {audio_id}")
        manual = str(verdict.get("verdict") or "unreviewed")
        verdict_counts[manual] += 1
        if manual in MANUAL_VERDICTS:
            reviewed_count += 1
        if manual == "correct":
            correct_by_teacher_label[teacher_label] += 1
        note = str(verdict.get("note") or "").strip()
        if note:
            notes.append({"audio_id": audio_id, "note": note})

    item_count = len(item_rows)
    complete = reviewed_count == item_count
    pass_gate = complete and verdict_counts["correct"] == item_count
    summary = {
        "schema": SUMMARY_SCHEMA,
        "item_count": item_count,
        "reviewed_count": reviewed_count,
        "teacher_label_counts": dict(sorted(teacher_label_counts.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "correct_by_teacher_label": dict(sorted(correct_by_teacher_label.items())),
        "manual_gate_complete": complete,
        "teacher_gate_pass": pass_gate,
        "expansion_ready": pass_gate,
        "approved_inventory_contract": (
            "teacher_all_semantic_only_after_fixed_manual_gate_v1"
        ),
        "notes": notes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Galgame semantic-core text teacher manual audit."
    )
    parser.add_argument("--items", required=True)
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                items=Path(args.items),
                verdicts=Path(args.verdicts),
                output=Path(args.output),
            ),
            ensure_ascii=False,
        )
    )
