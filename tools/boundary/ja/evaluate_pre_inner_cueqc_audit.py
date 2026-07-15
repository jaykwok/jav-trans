#!/usr/bin/env python3
"""Evaluate pre-Inner CueQC labels and route provisional sub-islands."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ITEM_SCHEMA = "pre_inner_cueqc_audit_item_v1"
VERDICT_SCHEMA = "pre_inner_cueqc_manual_verdict_v1"
SUMMARY_SCHEMA = "pre_inner_cueqc_gate_v1"
ALLOWED_LABELS = {"keep", "drop", "unsure"}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate(*, items: Path, verdicts: Path, output_dir: Path) -> dict[str, Any]:
    item_rows = _rows(items)
    verdict_by_id = {
        str(row["subisland_id"]): row
        for row in _rows(verdicts)
        if row.get("schema") == VERDICT_SCHEMA
    }
    counts: Counter[str] = Counter()
    labeled: list[dict[str, Any]] = []
    inner_inputs: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    complete = 0
    for item in item_rows:
        if item.get("schema") != ITEM_SCHEMA:
            raise ValueError("incompatible pre-Inner CueQC item schema")
        verdict = verdict_by_id.get(str(item["subisland_id"])) or {}
        source_audio = (items.parent / str(item["audio"])).resolve()
        label = str(verdict.get("label") or "unreviewed")
        is_complete = label in ALLOWED_LABELS
        complete += int(is_complete)
        if is_complete:
            counts[label] += 1
        routed = {
            **item,
            "audio": str(source_audio),
            "cueqc_label": label,
            "cueqc_note": str(verdict.get("note") or ""),
            "runtime_route": (
                "drop_whole_subisland"
                if label == "drop"
                else "preserve_then_inner"
                if label in {"keep", "unsure"}
                else "unreviewed"
            ),
        }
        labeled.append(routed)
        if label in {"keep", "unsure"}:
            inner_inputs.append(
                {
                    **routed,
                    "schema": "inner_edge_provisional_subisland_seed_v1",
                    "inner_label_status": "requires_edge_targets",
                }
            )
        elif label == "drop":
            removed.append(
                {
                    "schema": "cueqc_removed_provisional_span_v1",
                    "sample_id": item["sample_id"],
                    "subisland_id": item["subisland_id"],
                    "start_s": item["start_s"],
                    "end_s": item["end_s"],
                    "duration_s": item["duration_s"],
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "labeled_subislands.jsonl", labeled)
    _write(output_dir / "inner_inputs.jsonl", inner_inputs)
    _write(output_dir / "removed_spans.jsonl", removed)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "subisland_count": len(item_rows),
        "complete_count": complete,
        "label_counts": dict(sorted(counts.items())),
        "manual_gate_pass": complete == len(item_rows),
        "has_keep_supervision": counts["keep"] > 0,
        "has_drop_supervision": counts["drop"] > 0,
        "model_training_ready": complete == len(item_rows)
        and counts["keep"] > 0
        and counts["drop"] > 0,
        "inner_input_count": len(inner_inputs),
        "removed_span_count": len(removed),
        "unsure_runtime_route": "preserve_then_inner",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pre-Inner CueQC audit.")
    parser.add_argument("--items", required=True)
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                items=Path(args.items),
                verdicts=Path(args.verdicts),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
