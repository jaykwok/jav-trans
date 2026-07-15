#!/usr/bin/env python3
"""Evaluate bootstrap Inner edge quality over retained provisional sub-islands."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ITEM_SCHEMA = "inner_subisland_edge_audit_item_v1"
VERDICT_SCHEMA = "inner_subisland_edge_manual_verdict_v1"
SUMMARY_SCHEMA = "inner_subisland_edge_gate_v1"
ALLOWED = {"correct", "clipped", "too_wide", "unsure"}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(*, items: Path, verdicts: Path, output: Path) -> dict[str, Any]:
    item_rows = _rows(items)
    verdict_by_id = {
        str(row["subisland_id"]): row
        for row in _rows(verdicts)
        if row.get("schema") == VERDICT_SCHEMA
    }
    counts: Counter[str] = Counter()
    complete_edges = 0
    required_edges = 0
    model_abstain_edges = 0
    for item in item_rows:
        if item.get("schema") != ITEM_SCHEMA:
            raise ValueError("incompatible Inner edge audit item schema")
        verdict = verdict_by_id.get(str(item["subisland_id"])) or {}
        prediction = item.get("model_prediction") or item.get("bootstrap_prediction") or {}
        for edge in ("start", "end"):
            if not item.get(f"{edge}_requires_inner"):
                continue
            required_edges += 1
            label = str(verdict.get(f"{edge}_verdict") or "unreviewed")
            if label in ALLOWED:
                counts[label] += 1
                complete_edges += 1
            if str(prediction.get(f"{edge}_action") or "") == "abstain":
                model_abstain_edges += 1
    formal_model_gate = bool(item_rows) and all(
        row.get("teacher_usage") == "formal_inner_model_heldout_evaluation"
        for row in item_rows
    )
    promotion_ready = (
        formal_model_gate
        and complete_edges == required_edges
        and counts["correct"] == required_edges
        and model_abstain_edges == 0
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "subisland_count": len(item_rows),
        "required_edge_count": required_edges,
        "complete_edge_count": complete_edges,
        "verdict_counts": dict(sorted(counts.items())),
        "model_abstain_edge_count": model_abstain_edges,
        "manual_gate_complete": complete_edges == required_edges,
        "zero_clipping_pass": complete_edges == required_edges
        and counts["clipped"] == 0,
        "edge_cleanup_pass": complete_edges == required_edges
        and counts["clipped"] == 0
        and counts["too_wide"] == 0,
        "formal_inner_checkpoint_gate": formal_model_gate,
        "formal_inner_promotion_ready": promotion_ready,
        "bootstrap_inner_promotion_ready": False,
        "bootstrap_reason": (
            "not_applicable_formal_inner_checkpoint"
            if formal_model_gate
            else "Outer v2 preview is not an independently trained Inner checkpoint"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Inner sub-island edge audit.")
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
