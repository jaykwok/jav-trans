#!/usr/bin/env python3
"""Combine formal Inner v1 manual audits under the project 95% gate."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ITEM_SCHEMA = "inner_subisland_edge_audit_item_v1"
VERDICT_SCHEMA = "inner_subisland_edge_manual_verdict_v1"
SUMMARY_SCHEMA = "inner_edge_refiner_v1_combined_manual_gate_v1"
ALLOWED = {"correct", "clipped", "too_wide", "unsure"}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(
    *, audit_dirs: list[Path], output: Path, minimum_correct_rate: float = 0.95
) -> dict[str, Any]:
    if not audit_dirs:
        raise ValueError("at least one formal Inner audit is required")
    if not 0.0 < minimum_correct_rate <= 0.95:
        raise ValueError("Inner combined gate must be in (0, 0.95]")
    counts: Counter[str] = Counter()
    required = reviewed = model_abstain = 0
    checkpoint_shas: set[str] = set()
    audit_results: list[dict[str, Any]] = []
    for audit_dir in audit_dirs:
        items = _rows(audit_dir / "inner_items.jsonl")
        verdict_rows = [
            row
            for row in _rows(audit_dir / "manual_verdicts.jsonl")
            if row.get("schema") == VERDICT_SCHEMA
        ]
        verdict_by_id = {str(row["subisland_id"]): row for row in verdict_rows}
        local_counts: Counter[str] = Counter()
        local_required = local_reviewed = 0
        for item in items:
            if item.get("schema") != ITEM_SCHEMA:
                raise ValueError("incompatible Inner audit item schema")
            if item.get("teacher_usage") != "formal_inner_model_heldout_evaluation":
                raise ValueError("combined gate accepts only formal Inner audits")
            checkpoint_shas.add(str(item.get("checkpoint_sha256") or ""))
            verdict = verdict_by_id.get(str(item["subisland_id"])) or {}
            prediction = item.get("model_prediction") or {}
            for edge in ("start", "end"):
                if not item.get(f"{edge}_requires_inner"):
                    continue
                required += 1
                local_required += 1
                label = str(verdict.get(f"{edge}_verdict") or "unreviewed")
                if label in ALLOWED:
                    counts[label] += 1
                    local_counts[label] += 1
                    reviewed += 1
                    local_reviewed += 1
                if str(prediction.get(f"{edge}_action") or "") == "abstain":
                    model_abstain += 1
        audit_results.append(
            {
                "audit_dir": str(audit_dir),
                "required_edge_count": local_required,
                "reviewed_edge_count": local_reviewed,
                "verdict_counts": dict(sorted(local_counts.items())),
            }
        )
    checkpoint_shas.discard("")
    if len(checkpoint_shas) != 1:
        raise ValueError("formal Inner audits do not share exactly one checkpoint SHA")
    correct_rate = counts["correct"] / required if required else 0.0
    complete = required > 0 and reviewed == required
    promotion_ready = (
        complete
        and correct_rate >= minimum_correct_rate
        and counts["clipped"] == 0
        and counts["unsure"] == 0
        and model_abstain == 0
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "checkpoint_sha256": next(iter(checkpoint_shas)),
        "audit_count": len(audit_dirs),
        "required_edge_count": required,
        "reviewed_edge_count": reviewed,
        "verdict_counts": dict(sorted(counts.items())),
        "correct_rate": correct_rate,
        "minimum_correct_rate": minimum_correct_rate,
        "manual_gate_complete": complete,
        "zero_clipping_pass": complete and counts["clipped"] == 0,
        "zero_unsure_pass": complete and counts["unsure"] == 0,
        "model_abstain_edge_count": model_abstain,
        "combined_manual_gate_pass": promotion_ready,
        "promotion_ready": promotion_ready,
        "audits": audit_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate combined formal Inner v1 audits.")
    parser.add_argument("--audit-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-correct-rate", type=float, default=0.95)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                audit_dirs=[Path(path) for path in args.audit_dir],
                output=Path(args.output),
                minimum_correct_rate=args.minimum_correct_rate,
            ),
            ensure_ascii=False,
        )
    )
