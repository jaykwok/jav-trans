#!/usr/bin/env python3
"""Evaluate manual verdicts from the acoustic Split v3 fixed audit."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ITEM_SCHEMA = "acoustic_split_candidate_audit_v1"
VERDICT_SCHEMA = "acoustic_split_candidate_manual_verdict_v1"
SUMMARY_SCHEMA = "acoustic_split_candidate_gate_v1"
ALLOWED_LABELS = {"split", "continue", "unsure"}
ALLOWED_COVERAGE = {"complete", "missed", "unsure"}
ALLOWED_INNER_VERDICTS = {"correct", "clipped", "too_wide", "abstain", "unsure"}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(*, items: Path, verdicts: Path, output: Path) -> dict[str, Any]:
    item_rows = _rows(items)
    if not item_rows or any(row.get("schema") != ITEM_SCHEMA for row in item_rows):
        raise ValueError("acoustic Split audit items are empty or incompatible")
    verdict_by_id = {
        str(row["sample_id"]): row
        for row in _rows(verdicts)
        if row.get("schema") == VERDICT_SCHEMA
    }
    label_counts: Counter[str] = Counter()
    inner_counts: Counter[str] = Counter()
    coverage_counts: Counter[str] = Counter()
    complete_sources = 0
    source_results: list[dict[str, Any]] = []
    for item in item_rows:
        sample_id = str(item["sample_id"])
        verdict = verdict_by_id.get(sample_id) or {}
        candidate_by_id = {
            str(row.get("candidate_id") or ""): row
            for row in verdict.get("candidates") or []
        }
        labels: list[str] = []
        inner_verdicts: list[str] = []
        for candidate in item["candidates"]:
            verdict_candidate = candidate_by_id.get(str(candidate["candidate_id"])) or {}
            label = str(verdict_candidate.get("label") or "unreviewed")
            inner_verdict = str(verdict_candidate.get("inner_verdict") or "not_reviewed")
            labels.append(label)
            inner_verdicts.append(inner_verdict)
        coverage = str(verdict.get("coverage") or "unreviewed")
        complete = (
            all(label in ALLOWED_LABELS for label in labels)
            and coverage in ALLOWED_COVERAGE
        )
        complete_sources += int(complete)
        if complete:
            label_counts.update(labels)
            inner_counts.update(
                inner_verdict
                for label, inner_verdict in zip(labels, inner_verdicts, strict=True)
                if label == "split" and inner_verdict in ALLOWED_INNER_VERDICTS
            )
            coverage_counts.update([coverage])
        source_results.append(
            {
                "sample_id": sample_id,
                "complete": complete,
                "coverage": coverage,
                "label_counts": dict(Counter(labels)),
                "inner_verdict_counts": dict(
                    Counter(
                        inner_verdict
                        for label, inner_verdict in zip(
                            labels, inner_verdicts, strict=True
                        )
                        if label == "split"
                    )
                ),
                "note": str(verdict.get("note") or ""),
            }
        )
    source_count = len(item_rows)
    proposal_coverage = coverage_counts["complete"] / source_count if source_count else 0.0
    summary = {
        "schema": SUMMARY_SCHEMA,
        "source_count": source_count,
        "complete_source_count": complete_sources,
        "candidate_count": sum(len(row["candidates"]) for row in item_rows),
        "label_counts": dict(sorted(label_counts.items())),
        "bootstrap_inner_verdict_counts": dict(sorted(inner_counts.items())),
        "coverage_counts": dict(sorted(coverage_counts.items())),
        "proposal_coverage": proposal_coverage,
        "manual_fixed_gate_pass": complete_sources == source_count
        and coverage_counts["complete"] == source_count,
        "has_split_supervision": label_counts["split"] > 0,
        "has_continue_supervision": label_counts["continue"] > 0,
        "bootstrap_inner_review_complete": sum(inner_counts.values())
        == label_counts["split"],
        "bootstrap_inner_issue_count": sum(
            inner_counts[label]
            for label in ("clipped", "too_wide", "abstain", "unsure")
        ),
        "training_ready": complete_sources == source_count
        and coverage_counts["complete"] == source_count
        and label_counts["split"] > 0
        and label_counts["continue"] > 0,
        "decision_contract": "conservative_acoustic_boundary_fixed_gate_v1",
        "sources": source_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate acoustic Split v3 manual audit.")
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
