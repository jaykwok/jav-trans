#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_runs(candidates: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("label") == "safe":
            current.append(candidate)
            continue
        if current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def evaluate(verdicts: Path, *, output: Path | None = None) -> dict[str, Any]:
    rows = _rows(verdicts)
    if not rows:
        raise ValueError("manual verdicts are empty")
    label_counts: Counter[str] = Counter()
    event_rows: list[dict[str, Any]] = []
    for row in rows:
        candidates = list(row.get("candidates") or [])
        label_counts.update(str(candidate.get("label") or "") for candidate in candidates)
        runs = safe_runs(candidates)
        complete = bool(row.get("complete")) and all(
            candidate.get("label") in {
                "left_clipped",
                "safe",
                "right_clipped",
                "unsure",
            }
            for candidate in candidates
        )
        single_run = complete and len(runs) == 1
        event_rows.append(
            {
                "event_key": str(row["event_key"]),
                "complete": complete,
                "safe_candidate_count": sum(len(run) for run in runs),
                "safe_run_count": len(runs),
                "safe_runs": [
                    {
                        "start_s": float(run[0]["time_s"]),
                        "end_s": float(run[-1]["time_s"]),
                        "candidate_ids": [str(item["candidate_id"]) for item in run],
                    }
                    for run in runs
                ],
                "paired_edge_eligible": single_run,
                "route": (
                    "paired_inner_edges"
                    if single_run
                    else "abstain_and_review_semantic_event_decomposition"
                ),
            }
        )
    complete_count = sum(row["complete"] for row in event_rows)
    covered_count = sum(row["safe_run_count"] > 0 for row in event_rows)
    paired_count = sum(row["paired_edge_eligible"] for row in event_rows)
    total = len(event_rows)
    summary = {
        "schema": "semantic_anchor_learned_proposer_gate_v1",
        "verdicts": str(verdicts),
        "event_count": total,
        "complete_event_count": complete_count,
        "safe_covered_event_count": covered_count,
        "single_safe_run_event_count": paired_count,
        "multi_safe_run_event_count": sum(
            row["safe_run_count"] > 1 for row in event_rows
        ),
        "safe_coverage": covered_count / total,
        "paired_edge_eligibility": paired_count / total,
        "label_counts": dict(label_counts),
        "gate_threshold": 0.95,
        "coverage_gate_passed": covered_count / total >= 0.95,
        "paired_edge_gate_passed": paired_count / total >= 0.95,
        "training_ready": paired_count / total >= 0.95,
        "events": event_rows,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate safe coverage and contiguous paired-edge eligibility."
    )
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                Path(args.verdicts),
                output=Path(args.output) if args.output else None,
            ),
            ensure_ascii=False,
        )
    )
