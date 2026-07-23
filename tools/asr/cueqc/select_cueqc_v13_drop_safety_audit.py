#!/usr/bin/env python3
"""Select long and model-disputed Omni drops for manual safety audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _candidate_membership_score(row: dict[str, Any]) -> float:
    """Use only upstream Scorer evidence; Inner has not run before CueQC."""

    candidate = dict(row.get("pre_asr_candidate") or {})
    return max(
        (
            float(candidate.get("scorer_speech_p90") or 0.0),
            float(candidate.get("scorer_speech_mean") or 0.0),
            float(candidate.get("scorer_speech_active_ratio_05") or 0.0),
        )
    )


def select(
    *, runtime_chunks: Path, labels: Path, output: Path, per_axis: int
) -> dict[str, Any]:
    runtime = {str(row["subisland_id"]): row for row in _rows(runtime_chunks)}
    drops = [row for row in _rows(labels) if row.get("label") == "drop"]
    candidates = [runtime[str(row["subisland_id"])] for row in drops]
    selected: list[dict[str, Any]] = []
    seen_sources: set[str] = set()

    def add(rows: list[dict[str, Any]], axis: str) -> None:
        for row in rows:
            source = str(row["sample_id"])
            if source in seen_sources:
                continue
            selected.append({**row, "drop_safety_selection_axis": axis})
            seen_sources.add(source)
            if sum(item["drop_safety_selection_axis"] == axis for item in selected) >= per_axis:
                return

    add(
        sorted(candidates, key=lambda row: float(row["duration_s"]), reverse=True),
        "longest_drop",
    )
    add(
        sorted(candidates, key=_candidate_membership_score, reverse=True),
        "highest_upstream_candidate_membership",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema": "cueqc_v13_drop_safety_selection_summary_v2",
        "selected_count": len(selected),
        "per_axis": per_axis,
        "axes": ["longest_drop", "highest_upstream_candidate_membership"],
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-chunks", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-axis", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            select(
                runtime_chunks=Path(args.runtime_chunks),
                labels=Path(args.labels),
                output=Path(args.output),
                per_axis=args.per_axis,
            ),
            ensure_ascii=False,
        )
    )
