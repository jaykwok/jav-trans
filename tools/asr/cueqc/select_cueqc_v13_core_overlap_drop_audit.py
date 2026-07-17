#!/usr/bin/env python3
"""Select high-risk Omni drops that substantially overlap approved cores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SAMPLE_RATE = 16000


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(
    *,
    runtime_chunks: Path,
    teacher_labels: Path,
    exact_labels: Path,
    output: Path,
    count: int,
) -> dict[str, Any]:
    runtime = {str(row["subisland_id"]): row for row in _rows(runtime_chunks)}
    teacher = {str(row["subisland_id"]): row for row in _rows(teacher_labels)}
    exact = {str(row["subisland_id"]): row for row in _rows(exact_labels)}
    candidates: list[dict[str, Any]] = []
    for subisland_id, label in teacher.items():
        truth = exact.get(subisland_id)
        chunk = runtime.get(subisland_id)
        if label.get("label") != "drop" or truth is None or chunk is None:
            continue
        overlap_samples = sum(
            int(item.get("overlap_samples") or 0)
            for item in truth.get("semantic_core_overlaps") or []
        )
        if overlap_samples <= 0:
            continue
        overlap_s = overlap_samples / SAMPLE_RATE
        duration_s = max(float(chunk.get("duration_s") or 0.0), 1e-9)
        candidates.append(
            {
                **chunk,
                "teacher_label": "drop",
                "teacher_confidence": float(label.get("confidence") or 0.0),
                "exact_core_overlap_s": overlap_s,
                "exact_core_overlap_ratio": overlap_s / duration_s,
                "exact_semantic_core_overlaps": list(
                    truth.get("semantic_core_overlaps") or []
                ),
                "drop_safety_selection_axis": "highest_exact_core_overlap",
            }
        )
    candidates.sort(
        key=lambda row: (
            float(row["exact_core_overlap_s"]),
            float(row["exact_core_overlap_ratio"]),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for row in candidates:
        source = str(row["sample_id"])
        if source in seen_sources:
            continue
        selected.append(row)
        seen_sources.add(source)
        if len(selected) >= max(1, int(count)):
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema": "cueqc_v13_core_overlap_drop_selection_summary_v1",
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "unique_source_count": len(seen_sources),
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-chunks", required=True)
    parser.add_argument("--teacher-labels", required=True)
    parser.add_argument("--exact-labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            select(
                runtime_chunks=Path(args.runtime_chunks),
                teacher_labels=Path(args.teacher_labels),
                exact_labels=Path(args.exact_labels),
                output=Path(args.output),
                count=args.count,
            ),
            ensure_ascii=False,
        )
    )
