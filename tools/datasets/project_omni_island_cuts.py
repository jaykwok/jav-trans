#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "semantic_split_v3_island_candidate_projection_v1"
PROMPT_VERSION = "semantic_split_v3_island_candidate_projection_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def project_island(
    island: dict[str, Any], label: dict[str, Any]
) -> dict[str, Any]:
    candidates = sorted(
        island.get("candidates") or [], key=lambda row: float(row["relative_time_s"])
    )
    projected = []
    conflicts = []
    last_index = -1
    for cut in sorted(label.get("cuts") or [], key=lambda row: float(row["time_s"])):
        teacher_time = float(cut["time_s"])
        available = [
            (index, candidate)
            for index, candidate in enumerate(candidates)
            if index > last_index
        ]
        if not available:
            conflicts.append({"teacher_time_s": teacher_time, "reason": "no_candidate"})
            continue
        index, candidate = min(
            available,
            key=lambda item: (
                abs(float(item[1]["relative_time_s"]) - teacher_time),
                -float(item[1].get("p_cut") or 0.0),
            ),
        )
        last_index = index
        projected_time = float(candidate["relative_time_s"])
        projected.append(
            {
                **cut,
                "time_s": projected_time,
                "teacher_time_s": teacher_time,
                "projection_delta_s": projected_time - teacher_time,
                "candidate_feature_index": int(candidate["feature_index"]),
                "candidate_kind": str(candidate.get("kind") or ""),
                "candidate_p_cut": float(candidate.get("p_cut") or 0.0),
            }
        )
    projected.sort(key=lambda row: float(row["time_s"]))
    return {
        "schema": SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "island_id": str(island["island_id"]),
        "window_id": str(island["window_id"]),
        "video_id": str(island["video_id"]),
        "span_index": int(island["span_index"]),
        "span_start_s": float(island["span_start_s"]),
        "span_end_s": float(island["span_end_s"]),
        "duration_s": float(island["duration_s"]),
        "cuts": projected,
        "complete_search": not conflicts,
        "projection_conflicts": conflicts,
        "reason": "Omni semantic cuts projected monotonically to nearest proposer candidates.",
    }


def run(*, selected: Path, labels: Path, output_dir: Path) -> dict[str, Any]:
    islands = {str(row["island_id"]): row for row in _read_jsonl(selected)}
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        project_island(islands[str(label["island_id"])], label)
        for label in _read_jsonl(labels)
    ]
    output = output_dir / "projected_island_labels.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    deltas = [
        abs(float(cut["projection_delta_s"])) for row in rows for cut in row["cuts"]
    ]
    summary = {
        "schema": "semantic_split_v3_island_candidate_projection_summary_v1",
        "island_count": len(rows),
        "teacher_cut_count": sum(len(row.get("cuts") or []) for row in _read_jsonl(labels)),
        "projected_cut_count": sum(len(row["cuts"]) for row in rows),
        "conflict_count": sum(len(row["projection_conflicts"]) for row in rows),
        "max_abs_delta_s": max(deltas, default=0.0),
        "mean_abs_delta_s": sum(deltas) / len(deltas) if deltas else 0.0,
        "output": str(output),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                selected=Path(args.selected),
                labels=Path(args.labels),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
