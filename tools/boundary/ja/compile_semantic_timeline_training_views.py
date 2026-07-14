#!/usr/bin/env python3
"""Compile one semantic-timeline corpus into model-specific training manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.boundary.ja.label_semantic_timeline_with_omni import (
    PROMPT_VERSION,
    SCHEMA,
)


SUMMARY_SCHEMA = "semantic_timeline_training_views_summary_v1"
MAX_SOURCE_COUNT = 4096


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(materialized)


def _approved_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        str(row["sample_id"])
        for row in _rows(path)
        if row.get("schema") == "semantic_timeline_manual_verdict_v1"
        and row.get("verdict") == "approve"
    }


def compile_views(
    *,
    labels: Path,
    output_dir: Path,
    manual_verdicts: Path | None = None,
    max_sources: int = MAX_SOURCE_COUNT,
) -> dict[str, Any]:
    rows = _rows(labels)
    if not rows:
        raise ValueError("semantic timeline labels are empty")
    if len(rows) > max_sources:
        raise ValueError(f"semantic timeline source count exceeds cap {max_sources}")
    sample_ids = [str(row["sample_id"]) for row in rows]
    sources = [str(row.get("source") or "") for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("semantic timeline sample_id must be globally unique")
    if len(set(sources)) != len(sources):
        raise ValueError("each source/core may appear at most once in the corpus")
    for row in rows:
        if row.get("schema") != SCHEMA or row.get("prompt_version") != PROMPT_VERSION:
            raise ValueError("semantic timeline schema or prompt version mismatch")

    approved = _approved_ids(manual_verdicts)
    scorer_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    for row in rows:
        common = {
            "sample_id": str(row["sample_id"]),
            "source": str(row.get("source") or ""),
            "audio": str(row["audio"]),
            "duration_s": float(row["duration_s"]),
            "reference_text": str(row["reference_text"]),
            "partition_key": str(row.get("source") or row["sample_id"]),
            "teacher_schema": SCHEMA,
            "teacher_prompt_version": PROMPT_VERSION,
        }
        scorer = row["scorer_view"]
        scorer_rows.append(
            {
                "schema": "semantic_speech_scorer_source_labels_v1",
                **common,
                "semantic_spans": scorer["semantic_spans"],
                "nonsemantic_frame_spans": scorer["nonsemantic_complement_spans"],
                "source_membership": scorer["source_membership"],
                "label_contract": "semantic_content_plus_high_recall_membership_v1",
            }
        )
        outer = row["outer_refiner_view"]
        outer_rows.append(
            {
                "schema": "outer_edge_refiner_source_targets_v1",
                **common,
                "target": outer,
                "context_contract": "full_source_with_leading_and_trailing_nonsemantic_context",
            }
        )
        split_rows.append(
            {
                "schema": "semantic_split_event_sources_v1",
                **common,
                "text_units": row["text_units"],
                "semantic_events": row["semantic_events"],
                "projection_status": "requires_runtime_proposer_candidates",
                "timing_contract": "ordered_event_projection_not_shared_cut_time",
            }
        )
        for event in row["semantic_events"]:
            inner_rows.append(
                {
                    "schema": "inner_edge_event_seed_v1",
                    **common,
                    **event,
                    "label_status": "requires_candidate_safe_zone_teacher",
                    "allowed_labels": [
                        "left_clipped",
                        "safe",
                        "right_clipped",
                        "unsure",
                    ],
                    "training_eligible": False,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "scorer": output_dir / "scorer_sources.jsonl",
        "outer_refiner": output_dir / "outer_refiner_targets.jsonl",
        "semantic_split": output_dir / "semantic_split_event_sources.jsonl",
        "inner_refiner": output_dir / "inner_refiner_event_seeds.jsonl",
    }
    counts = {
        "scorer": _write(outputs["scorer"], scorer_rows),
        "outer_refiner": _write(outputs["outer_refiner"], outer_rows),
        "semantic_split": _write(outputs["semantic_split"], split_rows),
        "inner_refiner_event_seeds": _write(outputs["inner_refiner"], inner_rows),
    }
    all_approved = bool(manual_verdicts) and approved == set(sample_ids)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "source_count": len(rows),
        "unique_source_count": len(set(sources)),
        "max_source_use_count": 1,
        "source_cap": max_sources,
        "counts": counts,
        "teacher_gate_approved": all_approved,
        "training_ready": all_approved,
        "inner_training_ready": False,
        "inner_blocker": "candidate safe-zone labels have not been produced",
        "cueqc_status": "deferred_until_new_boundary_chunks_are_exported",
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile trainable Scorer/Outer/Split views and unlabeled Inner seeds from semantic timelines."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manual-verdicts", default="")
    parser.add_argument("--max-sources", type=int, default=MAX_SOURCE_COUNT)
    args = parser.parse_args()
    if not 1 <= args.max_sources <= MAX_SOURCE_COUNT:
        parser.error(f"--max-sources must be in 1..{MAX_SOURCE_COUNT}")
    return args


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            compile_views(
                labels=Path(args.labels),
                output_dir=Path(args.output_dir),
                manual_verdicts=(
                    Path(args.manual_verdicts) if args.manual_verdicts else None
                ),
                max_sources=args.max_sources,
            ),
            ensure_ascii=False,
        )
    )
