#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.boundary.boundary_preference import (  # noqa: E402
    CANDIDATE_SCHEMA,
    label_by_item,
    normalized_preference,
    read_jsonl,
    summarize_preferences,
    write_jsonl,
)
from tools.boundary.build_refiner_frame_sequence_dataset import DATASET_SCHEMA  # noqa: E402


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def _candidate_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("schema") != CANDIDATE_SCHEMA:
            raise ValueError(f"unsupported candidate schema: {row.get('schema')!r}")
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("candidate row is missing candidate_id")
        result[candidate_id] = row
    return result


def _winner_target(candidate: Mapping[str, Any], winner: str) -> tuple[list[float], list[float]]:
    if winner not in {"baseline", "challenger"}:
        raise ValueError(f"unsupported winner: {winner!r}")
    axis = str(candidate.get("axis") or "")
    boundary = candidate.get(f"{winner}_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError(f"candidate is missing {winner}_boundary")
    raw_left_end = float(candidate.get("raw_left_end_s") or 0.0)
    raw_right_start = float(candidate.get("raw_right_start_s") or 0.0)
    if axis == "right.start":
        target = float(boundary.get("right_start_s") or 0.0) - raw_right_start
        return [round(target, 6), 0.0], [1.0, 0.0]
    if axis == "left.end":
        target = float(boundary.get("left_end_s") or 0.0) - raw_left_end
        return [0.0, round(target, 6)], [0.0, 0.6]
    raise ValueError(f"unsupported preference axis: {axis!r}")


def compile_rows(
    *,
    labels: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gate_summary = summarize_preferences(labels, answers)
    if not gate_summary["gate_passed"]:
        raise ValueError("preference gate did not pass; refusing to compile v5.1 supervision")

    labels_by_id = label_by_item(labels)
    answers_by_id = {
        str(row.get("item_id") or ""): row
        for row in answers
        if row.get("item_id")
    }
    candidates_by_id = _candidate_map(candidates)
    output: list[dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    winner_counts: Counter[str] = Counter()
    axis_counts: Counter[str] = Counter()

    for item_id, answer in sorted(
        answers_by_id.items(),
        key=lambda item: int(item[1].get("display_index") or 0),
    ):
        if bool(answer.get("is_hidden_duplicate")):
            skip_counts["hidden_duplicate"] += 1
            continue
        winner = normalized_preference(labels_by_id.get(item_id), answer)
        if winner not in {"baseline", "challenger"}:
            skip_counts[winner or "missing"] += 1
            continue
        candidate_id = str(answer.get("candidate_id") or "")
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"missing candidate row for {candidate_id!r}")
        features = list(candidate.get("sequence_feature") or [])
        feature_names = list(candidate.get("feature_names") or [])
        if not features or len(features) != len(feature_names):
            raise ValueError(f"invalid sequence feature for {candidate_id}")
        targets, weights = _winner_target(candidate, winner)
        axis = str(candidate.get("axis") or "")
        output.append(
            {
                "schema": DATASET_SCHEMA,
                "feature_schema": str(candidate.get("feature_schema") or ""),
                "feature_schema_hash": str(candidate.get("feature_schema_hash") or ""),
                "feature_signature": dict(candidate.get("feature_signature") or {}),
                "audio_id": candidate_id,
                "source": "boundary_preference_v1",
                "label_index": len(output),
                "feature_names": feature_names,
                "feature_dim": len(feature_names),
                "sequence_features": [features],
                "sequence_boundary_delta_targets": [targets],
                "sequence_boundary_delta_weights": [weights],
                "sequence_reasons": [f"preference_{axis}_{winner}"],
                "gap_indexes": [int(candidate.get("boundary_index") or 0)],
                "metadata": {
                    "preference_item_id": item_id,
                    "candidate_id": candidate_id,
                    "video_id": str(candidate.get("video_id") or ""),
                    "axis": axis,
                    "winner": winner,
                    "offset_frames": int(candidate.get("offset_frames") or 0),
                    "offset_ms": int(candidate.get("offset_ms") or 0),
                    "only_compared_axis_supervised": True,
                },
            }
        )
        winner_counts[winner] += 1
        axis_counts[axis] += 1
    return output, {
        "gate": gate_summary,
        "compiled_rows": len(output),
        "compiled_sequence_items": len(output),
        "winner_counts": dict(winner_counts),
        "axis_counts": dict(axis_counts),
        "skip_counts": dict(skip_counts),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile gate-passing blind preference labels into "
            "boundary_refiner_frame_sequence_dataset_v5 rows."
        )
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels_path = project_path(args.labels)
    answer_key_path = project_path(args.answer_key)
    candidates_path = project_path(args.candidates)
    output_path = project_path(args.output_jsonl)
    summary_path = project_path(args.summary_json)
    labels = read_jsonl(labels_path)
    answers = read_jsonl(answer_key_path)
    candidates = read_jsonl(candidates_path)
    try:
        rows, summary = compile_rows(
            labels=labels,
            answers=answers,
            candidates=candidates,
        )
    except ValueError as exc:
        gate = summarize_preferences(labels, answers)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "compiled_rows": 0,
                    "error": str(exc),
                    "gate": gate,
                    "labels_path": str(labels_path),
                    "answer_key_path": str(answer_key_path),
                    "candidates_path": str(candidates_path),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"compile_refused={exc}", file=sys.stderr)
        print(f"summary={summary_path}")
        return 2

    write_jsonl(output_path, rows)
    summary.update(
        {
            "schema": DATASET_SCHEMA,
            "output_jsonl": str(output_path),
            "labels_path": str(labels_path),
            "answer_key_path": str(answer_key_path),
            "candidates_path": str(candidates_path),
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"dataset={output_path}")
    print(f"summary={summary_path}")
    print(f"compiled_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
