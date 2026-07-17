#!/usr/bin/env python3
"""Compile canonical CueQC v13 labels while excluding unsure from training."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_LABELS = {"drop", "keep", "unsure"}
TRAINING_LABELS = {"drop": 0, "keep": 1, "unsure": -100}


def _rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _label(row: dict[str, Any]) -> str:
    value = str(row.get("label") or row.get("verdict") or "").strip().lower()
    return value if value in VALID_LABELS else ""


def _unique_by_id(rows: list[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("subisland_id") or "").strip()
        if not item_id:
            raise ValueError(f"{name} row is missing subisland_id")
        if item_id in result:
            raise ValueError(f"duplicate {name} subisland_id: {item_id}")
        result[item_id] = row
    return result


def compile_labels(
    *,
    runtime_chunks: Path,
    teacher_labels: Path,
    output: Path,
    manual_overrides: Path | None = None,
    exact_labels: Path | None = None,
) -> dict[str, Any]:
    runtime_rows = _rows(runtime_chunks)
    runtime = _unique_by_id(runtime_rows, name="runtime")
    observed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(teacher_labels):
        item_id = str(row.get("subisland_id") or "").strip()
        label = _label(row)
        if not item_id or not label:
            raise ValueError("teacher row must contain subisland_id and keep/drop/unsure")
        if item_id not in runtime:
            raise ValueError(f"teacher label has no runtime chunk: {item_id}")
        observed[item_id].append(row)
    missing = sorted(set(runtime) - set(observed))
    if missing:
        raise ValueError(f"teacher labels are incomplete; missing {len(missing)} chunks")

    manual = _unique_by_id(_rows(manual_overrides), name="manual override")
    unknown_manual = sorted(set(manual) - set(runtime))
    if unknown_manual:
        raise ValueError(f"manual overrides have no runtime chunk: {unknown_manual[:3]}")
    exact = _unique_by_id(_rows(exact_labels), name="exact label")
    if exact_labels is not None:
        unknown_exact = sorted(set(exact) - set(runtime))
        if unknown_exact:
            raise ValueError(f"exact labels have no runtime chunk: {unknown_exact[:3]}")
        missing_exact = sorted(set(runtime) - set(exact))
        if missing_exact:
            raise ValueError(
                f"exact labels are incomplete; missing {len(missing_exact)} chunks"
            )

    result: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    conflicts = 0
    manual_count = 0
    for chunk in runtime_rows:
        item_id = str(chunk["subisland_id"])
        teacher_rows = observed[item_id]
        teacher_values = sorted({_label(row) for row in teacher_rows})
        conflict = len(teacher_values) > 1
        teacher_label = "unsure" if conflict else teacher_values[0]
        source = "duplicate_request_conflict_to_unsure" if conflict else str(
            teacher_rows[-1].get("label_source") or "omni_teacher"
        )
        if conflict:
            conflicts += 1

        override = manual.get(item_id)
        override_label = _label(override or {})
        if override is not None and not override_label:
            raise ValueError(f"manual override has invalid label: {item_id}")
        canonical = override_label or teacher_label
        if override_label:
            source = "manual_override"
            manual_count += 1
        counts[canonical] += 1
        exact_row = exact.get(item_id)
        result.append(
            {
                "schema": "cueqc_v13_canonical_label_v1",
                "sample_id": str(chunk["sample_id"]),
                "subisland_id": item_id,
                "source_partition": str(chunk.get("source_partition") or "train"),
                "audio": str(chunk["audio"]),
                "start_s": float(chunk["start_s"]),
                "end_s": float(chunk["end_s"]),
                "duration_s": float(chunk["duration_s"]),
                "teacher_label": teacher_label,
                "label": canonical,
                "training_label": TRAINING_LABELS[canonical],
                "training_label_included": canonical in {"drop", "keep"},
                "training_ignore_reason": "teacher_unsure" if canonical == "unsure" else "",
                "label_source": source,
                "teacher_labels_observed": teacher_values,
                "teacher_response_count": len(teacher_rows),
                "teacher_conflict": conflict,
                "manual_override_applied": bool(override_label),
                "exact_core_label": _label(exact_row or {}),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result),
        "utf-8",
    )
    summary = {
        "schema": "cueqc_v13_canonical_label_summary_v1",
        "runtime_chunk_count": len(runtime_rows),
        "canonical_label_counts": dict(sorted(counts.items())),
        "training_label_count": counts["drop"] + counts["keep"],
        "teacher_unsure_ignored": counts["unsure"],
        "duplicate_request_conflict_count": conflicts,
        "manual_override_count": manual_count,
        "exact_label_count": len(exact),
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-chunks", required=True)
    parser.add_argument("--teacher-labels", required=True)
    parser.add_argument("--manual-overrides", default="")
    parser.add_argument("--exact-labels", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(compile_labels(
        runtime_chunks=Path(args.runtime_chunks),
        teacher_labels=Path(args.teacher_labels),
        manual_overrides=Path(args.manual_overrides) if args.manual_overrides else None,
        exact_labels=Path(args.exact_labels) if args.exact_labels else None,
        output=Path(args.output),
    ), ensure_ascii=False))
