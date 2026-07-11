#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "timeline_teacher_fused_label_v2"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _index(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): row for row in _read_jsonl(path)}


def _unit_index(row: dict[str, Any], field: str) -> dict[str, dict[str, Any]]:
    return {str(unit["unit_id"]): unit for unit in row.get(field) or []}


def fuse_unit(
    forced: dict[str, Any],
    omni: dict[str, Any] | None,
    *,
    omni_min_confidence: float,
    agreement_tolerance_s: float,
) -> dict[str, Any]:
    forced_score = float(forced.get("alignment_score") or 0.0)
    forced_ready = (
        float(forced.get("end_s") or 0.0) > float(forced.get("start_s") or 0.0)
    )
    omni = omni or {}
    omni_confidence = float(omni.get("confidence") or 0.0)
    omni_ready = (
        str(omni.get("status") or "") == "matched"
        and omni_confidence >= omni_min_confidence
        and float(omni.get("end_s") or 0.0) > float(omni.get("start_s") or 0.0)
    )
    source = "unresolved"
    trainable = False
    review_required = False
    teacher_agreement = "unresolved"
    start_s = end_s = 0.0
    start_delta = end_delta = None
    if omni_ready:
        start_s = float(omni["start_s"])
        end_s = float(omni["end_s"])
        if forced_ready:
            start_delta = abs(float(forced["start_s"]) - start_s)
            end_delta = abs(float(forced["end_s"]) - end_s)
            if max(start_delta, end_delta) <= agreement_tolerance_s:
                source = "omni_primary_forced_agree"
                teacher_agreement = "agree"
                trainable = True
            else:
                source = "omni_primary_forced_conflict"
                teacher_agreement = "conflict"
                review_required = True
        else:
            source = "omni_primary_unchecked"
            teacher_agreement = "forced_missing"
            trainable = True
    elif forced_ready:
        source = "forced_only_review"
        teacher_agreement = "omni_missing"
        review_required = True
    return {
        "unit_id": forced["unit_id"],
        "text": forced["text"],
        "start_s": start_s,
        "end_s": end_s,
        "source": source,
        "trainable": trainable,
        "review_required": review_required,
        "teacher_agreement": teacher_agreement,
        "forced_score": forced_score,
        "omni_confidence": omni_confidence,
        "start_delta_s": start_delta,
        "end_delta_s": end_delta,
    }


def fuse(
    *,
    forced_labels: Path,
    omni_labels: Path,
    output_dir: Path,
    omni_min_confidence: float = 0.80,
    agreement_tolerance_s: float = 0.32,
    minimum_trainable_coverage: float = 0.60,
) -> dict[str, Any]:
    forced_index = _index(forced_labels)
    omni_index = _index(omni_labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    source_counts: dict[str, int] = {}
    trainable_items = 0
    for item_id, forced_row in sorted(forced_index.items()):
        omni_row = omni_index.get(item_id) or {}
        omni_units = _unit_index(omni_row, "units")
        units = [
            fuse_unit(
                forced,
                omni_units.get(str(forced["unit_id"])),
                omni_min_confidence=omni_min_confidence,
                agreement_tolerance_s=agreement_tolerance_s,
            )
            for forced in forced_row.get("word_units") or []
        ]
        for unit in units:
            source = str(unit["source"])
            source_counts[source] = source_counts.get(source, 0) + 1
        trainable_units = [unit for unit in units if unit["trainable"]]
        coverage = len(trainable_units) / len(units) if units else 0.0
        item_trainable = coverage >= minimum_trainable_coverage
        trainable_items += int(item_trainable)
        absolute_offset = float(forced_row.get("word_units", [{}])[0].get("absolute_start_s", 0.0))
        if forced_row.get("word_units"):
            first_forced = forced_row["word_units"][0]
            absolute_offset = float(first_forced["absolute_start_s"]) - float(first_forced["start_s"])
        display_start = min((unit["start_s"] for unit in trainable_units), default=0.0)
        display_end = max((unit["end_s"] for unit in trainable_units), default=0.0)
        rows.append(
            {
                "schema": SCHEMA,
                "item_id": item_id,
                "source_id": forced_row["source_id"],
                "source_chunk_index": forced_row["source_chunk_index"],
                "duration_s": forced_row["duration_s"],
                "transcript": forced_row["transcript"],
                "units": units,
                "trainable_unit_count": len(trainable_units),
                "unit_count": len(units),
                "trainable_coverage": coverage,
                "item_trainable": item_trainable,
                "display_start_s": display_start,
                "display_end_s": display_end,
                "absolute_display_start_s": absolute_offset + display_start,
                "absolute_display_end_s": absolute_offset + display_end,
            }
        )
    labels_path = output_dir / "fused_timeline_labels.jsonl"
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema": "timeline_teacher_fusion_summary_v2",
        "selection_policy": "omni_primary_forced_validation",
        "item_count": len(rows),
        "trainable_item_count": trainable_items,
        "source_counts": source_counts,
        "omni_min_confidence": omni_min_confidence,
        "agreement_tolerance_s": agreement_tolerance_s,
        "minimum_trainable_coverage": minimum_trainable_coverage,
        "labels": str(labels_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forced-labels", required=True)
    parser.add_argument("--omni-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--omni-min-confidence", type=float, default=0.80)
    parser.add_argument("--agreement-tolerance-s", type=float, default=0.32)
    parser.add_argument("--minimum-trainable-coverage", type=float, default=0.60)
    args = parser.parse_args()
    print(
        json.dumps(
            fuse(
                forced_labels=Path(args.forced_labels),
                omni_labels=Path(args.omni_labels),
                output_dir=Path(args.output_dir),
                omni_min_confidence=args.omni_min_confidence,
                agreement_tolerance_s=args.agreement_tolerance_s,
                minimum_trainable_coverage=args.minimum_trainable_coverage,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
