#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


OUTPUT_SCHEMA = "semantic_timeline_repaired_events_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _boundary_offset(reference_text: str, left_text: str, right_text: str) -> int:
    joined = left_text + right_text
    starts: list[int] = []
    cursor = 0
    while True:
        index = reference_text.find(joined, cursor)
        if index < 0:
            break
        starts.append(index)
        cursor = index + 1
    if len(starts) != 1:
        raise ValueError(
            f"repair text must occur exactly once in reference text: {left_text!r} | {right_text!r}"
        )
    return starts[0] + len(left_text)


def _original_boundaries(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    units = list(row["text_units"])
    unit_ends: dict[str, int] = {}
    offset = 0
    for unit in units:
        offset += len(str(unit["text"]))
        unit_ends[str(unit["unit_id"])] = offset
    result: dict[int, dict[str, Any]] = {}
    for event in row.get("semantic_events") or []:
        if event.get("status") != "matched":
            continue
        boundary_offset = unit_ends[str(event["left_unit_id"])]
        anchor = (
            float(event["interval_start_s"]) + float(event["interval_end_s"])
        ) / 2.0
        result[boundary_offset] = {
            "coarse_anchor_s": anchor,
            "anchor_source": "omni_semantic_timeline_coarse_anchor",
            "source_event_id": str(event["event_id"]),
        }
    return result


def apply_repairs(
    *, timeline_labels: Path, repair_verdicts: Path, output: Path
) -> list[dict[str, Any]]:
    repairs_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for repair in _rows(repair_verdicts):
        decision = str(repair.get("decision") or "")
        if decision not in {
            "semantic_split",
            "acoustic_continue",
            "outer_only",
            "unsure",
        }:
            raise ValueError(f"invalid or incomplete repair decision: {decision!r}")
        repairs_by_sample[str(repair["sample_id"])].append(repair)

    output_rows: list[dict[str, Any]] = []
    seen_repairs: set[str] = set()
    for source in _rows(timeline_labels):
        sample_id = str(source["sample_id"])
        reference_text = str(source["reference_text"])
        boundaries = _original_boundaries(source)
        blockers: list[dict[str, Any]] = []
        unsure_repairs: list[dict[str, Any]] = []
        for repair in repairs_by_sample.get(sample_id, []):
            seen_repairs.add(str(repair["repair_id"]))
            decision = str(repair["decision"])
            left_text = str(repair["left_text"])
            right_text = str(repair["right_text"])
            boundary_offset: int | None = None
            if not left_text.startswith("（source 开头"):
                boundary_offset = _boundary_offset(
                    reference_text, left_text, right_text
                )
            if decision == "semantic_split":
                if boundary_offset is None:
                    raise ValueError("semantic_split repair requires two text sides")
                boundaries[boundary_offset] = {
                    "coarse_anchor_s": float(repair["time_s"]),
                    "anchor_source": "manual_safe_candidate",
                    "repair_id": str(repair["repair_id"]),
                }
            elif decision in {"acoustic_continue", "outer_only"}:
                if boundary_offset is not None:
                    boundaries.pop(boundary_offset, None)
                blockers.append(
                    {
                        "time_s": float(repair["time_s"]),
                        "scope": decision,
                        "repair_id": str(repair["repair_id"]),
                        "note": str(repair.get("note") or ""),
                    }
                )
            else:
                unsure_repairs.append(dict(repair))

        offsets = sorted(boundaries)
        if any(offset <= 0 or offset >= len(reference_text) for offset in offsets):
            raise ValueError(f"invalid semantic boundary offset for {sample_id}")
        text_units: list[dict[str, Any]] = []
        start = 0
        for index, end in enumerate([*offsets, len(reference_text)]):
            text_units.append(
                {
                    "unit_id": f"u{index:02d}",
                    "text": reference_text[start:end],
                    "kind": "semantic",
                    "label_source": "original_plus_manual_event_repair",
                }
            )
            start = end
        semantic_events = [
            {
                "event_id": f"e{index:02d}",
                "left_unit_id": f"u{index:02d}",
                "right_unit_id": f"u{index + 1:02d}",
                "status": "matched",
                "boundary_text_offset": offset,
                **boundaries[offset],
            }
            for index, offset in enumerate(offsets)
        ]
        output_rows.append(
            {
                "schema": OUTPUT_SCHEMA,
                "sample_id": sample_id,
                "audio": str(source["audio"]),
                "duration_s": float(source["duration_s"]),
                "source": str(source.get("source") or ""),
                "reference_text": reference_text,
                "text_units": text_units,
                "semantic_events": semantic_events,
                "region_blocker_anchors": sorted(
                    blockers, key=lambda item: float(item["time_s"])
                ),
                "unsure_repairs": unsure_repairs,
                "source_teacher_schema": str(source.get("schema") or ""),
                "repair_verdicts": str(repair_verdicts),
            }
        )
    expected_repairs = {
        str(row["repair_id"]) for row in _rows(repair_verdicts)
    }
    if seen_repairs != expected_repairs:
        raise ValueError("repair verdict references an unknown timeline sample")
    _write(output, output_rows)
    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply approved semantic/outer repair decisions to coarse timeline events."
    )
    parser.add_argument("--timeline-labels", required=True)
    parser.add_argument("--repair-verdicts", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = apply_repairs(
        timeline_labels=Path(args.timeline_labels),
        repair_verdicts=Path(args.repair_verdicts),
        output=Path(args.output),
    )
    print(
        json.dumps(
            {
                "schema": OUTPUT_SCHEMA,
                "sample_count": len(rows),
                "semantic_event_count": sum(
                    len(row["semantic_events"]) for row in rows
                ),
                "region_blocker_count": sum(
                    len(row["region_blocker_anchors"]) for row in rows
                ),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
