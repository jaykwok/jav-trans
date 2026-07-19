#!/usr/bin/env python3
"""Analyze Scorer v10 argmax fragmentation against canonical compositions."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CONTRACT_ID = "boundary_acoustic_binary_v12"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in materialized),
        encoding="utf-8",
    )


def _truth_fragmentation(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    truth_spans = sorted(
        (
            span
            for span in row.get("truth_spans", [])
            if span.get("label") == "truth_speech"
        ),
        key=lambda span: (int(span["start_frame"]), int(span["end_frame"])),
    )
    predictions = sorted(
        row.get("prediction_spans", []),
        key=lambda span: (int(span["start_frame"]), int(span["end_frame"])),
    )
    fragmented_runs: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for ordinal, truth in enumerate(truth_spans, start=1):
        start = int(truth["start_frame"])
        end = int(truth["end_frame"])
        overlaps = [
            span
            for span in predictions
            if int(span["end_frame"]) > start and int(span["start_frame"]) < end
        ]
        if len(overlaps) <= 1:
            continue
        fragmented_runs.append(
            {
                "core_ordinal": ordinal,
                "truth_start_frame": start,
                "truth_end_frame": end,
                "predicted_run_count": len(overlaps),
            }
        )
        duration = max(1, end - start)
        for left, right in zip(overlaps, overlaps[1:]):
            gap_start = max(start, int(left["end_frame"]))
            gap_end = min(end, int(right["start_frame"]))
            gap_frames = max(0, gap_end - gap_start)
            center_ratio = ((gap_start + gap_end) / 2.0 - start) / duration
            gaps.append(
                {
                    "core_ordinal": ordinal,
                    "start_frame": gap_start,
                    "end_frame": gap_end,
                    "gap_frames": gap_frames,
                    "center_ratio": center_ratio,
                    "position": (
                        "outer_10pct"
                        if center_ratio < 0.1 or center_ratio > 0.9
                        else "middle_80pct"
                    ),
                }
            )
    return fragmented_runs, gaps


def _snr_bucket(value: float | None) -> str:
    if value is None:
        return "none"
    if value < 12.0:
        return "lt_12db"
    if value < 16.0:
        return "12_to_16db"
    if value < 20.0:
        return "16_to_20db"
    return "ge_20db"


def _rate_rows(counter: dict[tuple[str, str], list[int]]) -> list[dict[str, Any]]:
    return [
        {
            "partition": partition,
            "group": group,
            "row_count": counts[0],
            "fragmented_row_count": counts[1],
            "fragmentation_rate": counts[1] / max(counts[0], 1),
        }
        for (partition, group), counts in sorted(counter.items())
    ]


def analyze(
    *, predictions: Path, canonical_sources: Path, output_dir: Path
) -> dict[str, Any]:
    prediction_rows = _rows(predictions)
    canonical_rows = _rows(canonical_sources)
    canonical_by_id: dict[str, dict[str, Any]] = {}
    for row in canonical_rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in canonical_by_id:
            raise ValueError("canonical source IDs must be non-empty and unique")
        if row.get("boundary_serialization_contract_id") != CONTRACT_ID:
            raise ValueError("canonical source uses the wrong central contract")
        canonical_by_id[source_id] = row

    composition_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    snr_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    overlay_type_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    partition_rows = Counter()
    partition_fragmented = Counter()
    gap_positions = Counter()
    gap_lengths: dict[str, list[int]] = defaultdict(list)
    core_ordinals = Counter()
    train_hardcases: list[dict[str, Any]] = []
    heldout_hardcases: list[dict[str, Any]] = []

    for prediction in prediction_rows:
        source_id = str(prediction.get("source_id") or "")
        canonical = canonical_by_id.get(source_id)
        if canonical is None:
            raise ValueError(f"prediction source is absent from canonical sources: {source_id}")
        if prediction.get("row_role") != canonical.get("row_role"):
            raise ValueError(f"row role mismatch: {source_id}")
        if prediction.get("partition") != canonical.get("partition"):
            raise ValueError(f"partition mismatch: {source_id}")
        if prediction.get("row_role") != "speech":
            continue

        partition = str(prediction["partition"])
        fragmented_runs, gaps = _truth_fragmentation(prediction)
        fragmented = bool(fragmented_runs)
        overlay = dict(canonical.get("additive_overlay") or {})
        overlay_group = "overlay" if overlay else "clean"
        partition_rows[partition] += 1
        partition_fragmented[partition] += int(fragmented)
        composition_counts[(partition, overlay_group)][0] += 1
        composition_counts[(partition, overlay_group)][1] += int(fragmented)

        overlay_type = "none"
        achieved_snr_db: float | None = None
        if overlay:
            overlay_type = str(dict(overlay.get("source") or {}).get("background_type") or "unknown")
            raw_snr = dict(overlay.get("mix") or {}).get("achieved_snr_db")
            achieved_snr_db = None if raw_snr is None else float(raw_snr)
            overlay_type_counts[(partition, overlay_type)][0] += 1
            overlay_type_counts[(partition, overlay_type)][1] += int(fragmented)
            snr_counts[(partition, _snr_bucket(achieved_snr_db))][0] += 1
            snr_counts[(partition, _snr_bucket(achieved_snr_db))][1] += int(fragmented)

        for gap in gaps:
            gap_positions[(partition, str(gap["position"]))] += 1
            gap_lengths[partition].append(int(gap["gap_frames"]))
        for run in fragmented_runs:
            core_ordinals[(partition, int(run["core_ordinal"]))] += 1

        if not fragmented:
            continue
        hardcase = {
            "schema": "speech_scorer_v10_fragmentation_hardcase_v1",
            "boundary_serialization_contract_id": CONTRACT_ID,
            "source_id": source_id,
            "partition": partition,
            "audio": str(canonical["audio"]),
            "core_ids": list(canonical.get("core_ids") or ()),
            "overlay_group": overlay_group,
            "overlay_type": overlay_type,
            "achieved_snr_db": achieved_snr_db,
            "fragmented_truth_runs": fragmented_runs,
            "internal_gaps": gaps,
            "diagnostic_only": True,
        }
        if partition == "train":
            train_hardcases.append(hardcase)
        else:
            heldout_hardcases.append(hardcase)

    if any(row["partition"] != "train" for row in train_hardcases):
        raise AssertionError("held-out source leaked into train hardcases")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_fragmentation_hardcases.jsonl"
    heldout_path = output_dir / "heldout_fragmentation_hardcases.jsonl"
    _write_jsonl(train_path, train_hardcases)
    _write_jsonl(heldout_path, heldout_hardcases)

    gap_summary = {}
    for partition in ("train", "val", "test"):
        lengths = sorted(gap_lengths[partition])
        gap_summary[partition] = {
            "gap_count": len(lengths),
            "gap_le_1_frame": sum(value <= 1 for value in lengths),
            "gap_le_2_frames": sum(value <= 2 for value in lengths),
            "gap_le_5_frames": sum(value <= 5 for value in lengths),
            "gap_p50_frames": lengths[len(lengths) // 2] if lengths else 0,
            "gap_p95_frames": (
                lengths[int(0.95 * (len(lengths) - 1))] if lengths else 0
            ),
            "outer_10pct_count": gap_positions[(partition, "outer_10pct")],
            "middle_80pct_count": gap_positions[(partition, "middle_80pct")],
            "core1_fragmented_run_count": core_ordinals[(partition, 1)],
            "core2_fragmented_run_count": core_ordinals[(partition, 2)],
        }

    summary = {
        "schema": "speech_scorer_v10_fragmentation_distribution_audit_v1",
        "boundary_serialization_contract_id": CONTRACT_ID,
        "predictions": str(predictions),
        "canonical_sources": str(canonical_sources),
        "partition_speech_rows": dict(partition_rows),
        "partition_fragmented_rows": dict(partition_fragmented),
        "composition_fragmentation": _rate_rows(composition_counts),
        "snr_fragmentation": _rate_rows(snr_counts),
        "overlay_type_fragmentation": _rate_rows(overlay_type_counts),
        "gap_distribution": gap_summary,
        "train_hardcase_count": len(train_hardcases),
        "heldout_hardcase_count": len(heldout_hardcases),
        "train_hardcases": str(train_path),
        "heldout_hardcases": str(heldout_path),
        "hardcase_policy": {
            "diagnostic_only": True,
            "heldout_never_enters_training": True,
            "does_not_repeat_or_repartition_core_identity": True,
            "does_not_modify_training_manifest": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--canonical-sources", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = analyze(
        predictions=Path(args.predictions),
        canonical_sources=Path(args.canonical_sources),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
