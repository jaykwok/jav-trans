#!/usr/bin/env python3
"""Score a Scorer v10 checkpoint and select a listenable residual audit."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.ja.model import (  # noqa: E402
    SPEECH_ISLAND_SCORER_V10_TRAINING_ROW_SCHEMA,
    load_speech_island_scorer_checkpoint,
    score_binary_speech_class_probabilities,
)
from pipeline.memory_safety import (  # noqa: E402
    reset_shared_vram_baseline,
    runtime_memory_snapshot,
)


IGNORE_INDEX = -100
FRAME_HOP_S = 0.02
TRAINING_ROW_SCHEMA = SPEECH_ISLAND_SCORER_V10_TRAINING_ROW_SCHEMA
DIAGNOSTIC_ROW_SCHEMA = "speech_scorer_v10_binary_diagnostic_row_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_score_rows(rows: Iterable[dict[str, Any]]) -> str:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Scorer checkpoint audit dataset is empty")
    schemas = {str(row.get("schema") or "") for row in materialized}
    if schemas == {TRAINING_ROW_SCHEMA}:
        if any(row.get("diagnostic_only") for row in materialized):
            raise ValueError("Scorer training rows cannot be diagnostic-only")
        return "training_manifest"
    if schemas == {DIAGNOSTIC_ROW_SCHEMA}:
        if any(
            row.get("diagnostic_only") is not True
            or row.get("training_manifest_allowed") is not False
            for row in materialized
        ):
            raise ValueError("Scorer diagnostic rows require a strict read-only contract")
        return "diagnostic_rescore_manifest"
    raise ValueError("Scorer checkpoint audit rejects mixed or unknown row schemas")


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.r_[values.astype(bool), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index))
            start = None
    return result


def _span_rows(values: np.ndarray, *, label: str) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "start_frame": int(start),
            "end_frame": int(end),
            "start_s": float(start * FRAME_HOP_S),
            "end_s": float(end * FRAME_HOP_S),
        }
        for start, end in _runs(values)
    ]


def _row_continuity_from_spans(row: dict[str, Any]) -> dict[str, Any]:
    truth_spans = sorted(
        (
            span
            for span in row.get("truth_spans", [])
            if span.get("label") == "truth_speech"
        ),
        key=lambda span: (int(span["start_frame"]), int(span["end_frame"])),
    )
    prediction_spans = sorted(
        row.get("prediction_spans", []),
        key=lambda span: (int(span["start_frame"]), int(span["end_frame"])),
    )
    predictions_overlapping_truth = [
        span
        for span in prediction_spans
        if any(
            int(span["end_frame"]) > int(truth["start_frame"])
            and int(span["start_frame"]) < int(truth["end_frame"])
            for truth in truth_spans
        )
    ]
    predicted_duration_frames = [
        int(span["end_frame"]) - int(span["start_frame"])
        for span in predictions_overlapping_truth
    ]
    continuous = fragmented = predicted_runs = 0
    internal_gaps: list[int] = []
    for truth in truth_spans:
        truth_start = int(truth["start_frame"])
        truth_end = int(truth["end_frame"])
        overlaps = [
            span
            for span in prediction_spans
            if int(span["end_frame"]) > truth_start
            and int(span["start_frame"]) < truth_end
        ]
        predicted_runs += len(overlaps)
        continuous += int(len(overlaps) == 1)
        fragmented += int(len(overlaps) > 1)
        for left, right in zip(overlaps, overlaps[1:]):
            internal_gaps.append(
                max(0, int(right["start_frame"]) - int(left["end_frame"]))
            )
    truth_run_count = len(truth_spans)
    return {
        "truth_run_count": truth_run_count,
        "continuous_truth_run_count": continuous,
        "fragmented_truth_run_count": fragmented,
        "predicted_run_count_within_truth": predicted_runs,
        "speech_run_continuity": continuous / max(truth_run_count, 1),
        "prediction_to_truth_run_ratio": predicted_runs / max(truth_run_count, 1),
        "internal_drop_gap_count": len(internal_gaps),
        "internal_drop_frame_count": sum(internal_gaps),
        "max_internal_drop_gap_frames": max(internal_gaps, default=0),
        "predicted_run_under_100ms_count": sum(
            duration < 5 for duration in predicted_duration_frames
        ),
        "predicted_run_under_200ms_count": sum(
            duration < 10 for duration in predicted_duration_frames
        ),
        "predicted_run_under_500ms_count": sum(
            duration < 25 for duration in predicted_duration_frames
        ),
    }


def summarize_prediction_continuity(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    result: dict[str, Any] = {}
    for partition in ("train", "val", "test", "all"):
        selected = [
            row
            for row in materialized
            if row.get("row_role") == "speech"
            and (partition == "all" or row.get("partition") == partition)
        ]
        metrics = [_row_continuity_from_spans(row) for row in selected]
        truth_runs = sum(int(item["truth_run_count"]) for item in metrics)
        continuous = sum(int(item["continuous_truth_run_count"]) for item in metrics)
        fragmented = sum(int(item["fragmented_truth_run_count"]) for item in metrics)
        predicted_runs = sum(
            int(item["predicted_run_count_within_truth"]) for item in metrics
        )
        result[partition] = {
            "speech_row_count": len(selected),
            "truth_run_count": truth_runs,
            "continuous_truth_run_count": continuous,
            "fragmented_truth_run_count": fragmented,
            "speech_run_continuity": continuous / max(truth_runs, 1),
            "predicted_run_count_within_truth": predicted_runs,
            "prediction_to_truth_run_ratio": predicted_runs / max(truth_runs, 1),
            "internal_drop_gap_count": sum(
                int(item["internal_drop_gap_count"]) for item in metrics
            ),
            "internal_drop_frame_count": sum(
                int(item["internal_drop_frame_count"]) for item in metrics
            ),
            "max_internal_drop_gap_frames": max(
                (int(item["max_internal_drop_gap_frames"]) for item in metrics),
                default=0,
            ),
            "predicted_run_under_100ms_count": sum(
                int(item["predicted_run_under_100ms_count"]) for item in metrics
            ),
            "predicted_run_under_200ms_count": sum(
                int(item["predicted_run_under_200ms_count"]) for item in metrics
            ),
            "predicted_run_under_500ms_count": sum(
                int(item["predicted_run_under_500ms_count"]) for item in metrics
            ),
        }
    result["gate_threshold"] = 0.95
    result["heldout_continuity_gate_pass"] = min(
        float(result["val"]["speech_run_continuity"]),
        float(result["test"]["speech_run_continuity"]),
    ) >= 0.95
    result["short_run_counts_are_diagnostic_only"] = True
    return result


def _memory_check(*, stage: str, device: str) -> dict[str, Any]:
    import torch

    snapshot = runtime_memory_snapshot(require_shared_vram=device.startswith("cuda"))
    if snapshot["physical_ram_used_mb"] > snapshot["physical_ram_budget_mb"]:
        raise MemoryError("Scorer v10 audit exceeded the 0.95 physical RAM budget")
    if device.startswith("cuda") and float(snapshot.get("shared_vram_mb") or 0.0) > 0.0:
        raise MemoryError(
            "Scorer v10 audit shared VRAM spill is a soft OOM: "
            f"shared_vram_mb={snapshot.get('shared_vram_mb')} "
            f"raw_mb={snapshot.get('shared_vram_raw_mb')} "
            f"baseline_mb={snapshot.get('shared_vram_baseline_mb')}"
        )
    if device.startswith("cuda"):
        snapshot.update(
            cuda_allocated_mb=round(torch.cuda.memory_allocated() / 2**20, 3),
            cuda_reserved_mb=round(torch.cuda.memory_reserved() / 2**20, 3),
            cuda_max_allocated_mb=round(torch.cuda.max_memory_allocated() / 2**20, 3),
            cuda_max_reserved_mb=round(torch.cuda.max_memory_reserved() / 2**20, 3),
        )
    snapshot["stage"] = stage
    return snapshot


def _load_row_features(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(Path(str(row["feature_path"]))) as features:
        ptm = np.asarray(features["ptm"], dtype=np.float32)
        mfcc = np.asarray(features["mfcc"], dtype=np.float32)
    with np.load(Path(str(row["label_path"]))) as labels:
        canonical = np.asarray(labels["canonical_labels"], dtype=np.int64)
    if ptm.shape[0] != mfcc.shape[0] or ptm.shape[0] != canonical.shape[0]:
        raise ValueError(f"feature/label frame mismatch: {row['source_id']}")
    truth = np.where(canonical == 2, IGNORE_INDEX, canonical).astype(np.int64)
    return ptm, mfcc, truth


def _score_row(bundle: Any, row: dict[str, Any]) -> dict[str, Any]:
    ptm, mfcc, truth = _load_row_features(row)
    probabilities = score_binary_speech_class_probabilities(bundle, ptm=ptm, mfcc=mfcc)
    predicted = np.argmax(probabilities, axis=1).astype(np.int64)
    valid = truth != IGNORE_INDEX
    truth_speech = truth == 1
    predicted_speech = predicted == 1
    speech_runs = _runs(truth_speech & valid)
    edge_errors: list[dict[str, Any]] = []
    true_speech_deletions = 0
    for start, end in speech_runs:
        present = np.flatnonzero(predicted_speech[start:end] & valid[start:end])
        if not present.size:
            true_speech_deletions += 1
            continue
        predicted_start = start + int(present[0])
        predicted_end = start + int(present[-1])
        edge_errors.append(
            {
                "truth_start_s": start * FRAME_HOP_S,
                "truth_end_s": end * FRAME_HOP_S,
                "predicted_start_s": predicted_start * FRAME_HOP_S,
                "predicted_end_s": (predicted_end + 1) * FRAME_HOP_S,
                "start_error_frames": abs(predicted_start - start),
                "end_error_frames": abs(predicted_end - (end - 1)),
            }
        )
    false_negative_frames = int(np.sum((truth == 1) & (predicted == 0) & valid))
    false_positive_frames = int(np.sum((truth == 0) & (predicted == 1) & valid))
    prediction_spans = _span_rows(predicted_speech & valid, label="model_speech")
    truth_spans = [
        *_span_rows((truth == 1) & valid, label="truth_speech"),
        *_span_rows((truth == 0) & valid, label="truth_background"),
    ]
    max_predicted_speech_run_s = max(
        (span["end_s"] - span["start_s"] for span in prediction_spans),
        default=0.0,
    )
    if row["row_role"] == "all_background" and false_positive_frames:
        category = "background_false_keep"
    elif true_speech_deletions:
        category = "speech_deletion"
    elif any(
        max(item["start_error_frames"], item["end_error_frames"]) > 15
        for item in edge_errors
    ) or false_negative_frames:
        category = "speech_edge_or_partial"
    elif max_predicted_speech_run_s > 8.0:
        category = "long_residual"
    else:
        category = "normal"
    result = {
        "source_id": row["source_id"],
        "audio": row["audio"],
        "partition": row["partition"],
        "row_role": row["row_role"],
        "duration_s": float(row.get("duration_s") or int(row["frame_count"]) * FRAME_HOP_S),
        "frame_count": int(row["frame_count"]),
        "truth_speech_frames": int(np.sum((truth == 1) & valid)),
        "predicted_speech_frames": int(np.sum(predicted_speech & valid)),
        "false_negative_frames": false_negative_frames,
        "false_positive_frames": false_positive_frames,
        "true_speech_deletions": true_speech_deletions,
        "max_predicted_speech_run_s": max_predicted_speech_run_s,
        "edge_errors": edge_errors,
        "category": category,
        "truth_spans": truth_spans,
        "prediction_spans": prediction_spans,
    }
    result.update(_row_continuity_from_spans(result))
    return result


def _select_audit_rows(rows: Iterable[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["true_speech_deletions"] > 0
        or row["false_negative_frames"] > 0
        or float(row["max_predicted_speech_run_s"]) > 8.0
        or (
            row["partition"] in {"val", "test"}
            and row["category"] != "normal"
        )
    ]
    category_priority = {
        "speech_deletion": 0,
        "speech_edge_or_partial": 1,
        "long_residual": 2,
        "background_false_keep": 3,
        "normal": 4,
    }
    partition_priority = {"val": 0, "test": 1, "train": 2}

    def audit_category_priority(row: dict[str, Any]) -> int:
        category = str(row["category"])
        if category == "normal" and float(row["max_predicted_speech_run_s"]) > 8.0:
            category = "long_residual"
        return category_priority.get(category, 5)

    candidates.sort(
        key=lambda row: (
            audit_category_priority(row),
            partition_priority.get(str(row.get("partition") or ""), 3),
            -int(row["false_negative_frames"]),
            -int(row["false_positive_frames"]),
            -float(row.get("duration_s") or 0.0),
            str(row["source_id"]),
        )
    )
    return candidates[:max_items] if max_items > 0 else candidates


def score_checkpoint(
    *, checkpoint: Path, dataset_manifest: Path, output_dir: Path, device: str, max_audit_items: int
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(dataset_manifest)
    dataset_mode = _validate_score_rows(rows)
    if device.startswith("cuda"):
        torch.cuda.init()
        warmup = torch.ones(1, device=device)
        warmup.add_(1.0)
        torch.cuda.synchronize()
        del warmup
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
    memory_snapshots = [_memory_check(stage="context_baseline", device=device)]
    bundle = load_speech_island_scorer_checkpoint(checkpoint, device=device)
    first_ptm, first_mfcc, _truth = _load_row_features(rows[0])
    _score_row(bundle, {**rows[0], "feature_path": rows[0]["feature_path"], "label_path": rows[0]["label_path"]})
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        reset = reset_shared_vram_baseline(required=True)
        reset["stage"] = "execution_baseline"
        memory_snapshots.append(reset)
        torch.cuda.reset_peak_memory_stats()
    del first_ptm, first_mfcc, _truth
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        predictions.append(_score_row(bundle, row))
        if index % 100 == 0 or index == len(rows):
            memory_snapshots.append(_memory_check(stage=f"row_{index}", device=device))
            print(f"scored {index}/{len(rows)}", flush=True)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    selected = _select_audit_rows(predictions, max_items=max_audit_items)
    required_selection = _select_audit_rows(predictions, max_items=0)
    selected_path = output_dir / "audit_selection.jsonl"
    with selected_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema": "speech_scorer_v10_checkpoint_audit_summary_v2",
        "checkpoint": str(checkpoint),
        "dataset_manifest": str(dataset_manifest),
        "dataset_mode": dataset_mode,
        "training_manifest_allowed": dataset_mode == "training_manifest",
        "row_count": len(predictions),
        "selected_count": len(selected),
        "required_selection_count": len(required_selection),
        "audit_selection_complete": len(selected) == len(required_selection),
        "audit_selection_contract": (
            "all_truth_keep_model_drop_rows_plus_all_heldout_hard_cases_"
            "plus_all_over_8s_residuals"
        ),
        "category_counts": {
            category: sum(row["category"] == category for row in predictions)
            for category in sorted({row["category"] for row in predictions})
        },
        "prediction_drop_truth_keep_rows": sum(
            row["false_negative_frames"] > 0 for row in predictions if row["row_role"] == "speech"
        ),
        "true_speech_deletion_rows": sum(row["true_speech_deletions"] > 0 for row in predictions),
        "over_8s_residual_rows": sum(
            float(row["max_predicted_speech_run_s"]) > 8.0
            for row in predictions
        ),
        "selected_over_8s_residual_rows": sum(
            float(row["max_predicted_speech_run_s"]) > 8.0
            for row in selected
        ),
        "selected_prediction_drop_truth_keep_rows": sum(
            row["row_role"] == "speech" and row["false_negative_frames"] > 0
            for row in selected
        ),
        "continuity": summarize_prediction_continuity(predictions),
        "memory_snapshots": memory_snapshots,
        "predictions": str(predictions_path),
        "audit_selection": str(selected_path),
        "manual_gate": "pending",
    }
    if device.startswith("cuda"):
        del bundle
        gc.collect()
        torch.cuda.empty_cache()
        summary["memory_after_release"] = _memory_check(
            stage="post_release", device=device
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-audit-items", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    score_checkpoint(
        checkpoint=Path(args.checkpoint),
        dataset_manifest=Path(args.dataset_manifest),
        output_dir=Path(args.output_dir),
        device=args.device,
        max_audit_items=args.max_audit_items,
    )
