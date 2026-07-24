#!/usr/bin/env python3
"""Score a Scorer v11 checkpoint on complete canonical sources."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_CRF_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_QUERY_MASK_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA,
    load_speech_island_scorer_checkpoint,
    score_candidate_island_source_outputs,
)
from pipeline.memory_safety import (  # noqa: E402
    reset_shared_vram_baseline,
    runtime_memory_snapshot,
)


FRAME_HOP_S = 0.02
IGNORE_INDEX = -100
SUMMARY_SCHEMA = "candidate_island_scorer_v11_checkpoint_audit_summary_v1"
PREDICTION_SCHEMA = "candidate_island_scorer_v11_source_prediction_v1"
LABEL_ID = {
    "outside_candidate": 0,
    "inside_candidate": 1,
    "unsure": IGNORE_INDEX,
}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=np.bool_)
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _spans(mask: np.ndarray, *, label: str) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "start_frame": start,
            "end_frame": end,
            "start_s": round(start * FRAME_HOP_S, 6),
            "end_s": round(end * FRAME_HOP_S, 6),
        }
        for start, end in _runs(mask)
    ]


def _canonical_labels(row: dict[str, Any]) -> np.ndarray:
    frame_count = int(row.get("frame_count") or 0)
    if frame_count <= 0:
        raise ValueError("Scorer v11 canonical source requires frame_count")
    labels = np.full(frame_count, 999, dtype=np.int64)
    cursor = 0
    for span in row.get("canonical_spans") or ():
        start = int(span.get("start_frame", -1))
        end = int(span.get("end_frame", -1))
        label = str(span.get("label") or "")
        if start != cursor or end <= start or end > frame_count or label not in LABEL_ID:
            raise ValueError(f"invalid Scorer v11 canonical span: {row.get('source_id')}")
        labels[start:end] = LABEL_ID[label]
        cursor = end
    if cursor != frame_count or np.any(labels == 999):
        raise ValueError(f"Scorer v11 canonical coverage is incomplete: {row.get('source_id')}")
    return labels


def evaluate_source_prediction(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    tolerance_frames: int,
    long_residual_frames: int,
) -> dict[str, Any]:
    truth_values = np.asarray(truth, dtype=np.int64)
    predicted_values = np.asarray(predicted, dtype=np.int64)
    if truth_values.shape != predicted_values.shape or truth_values.ndim != 1:
        raise ValueError("Scorer v11 truth/prediction shape mismatch")
    if np.any(~np.isin(predicted_values, (0, 1))):
        raise ValueError("Scorer v11 prediction must contain binary argmax labels")
    valid = truth_values != IGNORE_INDEX
    inside_truth = truth_values == 1
    outside_truth = truth_values == 0
    inside_pred = predicted_values == 1
    drop_truth_keep = inside_truth & ~inside_pred
    keep_truth_drop = outside_truth & inside_pred
    confusion = np.zeros((2, 2), dtype=np.int64)
    for expected, actual in zip(
        truth_values[valid].tolist(), predicted_values[valid].tolist(), strict=True
    ):
        confusion[int(expected), int(actual)] += 1

    truth_runs = _runs(inside_truth)
    start_hits = end_hits = deletions = continuous = fragmented = 0
    internal_gap_frames = 0
    internal_gap_lengths: list[int] = []
    for start, end in truth_runs:
        local_runs = _runs(inside_pred[start:end])
        if not local_runs:
            deletions += 1
            continue
        continuous += int(len(local_runs) == 1)
        fragmented += int(len(local_runs) > 1)
        first_start = start + local_runs[0][0]
        last_end = start + local_runs[-1][1]
        start_hits += int(first_start - start <= tolerance_frames)
        end_hits += int(end - last_end <= tolerance_frames)
        local_gap_lengths = [
            max(0, right[0] - left[1])
            for left, right in zip(local_runs, local_runs[1:])
        ]
        internal_gap_lengths.extend(local_gap_lengths)
        internal_gap_frames += sum(local_gap_lengths)

    long_residuals = [
        (start, end)
        for start, end in _runs(keep_truth_drop)
        if end - start >= int(long_residual_frames)
    ]
    inside_total = int(confusion[1].sum())
    outside_total = int(confusion[0].sum())
    definite_total = int(confusion.sum())
    return {
        "confusion_truth_by_prediction": confusion.tolist(),
        "definite_frame_count": definite_total,
        "inside_candidate_recall": float(confusion[1, 1] / max(inside_total, 1)),
        "outside_candidate_recall": float(confusion[0, 0] / max(outside_total, 1)),
        "frame_accuracy": float(np.trace(confusion) / max(definite_total, 1)),
        "truth_inside_run_count": len(truth_runs),
        "start_hit_count": start_hits,
        "end_hit_count": end_hits,
        "start_coverage": start_hits / max(len(truth_runs), 1),
        "end_coverage": end_hits / max(len(truth_runs), 1),
        "true_inside_deletion_count": deletions,
        "continuous_truth_run_count": continuous,
        "fragmented_truth_run_count": fragmented,
        "truth_run_continuity": continuous / max(len(truth_runs), 1),
        "prediction_inside_run_count": len(_runs(inside_pred)),
        "internal_drop_gap_count": len(internal_gap_lengths),
        "internal_drop_gap_1_frame_count": internal_gap_lengths.count(1),
        "internal_drop_gap_2_frame_count": internal_gap_lengths.count(2),
        "internal_drop_gap_3_frame_count": internal_gap_lengths.count(3),
        "internal_drop_gap_4plus_frame_count": sum(
            length >= 4 for length in internal_gap_lengths
        ),
        "internal_drop_gap_frame_count": internal_gap_frames,
        "prediction_drop_truth_keep_frame_count": int(drop_truth_keep.sum()),
        "prediction_keep_truth_drop_frame_count": int(keep_truth_drop.sum()),
        "long_residual_count": len(long_residuals),
        "truth_spans": (
            _spans(inside_truth, label="truth_inside_candidate")
            + _spans(outside_truth, label="truth_outside_candidate")
            + _spans(truth_values == IGNORE_INDEX, label="truth_unsure")
        ),
        "prediction_spans": _spans(inside_pred, label="model_inside_candidate"),
        "prediction_drop_truth_keep_spans": _spans(
            drop_truth_keep, label="truth_inside_model_outside"
        ),
        "prediction_keep_truth_drop_spans": _spans(
            keep_truth_drop, label="truth_outside_model_inside"
        ),
        "long_residual_spans": [
            {
                "label": "long_truth_outside_model_inside",
                "start_frame": start,
                "end_frame": end,
                "start_s": round(start * FRAME_HOP_S, 6),
                "end_s": round(end * FRAME_HOP_S, 6),
            }
            for start, end in long_residuals
        ],
    }


def _sum_partition(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    confusion = np.zeros((2, 2), dtype=np.int64)
    summed = Counter()
    for row in selected:
        confusion += np.asarray(row["confusion_truth_by_prediction"], dtype=np.int64)
        for key in (
            "truth_inside_run_count",
            "start_hit_count",
            "end_hit_count",
            "true_inside_deletion_count",
            "continuous_truth_run_count",
            "fragmented_truth_run_count",
            "prediction_inside_run_count",
            "internal_drop_gap_count",
            "internal_drop_gap_1_frame_count",
            "internal_drop_gap_2_frame_count",
            "internal_drop_gap_3_frame_count",
            "internal_drop_gap_4plus_frame_count",
            "internal_drop_gap_frame_count",
            "prediction_drop_truth_keep_frame_count",
            "prediction_keep_truth_drop_frame_count",
            "long_residual_count",
        ):
            summed[key] += int(row[key])
    inside_total = int(confusion[1].sum())
    outside_total = int(confusion[0].sum())
    definite_total = int(confusion.sum())
    runs = int(summed["truth_inside_run_count"])
    start_hits = int(summed["start_hit_count"])
    end_hits = int(summed["end_hit_count"])
    return {
        "source_count": len(selected),
        "confusion_truth_by_prediction": confusion.tolist(),
        "definite_frame_count": definite_total,
        "inside_candidate_recall": float(confusion[1, 1] / max(inside_total, 1)),
        "outside_candidate_recall": float(confusion[0, 0] / max(outside_total, 1)),
        "frame_accuracy": float(np.trace(confusion) / max(definite_total, 1)),
        "truth_inside_run_count": runs,
        "start_coverage": start_hits / max(runs, 1),
        "end_coverage": end_hits / max(runs, 1),
        "truth_run_continuity": int(summed["continuous_truth_run_count"]) / max(runs, 1),
        **{key: int(value) for key, value in summed.items()},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    canonical_path = Path(args.canonical_sources).resolve()
    raw_manifest_path = Path(args.raw_feature_manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    canonical_rows = _rows(canonical_path)
    raw_rows = _rows(raw_manifest_path)
    if any(row.get("schema") != CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA for row in canonical_rows):
        raise ValueError("Scorer v11 audit requires current canonical rows")
    if any(row.get("schema") != CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA for row in raw_rows):
        raise ValueError("Scorer v11 audit requires current raw feature rows")
    contract = ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    if any(row.get("boundary_serialization_contract_id") != contract for row in canonical_rows + raw_rows):
        raise ValueError("Scorer v11 audit requires the central boundary contract")
    canonical_sha = _sha256(canonical_path)
    raw_by_id = {str(row.get("source_id") or ""): row for row in raw_rows}
    if len(raw_by_id) != len(raw_rows) or "" in raw_by_id:
        raise ValueError("Scorer v11 raw manifest source ids must be unique")

    requested_partitions = set(args.partition or ("val", "test"))
    if not requested_partitions <= {"train", "val", "test"}:
        raise ValueError("unsupported Scorer v11 audit partition")
    selected = [row for row in canonical_rows if row.get("partition") in requested_partitions]
    if not selected:
        raise ValueError("Scorer v11 audit selection is empty")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Scorer v11 checkpoint audit requires CUDA; CPU fallback is forbidden")
    if device.index is None:
        device = torch.device("cuda", int(torch.cuda.current_device()))
    torch.cuda.set_per_process_memory_fraction(0.95, device=device)
    bundle = load_speech_island_scorer_checkpoint(checkpoint_path, device=device)
    decoder_diagnostics: dict[str, Any] = {
        "schema": bundle.schema,
        "decision_mode": str(bundle.metadata.get("decision_mode") or ""),
    }
    if bundle.schema == CANDIDATE_ISLAND_SCORER_V11_CRF_SCHEMA:
        decoder_diagnostics["learned_transition_matrix"] = (
            bundle.model.crf.transitions.detach().cpu().tolist()
        )
    if bundle.schema == CANDIDATE_ISLAND_SCORER_V11_QUERY_MASK_SCHEMA:
        decoder_diagnostics.update(
            query_count=int(bundle.model.query_count),
            learned_residual_gate=float(
                torch.tanh(bundle.model.query_residual_gate).detach().cpu()
            ),
            query_mask_aggregation=str(
                bundle.metadata.get("query_mask_aggregation") or ""
            ),
            query_residual_fusion=str(
                bundle.metadata.get("query_residual_fusion") or ""
            ),
        )
    first = raw_by_id[str(selected[0]["source_id"])]
    with np.load(_resolve(first["feature_path"]), allow_pickle=False) as payload:
        warm_ptm = np.asarray(payload["ptm"], dtype=np.float32)
        warm_mfcc = np.asarray(payload["mfcc"], dtype=np.float32)
    score_candidate_island_source_outputs(
        bundle,
        ptm=warm_ptm,
        mfcc=warm_mfcc,
        max_padded_frames=int(args.max_padded_frames),
    )
    del warm_ptm, warm_mfcc
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    baseline = reset_shared_vram_baseline(required=True)
    torch.cuda.reset_peak_memory_stats(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, Any]] = []
    for index, canonical in enumerate(selected, start=1):
        source_id = str(canonical["source_id"])
        raw = raw_by_id.get(source_id)
        if raw is None or raw.get("canonical_sources_sha256") != canonical_sha:
            raise ValueError(f"Scorer v11 raw feature provenance mismatch: {source_id}")
        feature_path = _resolve(raw["feature_path"])
        if _sha256(feature_path) != raw.get("feature_sha256"):
            raise ValueError(f"Scorer v11 raw feature SHA mismatch: {source_id}")
        with np.load(feature_path, allow_pickle=False) as payload:
            ptm = np.asarray(payload["ptm"], dtype=np.float32)
            mfcc = np.asarray(payload["mfcc"], dtype=np.float32)
        if ptm.shape[0] != int(canonical["frame_count"]) or mfcc.shape[0] != ptm.shape[0]:
            raise ValueError(f"Scorer v11 source feature geometry mismatch: {source_id}")
        outputs = score_candidate_island_source_outputs(
            bundle,
            ptm=ptm,
            mfcc=mfcc,
            max_padded_frames=int(args.max_padded_frames),
        )
        probabilities = outputs.probabilities
        predicted = outputs.labels
        truth = _canonical_labels(canonical)
        metrics = evaluate_source_prediction(
            truth,
            predicted,
            tolerance_frames=int(args.tolerance_frames),
            long_residual_frames=int(args.long_residual_frames),
        )
        prediction_rows.append(
            {
                "schema": PREDICTION_SCHEMA,
                "boundary_serialization_contract_id": contract,
                "checkpoint": _display(checkpoint_path),
                "checkpoint_sha256": bundle.sha256,
                "source_id": source_id,
                "partition": str(canonical["partition"]),
                "source_kind": str(canonical.get("source_kind") or ""),
                "frame_count": int(canonical["frame_count"]),
                "frame_hop_s": FRAME_HOP_S,
                "duration_s": float(canonical["duration_s"]),
                "audio": str(canonical["audio"]),
                "audio_sha256": str(canonical["audio_sha256"]),
                **metrics,
            }
        )
        del ptm, mfcc, outputs, probabilities, predicted, truth
        if index % 5 == 0 or index == len(selected):
            snapshot = runtime_memory_snapshot(require_shared_vram=True)
            if float(snapshot.get("shared_vram_mb") or 0.0) > 0.0:
                raise MemoryError(
                    "Scorer v11 audit shared VRAM spill is a soft OOM: "
                    f"{snapshot.get('shared_vram_mb')} MiB"
                )
            print(f"scorer_v11_audit={index}/{len(selected)}", flush=True)

    predictions_path = output_dir / "source_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    partition_metrics = {
        partition: _sum_partition(
            row for row in prediction_rows if row["partition"] == partition
        )
        for partition in sorted(requested_partitions)
    }
    numeric_gate_pass = all(
        metrics["inside_candidate_recall"] >= 0.95
        and metrics["outside_candidate_recall"] >= 0.95
        and metrics["start_coverage"] >= 0.95
        and metrics["end_coverage"] >= 0.95
        and metrics["truth_run_continuity"] >= 0.95
        and metrics["true_inside_deletion_count"] == 0
        for metrics in partition_metrics.values()
    )
    before_cleanup = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    del bundle
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    after_cleanup = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": contract,
        "checkpoint": _display(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "canonical_sources": _display(canonical_path),
        "canonical_sources_sha256": canonical_sha,
        "raw_feature_manifest": _display(raw_manifest_path),
        "raw_feature_manifest_sha256": _sha256(raw_manifest_path),
        "partitions": sorted(requested_partitions),
        "decoder_diagnostics": decoder_diagnostics,
        "source_count": len(prediction_rows),
        "partition_metrics": partition_metrics,
        "numeric_gate_maximum_requirement": 0.95,
        "numeric_gate_pass": numeric_gate_pass,
        "manual_zero_clipping_gate": "pending",
        "manual_zero_true_speech_deletion_gate": "pending",
        "promotion_allowed": False,
        "predictions": _display(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "shared_vram_baseline": baseline,
        "stage_lifecycle": {
            "before_cleanup": before_cleanup,
            "after_cleanup": after_cleanup,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--canonical-sources", required=True)
    parser.add_argument("--raw-feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--partition", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-padded-frames", type=int, default=2000)
    parser.add_argument("--tolerance-frames", type=int, default=15)
    parser.add_argument("--long-residual-frames", type=int, default=400)
    args = parser.parse_args(argv)
    if args.max_padded_frames <= 0:
        parser.error("--max-padded-frames must be positive")
    if args.tolerance_frames < 0 or args.long_residual_frames <= 0:
        parser.error("audit frame thresholds are invalid")
    return args


if __name__ == "__main__":
    run(parse_args())
