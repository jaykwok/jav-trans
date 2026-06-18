#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.ja import (  # noqa: E402
    endpoint_targets_from_record,
    effective_frame_weights,
    load_cached_feature,
    load_feature_frame_scorer_checkpoint,
    load_label_records,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"feature manifest row must be an object: {path}:{line_number}")
            rows.append(dict(row))
    return rows


def _repo_display(path: str | Path) -> str:
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(PROJECT_ROOT.resolve())).replace("/", "\\")
    except ValueError:
        return str(raw)


def _default_thresholds() -> list[float]:
    values = [
        0.02,
        0.05,
        0.08,
        0.10,
        0.12,
        0.15,
        0.18,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]
    return values


def _empty_counts() -> dict[str, int]:
    return {
        "frames": 0,
        "positives": 0,
        "negatives": 0,
        "predicted_positives": 0,
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "correct": 0,
    }


def _speech_boundary_distances_s(labels: np.ndarray, *, frame_hop_s: float) -> np.ndarray:
    values = (np.asarray(labels, dtype=np.float32).reshape(-1) > 0.5).astype(np.int32)
    if values.size <= 0:
        return np.zeros(0, dtype=np.float32)
    transition_indices = np.flatnonzero(values[1:] != values[:-1]) + 1
    if transition_indices.size <= 0:
        return np.full(values.size, np.inf, dtype=np.float32)
    frame_centers = (np.arange(values.size, dtype=np.float32) + 0.5) * float(frame_hop_s)
    boundary_times = transition_indices.astype(np.float32) * float(frame_hop_s)
    distances = np.min(np.abs(frame_centers[:, None] - boundary_times[None, :]), axis=1)
    return np.asarray(distances, dtype=np.float32)


def _distance_bucket_key(distance_s: float, buckets_s: tuple[float, ...]) -> str:
    if not np.isfinite(float(distance_s)):
        return "no_boundary"
    for bound in buckets_s:
        if float(distance_s) <= float(bound) + 1e-9:
            return f"le_{float(bound):.3f}s"
    return f"gt_{float(buckets_s[-1]):.3f}s" if buckets_s else "all_boundary_distance"


def _binary_runs(values: np.ndarray) -> list[tuple[int, int]]:
    flags = np.asarray(values, dtype=np.float32).reshape(-1) > 0.5
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags.tolist() + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    return runs


def _overlap_frames(left: tuple[int, int], right: tuple[int, int]) -> int:
    start = max(int(left[0]), int(right[0]))
    end = min(int(left[1]), int(right[1]))
    return max(0, end - start)


def _empty_island_counts() -> dict[str, float | int]:
    return {
        "label_islands": 0,
        "detected_label_islands": 0,
        "missed_label_islands": 0,
        "label_island_coverage_sum": 0.0,
        "predicted_islands": 0,
        "matched_predicted_islands": 0,
        "false_predicted_islands": 0,
        "predicted_island_overlap_sum": 0.0,
        "cut_drop_zones": 0,
        "clean_cut_drop_zones": 0,
        "dirty_cut_drop_zones": 0,
        "cut_drop_zone_predicted_ratio_sum": 0.0,
    }


def _empty_diagnostic_state(
    *,
    kind: str,
    label: str,
    buckets_s: tuple[float, ...],
    near_boundary_s: float,
) -> dict[str, Any]:
    bucket_keys = [_distance_bucket_key(float(bound), buckets_s) for bound in buckets_s]
    if buckets_s:
        bucket_keys.append(f"gt_{float(buckets_s[-1]):.3f}s")
    bucket_keys.append("no_boundary")
    return {
        "kind": kind,
        "label": label,
        "near_boundary_s": float(near_boundary_s),
        "boundary_buckets_s": [float(value) for value in buckets_s],
        "overall": _empty_counts(),
        "distance_buckets": {key: _empty_counts() for key in dict.fromkeys(bucket_keys)},
        "regions": {
            "near_boundary": _empty_counts(),
            "far_from_boundary": _empty_counts(),
            "no_boundary": _empty_counts(),
            "cut_target": _empty_counts(),
            "non_cut_background": _empty_counts(),
            "speech_far_from_boundary": _empty_counts(),
            "background_far_from_boundary": _empty_counts(),
        },
        "islands": _empty_island_counts(),
        "top_error_rows": [],
    }


def _update_counts(
    counts: dict[str, int],
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> None:
    label_values = (labels > 0.5).astype(np.int32)
    pred_values = (predictions > 0.5).astype(np.int32)
    tp = int(np.logical_and(label_values == 1, pred_values == 1).sum())
    tn = int(np.logical_and(label_values == 0, pred_values == 0).sum())
    fp = int(np.logical_and(label_values == 0, pred_values == 1).sum())
    fn = int(np.logical_and(label_values == 1, pred_values == 0).sum())
    counts["frames"] += int(label_values.size)
    counts["positives"] += int(label_values.sum())
    counts["negatives"] += int((label_values == 0).sum())
    counts["predicted_positives"] += int(pred_values.sum())
    counts["true_positive"] += tp
    counts["true_negative"] += tn
    counts["false_positive"] += fp
    counts["false_negative"] += fn
    counts["correct"] += tp + tn


def _update_diagnostic_state(
    state: dict[str, Any],
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    boundary_distances_s: np.ndarray,
    cut_labels: np.ndarray,
    cut_drop_labels: np.ndarray,
    item: Mapping[str, Any],
    buckets_s: tuple[float, ...],
    near_boundary_s: float,
) -> None:
    label_values = (np.asarray(labels, dtype=np.float32).reshape(-1) > 0.5).astype(np.float32)
    pred_values = (np.asarray(predictions, dtype=np.float32).reshape(-1) > 0.5).astype(np.float32)
    distances = np.asarray(boundary_distances_s, dtype=np.float32).reshape(-1)
    cut_values = (np.asarray(cut_labels, dtype=np.float32).reshape(-1) > 0.5).astype(np.float32)
    cut_drop_values = (np.asarray(cut_drop_labels, dtype=np.float32).reshape(-1) > 0.5).astype(np.float32)
    frame_total = min(label_values.size, pred_values.size, distances.size, cut_values.size, cut_drop_values.size)
    if frame_total <= 0:
        return
    label_values = label_values[:frame_total]
    pred_values = pred_values[:frame_total]
    distances = distances[:frame_total]
    cut_values = cut_values[:frame_total]
    cut_drop_values = cut_drop_values[:frame_total]
    _update_counts(state["overall"], labels=label_values, predictions=pred_values)

    bucket_keys = np.asarray([_distance_bucket_key(float(value), buckets_s) for value in distances], dtype=object)
    for bucket_key in sorted(set(str(value) for value in bucket_keys.tolist())):
        mask = bucket_keys == bucket_key
        if bool(mask.any()):
            state["distance_buckets"].setdefault(str(bucket_key), _empty_counts())
            _update_counts(
                state["distance_buckets"][str(bucket_key)],
                labels=label_values[mask],
                predictions=pred_values[mask],
            )

    finite = np.isfinite(distances)
    near_mask = finite & (distances <= float(near_boundary_s))
    region_masks = {
        "near_boundary": near_mask,
        "far_from_boundary": finite & np.logical_not(near_mask),
        "no_boundary": np.logical_not(finite),
        "cut_target": cut_values > 0.5,
        "non_cut_background": (cut_values <= 0.5) & (label_values <= 0.5),
        "speech_far_from_boundary": (label_values > 0.5) & finite & np.logical_not(near_mask),
        "background_far_from_boundary": (label_values <= 0.5)
        & (cut_values <= 0.5)
        & finite
        & np.logical_not(near_mask),
    }
    for region, mask in region_masks.items():
        if bool(mask.any()):
            _update_counts(state["regions"][region], labels=label_values[mask], predictions=pred_values[mask])
    _update_island_counts(
        state["islands"],
        labels=label_values,
        predictions=pred_values,
        cut_drop_labels=cut_drop_values,
    )

    row_counts = _empty_counts()
    _update_counts(row_counts, labels=label_values, predictions=pred_values)
    error_frames = int(row_counts["false_positive"] + row_counts["false_negative"])
    if error_frames <= 0:
        return
    state["top_error_rows"].append(
        {
            "audio_id": str(item.get("audio_id") or ""),
            "label_index": int(item.get("label_index") or 0),
            "row_index": int(item.get("row_index") or 0),
            "duration_s": float(item.get("duration_s") or 0.0),
            "frames": int(row_counts["frames"]),
            "speech_positive_frames": int(row_counts["positives"]),
            "cut_positive_frames": int(cut_values.sum()),
            "cut_drop_frames": int(cut_drop_values.sum()),
            "false_positive": int(row_counts["false_positive"]),
            "false_negative": int(row_counts["false_negative"]),
            "error_frames": error_frames,
            "error_rate": error_frames / max(1, int(row_counts["frames"])),
            "boundary_count": int(item.get("boundary_count") or 0),
            "cut_drop_zone_count": int(item.get("cut_drop_zone_count") or 0),
            "cut_point_segment_count": int(item.get("cut_point_segment_count") or 0),
        }
    )


def _update_island_counts(
    counts: dict[str, float | int],
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    cut_drop_labels: np.ndarray,
    min_overlap_ratio: float = 0.5,
    clean_cut_drop_predicted_ratio: float = 0.10,
) -> None:
    label_runs = _binary_runs(labels)
    predicted_runs = _binary_runs(predictions)
    cut_drop_runs = _binary_runs(cut_drop_labels)
    counts["label_islands"] = int(counts["label_islands"]) + len(label_runs)
    counts["predicted_islands"] = int(counts["predicted_islands"]) + len(predicted_runs)
    counts["cut_drop_zones"] = int(counts["cut_drop_zones"]) + len(cut_drop_runs)

    for label_run in label_runs:
        length = max(1, int(label_run[1] - label_run[0]))
        best_overlap = max((_overlap_frames(label_run, pred_run) for pred_run in predicted_runs), default=0)
        coverage = best_overlap / length
        counts["label_island_coverage_sum"] = float(counts["label_island_coverage_sum"]) + coverage
        if coverage >= float(min_overlap_ratio):
            counts["detected_label_islands"] = int(counts["detected_label_islands"]) + 1
        else:
            counts["missed_label_islands"] = int(counts["missed_label_islands"]) + 1

    for pred_run in predicted_runs:
        length = max(1, int(pred_run[1] - pred_run[0]))
        best_overlap = max((_overlap_frames(pred_run, label_run) for label_run in label_runs), default=0)
        overlap_ratio = best_overlap / length
        counts["predicted_island_overlap_sum"] = float(counts["predicted_island_overlap_sum"]) + overlap_ratio
        if overlap_ratio >= float(min_overlap_ratio):
            counts["matched_predicted_islands"] = int(counts["matched_predicted_islands"]) + 1
        else:
            counts["false_predicted_islands"] = int(counts["false_predicted_islands"]) + 1

    predicted_bool = np.asarray(predictions, dtype=np.float32).reshape(-1) > 0.5
    for drop_run in cut_drop_runs:
        length = max(1, int(drop_run[1] - drop_run[0]))
        predicted_ratio = float(predicted_bool[drop_run[0] : drop_run[1]].sum()) / length
        counts["cut_drop_zone_predicted_ratio_sum"] = (
            float(counts["cut_drop_zone_predicted_ratio_sum"]) + predicted_ratio
        )
        if predicted_ratio <= float(clean_cut_drop_predicted_ratio):
            counts["clean_cut_drop_zones"] = int(counts["clean_cut_drop_zones"]) + 1
        else:
            counts["dirty_cut_drop_zones"] = int(counts["dirty_cut_drop_zones"]) + 1


def _summarize_diagnostic_state(state: Mapping[str, Any]) -> dict[str, Any]:
    overall = _metrics_from_counts(state.get("overall") or {}, threshold=0.0)
    bucket_metrics = {
        key: _metrics_from_counts(counts, threshold=0.0)
        for key, counts in sorted(dict(state.get("distance_buckets") or {}).items())
        if int(counts.get("frames") or 0) > 0
    }
    region_metrics = {
        key: _metrics_from_counts(counts, threshold=0.0)
        for key, counts in sorted(dict(state.get("regions") or {}).items())
        if int(counts.get("frames") or 0) > 0
    }
    near_counts = dict((state.get("regions") or {}).get("near_boundary") or {})
    overall_counts = dict(state.get("overall") or {})
    fp_total = int(overall_counts.get("false_positive") or 0)
    fn_total = int(overall_counts.get("false_negative") or 0)
    top_error_rows = sorted(
        list(state.get("top_error_rows") or []),
        key=lambda row: (-int(row.get("error_frames") or 0), -float(row.get("error_rate") or 0.0)),
    )[:20]
    island_counts = dict(state.get("islands") or {})
    label_islands = int(island_counts.get("label_islands") or 0)
    predicted_islands = int(island_counts.get("predicted_islands") or 0)
    cut_drop_zones = int(island_counts.get("cut_drop_zones") or 0)
    island_metrics = {
        **island_counts,
        "speech_island_recall": int(island_counts.get("detected_label_islands") or 0) / max(1, label_islands),
        "predicted_island_precision": int(island_counts.get("matched_predicted_islands") or 0)
        / max(1, predicted_islands),
        "mean_label_island_coverage": float(island_counts.get("label_island_coverage_sum") or 0.0)
        / max(1, label_islands),
        "mean_predicted_island_overlap": float(island_counts.get("predicted_island_overlap_sum") or 0.0)
        / max(1, predicted_islands),
        "cut_drop_zone_clean_rate": int(island_counts.get("clean_cut_drop_zones") or 0)
        / max(1, cut_drop_zones),
        "mean_cut_drop_zone_predicted_ratio": float(
            island_counts.get("cut_drop_zone_predicted_ratio_sum") or 0.0
        )
        / max(1, cut_drop_zones),
    }
    return {
        "kind": str(state.get("kind") or ""),
        "label": str(state.get("label") or ""),
        "near_boundary_s": float(state.get("near_boundary_s") or 0.0),
        "boundary_buckets_s": list(state.get("boundary_buckets_s") or []),
        "overall": overall,
        "distance_buckets": bucket_metrics,
        "regions": region_metrics,
        "islands": island_metrics,
        "near_boundary_error_share": {
            "false_positive": int(near_counts.get("false_positive") or 0) / max(1, fp_total),
            "false_negative": int(near_counts.get("false_negative") or 0) / max(1, fn_total),
        },
        "top_error_rows": top_error_rows,
    }


def _hysteresis_predictions(
    probabilities: np.ndarray,
    *,
    on_threshold: float,
    off_threshold: float,
) -> np.ndarray:
    if on_threshold < off_threshold:
        raise ValueError("speech on threshold must be greater than or equal to speech off threshold")
    values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    predictions = np.zeros(values.size, dtype=np.float32)
    if values.size <= 0:
        return predictions
    on_indices = np.flatnonzero(values >= float(on_threshold))
    if on_indices.size <= 0:
        return predictions
    off_indices = np.flatnonzero(values < float(off_threshold))
    search_from = 0
    while True:
        on_position = int(np.searchsorted(on_indices, search_from, side="left"))
        if on_position >= int(on_indices.size):
            break
        start = int(on_indices[on_position])
        off_position = int(np.searchsorted(off_indices, start + 1, side="left"))
        if off_position >= int(off_indices.size):
            predictions[start:] = 1.0
            break
        end = int(off_indices[off_position])
        predictions[start:end] = 1.0
        search_from = end + 1
    return predictions


def _apply_cut_gate(
    speech_scores: np.ndarray,
    cut_scores: np.ndarray,
    *,
    cut_threshold: float,
) -> np.ndarray:
    frame_total = min(int(speech_scores.size), int(cut_scores.size))
    if frame_total <= 0:
        return np.asarray(speech_scores, dtype=np.float32).reshape(-1)
    gated = np.asarray(speech_scores[:frame_total], dtype=np.float32).copy()
    gated[np.asarray(cut_scores[:frame_total], dtype=np.float32) >= float(cut_threshold)] = 0.0
    return gated


def _normalized_features(bundle: Any, *, ptm: np.ndarray, mfcc: np.ndarray, frame_total: int) -> np.ndarray:
    if int(ptm.shape[1]) != int(bundle.ptm_dim):
        raise ValueError(f"scorer expected ptm_dim={bundle.ptm_dim}, got {int(ptm.shape[1])}")
    if int(mfcc.shape[1]) != int(bundle.mfcc_dim):
        raise ValueError(f"scorer expected mfcc_dim={bundle.mfcc_dim}, got {int(mfcc.shape[1])}")
    features = np.concatenate(
        [
            np.asarray(ptm[:frame_total], dtype=np.float32),
            np.asarray(mfcc[:frame_total], dtype=np.float32),
        ],
        axis=1,
    )
    mean = np.asarray(bundle.normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(bundle.normalization["feature_std"], dtype=np.float32)
    if int(mean.shape[0]) != int(features.shape[1]) or int(std.shape[0]) != int(features.shape[1]):
        raise ValueError("scorer normalization dimension does not match feature dimension")
    return np.ascontiguousarray((features - mean) / np.maximum(std, 1e-6), dtype=np.float32)


def _score_feature_batches(
    bundle: Any,
    features: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    import torch

    if not features:
        return []
    max_frames = max(int(item.shape[0]) for item in features)
    input_dim = int(features[0].shape[1])
    if max_frames <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return [(empty, empty) for _item in features]
    batch = np.zeros((len(features), max_frames, input_dim), dtype=np.float32)
    attention_mask = np.zeros((len(features), max_frames), dtype=np.int64)
    lengths: list[int] = []
    for index, feature_array in enumerate(features):
        if int(feature_array.shape[1]) != input_dim:
            raise ValueError("all feature arrays in a batch must share input_dim")
        frame_count = int(feature_array.shape[0])
        lengths.append(frame_count)
        if frame_count <= 0:
            continue
        batch[index, :frame_count, :] = feature_array
        attention_mask[index, :frame_count] = 1
    with torch.inference_mode():
        tensor = torch.from_numpy(batch).to(bundle.device)
        mask = torch.from_numpy(attention_mask).to(bundle.device)
        logits = bundle.model(tensor, attention_mask=mask)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    if probabilities.ndim != 3 or int(probabilities.shape[-1]) != 2:
        raise ValueError("boundary scorer batch probabilities must have shape [batch, frames, 2]")
    scored: list[tuple[np.ndarray, np.ndarray]] = []
    for index, frame_count in enumerate(lengths):
        values = probabilities[index, :frame_count, :]
        scored.append(
            (
                np.ascontiguousarray(values[:, 0], dtype=np.float32),
                np.ascontiguousarray(values[:, 1], dtype=np.float32),
            )
        )
    return scored


def _row_frame_count(row: Mapping[str, Any]) -> int:
    try:
        return max(0, int(row.get("frame_count") or 0))
    except (TypeError, ValueError):
        return 0


def _metrics_from_counts(counts: Mapping[str, int], *, threshold: float) -> dict[str, Any]:
    frames = int(counts.get("frames") or 0)
    positives = int(counts.get("positives") or 0)
    negatives = int(counts.get("negatives") or 0)
    predicted = int(counts.get("predicted_positives") or 0)
    tp = int(counts.get("true_positive") or 0)
    fp = int(counts.get("false_positive") or 0)
    fn = int(counts.get("false_negative") or 0)
    tn = int(counts.get("true_negative") or 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    return {
        "threshold": float(threshold),
        "frames": frames,
        "positives": positives,
        "negatives": negatives,
        "predicted_positives": predicted,
        "positive_ratio": positives / max(1, frames),
        "predicted_positive_ratio": predicted / max(1, frames),
        "accuracy": int(counts.get("correct") or 0) / max(1, frames),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _choose_threshold(
    rows: list[dict[str, Any]],
    *,
    min_recall: float,
    max_false_positive_rate: float,
) -> dict[str, Any]:
    passing = [
        row
        for row in rows
        if float(row["recall"]) >= float(min_recall)
        and float(row["false_positive_rate"]) <= float(max_false_positive_rate)
    ]
    if passing:
        selected = sorted(
            passing,
            key=lambda row: (
                -float(row["f1"]),
                float(row["false_positive_rate"]),
                -float(row.get("threshold", row.get("speech_on_threshold", 0.0))),
            ),
        )[0]
        return {
            "status": "passes_policy",
            "policy": {
                "min_recall": float(min_recall),
                "max_false_positive_rate": float(max_false_positive_rate),
            },
            "selected": selected,
        }
    selected = sorted(rows, key=lambda row: (-float(row["f1"]), -float(row["recall"])))[0] if rows else {}
    return {
        "status": "no_threshold_passes_policy",
        "policy": {
            "min_recall": float(min_recall),
            "max_false_positive_rate": float(max_false_positive_rate),
        },
        "selected": selected,
    }


def evaluate_thresholds(
    *,
    checkpoint: Path,
    labels: Path,
    feature_manifest: Path,
    output_dir: Path,
    thresholds: Iterable[float],
    min_speech_recall: float = 0.995,
    max_speech_false_positive_rate: float = 0.02,
    min_cut_recall: float = 0.90,
    max_cut_false_positive_rate: float = 0.10,
    cut_min_gap_s: float = 0.5,
    cut_boundary_radius_frames: int = 1,
    device: str = "cpu",
    limit: int | None = None,
    batch_size: int = 8,
    runtime_profiles: Iterable[tuple[float, float, float]] | None = None,
    diagnostic_thresholds: Iterable[float] | None = None,
    diagnostic_runtime_profiles: Iterable[tuple[float, float, float]] | None = None,
    diagnostic_boundary_buckets_s: Iterable[float] = (0.04, 0.08, 0.16, 0.32, 0.64),
    diagnostic_near_boundary_s: float = 0.12,
) -> dict[str, Any]:
    records = load_label_records(labels)
    rows = _read_jsonl(feature_manifest)
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    threshold_values = sorted({round(float(value), 6) for value in thresholds})
    if not threshold_values:
        raise ValueError("at least one threshold is required")
    batch_size = max(1, int(batch_size))
    bundle = load_feature_frame_scorer_checkpoint(checkpoint, device=device)
    speech_counts = {threshold: _empty_counts() for threshold in threshold_values}
    cut_counts = {threshold: _empty_counts() for threshold in threshold_values}
    profile_values = sorted(
        {
            (
                round(float(on_threshold), 6),
                round(float(off_threshold), 6),
                round(float(cut_threshold), 6),
            )
            for on_threshold, off_threshold, cut_threshold in (runtime_profiles or [])
        }
    )
    for on_threshold, off_threshold, _cut_threshold in profile_values:
        if on_threshold < off_threshold:
            raise ValueError("runtime profile speech on threshold must be >= speech off threshold")
    runtime_profile_counts = {profile: _empty_counts() for profile in profile_values}
    diagnostic_threshold_values = sorted({round(float(value), 6) for value in (diagnostic_thresholds or [])})
    diagnostic_profile_values = sorted(
        {
            (
                round(float(on_threshold), 6),
                round(float(off_threshold), 6),
                round(float(cut_threshold), 6),
            )
            for on_threshold, off_threshold, cut_threshold in (diagnostic_runtime_profiles or [])
        }
    )
    for on_threshold, off_threshold, _cut_threshold in diagnostic_profile_values:
        if on_threshold < off_threshold:
            raise ValueError("diagnostic runtime profile speech on threshold must be >= speech off threshold")
    diagnostic_buckets = tuple(sorted({float(value) for value in diagnostic_boundary_buckets_s if float(value) >= 0.0}))
    diagnostic_near_boundary_s = max(0.0, float(diagnostic_near_boundary_s))
    diagnostic_states: dict[str, dict[str, Any]] = {}
    for threshold in diagnostic_threshold_values:
        key = f"threshold_{threshold:.6f}"
        diagnostic_states[key] = _empty_diagnostic_state(
            kind="threshold",
            label=f"threshold={threshold:g}",
            buckets_s=diagnostic_buckets,
            near_boundary_s=diagnostic_near_boundary_s,
        )
    for on_threshold, off_threshold, cut_threshold in diagnostic_profile_values:
        key = f"runtime_on{on_threshold:.6f}_off{off_threshold:.6f}_cut{cut_threshold:.6f}"
        diagnostic_states[key] = _empty_diagnostic_state(
            kind="runtime_profile",
            label=f"on={on_threshold:g},off={off_threshold:g},cut={cut_threshold:g}",
            buckets_s=diagnostic_buckets,
            near_boundary_s=diagnostic_near_boundary_s,
        )
    quality_counts: dict[str, dict[str, dict[float, dict[str, int]]]] = {}
    source_counts = Counter()
    label_quality_counts = Counter()
    skipped: list[dict[str, Any]] = []

    def _consume_scored_item(
        item: Mapping[str, Any],
        *,
        speech_scores: np.ndarray,
        cut_scores: np.ndarray,
    ) -> None:
        frame_total = int(item["frame_total"])
        mask = np.asarray(item["mask"], dtype=bool)
        speech_scores_array = np.asarray(speech_scores[:frame_total], dtype=np.float32)[mask]
        cut_scores_array = np.asarray(cut_scores[:frame_total], dtype=np.float32)[mask]
        speech_labels = np.asarray(item["speech_labels"], dtype=np.float32)
        cut_labels = np.asarray(item["cut_labels"], dtype=np.float32)
        cut_drop_labels = np.asarray(item["cut_drop_labels"], dtype=np.float32)
        boundary_distances_s = np.asarray(item["boundary_distances_s"], dtype=np.float32)
        quality = str(item["quality"])
        quality_counts.setdefault(
            quality,
            {
                "speech": {threshold: _empty_counts() for threshold in threshold_values},
                "cut": {threshold: _empty_counts() for threshold in threshold_values},
            },
        )
        source_counts[str(item["source"])] += 1
        label_quality_counts[quality] += 1
        for threshold in threshold_values:
            speech_predictions = (speech_scores_array >= threshold).astype(np.float32)
            cut_predictions = (cut_scores_array >= threshold).astype(np.float32)
            _update_counts(speech_counts[threshold], labels=speech_labels, predictions=speech_predictions)
            _update_counts(cut_counts[threshold], labels=cut_labels, predictions=cut_predictions)
            _update_counts(
                quality_counts[quality]["speech"][threshold],
                labels=speech_labels,
                predictions=speech_predictions,
            )
            _update_counts(
                quality_counts[quality]["cut"][threshold],
                labels=cut_labels,
                predictions=cut_predictions,
            )
        for profile in profile_values:
            on_threshold, off_threshold, cut_threshold = profile
            gated_scores = _apply_cut_gate(
                speech_scores_array,
                cut_scores_array,
                cut_threshold=cut_threshold,
            )
            profile_predictions = _hysteresis_predictions(
                gated_scores,
                on_threshold=on_threshold,
                off_threshold=off_threshold,
            )
            _update_counts(
                runtime_profile_counts[profile],
                labels=speech_labels,
                predictions=profile_predictions,
            )
        for threshold in diagnostic_threshold_values:
            key = f"threshold_{threshold:.6f}"
            speech_predictions = (speech_scores_array >= threshold).astype(np.float32)
            _update_diagnostic_state(
                diagnostic_states[key],
                labels=speech_labels,
                predictions=speech_predictions,
                boundary_distances_s=boundary_distances_s,
                cut_labels=cut_labels,
                cut_drop_labels=cut_drop_labels,
                item=item,
                buckets_s=diagnostic_buckets,
                near_boundary_s=diagnostic_near_boundary_s,
            )
        for profile in diagnostic_profile_values:
            on_threshold, off_threshold, cut_threshold = profile
            key = f"runtime_on{on_threshold:.6f}_off{off_threshold:.6f}_cut{cut_threshold:.6f}"
            gated_scores = _apply_cut_gate(
                speech_scores_array,
                cut_scores_array,
                cut_threshold=cut_threshold,
            )
            profile_predictions = _hysteresis_predictions(
                gated_scores,
                on_threshold=on_threshold,
                off_threshold=off_threshold,
            )
            _update_diagnostic_state(
                diagnostic_states[key],
                labels=speech_labels,
                predictions=profile_predictions,
                boundary_distances_s=boundary_distances_s,
                cut_labels=cut_labels,
                cut_drop_labels=cut_drop_labels,
                item=item,
                buckets_s=diagnostic_buckets,
                near_boundary_s=diagnostic_near_boundary_s,
            )

    indexed_rows = sorted(enumerate(rows), key=lambda item: (_row_frame_count(item[1]), item[0]))
    for batch_start in range(0, len(indexed_rows), batch_size):
        batch_items: list[dict[str, Any]] = []
        for row_index, row in indexed_rows[batch_start : batch_start + batch_size]:
            try:
                label_index = int(row["label_index"])
                record = records[label_index]
                ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
                weights = np.asarray(effective_frame_weights(record), dtype=np.float32)
                frame_total = min(int(ptm.shape[0]), int(mfcc.shape[0]), len(record.speech_frames), weights.shape[0])
                if frame_total <= 0:
                    skipped.append({"row_index": row_index, "label_index": label_index, "reason": "empty_frames"})
                    continue
                mask = weights[:frame_total] > 0.0
                if not bool(mask.any()):
                    skipped.append({"row_index": row_index, "label_index": label_index, "reason": "zero_weight"})
                    continue
                _starts, _ends, cut_drops, cut_points = endpoint_targets_from_record(
                    record,
                    frame_count=frame_total,
                    boundary_radius_frames=0,
                    cut_min_gap_s=cut_min_gap_s,
                    cut_boundary_radius_frames=cut_boundary_radius_frames,
                )
                metadata = dict(record.boundary_metadata or {})
                speech_label_array = np.asarray(record.speech_frames[:frame_total], dtype=np.float32)
                boundary_distances_s = _speech_boundary_distances_s(
                    speech_label_array,
                    frame_hop_s=float(record.frame_hop_s),
                )[mask]
                batch_items.append(
                    {
                        "row_index": row_index,
                        "label_index": label_index,
                        "audio_id": record.audio_id,
                        "duration_s": float(record.duration_s),
                        "features": _normalized_features(bundle, ptm=ptm, mfcc=mfcc, frame_total=frame_total),
                        "frame_total": frame_total,
                        "mask": mask,
                        "speech_labels": speech_label_array[mask],
                        "cut_labels": np.maximum(cut_drops[:frame_total], cut_points[:frame_total]).astype(np.float32)[
                            mask
                        ],
                        "cut_drop_labels": cut_drops[:frame_total].astype(np.float32)[mask],
                        "boundary_distances_s": boundary_distances_s,
                        "quality": str(record.label_quality or row.get("label_quality") or ""),
                        "source": str(record.source),
                        "boundary_count": len(list(metadata.get("utterance_boundaries") or [])),
                        "cut_drop_zone_count": len(list(metadata.get("cut_drop_zones") or [])),
                        "cut_point_segment_count": len(list(metadata.get("cut_point_segments") or [])),
                    }
                )
            except Exception as exc:  # pragma: no cover - exercised by real corrupt manifests
                skipped.append(
                    {
                        "row_index": row_index,
                        "label_index": row.get("label_index"),
                        "reason": "error",
                        "error": str(exc),
                    }
                )
        if not batch_items:
            continue
        try:
            scored_batch = _score_feature_batches(bundle, [np.asarray(item["features"]) for item in batch_items])
        except Exception as batch_exc:  # pragma: no cover - fallback for device/batch-size limits
            for item in batch_items:
                try:
                    speech_scores, cut_scores = _score_feature_batches(bundle, [np.asarray(item["features"])])[0]
                    _consume_scored_item(item, speech_scores=speech_scores, cut_scores=cut_scores)
                except Exception as item_exc:
                    skipped.append(
                        {
                            "row_index": item.get("row_index"),
                            "label_index": item.get("label_index"),
                            "reason": "error",
                            "error": f"{batch_exc}; fallback item error: {item_exc}",
                        }
                    )
            continue
        for item, (speech_scores, cut_scores) in zip(batch_items, scored_batch):
            try:
                _consume_scored_item(item, speech_scores=speech_scores, cut_scores=cut_scores)
            except Exception as item_exc:
                skipped.append(
                    {
                        "row_index": item.get("row_index"),
                        "label_index": item.get("label_index"),
                        "reason": "error",
                        "error": str(item_exc),
                    }
                )

    speech_metrics = [_metrics_from_counts(speech_counts[threshold], threshold=threshold) for threshold in threshold_values]
    cut_metrics = [_metrics_from_counts(cut_counts[threshold], threshold=threshold) for threshold in threshold_values]
    runtime_profile_metrics = [
        {
            "speech_on_threshold": on_threshold,
            "speech_off_threshold": off_threshold,
            "cut_threshold": cut_threshold,
            **{
                key: value
                for key, value in _metrics_from_counts(
                    runtime_profile_counts[(on_threshold, off_threshold, cut_threshold)],
                    threshold=on_threshold,
                ).items()
                if key != "threshold"
            },
        }
        for on_threshold, off_threshold, cut_threshold in profile_values
    ]
    by_quality = {
        quality: {
            "speech": [
                _metrics_from_counts(head_counts["speech"][threshold], threshold=threshold)
                for threshold in threshold_values
            ],
            "cut": [
                _metrics_from_counts(head_counts["cut"][threshold], threshold=threshold)
                for threshold in threshold_values
            ],
        }
        for quality, head_counts in sorted(quality_counts.items())
    }
    recommendation = {
        "speech": _choose_threshold(
            speech_metrics,
            min_recall=min_speech_recall,
            max_false_positive_rate=max_speech_false_positive_rate,
        ),
        "cut": _choose_threshold(
            cut_metrics,
            min_recall=min_cut_recall,
            max_false_positive_rate=max_cut_false_positive_rate,
        ),
        "runtime_profile": _choose_threshold(
            runtime_profile_metrics,
            min_recall=min_speech_recall,
            max_false_positive_rate=max_speech_false_positive_rate,
        ) if runtime_profile_metrics else {
            "status": "not_evaluated",
            "policy": {
                "min_recall": float(min_speech_recall),
                "max_false_positive_rate": float(max_speech_false_positive_rate),
            },
            "selected": {},
        },
    }
    speech_error_diagnostics = {
        key: _summarize_diagnostic_state(state)
        for key, state in sorted(diagnostic_states.items())
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "speech_boundary_ja_mamba2_frame_boundary_scorer_threshold_eval_v3",
        "checkpoint": _repo_display(checkpoint),
        "checkpoint_signature": bundle.signature(),
        "labels": _repo_display(labels),
        "feature_manifest": _repo_display(feature_manifest),
        "output_dir": _repo_display(output_dir),
        "device": str(device),
        "batch_size": int(batch_size),
        "batching": {"length_bucketed": True},
        "rows": len(rows),
        "skipped": len(skipped),
        "skipped_samples": skipped[:20],
        "label_quality_counts": dict(sorted(label_quality_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "thresholds": threshold_values,
        "speech_metrics": speech_metrics,
        "cut_metrics": cut_metrics,
        "runtime_profile_metrics": runtime_profile_metrics,
        "speech_error_diagnostics": speech_error_diagnostics,
        "by_label_quality": by_quality,
        "recommendation": recommendation,
        "cut_target_policy": {
            "cut_min_gap_s": float(cut_min_gap_s),
            "cut_boundary_radius_frames": int(cut_boundary_radius_frames),
            "positive_target": "cut_point_segments_or_cut_drop_zones",
        },
        "runtime_status": {
            "default_replaced": False,
            "opt_in_env": "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO",
            "note": "Threshold eval is offline. Do not promote without full workflow smoke and human audit.",
        },
    }
    (output_dir / "threshold_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "threshold_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for head, rows_for_head in (("speech", speech_metrics), ("cut", cut_metrics)):
            for row in rows_for_head:
                payload = {"head": head, **row}
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        for row in runtime_profile_metrics:
            payload = {"head": "runtime_profile", **row}
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    if speech_error_diagnostics:
        with (output_dir / "speech_error_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
            for key, row in speech_error_diagnostics.items():
                payload = {"diagnostic_id": key, **row}
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    speech_recommendation = dict(summary.get("recommendation", {}).get("speech") or {})
    cut_recommendation = dict(summary.get("recommendation", {}).get("cut") or {})
    runtime_recommendation = dict(summary.get("recommendation", {}).get("runtime_profile") or {})
    speech_selected = dict(speech_recommendation.get("selected") or {})
    cut_selected = dict(cut_recommendation.get("selected") or {})
    runtime_selected = dict(runtime_recommendation.get("selected") or {})
    lines = [
        "# SpeechBoundary-JA Mamba2 Frame Boundary Scorer Threshold Eval",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Skipped: `{summary['skipped']}`",
        f"- Speech recommendation status: `{speech_recommendation.get('status', '')}`",
        f"- Cut recommendation status: `{cut_recommendation.get('status', '')}`",
        f"- Runtime profile recommendation status: `{runtime_recommendation.get('status', '')}`",
    ]
    if speech_selected:
        lines.extend(
            [
                f"- Selected speech threshold: `{speech_selected.get('threshold')}`",
                f"- Speech recall: `{float(speech_selected.get('recall', 0.0)):.6f}`",
                f"- Speech FPR: `{float(speech_selected.get('false_positive_rate', 0.0)):.6f}`",
                f"- Speech F1: `{float(speech_selected.get('f1', 0.0)):.6f}`",
            ]
        )
    if runtime_selected:
        lines.extend(
            [
                f"- Selected runtime profile: `on={runtime_selected.get('speech_on_threshold')}, off={runtime_selected.get('speech_off_threshold')}, cut={runtime_selected.get('cut_threshold')}`",
                f"- Runtime profile recall: `{float(runtime_selected.get('recall', 0.0)):.6f}`",
                f"- Runtime profile FPR: `{float(runtime_selected.get('false_positive_rate', 0.0)):.6f}`",
                f"- Runtime profile F1: `{float(runtime_selected.get('f1', 0.0)):.6f}`",
            ]
        )
    if cut_selected:
        lines.extend(
            [
                f"- Selected cut threshold: `{cut_selected.get('threshold')}`",
                f"- Cut recall: `{float(cut_selected.get('recall', 0.0)):.6f}`",
                f"- Cut FPR: `{float(cut_selected.get('false_positive_rate', 0.0)):.6f}`",
                f"- Cut F1: `{float(cut_selected.get('f1', 0.0)):.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Speech",
            "",
            "| threshold | precision | recall | f1 | fpr | predicted_positive_ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in list(summary.get("speech_metrics") or []):
        lines.append(
            "| {threshold:.3f} | {precision:.6f} | {recall:.6f} | {f1:.6f} | {false_positive_rate:.6f} | {predicted_positive_ratio:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Cut",
            "",
            "| threshold | precision | recall | f1 | fpr | predicted_positive_ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in list(summary.get("cut_metrics") or []):
        lines.append(
            "| {threshold:.3f} | {precision:.6f} | {recall:.6f} | {f1:.6f} | {false_positive_rate:.6f} | {predicted_positive_ratio:.6f} |".format(
                **row
            )
        )
    if summary.get("runtime_profile_metrics"):
        lines.extend(
            [
                "",
                "## Runtime Profiles",
                "",
                "| speech_on | speech_off | cut | precision | recall | f1 | fpr | predicted_positive_ratio |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in list(summary.get("runtime_profile_metrics") or []):
            lines.append(
                "| {speech_on_threshold:.3f} | {speech_off_threshold:.3f} | {cut_threshold:.3f} | {precision:.6f} | {recall:.6f} | {f1:.6f} | {false_positive_rate:.6f} | {predicted_positive_ratio:.6f} |".format(
                    **row
                )
            )
    if summary.get("speech_error_diagnostics"):
        lines.extend(["", "## Speech Error Diagnostics", ""])
        for diagnostic_id, diagnostic in sorted(dict(summary.get("speech_error_diagnostics") or {}).items()):
            overall = dict(diagnostic.get("overall") or {})
            near_share = dict(diagnostic.get("near_boundary_error_share") or {})
            lines.extend(
                [
                    f"### {diagnostic.get('label') or diagnostic_id}",
                    "",
                    f"- Overall recall: `{float(overall.get('recall', 0.0)):.6f}`",
                    f"- Overall FPR: `{float(overall.get('false_positive_rate', 0.0)):.6f}`",
                    f"- FP near-boundary share: `{float(near_share.get('false_positive', 0.0)):.6f}`",
                    f"- FN near-boundary share: `{float(near_share.get('false_negative', 0.0)):.6f}`",
                    "",
                    "| distance bucket | frames | fp | fn | recall | fpr |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for bucket, row in sorted(dict(diagnostic.get("distance_buckets") or {}).items()):
                lines.append(
                    "| {bucket} | {frames} | {false_positive} | {false_negative} | {recall:.6f} | {false_positive_rate:.6f} |".format(
                        bucket=bucket,
                        **row,
                    )
                )
            lines.extend(["", "| region | frames | fp | fn | recall | fpr |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
            for region, row in sorted(dict(diagnostic.get("regions") or {}).items()):
                lines.append(
                    "| {region} | {frames} | {false_positive} | {false_negative} | {recall:.6f} | {false_positive_rate:.6f} |".format(
                        region=region,
                        **row,
                    )
                )
            islands = dict(diagnostic.get("islands") or {})
            if islands:
                lines.extend(
                    [
                        "",
                        "| island metric | value |",
                        "| --- | ---: |",
                        f"| label_islands | {int(islands.get('label_islands') or 0)} |",
                        f"| speech_island_recall | {float(islands.get('speech_island_recall') or 0.0):.6f} |",
                        f"| mean_label_island_coverage | {float(islands.get('mean_label_island_coverage') or 0.0):.6f} |",
                        f"| predicted_islands | {int(islands.get('predicted_islands') or 0)} |",
                        f"| predicted_island_precision | {float(islands.get('predicted_island_precision') or 0.0):.6f} |",
                        f"| mean_predicted_island_overlap | {float(islands.get('mean_predicted_island_overlap') or 0.0):.6f} |",
                        f"| cut_drop_zones | {int(islands.get('cut_drop_zones') or 0)} |",
                        f"| cut_drop_zone_clean_rate | {float(islands.get('cut_drop_zone_clean_rate') or 0.0):.6f} |",
                        f"| mean_cut_drop_zone_predicted_ratio | {float(islands.get('mean_cut_drop_zone_predicted_ratio') or 0.0):.6f} |",
                    ]
                )
            top_rows = list(diagnostic.get("top_error_rows") or [])[:10]
            if top_rows:
                lines.extend(
                    [
                        "",
                        "| top error audio_id | fp | fn | error_rate | boundary_count | cut_drop | cut_point |",
                        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                    ]
                )
                for row in top_rows:
                    lines.append(
                        "| {audio_id} | {false_positive} | {false_negative} | {error_rate:.6f} | {boundary_count} | {cut_drop_zone_count} | {cut_point_segment_count} |".format(
                            **row
                        )
                    )
            lines.append("")
    lines.extend(
        [
            "",
            "Do not promote this scorer from this offline metric alone. Run no-translate full workflow smoke and human audit first.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_runtime_profile(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(value or "").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--runtime-profile must be on,off,cut")
    try:
        on_threshold, off_threshold, cut_threshold = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--runtime-profile values must be numbers") from exc
    if not 0.0 <= on_threshold <= 1.0:
        raise argparse.ArgumentTypeError("runtime profile on threshold must be in [0, 1]")
    if not 0.0 <= off_threshold <= 1.0:
        raise argparse.ArgumentTypeError("runtime profile off threshold must be in [0, 1]")
    if not 0.0 <= cut_threshold <= 1.0:
        raise argparse.ArgumentTypeError("runtime profile cut threshold must be in [0, 1]")
    if on_threshold < off_threshold:
        raise argparse.ArgumentTypeError("runtime profile on threshold must be >= off threshold")
    return on_threshold, off_threshold, cut_threshold


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SpeechBoundary-JA speech+cut feature scorer thresholds from cached features."
    )
    parser.add_argument("--checkpoint", required=True, help="Feature scorer checkpoint.")
    parser.add_argument("--labels", required=True, help="SpeechBoundary-JA label JSONL.")
    parser.add_argument("--feature-manifest", required=True, help="Feature cache manifest JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", action="append", type=float, default=[])
    parser.add_argument("--min-speech-recall", type=float, default=0.995)
    parser.add_argument("--max-speech-false-positive-rate", type=float, default=0.02)
    parser.add_argument("--min-cut-recall", type=float, default=0.90)
    parser.add_argument("--max-cut-false-positive-rate", type=float, default=0.10)
    parser.add_argument("--cut-min-gap-s", type=float, default=0.5)
    parser.add_argument("--cut-boundary-radius-frames", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--runtime-profile",
        action="append",
        type=_parse_runtime_profile,
        default=[],
        help="Evaluate a runtime profile as speech_on,speech_off,cut after cut gate and hysteresis.",
    )
    parser.add_argument(
        "--diagnostic-threshold",
        action="append",
        type=float,
        default=[],
        help="Add speech error diagnostics for a simple speech probability threshold.",
    )
    parser.add_argument(
        "--diagnostic-runtime-profile",
        action="append",
        type=_parse_runtime_profile,
        default=[],
        help="Add speech error diagnostics for a runtime profile as speech_on,speech_off,cut.",
    )
    parser.add_argument(
        "--diagnostic-boundary-bucket-s",
        action="append",
        type=float,
        default=[],
        help="Boundary-distance bucket upper bound in seconds. Repeat to override defaults.",
    )
    parser.add_argument("--diagnostic-near-boundary-s", type=float, default=0.12)
    args = parser.parse_args(argv)
    if not 0.0 <= args.min_speech_recall <= 1.0:
        parser.error("--min-speech-recall must be in [0, 1]")
    if not 0.0 <= args.max_speech_false_positive_rate <= 1.0:
        parser.error("--max-speech-false-positive-rate must be in [0, 1]")
    if not 0.0 <= args.min_cut_recall <= 1.0:
        parser.error("--min-cut-recall must be in [0, 1]")
    if not 0.0 <= args.max_cut_false_positive_rate <= 1.0:
        parser.error("--max-cut-false-positive-rate must be in [0, 1]")
    if args.cut_min_gap_s < 0.0:
        parser.error("--cut-min-gap-s must be non-negative")
    if args.cut_boundary_radius_frames < 0:
        parser.error("--cut-boundary-radius-frames must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive when set")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    for threshold in args.diagnostic_threshold:
        if not 0.0 <= threshold <= 1.0:
            parser.error("--diagnostic-threshold must be in [0, 1]")
    for bucket in args.diagnostic_boundary_bucket_s:
        if bucket < 0.0:
            parser.error("--diagnostic-boundary-bucket-s must be non-negative")
    if args.diagnostic_near_boundary_s < 0.0:
        parser.error("--diagnostic-near-boundary-s must be non-negative")
    return args


def run(args: argparse.Namespace) -> None:
    thresholds = args.threshold if args.threshold else _default_thresholds()
    summary = evaluate_thresholds(
        checkpoint=Path(args.checkpoint),
        labels=Path(args.labels),
        feature_manifest=Path(args.feature_manifest),
        output_dir=Path(args.output_dir),
        thresholds=thresholds,
        min_speech_recall=args.min_speech_recall,
        max_speech_false_positive_rate=args.max_speech_false_positive_rate,
        min_cut_recall=args.min_cut_recall,
        max_cut_false_positive_rate=args.max_cut_false_positive_rate,
        cut_min_gap_s=args.cut_min_gap_s,
        cut_boundary_radius_frames=args.cut_boundary_radius_frames,
        device=args.device,
        limit=args.limit,
        batch_size=args.batch_size,
        runtime_profiles=args.runtime_profile,
        diagnostic_thresholds=args.diagnostic_threshold,
        diagnostic_runtime_profiles=args.diagnostic_runtime_profile,
        diagnostic_boundary_buckets_s=(
            args.diagnostic_boundary_bucket_s
            if args.diagnostic_boundary_bucket_s
            else (0.04, 0.08, 0.16, 0.32, 0.64)
        ),
        diagnostic_near_boundary_s=args.diagnostic_near_boundary_s,
    )
    speech_selected = summary["recommendation"]["speech"].get("selected") or {}
    cut_selected = summary["recommendation"]["cut"].get("selected") or {}
    print(f"summary={Path(args.output_dir) / 'threshold_eval_summary.json'}")
    print(f"speech_recommendation_status={summary['recommendation']['speech']['status']}")
    print(f"cut_recommendation_status={summary['recommendation']['cut']['status']}")
    print(f"runtime_profile_recommendation_status={summary['recommendation']['runtime_profile']['status']}")
    if speech_selected:
        print(f"selected_speech_threshold={speech_selected['threshold']}")
        print(f"selected_speech_recall={speech_selected['recall']:.6f}")
        print(f"selected_speech_fpr={speech_selected['false_positive_rate']:.6f}")
    if cut_selected:
        print(f"selected_cut_threshold={cut_selected['threshold']}")
        print(f"selected_cut_recall={cut_selected['recall']:.6f}")
        print(f"selected_cut_fpr={cut_selected['false_positive_rate']:.6f}")
    runtime_selected = summary["recommendation"]["runtime_profile"].get("selected") or {}
    if runtime_selected:
        print(
            "selected_runtime_profile="
            f"on={runtime_selected['speech_on_threshold']},"
            f"off={runtime_selected['speech_off_threshold']},"
            f"cut={runtime_selected['cut_threshold']}"
        )
        print(f"selected_runtime_profile_recall={runtime_selected['recall']:.6f}")
        print(f"selected_runtime_profile_fpr={runtime_selected['false_positive_rate']:.6f}")


if __name__ == "__main__":
    run(parse_args())
