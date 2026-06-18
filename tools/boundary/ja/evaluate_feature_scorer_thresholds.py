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
                batch_items.append(
                    {
                        "row_index": row_index,
                        "label_index": label_index,
                        "features": _normalized_features(bundle, ptm=ptm, mfcc=mfcc, frame_total=frame_total),
                        "frame_total": frame_total,
                        "mask": mask,
                        "speech_labels": np.asarray(record.speech_frames[:frame_total], dtype=np.float32)[mask],
                        "cut_labels": np.maximum(cut_drops[:frame_total], cut_points[:frame_total]).astype(np.float32)[
                            mask
                        ],
                        "quality": str(record.label_quality or row.get("label_quality") or ""),
                        "source": str(record.source),
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
        "by_label_quality": by_quality,
        "recommendation": recommendation,
        "cut_target_policy": {
            "cut_min_gap_s": float(cut_min_gap_s),
            "cut_boundary_radius_frames": int(cut_boundary_radius_frames),
            "positive_target": "cut_point_segments_or_cut_drop_zones",
        },
        "runtime_status": {
            "default_replaced": False,
            "opt_in_env": "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT",
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
