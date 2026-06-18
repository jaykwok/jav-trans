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
    effective_frame_weights,
    load_cached_feature,
    load_feature_frame_scorer_checkpoint,
    load_label_records,
    score_feature_frame_probabilities,
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
                -float(row["threshold"]),
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
    min_recall: float = 0.995,
    max_false_positive_rate: float = 0.02,
    device: str = "cpu",
    limit: int | None = None,
) -> dict[str, Any]:
    records = load_label_records(labels)
    rows = _read_jsonl(feature_manifest)
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    threshold_values = sorted({round(float(value), 6) for value in thresholds})
    if not threshold_values:
        raise ValueError("at least one threshold is required")
    bundle = load_feature_frame_scorer_checkpoint(checkpoint, device=device)
    overall_counts = {threshold: _empty_counts() for threshold in threshold_values}
    quality_counts: dict[str, dict[float, dict[str, int]]] = {}
    source_counts = Counter()
    label_quality_counts = Counter()
    skipped: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        try:
            label_index = int(row["label_index"])
            record = records[label_index]
            ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
            scores = score_feature_frame_probabilities(bundle, ptm=ptm, mfcc=mfcc)
            weights = np.asarray(effective_frame_weights(record), dtype=np.float32)
            frame_total = min(scores.shape[0], len(record.speech_frames), weights.shape[0])
            if frame_total <= 0:
                skipped.append({"row_index": row_index, "label_index": label_index, "reason": "empty_frames"})
                continue
            mask = weights[:frame_total] > 0.0
            if not bool(mask.any()):
                skipped.append({"row_index": row_index, "label_index": label_index, "reason": "zero_weight"})
                continue
            labels_array = np.asarray(record.speech_frames[:frame_total], dtype=np.float32)[mask]
            scores_array = scores[:frame_total][mask]
            quality = str(record.label_quality or row.get("label_quality") or "")
            quality_counts.setdefault(quality, {threshold: _empty_counts() for threshold in threshold_values})
            source_counts[str(record.source)] += 1
            label_quality_counts[quality] += 1
            for threshold in threshold_values:
                predictions = (scores_array >= threshold).astype(np.float32)
                _update_counts(overall_counts[threshold], labels=labels_array, predictions=predictions)
                _update_counts(
                    quality_counts[quality][threshold],
                    labels=labels_array,
                    predictions=predictions,
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

    metrics = [_metrics_from_counts(overall_counts[threshold], threshold=threshold) for threshold in threshold_values]
    by_quality = {
        quality: [
            _metrics_from_counts(counts_by_threshold[threshold], threshold=threshold)
            for threshold in threshold_values
        ]
        for quality, counts_by_threshold in sorted(quality_counts.items())
    }
    recommendation = _choose_threshold(
        metrics,
        min_recall=min_recall,
        max_false_positive_rate=max_false_positive_rate,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "speech_boundary_ja_feature_scorer_threshold_eval_v1",
        "checkpoint": _repo_display(checkpoint),
        "checkpoint_signature": bundle.signature(),
        "labels": _repo_display(labels),
        "feature_manifest": _repo_display(feature_manifest),
        "output_dir": _repo_display(output_dir),
        "device": str(device),
        "rows": len(rows),
        "skipped": len(skipped),
        "skipped_samples": skipped[:20],
        "label_quality_counts": dict(sorted(label_quality_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "thresholds": threshold_values,
        "metrics": metrics,
        "by_label_quality": by_quality,
        "recommendation": recommendation,
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
        for row in metrics:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    selected = dict(summary.get("recommendation", {}).get("selected") or {})
    lines = [
        "# SpeechBoundary-JA Feature Scorer Threshold Eval",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Skipped: `{summary['skipped']}`",
        f"- Recommendation status: `{summary['recommendation']['status']}`",
    ]
    if selected:
        lines.extend(
            [
                f"- Selected threshold: `{selected.get('threshold')}`",
                f"- Recall: `{float(selected.get('recall', 0.0)):.6f}`",
                f"- FPR: `{float(selected.get('false_positive_rate', 0.0)):.6f}`",
                f"- F1: `{float(selected.get('f1', 0.0)):.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "| threshold | precision | recall | f1 | fpr | predicted_positive_ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in list(summary.get("metrics") or []):
        lines.append(
            "| {threshold:.3f} | {precision:.6f} | {recall:.6f} | {f1:.6f} | {false_positive_rate:.6f} | {predicted_positive_ratio:.6f} |".format(
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SpeechBoundary-JA feature scorer thresholds from cached features."
    )
    parser.add_argument("--checkpoint", required=True, help="Feature scorer checkpoint.")
    parser.add_argument("--labels", required=True, help="SpeechBoundary-JA label JSONL.")
    parser.add_argument("--feature-manifest", required=True, help="Feature cache manifest JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", action="append", type=float, default=[])
    parser.add_argument("--min-recall", type=float, default=0.995)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if not 0.0 <= args.min_recall <= 1.0:
        parser.error("--min-recall must be in [0, 1]")
    if not 0.0 <= args.max_false_positive_rate <= 1.0:
        parser.error("--max-false-positive-rate must be in [0, 1]")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive when set")
    return args


def run(args: argparse.Namespace) -> None:
    thresholds = args.threshold if args.threshold else _default_thresholds()
    summary = evaluate_thresholds(
        checkpoint=Path(args.checkpoint),
        labels=Path(args.labels),
        feature_manifest=Path(args.feature_manifest),
        output_dir=Path(args.output_dir),
        thresholds=thresholds,
        min_recall=args.min_recall,
        max_false_positive_rate=args.max_false_positive_rate,
        device=args.device,
        limit=args.limit,
    )
    selected = summary["recommendation"].get("selected") or {}
    print(f"summary={Path(args.output_dir) / 'threshold_eval_summary.json'}")
    print(f"recommendation_status={summary['recommendation']['status']}")
    if selected:
        print(f"selected_threshold={selected['threshold']}")
        print(f"selected_recall={selected['recall']:.6f}")
        print(f"selected_fpr={selected['false_positive_rate']:.6f}")


if __name__ == "__main__":
    run(parse_args())
