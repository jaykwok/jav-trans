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
    score_feature_frame_boundary_probabilities,
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
    min_speech_recall: float = 0.995,
    max_speech_false_positive_rate: float = 0.02,
    min_cut_recall: float = 0.90,
    max_cut_false_positive_rate: float = 0.10,
    cut_min_gap_s: float = 0.5,
    cut_boundary_radius_frames: int = 1,
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
    speech_counts = {threshold: _empty_counts() for threshold in threshold_values}
    cut_counts = {threshold: _empty_counts() for threshold in threshold_values}
    quality_counts: dict[str, dict[str, dict[float, dict[str, int]]]] = {}
    source_counts = Counter()
    label_quality_counts = Counter()
    skipped: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        try:
            label_index = int(row["label_index"])
            record = records[label_index]
            ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
            speech_scores, cut_scores = score_feature_frame_boundary_probabilities(bundle, ptm=ptm, mfcc=mfcc)
            weights = np.asarray(effective_frame_weights(record), dtype=np.float32)
            frame_total = min(speech_scores.shape[0], cut_scores.shape[0], len(record.speech_frames), weights.shape[0])
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
            speech_labels = np.asarray(record.speech_frames[:frame_total], dtype=np.float32)[mask]
            cut_labels = np.maximum(cut_drops[:frame_total], cut_points[:frame_total]).astype(np.float32)[mask]
            speech_scores_array = speech_scores[:frame_total][mask]
            cut_scores_array = cut_scores[:frame_total][mask]
            quality = str(record.label_quality or row.get("label_quality") or "")
            quality_counts.setdefault(
                quality,
                {
                    "speech": {threshold: _empty_counts() for threshold in threshold_values},
                    "cut": {threshold: _empty_counts() for threshold in threshold_values},
                },
            )
            source_counts[str(record.source)] += 1
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
        except Exception as exc:  # pragma: no cover - exercised by real corrupt manifests
            skipped.append(
                {
                    "row_index": row_index,
                    "label_index": row.get("label_index"),
                    "reason": "error",
                    "error": str(exc),
                }
            )

    speech_metrics = [_metrics_from_counts(speech_counts[threshold], threshold=threshold) for threshold in threshold_values]
    cut_metrics = [_metrics_from_counts(cut_counts[threshold], threshold=threshold) for threshold in threshold_values]
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
        "rows": len(rows),
        "skipped": len(skipped),
        "skipped_samples": skipped[:20],
        "label_quality_counts": dict(sorted(label_quality_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "thresholds": threshold_values,
        "speech_metrics": speech_metrics,
        "cut_metrics": cut_metrics,
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
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    speech_recommendation = dict(summary.get("recommendation", {}).get("speech") or {})
    cut_recommendation = dict(summary.get("recommendation", {}).get("cut") or {})
    speech_selected = dict(speech_recommendation.get("selected") or {})
    cut_selected = dict(cut_recommendation.get("selected") or {})
    lines = [
        "# SpeechBoundary-JA Mamba2 Frame Boundary Scorer Threshold Eval",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Skipped: `{summary['skipped']}`",
        f"- Speech recommendation status: `{speech_recommendation.get('status', '')}`",
        f"- Cut recommendation status: `{cut_recommendation.get('status', '')}`",
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
    )
    speech_selected = summary["recommendation"]["speech"].get("selected") or {}
    cut_selected = summary["recommendation"]["cut"].get("selected") or {}
    print(f"summary={Path(args.output_dir) / 'threshold_eval_summary.json'}")
    print(f"speech_recommendation_status={summary['recommendation']['speech']['status']}")
    print(f"cut_recommendation_status={summary['recommendation']['cut']['status']}")
    if speech_selected:
        print(f"selected_speech_threshold={speech_selected['threshold']}")
        print(f"selected_speech_recall={speech_selected['recall']:.6f}")
        print(f"selected_speech_fpr={speech_selected['false_positive_rate']:.6f}")
    if cut_selected:
        print(f"selected_cut_threshold={cut_selected['threshold']}")
        print(f"selected_cut_recall={cut_selected['recall']:.6f}")
        print(f"selected_cut_fpr={cut_selected['false_positive_rate']:.6f}")


if __name__ == "__main__":
    run(parse_args())
