#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.fusionvad_ja import (  # noqa: E402
    effective_frame_weights,
    frame_classification_counts,
    load_cached_feature,
    load_label_records,
    metrics_from_frame_counts,
)

TOOLS_ROOT = PROJECT_ROOT / "tools" / "fusionvad_ja"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from export_addition_predictions import load_addition_model, probability_summary  # noqa: E402
from vad_recall_metrics import count_missed_speech_segments, padded_predictions  # noqa: E402


DEFAULT_FPS = "30000/1001"


def parse_fps(value: str) -> float:
    try:
        if "/" in value:
            fps = float(Fraction(value))
        else:
            fps = float(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(f"invalid fps: {value}") from exc
    if fps <= 0.0:
        raise argparse.ArgumentTypeError("--pad-video-fps must be positive")
    return fps


def parse_thresholds(values: list[str] | None) -> list[float]:
    thresholds: list[float] = []
    for raw in values or []:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            threshold = float(item)
            if threshold < 0.0:
                raise ValueError("thresholds must be non-negative")
            thresholds.append(threshold)
    return sorted(set(thresholds))


def build_threshold_grid(args: argparse.Namespace) -> list[float]:
    thresholds = parse_thresholds(args.threshold)
    if args.min_threshold is not None or args.max_threshold is not None or args.threshold_step is not None:
        if args.min_threshold is None or args.max_threshold is None or args.threshold_step is None:
            raise ValueError("--min-threshold, --max-threshold and --threshold-step must be provided together")
        if args.min_threshold < 0.0 or args.max_threshold < 0.0:
            raise ValueError("threshold range must be non-negative")
        if args.threshold_step <= 0.0:
            raise ValueError("--threshold-step must be positive")
        if args.max_threshold < args.min_threshold:
            raise ValueError("--max-threshold must be >= --min-threshold")
        count = int(round((args.max_threshold - args.min_threshold) / args.threshold_step))
        for index in range(count + 1):
            value = args.min_threshold + (index * args.threshold_step)
            if value <= args.max_threshold + (args.threshold_step / 2):
                thresholds.append(round(value, 10))
    if not thresholds:
        thresholds = [
            0.0,
            0.00002,
            0.00005,
            0.000075,
            0.0001,
            0.00015,
            0.0002,
            0.0003,
            0.0005,
            0.001,
            0.002,
            0.005,
            0.01,
        ]
    return sorted(set(float(value) for value in thresholds))


def load_probability_rows(
    *,
    labels_path: Path,
    feature_manifest_path: Path,
    checkpoint_path: Path,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    records = load_label_records(labels_path)
    feature_rows = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(feature_rows, list):
        raise ValueError("feature manifest must be a JSON list")

    model = load_addition_model(checkpoint_path=checkpoint_path, device=device)
    rows: list[dict[str, Any]] = []
    skipped = []
    all_probabilities: list[float] = []
    speech_probabilities: list[float] = []
    nonspeech_probabilities: list[float] = []

    with torch.inference_mode():
        for row_index, row in enumerate(feature_rows):
            label_index = int(row["label_index"])
            if label_index < 0 or label_index >= len(records):
                skipped.append({"row_index": row_index, "reason": "label_index_out_of_range"})
                continue
            record = records[label_index]
            try:
                whisper, mfcc = load_cached_feature(Path(str(row["feature_path"])))
            except Exception as exc:
                skipped.append(
                    {
                        "row_index": row_index,
                        "audio_id": str(row.get("audio_id") or record.audio_id),
                        "reason": "feature_load_error",
                        "error": str(exc),
                    }
                )
                continue

            weights = effective_frame_weights(record)
            frame_count = min(whisper.shape[0], mfcc.shape[0], len(record.speech_frames), len(weights))
            if frame_count <= 0:
                skipped.append(
                    {
                        "row_index": row_index,
                        "audio_id": str(row.get("audio_id") or record.audio_id),
                        "reason": "empty_frames",
                    }
                )
                continue

            whisper_tensor = torch.from_numpy(np.ascontiguousarray(whisper[:frame_count], dtype=np.float32)).to(
                device
            ).unsqueeze(0)
            mfcc_tensor = torch.from_numpy(np.ascontiguousarray(mfcc[:frame_count], dtype=np.float32)).to(
                device
            ).unsqueeze(0)
            logits = model(whisper_tensor, mfcc_tensor)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)[:frame_count]
            labels = np.asarray(record.speech_frames[:frame_count], dtype=np.int32)
            active_weights = np.asarray(weights[:frame_count], dtype=np.float32)
            active = active_weights > 0.0
            active_probabilities = probabilities[active]
            all_probabilities.extend(float(value) for value in active_probabilities)
            speech_probabilities.extend(float(value) for value in active_probabilities[labels[active] > 0])
            nonspeech_probabilities.extend(float(value) for value in active_probabilities[labels[active] <= 0])
            rows.append(
                {
                    "audio_id": record.audio_id,
                    "source": record.source,
                    "label_quality": record.label_quality,
                    "duration_s": record.duration_s,
                    "frame_hop_s": record.frame_hop_s,
                    "labels": labels,
                    "weights": active_weights,
                    "probabilities": probabilities,
                }
            )

    metadata = {
        "rows": len(rows),
        "skipped": len(skipped),
        "skipped_rows": skipped,
        "probability_summary": probability_summary(all_probabilities),
        "speech_label_probability_summary": probability_summary(speech_probabilities),
        "nonspeech_label_probability_summary": probability_summary(nonspeech_probabilities),
    }
    return rows, metadata


def compute_threshold_metrics(
    *,
    rows: list[dict[str, Any]],
    threshold: float,
    pad_frames: int,
) -> dict[str, Any]:
    raw_counts: Counter[str] = Counter()
    padded_counts: Counter[str] = Counter()
    missed_speech_frames = 0
    extra_audio_frames = 0
    missed_speech_segments = 0
    frame_hop_values: list[float] = []

    for row in rows:
        probabilities = np.asarray(row["probabilities"], dtype=np.float32)
        labels = np.asarray(row["labels"], dtype=np.int32)
        weights = np.asarray(row["weights"], dtype=np.float32)
        frame_count = min(probabilities.size, labels.size, weights.size)
        predictions = (probabilities[:frame_count] >= threshold).astype(np.int32)
        raw = frame_classification_counts(
            labels=labels[:frame_count],
            predictions=predictions[:frame_count],
            weights=weights[:frame_count],
        )
        raw_counts.update(raw)

        padded = np.asarray(
            padded_predictions(predictions[:frame_count].astype(int).tolist(), pad_frames=pad_frames),
            dtype=np.int32,
        )
        padded_count = frame_classification_counts(
            labels=labels[:frame_count],
            predictions=padded[:frame_count],
            weights=weights[:frame_count],
        )
        padded_counts.update(padded_count)
        missed_speech_frames += int(padded_count["false_negative"])
        extra_audio_frames += int(padded_count["false_positive"])
        missed_speech_segments += count_missed_speech_segments(
            labels=[int(value) for value in labels[:frame_count]],
            predictions=[int(value) for value in padded[:frame_count]],
        )
        frame_hop_values.append(float(row["frame_hop_s"]))

    raw_metrics = metrics_from_frame_counts(counts=dict(raw_counts), windows=len(rows), threshold=threshold)
    padded_metrics = metrics_from_frame_counts(counts=dict(padded_counts), windows=len(rows), threshold=threshold)
    frame_hop_s = frame_hop_values[0] if frame_hop_values else 0.02
    return {
        "threshold": float(threshold),
        "raw": {
            **asdict(raw_metrics),
            "counts": dict(raw_counts),
        },
        "padded": {
            **asdict(padded_metrics),
            "missed_speech_seconds": missed_speech_frames * frame_hop_s,
            "missed_speech_segments": missed_speech_segments,
            "extra_audio_seconds": extra_audio_frames * frame_hop_s,
            "extra_audio_ratio": (
                int(padded_counts.get("predicted_positives", 0)) / max(int(padded_counts.get("positives", 0)), 1)
            ),
            "counts": dict(padded_counts),
        },
    }


def choose_best(rows: list[dict[str, Any]], *, min_recall: float, max_extra_audio_ratio: float | None) -> dict[str, Any] | None:
    candidates = [row for row in rows if float(row["padded"]["recall"]) >= min_recall]
    if max_extra_audio_ratio is not None:
        candidates = [
            row for row in candidates if float(row["padded"]["extra_audio_ratio"]) <= max_extra_audio_ratio
        ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            float(row["padded"]["extra_audio_ratio"]),
            -float(row["padded"]["f1"]),
            -float(row["padded"]["precision"]),
            float(row["threshold"]),
        ),
    )[0]


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = build_threshold_grid(args)
    pad_s = float(args.pad_video_frames) / float(args.pad_video_fps)
    vad_pad_frames = int(round(pad_s / args.frame_hop_s))

    rows, metadata = load_probability_rows(
        labels_path=Path(args.labels),
        feature_manifest_path=Path(args.feature_manifest),
        checkpoint_path=Path(args.checkpoint),
        device=args.device,
    )
    sweep_rows = [
        compute_threshold_metrics(rows=rows, threshold=threshold, pad_frames=vad_pad_frames)
        for threshold in thresholds
    ]
    best = choose_best(
        sweep_rows,
        min_recall=args.min_recall,
        max_extra_audio_ratio=args.max_extra_audio_ratio,
    )
    summary = {
        "labels": args.labels,
        "feature_manifest": args.feature_manifest,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "thresholds": thresholds,
        "min_recall": args.min_recall,
        "max_extra_audio_ratio": args.max_extra_audio_ratio,
        "pad_video_frames": args.pad_video_frames,
        "pad_video_fps": args.pad_video_fps,
        "pad_s": pad_s,
        "frame_hop_s": args.frame_hop_s,
        "vad_pad_frames": vad_pad_frames,
        "operating_point": args.operating_point,
        "metadata": metadata,
        "best": best,
        "rows": sweep_rows,
    }
    summary_path = output_dir / "threshold_sweep_summary.json"
    jsonl_path = output_dir / "threshold_sweep_rows.jsonl"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in sweep_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"summary={summary_path}")
    print(f"rows={jsonl_path}")
    print(
        f"pad_video_frames={args.pad_video_frames:g} fps={args.pad_video_fps:.8f} "
        f"pad_s={pad_s:.4f} vad_pad_frames={vad_pad_frames}"
    )
    if best is None:
        print(f"best=none min_recall={args.min_recall:g}")
    else:
        print(
            f"best threshold={best['threshold']:g} recall={best['padded']['recall']:.4f} "
            f"precision={best['padded']['precision']:.4f} f1={best['padded']['f1']:.4f} "
            f"missed_speech_seconds={best['padded']['missed_speech_seconds']:.2f} "
            f"extra_audio_ratio={best['padded']['extra_audio_ratio']:.4f}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep FusionVAD-JA addition-BiLSTM thresholds with a high-recall operating-point target."
    )
    parser.add_argument("--labels", required=True, help="FusionVAD-JA label JSONL.")
    parser.add_argument("--feature-manifest", required=True, help="feature_manifest.json from build_feature_cache.py.")
    parser.add_argument("--checkpoint", required=True, help="FusionVAD-JA addition BiLSTM checkpoint.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", action="append", help="Threshold or comma-separated thresholds. Repeatable.")
    parser.add_argument("--min-threshold", type=float)
    parser.add_argument("--max-threshold", type=float)
    parser.add_argument("--threshold-step", type=float)
    parser.add_argument("--min-recall", type=float, default=0.98)
    parser.add_argument("--max-extra-audio-ratio", type=float)
    parser.add_argument(
        "--pad-video-frames",
        type=float,
        default=6.0,
        help="Pad by this many source video frames on each speech side.",
    )
    parser.add_argument(
        "--pad-video-fps",
        type=parse_fps,
        default=parse_fps(DEFAULT_FPS),
        help="Source video FPS used to convert --pad-video-frames to seconds. Supports 29.97 or 30000/1001.",
    )
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--operating-point", default="fusionvad-ja-threshold-sweep")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "threshold-sweep"),
    )
    args = parser.parse_args(argv)
    if args.min_recall < 0.0 or args.min_recall > 1.0:
        parser.error("--min-recall must be within [0, 1]")
    if args.max_extra_audio_ratio is not None and args.max_extra_audio_ratio < 0.0:
        parser.error("--max-extra-audio-ratio must be non-negative")
    if args.pad_video_frames < 0.0:
        parser.error("--pad-video-frames must be non-negative")
    if args.frame_hop_s <= 0.0:
        parser.error("--frame-hop-s must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
