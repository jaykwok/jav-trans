#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.fusionvad_ja import frame_classification_counts, load_label_records, metrics_from_frame_counts


def padded_predictions(predictions: list[int], *, pad_frames: int) -> list[int]:
    if pad_frames <= 0:
        return [int(value) for value in predictions]
    result = [0] * len(predictions)
    active = [index for index, value in enumerate(predictions) if int(value)]
    for index in active:
        start = max(0, index - pad_frames)
        end = min(len(result), index + pad_frames + 1)
        for offset in range(start, end):
            result[offset] = 1
    return result


def run(args: argparse.Namespace) -> None:
    labels = load_label_records(Path(args.labels))
    prediction_rows = [
        json.loads(line)
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_audio_id = {str(row.get("audio_id")): row for row in prediction_rows}
    pad_frames = max(0, int(round(args.pad_s / args.frame_hop_s)))
    total_counts = {
        "frames": 0,
        "correct": 0,
        "positives": 0,
        "predicted_positives": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    missed_speech_frames = 0
    extra_audio_frames = 0
    evaluated = 0
    skipped = []
    for record in labels:
        row = by_audio_id.get(record.audio_id)
        if row is None:
            skipped.append({"audio_id": record.audio_id, "reason": "missing_prediction"})
            continue
        raw_predictions = row.get("speech_frames") or row.get("predictions")
        if not isinstance(raw_predictions, list):
            skipped.append({"audio_id": record.audio_id, "reason": "missing_speech_frames"})
            continue
        predictions = padded_predictions([int(value) for value in raw_predictions], pad_frames=pad_frames)
        frame_total = min(len(record.speech_frames), len(predictions))
        counts = frame_classification_counts(
            labels=record.speech_frames[:frame_total],
            predictions=predictions[:frame_total],
        )
        for key, value in counts.items():
            total_counts[key] += int(value)
        missed_speech_frames += int(counts["false_negative"])
        extra_audio_frames += int(counts["false_positive"])
        evaluated += 1

    metrics = metrics_from_frame_counts(counts=total_counts, windows=evaluated)
    summary = {
        **asdict(metrics),
        "labels": args.labels,
        "predictions": args.predictions,
        "evaluated": evaluated,
        "skipped": len(skipped),
        "pad_s": args.pad_s,
        "frame_hop_s": args.frame_hop_s,
        "missed_speech_seconds": missed_speech_frames * args.frame_hop_s,
        "extra_audio_seconds": extra_audio_frames * args.frame_hop_s,
        "extra_audio_ratio": (
            total_counts["predicted_positives"] / max(total_counts["positives"], 1)
        ),
        "counts": total_counts,
        "skipped_rows": skipped,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"summary={output_path}")
    print(
        f"recall={summary['recall']:.4f} missed_speech_seconds={summary['missed_speech_seconds']:.2f} "
        f"extra_audio_ratio={summary['extra_audio_ratio']:.4f}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute high-recall VAD metrics from frame predictions.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True, help="JSONL rows with audio_id and speech_frames/predictions.")
    parser.add_argument("--pad-s", type=float, default=0.0)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
