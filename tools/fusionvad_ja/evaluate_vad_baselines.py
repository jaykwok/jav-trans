#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.fusionvad_ja import (
    effective_frame_weights,
    frame_classification_counts,
    get_research_vad_backend,
    load_label_records,
    load_manifest_audio_map,
    metrics_from_frame_counts,
    segments_to_frame_labels,
)
from vad.fusionvad_ja.manifest import build_training_examples


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_label_records(Path(args.labels))
    audio_map = load_manifest_audio_map(Path(args.manifest) if args.manifest else None)
    examples, skipped = build_training_examples(
        records,
        manifest_audio_map=audio_map,
        audio_root=Path(args.audio_root) if args.audio_root else None,
        extension_hints=args.extension,
        trainable_only=not args.include_non_trainable,
    )
    selected_examples = examples[: args.limit] if args.limit is not None else examples
    skipped_path = output_dir / "baseline_skipped.json"
    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "labels": args.labels,
        "manifest": args.manifest,
        "examples": len(examples),
        "selected_examples": len(selected_examples),
        "skipped": len(skipped),
        "backends": {},
    }
    for backend_name in args.backend:
        backend_dir = output_dir / backend_name.replace("/", "_")
        backend_dir.mkdir(parents=True, exist_ok=True)
        backend = get_research_vad_backend(backend_name)
        aggregate = Counter()
        rows = []
        errors = []
        threshold_override = args.threshold_override
        for index, example in enumerate(selected_examples):
            record = records[example.label_index]
            try:
                result = backend.segment(
                    example.audio_path,
                    target_sr=16000,
                    threshold_override=threshold_override,
                )
                predicted = segments_to_frame_labels(
                    result.segments,
                    duration_s=record.duration_s,
                    frame_hop_s=record.frame_hop_s,
                )
                counts = frame_classification_counts(
                    labels=record.speech_frames,
                    predictions=predicted,
                    weights=effective_frame_weights(record),
                )
                aggregate.update(counts)
                clip_metrics = metrics_from_frame_counts(
                    counts=counts,
                    windows=1,
                    threshold=float(threshold_override if threshold_override is not None else 0.5),
                )
                rows.append(
                    {
                        "audio_id": example.audio_id,
                        "source": example.source,
                        "label_quality": example.label_quality,
                        "duration_s": record.duration_s,
                        "frames": counts["frames"],
                        "speech_frame_count": counts["positives"],
                        "predicted_speech_frame_count": counts["predicted_positives"],
                        "segment_count": len(result.segments),
                        "processing_time_sec": result.processing_time_sec,
                        "precision": clip_metrics.precision,
                        "recall": clip_metrics.recall,
                        "f1": clip_metrics.f1,
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "index": index,
                        "audio_id": example.audio_id,
                        "audio_path": example.audio_path,
                        "label_index": example.label_index,
                        "error": str(exc),
                    }
                )
            print(
                f"backend={backend_name} processed={index + 1}/{len(selected_examples)} "
                f"ok={len(rows)} errors={len(errors)}",
                flush=True,
            )

        if not rows:
            raise RuntimeError(f"backend {backend_name} produced no successful evaluations")
        metrics_path = backend_dir / "eval_metrics.json"
        metrics = metrics_from_frame_counts(
            counts=dict(aggregate),
            windows=len(rows),
            metrics_path=metrics_path,
            threshold=float(threshold_override if threshold_override is not None else 0.5),
        )
        metrics_payload = asdict(metrics)
        metrics_payload["backend"] = backend_name
        metrics_payload["successful_examples"] = len(rows)
        metrics_payload["errors"] = len(errors)
        metrics_payload["signature"] = backend.signature()
        metrics_path.write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        rows_path = backend_dir / "per_clip_metrics.jsonl"
        with rows_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        errors_path = backend_dir / "errors.json"
        errors_path.write_text(
            json.dumps(errors, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary["backends"][backend_name] = {
            "metrics": str(metrics_path),
            "per_clip_metrics": str(rows_path),
            "errors": str(errors_path),
            "f1": metrics.f1,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "predicted_positive_ratio": metrics.predicted_positive_ratio,
            "successful_examples": len(rows),
            "error_count": len(errors),
        }
        print(
            f"backend={backend_name} f1={metrics.f1:.4f} "
            f"precision={metrics.precision:.4f} recall={metrics.recall:.4f} "
            f"predicted_positive_ratio={metrics.predicted_positive_ratio:.4f}",
            flush=True,
        )

    summary_path = output_dir / "baseline_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"summary={summary_path}")
    print(f"skipped={skipped_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing VAD baselines on FusionVAD-JA labels.")
    parser.add_argument("--labels", required=True, help="FusionVAD-JA label JSONL.")
    parser.add_argument("--manifest", help="JSON manifest containing audio paths.")
    parser.add_argument("--audio-root", help="Optional directory for resolving audio_id + extension.")
    parser.add_argument("--extension", action="append", default=[".wav", ".flac", ".ogg", ".mp3", ".m4a"])
    parser.add_argument("--include-non-trainable", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", action="append")
    parser.add_argument("--threshold-override", type=float)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "vad-baselines"),
    )
    args = parser.parse_args(argv)
    if args.backend is None:
        args.backend = ["fusion_lite", "whisperseg-adaptive"]
    return args


if __name__ == "__main__":
    run(parse_args())
