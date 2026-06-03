#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.fusionvad_ja import (
    TrainConfig,
    build_training_examples,
    load_label_records,
    load_manifest_audio_map,
    train_tiny_frame_classifier,
    write_training_manifest,
)


def run(args: argparse.Namespace) -> None:
    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else None
    audio_root = Path(args.audio_root) if args.audio_root else None
    records = load_label_records(labels_path)
    audio_map = load_manifest_audio_map(manifest_path)
    examples, skipped = build_training_examples(
        records,
        manifest_audio_map=audio_map,
        audio_root=audio_root,
        extension_hints=args.extension,
        trainable_only=not args.include_non_trainable,
    )
    training_manifest_path = output_dir / "training_manifest.jsonl"
    skipped_path = output_dir / "training_manifest_skipped.json"
    write_training_manifest(path=training_manifest_path, examples=examples)
    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metrics = train_tiny_frame_classifier(
        records=records,
        examples=examples,
        output_dir=output_dir,
        config=TrainConfig(
            window_s=args.window_s,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        ),
    )
    print(f"training_manifest={training_manifest_path}")
    print(f"skipped_report={skipped_path}")
    print(f"checkpoint={metrics.checkpoint}")
    print(f"metrics={metrics.metrics_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny FusionVAD-JA research frame classifier.")
    parser.add_argument("--labels", required=True, help="FusionVAD-JA label JSONL.")
    parser.add_argument("--manifest", help="Optional JSON manifest containing audio paths.")
    parser.add_argument("--audio-root", help="Optional directory for resolving audio_id + extension.")
    parser.add_argument("--extension", action="append", default=[".wav", ".flac", ".ogg", ".mp3", ".m4a"])
    parser.add_argument("--include-non-trainable", action="store_true")
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "tiny-train"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
