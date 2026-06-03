#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.fusionvad_ja import (
    build_training_examples,
    dry_run_batches,
    load_label_records,
    load_manifest_audio_map,
    write_dry_run,
    write_training_manifest,
)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = Path(args.labels)
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
    output_manifest_path = output_dir / args.output_jsonl
    skipped_path = output_dir / "training_manifest_skipped.json"
    dry_run_path = output_dir / "training_manifest_dry_run.json"
    summary_path = output_dir / "training_manifest_summary.json"
    write_training_manifest(path=output_manifest_path, examples=examples)
    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    batches = []
    if args.dry_run_batches > 0:
        batches = dry_run_batches(
            records,
            examples,
            window_s=args.window_s,
            max_batches=args.dry_run_batches,
        )
        write_dry_run(dry_run_path, batches)
    summary = {
        "labels": str(labels_path),
        "source_manifest": str(manifest_path) if manifest_path else None,
        "audio_root": str(audio_root) if audio_root else None,
        "records": len(records),
        "examples": len(examples),
        "skipped": len(skipped),
        "skipped_reasons": dict(sorted(Counter(row.get("reason", "unknown") for row in skipped).items())),
        "label_quality_counts": dict(sorted(Counter(example.label_quality for example in examples).items())),
        "source_counts": dict(sorted(Counter(example.source for example in examples).items())),
        "dry_run_batches": len(batches),
        "output_manifest": str(output_manifest_path),
        "skipped_report": str(skipped_path),
        "dry_run_report": str(dry_run_path) if args.dry_run_batches > 0 else None,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"training_manifest={output_manifest_path}")
    print(f"skipped_report={skipped_path}")
    if args.dry_run_batches > 0:
        print(f"dry_run_report={dry_run_path}")
    print(f"summary={summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and dry-run a FusionVAD-JA training manifest.")
    parser.add_argument("--labels", required=True, help="FusionVAD-JA label JSONL.")
    parser.add_argument("--manifest", help="Optional JSON manifest containing audio paths.")
    parser.add_argument("--audio-root", help="Optional directory for resolving audio_id + extension.")
    parser.add_argument("--extension", action="append", default=[".wav", ".flac", ".ogg", ".mp3", ".m4a"])
    parser.add_argument("--include-non-trainable", action="store_true")
    parser.add_argument("--dry-run-batches", type=int, default=4)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "training-manifest"))
    parser.add_argument("--output-jsonl", default="training_manifest.jsonl")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
