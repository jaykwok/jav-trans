#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.ja import build_negative_record, write_jsonl


def synthesize_negative(index: int, *, samples: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    mode = index % 4
    if mode == 0:
        return np.zeros(samples, dtype=np.float32)
    if mode == 1:
        return rng.normal(0.0, 0.015, samples).astype(np.float32)
    if mode == 2:
        time = np.arange(samples, dtype=np.float32) / sample_rate
        freq = 110.0 + 17.0 * (index % 7)
        return (0.04 * np.sin(2.0 * np.pi * freq * time)).astype(np.float32)
    base = rng.normal(0.0, 0.01, samples).astype(np.float32)
    envelope = np.linspace(0.2, 1.0, samples, dtype=np.float32)
    return (base * envelope).astype(np.float32)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    manifest_rows = []
    samples = max(1, int(round(args.duration_s * args.sample_rate)))
    for index in range(args.count):
        audio_id = f"{args.audio_id_prefix}-{index:06d}"
        audio = synthesize_negative(index, samples=samples, sample_rate=args.sample_rate, rng=rng)
        audio_path = audio_dir / f"{audio_id}.wav"
        sf.write(str(audio_path), audio, args.sample_rate)
        record = build_negative_record(
            audio_id=audio_id,
            source=args.source,
            duration_s=samples / args.sample_rate,
            frame_hop_s=args.frame_hop_s,
        )
        records.append(record)
        manifest_rows.append(
            {
                "audio_id": audio_id,
                "audio": str(audio_path),
                "duration_s": record.duration_s,
                "input": f"synthetic:{index}",
                "label_quality": record.label_quality,
                "source": args.source,
            }
        )

    labels_path = output_dir / args.output_jsonl
    manifest_path = output_dir / "synthetic_negative_manifest.json"
    summary_path = output_dir / "synthetic_negative_summary.json"
    write_jsonl(labels_path, records)
    manifest_path.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "records": len(records),
        "duration_s_total": sum(record.duration_s for record in records),
        "label_quality_counts": dict(sorted(Counter(record.label_quality for record in records).items())),
        "labels": str(labels_path),
        "manifest": str(manifest_path),
        "audio_dir": str(audio_dir),
        "seed": args.seed,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"negative_labels={labels_path}")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic synthetic negative labels for SpeechBoundary-JA smoke tests.")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--source", default="synthetic-negative")
    parser.add_argument("--audio-id-prefix", default="synthetic-negative")
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / "synthetic-negatives"))
    parser.add_argument("--output-jsonl", default="synthetic_negative_labels.jsonl")
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")
    if args.sample_rate <= 0:
        parser.error("--sample-rate must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
