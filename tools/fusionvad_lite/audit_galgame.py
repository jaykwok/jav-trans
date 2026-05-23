#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from vad.fusionvad_lite import audit_audio


def _load_dataset(*, name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required for HF streaming audit: uv pip install datasets") from exc
    return load_dataset(name, split=split, streaming=True)


def _sample_audio(example: dict[str, Any]) -> tuple[np.ndarray, int]:
    audio_obj = example.get("ogg") or example.get("audio")
    if isinstance(audio_obj, dict):
        array = audio_obj.get("array")
        sample_rate = int(audio_obj.get("sampling_rate") or 16000)
        if array is None:
            raise ValueError("audio sample has no array")
        return np.asarray(array, dtype=np.float32), sample_rate
    raise ValueError("expected an 'ogg' or 'audio' field decoded by datasets")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(name=args.dataset, split=args.split)
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(dataset):
        if index >= args.limit:
            break
        try:
            audio, sample_rate = _sample_audio(example)
            text = str(example.get("txt") or example.get("text") or "")
            audio_id = str(example.get("__key__") or example.get("id") or index)
            audit = audit_audio(
                audio_id=audio_id,
                source=args.dataset,
                audio=audio,
                sample_rate=sample_rate,
                text=text,
            )
            rows.append(asdict(audit))
        except Exception as exc:
            rows.append(
                {
                    "audio_id": str(example.get("__key__") or index),
                    "source": args.dataset,
                    "error": str(exc),
                }
            )

    samples_path = output_dir / "audit_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    valid = [row for row in rows if "error" not in row]
    durations = [float(row["duration_s"]) for row in valid]
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "requested_limit": args.limit,
        "rows": len(rows),
        "valid_rows": len(valid),
        "error_rows": len(rows) - len(valid),
        "duration_s_total": sum(durations),
        "duration_s_min": min(durations) if durations else None,
        "duration_s_max": max(durations) if durations else None,
        "duration_s_mean": sum(durations) / len(durations) if durations else None,
        "short_under_0_3s": sum(1 for value in durations if value < 0.3),
        "long_over_20s": sum(1 for value in durations if value > 20.0),
    }
    summary_path = output_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"audit_summary={summary_path}")
    print(f"audit_samples={samples_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Galgame ASR samples for FusionVAD-Lite research.")
    parser.add_argument("--dataset", default="litagin/Galgame_Speech_ASR_16kHz")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-lite" / "galgame-audit"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
