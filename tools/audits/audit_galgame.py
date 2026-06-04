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

from boundary.ja import audit_audio, sample_hf_audio_16k_mono


def _load_dataset(*, name: str, split: str):
    try:
        from datasets import Features, Value
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required for HF streaming audit: uv pip install datasets") from exc
    features = Features(
        {
            "ogg": Value("binary"),
            "txt": Value("string"),
            "__key__": Value("string"),
            "__url__": Value("string"),
        }
    )
    return load_dataset(name, split=split, streaming=True, features=features)


def _sample_audio(example: dict[str, Any]) -> tuple[np.ndarray, int]:
    return sample_hf_audio_16k_mono(example)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _bucket_counts(values: list[float], *, thresholds: list[float]) -> dict[str, int]:
    buckets: dict[str, int] = {}
    lower = 0.0
    for threshold in thresholds:
        key = f"{lower:g}-{threshold:g}"
        buckets[key] = sum(1 for value in values if lower <= value < threshold)
        lower = threshold
    buckets[f">={lower:g}"] = sum(1 for value in values if value >= lower)
    return buckets


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
    text_chars = [int(row["text_chars"]) for row in valid]
    rms_values = [float(row["rms_dbfs"]) for row in valid]
    head_silence_values = [float(row["head_silence_s"]) for row in valid]
    tail_silence_values = [float(row["tail_silence_s"]) for row in valid]
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "requested_limit": args.limit,
        "rows": len(rows),
        "valid_rows": len(valid),
        "error_rows": len(rows) - len(valid),
        "duration_s_total": sum(durations),
        "duration_s_min": min(durations) if durations else None,
        "duration_s_p50": _percentile(durations, 50),
        "duration_s_p90": _percentile(durations, 90),
        "duration_s_p99": _percentile(durations, 99),
        "duration_s_max": max(durations) if durations else None,
        "duration_s_mean": _mean(durations),
        "short_under_0_3s": sum(1 for value in durations if value < 0.3),
        "short_under_1s": sum(1 for value in durations if value < 1.0),
        "long_over_20s": sum(1 for value in durations if value > 20.0),
        "duration_buckets_s": _bucket_counts(durations, thresholds=[0.3, 1.0, 3.0, 10.0, 20.0]),
        "rms_dbfs_mean": _mean(rms_values),
        "rms_dbfs_p10": _percentile(rms_values, 10),
        "rms_dbfs_p50": _percentile(rms_values, 50),
        "head_silence_s_mean": _mean(head_silence_values),
        "tail_silence_s_mean": _mean(tail_silence_values),
        "text_chars_mean": _mean([float(value) for value in text_chars]),
        "text_chars_p50": _percentile([float(value) for value in text_chars], 50),
        "empty_text_rows": sum(1 for value in text_chars if value == 0),
    }
    summary_path = output_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"audit_summary={summary_path}")
    print(f"audit_samples={samples_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Galgame ASR samples for SpeechBoundary-JA research.")
    parser.add_argument("--dataset", default="litagin/Galgame_Speech_ASR_16kHz")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / "galgame-audit"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
