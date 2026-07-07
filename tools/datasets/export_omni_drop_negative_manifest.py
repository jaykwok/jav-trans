#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def source_partition(video_id: str, *, heldout_percent: int) -> str:
    """Keep the compiled joint-dataset heldout videos out of hardmix train."""

    bucket = int(hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket >= heldout_percent:
        return "train"
    return "test" if bucket < heldout_percent // 2 else "val"


def export_rows(
    paths: Iterable[Path],
    *,
    min_confidence: float,
    min_duration_s: float,
    max_duration_s: float,
    heldout_percent: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    exported: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for path in paths:
        for row in _read_jsonl(path):
            counts["input"] += 1
            if str(row.get("label") or "") != "definite_drop":
                counts["skip_label"] += 1
                continue
            if not bool(row.get("training_label_included", True)):
                counts["skip_not_training"] += 1
                continue
            if bool(row.get("omni_semantic_speech_detected", False)):
                counts["skip_semantic_speech"] += 1
                continue
            confidence = float(row.get("omni_confidence") or 0.0)
            if confidence < min_confidence:
                counts["skip_confidence"] += 1
                continue
            duration_s = float(row.get("duration_s") or 0.0)
            if duration_s < min_duration_s or duration_s > max_duration_s:
                counts["skip_duration"] += 1
                continue
            audio = Path(str(row.get("audio") or ""))
            if not audio.is_file():
                counts["skip_missing_audio"] += 1
                continue
            candidate_id = str(row.get("candidate_id") or row.get("sample_id") or "")
            identity = candidate_id or str(audio.resolve())
            if identity in seen:
                counts["skip_duplicate"] += 1
                continue
            seen.add(identity)
            video_id = str(row.get("video_id") or row.get("window_id") or "")
            partition = source_partition(video_id, heldout_percent=heldout_percent)
            flags = sorted(str(flag) for flag in row.get("omni_flags") or [])
            exported.append(
                {
                    "audio_id": identity,
                    "audio": str(audio.resolve()),
                    "duration_s": duration_s,
                    "sample_rate": 16000,
                    "source": "omni_definite_drop",
                    "source_partition": partition,
                    "video_id": video_id,
                    "window_id": str(row.get("window_id") or ""),
                    "omni_confidence": confidence,
                    "omni_flags": flags,
                    "background_type": "+".join(flags) if flags else "omni_drop",
                    "label_source": str(row.get("label_source") or ""),
                }
            )
            counts["exported"] += 1
            counts[f"partition_{partition}"] += 1
            for flag in flags:
                counts[f"flag_{flag}"] += 1
    return exported, counts


def run(args: argparse.Namespace) -> None:
    rows, counts = export_rows(
        [Path(path) for path in args.labels],
        min_confidence=args.min_confidence,
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
        heldout_percent=args.heldout_percent,
    )
    if not rows:
        raise ValueError("no strict Omni definite_drop rows passed the filters")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema": "omni_drop_negative_manifest_v1",
        "labels": [str(path) for path in args.labels],
        "output": str(output),
        "count": len(rows),
        "filters": {
            "min_confidence": args.min_confidence,
            "min_duration_s": args.min_duration_s,
            "max_duration_s": args.max_duration_s,
            "heldout_percent": args.heldout_percent,
            "semantic_speech_detected": False,
            "label": "definite_drop",
        },
        "counts": dict(sorted(counts.items())),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export strict Omni definite_drop WAV clips as partition-safe hardmix "
            "negative units."
        )
    )
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--min-duration-s", type=float, default=0.08)
    parser.add_argument("--max-duration-s", type=float, default=12.0)
    parser.add_argument("--heldout-percent", type=int, default=20)
    args = parser.parse_args()
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be in [0, 1]")
    if args.min_duration_s <= 0.0:
        parser.error("--min-duration-s must be positive")
    if args.max_duration_s < args.min_duration_s:
        parser.error("--max-duration-s must be >= --min-duration-s")
    if not 2 <= args.heldout_percent <= 50:
        parser.error("--heldout-percent must be in [2, 50]")
    return args


if __name__ == "__main__":
    run(parse_args())
