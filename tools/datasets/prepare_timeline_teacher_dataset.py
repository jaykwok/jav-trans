#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.asr.cueqc.label_pre_asr_with_omni import slice_audio_clip  # noqa: E402


SCHEMA = "timeline_teacher_item_v1"
HELD_OUT_SOURCE_IDS = frozenset({"FJIN-059", "NAMH-055"})


def validate_partition(*, source_id: str, split: str) -> None:
    normalized_source = source_id.strip().upper()
    if any(normalized_source.startswith(item) for item in HELD_OUT_SOURCE_IDS):
        if split.strip().lower() != "heldout":
            raise ValueError(f"{source_id} is a held-out gate and must use split=heldout")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _item_id(source_id: str, segment: dict[str, Any]) -> str:
    chunk_index = int(segment.get("source_chunk_index") or 0)
    start = float(segment.get("chunk_acoustic_start", segment["start"]))
    end = float(segment.get("chunk_acoustic_end", segment["end"]))
    digest = hashlib.sha256(
        f"{source_id}:{chunk_index}:{start:.6f}:{end:.6f}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{source_id}-chunk{chunk_index:05d}-{digest}"


def select_segments(
    segments: list[dict[str, Any]],
    *,
    limit: int,
    long_threshold_s: float,
    seed: int,
) -> list[dict[str, Any]]:
    usable = [segment for segment in segments if str(segment.get("text") or "").strip()]
    ordered = sorted(
        usable,
        key=lambda segment: (
            -float(segment.get("chunk_acoustic_duration", 0.0)),
            int(segment.get("source_chunk_index") or 0),
        ),
    )
    long_rows = [
        segment
        for segment in ordered
        if float(segment.get("chunk_acoustic_duration", 0.0)) > long_threshold_s
    ]
    if limit <= 0:
        return sorted(usable, key=lambda item: int(item.get("source_chunk_index") or 0))
    selected = long_rows[:limit]
    if len(selected) < limit:
        used = {int(item.get("source_chunk_index") or 0) for item in selected}
        remainder = [
            item
            for item in usable
            if int(item.get("source_chunk_index") or 0) not in used
        ]
        random.Random(seed).shuffle(remainder)
        selected.extend(remainder[: limit - len(selected)])
    return sorted(selected, key=lambda item: int(item.get("source_chunk_index") or 0))


def prepare(
    *,
    aligned_segments_path: Path,
    output_dir: Path,
    source_id: str,
    split: str,
    limit: int,
    long_threshold_s: float,
    seed: int,
) -> dict[str, Any]:
    validate_partition(source_id=source_id, split=split)
    aligned = _read_json(aligned_segments_path)
    source_audio = Path(str(aligned["audio_path"]))
    if not source_audio.is_absolute():
        source_audio = PROJECT_ROOT / source_audio
    if not source_audio.exists():
        raise FileNotFoundError(source_audio)
    segments = select_segments(
        [dict(item) for item in aligned.get("segments") or []],
        limit=limit,
        long_threshold_s=long_threshold_s,
        seed=seed,
    )
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for segment in segments:
        start = float(segment.get("chunk_acoustic_start", segment["start"]))
        end = float(segment.get("chunk_acoustic_end", segment["end"]))
        item_id = _item_id(source_id, segment)
        audio_path = audio_dir / f"{item_id}.wav"
        slice_audio_clip(
            source_audio=source_audio,
            row={"start": start, "end": end, "duration_s": end - start},
            output_path=audio_path,
            fmt="wav",
            bitrate="",
            sample_rate=16000,
            force=False,
        )
        rows.append(
            {
                "schema": SCHEMA,
                "item_id": item_id,
                "source_id": source_id,
                "split": split,
                "source_chunk_index": int(segment.get("source_chunk_index") or 0),
                "absolute_start_s": start,
                "absolute_end_s": end,
                "duration_s": end - start,
                "transcript": str(segment.get("text") or "").strip(),
                "audio_path": str(audio_path),
                "alignment_mode": str(segment.get("alignment_mode") or ""),
            }
        )
    items_path = output_dir / "items.jsonl"
    items_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema": "timeline_teacher_dataset_summary_v1",
        "source": str(aligned_segments_path),
        "source_audio": str(source_audio),
        "source_id": source_id,
        "split": split,
        "item_count": len(rows),
        "long_item_count": sum(row["duration_s"] > long_threshold_s for row in rows),
        "long_threshold_s": long_threshold_s,
        "items": str(items_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-segments", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--long-threshold-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()
    summary = prepare(
        aligned_segments_path=Path(args.aligned_segments),
        output_dir=Path(args.output_dir),
        source_id=args.source_id,
        split=args.split,
        limit=max(0, args.limit),
        long_threshold_s=max(0.0, args.long_threshold_s),
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
