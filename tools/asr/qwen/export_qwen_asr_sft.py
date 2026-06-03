#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_manifest_rows(paths: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_text in paths:
        path = Path(path_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"manifest must be a JSON list: {path}")
        for row in payload:
            if isinstance(row, Mapping):
                copied = dict(row)
                copied["_manifest"] = str(path)
                rows.append(copied)
    return rows


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def should_keep_row(row: Mapping[str, Any], *, min_duration_s: float, max_duration_s: float) -> tuple[bool, str]:
    audio_path = row.get("audio")
    if not audio_path:
        return False, "missing_audio"
    if not Path(str(audio_path)).exists():
        return False, "audio_not_found"
    text = normalized_text(row.get("text"))
    if not text:
        return False, "missing_text"
    try:
        duration_s = float(row.get("duration_s") or 0.0)
    except (TypeError, ValueError):
        duration_s = 0.0
    if duration_s < min_duration_s:
        return False, "duration_too_short"
    if max_duration_s > 0 and duration_s > max_duration_s:
        return False, "duration_too_long"
    return True, ""


def target_audio_path(*, row: Mapping[str, Any], audio_dir: Path, copy_audio: bool) -> str:
    source = Path(str(row["audio"]))
    if not copy_audio:
        return str(source)
    suffix = source.suffix or ".wav"
    audio_id = str(row.get("audio_id") or source.stem)
    target = audio_dir / f"{audio_id}{suffix}"
    if not target.exists():
        shutil.copy2(source, target)
    return str(target)


def export_rows(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    split: str,
    copy_audio: bool,
    min_duration_s: float,
    max_duration_s: float,
    limit: int | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    if copy_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / f"{split}.jsonl"
    skipped_path = output_dir / f"{split}_skipped.json"
    summary_path = output_dir / f"{split}_summary.json"

    kept = 0
    skipped: list[dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    total_duration_s = 0.0
    total_text_chars = 0

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            keep, reason = should_keep_row(
                row,
                min_duration_s=min_duration_s,
                max_duration_s=max_duration_s,
            )
            if not keep:
                skip_counts[reason] += 1
                skipped.append(
                    {
                        "audio_id": row.get("audio_id"),
                        "audio": row.get("audio"),
                        "text": row.get("text"),
                        "reason": reason,
                    }
                )
                continue
            if limit is not None and kept >= limit:
                skip_counts["limit"] += 1
                continue

            audio_path = target_audio_path(row=row, audio_dir=audio_dir, copy_audio=copy_audio)
            text = normalized_text(row.get("text"))
            duration_s = float(row.get("duration_s") or 0.0)
            payload = {
                "audio": audio_path,
                "text": text,
                "language": "Japanese",
                "audio_id": str(row.get("audio_id") or Path(audio_path).stem),
                "source": str(row.get("source") or ""),
                "duration_s": duration_s,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            kept += 1
            total_duration_s += duration_s
            total_text_chars += len(text)

    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "split": split,
        "output_jsonl": str(output_jsonl),
        "copy_audio": copy_audio,
        "records": len(rows),
        "kept": kept,
        "skipped": len(skipped),
        "skip_counts": dict(sorted(skip_counts.items())),
        "total_duration_s": total_duration_s,
        "total_text_chars": total_text_chars,
        "avg_duration_s": total_duration_s / kept if kept else 0.0,
        "avg_text_chars": total_text_chars / kept if kept else 0.0,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def run(args: argparse.Namespace) -> None:
    rows = load_manifest_rows(args.manifest)
    summary = export_rows(
        rows,
        output_dir=Path(args.output_dir),
        split=args.split,
        copy_audio=args.copy_audio,
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
        limit=args.limit,
    )
    print(f"output_jsonl={summary['output_jsonl']}")
    print(f"kept={summary['kept']} skipped={summary['skipped']} duration_s={summary['total_duration_s']:.2f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Galgame audio/text rows for Qwen3-ASR SFT packaging.")
    parser.add_argument("--manifest", action="append", required=True, help="Materialized HF audio manifest JSON.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copy-audio", action="store_true", help="Copy audio into output-dir/audio for upload.")
    parser.add_argument("--min-duration-s", type=float, default=0.5)
    parser.add_argument("--max-duration-s", type=float, default=30.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
