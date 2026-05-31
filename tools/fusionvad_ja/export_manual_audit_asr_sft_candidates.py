#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def read_manifest_rows(paths: Iterable[str]) -> list[dict[str, Any]]:
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


def parse_candidate_asr_text(value: Any) -> str:
    lines = str(value or "").splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ASR:"):
            return normalized_text(stripped[4:])
    return normalized_text(lines[0] if lines else "")


def resolve_audio_path(value: Any) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def safe_audio_name(row: Mapping[str, Any], audio_path: Path) -> str:
    audio_id = str(row.get("audio_id") or audio_path.stem or "audio")
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in audio_id)
    suffix = audio_path.suffix or ".wav"
    return f"{safe}{suffix}"


def exported_audio_path(*, row: Mapping[str, Any], audio_dir: Path, copy_audio: bool) -> str:
    source = resolve_audio_path(row.get("audio"))
    if not copy_audio:
        return str(source)
    audio_dir.mkdir(parents=True, exist_ok=True)
    target = audio_dir / safe_audio_name(row, source)
    if source.exists() and not target.exists():
        shutil.copy2(source, target)
    return str(target)


def duration_s(row: Mapping[str, Any]) -> float:
    try:
        return float(row.get("duration_s") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def export_candidates(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    split: str,
    copy_audio: bool,
    require_audio_exists: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    empty_sft_path = output_dir / f"{split}_empty_hard_negative.jsonl"
    speech_review_path = output_dir / f"{split}_speech_review.jsonl"
    speech_review_csv_path = output_dir / f"{split}_speech_review.csv"
    skipped_path = output_dir / f"{split}_skipped.json"
    summary_path = output_dir / f"{split}_summary.json"

    empty_rows: list[dict[str, Any]] = []
    speech_review_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for row in rows:
        label_quality = str(row.get("label_quality") or "")
        label_counts[label_quality] += 1
        audio_path = resolve_audio_path(row.get("audio"))
        audio_id = str(row.get("audio_id") or audio_path.stem)
        if not row.get("audio"):
            skip_counts["missing_audio"] += 1
            skipped.append({"audio_id": audio_id, "reason": "missing_audio"})
            continue
        if require_audio_exists and not audio_path.exists():
            skip_counts["audio_not_found"] += 1
            skipped.append({"audio_id": audio_id, "audio": str(audio_path), "reason": "audio_not_found"})
            continue

        exported_audio = exported_audio_path(row=row, audio_dir=audio_dir, copy_audio=copy_audio)
        common = {
            "audio": exported_audio,
            "audio_id": audio_id,
            "duration_s": duration_s(row),
            "language": "Japanese",
            "source": str(row.get("source") or ""),
            "manual_reason": str(row.get("manual_reason") or ""),
        }
        if label_quality == "negative":
            empty_rows.append(
                {
                    **common,
                    "text": "",
                    "label_type": "empty_hard_negative",
                }
            )
        elif label_quality == "supervised":
            speech_review_rows.append(
                {
                    **common,
                    "candidate_asr_text": parse_candidate_asr_text(row.get("text")),
                    "original_text_field": str(row.get("text") or ""),
                    "label_type": "needs_manual_transcript",
                    "text": "",
                }
            )
        else:
            skip_counts["unsupported_label_quality"] += 1
            skipped.append({"audio_id": audio_id, "label_quality": label_quality, "reason": "unsupported_label_quality"})

    write_jsonl(empty_sft_path, empty_rows)
    write_jsonl(speech_review_path, speech_review_rows)
    write_speech_review_csv(speech_review_csv_path, speech_review_rows)
    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "records": len(rows),
        "label_quality_counts": dict(sorted(label_counts.items())),
        "empty_hard_negative_records": len(empty_rows),
        "speech_review_records": len(speech_review_rows),
        "skipped": len(skipped),
        "skip_counts": dict(sorted(skip_counts.items())),
        "copy_audio": copy_audio,
        "empty_hard_negative_jsonl": str(empty_sft_path),
        "speech_review_jsonl": str(speech_review_path),
        "speech_review_csv": str(speech_review_csv_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_speech_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "audio_id",
        "audio",
        "duration_s",
        "manual_reason",
        "candidate_asr_text",
        "text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    rows = read_manifest_rows(args.manifest)
    summary = export_candidates(
        rows,
        output_dir=Path(args.output_dir),
        split=args.split,
        copy_audio=args.copy_audio,
        require_audio_exists=args.require_audio_exists,
    )
    print(f"empty_hard_negative_jsonl={summary['empty_hard_negative_jsonl']}")
    print(f"speech_review_jsonl={summary['speech_review_jsonl']}")
    print(
        f"empty={summary['empty_hard_negative_records']} "
        f"speech_review={summary['speech_review_records']} skipped={summary['skipped']}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export manual-audit ASR SFT candidates without trusting ASR text as positive labels."
    )
    parser.add_argument("--manifest", action="append", required=True, help="Manual-audit strong manifest JSON.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copy-audio", action="store_true")
    parser.add_argument("--require-audio-exists", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
