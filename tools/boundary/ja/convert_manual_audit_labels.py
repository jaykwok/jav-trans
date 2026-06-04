#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.ja import build_negative_record, build_supervised_record, write_jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
    return rows


def normalize_segments(row: Mapping[str, Any]) -> list[dict[str, float]]:
    duration_s = float(row.get("duration_s") or 0.0)
    out: list[dict[str, float]] = []
    for item in list(row.get("speech_segments") or []):
        if not isinstance(item, Mapping):
            continue
        try:
            has_start = item.get("start") not in (None, "")
            has_end = item.get("end") not in (None, "")
            if not has_start and not has_end:
                continue
            start = max(0.0, min(float(item.get("start") if has_start else 0.0), duration_s))
            end = max(0.0, min(float(item.get("end") if has_end else duration_s), duration_s))
        except (TypeError, ValueError):
            continue
        if end > start:
            out.append({"start": start, "end": end})
    out.sort(key=lambda segment: segment["start"])
    merged: list[dict[str, float]] = []
    for segment in out:
        last = merged[-1] if merged else None
        if last and segment["start"] <= last["end"]:
            last["end"] = max(last["end"], segment["end"])
        else:
            merged.append(segment)
    return merged


def row_audio_path(row: Mapping[str, Any]) -> str:
    audio = str(row.get("audio") or "")
    if not audio:
        return ""
    path = Path(audio)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(input_path)

    records = []
    manifest_rows = []
    skipped = []
    manual_speech_s = 0.0
    for index, row in enumerate(rows):
        audio_id = str(row.get("audio_id") or "")
        duration_s = float(row.get("duration_s") or 0.0)
        source = str(row.get("source") or "")
        text = str(row.get("text") or "")
        skip_reason = str(row.get("skip_reason") or "").strip()
        reviewed = bool(row.get("reviewed"))
        segments = normalize_segments(row)
        audio_path = row_audio_path(row)

        skip_payload = {
            "index": index,
            "audio_id": audio_id,
            "source": source,
            "skip_reason": skip_reason,
        }
        if not reviewed:
            skipped.append({**skip_payload, "reason": "not_reviewed"})
            continue
        if skip_reason and not args.include_skipped:
            skipped.append({**skip_payload, "reason": "skip_reason"})
            continue
        if duration_s <= 0.0:
            skipped.append({**skip_payload, "reason": "non_positive_duration_s", "duration_s": duration_s})
            continue
        if not audio_path:
            skipped.append({**skip_payload, "reason": "missing_audio_path"})
            continue
        if args.require_audio_exists and not Path(audio_path).exists():
            skipped.append({**skip_payload, "reason": "audio_path_not_found", "audio": audio_path})
            continue

        if segments:
            record = build_supervised_record(
                audio_id=audio_id,
                source=source,
                duration_s=duration_s,
                text=text,
                speech_segments=segments,
                frame_hop_s=args.frame_hop_s,
            )
            label_quality = "supervised"
            manual_speech_s += sum(segment["end"] - segment["start"] for segment in segments)
        else:
            record = build_negative_record(
                audio_id=audio_id,
                source=source,
                duration_s=duration_s,
                text=text,
                frame_hop_s=args.frame_hop_s,
            )
            label_quality = "negative"
        records.append(record)
        manifest_rows.append(
            {
                "audio_id": audio_id,
                "audio": audio_path,
                "duration_s": duration_s,
                "source": source,
                "text": text,
                "manual_reason": str(row.get("reason") or ""),
                "label_quality": label_quality,
            }
        )

    labels_path = output_dir / args.output_labels
    manifest_path = output_dir / args.output_manifest
    skipped_path = output_dir / "manual_label_skipped.json"
    summary_path = output_dir / "manual_label_summary.json"
    write_jsonl(labels_path, records)
    manifest_path.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    frame_total = sum(len(record.speech_frames) for record in records)
    speech_frame_total = sum(sum(int(value) for value in record.speech_frames) for record in records)
    summary = {
        "input": str(input_path),
        "labels": str(labels_path),
        "manifest": str(manifest_path),
        "manual_rows": len(rows),
        "records": len(records),
        "skipped": len(skipped),
        "reviewed": sum(bool(row.get("reviewed")) for row in rows),
        "with_manual_segments": sum(bool(row.get("speech_segments")) for row in rows),
        "without_manual_segments": sum(not bool(row.get("speech_segments")) for row in rows),
        "manual_duration_s": sum(float(row.get("duration_s") or 0.0) for row in rows),
        "manual_speech_s": manual_speech_s,
        "manual_speech_ratio": manual_speech_s / sum(float(row.get("duration_s") or 0.0) for row in rows) if rows else 0.0,
        "frame_count": frame_total,
        "speech_frame_count": speech_frame_total,
        "speech_frame_ratio": speech_frame_total / frame_total if frame_total else 0.0,
        "label_quality_counts": dict(sorted(Counter(record.label_quality for record in records).items())),
        "manual_reason_counts": dict(sorted(Counter(str(row.get("reason") or "") for row in rows).items())),
        "skip_reason_counts": dict(sorted(Counter(str(row.get("skip_reason") or "") for row in rows).items())),
        "skipped_reasons": dict(sorted(Counter(row.get("reason", "unknown") for row in skipped).items())),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"labels={labels_path}")
    print(f"manifest={manifest_path}")
    print(f"skipped={skipped_path}")
    print(f"summary={summary_path}")
    print(f"records={len(records)} skipped={len(skipped)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert manual audit JSONL into SpeechBoundary-JA labels.")
    parser.add_argument("--input", required=True, help="manual_labels.jsonl exported from the audit HTML.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "datasets" / "val" / "speech-boundary-ja" / "v1-3" / "manual-audit-galgame" / "strong-labels"),
    )
    parser.add_argument("--output-labels", default="labels.jsonl")
    parser.add_argument("--output-manifest", default="manifest.json")
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--include-skipped", action="store_true", help="Convert rows with skip_reason instead of dropping them.")
    parser.add_argument("--require-audio-exists", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.frame_hop_s <= 0:
        parser.error("--frame-hop-s must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
