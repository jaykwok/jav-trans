#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.fusionvad_ja import LabelRecord, load_label_records


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
    return rows


def load_manifest_audio_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"manifest must be a JSON list: {path}")
    rows = [dict(row) for row in payload if isinstance(row, Mapping)]
    return {str(row.get("audio_id")): row for row in rows if row.get("audio_id")}


def record_segments(record: LabelRecord) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [asdict(segment) for segment in segments]
        for name, segments in record.teacher_segments.items()
    }


def enrich_audit_row(
    row: Mapping[str, Any],
    *,
    records_by_id: Mapping[str, LabelRecord],
    manifest_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    audio_id = str(row.get("audio_id") or "")
    record = records_by_id.get(audio_id)
    manifest = manifest_by_id.get(audio_id, {})
    frames = int(row.get("frames") or 0)
    weighted_speech = int(row.get("weighted_speech_frames") or 0)
    weighted_negative = int(row.get("weighted_negative_frames") or 0)
    enriched = {
        **dict(row),
        "audio_id": audio_id,
        "audio": str(row.get("audio") or manifest.get("audio") or ""),
        "text": record.text if record is not None else str(manifest.get("text") or ""),
        "source": record.source if record is not None else str(manifest.get("source") or ""),
        "teacher_segments": record_segments(record) if record is not None else {},
        "weighted_speech_frame_ratio": (weighted_speech / frames) if frames else 0.0,
        "weighted_negative_frame_ratio": (weighted_negative / frames) if frames else 0.0,
    }
    return enriched


def add_bucket(
    selected: list[dict[str, Any]],
    seen: set[str],
    rows: Iterable[dict[str, Any]],
    *,
    reason: str,
    per_bucket: int,
) -> None:
    added = 0
    for row in rows:
        audio_id = str(row.get("audio_id") or "")
        if not audio_id or audio_id in seen:
            continue
        candidate = {**row, "reason": reason}
        selected.append(candidate)
        seen.add(audio_id)
        added += 1
        if added >= per_bucket:
            return


def select_candidates(
    rows: list[dict[str, Any]],
    *,
    per_bucket: int,
    min_conflict_ratio: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    conflict_rows = sorted(
        [
            row
            for row in rows
            if row.get("label_quality") == "teacher_conflict"
            or float(row.get("conflict_frame_ratio") or 0.0) >= min_conflict_ratio
        ],
        key=lambda row: (
            float(row.get("conflict_frame_ratio") or 0.0),
            float(row.get("duration_s") or 0.0),
        ),
        reverse=True,
    )
    add_bucket(selected, seen, conflict_rows, reason="teacher_conflict_high", per_bucket=per_bucket)

    low_active_text = sorted(
        [row for row in rows if str(row.get("text") or "").strip()],
        key=lambda row: (
            float(row.get("active_frame_ratio") or 0.0),
            -float(row.get("duration_s") or 0.0),
        ),
    )
    add_bucket(selected, seen, low_active_text, reason="text_but_low_active", per_bucket=per_bucket)

    ignored_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("ignored_frame_ratio") or 0.0),
            float(row.get("conflict_frame_ratio") or 0.0),
        ),
        reverse=True,
    )
    add_bucket(selected, seen, ignored_rows, reason="ignored_ratio_high", per_bucket=per_bucket)

    negative_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("weighted_negative_frame_ratio") or 0.0),
            float(row.get("duration_s") or 0.0),
        ),
        reverse=True,
    )
    add_bucket(selected, seen, negative_rows, reason="negative_gap_high", per_bucket=per_bucket)

    high_agree_rows = sorted(
        [
            row
            for row in rows
            if row.get("label_quality") == "teacher_agree"
            and float(row.get("conflict_frame_ratio") or 0.0) <= min_conflict_ratio
        ],
        key=lambda row: (
            float(row.get("active_frame_ratio") or 0.0),
            -float(row.get("ignored_frame_ratio") or 0.0),
        ),
        reverse=True,
    )
    add_bucket(selected, seen, high_agree_rows, reason="clean_teacher_agree", per_bucket=per_bucket)

    long_rows = sorted(rows, key=lambda row: float(row.get("duration_s") or 0.0), reverse=True)
    add_bucket(selected, seen, long_rows, reason="long_clip", per_bucket=per_bucket)

    return selected


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "reason",
        "audio_id",
        "audio",
        "duration_s",
        "source",
        "label_quality",
        "active_frame_ratio",
        "ignored_frame_ratio",
        "conflict_frame_ratio",
        "weighted_speech_frame_ratio",
        "weighted_negative_frame_ratio",
        "teacher_segment_counts",
        "teacher_speech_ratios",
        "text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["teacher_segment_counts"] = json.dumps(
                payload.get("teacher_segment_counts") or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            payload["teacher_speech_ratios"] = json.dumps(
                payload.get("teacher_speech_ratios") or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            writer.writerow(payload)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_label_records(Path(args.labels))
    records_by_id = {record.audio_id: record for record in records}
    manifest_by_id = load_manifest_audio_map(Path(args.manifest) if args.manifest else None)
    audit_rows = read_jsonl_rows(Path(args.audit))
    enriched_rows = [
        enrich_audit_row(row, records_by_id=records_by_id, manifest_by_id=manifest_by_id)
        for row in audit_rows
    ]
    candidates = select_candidates(
        enriched_rows,
        per_bucket=args.per_bucket,
        min_conflict_ratio=args.min_conflict_ratio,
    )

    jsonl_path = output_dir / "audit_candidates.jsonl"
    csv_path = output_dir / "audit_candidates.csv"
    summary_path = output_dir / "audit_candidate_summary.json"
    write_jsonl(jsonl_path, candidates)
    write_csv(csv_path, candidates)
    summary = {
        "labels": args.labels,
        "audit": args.audit,
        "manifest": args.manifest,
        "records": len(records),
        "audit_rows": len(audit_rows),
        "candidates": len(candidates),
        "per_bucket": args.per_bucket,
        "min_conflict_ratio": args.min_conflict_ratio,
        "reason_counts": dict(sorted(Counter(row["reason"] for row in candidates).items())),
        "label_quality_counts": dict(sorted(Counter(row.get("label_quality", "") for row in candidates).items())),
        "output_jsonl": str(jsonl_path),
        "output_csv": str(csv_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"candidates={jsonl_path}")
    print(f"csv={csv_path}")
    print(f"summary={summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select FusionVAD-JA clips for manual VAD audit.")
    parser.add_argument("--labels", required=True, help="FusionVAD-JA label JSONL.")
    parser.add_argument("--audit", required=True, help="Manual or pseudo-label audit JSONL.")
    parser.add_argument("--manifest", help="Optional audio manifest JSON.")
    parser.add_argument("--per-bucket", type=int, default=12)
    parser.add_argument("--min-conflict-ratio", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "audit-candidates"),
    )
    args = parser.parse_args(argv)
    if args.per_bucket <= 0:
        parser.error("--per-bucket must be positive")
    if args.min_conflict_ratio < 0.0:
        parser.error("--min-conflict-ratio must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
