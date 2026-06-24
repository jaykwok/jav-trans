#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.ja import TeacherSegment, build_supervised_record, write_jsonl  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    yield dict(payload)


def parse_segments(items: Iterable[Mapping[str, Any]], *, duration_s: float) -> list[TeacherSegment]:
    segments: list[TeacherSegment] = []
    for item in items:
        try:
            start = max(0.0, min(float(item.get("start", item.get("start_s", 0.0))), duration_s))
            end = max(0.0, min(float(item.get("end", item.get("end_s", 0.0))), duration_s))
        except (TypeError, ValueError):
            continue
        if end > start:
            segments.append(TeacherSegment(start=start, end=end, score=1.0))
    return sorted(segments, key=lambda segment: (segment.start, segment.end))


def build_exact_island_labels(
    *,
    boundary_manifest: Path,
    output_jsonl: Path,
    segment_field: str = "actual_speech_segments",
    source_suffix: str = "exact-island",
) -> dict[str, Any]:
    records = []
    skipped = []
    segment_counts: Counter[int] = Counter()
    frame_total = 0
    speech_frame_total = 0

    for row_index, payload in enumerate(iter_jsonl(boundary_manifest)):
        audio_id = str(payload.get("audio_id") or "")
        if not audio_id:
            skipped.append({"row_index": row_index, "reason": "missing_audio_id"})
            continue
        duration_s = float(payload.get("duration_s") or 0.0)
        frame_hop_s = float(payload.get("frame_hop_s") or 0.02)
        if duration_s <= 0.0 or frame_hop_s <= 0.0:
            skipped.append({"row_index": row_index, "audio_id": audio_id, "reason": "invalid_timing"})
            continue
        raw_segments = payload.get(segment_field) or []
        if not isinstance(raw_segments, list):
            skipped.append({"row_index": row_index, "audio_id": audio_id, "reason": "invalid_segments"})
            continue
        segments = parse_segments(raw_segments, duration_s=duration_s)
        record = build_supervised_record(
            audio_id=audio_id,
            source=f"{payload.get('source') or 'boundary'}-{source_suffix}",
            duration_s=duration_s,
            text=str(payload.get("text") or ""),
            speech_segments=segments,
            frame_hop_s=frame_hop_s,
        )
        boundary_metadata = {
            "source_audio_ids": list(payload.get("source_audio_ids") or []),
            "utterance_boundaries": list(payload.get("utterance_boundaries") or []),
            "cut_point_segments": list(payload.get("cut_point_segments") or []),
            "actual_speech_segments": list(payload.get("actual_speech_segments") or []),
            "speech_label_dilation_s": payload.get("speech_label_dilation_s"),
        }
        record = type(record)(
            audio_id=record.audio_id,
            source=record.source,
            duration_s=record.duration_s,
            text=record.text,
            teacher_segments=record.teacher_segments,
            frame_hop_s=record.frame_hop_s,
            speech_frames=record.speech_frames,
            label_quality=record.label_quality,
            frame_weights=record.frame_weights,
            boundary_metadata=boundary_metadata,
        )
        records.append(record)
        segment_counts[len(segments)] += 1
        frame_total += len(record.speech_frames)
        speech_frame_total += sum(int(value) for value in record.speech_frames)

    write_jsonl(output_jsonl, records)
    summary = {
        "boundary_manifest": str(boundary_manifest),
        "output_jsonl": str(output_jsonl),
        "segment_field": segment_field,
        "records": len(records),
        "skipped": len(skipped),
        "segment_count_distribution": {str(key): value for key, value in sorted(segment_counts.items())},
        "frame_count": frame_total,
        "speech_frame_count": speech_frame_total,
        "speech_frame_ratio": speech_frame_total / frame_total if frame_total else 0.0,
        "skipped_rows": skipped,
    }
    summary_path = output_jsonl.with_name(output_jsonl.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"labels={output_jsonl}")
    print(f"summary={summary_path}")
    print(
        f"records={summary['records']} skipped={summary['skipped']} "
        f"speech_frame_ratio={summary['speech_frame_ratio']:.4f}"
    )
    return summary


def run(args: argparse.Namespace) -> None:
    build_exact_island_labels(
        boundary_manifest=Path(args.boundary_manifest),
        output_jsonl=Path(args.output_jsonl),
        segment_field=args.segment_field,
        source_suffix=args.source_suffix,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact speech-island SpeechBoundary-JA labels from synthetic boundary_manifest.jsonl."
    )
    parser.add_argument("--boundary-manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--segment-field", default="actual_speech_segments")
    parser.add_argument("--source-suffix", default="exact-island")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
