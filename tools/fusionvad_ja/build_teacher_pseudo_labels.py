#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono
from vad.fusionvad_ja import (
    TeacherSegment,
    build_weighted_teacher_record,
    effective_frame_weights,
    get_research_vad_backend,
    segments_to_frame_labels,
    write_jsonl,
)


def load_manifest_rows(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("manifest must be a JSON list")
    return [row for row in payload if isinstance(row, Mapping)]


def manifest_audio_path(row: Mapping[str, Any]) -> Path | None:
    value = row.get("audio")
    if not value:
        return None
    return Path(str(value))


def run_teacher(backend_cache: dict[str, Any], name: str, audio_path: Path) -> list[TeacherSegment]:
    backend = backend_cache.get(name)
    if backend is None:
        backend = get_research_vad_backend(name)
        backend_cache[name] = backend
    result = backend.segment(str(audio_path), target_sr=16000)
    return [
        TeacherSegment(start=float(segment.start), end=float(segment.end), score=segment.score)
        for segment in result.segments
    ]


def teacher_frame_votes(
    teacher_segments: Mapping[str, list[TeacherSegment]],
    *,
    duration_s: float,
    frame_hop_s: float,
) -> np.ndarray:
    votes: np.ndarray | None = None
    for segments in teacher_segments.values():
        labels = np.asarray(
            segments_to_frame_labels(segments, duration_s=duration_s, frame_hop_s=frame_hop_s),
            dtype=np.int32,
        )
        if votes is None:
            votes = np.zeros_like(labels, dtype=np.int32)
        votes[: labels.shape[0]] += labels
    return votes if votes is not None else np.zeros(0, dtype=np.int32)


def audit_record(
    *,
    audio_id: str,
    audio_path: Path,
    record: Any,
    teacher_segments: Mapping[str, list[TeacherSegment]],
    teacher_errors: Mapping[str, str],
    min_speech_teachers: int,
) -> dict[str, Any]:
    weights = effective_frame_weights(record)
    weighted = np.asarray(weights, dtype=np.float32)
    labels = np.asarray(record.speech_frames, dtype=np.int32)
    votes = teacher_frame_votes(
        teacher_segments,
        duration_s=record.duration_s,
        frame_hop_s=record.frame_hop_s,
    )
    frame_total = min(labels.size, weighted.size, votes.size)
    labels = labels[:frame_total]
    weighted = weighted[:frame_total]
    votes = votes[:frame_total]
    active = weighted > 0.0
    conflict = np.logical_and(votes > 0, votes < min_speech_teachers)
    teacher_ratios = {
        name: (
            sum(segments_to_frame_labels(segments, duration_s=record.duration_s, frame_hop_s=record.frame_hop_s))
            / frame_total
        )
        if frame_total
        else 0.0
        for name, segments in teacher_segments.items()
    }
    return {
        "audio_id": audio_id,
        "audio": str(audio_path),
        "duration_s": record.duration_s,
        "label_quality": record.label_quality,
        "frames": int(frame_total),
        "active_frames": int(active.sum()),
        "ignored_frames": int((~active).sum()),
        "active_frame_ratio": float(active.mean()) if frame_total else 0.0,
        "ignored_frame_ratio": float((~active).mean()) if frame_total else 0.0,
        "speech_frames": int(labels.sum()),
        "weighted_speech_frames": int(np.logical_and(labels > 0, active).sum()),
        "weighted_negative_frames": int(np.logical_and(labels == 0, active).sum()),
        "conflict_frames": int(conflict.sum()),
        "conflict_frame_ratio": float(conflict.mean()) if frame_total else 0.0,
        "teacher_errors": dict(teacher_errors),
        "teacher_segment_counts": {name: len(segments) for name, segments in teacher_segments.items()},
        "teacher_speech_ratios": teacher_ratios,
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest_rows(Path(args.manifest))
    selected_rows = rows[args.start_index :]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    teachers = args.backend or ["whisperseg-adaptive", "fusion_lite"]
    records = []
    pseudo_manifest = []
    audit_rows = []
    conflict_rows = []
    errors = []
    backend_cache: dict[str, Any] = {}

    for offset, row in enumerate(selected_rows):
        index = args.start_index + offset
        audio_path = manifest_audio_path(row)
        audio_id = str(row.get("audio_id") or (audio_path.stem if audio_path else f"row-{index:06d}"))
        if audio_path is None or not audio_path.exists():
            errors.append({"index": index, "audio_id": audio_id, "reason": "missing_audio", "audio": str(audio_path)})
            continue
        try:
            duration_s = float(row.get("duration_s") or 0.0)
            if duration_s <= 0.0:
                audio, sample_rate = load_audio_16k_mono(str(audio_path))
                duration_s = len(audio) / sample_rate if sample_rate else 0.0
            if duration_s <= 0.0:
                raise ValueError("invalid duration")
            teacher_segments: dict[str, list[TeacherSegment]] = {}
            teacher_errors: dict[str, str] = {}
            for teacher in teachers:
                try:
                    teacher_segments[teacher] = run_teacher(backend_cache, teacher, audio_path)
                except Exception as exc:
                    teacher_segments[teacher] = []
                    teacher_errors[teacher] = str(exc)
            source = str(row.get("source") or args.source)
            text = str(row.get("text") or "")
            record = build_weighted_teacher_record(
                audio_id=audio_id,
                source=source,
                duration_s=duration_s,
                text=text,
                teacher_segments=teacher_segments,
                frame_hop_s=args.frame_hop_s,
                min_speech_teachers=args.min_speech_teachers,
                min_negative_gap_s=args.min_negative_gap_s,
                boundary_pad_s=args.boundary_pad_s,
                positive_weight=args.teacher_weight,
                negative_weight=args.teacher_weight,
            )
            audit = audit_record(
                audio_id=audio_id,
                audio_path=audio_path,
                record=record,
                teacher_segments=teacher_segments,
                teacher_errors=teacher_errors,
                min_speech_teachers=args.min_speech_teachers,
            )
            records.append(record)
            pseudo_manifest.append(
                {
                    "audio_id": audio_id,
                    "audio": str(audio_path),
                    "duration_s": duration_s,
                    "input": row.get("input"),
                    "source": source,
                    "text": text,
                    "label_quality": record.label_quality,
                }
            )
            audit_rows.append(audit)
            if record.label_quality == "teacher_conflict" or audit["conflict_frames"] > 0:
                conflict_rows.append(
                    {
                        **audit,
                        "teacher_segments": {
                            name: [asdict(segment) for segment in segments]
                            for name, segments in teacher_segments.items()
                        },
                    }
                )
            print(
                f"pseudo_labeled {offset + 1}/{len(selected_rows)} audio_id={audio_id} "
                f"quality={record.label_quality} active_ratio={audit['active_frame_ratio']:.3f} "
                f"conflict_ratio={audit['conflict_frame_ratio']:.3f}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"index": index, "audio_id": audio_id, "audio": str(audio_path), "error": str(exc)})
            print(f"error {offset + 1}/{len(selected_rows)} audio_id={audio_id} error={exc}", flush=True)

    labels_path = output_dir / args.output_jsonl
    manifest_path = output_dir / "teacher_pseudo_manifest.json"
    audit_path = output_dir / "teacher_pseudo_audit.jsonl"
    conflict_path = output_dir / "teacher_conflicts.jsonl"
    errors_path = output_dir / "teacher_pseudo_errors.json"
    summary_path = output_dir / "teacher_pseudo_summary.json"

    write_jsonl(labels_path, records)
    manifest_path.write_text(
        json.dumps(pseudo_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with audit_path.open("w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with conflict_path.open("w", encoding="utf-8") as handle:
        for row in conflict_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    errors_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    frame_total = sum(int(row["frames"]) for row in audit_rows)
    active_total = sum(int(row["active_frames"]) for row in audit_rows)
    ignored_total = sum(int(row["ignored_frames"]) for row in audit_rows)
    weighted_speech_total = sum(int(row["weighted_speech_frames"]) for row in audit_rows)
    weighted_negative_total = sum(int(row["weighted_negative_frames"]) for row in audit_rows)
    conflict_total = sum(int(row["conflict_frames"]) for row in audit_rows)
    summary = {
        "manifest": args.manifest,
        "labels": str(labels_path),
        "output_manifest": str(manifest_path),
        "audit": str(audit_path),
        "conflicts": str(conflict_path),
        "errors": str(errors_path),
        "teachers": teachers,
        "frame_hop_s": args.frame_hop_s,
        "min_speech_teachers": args.min_speech_teachers,
        "min_negative_gap_s": args.min_negative_gap_s,
        "boundary_pad_s": args.boundary_pad_s,
        "teacher_weight": args.teacher_weight,
        "records": len(records),
        "errors_count": len(errors),
        "label_quality_counts": dict(sorted(Counter(record.label_quality for record in records).items())),
        "frames": frame_total,
        "active_frames": active_total,
        "ignored_frames": ignored_total,
        "active_frame_ratio": (active_total / frame_total) if frame_total else 0.0,
        "ignored_frame_ratio": (ignored_total / frame_total) if frame_total else 0.0,
        "weighted_speech_frame_ratio": (weighted_speech_total / frame_total) if frame_total else 0.0,
        "weighted_negative_frame_ratio": (weighted_negative_total / frame_total) if frame_total else 0.0,
        "conflict_frame_ratio": (conflict_total / frame_total) if frame_total else 0.0,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"teacher_pseudo_labels={labels_path}")
    print(f"teacher_pseudo_manifest={manifest_path}")
    print(f"audit={audit_path}")
    print(f"conflicts={conflict_path}")
    print(f"errors={errors_path}")
    print(f"summary={summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weighted teacher pseudo-labels from materialized audio.")
    parser.add_argument("--manifest", required=True, help="Manifest with audio_id/audio/duration_s fields.")
    parser.add_argument("--backend", action="append", help="Teacher VAD backend. Repeatable.")
    parser.add_argument("--source", default="litagin/Galgame_Speech_ASR_16kHz")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--min-speech-teachers", type=int, default=2)
    parser.add_argument("--min-negative-gap-s", type=float, default=0.5)
    parser.add_argument("--boundary-pad-s", type=float, default=0.08)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "teacher-pseudo-labels"),
    )
    parser.add_argument("--output-jsonl", default="teacher_pseudo_labels.jsonl")
    args = parser.parse_args(argv)
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.min_speech_teachers <= 0:
        parser.error("--min-speech-teachers must be positive")
    if args.teacher_weight < 0.0:
        parser.error("--teacher-weight must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
