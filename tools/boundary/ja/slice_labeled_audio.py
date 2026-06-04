#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono
from boundary.ja import (
    LabelRecord,
    TeacherSegment,
    build_training_examples,
    load_label_records,
    load_manifest_audio_map,
    write_jsonl,
)


def adjusted_segments(
    record: LabelRecord,
    *,
    start_s: float,
    duration_s: float,
) -> dict[str, list[TeacherSegment]]:
    adjusted: dict[str, list[TeacherSegment]] = {}
    end_s = start_s + duration_s
    for name, segments in record.teacher_segments.items():
        clipped: list[TeacherSegment] = []
        for segment in segments:
            start = max(segment.start, start_s)
            end = min(segment.end, end_s)
            if end <= start:
                continue
            clipped.append(
                TeacherSegment(
                    start=start - start_s,
                    end=end - start_s,
                    score=segment.score,
                )
            )
        adjusted[name] = clipped
    return adjusted


def slice_records(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records = load_label_records(Path(args.labels))
    audio_map = load_manifest_audio_map(Path(args.manifest) if args.manifest else None)
    examples, skipped = build_training_examples(
        records,
        manifest_audio_map=audio_map,
        audio_root=Path(args.audio_root) if args.audio_root else None,
        extension_hints=args.extension,
    )
    clip_records: list[LabelRecord] = []
    manifest_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    clip_frames = max(1, int(round(args.clip_s / args.frame_hop_s)))
    stride_frames = max(1, int(round(args.stride_s / args.frame_hop_s)))
    min_frames = max(1, int(math.ceil(args.min_clip_s / args.frame_hop_s)))
    for example in examples:
        record = records[example.label_index]
        try:
            audio, sample_rate = load_audio_16k_mono(example.audio_path)
            audio_frame_count = int(math.floor((len(audio) / sample_rate) / args.frame_hop_s))
            usable_frames = min(len(record.speech_frames), audio_frame_count)
            if usable_frames < min_frames:
                skipped.append(
                    {
                        "audio_id": record.audio_id,
                        "source": record.source,
                        "label_quality": record.label_quality,
                        "reason": "too_short_for_clip",
                        "usable_frames": usable_frames,
                    }
                )
                continue
            clip_index = 0
            for start_frame in range(0, usable_frames, stride_frames):
                end_frame = min(start_frame + clip_frames, usable_frames)
                if end_frame - start_frame < min_frames:
                    continue
                start_s = start_frame * args.frame_hop_s
                duration_s = (end_frame - start_frame) * args.frame_hop_s
                sample_start = int(round(start_s * sample_rate))
                sample_end = int(round((start_s + duration_s) * sample_rate))
                audio_clip = audio[sample_start:sample_end]
                if audio_clip.shape[0] <= 0:
                    continue
                audio_stem = Path(record.audio_id).stem
                clip_audio_id = f"{audio_stem}-{start_frame:08d}-{end_frame:08d}"
                output_audio_path = audio_dir / f"{clip_audio_id}.wav"
                sf.write(str(output_audio_path), audio_clip, sample_rate)
                speech_frames = [int(value) for value in record.speech_frames[start_frame:end_frame]]
                clip_records.append(
                    LabelRecord(
                        audio_id=clip_audio_id,
                        source=record.source,
                        duration_s=duration_s,
                        text=record.text,
                        teacher_segments=adjusted_segments(
                            record,
                            start_s=start_s,
                            duration_s=duration_s,
                        ),
                        frame_hop_s=args.frame_hop_s,
                        speech_frames=speech_frames,
                        label_quality=record.label_quality,
                    )
                )
                manifest_rows.append(
                    {
                        "audio": str(output_audio_path),
                        "audio_id": clip_audio_id,
                        "duration_s": duration_s,
                        "input": f"{example.audio_path}:{start_s:.3f}-{start_s + duration_s:.3f}",
                        "label_quality": record.label_quality,
                        "sample_rate": sample_rate,
                        "source": record.source,
                        "speech_frame_count": sum(speech_frames),
                        "speech_ratio": (sum(speech_frames) / len(speech_frames)) if speech_frames else 0.0,
                    }
                )
                clip_index += 1
                if args.max_clips_per_audio and clip_index >= args.max_clips_per_audio:
                    break
        except Exception as exc:
            errors.append(
                {
                    "audio_id": record.audio_id,
                    "audio_path": example.audio_path,
                    "source": record.source,
                    "error": str(exc),
                }
            )

    labels_path = output_dir / args.output_jsonl
    manifest_path = output_dir / "clip_manifest.json"
    skipped_path = output_dir / "clip_skipped.json"
    errors_path = output_dir / "clip_errors.json"
    summary_path = output_dir / "clip_summary.json"
    write_jsonl(labels_path, clip_records)
    manifest_path.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    errors_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "labels": str(labels_path),
        "manifest": str(manifest_path),
        "source_labels": args.labels,
        "source_manifest": args.manifest,
        "source_examples": len(examples),
        "clip_records": len(clip_records),
        "skipped": len(skipped),
        "errors": len(errors),
        "duration_s_total": sum(record.duration_s for record in clip_records),
        "label_quality_counts": dict(sorted(Counter(record.label_quality for record in clip_records).items())),
        "source_counts": dict(sorted(Counter(record.source for record in clip_records).items())),
        "clip_s": args.clip_s,
        "stride_s": args.stride_s,
        "min_clip_s": args.min_clip_s,
        "max_clips_per_audio": args.max_clips_per_audio,
        "speech_ratio_mean": (
            sum(float(row["speech_ratio"]) for row in manifest_rows) / len(manifest_rows)
            if manifest_rows
            else 0.0
        ),
        "skipped_report": str(skipped_path),
        "errors_report": str(errors_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"clip_labels={labels_path}")
    print(f"manifest={manifest_path}")
    print(f"skipped={skipped_path}")
    print(f"errors={errors_path}")
    print(f"summary={summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice long labeled SpeechBoundary-JA audio into short trainable clips.")
    parser.add_argument("--labels", required=True, help="SpeechBoundary-JA label JSONL.")
    parser.add_argument("--manifest", help="JSON manifest containing audio paths.")
    parser.add_argument("--audio-root", help="Optional directory for resolving audio_id + extension.")
    parser.add_argument("--extension", action="append", default=[".wav", ".flac", ".ogg", ".mp3", ".m4a"])
    parser.add_argument("--clip-s", type=float, default=20.0)
    parser.add_argument("--stride-s", type=float, default=20.0)
    parser.add_argument("--min-clip-s", type=float, default=4.0)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--max-clips-per-audio", type=int, default=0)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / "labeled-clips"))
    parser.add_argument("--output-jsonl", default="clip_labels.jsonl")
    args = parser.parse_args(argv)
    if args.clip_s <= 0.0:
        parser.error("--clip-s must be positive")
    if args.stride_s <= 0.0:
        parser.error("--stride-s must be positive")
    if args.min_clip_s <= 0.0:
        parser.error("--min-clip-s must be positive")
    if args.frame_hop_s <= 0.0:
        parser.error("--frame-hop-s must be positive")
    if args.max_clips_per_audio < 0:
        parser.error("--max-clips-per-audio must be non-negative")
    return args


if __name__ == "__main__":
    slice_records(parse_args())
