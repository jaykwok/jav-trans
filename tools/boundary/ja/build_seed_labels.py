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

from audio.loading import load_audio_16k_mono
from pipeline.audio import extract_audio
from boundary.ja import (
    DEFAULT_TRAINABLE_LABEL_QUALITIES,
    LabelRecord,
    build_negative_record,
    build_supervised_record,
    default_trainable_records,
    stable_hf_audio_id,
    write_jsonl,
)


AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm"}


def discover_media_inputs(values: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            for suffix in sorted(AUDIO_SUFFIXES | VIDEO_SUFFIXES):
                paths.extend(path.rglob(f"*{suffix}"))
        elif path.exists():
            paths.append(path)
        else:
            raise SystemExit(f"input not found: {path}")
    return sorted(set(paths), key=lambda item: str(item))


def load_hf_dataset(*, name: str, split: str):
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required for seed label generation: uv pip install datasets") from exc
    dataset = load_dataset(name, split=split, streaming=True)
    try:
        features = dataset.features or {}
        if "audio" in features and isinstance(features["audio"], Audio):
            dataset = dataset.cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    return dataset


def segments_from_ava_example(example: Mapping[str, Any]) -> list[tuple[float, float]]:
    onsets = list(example.get("onset") or [])
    offsets = list(example.get("offset") or [])
    clusters = list(example.get("cluster") or [])
    segments: list[tuple[float, float]] = []
    for index, (start, end) in enumerate(zip(onsets, offsets, strict=False)):
        cluster = str(clusters[index]) if index < len(clusters) else ""
        if cluster and "speech" not in cluster.lower():
            continue
        segments.append((float(start), float(end)))
    return segments


def record_from_ava_example(
    example: Mapping[str, Any],
    *,
    index: int,
    dataset_name: str,
    split: str,
    source: str,
    frame_hop_s: float,
) -> LabelRecord | None:
    segments = segments_from_ava_example(example)
    if not segments:
        return None
    duration_s = max(end for _, end in segments)
    return build_supervised_record(
        audio_id=str(
            example.get("id")
            or example.get("__key__")
            or stable_hf_audio_id(dataset_name=dataset_name, split=split, index=index)
        ),
        source=source,
        duration_s=duration_s,
        speech_segments=segments,
        frame_hop_s=frame_hop_s,
    )


def segments_from_voxconverse_example(example: Mapping[str, Any]) -> list[tuple[float, float]]:
    starts = list(example.get("timestamps_start") or [])
    ends = list(example.get("timestamps_end") or [])
    return [(float(start), float(end)) for start, end in zip(starts, ends, strict=False)]


def record_from_voxconverse_example(
    example: Mapping[str, Any],
    *,
    index: int,
    dataset_name: str,
    split: str,
    source: str,
    frame_hop_s: float,
) -> LabelRecord | None:
    segments = segments_from_voxconverse_example(example)
    if not segments:
        return None
    audio_obj = example.get("audio")
    audio_id = ""
    if isinstance(audio_obj, Mapping):
        audio_id = str(audio_obj.get("path") or "")
    duration_s = max(end for _, end in segments)
    return build_supervised_record(
        audio_id=str(
            example.get("id")
            or example.get("__key__")
            or stable_hf_audio_id(dataset_name=dataset_name, split=split, index=index)
        ),
        source=source,
        duration_s=duration_s,
        speech_segments=segments,
        frame_hop_s=frame_hop_s,
    )


def build_hf_supervised_records(
    *,
    dataset_name: str,
    split: str,
    start_index: int,
    limit: int,
    source: str,
    frame_hop_s: float,
    kind: str,
) -> tuple[list[LabelRecord], list[dict[str, Any]]]:
    dataset = load_hf_dataset(name=dataset_name, split=split)
    records: list[LabelRecord] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, example in enumerate(dataset):
        if index < start_index:
            continue
        if index >= start_index + limit:
            break
        try:
            if kind == "ava":
                record = record_from_ava_example(
                    example,
                    index=index,
                    dataset_name=dataset_name,
                    split=split,
                    source=source,
                    frame_hop_s=frame_hop_s,
                )
            elif kind == "voxconverse":
                record = record_from_voxconverse_example(
                    example,
                    index=index,
                    dataset_name=dataset_name,
                    split=split,
                    source=source,
                    frame_hop_s=frame_hop_s,
                )
            else:
                raise ValueError(f"unsupported supervised kind: {kind}")
            if record is None:
                manifest_rows.append({"input": f"{dataset_name}:{split}:{index}", "label_quality": "skipped"})
                continue
            records.append(record)
            manifest_rows.append(
                {
                    "input": f"{dataset_name}:{split}:{index}",
                    "audio_id": record.audio_id,
                    "duration_s": record.duration_s,
                    "label_quality": record.label_quality,
                    "speech_frame_count": sum(record.speech_frames),
                    "teacher_segment_counts": {
                        name: len(segments) for name, segments in record.teacher_segments.items()
                    },
                }
            )
        except Exception as exc:
            manifest_rows.append(
                {
                    "input": f"{dataset_name}:{split}:{index}",
                    "label_quality": "error",
                    "error": str(exc),
                }
            )
    return records, manifest_rows


def prepared_audio_path(path: Path, *, output_dir: Path) -> Path:
    if path.suffix.lower() in AUDIO_SUFFIXES:
        return path
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise SystemExit(f"unsupported input suffix: {path}")
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{path.stem}.wav"
    if not audio_path.exists():
        extract_audio(str(path), str(audio_path))
    return audio_path


def build_negative_records(
    *,
    inputs: Iterable[str],
    output_dir: Path,
    source: str,
    frame_hop_s: float,
    start_index: int = 0,
    limit: int = 0,
) -> tuple[list[LabelRecord], list[dict[str, Any]]]:
    records: list[LabelRecord] = []
    manifest_rows: list[dict[str, Any]] = []
    source_paths = discover_media_inputs(inputs)
    end_index = None if limit <= 0 else start_index + limit
    for input_index, source_path in enumerate(source_paths):
        if input_index < start_index:
            continue
        if end_index is not None and input_index >= end_index:
            break
        try:
            audio_path = prepared_audio_path(source_path, output_dir=output_dir)
            audio, sample_rate = load_audio_16k_mono(str(audio_path))
            duration_s = len(audio) / sample_rate if sample_rate else 0.0
            record = build_negative_record(
                audio_id=source_path.stem,
                source=source,
                duration_s=duration_s,
                frame_hop_s=frame_hop_s,
            )
            records.append(record)
            manifest_rows.append(
                {
                    "input_index": input_index,
                    "input": str(source_path),
                    "audio": str(audio_path),
                    "duration_s": duration_s,
                    "label_quality": record.label_quality,
                }
            )
        except Exception as exc:
            manifest_rows.append(
                {
                    "input_index": input_index,
                    "input": str(source_path),
                    "label_quality": "error",
                    "error": str(exc),
                }
            )
    return records, manifest_rows


def write_summary(
    *,
    output_dir: Path,
    records: list[LabelRecord],
    manifest_rows: list[dict[str, Any]],
    trainable_records: list[LabelRecord],
    labels_path: Path,
    trainable_path: Path,
    manifest_path: Path,
) -> Path:
    quality_counts = Counter(record.label_quality for record in records)
    source_counts = Counter(record.source for record in records)
    summary = {
        "rows": len(records),
        "manifest_rows": len(manifest_rows),
        "duration_s_total": sum(record.duration_s for record in records),
        "label_quality_counts": dict(sorted(quality_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "default_trainable_label_qualities": sorted(DEFAULT_TRAINABLE_LABEL_QUALITIES),
        "default_trainable_rows": len(trainable_records),
        "output_jsonl": str(labels_path),
        "trainable_jsonl": str(trainable_path),
        "manifest": str(manifest_path),
    }
    summary_path = output_dir / "seed_label_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[LabelRecord] = []
    manifest_rows: list[dict[str, Any]] = []

    if args.ava_limit > 0:
        ava_records, ava_manifest = build_hf_supervised_records(
            dataset_name=args.ava_dataset,
            split=args.ava_split,
            start_index=args.ava_start_index,
            limit=args.ava_limit,
            source="ava-speech",
            frame_hop_s=args.frame_hop_s,
            kind="ava",
        )
        records.extend(ava_records)
        manifest_rows.extend(ava_manifest)

    if args.voxconverse_limit > 0:
        vox_records, vox_manifest = build_hf_supervised_records(
            dataset_name=args.voxconverse_dataset,
            split=args.voxconverse_split,
            start_index=args.voxconverse_start_index,
            limit=args.voxconverse_limit,
            source="voxconverse",
            frame_hop_s=args.frame_hop_s,
            kind="voxconverse",
        )
        records.extend(vox_records)
        manifest_rows.extend(vox_manifest)

    if args.negative_input:
        negative_records, negative_manifest = build_negative_records(
            inputs=args.negative_input,
            output_dir=output_dir,
            source=args.negative_source,
            frame_hop_s=args.frame_hop_s,
            start_index=args.negative_start_index,
            limit=args.negative_limit,
        )
        records.extend(negative_records)
        manifest_rows.extend(negative_manifest)

    labels_path = output_dir / args.output_jsonl
    trainable_path = output_dir / args.trainable_jsonl
    manifest_path = output_dir / "seed_label_manifest.json"
    trainable_records = default_trainable_records(records)
    write_jsonl(labels_path, records)
    write_jsonl(trainable_path, trainable_records)
    manifest_path.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path = write_summary(
        output_dir=output_dir,
        records=records,
        manifest_rows=manifest_rows,
        trainable_records=trainable_records,
        labels_path=labels_path,
        trainable_path=trainable_path,
        manifest_path=manifest_path,
    )
    print(f"seed_labels={labels_path}")
    print(f"trainable_labels={trainable_path}")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SpeechBoundary-JA supervised/negative seed labels.")
    parser.add_argument("--ava-dataset", default="nccratliri/vad-human-ava-speech")
    parser.add_argument("--ava-split", default="train")
    parser.add_argument("--ava-start-index", type=int, default=0)
    parser.add_argument("--ava-limit", type=int, default=0)
    parser.add_argument("--voxconverse-dataset", default="diarizers-community/voxconverse")
    parser.add_argument("--voxconverse-split", default="dev")
    parser.add_argument("--voxconverse-start-index", type=int, default=0)
    parser.add_argument("--voxconverse-limit", type=int, default=0)
    parser.add_argument("--negative-input", action="append", help="MUSAN/DNS/local audio/video file or directory. Repeatable.")
    parser.add_argument("--negative-source", default="negative")
    parser.add_argument("--negative-start-index", type=int, default=0)
    parser.add_argument("--negative-limit", type=int, default=0, help="Maximum sorted negative media inputs to use. 0 means all.")
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / "seed-labels"))
    parser.add_argument("--output-jsonl", default="seed_labels.jsonl")
    parser.add_argument("--trainable-jsonl", default="seed_labels.trainable.jsonl")
    args = parser.parse_args(argv)
    if args.ava_start_index < 0:
        parser.error("--ava-start-index must be non-negative")
    if args.voxconverse_start_index < 0:
        parser.error("--voxconverse-start-index must be non-negative")
    if args.negative_start_index < 0:
        parser.error("--negative-start-index must be non-negative")
    if args.negative_limit < 0:
        parser.error("--negative-limit must be non-negative")
    if args.ava_limit <= 0 and args.voxconverse_limit <= 0 and not args.negative_input:
        parser.error("provide --ava-limit, --voxconverse-limit, or --negative-input")
    return args


if __name__ == "__main__":
    run(parse_args())
