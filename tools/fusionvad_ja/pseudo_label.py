#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono
from pipeline.audio import extract_audio
from vad.base import SpeechSegment
from vad.fusionvad_ja import (
    build_teacher_record,
    get_research_vad_backend,
    sample_hf_audio_16k_mono,
    write_jsonl,
)


AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm"}


def discover_inputs(values: Iterable[str]) -> list[Path]:
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
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required for HF pseudo-labeling: uv pip install datasets") from exc
    return load_dataset(name, split=split, streaming=True)


def sample_audio(example: dict[str, Any]) -> tuple[np.ndarray, int, str, str]:
    audio, sample_rate = sample_hf_audio_16k_mono(example)
    text = str(example.get("txt") or example.get("text") or "")
    audio_id = str(example.get("__key__") or example.get("id") or "")
    return audio, sample_rate, text, audio_id


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


def run_teacher(name: str, audio_path: Path) -> list[SpeechSegment]:
    try:
        return get_research_vad_backend(name).segment(str(audio_path)).segments
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def label_audio_file(
    *,
    audio_path: Path,
    audio_id: str,
    source: str,
    text: str,
    teachers: list[str],
    frame_hop_s: float,
    min_speech_teachers: int,
) -> tuple[object, dict]:
    audio, sample_rate = load_audio_16k_mono(str(audio_path))
    duration_s = len(audio) / sample_rate if sample_rate else 0.0
    teacher_segments = {}
    teacher_errors = {}
    for teacher in teachers:
        try:
            teacher_segments[teacher] = run_teacher(teacher, audio_path)
        except Exception as exc:
            teacher_segments[teacher] = []
            teacher_errors[teacher] = str(exc)
    record = build_teacher_record(
        audio_id=audio_id,
        source=source,
        duration_s=duration_s,
        text=text,
        teacher_segments=teacher_segments,
        frame_hop_s=frame_hop_s,
        min_speech_teachers=min_speech_teachers,
    )
    manifest_row = {
        "audio": str(audio_path),
        "duration_s": duration_s,
        "label_quality": record.label_quality,
        "teacher_errors": teacher_errors,
        "teacher_segment_counts": {
            name: len(segments) for name, segments in record.teacher_segments.items()
        },
    }
    return record, manifest_row


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teachers = args.teacher or ["whisperseg-adaptive", "fusion_lite", "ten_silero"]
    records = []
    manifest_rows = []
    for source_path in discover_inputs(args.input or []):
        audio_path = prepared_audio_path(source_path, output_dir=output_dir)
        record, manifest_row = label_audio_file(
            audio_path=audio_path,
            audio_id=source_path.stem,
            source=args.source,
            text="",
            teachers=teachers,
            frame_hop_s=args.frame_hop_s,
            min_speech_teachers=args.min_speech_teachers,
        )
        records.append(record)
        manifest_row["input"] = str(source_path)
        manifest_rows.append(manifest_row)

    if args.hf_dataset:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise SystemExit("soundfile is required for HF pseudo-labeling") from exc
        dataset = load_hf_dataset(name=args.hf_dataset, split=args.hf_split)
        hf_audio_dir = output_dir / "hf_audio"
        hf_audio_dir.mkdir(parents=True, exist_ok=True)
        for index, example in enumerate(dataset):
            if index >= args.hf_limit:
                break
            try:
                audio, sample_rate, text, audio_id = sample_audio(example)
                if not audio_id:
                    audio_id = f"hf-{index:06d}"
                with tempfile.NamedTemporaryFile(
                    suffix=".wav",
                    prefix=f"{audio_id[:24]}-",
                    dir=hf_audio_dir,
                    delete=False,
                ) as handle:
                    temp_audio_path = Path(handle.name)
                sf.write(str(temp_audio_path), audio, sample_rate)
                record, manifest_row = label_audio_file(
                    audio_path=temp_audio_path,
                    audio_id=audio_id,
                    source=args.hf_dataset,
                    text=text,
                    teachers=teachers,
                    frame_hop_s=args.frame_hop_s,
                    min_speech_teachers=args.min_speech_teachers,
                )
                records.append(record)
                manifest_row["input"] = f"{args.hf_dataset}:{index}"
                manifest_row["audio_id"] = audio_id
                manifest_rows.append(manifest_row)
            except Exception as exc:
                manifest_rows.append(
                    {
                        "input": f"{args.hf_dataset}:{index}",
                        "label_quality": "error",
                        "error": str(exc),
                    }
                )

    labels_path = output_dir / args.output_jsonl
    write_jsonl(labels_path, records)
    manifest_path = output_dir / "pseudo_label_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"pseudo_labels={labels_path}")
    print(f"manifest={manifest_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weak FusionVAD-JA frame labels from VAD teachers.")
    parser.add_argument("--input", action="append", help="Audio/video file or directory. Repeatable.")
    parser.add_argument("--hf-dataset", help="Optional Hugging Face streaming dataset, e.g. litagin/Galgame_Speech_ASR_16kHz.")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-limit", type=int, default=100)
    parser.add_argument("--teacher", action="append", help="Teacher VAD name. Repeatable. Defaults to whisperseg-adaptive, fusion_lite, ten_silero.")
    parser.add_argument("--source", default="local")
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--min-speech-teachers", type=int, default=2)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "pseudo-labels"))
    parser.add_argument("--output-jsonl", default="pseudo_labels.jsonl")
    args = parser.parse_args(argv)
    if not args.input and not args.hf_dataset:
        parser.error("at least one --input or --hf-dataset is required")
    return args


if __name__ == "__main__":
    os.environ.setdefault("ASR_VAD_ADAPTIVE", "1")
    run(parse_args())
