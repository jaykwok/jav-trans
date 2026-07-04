#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from asr.pipeline import (  # noqa: E402
    _build_processing_spans,
    _pre_asr_candidates_for_spans,
)
from pipeline.audio import build_audio_filter_chain  # noqa: E402
from utils.subprocess_tools import no_window_subprocess_kwargs  # noqa: E402


VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".m4v"}
SCHEMA = "joint_boundary_omni_source_window_v1"


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-._")
    return cleaned[:48] or "video"


def _stable_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{_safe_stem(path.stem)}-{digest}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_duration(path: Path) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **no_window_subprocess_kwargs(),
        )
        duration = float(completed.stdout.strip())
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError):
        return None
    return duration if duration > 0.0 else None


def _extract_window(
    *,
    source: Path,
    start_s: float,
    duration_s: float,
    wav_path: Path,
    mp3_path: Path,
) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    if not wav_path.exists():
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_s:.6f}",
            "-i",
            str(source),
            "-t",
            f"{duration_s:.6f}",
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-codec:a",
            "pcm_s16le",
        ]
        filter_chain = build_audio_filter_chain()
        if filter_chain:
            command.extend(["-af", filter_chain])
        command.append(str(wav_path))
        subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            check=True,
            **no_window_subprocess_kwargs(),
        )
    if not mp3_path.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(wav_path),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "32k",
                str(mp3_path),
            ],
            cwd=str(PROJECT_ROOT),
            check=True,
            **no_window_subprocess_kwargs(),
        )


def _window_starts(
    *,
    duration_s: float,
    window_s: float,
    count: int,
    rng: random.Random,
) -> list[float]:
    if duration_s <= window_s:
        return [0.0]
    margin = min(60.0, max(0.0, (duration_s - window_s) / 4.0))
    low = margin
    high = max(low, duration_s - window_s - margin)
    if count <= 1 or high <= low:
        return [round((low + high) / 2.0, 6)]
    cells = count
    width = (high - low) / cells
    starts = [
        low + width * index + rng.random() * max(width, 1e-6)
        for index in range(cells)
    ]
    return [round(min(high, max(low, value)), 6) for value in starts]


def _select_sources(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.source_root)
    excluded = tuple(item.lower() for item in args.exclude_contains)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_SUFFIXES
        and not any(item in str(path).lower() for item in excluded)
    )
    rng = random.Random(args.seed)
    rng.shuffle(files)
    selected: list[dict[str, Any]] = []
    for path in files:
        duration = _probe_duration(path)
        if duration is None or duration < args.min_video_duration_s:
            continue
        selected.append(
            {
                "video_id": _stable_id(path),
                "source_video": str(path.resolve()),
                "source_duration_s": round(duration, 6),
                "source_size_bytes": path.stat().st_size,
            }
        )
        if len(selected) >= args.video_count:
            break
    if len(selected) < args.video_count:
        raise RuntimeError(
            f"only {len(selected)} eligible videos found; requested {args.video_count}"
        )
    return selected


def _export_boundary_features(
    *,
    wav_path: Path,
    feature_dir: Path,
) -> dict[str, Any]:
    split_path = feature_dir / "semantic_split_features.npz"
    speech_path = feature_dir / "speech_sequence_features.npz"
    candidates_path = feature_dir / "pre_asr_candidates.jsonl"
    audit_path = feature_dir / "boundary_audit.jsonl"
    if (
        split_path.exists()
        and split_path.with_suffix(".jsonl").exists()
        and speech_path.exists()
        and candidates_path.exists()
        and audit_path.exists()
    ):
        return {
            "semantic_split_features": str(split_path),
            "semantic_split_metadata": str(split_path.with_suffix(".jsonl")),
            "speech_sequence_features": str(speech_path),
            "pre_asr_candidates": str(candidates_path),
            "boundary_audit": str(audit_path),
            "resumed": True,
        }
    feature_dir.mkdir(parents=True, exist_ok=True)
    os.environ["BOUNDARY_CACHE_ENABLED"] = "0"
    os.environ["PRE_ASR_CUEQC_ENABLED"] = "0"
    os.environ["SEMANTIC_SPLIT_FEATURE_EXPORT_PATH"] = str(split_path)
    os.environ["SPEECH_ISLAND_FEATURE_EXPORT_PATH"] = str(speech_path)
    spans = _build_processing_spans(str(wav_path))
    candidates = _pre_asr_candidates_for_spans(str(wav_path), spans)
    _write_jsonl(candidates_path, candidates)
    audit_rows: list[dict[str, Any]] = []
    for span_index, span in enumerate(spans):
        for accepted, items in (
            (True, getattr(span, "primary_cut_candidates", None) or []),
            (False, getattr(span, "weak_cut_candidates", None) or []),
        ):
            for candidate in items:
                audit_rows.append(
                    {
                        "span_index": span_index,
                        "span_start": float(span.start),
                        "span_end": float(span.end),
                        "accepted": accepted,
                        **candidate,
                    }
                )
    _write_jsonl(audit_path, audit_rows)
    return {
        "semantic_split_features": str(split_path),
        "semantic_split_metadata": str(split_path.with_suffix(".jsonl")),
        "speech_sequence_features": str(speech_path),
        "pre_asr_candidates": str(candidates_path),
        "boundary_audit": str(audit_path),
        "span_count": len(spans),
        "candidate_count": len(candidates),
        "resumed": False,
    }


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "source_selection.json"
    if selection_path.exists() and not args.reselect:
        selected = json.loads(selection_path.read_text(encoding="utf-8"))["videos"]
    else:
        selected = _select_sources(args)
        _json_dump(
            selection_path,
            {
                "schema": "joint_boundary_omni_source_selection_v1",
                "seed": args.seed,
                "source_root": str(Path(args.source_root).resolve()),
                "video_count": len(selected),
                "videos": selected,
            },
        )
    existing = {
        str(row["window_id"]): row
        for row in _read_jsonl(output / "source_windows.jsonl")
    }
    rows: list[dict[str, Any]] = []
    for video_position, source_row in enumerate(selected, start=1):
        source = Path(source_row["source_video"])
        video_id = str(source_row["video_id"])
        source_duration = float(source_row["source_duration_s"])
        rng = random.Random(f"{args.seed}:{video_id}")
        starts = _window_starts(
            duration_s=source_duration,
            window_s=args.window_s,
            count=args.windows_per_video,
            rng=rng,
        )
        for window_index, start_s in enumerate(starts):
            duration_s = min(args.window_s, source_duration - start_s)
            window_id = f"{video_id}-w{window_index:02d}"
            wav_path = output / "audio_wav" / f"{window_id}.wav"
            mp3_path = output / "omni_mp3_32k" / f"{window_id}.mp3"
            feature_dir = output / "features" / window_id
            previous = existing.get(window_id)
            if previous and all(
                Path(previous[key]).exists()
                for key in (
                    "audio_wav",
                    "omni_mp3_32k",
                    "semantic_split_features",
                    "semantic_split_metadata",
                    "speech_sequence_features",
                    "pre_asr_candidates",
                )
            ):
                rows.append(previous)
                print(
                    f"resumed video={video_position}/{len(selected)} window={window_id}",
                    flush=True,
                )
                continue
            _extract_window(
                source=source,
                start_s=start_s,
                duration_s=duration_s,
                wav_path=wav_path,
                mp3_path=mp3_path,
            )
            feature_payload: dict[str, Any] = {}
            if not args.skip_boundary:
                feature_payload = _export_boundary_features(
                    wav_path=wav_path,
                    feature_dir=feature_dir,
                )
            row = {
                "schema": SCHEMA,
                "window_id": window_id,
                "video_id": video_id,
                "source_video": str(source.resolve()),
                "source_duration_s": source_duration,
                "source_start_s": start_s,
                "source_end_s": round(start_s + duration_s, 6),
                "duration_s": round(duration_s, 6),
                "audio_wav": str(wav_path.resolve()),
                "audio_wav_sha256": _sha256(wav_path),
                "omni_mp3_32k": str(mp3_path.resolve()),
                "omni_mp3_32k_sha256": _sha256(mp3_path),
                "omni_audio_bitrate": "32k",
                "sample_rate": 16000,
                "channels": 1,
                "audio_filter": build_audio_filter_chain(),
                **feature_payload,
            }
            rows.append(row)
            _write_jsonl(output / "source_windows.jsonl", rows)
            print(
                f"prepared video={video_position}/{len(selected)} "
                f"window={window_id} duration={duration_s:.1f}s",
                flush=True,
            )
            gc.collect()
    _write_jsonl(output / "source_windows.jsonl", rows)
    _json_dump(
        output / "summary.json",
        {
            "schema": "joint_boundary_omni_preparation_summary_v1",
            "seed": args.seed,
            "video_count": len(selected),
            "window_count": len(rows),
            "total_audio_s": round(sum(float(row["duration_s"]) for row in rows), 3),
            "window_s": args.window_s,
            "windows_per_video": args.windows_per_video,
            "omni_audio_format": "mp3",
            "omni_audio_bitrate": "32k",
            "training_audio_format": "wav",
            "sample_rate": 16000,
            "channels": 1,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically sample video windows, preserve runtime-equivalent "
            "16k mono WAV, make 32k MP3 Omni inputs, and export boundary features."
        )
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--output-dir",
        default="datasets/train/omni-joint-boundary-preasr-v1",
    )
    parser.add_argument("--video-count", type=int, default=30)
    parser.add_argument("--windows-per-video", type=int, default=2)
    parser.add_argument("--window-s", type=float, default=75.0)
    parser.add_argument("--min-video-duration-s", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument(
        "--exclude-contains",
        action="append",
        default=["FJIN-059", "BONY-173", "NAMH-055", "867HTTM-0045"],
    )
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--reselect", action="store_true")
    args = parser.parse_args()
    if args.video_count <= 0:
        parser.error("--video-count must be positive")
    if args.windows_per_video <= 0:
        parser.error("--windows-per-video must be positive")
    if args.window_s <= 1.0:
        parser.error("--window-s must be greater than one second")
    return args


if __name__ == "__main__":
    run(parse_args())
