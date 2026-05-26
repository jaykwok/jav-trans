#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def safe_stem(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned.strip("_") or "video"


def discover_videos(paths: Iterable[str], *, recursive: bool) -> list[Path]:
    videos: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path.resolve())
            continue
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            videos.extend(item.resolve() for item in iterator if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS)
    return sorted(dict.fromkeys(videos), key=lambda item: item.as_posix())


def ffprobe_duration_s(video_path: Path, *, ffprobe_bin: str) -> float:
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def choose_start_times(
    *,
    duration_s: float,
    clip_duration_s: float,
    count: int,
    seed: int,
    exclude_head_s: float,
    exclude_tail_s: float,
) -> list[float]:
    import numpy as np

    if count <= 0:
        return []
    latest = max(0.0, duration_s - clip_duration_s)
    lower = min(max(0.0, exclude_head_s), latest)
    upper = max(lower, min(latest, duration_s - exclude_tail_s - clip_duration_s))
    if upper <= lower:
        lower = 0.0
        upper = latest
    if upper <= lower:
        return [0.0]
    rng = np.random.default_rng(seed)
    starts = sorted(float(value) for value in rng.uniform(lower, upper, size=count))
    return [round(value, 3) for value in starts]


def extract_wav_clip(
    *,
    video_path: Path,
    output_path: Path,
    start_s: float,
    duration_s: float,
    ffmpeg_bin: str,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration_s:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def stable_audio_id(*, video_path: Path, start_s: float, duration_s: float, prefix: str) -> str:
    digest = hashlib.sha1(f"{video_path.resolve()}:{start_s:.3f}:{duration_s:.3f}".encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{safe_stem(video_path.stem)[:48]}-{int(round(start_s * 1000)):010d}-{digest}"


def build_candidates(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = discover_videos(args.input, recursive=args.recursive)
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]
    if not videos:
        raise ValueError("no input videos found")

    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for video_index, video_path in enumerate(videos):
        try:
            video_duration_s = ffprobe_duration_s(video_path, ffprobe_bin=args.ffprobe_bin)
        except Exception as exc:
            errors.append({"video": project_relative(video_path), "reason": "ffprobe_error", "error": str(exc)})
            continue
        if not math.isfinite(video_duration_s) or video_duration_s <= 0.0:
            errors.append({"video": project_relative(video_path), "reason": "invalid_duration", "duration_s": video_duration_s})
            continue
        starts = choose_start_times(
            duration_s=video_duration_s,
            clip_duration_s=args.clip_duration_s,
            count=args.clips_per_video,
            seed=args.seed + video_index,
            exclude_head_s=args.exclude_head_s,
            exclude_tail_s=args.exclude_tail_s,
        )
        for clip_index, start_s in enumerate(starts):
            clip_duration_s = min(args.clip_duration_s, max(0.0, video_duration_s - start_s))
            if clip_duration_s < args.min_clip_duration_s:
                errors.append(
                    {
                        "video": project_relative(video_path),
                        "start_s": start_s,
                        "duration_s": clip_duration_s,
                        "reason": "clip_too_short",
                    }
                )
                continue
            audio_id = stable_audio_id(
                video_path=video_path,
                start_s=start_s,
                duration_s=clip_duration_s,
                prefix=args.audio_id_prefix,
            )
            audio_path = audio_dir / f"{audio_id}.wav"
            try:
                extract_wav_clip(
                    video_path=video_path,
                    output_path=audio_path,
                    start_s=start_s,
                    duration_s=clip_duration_s,
                    ffmpeg_bin=args.ffmpeg_bin,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                errors.append(
                    {
                        "video": project_relative(video_path),
                        "audio_id": audio_id,
                        "start_s": start_s,
                        "duration_s": clip_duration_s,
                        "reason": "ffmpeg_error",
                        "error": str(exc),
                    }
                )
                continue
            row = {
                "audio_id": audio_id,
                "audio": project_relative(audio_path),
                "duration_s": clip_duration_s,
                "source": args.source,
                "text": "",
                "reason": args.reason,
                "label_quality": "manual_pending",
                "video": project_relative(video_path),
                "video_duration_s": video_duration_s,
                "video_start_s": start_s,
                "video_end_s": start_s + clip_duration_s,
                "clip_index": clip_index,
                "teacher_segments": {},
            }
            rows.append(row)
            manifest_rows.append(
                {
                    "audio_id": audio_id,
                    "audio": project_relative(audio_path),
                    "duration_s": clip_duration_s,
                    "sample_rate": 16000,
                    "source": args.source,
                    "label_quality": "manual_pending",
                    "video": project_relative(video_path),
                    "video_start_s": start_s,
                    "video_end_s": start_s + clip_duration_s,
                }
            )

    candidates_path = output_dir / args.output_candidates
    manifest_path = output_dir / args.output_manifest
    errors_path = output_dir / "candidate_errors.json"
    summary_path = output_dir / "candidate_summary.json"
    with candidates_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    errors_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "inputs": args.input,
        "videos": len(videos),
        "clips": len(rows),
        "errors": len(errors),
        "clips_per_video": args.clips_per_video,
        "clip_duration_s": args.clip_duration_s,
        "exclude_head_s": args.exclude_head_s,
        "exclude_tail_s": args.exclude_tail_s,
        "seed": args.seed,
        "source": args.source,
        "reason_counts": dict(sorted(Counter(row["reason"] for row in rows).items())),
        "video_counts": dict(sorted(Counter(row["video"] for row in rows).items())),
        "candidates": str(candidates_path),
        "manifest": str(manifest_path),
        "errors_report": str(errors_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"candidates={candidates_path}")
    print(f"manifest={manifest_path}")
    print(f"errors={errors_path}")
    print(f"summary={summary_path}")
    print(f"clips={len(rows)} errors={len(errors)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local-video manual VAD audit candidates for FusionVAD-JA.")
    parser.add_argument("--input", action="append", required=True, help="Video file or directory. Repeatable.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--clips-per-video", type=int, default=8)
    parser.add_argument("--clip-duration-s", type=float, default=8.0)
    parser.add_argument("--min-clip-duration-s", type=float, default=1.0)
    parser.add_argument("--exclude-head-s", type=float, default=60.0)
    parser.add_argument("--exclude-tail-s", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--source", default="local-video-heldout")
    parser.add_argument("--reason", default="local_video_random")
    parser.add_argument("--audio-id-prefix", default="localvid")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "datasets" / "val" / "fusionvad-ja" / "v1-6" / "real-heldout-local-video-audit"),
    )
    parser.add_argument("--output-candidates", default="audit_candidates.jsonl")
    parser.add_argument("--output-manifest", default="manifest.json")
    args = parser.parse_args(argv)
    if args.limit_videos is not None and args.limit_videos <= 0:
        parser.error("--limit-videos must be positive")
    if args.clips_per_video <= 0:
        parser.error("--clips-per-video must be positive")
    if args.clip_duration_s <= 0:
        parser.error("--clip-duration-s must be positive")
    if args.min_clip_duration_s <= 0:
        parser.error("--min-clip-duration-s must be positive")
    if args.exclude_head_s < 0 or args.exclude_tail_s < 0:
        parser.error("--exclude-head-s and --exclude-tail-s must be non-negative")
    return args


if __name__ == "__main__":
    build_candidates(parse_args())
