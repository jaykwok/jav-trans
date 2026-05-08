from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import xxhash

_AUDIO_SAMPLE_RATE = max(8000, int(os.getenv("AUDIO_SAMPLE_RATE", "16000")))
_AUDIO_CHANNELS = max(1, int(os.getenv("AUDIO_CHANNELS", "1")))
_AUDIO_BASE_FILTER = os.getenv("AUDIO_FILTER", "highpass=f=70,lowpass=f=7600").strip()
_AUDIO_CACHE_SAMPLE_BYTES = 4 * 1024 * 1024
_AUDIO_DYNAUDNORM_FILTER = "agate=threshold=0.01,dynaudnorm=f=250:g=15"
_AUDIO_USE_LOUDNORM = os.getenv("AUDIO_USE_LOUDNORM", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_AUDIO_DYNAUDNORM = os.getenv("AUDIO_DYNAUDNORM", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def build_audio_filter_chain() -> str:
    dynaudnorm_enabled = os.getenv(
        "AUDIO_DYNAUDNORM",
        "1" if _AUDIO_DYNAUDNORM else "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    loudnorm_enabled = os.getenv(
        "AUDIO_USE_LOUDNORM",
        "1" if _AUDIO_USE_LOUDNORM else "0",
    ).strip().lower() in {"1", "true", "yes", "on"}

    filters = []
    if _AUDIO_BASE_FILTER:
        filters.append(_AUDIO_BASE_FILTER)
    if dynaudnorm_enabled:
        filters.append(_AUDIO_DYNAUDNORM_FILTER)
        if loudnorm_enabled:
            print("[WARN] AUDIO_USE_LOUDNORM ignored when AUDIO_DYNAUDNORM=1")
    elif loudnorm_enabled:
        filters.append("loudnorm=I=-16:LRA=11:TP=-1.5")
    return ",".join(filters)


def video_content_sample_hash(video_path: str) -> str:
    path = Path(video_path)
    size = path.stat().st_size
    hasher = xxhash.xxh3_64()
    with path.open("rb") as reader:
        if size <= _AUDIO_CACHE_SAMPLE_BYTES * 2:
            hasher.update(reader.read())
        else:
            hasher.update(reader.read(_AUDIO_CACHE_SAMPLE_BYTES))
            reader.seek(size - _AUDIO_CACHE_SAMPLE_BYTES)
            hasher.update(reader.read(_AUDIO_CACHE_SAMPLE_BYTES))
    return hasher.hexdigest()


def get_audio_cache_key(video_path: str) -> str:
    video_hash = video_content_sample_hash(video_path)
    payload = (
        f"{_AUDIO_SAMPLE_RATE}|{_AUDIO_CHANNELS}|"
        f"{build_audio_filter_chain()}|{video_hash}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def extract_audio(video_path: str, out_path: str) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(_AUDIO_SAMPLE_RATE),
        "-ac",
        str(_AUDIO_CHANNELS),
    ]
    filter_chain = build_audio_filter_chain()
    if filter_chain:
        command.extend(["-af", filter_chain])
    command.extend([out_path, "-loglevel", "error"])
    subprocess.run(command, check=True)


def probe_video_duration_s(video_path: str) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        duration = float(completed.stdout.strip())
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError):
        return None
    return duration if duration > 0 else None

