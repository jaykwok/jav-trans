from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import xxhash

from utils.subprocess_tools import no_window_subprocess_kwargs

_AUDIO_SAMPLE_RATE = max(8000, int(os.getenv("AUDIO_SAMPLE_RATE", "16000")))
_AUDIO_CHANNELS = max(1, int(os.getenv("AUDIO_CHANNELS", "1")))
_AUDIO_BASE_FILTER = os.getenv("AUDIO_FILTER", "highpass=f=70,lowpass=f=7600").strip()
_AUDIO_CACHE_SAMPLE_BYTES = 4 * 1024 * 1024
_AUDIO_DYNAUDNORM_FILTER = "agate=threshold=0.01,dynaudnorm=f=250:g=15"
_DEFAULT_AUDIO_EXTRACT_TIMEOUT_S = 3600.0
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
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise OSError(f"Cannot stat video for audio cache key: {path}") from exc
    hasher = xxhash.xxh3_64()
    try:
        with path.open("rb") as reader:
            if size <= _AUDIO_CACHE_SAMPLE_BYTES * 2:
                hasher.update(reader.read())
            else:
                hasher.update(reader.read(_AUDIO_CACHE_SAMPLE_BYTES))
                reader.seek(size - _AUDIO_CACHE_SAMPLE_BYTES)
                hasher.update(reader.read(_AUDIO_CACHE_SAMPLE_BYTES))
    except OSError as exc:
        raise OSError(f"Cannot sample video for audio cache key: {path}") from exc
    return hasher.hexdigest()


def get_audio_cache_key(video_path: str) -> str:
    video_hash = video_content_sample_hash(video_path)
    payload = (
        f"{_AUDIO_SAMPLE_RATE}|{_AUDIO_CHANNELS}|"
        f"{build_audio_filter_chain()}|{video_hash}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _audio_extract_timeout_s(video_path: str) -> float:
    override = os.getenv("AUDIO_EXTRACT_TIMEOUT_S", "").strip()
    if override:
        try:
            parsed = float(override)
        except ValueError as exc:
            raise ValueError(f"AUDIO_EXTRACT_TIMEOUT_S must be a positive number, got {override!r}") from exc
        if parsed <= 0:
            raise ValueError(f"AUDIO_EXTRACT_TIMEOUT_S must be positive, got {override!r}")
        return parsed
    duration = probe_video_duration_s(video_path)
    if duration:
        return max(300.0, duration * 4.0 + 120.0)
    return _DEFAULT_AUDIO_EXTRACT_TIMEOUT_S


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
    timeout_s = _audio_extract_timeout_s(video_path)
    try:
        subprocess.run(
            command,
            check=True,
            timeout=timeout_s,
            **no_window_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"ffmpeg audio extraction timed out after {timeout_s:.1f}s: {video_path}") from exc


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
            **no_window_subprocess_kwargs(),
        )
        duration = float(completed.stdout.strip())
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError):
        return None
    return duration if duration > 0 else None
