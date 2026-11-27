from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

import xxhash

from core import atomic_io
from core.cancellation import PipelineCancelledError, cancel_requested
from core.diagnostics import format_exit_code
from core.typed_config import env_bool, env_float, env_int, env_text
from utils.subprocess_tools import SubprocessCancelledError, run_cancellable

logger = logging.getLogger(__name__)

_AUDIO_CACHE_SAMPLE_BYTES = 4 * 1024 * 1024
_AUDIO_DYNAUDNORM_FILTER = "agate=threshold=0.01,dynaudnorm=f=250:g=15"
_DEFAULT_AUDIO_EXTRACT_TIMEOUT_S = 3600.0
# ffprobe reads a header; when it does not answer in a minute it is stuck on
# unreadable media, not working. Unbounded, it blocked the whole job forever.
_DEFAULT_PROBE_TIMEOUT_S = 60.0
_MEDIA_CANCEL_POLL_S = 0.25
# Extraction writes here and is published by an atomic replace. The suffix is
# what the cache reader skips, and the random middle keeps two jobs extracting
# the same input from writing to each other's file.
_PARTIAL_SUFFIX = ".partial"
_PARTIAL_GLOB = f"*{_PARTIAL_SUFFIX}"
_STALE_PARTIAL_AGE_S = 1800.0
_WAV_SAMPLE_WIDTH = 2


def audio_sample_rate() -> int:
    return env_int("AUDIO_SAMPLE_RATE", 16000, minimum=8000)


def audio_channels() -> int:
    return env_int("AUDIO_CHANNELS", 1, minimum=1)


def audio_base_filter() -> str:
    return env_text("AUDIO_FILTER", "highpass=f=70,lowpass=f=7600")


def audio_timeline_filter(*, sample_rate: int | None = None) -> str:
    # PCM WAV has no discontinuous timestamp axis. Without async resampling, ffmpeg
    # concatenates decoded packets across edit-list / PTS gaps and silently shortens
    # the ASR audio. Every CTC timestamp after such a gap then becomes early against
    # the source video. Fill/drop samples against input PTS before any other audio
    # processing so sample index / sample_rate remains the video's time coordinate.
    rate = audio_sample_rate() if sample_rate is None else int(sample_rate)
    return f"aresample={rate}:async=1000:first_pts=0"


def audio_use_loudnorm() -> bool:
    return env_bool("AUDIO_USE_LOUDNORM", False)


def audio_use_dynaudnorm() -> bool:
    return env_bool("AUDIO_DYNAUDNORM", True)


def build_audio_filter_chain(*, sample_rate: int | None = None) -> str:
    dynaudnorm_enabled = audio_use_dynaudnorm()
    loudnorm_enabled = audio_use_loudnorm()
    base_filter = audio_base_filter()

    filters = [audio_timeline_filter(sample_rate=sample_rate)]
    if base_filter:
        filters.append(base_filter)
    if dynaudnorm_enabled:
        filters.append(_AUDIO_DYNAUDNORM_FILTER)
        if loudnorm_enabled:
            logger.warning("AUDIO_USE_LOUDNORM ignored when AUDIO_DYNAUDNORM=1")
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
    sample_rate = audio_sample_rate()
    channels = audio_channels()
    payload = (
        f"{sample_rate}|{channels}|"
        f"{build_audio_filter_chain(sample_rate=sample_rate)}|{video_hash}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _audio_extract_timeout_s(
    video_path: str,
    *,
    cancel_event: threading.Event | None = None,
    duration_s: float | None = None,
) -> float:
    # 0 (or unset) means "derive it from the video's own duration", which is
    # what the caller wants far more often than a fixed number. A value that
    # cannot be a timeout is reported as a config problem and derived instead of
    # aborting the extraction - the run is not wrong, the setting is.
    override = env_float("AUDIO_EXTRACT_TIMEOUT_S", 0.0, minimum=0.0)
    if override > 0:
        return override
    duration = (
        float(duration_s)
        if duration_s is not None and duration_s > 0
        else probe_video_duration_s(video_path, cancel_event=cancel_event)
    )
    if duration:
        return max(300.0, duration * 4.0 + 120.0)
    return _DEFAULT_AUDIO_EXTRACT_TIMEOUT_S


def _partial_audio_path(out_path: str) -> str:
    return f"{out_path}.{uuid.uuid4().hex[:8]}{_PARTIAL_SUFFIX}"


def _discard_partial(path: str) -> None:
    """Best-effort disk hygiene only - correctness comes from the naming."""
    try:
        os.remove(path)
    except OSError:
        pass


def _wav_header(path: str) -> tuple[int, int, int, int] | None:
    """(channels, sample_rate, sample_width, frames), or None if unreadable."""
    try:
        with wave.open(path, "rb") as reader:
            return (
                reader.getnchannels(),
                reader.getframerate(),
                reader.getsampwidth(),
                reader.getnframes(),
            )
    except Exception:
        return None


def audio_file_is_complete(path: str) -> bool:
    """Would this file decode as the audio we asked ffmpeg for?

    ffmpeg writes the RIFF/data sizes as placeholders and patches them when it
    closes the file, so a killed extraction leaves a large file whose header
    claims zero (or 0xFFFFFFFF) frames. That is the signature checked here,
    together with the format the ASR stage expects.
    """
    header = _wav_header(path)
    if header is None:
        return False
    channels, sample_rate, sample_width, frames = header
    if channels != audio_channels() or sample_rate != audio_sample_rate():
        return False
    if sample_width != _WAV_SAMPLE_WIDTH or frames <= 0:
        return False
    declared_bytes = frames * channels * sample_width
    try:
        actual_bytes = os.path.getsize(path)
    except OSError:
        return False
    # The header is 44 bytes canonically; ffmpeg may add a small LIST/INFO chunk.
    return declared_bytes <= actual_bytes <= declared_bytes + 4096


def audio_cache_is_usable(path: str) -> bool:
    """Is this cached extraction safe to reuse instead of re-extracting?

    Deliberately narrower than `audio_file_is_complete`: it rejects empty files
    and any WAV that ffmpeg started and never finished. Files published before
    this project wrote through a partial (and non-empty non-RIFF/WAVE files)
    are left to the ASR stage to accept or reject, so this check can only heal
    truncation / zero-byte files, never invent a new way to miss the cache.
    """
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as reader:
            prefix = reader.read(12)
    except OSError:
        return False
    if not prefix:
        return False
    if not (prefix[:4] == b"RIFF" and prefix[8:12] == b"WAVE"):
        return True
    return audio_file_is_complete(path)


def publish_audio_file(temp_path: str, out_path: str) -> None:
    """Make a validated extraction visible under its cache name, atomically."""
    atomic_io.replace_atomically(temp_path, out_path, retries=5, base_delay_s=0.05)


def reclaim_stale_partials(root: str | Path, *, max_age_s: float | None = None) -> int:
    """Delete leftover partials under `root` that no live extraction can own.

    Age, not "is this run mine": a partial being written right now by another
    process (a CLI run beside the app) has a fresh mtime, and on Windows an
    open file cannot be unlinked anyway.
    """
    age_limit = _STALE_PARTIAL_AGE_S if max_age_s is None else float(max_age_s)
    base = Path(root)
    if not base.is_dir():
        return 0
    now = time.time()
    removed = 0
    for path in base.rglob(_PARTIAL_GLOB):
        try:
            if not path.is_file() or now - path.stat().st_mtime < age_limit:
                continue
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def extract_audio(
    video_path: str,
    out_path: str,
    *,
    cancel_event: threading.Event | None = None,
    duration_s: float | None = None,
) -> None:
    """Extract to a partial file and publish it only once it is complete.

    Nothing under the cache name is ever half-written: cancellation, a timeout,
    a crash or a hard kill of this process can only leave a `.partial` behind,
    which the cache reader does not look at. Deleting those is disk hygiene, not
    correctness - which matters, because a hard kill runs no cleanup at all.
    """
    sample_rate = audio_sample_rate()
    channels = audio_channels()
    temp_path = _partial_audio_path(out_path)
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
    ]
    filter_chain = build_audio_filter_chain(sample_rate=sample_rate)
    if filter_chain:
        command.extend(["-af", filter_chain])
    # -f wav explicitly: the muxer used to be chosen from the ".wav" extension,
    # and the file being written is now named ".partial".
    command.extend(["-f", "wav", temp_path, "-loglevel", "error"])
    timeout_s = _audio_extract_timeout_s(
        video_path,
        cancel_event=cancel_event,
        duration_s=duration_s,
    )
    try:
        run_cancellable(
            command,
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout_s=timeout_s,
            cancel_requested=lambda: cancel_requested(cancel_event),
            poll_interval_s=_MEDIA_CANCEL_POLL_S,
        )
        if not audio_file_is_complete(temp_path):
            raise RuntimeError(
                "ffmpeg exited successfully but produced an unusable wav "
                f"(expected {sample_rate}Hz/{channels}ch PCM16): "
                f"{video_path}"
            )
        publish_audio_file(temp_path, out_path)
    except SubprocessCancelledError as exc:
        raise PipelineCancelledError("任务已取消") from exc
    except subprocess.CalledProcessError as exc:
        stderr_tail = (exc.stderr or "").strip()
        if len(stderr_tail) > 1000:
            stderr_tail = stderr_tail[-1000:]
        logger.error(
            "ffmpeg audio extraction failed (exit=%s) for %s: %s",
            format_exit_code(exc.returncode),
            os.path.basename(video_path),
            stderr_tail or "(no stderr output)",
        )
        raise
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"ffmpeg audio extraction timed out after {timeout_s:.1f}s: {video_path}"
        ) from exc
    finally:
        _discard_partial(temp_path)


def _probe_timeout_s() -> float:
    override = env_float("FFPROBE_TIMEOUT_S", 0.0, minimum=0.0)
    return override if override > 0 else _DEFAULT_PROBE_TIMEOUT_S


def probe_video_duration_s(
    video_path: str,
    *,
    cancel_event: threading.Event | None = None,
) -> float | None:
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
        completed = run_cancellable(
            command,
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout_s=_probe_timeout_s(),
            cancel_requested=lambda: cancel_requested(cancel_event),
            poll_interval_s=_MEDIA_CANCEL_POLL_S,
        )
        duration = float(completed.stdout.strip())
    except SubprocessCancelledError as exc:
        # Cancellation is not "this video has no duration": swallowing it here
        # would let the caller go on to a multi-hour extraction.
        raise PipelineCancelledError("任务已取消") from exc
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        TypeError,
        ValueError,
    ) as exc:
        stderr_detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            stderr_detail = f" stderr={str(exc.stderr).strip()[:300]}"
        logger.warning(
            "ffprobe duration probe failed for %s: %s: %s%s",
            os.path.basename(video_path),
            type(exc).__name__,
            exc,
            stderr_detail,
        )
        return None
    return duration if duration > 0 else None
