from __future__ import annotations

import logging
import os
import re
import subprocess
import time
import wave

from vad.base import SegmentationResult, SpeechSegment

log = logging.getLogger(__name__)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
log.setLevel(logging.INFO)
log.propagate = False

_SEGMENT_MIN_SILENCE_S = float(os.getenv("SEGMENT_MIN_SILENCE", "0.35"))
_SEGMENT_MIN_CHUNK_S = float(os.getenv("SEGMENT_MIN_CHUNK", "1.2"))
_SEGMENT_MAX_CHUNK_S = float(os.getenv("SEGMENT_MAX_CHUNK", "18.0"))
_SEGMENT_TARGET_CHUNK_S = min(
    _SEGMENT_MAX_CHUNK_S,
    float(os.getenv("SEGMENT_TARGET_CHUNK", str(min(60.0, _SEGMENT_MAX_CHUNK_S)))),
)
_SEGMENT_MIN_SPEECH_S = float(os.getenv("SEGMENT_MIN_SPEECH", "0.25"))
_SEGMENT_PAD_S = float(os.getenv("SEGMENT_PAD", "0.15"))
_SEGMENT_CUT_MIN_SILENCE_S = float(os.getenv("SEGMENT_CUT_MIN_SILENCE", "0.5"))
_SEGMENT_SILENCE_DB = os.getenv("SEGMENT_SILENCE_DB", "-32dB").strip()

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


def _get_wav_duration(audio_path: str) -> float:
    with wave.open(audio_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    return frames / rate if rate else 0.0


def _run_silence_detect(audio_path: str) -> list[tuple[float, float]]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        audio_path,
        "-af",
        f"silencedetect=noise={_SEGMENT_SILENCE_DB}:d={_SEGMENT_MIN_SILENCE_S}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in {0, 255}:
        raise RuntimeError(result.stderr.strip() or "ffmpeg silencedetect failed")

    silences: list[tuple[float, float]] = []
    current_start: float | None = None
    for line in result.stderr.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            current_start = float(start_match.group(1))
            continue

        end_match = _SILENCE_END_RE.search(line)
        if end_match and current_start is not None:
            silences.append(
                (max(0.0, current_start), max(0.0, float(end_match.group(1))))
            )
            current_start = None

    return silences


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []

    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _speech_spans_from_silences(
    duration: float,
    silences: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if duration <= 0:
        return []

    if not silences:
        return [(0.0, duration)]

    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for silence_start, silence_end in silences:
        speech_start = max(0.0, cursor - _SEGMENT_PAD_S)
        speech_end = min(duration, silence_start + _SEGMENT_PAD_S)
        if speech_end - speech_start >= _SEGMENT_MIN_SPEECH_S:
            spans.append((speech_start, speech_end))
        cursor = silence_end

    tail_start = max(0.0, cursor - _SEGMENT_PAD_S)
    if duration - tail_start >= _SEGMENT_MIN_SPEECH_S:
        spans.append((tail_start, duration))

    return _merge_spans(spans)


def _split_long_span(start: float, end: float) -> list[tuple[float, float]]:
    if end - start <= _SEGMENT_MAX_CHUNK_S:
        return [(start, end)]

    chunks: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + _SEGMENT_MAX_CHUNK_S)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def _group_speech_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []

    chunks: list[tuple[float, float]] = []
    current_start, current_end = spans[0]

    for start, end in spans[1:]:
        current_duration = current_end - current_start
        gap_after_current = max(0.0, start - current_end)
        proposed_duration = end - current_start

        if (
            current_duration >= _SEGMENT_TARGET_CHUNK_S
            and gap_after_current >= _SEGMENT_CUT_MIN_SILENCE_S
        ):
            chunks.extend(_split_long_span(current_start, current_end))
            current_start, current_end = start, end
            continue

        if (
            current_duration < _SEGMENT_MIN_CHUNK_S
            or proposed_duration <= _SEGMENT_MAX_CHUNK_S
        ):
            current_end = end
            continue

        chunks.extend(_split_long_span(current_start, current_end))
        current_start, current_end = start, end

    chunks.extend(_split_long_span(current_start, current_end))

    if len(chunks) >= 2:
        last_start, last_end = chunks[-1]
        prev_start, prev_end = chunks[-2]
        if (
            last_end - last_start < _SEGMENT_MIN_CHUNK_S
            and last_end - prev_start <= _SEGMENT_MAX_CHUNK_S * 1.25
        ):
            chunks[-2] = (prev_start, last_end)
            chunks.pop()

    return chunks


class FfmpegSilencedetectBackend:
    name = "ffmpeg_silencedetect"

    def signature(self) -> dict:
        return {
            "backend": self.name,
            "min_silence": _SEGMENT_MIN_SILENCE_S,
            "silence_db": _SEGMENT_SILENCE_DB,
            "pad": _SEGMENT_PAD_S,
            "min_speech": _SEGMENT_MIN_SPEECH_S,
            "min_chunk": _SEGMENT_MIN_CHUNK_S,
            "max_chunk": _SEGMENT_MAX_CHUNK_S,
            "target_chunk": _SEGMENT_TARGET_CHUNK_S,
            "cut_min_silence": _SEGMENT_CUT_MIN_SILENCE_S,
        }

    def segment(self, audio_path: str, *, target_sr: int = 16000) -> SegmentationResult:
        _ = target_sr
        t0 = time.monotonic()
        duration = _get_wav_duration(audio_path)
        silences = _run_silence_detect(audio_path)
        speech_spans = _speech_spans_from_silences(duration, silences)
        chunks = _group_speech_spans(speech_spans)
        elapsed = time.monotonic() - t0

        segments = [SpeechSegment(start=start, end=end) for start, end in chunks]
        groups = [[segment] for segment in segments]
        mean_dur = (
            sum(segment.end - segment.start for segment in segments) / len(segments)
            if segments
            else 0.0
        )
        log.info(
            "[vad] backend=ffmpeg_silencedetect chunks=%d mean_dur=%.2fs",
            len(groups),
            mean_dur,
        )
        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=duration,
            parameters=self.signature(),
            processing_time_sec=elapsed,
        )

