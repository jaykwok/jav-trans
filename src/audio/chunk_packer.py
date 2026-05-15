from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from vad.base import SpeechSegment


@dataclass(frozen=True)
class PackedChunk:
    start: float
    end: float
    vad_segments: list[SpeechSegment]
    duration: float


def pack_vad_segments(
    segments: Sequence[SpeechSegment],
    *,
    max_s: float = 28.0,
    gap_merge_s: float = 1.5,
    padding_s: float = 2.0,
) -> list[PackedChunk]:
    """Greedily pack VAD speech segments into padded ASR chunks."""
    if max_s <= 0:
        raise ValueError("max_s must be positive")
    if gap_merge_s < 0:
        raise ValueError("gap_merge_s must be non-negative")
    if padding_s < 0:
        raise ValueError("padding_s must be non-negative")

    ordered_segments = sorted(segments, key=lambda segment: (segment.start, segment.end))
    _validate_segments(ordered_segments)

    chunks: list[PackedChunk] = []
    current: list[SpeechSegment] = []

    for segment in ordered_segments:
        if not current:
            current = [segment]
            continue

        gap_s = segment.start - current[-1].end
        proposed = [*current, segment]
        if gap_s <= gap_merge_s and _padded_duration(proposed, padding_s) <= max_s:
            current = proposed
            continue

        chunks.append(_make_chunk(current, padding_s))
        current = [segment]

    if current:
        chunks.append(_make_chunk(current, padding_s))

    return chunks


def _validate_segments(segments: Sequence[SpeechSegment]) -> None:
    for segment in segments:
        if segment.end < segment.start:
            raise ValueError("segment end must be greater than or equal to start")


def _padded_duration(segments: Sequence[SpeechSegment], padding_s: float) -> float:
    return _padded_end(segments, padding_s) - _padded_start(segments, padding_s)


def _make_chunk(segments: Sequence[SpeechSegment], padding_s: float) -> PackedChunk:
    start = _padded_start(segments, padding_s)
    end = _padded_end(segments, padding_s)
    return PackedChunk(
        start=start,
        end=end,
        vad_segments=list(segments),
        duration=end - start,
    )


def _padded_start(segments: Sequence[SpeechSegment], padding_s: float) -> float:
    return max(0.0, segments[0].start - padding_s)


def _padded_end(segments: Sequence[SpeechSegment], padding_s: float) -> float:
    return segments[-1].end + padding_s
