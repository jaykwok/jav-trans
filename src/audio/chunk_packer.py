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
    left_padding_s: float
    right_padding_s: float
    split_reason: str


@dataclass(frozen=True)
class FramePackingConfig:
    frame_hop_s: float = 1.0 / 29.97
    window_frames: int = 899
    reserve_frames: int = 45
    target_padding_frames: int = 60
    gap_merge_frames: int = 45

    @property
    def chunk_cap_s(self) -> float:
        return max(1, self.window_frames - self.reserve_frames) * self.frame_hop_s

    @property
    def target_padding_s(self) -> float:
        return max(0, self.target_padding_frames) * self.frame_hop_s

    @property
    def gap_merge_s(self) -> float:
        return max(0, self.gap_merge_frames) * self.frame_hop_s


@dataclass(frozen=True)
class _PackSegment:
    start: float
    end: float
    score: float | None = None
    split_left: bool = False
    split_right: bool = False

    def to_speech_segment(self) -> SpeechSegment:
        return SpeechSegment(start=self.start, end=self.end, score=self.score)


def pack_vad_segments(
    segments: Sequence[SpeechSegment],
    *,
    frame_hop_s: float = 1.0 / 29.97,
    window_frames: int = 899,
    reserve_frames: int = 45,
    target_padding_frames: int = 60,
    gap_merge_frames: int = 45,
) -> list[PackedChunk]:
    """Pack VAD speech into frame-derived ASR chunks with dynamic gap-aware padding."""
    config = FramePackingConfig(
        frame_hop_s=frame_hop_s,
        window_frames=window_frames,
        reserve_frames=reserve_frames,
        target_padding_frames=target_padding_frames,
        gap_merge_frames=gap_merge_frames,
    )
    _validate_config(config)

    ordered_speech_segments = sorted(segments, key=lambda segment: (segment.start, segment.end))
    _validate_segments(ordered_speech_segments)
    ordered_segments = _split_overlong_segments(ordered_speech_segments, config=config)

    chunks: list[PackedChunk] = []
    current: list[_PackSegment] = []

    for index, segment in enumerate(ordered_segments):
        if not current:
            current = [segment]
            continue

        gap_s = segment.start - current[-1].end
        proposed = [*current, segment]
        if gap_s <= config.gap_merge_s and _core_duration(proposed) <= config.chunk_cap_s:
            current = proposed
            continue

        next_start = segment.start
        chunks.append(
            _make_chunk(
                current,
                config=config,
                previous_end=chunks[-1].vad_segments[-1].end if chunks else None,
                next_start=next_start,
                split_reason="gap" if gap_s > config.gap_merge_s else "capacity",
            )
        )
        current = [segment]

    if current:
        chunks.append(
            _make_chunk(
                current,
                config=config,
                previous_end=chunks[-1].vad_segments[-1].end if chunks else None,
                next_start=None,
                split_reason="tail",
            )
        )

    return chunks


def _validate_config(config: FramePackingConfig) -> None:
    if config.frame_hop_s <= 0:
        raise ValueError("frame_hop_s must be positive")
    if config.window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if config.reserve_frames < 0:
        raise ValueError("reserve_frames must be non-negative")
    if config.reserve_frames >= config.window_frames:
        raise ValueError("reserve_frames must be smaller than window_frames")
    if config.target_padding_frames < 0:
        raise ValueError("target_padding_frames must be non-negative")
    if config.gap_merge_frames < 0:
        raise ValueError("gap_merge_frames must be non-negative")


def _validate_segments(segments: Sequence[SpeechSegment]) -> None:
    for segment in segments:
        if segment.end < segment.start:
            raise ValueError("segment end must be greater than or equal to start")


def _split_overlong_segments(
    segments: Sequence[SpeechSegment],
    *,
    config: FramePackingConfig,
) -> list[_PackSegment]:
    speech_limit_s = config.chunk_cap_s - 2.0 * config.target_padding_s
    if speech_limit_s <= 0.0:
        speech_limit_s = config.chunk_cap_s
    split: list[_PackSegment] = []
    for segment in segments:
        duration_s = segment.end - segment.start
        if duration_s <= speech_limit_s:
            split.append(_PackSegment(start=segment.start, end=segment.end, score=segment.score))
            continue

        cursor = segment.start
        first = True
        while cursor < segment.end:
            next_end = min(segment.end, cursor + speech_limit_s)
            split.append(
                _PackSegment(
                    start=cursor,
                    end=next_end,
                    score=segment.score,
                    split_left=not first,
                    split_right=next_end < segment.end,
                )
            )
            cursor = next_end
            first = False
    return split


def _core_duration(segments: Sequence[_PackSegment]) -> float:
    if not segments:
        return 0.0
    return max(0.0, segments[-1].end - segments[0].start)


def _make_chunk(
    segments: Sequence[_PackSegment],
    *,
    config: FramePackingConfig,
    previous_end: float | None,
    next_start: float | None,
    split_reason: str,
) -> PackedChunk:
    core_start = segments[0].start
    core_end = segments[-1].end
    core_duration = max(0.0, core_end - core_start)
    remaining = max(0.0, config.chunk_cap_s - core_duration)
    target = min(config.target_padding_s, remaining / 2.0)

    if segments[0].split_left:
        left_limit = config.target_padding_s
    else:
        left_gap = core_start if previous_end is None else max(0.0, core_start - previous_end)
        left_limit = left_gap / 2.0 if previous_end is not None else left_gap

    if segments[-1].split_right:
        right_limit = config.target_padding_s
    else:
        right_gap = config.target_padding_s * 2.0
        if next_start is not None:
            right_gap = max(0.0, next_start - core_end)
        right_limit = right_gap / 2.0 if next_start is not None else right_gap
    left_padding = min(target, left_limit)
    right_padding = min(target, right_limit)

    unused = max(0.0, remaining - left_padding - right_padding)
    if unused > 0.0:
        extra_right = min(config.target_padding_s - right_padding, right_limit - right_padding, unused)
        if extra_right > 0.0:
            right_padding += extra_right
            unused -= extra_right
        extra_left = min(config.target_padding_s - left_padding, left_limit - left_padding, unused)
        if extra_left > 0.0:
            left_padding += extra_left

    start = max(0.0, core_start - left_padding)
    end = core_end + right_padding
    if any(segment.split_left or segment.split_right for segment in segments):
        split_reason = "overlong"
    return PackedChunk(
        start=start,
        end=end,
        vad_segments=[segment.to_speech_segment() for segment in segments],
        duration=end - start,
        left_padding_s=left_padding,
        right_padding_s=right_padding,
        split_reason=split_reason,
    )
