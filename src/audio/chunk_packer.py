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
    parent_chunk_id: int | None = None
    island_id: int | None = None
    island_count: int | None = None
    core_start: float | None = None
    core_end: float | None = None
    internal_gap_count: int = 0
    internal_gap_max_s: float = 0.0
    split_policy: str = ""
    valley_split_count: int = 0
    valley_score_min: float | None = None
    cut_split_count: int = 0
    cut_score_max: float | None = None


@dataclass(frozen=True)
class FramePackingConfig:
    frame_hop_s: float = 1.0 / 29.97
    score_frame_hop_s: float | None = None
    window_frames: int = 899
    reserve_frames: int = 45
    target_padding_frames: int = 60
    gap_merge_frames: int = 45
    # Soft cap on packed core duration. 0 = disabled (only the hard chunk_cap
    # applies). When > 0, accumulation stops at the next inter-island gap once
    # the proposed core would exceed it, so high-recall VAD does not merge many
    # speech islands into one aligner-hostile super-chunk. See R14 Phase 1a.
    max_core_frames: int = 0
    # R15/R16 pre-ASR speech-island splitting. Disabled by default. Unlike the
    # R14 soft cap, this only splits already-packed high-risk chunks that contain
    # multiple islands and an explicit internal gap.
    pre_asr_island_split_enabled: bool = False
    pre_asr_island_split_min_core_frames: int = 420
    pre_asr_island_split_min_gap_frames: int = 18
    pre_asr_island_split_min_island_frames: int = 3
    pre_asr_island_split_max_children: int = 8
    # R16 opt-in: split a long continuous positive island at low-score valleys
    # before ASR. This only runs when per-frame VAD scores are supplied.
    pre_asr_valley_split_enabled: bool = False
    pre_asr_valley_split_min_core_frames: int = 420
    pre_asr_valley_split_target_core_frames: int = 270
    pre_asr_valley_split_min_valley_frames: int = 6
    pre_asr_valley_split_min_child_frames: int = 45
    pre_asr_valley_split_max_children: int = 8
    pre_asr_valley_split_threshold: float = 0.20
    # R17 opt-in: split a long continuous positive island at high endpoint
    # refiner cut scores. This is boundary packing, not recall tuning.
    pre_asr_cut_split_enabled: bool = False
    pre_asr_cut_split_min_core_frames: int = 420
    pre_asr_cut_split_target_core_frames: int = 270
    pre_asr_cut_split_min_cut_frames: int = 3
    pre_asr_cut_split_min_child_frames: int = 45
    pre_asr_cut_split_max_children: int = 8
    pre_asr_cut_split_threshold: float = 0.94

    @property
    def chunk_cap_s(self) -> float:
        return max(1, self.window_frames - self.reserve_frames) * self.frame_hop_s

    @property
    def max_core_s(self) -> float:
        return max(0, self.max_core_frames) * self.frame_hop_s

    @property
    def target_padding_s(self) -> float:
        return max(0, self.target_padding_frames) * self.frame_hop_s

    @property
    def gap_merge_s(self) -> float:
        return max(0, self.gap_merge_frames) * self.frame_hop_s

    @property
    def pre_asr_island_split_min_core_s(self) -> float:
        return max(0, self.pre_asr_island_split_min_core_frames) * self.frame_hop_s

    @property
    def pre_asr_island_split_min_gap_s(self) -> float:
        return max(0, self.pre_asr_island_split_min_gap_frames) * self.frame_hop_s

    @property
    def pre_asr_island_split_min_island_s(self) -> float:
        return max(0, self.pre_asr_island_split_min_island_frames) * self.frame_hop_s

    @property
    def pre_asr_valley_split_min_core_s(self) -> float:
        return max(0, self.pre_asr_valley_split_min_core_frames) * self.frame_hop_s

    @property
    def pre_asr_cut_split_min_core_s(self) -> float:
        return max(0, self.pre_asr_cut_split_min_core_frames) * self.frame_hop_s

    @property
    def effective_score_frame_hop_s(self) -> float:
        return self.score_frame_hop_s if self.score_frame_hop_s is not None else self.frame_hop_s


@dataclass(frozen=True)
class _PackSegment:
    start: float
    end: float
    score: float | None = None
    split_left: bool = False
    split_right: bool = False
    force_break_before: bool = False
    split_policy: str = ""
    valley_score_min: float | None = None
    cut_score_max: float | None = None

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
    max_core_frames: int = 0,
    pre_asr_island_split_enabled: bool = False,
    pre_asr_island_split_min_core_frames: int = 420,
    pre_asr_island_split_min_gap_frames: int = 18,
    pre_asr_island_split_min_island_frames: int = 3,
    pre_asr_island_split_max_children: int = 8,
    pre_asr_valley_split_enabled: bool = False,
    pre_asr_valley_split_min_core_frames: int = 420,
    pre_asr_valley_split_target_core_frames: int = 270,
    pre_asr_valley_split_min_valley_frames: int = 6,
    pre_asr_valley_split_min_child_frames: int = 45,
    pre_asr_valley_split_max_children: int = 8,
    pre_asr_valley_split_threshold: float = 0.20,
    frame_scores: Sequence[float] | None = None,
    score_frame_hop_s: float | None = None,
    pre_asr_cut_split_enabled: bool = False,
    pre_asr_cut_split_min_core_frames: int = 420,
    pre_asr_cut_split_target_core_frames: int = 270,
    pre_asr_cut_split_min_cut_frames: int = 3,
    pre_asr_cut_split_min_child_frames: int = 45,
    pre_asr_cut_split_max_children: int = 8,
    pre_asr_cut_split_threshold: float = 0.94,
    cut_frame_scores: Sequence[float] | None = None,
) -> list[PackedChunk]:
    """Pack VAD speech into frame-derived ASR chunks with dynamic gap-aware padding."""
    config = FramePackingConfig(
        frame_hop_s=frame_hop_s,
        score_frame_hop_s=score_frame_hop_s,
        window_frames=window_frames,
        reserve_frames=reserve_frames,
        target_padding_frames=target_padding_frames,
        gap_merge_frames=gap_merge_frames,
        max_core_frames=max_core_frames,
        pre_asr_island_split_enabled=pre_asr_island_split_enabled,
        pre_asr_island_split_min_core_frames=pre_asr_island_split_min_core_frames,
        pre_asr_island_split_min_gap_frames=pre_asr_island_split_min_gap_frames,
        pre_asr_island_split_min_island_frames=pre_asr_island_split_min_island_frames,
        pre_asr_island_split_max_children=pre_asr_island_split_max_children,
        pre_asr_valley_split_enabled=pre_asr_valley_split_enabled,
        pre_asr_valley_split_min_core_frames=pre_asr_valley_split_min_core_frames,
        pre_asr_valley_split_target_core_frames=pre_asr_valley_split_target_core_frames,
        pre_asr_valley_split_min_valley_frames=pre_asr_valley_split_min_valley_frames,
        pre_asr_valley_split_min_child_frames=pre_asr_valley_split_min_child_frames,
        pre_asr_valley_split_max_children=pre_asr_valley_split_max_children,
        pre_asr_valley_split_threshold=pre_asr_valley_split_threshold,
        pre_asr_cut_split_enabled=pre_asr_cut_split_enabled,
        pre_asr_cut_split_min_core_frames=pre_asr_cut_split_min_core_frames,
        pre_asr_cut_split_target_core_frames=pre_asr_cut_split_target_core_frames,
        pre_asr_cut_split_min_cut_frames=pre_asr_cut_split_min_cut_frames,
        pre_asr_cut_split_min_child_frames=pre_asr_cut_split_min_child_frames,
        pre_asr_cut_split_max_children=pre_asr_cut_split_max_children,
        pre_asr_cut_split_threshold=pre_asr_cut_split_threshold,
    )
    _validate_config(config)

    ordered_speech_segments = sorted(segments, key=lambda segment: (segment.start, segment.end))
    _validate_segments(ordered_speech_segments)
    ordered_segments = _segments_to_pack_segments(ordered_speech_segments)
    ordered_segments = _split_segments_on_cut_scores(
        ordered_segments,
        config=config,
        cut_frame_scores=cut_frame_scores,
    )
    ordered_segments = _split_segments_on_score_valleys(
        ordered_segments,
        config=config,
        frame_scores=frame_scores,
    )
    ordered_segments = _split_overlong_segments(ordered_segments, config=config)

    chunks: list[PackedChunk] = []
    current: list[_PackSegment] = []

    for index, segment in enumerate(ordered_segments):
        if not current:
            current = [segment]
            continue

        if segment.force_break_before:
            chunks.append(
                _make_chunk(
                    current,
                    config=config,
                    previous_end=chunks[-1].vad_segments[-1].end if chunks else None,
                    next_start=segment.start,
                    split_reason="pre_asr_valley_split",
                )
            )
            current = [segment]
            continue

        gap_s = segment.start - current[-1].end
        proposed = [*current, segment]
        proposed_core = _core_duration(proposed)
        within_hard_cap = proposed_core <= config.chunk_cap_s
        within_soft_cap = config.max_core_s <= 0.0 or proposed_core <= config.max_core_s
        if gap_s <= config.gap_merge_s and within_hard_cap and within_soft_cap:
            current = proposed
            continue

        if gap_s > config.gap_merge_s:
            split_reason = "gap"
        elif not within_hard_cap:
            split_reason = "capacity"
        else:
            # Soft cap tripped at a mergeable gap: split here so multiple speech
            # islands do not accumulate into one aligner-hostile super-chunk.
            split_reason = "soft_cap"

        next_start = segment.start
        chunks.append(
            _make_chunk(
                current,
                config=config,
                previous_end=chunks[-1].vad_segments[-1].end if chunks else None,
                next_start=next_start,
                split_reason=split_reason,
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

    return _split_pre_asr_islands(chunks, config=config)


def _validate_config(config: FramePackingConfig) -> None:
    if config.frame_hop_s <= 0:
        raise ValueError("frame_hop_s must be positive")
    if config.score_frame_hop_s is not None and config.score_frame_hop_s <= 0:
        raise ValueError("score_frame_hop_s must be positive")
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
    if config.pre_asr_island_split_min_core_frames < 0:
        raise ValueError("pre_asr_island_split_min_core_frames must be non-negative")
    if config.pre_asr_island_split_min_gap_frames < 0:
        raise ValueError("pre_asr_island_split_min_gap_frames must be non-negative")
    if config.pre_asr_island_split_min_island_frames < 0:
        raise ValueError("pre_asr_island_split_min_island_frames must be non-negative")
    if config.pre_asr_island_split_max_children <= 0:
        raise ValueError("pre_asr_island_split_max_children must be positive")
    if config.pre_asr_valley_split_min_core_frames < 0:
        raise ValueError("pre_asr_valley_split_min_core_frames must be non-negative")
    if config.pre_asr_valley_split_target_core_frames < 0:
        raise ValueError("pre_asr_valley_split_target_core_frames must be non-negative")
    if config.pre_asr_valley_split_min_valley_frames < 0:
        raise ValueError("pre_asr_valley_split_min_valley_frames must be non-negative")
    if config.pre_asr_valley_split_min_child_frames < 0:
        raise ValueError("pre_asr_valley_split_min_child_frames must be non-negative")
    if config.pre_asr_valley_split_max_children <= 0:
        raise ValueError("pre_asr_valley_split_max_children must be positive")
    if config.pre_asr_valley_split_threshold < 0.0:
        raise ValueError("pre_asr_valley_split_threshold must be non-negative")
    if config.pre_asr_cut_split_min_core_frames < 0:
        raise ValueError("pre_asr_cut_split_min_core_frames must be non-negative")
    if config.pre_asr_cut_split_target_core_frames < 0:
        raise ValueError("pre_asr_cut_split_target_core_frames must be non-negative")
    if config.pre_asr_cut_split_min_cut_frames < 0:
        raise ValueError("pre_asr_cut_split_min_cut_frames must be non-negative")
    if config.pre_asr_cut_split_min_child_frames < 0:
        raise ValueError("pre_asr_cut_split_min_child_frames must be non-negative")
    if config.pre_asr_cut_split_max_children <= 0:
        raise ValueError("pre_asr_cut_split_max_children must be positive")
    if config.pre_asr_cut_split_threshold < 0.0:
        raise ValueError("pre_asr_cut_split_threshold must be non-negative")


def _validate_segments(segments: Sequence[SpeechSegment]) -> None:
    for segment in segments:
        if segment.end < segment.start:
            raise ValueError("segment end must be greater than or equal to start")


def _segments_to_pack_segments(segments: Sequence[SpeechSegment]) -> list[_PackSegment]:
    return [
        _PackSegment(start=segment.start, end=segment.end, score=segment.score)
        for segment in segments
    ]


def _split_overlong_segments(
    segments: Sequence[_PackSegment],
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
            split.append(segment)
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
                    force_break_before=segment.force_break_before if first else False,
                    split_policy=segment.split_policy,
                    valley_score_min=segment.valley_score_min,
                    cut_score_max=segment.cut_score_max,
                )
            )
            cursor = next_end
            first = False
    return split


def _split_segments_on_cut_scores(
    segments: Sequence[_PackSegment],
    *,
    config: FramePackingConfig,
    cut_frame_scores: Sequence[float] | None,
) -> list[_PackSegment]:
    if not config.pre_asr_cut_split_enabled or cut_frame_scores is None:
        return list(segments)
    scores = (
        cut_frame_scores
        if isinstance(cut_frame_scores, list)
        else [float(value) for value in cut_frame_scores]
    )
    if not scores:
        return list(segments)

    result: list[_PackSegment] = []
    for segment in segments:
        split_frames = _plan_cut_split_frames_for_segment(
            segment,
            scores=scores,
            config=config,
        )
        if not split_frames:
            result.append(segment)
            continue
        score_hop = config.effective_score_frame_hop_s
        boundaries = [segment.start, *[frame * score_hop for frame in split_frames], segment.end]
        for child_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if end <= start:
                continue
            start_frame = max(0, int(start / score_hop))
            end_frame = min(len(scores), max(start_frame + 1, int(end / score_hop)))
            cut_max = max(scores[start_frame:end_frame]) if end_frame > start_frame else None
            result.append(
                _PackSegment(
                    start=start,
                    end=end,
                    score=segment.score,
                    split_left=child_index > 0,
                    split_right=child_index < len(boundaries) - 2,
                    force_break_before=child_index > 0,
                    split_policy="r17_pre_asr_cut_v1",
                    cut_score_max=cut_max,
                )
            )
    return result


def _plan_cut_split_frames_for_segment(
    segment: _PackSegment,
    *,
    scores: Sequence[float],
    config: FramePackingConfig,
) -> list[int]:
    if config.pre_asr_cut_split_min_core_s > 0.0:
        if segment.end - segment.start < config.pre_asr_cut_split_min_core_s:
            return []
    score_hop = config.effective_score_frame_hop_s
    start_frame = max(0, int(round(segment.start / score_hop)))
    end_frame = min(len(scores), int(round(segment.end / score_hop)))
    min_core_score_frames = max(1, int(round(config.pre_asr_cut_split_min_core_s / score_hop)))
    if end_frame - start_frame < min_core_score_frames:
        return []

    min_child_score_frames = max(
        1,
        int(round(max(0, config.pre_asr_cut_split_min_child_frames) * config.frame_hop_s / score_hop)),
    )
    target_score_frames = max(
        1,
        int(round(max(0, config.pre_asr_cut_split_target_core_frames) * config.frame_hop_s / score_hop)),
    )
    min_cut_score_frames = max(
        1,
        int(round(max(0, config.pre_asr_cut_split_min_cut_frames) * config.frame_hop_s / score_hop)),
    )
    target_frames = max(
        min_child_score_frames * 2,
        target_score_frames,
    )
    if target_frames <= 0:
        target_frames = min_core_score_frames

    groups: list[tuple[int, int]] = [(start_frame, end_frame)]
    split_frames: list[int] = []
    while len(groups) < config.pre_asr_cut_split_max_children:
        candidates = [
            (index, group)
            for index, group in enumerate(groups)
            if group[1] - group[0] > target_frames
        ]
        if not candidates:
            break
        group_index, (group_start, group_end) = max(
            candidates,
            key=lambda item: item[1][1] - item[1][0],
        )
        split_frame = _pick_cut_split_frame(
            scores,
            start_frame=group_start,
            end_frame=group_end,
            config=config,
            min_child_frames=min_child_score_frames,
            min_cut_frames=min_cut_score_frames,
            target_core_frames=target_score_frames,
        )
        if split_frame is None:
            break
        left = (group_start, split_frame)
        right = (split_frame, group_end)
        groups[group_index : group_index + 1] = [left, right]
        split_frames.append(split_frame)

    return sorted(set(split_frames))


def _pick_cut_split_frame(
    scores: Sequence[float],
    *,
    start_frame: int,
    end_frame: int,
    config: FramePackingConfig,
    min_child_frames: int | None = None,
    min_cut_frames: int | None = None,
    target_core_frames: int | None = None,
) -> int | None:
    min_child = max(1, min_child_frames or config.pre_asr_cut_split_min_child_frames)
    min_cut = max(1, min_cut_frames or config.pre_asr_cut_split_min_cut_frames)
    lower = start_frame + min_child
    upper = end_frame - min_child
    if upper <= lower:
        return None

    runs: list[tuple[int, int, float]] = []
    run_start: int | None = None
    threshold = float(config.pre_asr_cut_split_threshold)
    for frame in range(lower, upper):
        if float(scores[frame]) >= threshold:
            if run_start is None:
                run_start = frame
            continue
        if run_start is not None and frame - run_start >= min_cut:
            run = scores[run_start:frame]
            runs.append((run_start, frame, max(float(value) for value in run)))
        run_start = None
    if run_start is not None and upper - run_start >= min_cut:
        run = scores[run_start:upper]
        runs.append((run_start, upper, max(float(value) for value in run)))
    if not runs:
        return None

    target = target_core_frames if target_core_frames is not None else config.pre_asr_cut_split_target_core_frames
    ideal = min(end_frame - min_child, start_frame + max(min_child, target))
    run_start, run_end, _max_score = min(
        runs,
        key=lambda item: (abs(((item[0] + item[1]) / 2.0) - ideal), -item[2]),
    )
    return int(round((run_start + run_end) / 2.0))


def _split_segments_on_score_valleys(
    segments: Sequence[_PackSegment],
    *,
    config: FramePackingConfig,
    frame_scores: Sequence[float] | None,
) -> list[_PackSegment]:
    if not config.pre_asr_valley_split_enabled or frame_scores is None:
        return list(segments)
    scores = frame_scores if isinstance(frame_scores, list) else [float(value) for value in frame_scores]
    if not scores:
        return list(segments)

    result: list[_PackSegment] = []
    for segment in segments:
        split_frames = _plan_valley_split_frames_for_segment(
            segment,
            scores=scores,
            config=config,
        )
        if not split_frames:
            result.append(segment)
            continue
        score_hop = config.effective_score_frame_hop_s
        boundaries = [segment.start, *[frame * score_hop for frame in split_frames], segment.end]
        for child_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if end <= start:
                continue
            start_frame = max(0, int(start / score_hop))
            end_frame = min(len(scores), max(start_frame + 1, int(end / score_hop)))
            valley_min = min(scores[start_frame:end_frame]) if end_frame > start_frame else None
            result.append(
                _PackSegment(
                    start=start,
                    end=end,
                    score=segment.score,
                    split_left=child_index > 0,
                    split_right=child_index < len(boundaries) - 2,
                    force_break_before=child_index > 0,
                    split_policy="r16_pre_asr_valley_v1",
                    valley_score_min=valley_min,
                )
            )
    return result


def _plan_valley_split_frames_for_segment(
    segment: _PackSegment,
    *,
    scores: Sequence[float],
    config: FramePackingConfig,
) -> list[int]:
    if config.pre_asr_valley_split_min_core_s > 0.0:
        if segment.end - segment.start < config.pre_asr_valley_split_min_core_s:
            return []
    score_hop = config.effective_score_frame_hop_s
    start_frame = max(0, int(round(segment.start / score_hop)))
    end_frame = min(len(scores), int(round(segment.end / score_hop)))
    min_core_score_frames = max(1, int(round(config.pre_asr_valley_split_min_core_s / score_hop)))
    if end_frame - start_frame < min_core_score_frames:
        return []

    min_child_score_frames = max(
        1,
        int(round(max(0, config.pre_asr_valley_split_min_child_frames) * config.frame_hop_s / score_hop)),
    )
    target_score_frames = max(
        1,
        int(round(max(0, config.pre_asr_valley_split_target_core_frames) * config.frame_hop_s / score_hop)),
    )
    min_valley_score_frames = max(
        1,
        int(round(max(0, config.pre_asr_valley_split_min_valley_frames) * config.frame_hop_s / score_hop)),
    )
    target_frames = max(
        min_child_score_frames * 2,
        target_score_frames,
    )
    if target_frames <= 0:
        target_frames = min_core_score_frames

    groups: list[tuple[int, int]] = [(start_frame, end_frame)]
    split_frames: list[int] = []
    while len(groups) < config.pre_asr_valley_split_max_children:
        candidates = [
            (index, group)
            for index, group in enumerate(groups)
            if group[1] - group[0] > target_frames
        ]
        if not candidates:
            break
        group_index, (group_start, group_end) = max(
            candidates,
            key=lambda item: item[1][1] - item[1][0],
        )
        split_frame = _pick_valley_split_frame(
            scores,
            start_frame=group_start,
            end_frame=group_end,
            config=config,
            min_child_frames=min_child_score_frames,
            min_valley_frames=min_valley_score_frames,
            target_core_frames=target_score_frames,
        )
        if split_frame is None:
            break
        left = (group_start, split_frame)
        right = (split_frame, group_end)
        groups[group_index : group_index + 1] = [left, right]
        split_frames.append(split_frame)

    return sorted(set(split_frames))


def _pick_valley_split_frame(
    scores: Sequence[float],
    *,
    start_frame: int,
    end_frame: int,
    config: FramePackingConfig,
    min_child_frames: int | None = None,
    min_valley_frames: int | None = None,
    target_core_frames: int | None = None,
) -> int | None:
    min_child = max(1, min_child_frames or config.pre_asr_valley_split_min_child_frames)
    min_valley = max(1, min_valley_frames or config.pre_asr_valley_split_min_valley_frames)
    lower = start_frame + min_child
    upper = end_frame - min_child
    if upper <= lower:
        return None

    runs: list[tuple[int, int, float]] = []
    run_start: int | None = None
    threshold = float(config.pre_asr_valley_split_threshold)
    for frame in range(lower, upper):
        if float(scores[frame]) <= threshold:
            if run_start is None:
                run_start = frame
            continue
        if run_start is not None and frame - run_start >= min_valley:
            run = scores[run_start:frame]
            runs.append((run_start, frame, sum(float(value) for value in run) / len(run)))
        run_start = None
    if run_start is not None and upper - run_start >= min_valley:
        run = scores[run_start:upper]
        runs.append((run_start, upper, sum(float(value) for value in run) / len(run)))
    if not runs:
        return None

    target = target_core_frames if target_core_frames is not None else config.pre_asr_valley_split_target_core_frames
    ideal = min(end_frame - min_child, start_frame + max(min_child, target))
    run_start, run_end, _mean_score = min(
        runs,
        key=lambda item: (abs(((item[0] + item[1]) / 2.0) - ideal), item[2]),
    )
    return int(round((run_start + run_end) / 2.0))


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
        if any(segment.split_policy == "r17_pre_asr_cut_v1" for segment in segments):
            split_reason = "pre_asr_cut_split"
        elif any(segment.split_policy == "r16_pre_asr_valley_v1" for segment in segments):
            split_reason = "pre_asr_valley_split"
        else:
            split_reason = "overlong"
    split_policies = sorted({segment.split_policy for segment in segments if segment.split_policy})
    valley_scores = [
        float(segment.valley_score_min)
        for segment in segments
        if segment.valley_score_min is not None
    ]
    cut_scores = [
        float(segment.cut_score_max)
        for segment in segments
        if segment.cut_score_max is not None
    ]
    return PackedChunk(
        start=start,
        end=end,
        vad_segments=[segment.to_speech_segment() for segment in segments],
        duration=end - start,
        left_padding_s=left_padding,
        right_padding_s=right_padding,
        split_reason=split_reason,
        core_start=core_start,
        core_end=core_end,
        internal_gap_count=_internal_gap_count(segments),
        internal_gap_max_s=_internal_gap_max_s(segments),
        split_policy=",".join(split_policies),
        valley_split_count=sum(
            1 for segment in segments if segment.split_policy == "r16_pre_asr_valley_v1"
        ),
        valley_score_min=min(valley_scores) if valley_scores else None,
        cut_split_count=sum(
            1 for segment in segments if segment.split_policy == "r17_pre_asr_cut_v1"
        ),
        cut_score_max=max(cut_scores) if cut_scores else None,
    )


def _split_pre_asr_islands(
    chunks: Sequence[PackedChunk],
    *,
    config: FramePackingConfig,
) -> list[PackedChunk]:
    if not config.pre_asr_island_split_enabled:
        return list(chunks)

    result: list[PackedChunk] = []
    for parent_index, chunk in enumerate(chunks):
        children = _split_pre_asr_island_chunk(
            chunk,
            parent_index=parent_index,
            config=config,
        )
        result.extend(children)
    return result


def _split_pre_asr_island_chunk(
    chunk: PackedChunk,
    *,
    parent_index: int,
    config: FramePackingConfig,
) -> list[PackedChunk]:
    segments = [
        _PackSegment(start=segment.start, end=segment.end, score=segment.score)
        for segment in chunk.vad_segments
    ]
    if len(segments) < 2:
        return [chunk]

    core_duration = _core_duration(segments)
    if (
        config.pre_asr_island_split_min_core_s > 0.0
        and core_duration < config.pre_asr_island_split_min_core_s
    ):
        return [chunk]
    if _internal_gap_max_s(segments) < config.pre_asr_island_split_min_gap_s:
        return [chunk]

    groups = _pre_asr_island_groups(segments, config=config)
    if len(groups) < 2:
        return [chunk]

    children: list[PackedChunk] = []
    for child_index, group in enumerate(groups):
        previous_end = groups[child_index - 1][-1].end if child_index > 0 else None
        next_start = groups[child_index + 1][0].start if child_index + 1 < len(groups) else None
        child = _make_chunk(
            group,
            config=config,
            previous_end=previous_end,
            next_start=next_start,
            split_reason="pre_asr_island_split",
        )
        child = _clamp_child_to_parent(
            child,
            parent=chunk,
            parent_index=parent_index,
            child_index=child_index,
            child_count=len(groups),
        )
        children.append(child)
    return children


def _pre_asr_island_groups(
    segments: Sequence[_PackSegment],
    *,
    config: FramePackingConfig,
) -> list[list[_PackSegment]]:
    groups: list[list[_PackSegment]] = []
    current: list[_PackSegment] = [segments[0]]
    max_children = max(1, config.pre_asr_island_split_max_children)
    for segment in segments[1:]:
        gap_s = max(0.0, segment.start - current[-1].end)
        can_split = gap_s >= config.pre_asr_island_split_min_gap_s and len(groups) + 1 < max_children
        if can_split:
            groups.append(current)
            current = [segment]
        else:
            current.append(segment)
    groups.append(current)
    return _merge_tiny_pre_asr_island_groups(groups, config=config)


def _merge_tiny_pre_asr_island_groups(
    groups: Sequence[Sequence[_PackSegment]],
    *,
    config: FramePackingConfig,
) -> list[list[_PackSegment]]:
    merged: list[list[_PackSegment]] = []
    min_s = config.pre_asr_island_split_min_island_s
    for group in groups:
        current = list(group)
        duration_s = _core_duration(current)
        if merged and duration_s < min_s:
            merged[-1].extend(current)
        else:
            merged.append(current)

    if len(merged) > 1 and _core_duration(merged[0]) < min_s:
        first = merged.pop(0)
        merged[0] = [*first, *merged[0]]
    return merged


def _clamp_child_to_parent(
    child: PackedChunk,
    *,
    parent: PackedChunk,
    parent_index: int,
    child_index: int,
    child_count: int,
) -> PackedChunk:
    core_start, core_end = _chunk_core_bounds(child)
    start = max(parent.start, child.start)
    end = min(parent.end, child.end)
    start = min(start, core_start)
    end = max(end, core_end)
    return PackedChunk(
        start=start,
        end=end,
        vad_segments=child.vad_segments,
        duration=max(0.0, end - start),
        left_padding_s=max(0.0, core_start - start),
        right_padding_s=max(0.0, end - core_end),
        split_reason=child.split_reason,
        parent_chunk_id=parent_index,
        island_id=child_index,
        island_count=child_count,
        core_start=core_start,
        core_end=core_end,
        internal_gap_count=child.internal_gap_count,
        internal_gap_max_s=child.internal_gap_max_s,
        split_policy="r15_pre_asr_island_v1",
        valley_split_count=child.valley_split_count,
        valley_score_min=child.valley_score_min,
        cut_split_count=child.cut_split_count,
        cut_score_max=child.cut_score_max,
    )


def _chunk_core_bounds(chunk: PackedChunk) -> tuple[float, float]:
    if chunk.vad_segments:
        return chunk.vad_segments[0].start, chunk.vad_segments[-1].end
    start = chunk.core_start if chunk.core_start is not None else chunk.start
    end = chunk.core_end if chunk.core_end is not None else chunk.end
    return start, end


def _internal_gap_count(segments: Sequence[_PackSegment]) -> int:
    return sum(
        1
        for previous, current in zip(segments, segments[1:])
        if current.start > previous.end
    )


def _internal_gap_max_s(segments: Sequence[_PackSegment]) -> float:
    if len(segments) < 2:
        return 0.0
    return max(
        max(0.0, current.start - previous.end)
        for previous, current in zip(segments, segments[1:])
    )
