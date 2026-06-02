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
    drop_gap_split_count: int = 0
    drop_gap_score_max: float | None = None
    risk_split_count: int = 0
    risk_score: float | None = None
    risk_reasons: tuple[str, ...] = ()


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
    # R21 opt-in: use a drop-gap-only imitation head to delete high-confidence
    # non-speech runs inside a long VAD-positive parent segment, then force a
    # pre-ASR boundary so gap_merge_frames cannot glue the children back.
    pre_asr_drop_gap_split_enabled: bool = False
    pre_asr_drop_gap_split_min_parent_frames: int = 400
    pre_asr_drop_gap_split_min_gap_frames: int = 3
    pre_asr_drop_gap_split_min_gap_s: float = 0.60
    pre_asr_drop_gap_split_min_child_frames: int = 25
    pre_asr_drop_gap_split_max_children: int = 8
    pre_asr_drop_gap_split_threshold: float = 0.80
    # R18 opt-in: split high fallback-risk packed chunks with a global objective
    # instead of relying on one local cut/valley signal. The packer prefers
    # explicit internal gaps, then falls back to cut/valley frame-score evidence.
    pre_asr_risk_split_enabled: bool = False
    pre_asr_risk_split_min_core_frames: int = 420
    pre_asr_risk_split_target_core_frames: int = 270
    pre_asr_risk_split_safe_core_frames: int = 360
    pre_asr_risk_split_min_gap_frames: int = 6
    pre_asr_risk_split_min_child_frames: int = 45
    pre_asr_risk_split_max_children: int = 8
    pre_asr_risk_split_threshold: float = 1.0
    # Long continuous islands are riskier to cut than chunks with explicit
    # internal gaps. Require a higher risk score before using cut/valley scores.
    pre_asr_risk_split_continuous_threshold: float = 2.0
    pre_asr_risk_split_valley_threshold: float = 0.20
    pre_asr_risk_split_cut_threshold: float = 0.94

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
    def pre_asr_drop_gap_split_min_parent_s(self) -> float:
        return max(0, self.pre_asr_drop_gap_split_min_parent_frames) * self.frame_hop_s

    @property
    def pre_asr_drop_gap_split_min_gap_s_effective(self) -> float:
        return max(
            0.0,
            max(0, self.pre_asr_drop_gap_split_min_gap_frames) * self.frame_hop_s,
            self.pre_asr_drop_gap_split_min_gap_s,
        )

    @property
    def pre_asr_drop_gap_split_min_child_s(self) -> float:
        return max(0, self.pre_asr_drop_gap_split_min_child_frames) * self.frame_hop_s

    @property
    def pre_asr_risk_split_min_core_s(self) -> float:
        return max(0, self.pre_asr_risk_split_min_core_frames) * self.frame_hop_s

    @property
    def pre_asr_risk_split_target_core_s(self) -> float:
        return max(0, self.pre_asr_risk_split_target_core_frames) * self.frame_hop_s

    @property
    def pre_asr_risk_split_safe_core_s(self) -> float:
        return max(0, self.pre_asr_risk_split_safe_core_frames) * self.frame_hop_s

    @property
    def pre_asr_risk_split_min_gap_s(self) -> float:
        return max(0, self.pre_asr_risk_split_min_gap_frames) * self.frame_hop_s

    @property
    def pre_asr_risk_split_min_child_s(self) -> float:
        return max(0, self.pre_asr_risk_split_min_child_frames) * self.frame_hop_s

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
    drop_gap_score_max: float | None = None

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
    pre_asr_drop_gap_split_enabled: bool = False,
    pre_asr_drop_gap_split_min_parent_frames: int = 400,
    pre_asr_drop_gap_split_min_gap_frames: int = 3,
    pre_asr_drop_gap_split_min_gap_s: float = 0.60,
    pre_asr_drop_gap_split_min_child_frames: int = 25,
    pre_asr_drop_gap_split_max_children: int = 8,
    pre_asr_drop_gap_split_threshold: float = 0.80,
    drop_gap_frame_scores: Sequence[float] | None = None,
    pre_asr_risk_split_enabled: bool = False,
    pre_asr_risk_split_min_core_frames: int = 420,
    pre_asr_risk_split_target_core_frames: int = 270,
    pre_asr_risk_split_safe_core_frames: int = 360,
    pre_asr_risk_split_min_gap_frames: int = 6,
    pre_asr_risk_split_min_child_frames: int = 45,
    pre_asr_risk_split_max_children: int = 8,
    pre_asr_risk_split_threshold: float = 1.0,
    pre_asr_risk_split_continuous_threshold: float = 2.0,
    pre_asr_risk_split_valley_threshold: float = 0.20,
    pre_asr_risk_split_cut_threshold: float = 0.94,
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
        pre_asr_drop_gap_split_enabled=pre_asr_drop_gap_split_enabled,
        pre_asr_drop_gap_split_min_parent_frames=pre_asr_drop_gap_split_min_parent_frames,
        pre_asr_drop_gap_split_min_gap_frames=pre_asr_drop_gap_split_min_gap_frames,
        pre_asr_drop_gap_split_min_gap_s=pre_asr_drop_gap_split_min_gap_s,
        pre_asr_drop_gap_split_min_child_frames=pre_asr_drop_gap_split_min_child_frames,
        pre_asr_drop_gap_split_max_children=pre_asr_drop_gap_split_max_children,
        pre_asr_drop_gap_split_threshold=pre_asr_drop_gap_split_threshold,
        pre_asr_risk_split_enabled=pre_asr_risk_split_enabled,
        pre_asr_risk_split_min_core_frames=pre_asr_risk_split_min_core_frames,
        pre_asr_risk_split_target_core_frames=pre_asr_risk_split_target_core_frames,
        pre_asr_risk_split_safe_core_frames=pre_asr_risk_split_safe_core_frames,
        pre_asr_risk_split_min_gap_frames=pre_asr_risk_split_min_gap_frames,
        pre_asr_risk_split_min_child_frames=pre_asr_risk_split_min_child_frames,
        pre_asr_risk_split_max_children=pre_asr_risk_split_max_children,
        pre_asr_risk_split_threshold=pre_asr_risk_split_threshold,
        pre_asr_risk_split_continuous_threshold=pre_asr_risk_split_continuous_threshold,
        pre_asr_risk_split_valley_threshold=pre_asr_risk_split_valley_threshold,
        pre_asr_risk_split_cut_threshold=pre_asr_risk_split_cut_threshold,
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
    ordered_segments = _split_segments_on_drop_gap_scores(
        ordered_segments,
        config=config,
        drop_gap_frame_scores=drop_gap_frame_scores,
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

    chunks = _split_pre_asr_islands(chunks, config=config)
    return _split_pre_asr_risk_chunks(
        chunks,
        config=config,
        frame_scores=frame_scores,
        cut_frame_scores=cut_frame_scores,
    )


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
    if config.pre_asr_drop_gap_split_min_parent_frames < 0:
        raise ValueError("pre_asr_drop_gap_split_min_parent_frames must be non-negative")
    if config.pre_asr_drop_gap_split_min_gap_frames < 0:
        raise ValueError("pre_asr_drop_gap_split_min_gap_frames must be non-negative")
    if config.pre_asr_drop_gap_split_min_gap_s < 0.0:
        raise ValueError("pre_asr_drop_gap_split_min_gap_s must be non-negative")
    if config.pre_asr_drop_gap_split_min_child_frames < 0:
        raise ValueError("pre_asr_drop_gap_split_min_child_frames must be non-negative")
    if config.pre_asr_drop_gap_split_max_children <= 0:
        raise ValueError("pre_asr_drop_gap_split_max_children must be positive")
    if config.pre_asr_drop_gap_split_threshold < 0.0:
        raise ValueError("pre_asr_drop_gap_split_threshold must be non-negative")
    if config.pre_asr_risk_split_min_core_frames < 0:
        raise ValueError("pre_asr_risk_split_min_core_frames must be non-negative")
    if config.pre_asr_risk_split_target_core_frames < 0:
        raise ValueError("pre_asr_risk_split_target_core_frames must be non-negative")
    if config.pre_asr_risk_split_safe_core_frames < 0:
        raise ValueError("pre_asr_risk_split_safe_core_frames must be non-negative")
    if config.pre_asr_risk_split_min_gap_frames < 0:
        raise ValueError("pre_asr_risk_split_min_gap_frames must be non-negative")
    if config.pre_asr_risk_split_min_child_frames < 0:
        raise ValueError("pre_asr_risk_split_min_child_frames must be non-negative")
    if config.pre_asr_risk_split_max_children <= 0:
        raise ValueError("pre_asr_risk_split_max_children must be positive")
    if config.pre_asr_risk_split_threshold < 0.0:
        raise ValueError("pre_asr_risk_split_threshold must be non-negative")
    if config.pre_asr_risk_split_continuous_threshold < 0.0:
        raise ValueError("pre_asr_risk_split_continuous_threshold must be non-negative")
    if config.pre_asr_risk_split_valley_threshold < 0.0:
        raise ValueError("pre_asr_risk_split_valley_threshold must be non-negative")
    if config.pre_asr_risk_split_cut_threshold < 0.0:
        raise ValueError("pre_asr_risk_split_cut_threshold must be non-negative")


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
                    drop_gap_score_max=segment.drop_gap_score_max,
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
                    drop_gap_score_max=segment.drop_gap_score_max,
                )
            )
    return result


def _split_segments_on_drop_gap_scores(
    segments: Sequence[_PackSegment],
    *,
    config: FramePackingConfig,
    drop_gap_frame_scores: Sequence[float] | None,
) -> list[_PackSegment]:
    if not config.pre_asr_drop_gap_split_enabled or drop_gap_frame_scores is None:
        return list(segments)
    scores = (
        drop_gap_frame_scores
        if isinstance(drop_gap_frame_scores, list)
        else [float(value) for value in drop_gap_frame_scores]
    )
    if not scores:
        return list(segments)

    result: list[_PackSegment] = []
    for segment in segments:
        children = _split_segment_on_drop_gap_scores(segment, scores=scores, config=config)
        result.extend(children)
    return result


def _split_segment_on_drop_gap_scores(
    segment: _PackSegment,
    *,
    scores: Sequence[float],
    config: FramePackingConfig,
) -> list[_PackSegment]:
    if (
        config.pre_asr_drop_gap_split_min_parent_s > 0.0
        and segment.end - segment.start < config.pre_asr_drop_gap_split_min_parent_s
    ):
        return [segment]

    score_hop = config.effective_score_frame_hop_s
    start_frame = max(0, int(round(segment.start / score_hop)))
    end_frame = min(len(scores), int(round(segment.end / score_hop)))
    if end_frame <= start_frame:
        return [segment]

    min_gap_frames = max(
        1,
        int(round(config.pre_asr_drop_gap_split_min_gap_s_effective / score_hop)),
    )
    min_child_frames = max(
        1,
        int(round(config.pre_asr_drop_gap_split_min_child_s / score_hop)),
    )
    if end_frame - start_frame < min_child_frames * 2 + min_gap_frames:
        return [segment]

    runs = _high_score_runs(
        scores,
        start_frame=start_frame + min_child_frames,
        end_frame=end_frame - min_child_frames,
        threshold=config.pre_asr_drop_gap_split_threshold,
        min_frames=min_gap_frames,
    )
    if not runs:
        return [segment]

    max_children = max(1, config.pre_asr_drop_gap_split_max_children)
    parts: list[tuple[int, int, float | None]] = [(start_frame, end_frame, segment.drop_gap_score_max)]
    applied_runs: list[tuple[int, int, float]] = []
    for run_start, run_end, run_score in runs:
        if len(parts) >= max_children:
            break
        next_parts: list[tuple[int, int, float | None]] = []
        changed = False
        for part_start, part_end, part_score in parts:
            if changed:
                next_parts.append((part_start, part_end, part_score))
                continue
            if run_end <= part_start or run_start >= part_end:
                next_parts.append((part_start, part_end, part_score))
                continue
            zone_start = max(part_start, run_start)
            zone_end = min(part_end, run_end)
            if zone_start - part_start < min_child_frames or part_end - zone_end < min_child_frames:
                next_parts.append((part_start, part_end, part_score))
                continue
            next_parts.append((part_start, zone_start, run_score))
            next_parts.append((zone_end, part_end, run_score))
            applied_runs.append((zone_start, zone_end, run_score))
            changed = True
        parts = sorted(next_parts, key=lambda item: (item[0], item[1]))

    if not applied_runs:
        return [segment]

    split: list[_PackSegment] = []
    for child_index, (part_start, part_end, part_score) in enumerate(parts):
        if part_end <= part_start:
            continue
        split.append(
            _PackSegment(
                start=part_start * score_hop,
                end=part_end * score_hop,
                score=segment.score,
                split_left=child_index > 0 or segment.split_left,
                split_right=child_index < len(parts) - 1 or segment.split_right,
                force_break_before=child_index > 0 or segment.force_break_before,
                split_policy="r21_pre_asr_drop_gap_v1",
                valley_score_min=segment.valley_score_min,
                cut_score_max=segment.cut_score_max,
                drop_gap_score_max=part_score,
            )
        )
    return split or [segment]


def _high_score_runs(
    scores: Sequence[float],
    *,
    start_frame: int,
    end_frame: int,
    threshold: float,
    min_frames: int,
) -> list[tuple[int, int, float]]:
    lower = max(0, start_frame)
    upper = min(len(scores), end_frame)
    if upper <= lower:
        return []
    runs: list[tuple[int, int, float]] = []
    run_start: int | None = None
    for frame in range(lower, upper):
        if float(scores[frame]) >= threshold:
            if run_start is None:
                run_start = frame
            continue
        if run_start is not None and frame - run_start >= min_frames:
            run = scores[run_start:frame]
            runs.append((run_start, frame, max(float(value) for value in run)))
        run_start = None
    if run_start is not None and upper - run_start >= min_frames:
        run = scores[run_start:upper]
        runs.append((run_start, upper, max(float(value) for value in run)))
    return runs


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
                    cut_score_max=segment.cut_score_max,
                    drop_gap_score_max=segment.drop_gap_score_max,
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
        if any(segment.split_policy == "r21_pre_asr_drop_gap_v1" for segment in segments):
            split_reason = "pre_asr_drop_gap_split"
        elif any(segment.split_policy == "r17_pre_asr_cut_v1" for segment in segments):
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
    drop_gap_scores = [
        float(segment.drop_gap_score_max)
        for segment in segments
        if segment.drop_gap_score_max is not None
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
        drop_gap_split_count=sum(
            1 for segment in segments if segment.split_policy == "r21_pre_asr_drop_gap_v1"
        ),
        drop_gap_score_max=max(drop_gap_scores) if drop_gap_scores else None,
    )


def _split_pre_asr_risk_chunks(
    chunks: Sequence[PackedChunk],
    *,
    config: FramePackingConfig,
    frame_scores: Sequence[float] | None,
    cut_frame_scores: Sequence[float] | None,
) -> list[PackedChunk]:
    if not config.pre_asr_risk_split_enabled:
        return list(chunks)

    vad_scores = (
        frame_scores
        if isinstance(frame_scores, list) or frame_scores is None
        else [float(value) for value in frame_scores]
    )
    cut_scores = (
        cut_frame_scores
        if isinstance(cut_frame_scores, list) or cut_frame_scores is None
        else [float(value) for value in cut_frame_scores]
    )
    result: list[PackedChunk] = []
    for parent_index, chunk in enumerate(chunks):
        children = _split_pre_asr_risk_chunk(
            chunk,
            parent_index=parent_index,
            config=config,
            frame_scores=vad_scores,
            cut_frame_scores=cut_scores,
        )
        result.extend(children)
    return result


def _split_pre_asr_risk_chunk(
    chunk: PackedChunk,
    *,
    parent_index: int,
    config: FramePackingConfig,
    frame_scores: Sequence[float] | None,
    cut_frame_scores: Sequence[float] | None,
) -> list[PackedChunk]:
    segments = [
        _PackSegment(
            start=segment.start,
            end=segment.end,
            score=segment.score,
            split_policy=chunk.split_policy,
        )
        for segment in chunk.vad_segments
    ]
    if not segments:
        return [chunk]

    risk_score, risk_reasons = _pre_asr_fallback_risk(segments, config=config)
    if risk_score < config.pre_asr_risk_split_threshold:
        return [chunk]

    boundaries = _plan_pre_asr_risk_boundaries(
        segments,
        config=config,
        risk_score=risk_score,
        frame_scores=frame_scores,
        cut_frame_scores=cut_frame_scores,
    )
    if not boundaries:
        return [chunk]

    groups = _groups_from_boundaries(segments, boundaries)
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
            split_reason="pre_asr_risk_split",
        )
        child = _clamp_child_to_parent(
            child,
            parent=chunk,
            parent_index=parent_index,
            child_index=child_index,
            child_count=len(groups),
            split_policy="r18_pre_asr_risk_v1",
        )
        child = _with_risk_metadata(
            child,
            risk_score=risk_score,
            risk_reasons=risk_reasons,
        )
        children.append(child)
    return children


def _pre_asr_fallback_risk(
    segments: Sequence[_PackSegment],
    *,
    config: FramePackingConfig,
) -> tuple[float, tuple[str, ...]]:
    core_s = _core_duration(segments)
    reasons: list[str] = []
    score = 0.0

    if (
        config.pre_asr_risk_split_min_core_s > 0.0
        and core_s >= config.pre_asr_risk_split_min_core_s
    ):
        score += 1.0
        reasons.append("long_core")
    if (
        config.pre_asr_risk_split_safe_core_s > 0.0
        and core_s >= config.pre_asr_risk_split_safe_core_s
    ):
        score += 0.5
        reasons.append("unsafe_duration")
    if (
        len(segments) == 1
        and "r17_pre_asr_cut_v1" in str(segments[0].split_policy or "")
        and config.pre_asr_risk_split_target_core_s > 0.0
        and core_s > config.pre_asr_risk_split_target_core_s
    ):
        score += 0.5
        reasons.append("residual_cut_child")
    if len(segments) >= 2:
        score += 0.75
        reasons.append("multi_island")
    if _internal_gap_max_s(segments) >= config.pre_asr_risk_split_min_gap_s:
        score += 0.75
        reasons.append("internal_gap")

    return score, tuple(reasons)


def _plan_pre_asr_risk_boundaries(
    segments: Sequence[_PackSegment],
    *,
    config: FramePackingConfig,
    risk_score: float,
    frame_scores: Sequence[float] | None,
    cut_frame_scores: Sequence[float] | None,
) -> list[float]:
    groups: list[list[_PackSegment]] = [list(segments)]
    boundaries: list[float] = []
    max_children = max(1, config.pre_asr_risk_split_max_children)

    while len(groups) < max_children:
        candidates = [
            (index, group)
            for index, group in enumerate(groups)
            if _core_duration(group) > config.pre_asr_risk_split_target_core_s
        ]
        if not candidates:
            break
        group_index, group = max(candidates, key=lambda item: _core_duration(item[1]))
        boundary = _pick_pre_asr_risk_boundary(
            group,
            config=config,
            risk_score=risk_score,
            frame_scores=frame_scores,
            cut_frame_scores=cut_frame_scores,
        )
        if boundary is None:
            break
        left, right = _split_group_at_boundary(group, boundary)
        if not left or not right:
            break
        groups[group_index : group_index + 1] = [left, right]
        boundaries.append(boundary)

    return sorted(set(boundaries))


def _pick_pre_asr_risk_boundary(
    segments: Sequence[_PackSegment],
    *,
    config: FramePackingConfig,
    risk_score: float,
    frame_scores: Sequence[float] | None,
    cut_frame_scores: Sequence[float] | None,
) -> float | None:
    if not segments:
        return None
    core_start = segments[0].start
    core_end = segments[-1].end
    min_child_s = max(0.0, config.pre_asr_risk_split_min_child_s)
    lower = core_start + min_child_s
    upper = core_end - min_child_s
    if upper <= lower:
        return None

    target = min(upper, max(lower, core_start + config.pre_asr_risk_split_target_core_s))
    gap_candidates = _internal_gap_boundary_candidates(
        segments,
        config=config,
        lower=lower,
        upper=upper,
    )
    if gap_candidates:
        return min(gap_candidates, key=lambda value: abs(value - target))

    if risk_score < config.pre_asr_risk_split_continuous_threshold:
        return None

    cut_boundary = _score_run_boundary(
        cut_frame_scores,
        score_frame_hop_s=config.effective_score_frame_hop_s,
        lower_s=lower,
        upper_s=upper,
        target_s=target,
        threshold=config.pre_asr_risk_split_cut_threshold,
        mode="high",
        min_frames=max(1, config.pre_asr_cut_split_min_cut_frames),
    )
    if cut_boundary is not None:
        return cut_boundary

    return _score_run_boundary(
        frame_scores,
        score_frame_hop_s=config.effective_score_frame_hop_s,
        lower_s=lower,
        upper_s=upper,
        target_s=target,
        threshold=config.pre_asr_risk_split_valley_threshold,
        mode="low",
        min_frames=max(1, config.pre_asr_valley_split_min_valley_frames),
    )


def _internal_gap_boundary_candidates(
    segments: Sequence[_PackSegment],
    *,
    config: FramePackingConfig,
    lower: float,
    upper: float,
) -> list[float]:
    candidates: list[float] = []
    for previous, current in zip(segments, segments[1:]):
        gap_s = max(0.0, current.start - previous.end)
        if gap_s < config.pre_asr_risk_split_min_gap_s:
            continue
        boundary = (previous.end + current.start) / 2.0
        if lower <= boundary <= upper:
            candidates.append(boundary)
    return candidates


def _score_run_boundary(
    scores: Sequence[float] | None,
    *,
    score_frame_hop_s: float,
    lower_s: float,
    upper_s: float,
    target_s: float,
    threshold: float,
    mode: str,
    min_frames: int,
) -> float | None:
    if not scores:
        return None
    lower = max(0, int(round(lower_s / score_frame_hop_s)))
    upper = min(len(scores), int(round(upper_s / score_frame_hop_s)))
    if upper <= lower:
        return None

    runs: list[tuple[int, int, float]] = []
    run_start: int | None = None
    for frame in range(lower, upper):
        value = float(scores[frame])
        hit = value >= threshold if mode == "high" else value <= threshold
        if hit:
            if run_start is None:
                run_start = frame
            continue
        if run_start is not None and frame - run_start >= min_frames:
            run = scores[run_start:frame]
            score = max(float(v) for v in run) if mode == "high" else min(float(v) for v in run)
            runs.append((run_start, frame, score))
        run_start = None
    if run_start is not None and upper - run_start >= min_frames:
        run = scores[run_start:upper]
        score = max(float(v) for v in run) if mode == "high" else min(float(v) for v in run)
        runs.append((run_start, upper, score))
    if not runs:
        return None

    target_frame = target_s / score_frame_hop_s
    if mode == "high":
        run_start, run_end, _score = min(
            runs,
            key=lambda item: (abs(((item[0] + item[1]) / 2.0) - target_frame), -item[2]),
        )
    else:
        run_start, run_end, _score = min(
            runs,
            key=lambda item: (abs(((item[0] + item[1]) / 2.0) - target_frame), item[2]),
        )
    return ((run_start + run_end) / 2.0) * score_frame_hop_s


def _groups_from_boundaries(
    segments: Sequence[_PackSegment],
    boundaries: Sequence[float],
) -> list[list[_PackSegment]]:
    groups: list[list[_PackSegment]] = [list(segments)]
    for boundary in sorted(boundaries):
        next_groups: list[list[_PackSegment]] = []
        did_split = False
        for group in groups:
            if did_split:
                next_groups.append(group)
                continue
            left, right = _split_group_at_boundary(group, boundary)
            if left and right:
                next_groups.extend([left, right])
                did_split = True
            else:
                next_groups.append(group)
        groups = next_groups
    return groups


def _split_group_at_boundary(
    segments: Sequence[_PackSegment],
    boundary: float,
) -> tuple[list[_PackSegment], list[_PackSegment]]:
    left: list[_PackSegment] = []
    right: list[_PackSegment] = []
    for segment in segments:
        if boundary <= segment.start:
            right.append(segment)
        elif boundary >= segment.end:
            left.append(segment)
        else:
            left.append(
                _PackSegment(
                    start=segment.start,
                    end=boundary,
                    score=segment.score,
                    split_left=segment.split_left,
                    split_right=True,
                    split_policy=segment.split_policy,
                    valley_score_min=segment.valley_score_min,
                    cut_score_max=segment.cut_score_max,
                    drop_gap_score_max=segment.drop_gap_score_max,
                )
            )
            right.append(
                _PackSegment(
                    start=boundary,
                    end=segment.end,
                    score=segment.score,
                    split_left=True,
                    split_right=segment.split_right,
                    split_policy=segment.split_policy,
                    valley_score_min=segment.valley_score_min,
                    cut_score_max=segment.cut_score_max,
                    drop_gap_score_max=segment.drop_gap_score_max,
                )
            )
    return left, right


def _with_risk_metadata(
    chunk: PackedChunk,
    *,
    risk_score: float,
    risk_reasons: tuple[str, ...],
) -> PackedChunk:
    return PackedChunk(
        start=chunk.start,
        end=chunk.end,
        vad_segments=chunk.vad_segments,
        duration=chunk.duration,
        left_padding_s=chunk.left_padding_s,
        right_padding_s=chunk.right_padding_s,
        split_reason=chunk.split_reason,
        parent_chunk_id=chunk.parent_chunk_id,
        island_id=chunk.island_id,
        island_count=chunk.island_count,
        core_start=chunk.core_start,
        core_end=chunk.core_end,
        internal_gap_count=chunk.internal_gap_count,
        internal_gap_max_s=chunk.internal_gap_max_s,
        split_policy=chunk.split_policy,
        valley_split_count=chunk.valley_split_count,
        valley_score_min=chunk.valley_score_min,
        cut_split_count=chunk.cut_split_count,
        cut_score_max=chunk.cut_score_max,
        drop_gap_split_count=chunk.drop_gap_split_count,
        drop_gap_score_max=chunk.drop_gap_score_max,
        risk_split_count=1,
        risk_score=risk_score,
        risk_reasons=risk_reasons,
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
    split_policy: str = "r15_pre_asr_island_v1",
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
        split_policy=split_policy,
        valley_split_count=child.valley_split_count,
        valley_score_min=child.valley_score_min,
        cut_split_count=child.cut_split_count,
        cut_score_max=child.cut_score_max,
        drop_gap_split_count=child.drop_gap_split_count,
        drop_gap_score_max=child.drop_gap_score_max,
        risk_split_count=child.risk_split_count,
        risk_score=child.risk_score,
        risk_reasons=child.risk_reasons,
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
