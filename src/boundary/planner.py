from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from boundary.refiner import BoundaryDecision, SequenceBoundaryRefiner
from boundary.base import SpeechSegment


@dataclass(frozen=True)
class BoundaryPlannerConfig:
    frame_hop_s: float = 1.0 / 29.97
    sequence_batch_size: int = 256

    def signature(self) -> dict:
        return {
            "planner": "edge_sequence_island_planner_v7",
            "frame_hop_s": self.frame_hop_s,
            "sequence_batch_size": self.sequence_batch_size,
        }


@dataclass(frozen=True)
class PlannedIsland:
    start: float
    end: float
    score: float | None = None
    split_left: bool = False
    split_right: bool = False
    force_break_before: bool = False
    boundary_score: float | None = None
    boundary_reason: str = ""
    boundary_source: str = ""
    subtitle_min_duration_s: float | None = None
    below_subtitle_min_duration: bool = False
    micro_chunk_candidate: bool = False
    micro_resolve_action: str = ""
    micro_resolve_reason: str = ""
    left_split_score: float | None = None
    right_split_score: float | None = None
    left_split_prominence: float | None = None
    right_split_prominence: float | None = None
    left_split_speech_valley: float | None = None
    right_split_speech_valley: float | None = None
    primary_cut_candidates: list[dict[str, Any]] | None = None
    weak_cut_candidates: list[dict[str, Any]] | None = None

    def to_speech_segment(self) -> SpeechSegment:
        return SpeechSegment(
            start=self.start,
            end=self.end,
            score=self.score,
            subtitle_min_duration_s=self.subtitle_min_duration_s,
            below_subtitle_min_duration=self.below_subtitle_min_duration,
            micro_chunk_candidate=self.micro_chunk_candidate,
            micro_resolve_action=self.micro_resolve_action,
            micro_resolve_reason=self.micro_resolve_reason,
            left_split_score=self.left_split_score,
            right_split_score=self.right_split_score,
            left_split_prominence=self.left_split_prominence,
            right_split_prominence=self.right_split_prominence,
            left_split_speech_valley=self.left_split_speech_valley,
            right_split_speech_valley=self.right_split_speech_valley,
            primary_cut_candidates=list(self.primary_cut_candidates or []),
            weak_cut_candidates=list(self.weak_cut_candidates or []),
        )


@dataclass(frozen=True)
class PlannedChunk:
    islands: list[PlannedIsland]
    split_reason: str
    boundary_decision: BoundaryDecision | None = None


class BoundarySequenceFeatureProvider(Protocol):
    def features_for_boundary(
        self,
        *,
        left_start_s: float,
        left_end_s: float,
        right_start_s: float,
        right_end_s: float,
    ) -> list[float]: ...


def plan_boundary_chunks(
    segments: Sequence[SpeechSegment],
    *,
    config: BoundaryPlannerConfig,
    sequence_refiner: SequenceBoundaryRefiner | None = None,
    sequence_feature_provider: BoundarySequenceFeatureProvider | None = None,
) -> list[PlannedChunk]:
    _validate_config(config)
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    _validate_segments(ordered)
    islands = [_island_from_segment(segment) for segment in ordered]
    return _pack_islands(
        islands,
        config=config,
        sequence_refiner=sequence_refiner,
        sequence_feature_provider=sequence_feature_provider,
    )


def _validate_config(config: BoundaryPlannerConfig) -> None:
    if config.frame_hop_s <= 0:
        raise ValueError("frame_hop_s must be positive")
    if config.sequence_batch_size <= 0:
        raise ValueError("sequence_batch_size must be positive")


def _validate_segments(segments: Sequence[SpeechSegment]) -> None:
    for segment in segments:
        if segment.end < segment.start:
            raise ValueError("segment end must be greater than or equal to start")


def _island_from_segment(segment: SpeechSegment) -> PlannedIsland:
    return PlannedIsland(
        start=segment.start,
        end=segment.end,
        score=segment.score,
        boundary_score=segment.right_split_score,
        boundary_reason=segment.micro_resolve_reason,
        boundary_source="micro_split_resolver" if segment.micro_resolve_reason else "",
        subtitle_min_duration_s=segment.subtitle_min_duration_s,
        below_subtitle_min_duration=segment.below_subtitle_min_duration,
        micro_chunk_candidate=segment.micro_chunk_candidate,
        micro_resolve_action=segment.micro_resolve_action,
        micro_resolve_reason=segment.micro_resolve_reason,
        left_split_score=segment.left_split_score,
        right_split_score=segment.right_split_score,
        left_split_prominence=segment.left_split_prominence,
        right_split_prominence=segment.right_split_prominence,
        left_split_speech_valley=segment.left_split_speech_valley,
        right_split_speech_valley=segment.right_split_speech_valley,
        primary_cut_candidates=list(segment.primary_cut_candidates or []),
        weak_cut_candidates=list(segment.weak_cut_candidates or []),
    )


def _pack_islands(
    islands: Sequence[PlannedIsland],
    *,
    config: BoundaryPlannerConfig,
    sequence_refiner: SequenceBoundaryRefiner | None,
    sequence_feature_provider: BoundarySequenceFeatureProvider | None,
) -> list[PlannedChunk]:
    chunks: list[PlannedChunk] = []
    sequence_decisions = _precompute_sequence_decisions(
        islands,
        sequence_refiner=sequence_refiner,
        sequence_feature_provider=sequence_feature_provider,
        sequence_batch_size=config.sequence_batch_size,
    )
    for index, island in enumerate(islands):
        decision = sequence_decisions[index] if index + 1 < len(islands) else None
        chunks.append(
            PlannedChunk(
                islands=[island],
                split_reason=island.boundary_reason or "speech_island",
                boundary_decision=decision,
            )
        )
    return chunks


def _precompute_sequence_decisions(
    islands: Sequence[PlannedIsland],
    *,
    sequence_refiner: SequenceBoundaryRefiner | None,
    sequence_feature_provider: BoundarySequenceFeatureProvider | None,
    sequence_batch_size: int,
) -> list[BoundaryDecision | None]:
    if len(islands) < 2:
        return []
    if sequence_refiner is None or sequence_feature_provider is None:
        return [None] * (len(islands) - 1)

    features_by_gap: list[list[float]] = []
    gap_indexes: list[int] = []
    for index, (left, right) in enumerate(zip(islands, islands[1:])):
        if right.start < left.end:
            continue
        features_by_gap.append(
            sequence_feature_provider.features_for_boundary(
                left_start_s=left.start,
                left_end_s=left.end,
                right_start_s=right.start,
                right_end_s=right.end,
            )
        )
        gap_indexes.append(index)

    decisions: list[BoundaryDecision | None] = [None] * (len(islands) - 1)
    if not features_by_gap:
        return decisions
    for start in range(0, len(features_by_gap), sequence_batch_size):
        end = min(len(features_by_gap), start + sequence_batch_size)
        sequence_decisions = sequence_refiner.decide_sequence(features_by_gap[start:end])
        if len(sequence_decisions) != end - start:
            raise ValueError("sequence boundary refiner returned a decision count mismatch")
        for index, decision in zip(gap_indexes[start:end], sequence_decisions):
            decisions[index] = decision
    return decisions
