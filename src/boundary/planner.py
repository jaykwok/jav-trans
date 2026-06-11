from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from boundary.candidates import (
    BoundaryCandidate,
    CandidateExtractionConfig,
    best_candidate_near_target,
    extract_boundary_candidates,
    soft_candidate_near_target,
)
from boundary.features import BoundaryFeatureBundle
from boundary.refiner import BoundaryDecision, SequenceBoundaryRefiner
from boundary.base import SpeechSegment


@dataclass(frozen=True)
class BoundaryPlannerConfig:
    frame_hop_s: float = 1.0 / 29.97
    max_core_chunk_s: float = 5.0
    target_chunk_s: float = 3.0
    min_chunk_s: float = 0.4
    max_splits_per_segment: int = 16
    sequence_batch_size: int = 256

    def signature(self) -> dict:
        return {
            "planner": "candidate_sequence_core_planner_v5",
            "frame_hop_s": self.frame_hop_s,
            "max_core_chunk_s": self.max_core_chunk_s,
            "target_chunk_s": self.target_chunk_s,
            "min_chunk_s": self.min_chunk_s,
            "max_splits_per_segment": self.max_splits_per_segment,
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

    def to_speech_segment(self) -> SpeechSegment:
        return SpeechSegment(start=self.start, end=self.end, score=self.score)


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
    features: BoundaryFeatureBundle,
    config: BoundaryPlannerConfig,
    sequence_refiner: SequenceBoundaryRefiner | None = None,
    sequence_feature_provider: BoundarySequenceFeatureProvider | None = None,
) -> list[PlannedChunk]:
    _validate_config(config)
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    _validate_segments(ordered)
    islands = [_island_from_segment(segment) for segment in ordered]
    islands = _split_on_boundary_candidates(islands, features=features, config=config)
    islands = _split_overlong_islands(islands, config=config)
    return _pack_islands(
        islands,
        config=config,
        sequence_refiner=sequence_refiner,
        sequence_feature_provider=sequence_feature_provider,
    )


def _validate_config(config: BoundaryPlannerConfig) -> None:
    if config.frame_hop_s <= 0:
        raise ValueError("frame_hop_s must be positive")
    if config.max_core_chunk_s <= 0:
        raise ValueError("max_core_chunk_s must be positive")
    if config.target_chunk_s <= 0:
        raise ValueError("target_chunk_s must be positive")
    if config.min_chunk_s < 0:
        raise ValueError("min_chunk_s must be non-negative")
    if config.max_splits_per_segment < 0:
        raise ValueError("max_splits_per_segment must be non-negative")
    if config.sequence_batch_size <= 0:
        raise ValueError("sequence_batch_size must be positive")


def _validate_segments(segments: Sequence[SpeechSegment]) -> None:
    for segment in segments:
        if segment.end < segment.start:
            raise ValueError("segment end must be greater than or equal to start")


def _island_from_segment(segment: SpeechSegment) -> PlannedIsland:
    return PlannedIsland(start=segment.start, end=segment.end, score=segment.score)


def _split_on_boundary_candidates(
    islands: Sequence[PlannedIsland],
    *,
    features: BoundaryFeatureBundle,
    config: BoundaryPlannerConfig,
) -> list[PlannedIsland]:
    result: list[PlannedIsland] = []
    extractor_config = CandidateExtractionConfig(
        min_chunk_s=config.min_chunk_s,
        target_chunk_s=config.target_chunk_s,
    )
    for island in islands:
        result.extend(
            _split_single_island(
                island,
                features=features,
                config=config,
                extractor_config=extractor_config,
            )
        )
    return result


def _split_single_island(
    island: PlannedIsland,
    *,
    features: BoundaryFeatureBundle,
    config: BoundaryPlannerConfig,
    extractor_config: CandidateExtractionConfig,
) -> list[PlannedIsland]:
    if island.end - island.start <= config.target_chunk_s:
        return [island]
    if config.max_splits_per_segment == 0:
        return [island]

    parts: list[PlannedIsland] = [island]
    split_count = 0
    while split_count < config.max_splits_per_segment:
        index = _longest_part_index(parts, config.target_chunk_s)
        if index is None:
            break
        part = parts[index]
        target = min(
            part.end - config.min_chunk_s,
            max(part.start + config.min_chunk_s, part.start + config.target_chunk_s),
        )
        candidates = extract_boundary_candidates(
            start_s=part.start,
            end_s=part.end,
            features=features,
            config=extractor_config,
        )
        candidate = best_candidate_near_target(candidates, target_s=target)
        if candidate is None:
            candidate = soft_candidate_near_target(
                start_s=part.start,
                end_s=part.end,
                target_s=target,
                features=features,
                config=extractor_config,
            )
        if candidate is None:
            break
        left, right = _split_island_at_candidate(part, candidate)
        if left is None or right is None:
            break
        parts[index : index + 1] = [left, right]
        split_count += 1
    return parts


def _longest_part_index(parts: Sequence[PlannedIsland], target_s: float) -> int | None:
    candidates = [
        (index, part.end - part.start)
        for index, part in enumerate(parts)
        if part.end - part.start > target_s
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def _split_island_at_candidate(
    island: PlannedIsland,
    candidate: BoundaryCandidate,
) -> tuple[PlannedIsland | None, PlannedIsland | None]:
    if candidate.time_s <= island.start or candidate.time_s >= island.end:
        return None, None
    left = PlannedIsland(
        start=island.start,
        end=candidate.time_s,
        score=island.score,
        split_left=island.split_left,
        split_right=True,
        force_break_before=island.force_break_before,
        boundary_score=candidate.score,
        boundary_reason=candidate.reason,
        boundary_source=candidate.source,
    )
    right = PlannedIsland(
        start=candidate.time_s,
        end=island.end,
        score=island.score,
        split_left=True,
        split_right=island.split_right,
        force_break_before=True,
        boundary_score=candidate.score,
        boundary_reason=candidate.reason,
        boundary_source=candidate.source,
    )
    return left, right


def _split_overlong_islands(
    islands: Sequence[PlannedIsland],
    *,
    config: BoundaryPlannerConfig,
) -> list[PlannedIsland]:
    speech_limit_s = config.max_core_chunk_s
    split: list[PlannedIsland] = []
    for island in islands:
        duration_s = island.end - island.start
        if duration_s <= config.max_core_chunk_s:
            split.append(island)
            continue

        cursor = island.start
        first = True
        while cursor < island.end:
            next_end = min(island.end, cursor + speech_limit_s)
            split.append(
                PlannedIsland(
                    start=cursor,
                    end=next_end,
                    score=island.score,
                    split_left=not first or island.split_left,
                    split_right=next_end < island.end or island.split_right,
                    force_break_before=island.force_break_before if first else True,
                    boundary_score=island.boundary_score,
                    boundary_reason="overlong",
                    boundary_source=island.boundary_source,
                )
            )
            cursor = next_end
            first = False
    return split


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
        reason = island.boundary_reason or "speech_island"
        chunks.append(
            PlannedChunk(
                islands=[island],
                split_reason=reason,
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
        if right.force_break_before or right.start - left.end < 0.0:
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
