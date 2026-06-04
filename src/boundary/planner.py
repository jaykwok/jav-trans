from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from boundary.candidates import (
    BoundaryCandidate,
    CandidateExtractionConfig,
    best_candidate_near_target,
    extract_boundary_candidates,
    gap_midpoint_candidate,
)
from boundary.features import BoundaryFeatureBundle, summarize_gap_features
from boundary.refiner import BoundaryDecision, BoundaryRefiner, RefinerInput
from boundary.base import SpeechSegment


@dataclass(frozen=True)
class BoundaryPlannerConfig:
    frame_hop_s: float = 1.0 / 29.97
    max_chunk_s: float = 30.0
    target_chunk_s: float = 9.0
    min_chunk_s: float = 0.4
    start_weight: float = 1.5
    target_padding_s: float = 2.0
    max_splits_per_segment: int = 16

    def signature(self) -> dict:
        return {
            "planner": "constrained_boundary_planner_v1",
            "frame_hop_s": self.frame_hop_s,
            "max_chunk_s": self.max_chunk_s,
            "target_chunk_s": self.target_chunk_s,
            "min_chunk_s": self.min_chunk_s,
            "start_weight": self.start_weight,
            "target_padding_s": self.target_padding_s,
            "max_splits_per_segment": self.max_splits_per_segment,
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


def plan_boundary_chunks(
    segments: Sequence[SpeechSegment],
    *,
    features: BoundaryFeatureBundle,
    config: BoundaryPlannerConfig,
    refiner: BoundaryRefiner | None = None,
) -> list[PlannedChunk]:
    _validate_config(config)
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    _validate_segments(ordered)
    islands = [_island_from_segment(segment) for segment in ordered]
    islands = _split_on_boundary_candidates(islands, features=features, config=config)
    islands = _split_overlong_islands(islands, config=config)
    return _pack_islands(islands, features=features, config=config, refiner=refiner)


def _validate_config(config: BoundaryPlannerConfig) -> None:
    if config.frame_hop_s <= 0:
        raise ValueError("frame_hop_s must be positive")
    if config.max_chunk_s <= 0:
        raise ValueError("max_chunk_s must be positive")
    if config.target_chunk_s <= 0:
        raise ValueError("target_chunk_s must be positive")
    if config.min_chunk_s < 0:
        raise ValueError("min_chunk_s must be non-negative")
    if config.start_weight <= 0:
        raise ValueError("start_weight must be positive")
    if config.target_padding_s < 0:
        raise ValueError("target_padding_s must be non-negative")
    if config.max_splits_per_segment < 0:
        raise ValueError("max_splits_per_segment must be non-negative")


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
    speech_limit_s = config.max_chunk_s - 2.0 * config.target_padding_s
    if speech_limit_s <= 0.0:
        speech_limit_s = config.max_chunk_s
    split: list[PlannedIsland] = []
    for island in islands:
        duration_s = island.end - island.start
        if duration_s <= config.max_chunk_s:
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
    features: BoundaryFeatureBundle,
    config: BoundaryPlannerConfig,
    refiner: BoundaryRefiner | None,
) -> list[PlannedChunk]:
    chunks: list[PlannedChunk] = []
    current: list[PlannedIsland] = []
    current_decision: BoundaryDecision | None = None

    for island in islands:
        if not current:
            current = [island]
            current_decision = None
            continue

        if island.force_break_before:
            chunks.append(
                PlannedChunk(
                    islands=current,
                    split_reason=island.boundary_reason or "boundary_candidate",
                    boundary_decision=current_decision,
                )
            )
            current = [island]
            current_decision = None
            continue

        gap_s = island.start - current[-1].end
        proposed = [*current, island]
        proposed_core_s = _core_duration(proposed)
        within_cap = proposed_core_s <= config.max_chunk_s
        decision = _score_gap(
            current,
            island,
            features=features,
            config=config,
            refiner=refiner,
            gap_s=gap_s,
            proposed_core_s=proposed_core_s,
        )
        should_merge = decision.merge if decision is not None else gap_s <= _bootstrap_gap_scale_s(config)

        if should_merge and within_cap:
            current = proposed
            current_decision = decision
            continue

        if decision is not None and not decision.merge:
            reason = f"boundary_refiner:{decision.reason}"
        elif not within_cap:
            reason = "planner_max"
        else:
            reason = "gap"
        chunks.append(
            PlannedChunk(
                islands=current,
                split_reason=reason,
                boundary_decision=decision,
            )
        )
        current = [island]
        current_decision = None

    if current:
        chunks.append(
            PlannedChunk(
                islands=current,
                split_reason="tail",
                boundary_decision=current_decision,
            )
        )
    return chunks


def _score_gap(
    current: Sequence[PlannedIsland],
    island: PlannedIsland,
    *,
    features: BoundaryFeatureBundle,
    config: BoundaryPlannerConfig,
    refiner: BoundaryRefiner | None,
    gap_s: float,
    proposed_core_s: float,
) -> BoundaryDecision | None:
    if refiner is None or gap_s < 0.0:
        return None
    left = current[-1]
    gap_features = summarize_gap_features(features, start_s=left.end, end_s=island.start)
    gap_candidate = gap_midpoint_candidate(
        left_end_s=left.end,
        right_start_s=island.start,
        target_chunk_s=config.target_chunk_s,
    )
    item = RefinerInput(
        gap_s=gap_s,
        left_start=left.start,
        left_end=left.end,
        right_start=island.start,
        right_end=island.end,
        current_core_s=_core_duration(current),
        proposed_core_s=proposed_core_s,
        gap_merge_s=_bootstrap_gap_scale_s(config),
        left_score=left.score,
        right_score=island.score,
        valley_score_min=gap_features.valley_score_min,
        cut_score_max=max_optional(
            gap_features.cut_score_max,
            max_optional(left.boundary_score, island.boundary_score)
            if {left.boundary_source, island.boundary_source} & {"cut"}
            else None,
        ),
        gap_boundary_score=None if gap_candidate is None else gap_candidate.score,
    )
    return refiner.decide_gap(item)


def _bootstrap_gap_scale_s(config: BoundaryPlannerConfig) -> float:
    return max(0.2, min(1.5, config.target_chunk_s / 6.0))


def max_optional(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _core_duration(islands: Sequence[PlannedIsland]) -> float:
    if not islands:
        return 0.0
    return max(0.0, islands[-1].end - islands[0].start)
