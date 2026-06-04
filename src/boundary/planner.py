from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from boundary.candidates import (
    BoundaryCandidate,
    CandidateExtractionConfig,
    best_candidate_near_target,
    extract_boundary_candidates,
    gap_midpoint_candidate,
)
from boundary.features import BoundaryFeatureBundle, summarize_gap_features
from boundary.refiner import BoundaryDecision, BoundaryRefiner, RefinerInput, SequenceBoundaryRefiner
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
    sequence_batch_size: int = 256
    dp_chunk_base_cost: float = 0.04
    dp_over_target_weight: float = 0.30
    dp_far_over_target_weight: float = 1.50
    dp_under_min_weight: float = 0.20
    dp_long_gap_weight: float = 0.35
    dp_split_merge_weight: float = 0.35

    def signature(self) -> dict:
        return {
            "planner": "constrained_sequence_dp_planner_v2",
            "frame_hop_s": self.frame_hop_s,
            "max_chunk_s": self.max_chunk_s,
            "target_chunk_s": self.target_chunk_s,
            "min_chunk_s": self.min_chunk_s,
            "start_weight": self.start_weight,
            "target_padding_s": self.target_padding_s,
            "max_splits_per_segment": self.max_splits_per_segment,
            "sequence_batch_size": self.sequence_batch_size,
            "dp_chunk_base_cost": self.dp_chunk_base_cost,
            "dp_over_target_weight": self.dp_over_target_weight,
            "dp_far_over_target_weight": self.dp_far_over_target_weight,
            "dp_under_min_weight": self.dp_under_min_weight,
            "dp_long_gap_weight": self.dp_long_gap_weight,
            "dp_split_merge_weight": self.dp_split_merge_weight,
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


@dataclass(frozen=True)
class _ForcedRun:
    islands: list[PlannedIsland]
    terminal_reason: str


class GapSequenceFeatureProvider(Protocol):
    def features_for_gap(
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
    refiner: BoundaryRefiner | None = None,
    sequence_refiner: SequenceBoundaryRefiner | None = None,
    sequence_feature_provider: GapSequenceFeatureProvider | None = None,
) -> list[PlannedChunk]:
    _validate_config(config)
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    _validate_segments(ordered)
    islands = [_island_from_segment(segment) for segment in ordered]
    islands = _split_on_boundary_candidates(islands, features=features, config=config)
    islands = _split_overlong_islands(islands, config=config)
    return _pack_islands(
        islands,
        features=features,
        config=config,
        refiner=refiner,
        sequence_refiner=sequence_refiner,
        sequence_feature_provider=sequence_feature_provider,
    )


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
    if config.sequence_batch_size <= 0:
        raise ValueError("sequence_batch_size must be positive")
    for field_name in (
        "dp_chunk_base_cost",
        "dp_over_target_weight",
        "dp_far_over_target_weight",
        "dp_under_min_weight",
        "dp_long_gap_weight",
        "dp_split_merge_weight",
    ):
        if getattr(config, field_name) < 0:
            raise ValueError(f"{field_name} must be non-negative")


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
    sequence_refiner: SequenceBoundaryRefiner | None,
    sequence_feature_provider: GapSequenceFeatureProvider | None,
) -> list[PlannedChunk]:
    sequence_decisions = _precompute_sequence_decisions(
        islands,
        sequence_refiner=sequence_refiner,
        sequence_feature_provider=sequence_feature_provider,
        sequence_batch_size=config.sequence_batch_size,
    )
    if sequence_refiner is not None and sequence_feature_provider is not None:
        return _pack_islands_with_dp(
            islands,
            config=config,
            refiner=refiner,
            sequence_decisions=sequence_decisions,
        )

    chunks: list[PlannedChunk] = []
    current: list[PlannedIsland] = []
    current_decision: BoundaryDecision | None = None

    for index, island in enumerate(islands):
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
            sequence_decision=sequence_decisions[index - 1] if index > 0 else None,
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


def _pack_islands_with_dp(
    islands: Sequence[PlannedIsland],
    *,
    config: BoundaryPlannerConfig,
    refiner: BoundaryRefiner | None,
    sequence_decisions: Sequence[BoundaryDecision | None],
) -> list[PlannedChunk]:
    chunks: list[PlannedChunk] = []
    cursor = 0
    for run in _forced_runs(islands):
        run_decisions = list(sequence_decisions[cursor : cursor + max(0, len(run.islands) - 1)])
        chunks.extend(
            _dp_pack_run(
                run.islands,
                terminal_reason=run.terminal_reason,
                config=config,
                refiner=refiner,
                gap_decisions=run_decisions,
            )
        )
        cursor += len(run.islands)
    return chunks


def _forced_runs(islands: Sequence[PlannedIsland]) -> list[_ForcedRun]:
    if not islands:
        return []
    runs: list[_ForcedRun] = []
    current: list[PlannedIsland] = []
    for island in islands:
        if island.force_break_before and current:
            runs.append(
                _ForcedRun(
                    islands=current,
                    terminal_reason=island.boundary_reason or "boundary_candidate",
                )
            )
            current = [island]
            continue
        current.append(island)
    if current:
        runs.append(_ForcedRun(islands=current, terminal_reason="tail"))
    return runs


def _dp_pack_run(
    islands: Sequence[PlannedIsland],
    *,
    terminal_reason: str,
    config: BoundaryPlannerConfig,
    refiner: BoundaryRefiner | None,
    gap_decisions: Sequence[BoundaryDecision | None],
) -> list[PlannedChunk]:
    if not islands:
        return []
    if len(islands) == 1:
        return [PlannedChunk(islands=list(islands), split_reason=terminal_reason)]

    gap_infos = [
        _gap_decision_for_dp(
            islands[index],
            islands[index + 1],
            decision=gap_decisions[index] if index < len(gap_decisions) else None,
            config=config,
            refiner=refiner,
        )
        for index in range(len(islands) - 1)
    ]

    count = len(islands)
    inf = 1e18
    costs = [inf] * (count + 1)
    previous = [-1] * (count + 1)
    costs[0] = 0.0
    for end in range(1, count + 1):
        for start in range(end - 1, -1, -1):
            core_s = islands[end - 1].end - islands[start].start
            if core_s > config.max_chunk_s and start < end - 1:
                continue
            candidate_cost = costs[start] + _chunk_dp_cost(
                islands,
                start=start,
                end=end,
                gap_infos=gap_infos,
                config=config,
            )
            if start > 0:
                candidate_cost += _split_gap_cost(gap_infos[start - 1], config=config)
            if candidate_cost < costs[end]:
                costs[end] = candidate_cost
                previous[end] = start

    if previous[count] < 0:
        return [
            PlannedChunk(
                islands=[island],
                split_reason=terminal_reason if index == count - 1 else "planner_dp",
                boundary_decision=gap_infos[index].decision if index < len(gap_infos) else None,
            )
            for index, island in enumerate(islands)
        ]

    spans: list[tuple[int, int]] = []
    cursor = count
    while cursor > 0:
        start = previous[cursor]
        if start < 0:
            break
        spans.append((start, cursor))
        cursor = start
    spans.reverse()

    planned: list[PlannedChunk] = []
    for start, end in spans:
        if end < count:
            boundary_decision = gap_infos[end - 1].decision
            split_reason = _dp_split_reason(boundary_decision)
        else:
            boundary_decision = None
            split_reason = terminal_reason
        planned.append(
            PlannedChunk(
                islands=list(islands[start:end]),
                split_reason=split_reason,
                boundary_decision=_dp_boundary_decision(boundary_decision, gap_infos[end - 1]) if end < count else None,
            )
        )
    return planned


@dataclass(frozen=True)
class _GapInfo:
    gap_s: float
    merge_score: float
    decision: BoundaryDecision | None


def _gap_decision_for_dp(
    left: PlannedIsland,
    right: PlannedIsland,
    *,
    decision: BoundaryDecision | None,
    config: BoundaryPlannerConfig,
    refiner: BoundaryRefiner | None,
) -> _GapInfo:
    gap_s = right.start - left.end
    if decision is not None:
        return _GapInfo(gap_s=gap_s, merge_score=_clamp01(decision.score), decision=decision)
    if gap_s < 0.0:
        return _GapInfo(gap_s=gap_s, merge_score=1.0, decision=None)
    if refiner is not None:
        item = RefinerInput(
            gap_s=gap_s,
            left_start=left.start,
            left_end=left.end,
            right_start=right.start,
            right_end=right.end,
            current_core_s=max(0.0, left.end - left.start),
            proposed_core_s=max(0.0, right.end - left.start),
            gap_merge_s=_bootstrap_gap_scale_s(config),
            left_score=left.score,
            right_score=right.score,
        )
        refiner_decision = refiner.decide_gap(item)
        return _GapInfo(
            gap_s=gap_s,
            merge_score=_clamp01(refiner_decision.score),
            decision=refiner_decision,
        )
    score = max(0.0, 1.0 - gap_s / max(_bootstrap_gap_scale_s(config), 1e-6))
    return _GapInfo(gap_s=gap_s, merge_score=_clamp01(score), decision=None)


def _chunk_dp_cost(
    islands: Sequence[PlannedIsland],
    *,
    start: int,
    end: int,
    gap_infos: Sequence[_GapInfo],
    config: BoundaryPlannerConfig,
) -> float:
    core_s = max(0.0, islands[end - 1].end - islands[start].start)
    cost = config.dp_chunk_base_cost
    target = max(config.target_chunk_s, 1e-6)
    if core_s > config.target_chunk_s:
        cost += config.dp_over_target_weight * ((core_s - config.target_chunk_s) / target) ** 2
    if core_s > 2.0 * config.target_chunk_s:
        cost += config.dp_far_over_target_weight * ((core_s - 2.0 * config.target_chunk_s) / target) ** 2
    if core_s < config.min_chunk_s:
        cost += config.dp_under_min_weight * ((config.min_chunk_s - core_s) / max(config.min_chunk_s, 1e-6)) ** 2
    for index in range(start, end - 1):
        cost += _merge_gap_cost(gap_infos[index], config=config)
    return cost


def _merge_gap_cost(info: _GapInfo, *, config: BoundaryPlannerConfig) -> float:
    score = _clamp01(info.merge_score)
    cost = config.start_weight * (1.0 - score)
    if info.gap_s > _bootstrap_gap_scale_s(config):
        cost += config.dp_long_gap_weight * min(
            4.0,
            (info.gap_s - _bootstrap_gap_scale_s(config)) / max(config.target_chunk_s, 1e-6),
        )
    return cost


def _split_gap_cost(info: _GapInfo, *, config: BoundaryPlannerConfig) -> float:
    score = _clamp01(info.merge_score)
    return config.dp_split_merge_weight * score * score


def _dp_split_reason(decision: BoundaryDecision | None) -> str:
    if decision is not None and not decision.merge:
        return f"boundary_refiner:{decision.reason}"
    return "planner_dp"


def _dp_boundary_decision(
    decision: BoundaryDecision | None,
    info: _GapInfo,
) -> BoundaryDecision:
    if decision is not None and not decision.merge:
        return decision
    return BoundaryDecision(
        False,
        _clamp01(info.merge_score),
        "planner_dp",
        source="boundary_planner",
    )


def _score_gap(
    current: Sequence[PlannedIsland],
    island: PlannedIsland,
    *,
    features: BoundaryFeatureBundle,
    config: BoundaryPlannerConfig,
    refiner: BoundaryRefiner | None,
    sequence_decision: BoundaryDecision | None,
    gap_s: float,
    proposed_core_s: float,
) -> BoundaryDecision | None:
    if gap_s < 0.0:
        return None
    if sequence_decision is not None:
        return sequence_decision
    if refiner is None:
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


def _precompute_sequence_decisions(
    islands: Sequence[PlannedIsland],
    *,
    sequence_refiner: SequenceBoundaryRefiner | None,
    sequence_feature_provider: GapSequenceFeatureProvider | None,
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
            sequence_feature_provider.features_for_gap(
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


def _bootstrap_gap_scale_s(config: BoundaryPlannerConfig) -> float:
    return max(0.2, min(1.5, config.target_chunk_s / 6.0))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def max_optional(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _core_duration(islands: Sequence[PlannedIsland]) -> float:
    if not islands:
        return 0.0
    return max(0.0, islands[-1].end - islands[0].start)
