from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from boundary.refiner import BoundaryDecision, SequenceBoundaryRefiner
from boundary.base import SpeechSegment


class BoundarySequenceFeatureProvider(Protocol):
    def features_for_boundary(
        self,
        *,
        left_start_s: float,
        left_end_s: float,
        right_start_s: float,
        right_end_s: float,
    ) -> list[float]: ...


@dataclass(frozen=True)
class _PlannedIsland:
    start: float
    end: float
    score: float | None = None
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
class _PlannedChunk:
    islands: list[_PlannedIsland]
    split_reason: str
    boundary_decision: BoundaryDecision | None = None


@dataclass(frozen=True)
class PackedChunk:
    start: float
    end: float
    speech_segments: list[SpeechSegment]
    duration: float
    split_reason: str
    parent_chunk_id: int | None = None
    island_id: int | None = None
    island_count: int | None = None
    core_start: float | None = None
    core_end: float | None = None
    raw_start: float | None = None
    raw_end: float | None = None
    raw_duration: float | None = None
    acoustic_start: float | None = None
    acoustic_end: float | None = None
    acoustic_duration: float | None = None
    internal_gap_count: int = 0
    internal_gap_max_s: float = 0.0
    boundary_score: float | None = None
    boundary_reason: str = ""
    boundary_source: str = ""
    boundary_start_refine_delta_s: float | None = None
    boundary_end_refine_delta_s: float | None = None
    boundary_decision_source: str = ""
    refiner_pred_start_delta_s: float | None = None
    refiner_pred_end_delta_s: float | None = None
    refiner_applied_start_delta_s: float | None = None
    refiner_applied_end_delta_s: float | None = None
    refiner_start_confidence: float | None = None
    refiner_end_confidence: float | None = None
    refiner_start_source: str = ""
    refiner_end_source: str = ""
    refiner_safety_action: str = ""
    refiner_safety_reason: str = ""
    refiner_effective_start_delta_max_s: float | None = None
    refiner_effective_end_delta_max_s: float | None = None
    refiner_fallback_used: bool = False
    refiner_shared_boundary_adjusted: bool = False
    scorer_speech_mean: float | None = None
    scorer_speech_max: float | None = None
    scorer_speech_p90: float | None = None
    scorer_speech_p10: float | None = None
    scorer_speech_p50: float | None = None
    scorer_speech_std: float | None = None
    scorer_speech_active_ratio_05: float | None = None
    scorer_speech_active_ratio_07: float | None = None
    scorer_speech_active_ratio_09: float | None = None
    scorer_split_mean: float | None = None
    scorer_split_max: float | None = None
    scorer_split_p90: float | None = None
    scorer_split_std: float | None = None
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
    pre_asr_ptm_pooling_schema: str = ""
    pre_asr_ptm_pooling_bins: int | None = None
    pre_asr_ptm_pooling_dim: int | None = None
    pre_asr_ptm_pooled_features: list[float] | None = None


@dataclass(frozen=True)
class PackingLayoutConfig:
    min_core_s: float = 0.05


def pack_speech_segments(
    segments: Sequence[SpeechSegment],
    *,
    sequence_boundary_refiner: SequenceBoundaryRefiner | None = None,
    sequence_feature_provider: BoundarySequenceFeatureProvider | None = None,
    sequence_batch_size: int = 256,
) -> list[PackedChunk]:
    """Convert scorer-produced speech islands into ASR chunks.

    Scorer v7 is the only island splitter. The planner no longer creates
    duration-driven splits or secondary boundary candidates; Boundary Refiner v8
    may only adjust the start/end of already-planned island chunks.
    """

    planned = _plan_boundary_chunks(
        segments,
        sequence_refiner=sequence_boundary_refiner,
        sequence_feature_provider=sequence_feature_provider,
        sequence_batch_size=sequence_batch_size,
    )
    return _materialize_packed_chunks(planned, layout=PackingLayoutConfig())


def _plan_boundary_chunks(
    segments: Sequence[SpeechSegment],
    *,
    sequence_refiner: SequenceBoundaryRefiner | None = None,
    sequence_feature_provider: BoundarySequenceFeatureProvider | None = None,
    sequence_batch_size: int = 256,
) -> list[_PlannedChunk]:
    if sequence_batch_size <= 0:
        raise ValueError("sequence_batch_size must be positive")
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    _validate_segments(ordered)
    islands = [_island_from_segment(segment) for segment in ordered]
    sequence_decisions = _precompute_sequence_decisions(
        islands,
        sequence_refiner=sequence_refiner,
        sequence_feature_provider=sequence_feature_provider,
        sequence_batch_size=sequence_batch_size,
    )
    chunks: list[_PlannedChunk] = []
    for index, island in enumerate(islands):
        decision = sequence_decisions[index] if index + 1 < len(islands) else None
        chunks.append(
            _PlannedChunk(
                islands=[island],
                split_reason=island.boundary_reason or "speech_island",
                boundary_decision=decision,
            )
        )
    return chunks


def _validate_segments(segments: Sequence[SpeechSegment]) -> None:
    for segment in segments:
        if segment.end < segment.start:
            raise ValueError("segment end must be greater than or equal to start")


def _island_from_segment(segment: SpeechSegment) -> _PlannedIsland:
    return _PlannedIsland(
        start=segment.start,
        end=segment.end,
        score=segment.score,
        boundary_score=segment.right_split_score,
        boundary_reason=segment.micro_resolve_reason,
        boundary_source="split_resolver" if segment.micro_resolve_reason else "",
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


def _precompute_sequence_decisions(
    islands: Sequence[_PlannedIsland],
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


def _materialize_packed_chunks(
    planned: Sequence[_PlannedChunk],
    *,
    layout: PackingLayoutConfig,
) -> list[PackedChunk]:
    chunks: list[PackedChunk] = []
    for index, item in enumerate(planned):
        previous_item = planned[index - 1] if index > 0 else None
        next_item = planned[index + 1] if index + 1 < len(planned) else None
        previous_decision = previous_item.boundary_decision if previous_item is not None else None
        chunks.append(
            _make_chunk(
                item.islands,
                layout=layout,
                previous_decision=previous_decision,
                previous_core_end=(
                    previous_item.islands[-1].end if previous_item is not None else None
                ),
                next_core_start=(
                    next_item.islands[0].start if next_item is not None else None
                ),
                split_reason=item.split_reason,
                boundary_decision=item.boundary_decision,
            )
        )
    return chunks


def _make_chunk(
    islands: Sequence[_PlannedIsland],
    *,
    layout: PackingLayoutConfig,
    previous_decision: BoundaryDecision | None,
    previous_core_end: float | None,
    next_core_start: float | None,
    split_reason: str,
    boundary_decision: BoundaryDecision | None,
) -> PackedChunk:
    raw_start = islands[0].start
    raw_end = islands[-1].end
    core_start = raw_start
    core_end = raw_end
    applied_start_delta_s = (
        previous_decision.start_refine_delta_s
        if previous_decision is not None
        else None
    )
    applied_end_delta_s = (
        boundary_decision.end_refine_delta_s
        if boundary_decision is not None
        else None
    )
    core_start, core_end, safety_action, safety_reason = _apply_boundary_delta(
        core_start,
        core_end,
        previous_core_end=previous_core_end,
        next_core_start=next_core_start,
        previous_decision=previous_decision,
        next_decision=boundary_decision,
        min_core_s=layout.min_core_s,
    )

    boundary_scores = [
        float(island.boundary_score)
        for island in islands
        if island.boundary_score is not None
    ]
    boundary_reasons = [island.boundary_reason for island in islands if island.boundary_reason]
    boundary_sources = [island.boundary_source for island in islands if island.boundary_source]
    micro_actions = [island.micro_resolve_action for island in islands if island.micro_resolve_action]
    micro_reasons = [island.micro_resolve_reason for island in islands if island.micro_resolve_reason]

    start = max(0.0, core_start)
    end = core_end
    subtitle_min_duration_s = _first_float(
        island.subtitle_min_duration_s for island in islands
    )
    return PackedChunk(
        start=start,
        end=end,
        speech_segments=[island.to_speech_segment() for island in islands],
        duration=end - start,
        split_reason=split_reason,
        raw_start=raw_start,
        raw_end=raw_end,
        raw_duration=max(0.0, raw_end - raw_start),
        acoustic_start=start,
        acoustic_end=end,
        acoustic_duration=max(0.0, end - start),
        core_start=core_start,
        core_end=core_end,
        internal_gap_count=_internal_gap_count(islands),
        internal_gap_max_s=_internal_gap_max_s(islands),
        boundary_score=(max(boundary_scores) if boundary_scores else None),
        boundary_reason=(",".join(sorted(set(boundary_reasons)))),
        boundary_source=",".join(sorted(set(boundary_sources))),
        boundary_start_refine_delta_s=applied_start_delta_s,
        boundary_end_refine_delta_s=applied_end_delta_s,
        boundary_decision_source=(
            boundary_decision.source if boundary_decision is not None else ""
        ),
        refiner_pred_start_delta_s=_decision_float(previous_decision, "raw_start_refine_delta_s"),
        refiner_pred_end_delta_s=_decision_float(boundary_decision, "raw_end_refine_delta_s"),
        refiner_applied_start_delta_s=applied_start_delta_s,
        refiner_applied_end_delta_s=applied_end_delta_s,
        refiner_start_confidence=_decision_float(previous_decision, "start_confidence"),
        refiner_end_confidence=_decision_float(boundary_decision, "end_confidence"),
        refiner_start_source=_decision_str(previous_decision, "start_source"),
        refiner_end_source=_decision_str(boundary_decision, "end_source"),
        refiner_safety_action=safety_action,
        refiner_safety_reason=safety_reason,
        refiner_effective_start_delta_max_s=_decision_float(
            previous_decision,
            "effective_start_delta_max_s",
        ),
        refiner_effective_end_delta_max_s=_decision_float(
            boundary_decision,
            "effective_end_delta_max_s",
        ),
        refiner_fallback_used=(
            _decision_bool(previous_decision, "fallback_used")
            or _decision_bool(boundary_decision, "fallback_used")
        ),
        refiner_shared_boundary_adjusted=(
            _decision_bool(previous_decision, "shared_boundary_adjusted")
            or _decision_bool(boundary_decision, "shared_boundary_adjusted")
        ),
        subtitle_min_duration_s=subtitle_min_duration_s,
        below_subtitle_min_duration=(
            bool(subtitle_min_duration_s is not None and end - start < subtitle_min_duration_s)
            or any(island.below_subtitle_min_duration for island in islands)
        ),
        micro_chunk_candidate=any(island.micro_chunk_candidate for island in islands),
        micro_resolve_action=",".join(sorted(set(micro_actions))),
        micro_resolve_reason=",".join(sorted(set(micro_reasons))),
        left_split_score=_max_float(island.left_split_score for island in islands),
        right_split_score=_max_float(island.right_split_score for island in islands),
        left_split_prominence=_max_float(island.left_split_prominence for island in islands),
        right_split_prominence=_max_float(island.right_split_prominence for island in islands),
        left_split_speech_valley=_max_float(island.left_split_speech_valley for island in islands),
        right_split_speech_valley=_max_float(island.right_split_speech_valley for island in islands),
        primary_cut_candidates=_merge_cut_candidates(
            island.primary_cut_candidates for island in islands
        ),
        weak_cut_candidates=_merge_cut_candidates(
            island.weak_cut_candidates for island in islands
        ),
    )


def _first_float(values) -> float | None:
    for value in values:
        if value is not None:
            return float(value)
    return None


def _max_float(values) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return max(finite) if finite else None


def _decision_float(decision: BoundaryDecision | None, name: str) -> float | None:
    if decision is None:
        return None
    value = getattr(decision, name, None)
    return None if value is None else float(value)


def _decision_str(decision: BoundaryDecision | None, name: str) -> str:
    if decision is None:
        return ""
    return str(getattr(decision, name, "") or "")


def _decision_bool(decision: BoundaryDecision | None, name: str) -> bool:
    return bool(decision is not None and getattr(decision, name, False))


def _merge_cut_candidates(values) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for candidates in values:
        for candidate in candidates or []:
            if not isinstance(candidate, dict):
                continue
            try:
                time_s = float(candidate["time_s"])
            except (KeyError, TypeError, ValueError):
                continue
            kind = str(candidate.get("kind") or "")
            key = (kind, int(round(time_s * 1000.0)))
            existing = by_key.get(key)
            strength = float(candidate.get("strength") or 0.0)
            existing_strength = float(existing.get("strength") or 0.0) if existing else -1.0
            if existing is None or strength > existing_strength:
                by_key[key] = dict(candidate)
    return [
        by_key[key]
        for key in sorted(by_key, key=lambda item: (float(by_key[item].get("time_s") or 0.0), item[0]))
    ]


def _internal_gap_count(islands: Sequence[_PlannedIsland]) -> int:
    return sum(
        1
        for previous, current in zip(islands, islands[1:])
        if current.start > previous.end
    )


def _internal_gap_max_s(islands: Sequence[_PlannedIsland]) -> float:
    if len(islands) < 2:
        return 0.0
    return max(
        max(0.0, current.start - previous.end)
        for previous, current in zip(islands, islands[1:])
    )


def _apply_boundary_delta(
    core_start: float,
    core_end: float,
    *,
    previous_core_end: float | None,
    next_core_start: float | None,
    previous_decision: BoundaryDecision | None,
    next_decision: BoundaryDecision | None,
    min_core_s: float,
) -> tuple[float, float, str, str]:
    min_duration = max(0.0, float(min_core_s))
    start = float(core_start)
    end = float(core_end)
    start_delta = (
        previous_decision.start_refine_delta_s
        if previous_decision is not None
        else None
    )
    end_delta = next_decision.end_refine_delta_s if next_decision is not None else None
    if start_delta is not None:
        start += float(start_delta)
    if end_delta is not None:
        end += float(end_delta)
    desired_start = start
    desired_end = end

    left_limit = 0.0 if previous_core_end is None else float(previous_core_end)
    right_limit = float("inf") if next_core_start is None else float(next_core_start)
    start = max(left_limit, min(start, max(left_limit, right_limit - min_duration)))
    end = min(right_limit, max(end, start + min_duration))
    reasons: list[str] = []
    if abs(start - desired_start) > 1e-9:
        reasons.append("start_clamped")
    if abs(end - desired_end) > 1e-9:
        reasons.append("end_clamped")
    if end < start:
        end = start
        reasons.append("invalid_duration_rollback")
    action = "clamp" if reasons else ""
    return start, end, action, ",".join(reasons)
