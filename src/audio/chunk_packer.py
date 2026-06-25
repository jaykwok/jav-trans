from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from boundary.planner import (
    BoundarySequenceFeatureProvider,
    BoundaryPlannerConfig,
    PlannedChunk,
    PlannedIsland,
    plan_boundary_chunks,
)
from boundary.refiner import BoundaryDecision, SequenceBoundaryRefiner
from boundary.base import SpeechSegment


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
    internal_gap_count: int = 0
    internal_gap_max_s: float = 0.0
    boundary_score: float | None = None
    boundary_reason: str = ""
    boundary_source: str = ""
    boundary_start_refine_delta_s: float | None = None
    boundary_end_refine_delta_s: float | None = None
    boundary_decision_source: str = ""
    scorer_speech_mean: float | None = None
    scorer_speech_max: float | None = None
    scorer_speech_p90: float | None = None
    scorer_split_mean: float | None = None
    scorer_split_max: float | None = None
    scorer_split_p90: float | None = None
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
    frame_hop_s: float = 1.0 / 29.97,
    sequence_boundary_refiner: SequenceBoundaryRefiner | None = None,
    sequence_feature_provider: BoundarySequenceFeatureProvider | None = None,
    sequence_batch_size: int = 256,
) -> list[PackedChunk]:
    """Convert scorer-produced speech islands into ASR chunks.

    Scorer v5 is the only island splitter. The planner no longer creates
    duration-driven splits or secondary boundary candidates; Boundary Refiner v6
    may only adjust the start/end of already-planned island chunks.
    """

    planner_config = BoundaryPlannerConfig(
        frame_hop_s=frame_hop_s,
        sequence_batch_size=sequence_batch_size,
    )
    planned = plan_boundary_chunks(
        segments,
        config=planner_config,
        sequence_refiner=sequence_boundary_refiner,
        sequence_feature_provider=sequence_feature_provider,
    )
    return _materialize_packed_chunks(planned, layout=PackingLayoutConfig())


def _materialize_packed_chunks(
    planned: Sequence[PlannedChunk],
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
    islands: Sequence[PlannedIsland],
    *,
    layout: PackingLayoutConfig,
    previous_decision: BoundaryDecision | None,
    previous_core_end: float | None,
    next_core_start: float | None,
    split_reason: str,
    boundary_decision: BoundaryDecision | None,
) -> PackedChunk:
    core_start = islands[0].start
    core_end = islands[-1].end
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
    core_start, core_end = _apply_boundary_delta(
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


def _internal_gap_count(islands: Sequence[PlannedIsland]) -> int:
    return sum(
        1
        for previous, current in zip(islands, islands[1:])
        if current.start > previous.end
    )


def _internal_gap_max_s(islands: Sequence[PlannedIsland]) -> float:
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
) -> tuple[float, float]:
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

    left_limit = 0.0 if previous_core_end is None else float(previous_core_end)
    right_limit = float("inf") if next_core_start is None else float(next_core_start)
    start = max(left_limit, min(start, max(left_limit, right_limit - min_duration)))
    end = min(right_limit, max(end, start + min_duration))
    if end < start:
        end = start
    return start, end
