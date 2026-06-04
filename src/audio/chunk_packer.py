from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from boundary.features import make_feature_bundle
from boundary.planner import BoundaryPlannerConfig, PlannedChunk, PlannedIsland, plan_boundary_chunks
from boundary.refiner import BoundaryDecision, BoundaryRefiner
from boundary.base import SpeechSegment


@dataclass(frozen=True)
class PackedChunk:
    start: float
    end: float
    speech_segments: list[SpeechSegment]
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
    boundary_score: float | None = None
    boundary_reason: str = ""
    boundary_source: str = ""


@dataclass(frozen=True)
class PackingLayoutConfig:
    max_chunk_s: float = 30.0
    target_padding_s: float = 2.0


def pack_speech_segments(
    segments: Sequence[SpeechSegment],
    *,
    frame_hop_s: float = 1.0 / 29.97,
    max_chunk_s: float = 30.0,
    target_chunk_s: float = 9.0,
    min_chunk_s: float = 0.4,
    target_padding_s: float = 2.0,
    start_weight: float = 1.5,
    frame_scores: Sequence[float] | None = None,
    score_frame_hop_s: float | None = None,
    cut_frame_scores: Sequence[float] | None = None,
    boundary_refiner: BoundaryRefiner | None = None,
    max_splits_per_segment: int = 16,
) -> list[PackedChunk]:
    """Convert Boundary Planner output into padded ASR chunks.

    Candidate extraction, refiner scoring, and constrained planning live under
    ``src/boundary``. This module only preserves the ASR-facing PackedChunk
    contract and applies dynamic padding around planner cores.
    """

    score_hop = score_frame_hop_s if score_frame_hop_s is not None else frame_hop_s
    features = make_feature_bundle(
        frame_hop_s=score_hop,
        speech_scores=frame_scores,
        cut_scores=cut_frame_scores,
    )
    planner_config = BoundaryPlannerConfig(
        frame_hop_s=frame_hop_s,
        max_chunk_s=max_chunk_s,
        target_chunk_s=target_chunk_s,
        min_chunk_s=min_chunk_s,
        start_weight=start_weight,
        target_padding_s=target_padding_s,
        max_splits_per_segment=max_splits_per_segment,
    )
    planned = plan_boundary_chunks(
        segments,
        features=features,
        config=planner_config,
        refiner=boundary_refiner,
    )
    layout = PackingLayoutConfig(
        max_chunk_s=max_chunk_s,
        target_padding_s=target_padding_s,
    )
    return _materialize_packed_chunks(planned, layout=layout)


def _materialize_packed_chunks(
    planned: Sequence[PlannedChunk],
    *,
    layout: PackingLayoutConfig,
) -> list[PackedChunk]:
    chunks: list[PackedChunk] = []
    for index, item in enumerate(planned):
        next_start = planned[index + 1].islands[0].start if index + 1 < len(planned) else None
        previous_end = chunks[-1].speech_segments[-1].end if chunks else None
        chunks.append(
            _make_chunk(
                item.islands,
                layout=layout,
                previous_end=previous_end,
                next_start=next_start,
                split_reason=item.split_reason,
                boundary_decision=item.boundary_decision,
            )
        )
    return chunks


def _make_chunk(
    islands: Sequence[PlannedIsland],
    *,
    layout: PackingLayoutConfig,
    previous_end: float | None,
    next_start: float | None,
    split_reason: str,
    boundary_decision: BoundaryDecision | None,
) -> PackedChunk:
    core_start = islands[0].start
    core_end = islands[-1].end
    core_duration = max(0.0, core_end - core_start)
    remaining = max(0.0, layout.max_chunk_s - core_duration)
    target = min(layout.target_padding_s, remaining / 2.0)

    if islands[0].split_left:
        left_limit = layout.target_padding_s
    else:
        left_gap = core_start if previous_end is None else max(0.0, core_start - previous_end)
        left_limit = left_gap / 2.0 if previous_end is not None else left_gap

    if islands[-1].split_right:
        right_limit = layout.target_padding_s
    else:
        right_gap = layout.target_padding_s * 2.0
        if next_start is not None:
            right_gap = max(0.0, next_start - core_end)
        right_limit = right_gap / 2.0 if next_start is not None else right_gap

    left_padding = min(target, left_limit)
    right_padding = min(target, right_limit)

    unused = max(0.0, remaining - left_padding - right_padding)
    if unused > 0.0:
        extra_right = min(
            layout.target_padding_s - right_padding,
            right_limit - right_padding,
            unused,
        )
        if extra_right > 0.0:
            right_padding += extra_right
            unused -= extra_right
        extra_left = min(
            layout.target_padding_s - left_padding,
            left_limit - left_padding,
            unused,
        )
        if extra_left > 0.0:
            left_padding += extra_left

    if any(island.split_left or island.split_right for island in islands):
        split_reason = "boundary_candidate" if split_reason == "tail" else split_reason

    boundary_scores = [
        float(island.boundary_score)
        for island in islands
        if island.boundary_score is not None
    ]
    boundary_reasons = [island.boundary_reason for island in islands if island.boundary_reason]
    boundary_sources = [island.boundary_source for island in islands if island.boundary_source]

    start = max(0.0, core_start - left_padding)
    end = core_end + right_padding
    return PackedChunk(
        start=start,
        end=end,
        speech_segments=[island.to_speech_segment() for island in islands],
        duration=end - start,
        left_padding_s=left_padding,
        right_padding_s=right_padding,
        split_reason=split_reason,
        core_start=core_start,
        core_end=core_end,
        internal_gap_count=_internal_gap_count(islands),
        internal_gap_max_s=_internal_gap_max_s(islands),
        boundary_score=(
            boundary_decision.score
            if boundary_decision is not None
            else (max(boundary_scores) if boundary_scores else None)
        ),
        boundary_reason=(
            boundary_decision.reason
            if boundary_decision is not None
            else ",".join(sorted(set(boundary_reasons)))
        ),
        boundary_source=",".join(sorted(set(boundary_sources))),
    )


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
