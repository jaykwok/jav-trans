from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

import numpy as np

from audio.chunk_packer import PackedChunk
from boundary.base import SpeechSegment
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.inner_refiner_v2 import InnerEdgeRefinerV2
from boundary.outer_refiner_v3 import OuterEdgeRefinerV3
from boundary.sequence_features import FrameSequenceFeatureProvider
from boundary.split_model import AcousticSplitV4Planner, aggregate_cut_event_runs


def build_acoustic_split_v4_provisional_chunks(
    segments: Sequence[SpeechSegment],
    *,
    duration_s: float,
    speech_probabilities: Sequence[float],
    feature_provider: FrameSequenceFeatureProvider,
    outer_refiner: OuterEdgeRefinerV3,
    split_planner: AcousticSplitV4Planner,
    on_stage: Callable[[str], None] | None = None,
) -> list[PackedChunk]:
    """Build current-contract provisional sub-islands with binary argmax events."""

    split_source = "acoustic_split_v4"
    split_adapter = str(split_planner.signature().get("runtime_adapter") or "")
    speech = np.asarray(speech_probabilities, dtype=np.float32)
    if on_stage is not None:
        on_stage("外边界精修 0/1")
    refined = _refine_outer_edges(
        segments,
        duration_s=duration_s,
        provider=feature_provider,
        refiner=outer_refiner,
    )
    if on_stage is not None:
        on_stage("外边界精修 1/1")

    prepared_inputs: list[tuple] = []
    split_input_indexes: list[int] = []
    split_frame_groups: list[np.ndarray] = []
    split_scalar_groups: list[np.ndarray] = []
    for segment, core_start, core_end, outer_prediction in refined:
        proposals = [
            dict(candidate)
            for candidate in segment.weak_cut_candidates
            if core_start < float(candidate["time_s"]) < core_end
        ]
        if proposals:
            frame_features, scalar_features = _split_features(
                proposals,
                core_start=core_start,
                core_end=core_end,
                speech=speech,
                provider=feature_provider,
                planner=split_planner,
            )
            split_input_indexes.append(len(prepared_inputs))
            split_frame_groups.append(frame_features)
            split_scalar_groups.append(scalar_features)
        prepared_inputs.append(
            (segment, core_start, core_end, outer_prediction, proposals)
        )

    decisions_by_island: list[list] = [[] for _item in prepared_inputs]
    if split_frame_groups:
        batched_decisions = split_planner.decide_islands(
            island_frame_features=split_frame_groups,
            island_scalar_features=split_scalar_groups,
        )
        for prepared_index, decisions in zip(
            split_input_indexes, batched_decisions, strict=True
        ):
            decisions_by_island[prepared_index] = decisions

    prepared: list[tuple] = []
    for island_index, (prepared_input, decisions) in enumerate(
        zip(prepared_inputs, decisions_by_island, strict=True)
    ):
        segment, core_start, core_end, outer_prediction, proposals = prepared_input
        events = aggregate_cut_event_runs(
            candidate_times_s=[float(candidate["time_s"]) for candidate in proposals],
            decisions=decisions,
            event_id_prefix=f"island-{island_index:04d}",
        )
        prepared.append(
            (
                segment,
                core_start,
                core_end,
                outer_prediction,
                proposals,
                decisions,
                events,
            )
        )

    chunks: list[PackedChunk] = []
    for island_index, prepared_island in enumerate(prepared):
        (
            segment,
            core_start,
            core_end,
            outer_prediction,
            proposals,
            decisions,
            events,
        ) = prepared_island
        event_rows = []
        for event in events:
            decision = decisions[event.representative_index]
            candidate = proposals[event.representative_index]
            event_rows.append(
                {
                    "event_id": event.event_id,
                    "time_s": event.representative_time_s,
                    "frame": int(candidate["frame"]),
                    "candidate_start_index": event.candidate_start_index,
                    "candidate_end_index": event.candidate_end_index,
                    "representative_index": event.representative_index,
                    "p_cut": decision.p_cut,
                    "p_continue": decision.p_continue,
                    "p_unsure": decision.p_unsure,
                    "score": float(candidate.get("score") or 0.0),
                    "prominence": float(candidate.get("prominence") or 0.0),
                    "speech_valley": float(candidate.get("speech_valley") or 0.0),
                }
            )
        event_by_time = {
            round(float(row["time_s"]), 6): row for row in event_rows
        }
        boundaries = [core_start, *(row["time_s"] for row in event_rows), core_end]
        representative_indexes = {
            event.representative_index for event in events
        }
        weak = [
            {
                **candidate,
                "kind": "weak",
                "label": decisions[index].label,
                "p_cut": decisions[index].p_cut,
                "p_continue": decisions[index].p_continue,
                "p_unsure": decisions[index].p_unsure,
            }
            for index, candidate in enumerate(proposals)
            if index not in representative_indexes
        ]
        for piece_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            adjacent = [
                dict(event_by_time[key])
                for key in (round(float(start), 6), round(float(end), 6))
                if key in event_by_time
            ]
            piece = replace(
                segment,
                start=float(start),
                end=float(end),
                primary_cut_candidates=adjacent,
                weak_cut_candidates=[
                    dict(item) for item in weak if start < float(item["time_s"]) < end
                ],
            )
            first_piece = piece_index == 0
            last_piece = piece_index + 1 == len(boundaries) - 1
            raw_start = float(segment.start) if first_piece else float(start)
            raw_end = float(segment.end) if last_piece else float(end)
            chunks.append(
                PackedChunk(
                    start=float(start),
                    end=float(end),
                    duration=float(end - start),
                    speech_segments=[piece],
                    split_reason=split_source if events else "speech_core",
                    source_abs_start=float(start),
                    source_abs_end=float(end),
                    parent_chunk_id=island_index,
                    island_id=piece_index,
                    island_count=len(boundaries) - 1,
                    core_start=float(start),
                    core_end=float(end),
                    raw_start=raw_start,
                    raw_end=raw_end,
                    raw_duration=raw_end - raw_start,
                    acoustic_start=float(start),
                    acoustic_end=float(end),
                    acoustic_duration=float(end - start),
                    display_start=float(start),
                    display_end=float(end),
                    display_duration=float(end - start),
                    boundary_contract_id=ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                    semantic_event_ids=[row["event_id"] for row in adjacent],
                    semantic_event_probabilities=[
                        {
                            "p_cut": float(row["p_cut"]),
                            "p_continue": float(row["p_continue"]),
                            "p_unsure": float(row["p_unsure"]),
                        }
                        for row in adjacent
                    ],
                    boundary_source=(
                        split_source if adjacent else "outer_edge_refiner_v3"
                    ),
                    boundary_decision_source=(
                        split_adapter if adjacent else "outer_edge_refiner_v3"
                    ),
                    boundary_start_refine_delta_s=(
                        outer_prediction.start_delta_s if first_piece else None
                    ),
                    boundary_end_refine_delta_s=(
                        outer_prediction.end_delta_s if last_piece else None
                    ),
                    primary_cut_candidates=adjacent,
                    weak_cut_candidates=piece.weak_cut_candidates,
                )
            )
    return chunks


def annotate_inner_edge_predictions(
    chunks: Sequence[PackedChunk],
    *,
    feature_provider: FrameSequenceFeatureProvider,
    inner_refiner: InnerEdgeRefinerV2,
) -> list[PackedChunk]:
    """Predict binary acoustic semantic-core spans before CueQC routing."""

    if not chunks:
        return []
    if any(
        not ACOUSTIC_BINARY_V12_CONTRACT.matches(chunk.boundary_contract_id)
        for chunk in chunks
    ):
        raise ValueError("Inner v2 received an unsupported Boundary contract")
    raw_ptm_dim = int(
        inner_refiner.feature_config.get("raw_ptm_dim")
        or inner_refiner.model_config.get("ptm_input_dim")
        or 0
    )
    feature_groups = [
        feature_provider.features_for_outer_island(
            start_s=float(chunk.start),
            end_s=float(chunk.end),
            raw_ptm_dim=raw_ptm_dim,
        )
        for chunk in chunks
    ]
    predictions = inner_refiner.predict_subislands(
        frame_feature_groups=feature_groups,
        raw_spans=[(float(chunk.start), float(chunk.end)) for chunk in chunks],
        frame_hop_s=float(feature_provider.frame_hop_s),
    )
    return [
        replace(
            chunk,
            inner_edge_prediction={
                "schema": ACOUSTIC_BINARY_V12_CONTRACT.inner_prediction_schema,
                "start_s": float(prediction.start_s),
                "end_s": float(prediction.end_s),
                "start_action": prediction.start_action,
                "end_action": prediction.end_action,
                "action": (
                    "drop"
                    if prediction.start_action == "drop"
                    and prediction.end_action == "drop"
                    else "refined"
                ),
                "abstain_reason": prediction.abstain_reason,
                "start_probabilities": dict(prediction.start_probabilities),
                "end_probabilities": dict(prediction.end_probabilities),
            },
        )
        for chunk, prediction in zip(chunks, predictions)
    ]


def apply_binary_inner_edges_after_cueqc(
    chunks: Sequence[PackedChunk],
) -> list[PackedChunk]:
    """Apply learned semantic-core spans before waveform extraction and ASR."""

    result: list[PackedChunk] = []
    for chunk in chunks:
        if not ACOUSTIC_BINARY_V12_CONTRACT.matches(chunk.boundary_contract_id):
            raise ValueError("binary Inner v2 apply received an unsupported contract")
        prediction = dict(chunk.inner_edge_prediction or {})
        if (
            prediction.get("schema")
            != ACOUSTIC_BINARY_V12_CONTRACT.inner_prediction_schema
        ):
            raise ValueError("binary Inner v2 apply requires current predictions")
        if prediction.get("action") == "drop":
            continue
        start = float(prediction["start_s"])
        end = float(prediction["end_s"])
        if start < float(chunk.start) or end > float(chunk.end) or end <= start:
            raise ValueError("binary Inner v2 emitted an invalid acoustic core span")
        removed = list(chunk.removed_gap_spans or [])
        if start > float(chunk.start):
            removed.append(
                {
                    "event_ids": list(chunk.semantic_event_ids or []),
                    "start": float(chunk.start),
                    "end": start,
                    "duration": start - float(chunk.start),
                }
            )
        if end < float(chunk.end):
            removed.append(
                {
                    "event_ids": list(chunk.semantic_event_ids or []),
                    "start": end,
                    "end": float(chunk.end),
                    "duration": float(chunk.end) - end,
                }
            )
        result.append(
            replace(
                chunk,
                start=start,
                end=end,
                duration=end - start,
                source_abs_start=start,
                source_abs_end=end,
                acoustic_start=start,
                acoustic_end=end,
                acoustic_duration=end - start,
                display_start=start,
                display_end=end,
                display_duration=end - start,
                paired_inner_edges={
                    "event_ids": list(chunk.semantic_event_ids or []),
                    "speech_start": start,
                    "speech_end": end,
                    "action": "binary_core",
                },
                removed_gap_spans=removed,
                removed_gap_duration_s=sum(
                    float(item["duration"]) for item in removed
                ),
            )
        )
    return result


def _refine_outer_edges(
    segments: Sequence[SpeechSegment],
    *,
    duration_s: float,
    provider: FrameSequenceFeatureProvider,
    refiner: OuterEdgeRefinerV3,
) -> list[tuple]:
    frame_feature_groups = [
        provider.features_for_outer_island(
            start_s=segment.start,
            end_s=segment.end,
            raw_ptm_dim=int(refiner.feature_config["raw_ptm_dim"]),
        )
        for segment in segments
    ]
    predictions = refiner.predict_islands(
        frame_feature_groups=frame_feature_groups,
        raw_spans=[(float(segment.start), float(segment.end)) for segment in segments],
        frame_hop_s=float(provider.frame_hop_s),
    )
    refined = []
    for segment, prediction in zip(segments, predictions, strict=True):
        if prediction.start_action == "drop" and prediction.end_action == "drop":
            continue
        start = max(0.0, min(float(prediction.start_s), duration_s))
        end = max(start, min(float(prediction.end_s), duration_s))
        refined.append((segment, start, end, prediction))
    return refined


def _split_features(
    proposals: Sequence[dict],
    *,
    core_start: float,
    core_end: float,
    speech: np.ndarray,
    provider: FrameSequenceFeatureProvider,
    planner: AcousticSplitV4Planner,
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    scalar_rows: list[np.ndarray] = []
    config = planner.feature_config
    if config.get("ptm_projection"):
        raise ValueError("Acoustic Split v4 must learn its PTM projection in-model")
    for candidate in proposals:
        frames, scalars = provider.features_for_split_candidate(
            core_start_s=core_start,
            core_end_s=core_end,
            candidate=candidate,
            speech_probabilities=speech,
            left_context_s=float(config["left_context_s"]),
            right_context_s=float(config["right_context_s"]),
            gap_context_s=float(config["gap_context_s"]),
            left_bins=int(config["left_bins"]),
            gap_bins=int(config["gap_bins"]),
            right_bins=int(config["right_bins"]),
            ptm_dim=int(config["ptm_dim"]),
            extra_context_scales=tuple(config.get("extra_context_scales") or ()),
            ptm_projection_mean=None,
            ptm_projection_components=None,
        )
        feature_rows.append(frames)
        scalar_rows.append(scalars)
    return np.stack(feature_rows), np.stack(scalar_rows)
