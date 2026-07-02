from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

from audio.chunk_packer import PackedChunk
from boundary.base import SpeechSegment
from boundary.cut_refiner import CutEdgeRefiner
from boundary.outer_refiner import OuterEdgeRefiner
from boundary.sequence_features import FrameSequenceFeatureProvider
from boundary.split_model import SemanticSplitVerifier, SplitDecision


@dataclass(frozen=True)
class SemanticBoundaryConfig:
    short_core_max_s: float = 6.0
    short_core_cut_threshold: float = 0.90
    normal_cut_threshold: float = 0.75
    min_chunk_after_split_s: float = 1.2


def build_semantic_boundary_chunks(
    segments: Sequence[SpeechSegment],
    *,
    duration_s: float,
    speech_probabilities: Sequence[float],
    feature_provider: FrameSequenceFeatureProvider,
    outer_refiner: OuterEdgeRefiner,
    split_verifier: SemanticSplitVerifier,
    cut_refiner: CutEdgeRefiner,
    config: SemanticBoundaryConfig = SemanticBoundaryConfig(),
    split_audit_records: list[dict] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> list[PackedChunk]:
    def progress(label: str, current: int, total: int) -> None:
        if on_stage is not None:
            on_stage(f"{label} {current}/{total}")

    speech = np.asarray(speech_probabilities, dtype=np.float32)
    progress("外边界精修", 0, 1)
    refined = _refine_outer_edges(
        segments,
        duration_s=duration_s,
        speech=speech,
        provider=feature_provider,
        refiner=outer_refiner,
    )
    progress("外边界精修", 1, 1)

    prepared: list[tuple] = []
    split_total = max(1, len(refined))
    progress("语义切分判断", 0, split_total)
    for segment, core_start, core_end, outer_prediction in refined:
        proposals = [
            dict(candidate)
            for candidate in segment.weak_cut_candidates
            if core_start < float(candidate["time_s"]) < core_end
        ]
        decisions, frame_features, scalar_features = _split_decisions(
            proposals,
            core_start=core_start,
            core_end=core_end,
            speech=speech,
            provider=feature_provider,
            verifier=split_verifier,
        )
        accepted = _accepted_proposals(
            proposals,
            decisions,
            core_start=core_start,
            core_end=core_end,
            config=config,
        )
        if split_audit_records is not None:
            accepted_times = {
                round(float(candidate["time_s"]), 6)
                for candidate, _decision in accepted
            }
            for candidate, decision, frames, scalars in zip(
                proposals,
                decisions,
                frame_features,
                scalar_features,
            ):
                split_audit_records.append(
                    {
                        "candidate": dict(candidate),
                        "core_start": core_start,
                        "core_end": core_end,
                        "accepted": round(float(candidate["time_s"]), 6)
                        in accepted_times,
                        "label": decision.label,
                        "p_cut": decision.p_cut,
                        "p_continue": decision.p_continue,
                        "p_unsure": decision.p_unsure,
                        "frame_features": frames,
                        "scalar_features": scalars,
                    }
                )
        prepared.append(
            (
                segment,
                core_start,
                core_end,
                outer_prediction,
                proposals,
                decisions,
                accepted,
            )
        )
        progress("语义切分判断", len(prepared), split_total)
    if not refined:
        progress("语义切分判断", 1, 1)

    chunks: list[PackedChunk] = []
    cut_total = max(1, len(prepared))
    progress("内部切点精修", 0, cut_total)
    for index, (
        segment,
        core_start,
        core_end,
        outer_prediction,
        proposals,
        decisions,
        accepted,
    ) in enumerate(prepared, start=1):
        cuts = _refine_cuts(
            accepted,
            core_start=core_start,
            core_end=core_end,
            speech=speech,
            provider=feature_provider,
            refiner=cut_refiner,
            min_chunk_s=config.min_chunk_after_split_s,
        )
        chunks.extend(
            _materialize_chunks(
                segment,
                raw_start=float(segment.start),
                raw_end=float(segment.end),
                core_start=core_start,
                core_end=core_end,
                cuts=cuts,
                proposals=proposals,
                decisions=decisions,
                outer_prediction=outer_prediction,
            )
        )
        progress("内部切点精修", index, cut_total)
    if not prepared:
        progress("内部切点精修", 1, 1)
    return chunks


def _refine_outer_edges(
    segments,
    *,
    duration_s,
    speech,
    provider,
    refiner,
):
    feature_rows: list[np.ndarray] = []
    scalar_rows: list[np.ndarray] = []
    for segment in segments:
        frames, scalars = provider.features_for_outer_island(
            start_s=segment.start,
            end_s=segment.end,
            speech_probabilities=speech,
            context_s=float(refiner.feature_config["context_s"]),
            ptm_dim=int(refiner.feature_config["ptm_dim"]),
        )
        feature_rows.append(frames)
        scalar_rows.append(scalars)
    predictions = refiner.predict(
        frame_features=np.stack(feature_rows),
        scalar_features=np.stack(scalar_rows),
    )
    result = []
    for segment, prediction in zip(segments, predictions):
        start = float(segment.start)
        end = float(segment.end)
        if prediction.start_confidence >= 0.4:
            start += prediction.start_delta_s
        if prediction.end_confidence >= 0.4:
            end += prediction.end_delta_s
        start = max(0.0, min(start, duration_s))
        end = max(start, min(end, duration_s))
        result.append((segment, start, end, prediction))
    return result


def _split_decisions(
    proposals,
    *,
    core_start,
    core_end,
    speech,
    provider,
    verifier,
) -> tuple[list[SplitDecision], np.ndarray, np.ndarray]:
    if not proposals:
        return [], np.empty((0, 0, 0), dtype=np.float32), np.empty(
            (0, 0), dtype=np.float32
        )
    feature_rows: list[np.ndarray] = []
    scalar_rows: list[np.ndarray] = []
    cfg = verifier.feature_config
    for candidate in proposals:
        frames, scalars = provider.features_for_split_candidate(
            core_start_s=core_start,
            core_end_s=core_end,
            candidate=candidate,
            speech_probabilities=speech,
            left_context_s=float(cfg["left_context_s"]),
            right_context_s=float(cfg["right_context_s"]),
            gap_context_s=float(cfg["gap_context_s"]),
            left_bins=int(cfg["left_bins"]),
            gap_bins=int(cfg["gap_bins"]),
            right_bins=int(cfg["right_bins"]),
            ptm_dim=int(cfg["ptm_dim"]),
        )
        feature_rows.append(frames)
        scalar_rows.append(scalars)
    frame_array = np.stack(feature_rows)
    scalar_array = np.stack(scalar_rows)
    return (
        verifier.decide(
            frame_features=frame_array,
            scalar_features=scalar_array,
        ),
        frame_array,
        scalar_array,
    )


def _accepted_proposals(
    proposals,
    decisions,
    *,
    core_start,
    core_end,
    config,
):
    threshold = (
        config.short_core_cut_threshold
        if core_end - core_start <= config.short_core_max_s
        else config.normal_cut_threshold
    )
    accepted = [
        (candidate, decision)
        for candidate, decision in zip(proposals, decisions)
        if decision.label == "cut" and decision.p_cut >= threshold
    ]
    accepted.sort(key=lambda item: item[1].p_cut, reverse=True)
    selected = []
    for candidate, decision in accepted:
        time_s = float(candidate["time_s"])
        anchors = [core_start, core_end, *(float(item[0]["time_s"]) for item in selected)]
        if min(abs(time_s - anchor) for anchor in anchors) >= config.min_chunk_after_split_s:
            selected.append((candidate, decision))
    return sorted(selected, key=lambda item: float(item[0]["time_s"]))


def _refine_cuts(
    accepted,
    *,
    core_start,
    core_end,
    speech,
    provider,
    refiner,
    min_chunk_s,
):
    if not accepted:
        return []
    cfg = refiner.feature_config
    features: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    proposals: list[float] = []
    for candidate, _decision in accepted:
        frame_row, scalar_row = provider.features_for_split_candidate(
            core_start_s=core_start,
            core_end_s=core_end,
            candidate=candidate,
            speech_probabilities=speech,
            left_context_s=float(cfg["context_s"]),
            right_context_s=float(cfg["context_s"]),
            gap_context_s=float(cfg["gap_context_s"]),
            left_bins=int(cfg["bins"][0]),
            gap_bins=int(cfg["bins"][1]),
            right_bins=int(cfg["bins"][2]),
            ptm_dim=int(cfg["ptm_dim"]),
        )
        features.append(frame_row)
        scalars.append(scalar_row)
        proposals.append(float(candidate["time_s"]))
    refined = refiner.refine(
        proposal_times_s=np.asarray(proposals),
        frame_features=np.stack(features),
        scalar_features=np.stack(scalars),
        core_start_s=np.full(len(proposals), core_start),
        core_end_s=np.full(len(proposals), core_end),
    )
    result = []
    cursor = core_start
    for (candidate, decision), cut_time in zip(accepted, refined):
        cut = float(cut_time)
        if cut - cursor < min_chunk_s or core_end - cut < min_chunk_s:
            continue
        result.append(
            {
                **candidate,
                "kind": "primary",
                "proposal_time_s": float(candidate["time_s"]),
                "time_s": cut,
                "frame": int(round(cut / provider.frame_hop_s)),
                "label": decision.label,
                "p_cut": decision.p_cut,
                "p_continue": decision.p_continue,
                "p_unsure": decision.p_unsure,
                "shared_absolute_timestamp": True,
            }
        )
        cursor = cut
    return result


def _materialize_chunks(
    segment,
    *,
    raw_start,
    raw_end,
    core_start,
    core_end,
    cuts,
    proposals,
    decisions,
    outer_prediction,
):
    decision_by_time = {
        round(float(candidate["time_s"]), 6): decision
        for candidate, decision in zip(proposals, decisions)
    }
    accepted_proposals = {
        round(float(candidate["proposal_time_s"]), 6)
        for candidate in cuts
    }
    weak = []
    for candidate in proposals:
        key = round(float(candidate["time_s"]), 6)
        if key in accepted_proposals:
            continue
        decision = decision_by_time[key]
        weak.append(
            {
                **candidate,
                "kind": "weak",
                "label": decision.label,
                "p_cut": decision.p_cut,
                "p_continue": decision.p_continue,
                "p_unsure": decision.p_unsure,
            }
        )
    boundaries = [core_start, *(float(item["time_s"]) for item in cuts), core_end]
    chunks: list[PackedChunk] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        primary = [
            dict(item)
            for item in cuts
            if abs(float(item["time_s"]) - start) < 1e-6
            or abs(float(item["time_s"]) - end) < 1e-6
        ]
        piece = replace(
            segment,
            start=start,
            end=end,
            primary_cut_candidates=primary,
            weak_cut_candidates=[
                dict(item) for item in weak if start < float(item["time_s"]) < end
            ],
        )
        chunks.append(
            PackedChunk(
                start=start,
                end=end,
                duration=end - start,
                speech_segments=[piece],
                split_reason="semantic_split_model" if cuts else "speech_core",
                source_abs_start=start,
                source_abs_end=end,
                core_start=start,
                core_end=end,
                raw_start=raw_start if index == 0 else start,
                raw_end=raw_end if index + 1 == len(boundaries) - 1 else end,
                raw_duration=(
                    (raw_end if index + 1 == len(boundaries) - 1 else end)
                    - (raw_start if index == 0 else start)
                ),
                acoustic_start=start,
                acoustic_end=end,
                acoustic_duration=end - start,
                boundary_source="semantic_split_model" if primary else "outer_edge_refiner",
                boundary_start_refine_delta_s=(
                    outer_prediction.start_delta_s if index == 0 else None
                ),
                boundary_end_refine_delta_s=(
                    outer_prediction.end_delta_s
                    if index + 1 == len(boundaries) - 1
                    else None
                ),
                boundary_decision_source=(
                    "outer_edge_refiner_v1" if not primary else "shared_absolute_cut_v1"
                ),
                refiner_start_confidence=(
                    outer_prediction.start_confidence if index == 0 else None
                ),
                refiner_end_confidence=(
                    outer_prediction.end_confidence
                    if index + 1 == len(boundaries) - 1
                    else None
                ),
                primary_cut_candidates=primary,
                weak_cut_candidates=piece.weak_cut_candidates,
            )
        )
    return chunks
