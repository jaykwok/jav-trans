from __future__ import annotations

import os
import math
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

from audio.chunk_packer import PackedChunk
from boundary.base import SpeechSegment
from boundary.cut_refiner import CutEdgeRefiner
from boundary.outer_refiner import OuterEdgeRefiner
from boundary.outer_refiner_v2 import OuterEdgeRefinerV2, PairedOuterEdgePrediction
from boundary.sequence_features import (
    FrameSequenceFeatureProvider,
    ptm_projection_digest,
)
from boundary.split_model import (
    SemanticSplitIslandVerifier,
    SplitDecision,
)


@dataclass(frozen=True)
class SemanticBoundaryConfig:
    short_core_max_s: float = 6.0
    short_core_cut_threshold: float = 0.90
    normal_cut_threshold: float = 0.75
    min_chunk_after_split_s: float = 1.2
    duration_pressure_enabled: bool = False
    duration_pressure_log_median: float = 0.0
    duration_pressure_log_mad: float = 0.0
    duration_pressure_z: float = 0.0
    duration_pressure_floor: float = 0.50


def _decision_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def effective_semantic_config(
    split_verifier,
    config: SemanticBoundaryConfig,
) -> SemanticBoundaryConfig:
    """Prefer the calibrated decision_config embedded in a v2 checkpoint."""

    decision = getattr(split_verifier, "decision_config", None)
    if not isinstance(decision, dict) or not decision:
        return config
    pressure_from_config = bool(config.duration_pressure_enabled)
    return SemanticBoundaryConfig(
        short_core_max_s=float(
            decision.get("short_core_max_s", config.short_core_max_s)
        ),
        short_core_cut_threshold=float(
            decision.get("short_core_cut_threshold", config.short_core_cut_threshold)
        ),
        normal_cut_threshold=float(
            decision.get("normal_cut_threshold", config.normal_cut_threshold)
        ),
        min_chunk_after_split_s=float(
            decision.get("min_chunk_after_split_s", config.min_chunk_after_split_s)
        ),
        duration_pressure_enabled=pressure_from_config
        or _decision_bool(decision.get("duration_pressure_enabled"), False),
        duration_pressure_log_median=float(
            config.duration_pressure_log_median
            if pressure_from_config
            else decision.get(
                "duration_pressure_log_median", config.duration_pressure_log_median
            )
        ),
        duration_pressure_log_mad=float(
            config.duration_pressure_log_mad
            if pressure_from_config
            else decision.get(
                "duration_pressure_log_mad", config.duration_pressure_log_mad
            )
        ),
        duration_pressure_z=float(
            config.duration_pressure_z
            if pressure_from_config
            else decision.get("duration_pressure_z", config.duration_pressure_z)
        ),
        duration_pressure_floor=float(
            config.duration_pressure_floor
            if pressure_from_config
            else decision.get(
                "duration_pressure_floor", config.duration_pressure_floor
            )
        ),
    )


def semantic_config_payload(config: SemanticBoundaryConfig) -> dict[str, float | bool | str]:
    return {
        "decision_mode": "threshold",
        "short_core_max_s": float(config.short_core_max_s),
        "short_core_cut_threshold": float(config.short_core_cut_threshold),
        "normal_cut_threshold": float(config.normal_cut_threshold),
        "min_chunk_after_split_s": float(config.min_chunk_after_split_s),
        "duration_pressure_enabled": bool(config.duration_pressure_enabled),
        "duration_pressure_log_median": float(config.duration_pressure_log_median),
        "duration_pressure_log_mad": float(config.duration_pressure_log_mad),
        "duration_pressure_z": float(config.duration_pressure_z),
        "duration_pressure_floor": float(config.duration_pressure_floor),
    }


def build_semantic_boundary_chunks(
    segments: Sequence[SpeechSegment],
    *,
    duration_s: float,
    speech_probabilities: Sequence[float],
    feature_provider: FrameSequenceFeatureProvider,
    outer_refiner: OuterEdgeRefiner | OuterEdgeRefinerV2,
    split_verifier: SemanticSplitIslandVerifier,
    cut_refiner: CutEdgeRefiner,
    config: SemanticBoundaryConfig = SemanticBoundaryConfig(),
    split_audit_records: list[dict] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> list[PackedChunk]:
    def progress(label: str, current: int, total: int) -> None:
        interval = max(1, int(total) // 100)
        if on_stage is not None and (
            current <= 0
            or current >= total
            or current % interval == 0
        ):
            on_stage(f"{label} {current}/{total}")

    speech = np.asarray(speech_probabilities, dtype=np.float32)
    config = effective_semantic_config(split_verifier, config)
    progress("外边界精修", 0, 1)
    refined = _refine_outer_edges(
        segments,
        duration_s=duration_s,
        speech=speech,
        provider=feature_provider,
        refiner=outer_refiner,
    )
    progress("外边界精修", 1, 1)

    split_total = max(1, len(refined))
    progress("语义切分判断", 0, split_total)
    prepared: list[tuple] = []

    def append_audit_records(
        *,
        proposals,
        decisions,
        frame_features,
        scalar_features,
        core_start,
        core_end,
        accepted,
    ) -> None:
        if split_audit_records is None:
            return
        accepted_by_time = {
            round(float(candidate["time_s"]), 6): dict(candidate)
            for candidate, _decision in accepted
        }
        for candidate, decision, frames, scalars in zip(
            proposals,
            decisions,
            frame_features,
            scalar_features,
        ):
            accepted_candidate = accepted_by_time.get(round(float(candidate["time_s"]), 6))
            split_audit_records.append(
                {
                    "candidate": dict(candidate),
                    "core_start": core_start,
                    "core_end": core_end,
                    "accepted": accepted_candidate is not None,
                    "accepted_candidate": accepted_candidate or {},
                    "label": decision.label,
                    "p_cut": decision.p_cut,
                    "p_continue": decision.p_continue,
                    "p_unsure": decision.p_unsure,
                    "frame_features": frames,
                    "scalar_features": scalars,
                }
            )

    def finish_group(
        *,
        segment,
        core_start,
        core_end,
        outer_prediction,
        proposals,
        decisions,
        audit_chunks=(),
    ) -> None:
        accepted = _accepted_proposals(
            proposals,
            decisions,
            core_start=core_start,
            core_end=core_end,
            config=config,
        )
        for audit_proposals, audit_decisions, audit_frames, audit_scalars in audit_chunks:
            append_audit_records(
                proposals=audit_proposals,
                decisions=audit_decisions,
                frame_features=audit_frames,
                scalar_features=audit_scalars,
                core_start=core_start,
                core_end=core_end,
                accepted=accepted,
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

    if hasattr(split_verifier, "decide_islands"):
        cap = _semantic_split_island_candidate_cap()
        pending: list[tuple] = []
        pending_candidates = 0

        def flush_pending() -> None:
            nonlocal pending, pending_candidates
            if not pending:
                return
            decision_groups = _batched_island_split_decisions(
                split_verifier,
                frame_feature_groups=[item[5] for item in pending],
                scalar_feature_groups=[item[6] for item in pending],
            )
            for item, decisions in zip(pending, decision_groups):
                (
                    segment,
                    core_start,
                    core_end,
                    outer_prediction,
                    proposals,
                    frame_features,
                    scalar_features,
                ) = item
                audit_chunks = (
                    ((proposals, decisions, frame_features, scalar_features),)
                    if split_audit_records is not None
                    else ()
                )
                finish_group(
                    segment=segment,
                    core_start=core_start,
                    core_end=core_end,
                    outer_prediction=outer_prediction,
                    proposals=proposals,
                    decisions=decisions,
                    audit_chunks=audit_chunks,
                )
            pending = []
            pending_candidates = 0

        for segment, core_start, core_end, outer_prediction in refined:
            proposals = [
                dict(candidate)
                for candidate in segment.weak_cut_candidates
                if core_start < float(candidate["time_s"]) < core_end
            ]
            if not proposals:
                flush_pending()
                finish_group(
                    segment=segment,
                    core_start=core_start,
                    core_end=core_end,
                    outer_prediction=outer_prediction,
                    proposals=proposals,
                    decisions=[],
                )
                continue
            if len(proposals) > cap:
                flush_pending()
                decisions: list[SplitDecision] = []
                audit_chunks = []
                for start in range(0, len(proposals), cap):
                    slab = proposals[start : start + cap]
                    frame_features, scalar_features = _split_features(
                        slab,
                        core_start=core_start,
                        core_end=core_end,
                        speech=speech,
                        provider=feature_provider,
                        verifier=split_verifier,
                    )
                    slab_decisions = _batched_island_split_decisions(
                        split_verifier,
                        frame_feature_groups=[frame_features],
                        scalar_feature_groups=[scalar_features],
                    )[0]
                    decisions.extend(slab_decisions)
                    if split_audit_records is not None:
                        audit_chunks.append(
                            (slab, slab_decisions, frame_features, scalar_features)
                        )
                finish_group(
                    segment=segment,
                    core_start=core_start,
                    core_end=core_end,
                    outer_prediction=outer_prediction,
                    proposals=proposals,
                    decisions=decisions,
                    audit_chunks=audit_chunks,
                )
                continue
            if pending and pending_candidates + len(proposals) > cap:
                flush_pending()
            frame_features, scalar_features = _split_features(
                proposals,
                core_start=core_start,
                core_end=core_end,
                speech=speech,
                provider=feature_provider,
                verifier=split_verifier,
            )
            pending.append(
                (
                    segment,
                    core_start,
                    core_end,
                    outer_prediction,
                    proposals,
                    frame_features,
                    scalar_features,
                )
            )
            pending_candidates += len(proposals)
        flush_pending()
    else:
        raise TypeError("Semantic Split runtime requires a v2 island verifier")

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
    if isinstance(refiner, OuterEdgeRefinerV2):
        frame_feature_groups = [
            provider.features_for_outer_island_v2(
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
        return [
            (
                segment,
                max(0.0, min(float(prediction.start_s), duration_s)),
                max(
                    max(0.0, min(float(prediction.start_s), duration_s)),
                    min(float(prediction.end_s), duration_s),
                ),
                prediction,
            )
            for segment, prediction in zip(segments, predictions)
        ]

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


def _split_features(
    proposals,
    *,
    core_start,
    core_end,
    speech,
    provider,
    verifier,
) -> tuple[np.ndarray, np.ndarray]:
    if not proposals:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    feature_rows: list[np.ndarray] = []
    scalar_rows: list[np.ndarray] = []
    cfg = verifier.feature_config
    projection = cfg.get("ptm_projection") or None
    projection_mean = (
        np.asarray(projection["mean"], dtype=np.float32)
        if projection is not None
        else None
    )
    projection_components = (
        np.asarray(projection["components"], dtype=np.float32)
        if projection is not None
        else None
    )
    if projection is not None and provider.has_pre_projected_ptm:
        expected_digest = str(projection.get("digest") or "") or ptm_projection_digest(
            projection_mean, projection_components
        )
        if provider.ptm_projected_digest != expected_digest:
            raise ValueError(
                "semantic split ptm projection mismatch: runtime payload digest "
                f"{provider.ptm_projected_digest or '(empty)'} != checkpoint digest "
                f"{expected_digest}; point SPEECH_BOUNDARY_JA_SEQUENCE_PTM_PROJECTION "
                "at the projection npz this checkpoint was trained with"
            )
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
            extra_context_scales=tuple(cfg.get("extra_context_scales") or ()),
            ptm_projection_mean=projection_mean,
            ptm_projection_components=projection_components,
        )
        feature_rows.append(frames)
        scalar_rows.append(scalars)
    frame_array = np.stack(feature_rows)
    scalar_array = np.stack(scalar_rows)
    return frame_array, scalar_array


def _semantic_split_inference_batch_size() -> int:
    raw = os.getenv("SEMANTIC_SPLIT_INFERENCE_BATCH_SIZE", "auto").strip().lower()
    if raw in {"", "auto"}:
        return 128
    try:
        return max(1, int(raw))
    except ValueError:
        return 128


_MAX_ISLAND_BATCH_CANDIDATES = 256


def _semantic_split_island_candidate_cap() -> int:
    raw = os.getenv("SEMANTIC_SPLIT_ISLAND_MAX_CANDIDATES", "").strip()
    if not raw:
        return _MAX_ISLAND_BATCH_CANDIDATES
    try:
        return max(1, int(raw))
    except ValueError:
        return _MAX_ISLAND_BATCH_CANDIDATES


def _release_cuda_cache_for_verifier(verifier) -> None:
    if not str(getattr(verifier, "device", "")).startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        return


def _batched_island_split_decisions(
    verifier,
    *,
    frame_feature_groups: Sequence[np.ndarray],
    scalar_feature_groups: Sequence[np.ndarray],
) -> list[list[SplitDecision]]:
    """Batch whole islands; VRAM is bounded by both island count and candidates.

    A single island with more than ``_MAX_ISLAND_BATCH_CANDIDATES`` candidates
    is split into contiguous candidate slabs (each run as its own forward) and
    the per-candidate decisions are merged back in order; otherwise the
    island-sequence Mamba2 forward over an oversized island allocates an SSM
    states tensor that OOMs the GPU on multi-minute audio. The slab boundary
    costs the island encoder cross-slab context, which is acceptable for islands
    long enough to exceed the cap and is the only way to keep the forward
    bounded by candidate count rather than island length."""

    max_islands = _semantic_split_inference_batch_size()
    cap = _semantic_split_island_candidate_cap()

    # Expand oversized islands into contiguous candidate slabs; each slab keeps
    # its originating island index so decisions merge back in candidate order.
    slabs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for island_index, (frames, scalars) in enumerate(
        zip(frame_feature_groups, scalar_feature_groups)
    ):
        count = int(frames.shape[0])
        if count <= cap:
            slabs.append((island_index, frames, scalars))
            continue
        for start in range(0, count, cap):
            end = min(count, start + cap)
            slabs.append((island_index, frames[start:end], scalars[start:end]))

    slab_decisions: list[list[SplitDecision]] = [[] for _ in slabs]
    batch: list[tuple[int, np.ndarray, np.ndarray]] = []
    batch_max_count = 0

    def flush() -> None:
        nonlocal batch, batch_max_count
        if not batch:
            return
        slab_indexes = [item[0] for item in batch]
        decisions = verifier.decide_islands(
            island_frame_features=[item[1] for item in batch],
            island_scalar_features=[item[2] for item in batch],
        )
        for slab_index, slab_decision in zip(slab_indexes, decisions):
            slab_decisions[slab_index] = slab_decision
        _release_cuda_cache_for_verifier(verifier)
        batch = []
        batch_max_count = 0

    for slab_index, (_island_index, frames, scalars) in enumerate(slabs):
        count = int(frames.shape[0])
        padded_cost = max(batch_max_count, count) * (len(batch) + 1)
        if batch and (
            len(batch) >= max_islands or padded_cost > cap
        ):
            flush()
        batch.append((slab_index, frames, scalars))
        batch_max_count = max(batch_max_count, count)
    flush()

    results: list[list[SplitDecision]] = [[] for _ in frame_feature_groups]
    for slab_index, (island_index, _frames, _scalars) in enumerate(slabs):
        results[island_index].extend(slab_decisions[slab_index])
    return results


def _is_noise_bracket_pair(
    earlier_role: str,
    later_role: str,
) -> bool:
    """A speech_to_noise cut followed by noise_to_speech isolates one noise run."""

    return earlier_role == "speech_to_noise" and later_role == "noise_to_speech"


def _duration_pressure_score(duration_s: float, config: SemanticBoundaryConfig) -> float:
    scaled_log_mad = 1.4826 * float(config.duration_pressure_log_mad)
    if scaled_log_mad <= 0.0:
        raise ValueError("duration pressure log MAD must be positive when enabled")
    return (math.log(float(duration_s)) - float(config.duration_pressure_log_median)) / scaled_log_mad


def _duration_pressure_trigger_s(config: SemanticBoundaryConfig) -> float:
    return math.exp(
        float(config.duration_pressure_log_median)
        + float(config.duration_pressure_z) * 1.4826 * float(config.duration_pressure_log_mad)
    )


def _select_spaced_cut_candidates(
    candidates,
    *,
    core_start,
    core_end,
    min_chunk_after_split_s: float,
):
    accepted = list(candidates)
    accepted.sort(key=lambda item: item[1].p_cut, reverse=True)
    selected: list[tuple[dict, SplitDecision]] = []
    brackets: set[int] = set()
    for candidate, decision in accepted:
        candidate = dict(candidate)
        time_s = float(candidate["time_s"])
        edge_gap = min(abs(time_s - core_start), abs(time_s - core_end))
        if edge_gap < min_chunk_after_split_s:
            continue
        near = [
            index
            for index, (existing, _decision) in enumerate(selected)
            if abs(time_s - float(existing["time_s"]))
            < min_chunk_after_split_s
        ]
        if not near:
            selected.append((candidate, decision))
            continue
        # Min-spacing exemption: exactly one nearby accepted cut whose role
        # forms a speech_to_noise -> noise_to_speech bracket with this one, so
        # the short run in between becomes its own chunk for CueQC to drop.
        if len(near) != 1 or near[0] in brackets:
            continue
        partner_candidate, partner_decision = selected[near[0]]
        partner_time = float(partner_candidate["time_s"])
        first_role, second_role = (
            (partner_decision.role, decision.role)
            if partner_time < time_s
            else (decision.role, partner_decision.role)
        )
        if not _is_noise_bracket_pair(first_role, second_role):
            continue
        pair_start = min(partner_time, time_s)
        pair_end = max(partner_time, time_s)
        pair_id = f"noise-bracket-{pair_start:.6f}-{pair_end:.6f}"
        partner_candidate["bracket_pair_id"] = pair_id
        candidate["bracket_pair_id"] = pair_id
        brackets.add(near[0])
        brackets.add(len(selected))
        selected.append((candidate, decision))
    bracket_times = {
        round(float(selected[index][0]["time_s"]), 6) for index in brackets
    }
    result = []
    for candidate, decision in selected:
        entry = dict(candidate)
        if round(float(candidate["time_s"]), 6) in bracket_times:
            entry["noise_isolation_bracket"] = True
        result.append((entry, decision))
    return sorted(result, key=lambda item: float(item[0]["time_s"]))


def _add_duration_pressure_cuts(
    selected,
    proposals,
    decisions,
    *,
    core_start,
    core_end,
    config: SemanticBoundaryConfig,
):
    if not config.duration_pressure_enabled:
        return selected
    trigger_s = _duration_pressure_trigger_s(config)
    selected_by_time = {
        round(float(candidate["time_s"]), 6) for candidate, _decision in selected
    }
    floor_pool = [
        (dict(candidate), decision)
        for candidate, decision in zip(proposals, decisions)
        if decision.label == "cut"
        and decision.p_cut >= float(config.duration_pressure_floor)
        and round(float(candidate["time_s"]), 6) not in selected_by_time
    ]
    if not floor_pool:
        return selected
    result = list(selected)
    while True:
        added: list[tuple[dict, SplitDecision]] = []
        bounds = sorted([float(core_start), *(float(item[0]["time_s"]) for item in result), float(core_end)])
        for seg_start, seg_end in zip(bounds, bounds[1:]):
            duration_s = seg_end - seg_start
            if duration_s < trigger_s:
                continue
            score = _duration_pressure_score(duration_s, config)
            if score < float(config.duration_pressure_z):
                continue
            eligible = [
                (candidate, decision)
                for candidate, decision in floor_pool
                if round(float(candidate["time_s"]), 6) not in selected_by_time
                and seg_start + config.min_chunk_after_split_s
                <= float(candidate["time_s"])
                <= seg_end - config.min_chunk_after_split_s
            ]
            if not eligible:
                continue
            candidate, decision = max(eligible, key=lambda item: item[1].p_cut)
            entry = {
                **candidate,
                "duration_pressure_acceptance": True,
                "duration_pressure_score": score,
                "duration_pressure_trigger_s": trigger_s,
                "duration_pressure_floor": float(config.duration_pressure_floor),
            }
            added.append((entry, decision))
            selected_by_time.add(round(float(candidate["time_s"]), 6))
        if not added:
            break
        result.extend(added)
    return sorted(result, key=lambda item: float(item[0]["time_s"]))


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
    accepted = _select_spaced_cut_candidates(
        [
            (candidate, decision)
            for candidate, decision in zip(proposals, decisions)
            if decision.label == "cut" and decision.p_cut >= threshold
        ],
        core_start=core_start,
        core_end=core_end,
        min_chunk_after_split_s=config.min_chunk_after_split_s,
    )
    return _add_duration_pressure_cuts(
        accepted,
        proposals,
        decisions,
        core_start=core_start,
        core_end=core_end,
        config=config,
    )


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
    previous_bracket = False
    for (candidate, decision), cut_time in zip(accepted, refined):
        cut = float(cut_time)
        is_bracket = bool(candidate.get("noise_isolation_bracket"))
        spacing_exempt = is_bracket and previous_bracket and result
        if cut <= cursor:
            continue
        if (cut - cursor < min_chunk_s and not spacing_exempt) or (
            core_end - cut < min_chunk_s
        ):
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
                "role": decision.role,
                "p_role": decision.p_role,
                "shared_absolute_timestamp": True,
            }
        )
        cursor = cut
        previous_bracket = is_bracket
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
    if isinstance(outer_prediction, PairedOuterEdgePrediction):
        outer_decision_source = "outer_edge_refiner_v2"
        outer_start_confidence = float(
            outer_prediction.start_probabilities["semantic_target"]
        )
        outer_end_confidence = float(
            outer_prediction.end_probabilities["semantic_target"]
        )
    else:
        outer_decision_source = "outer_edge_refiner_v1"
        outer_start_confidence = float(outer_prediction.start_confidence)
        outer_end_confidence = float(outer_prediction.end_confidence)
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
                "role": decision.role,
                "p_role": decision.p_role,
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
                    outer_decision_source if not primary else "shared_absolute_cut_v1"
                ),
                refiner_start_confidence=(
                    outer_start_confidence if index == 0 else None
                ),
                refiner_end_confidence=(
                    outer_end_confidence
                    if index + 1 == len(boundaries) - 1
                    else None
                ),
                primary_cut_candidates=primary,
                weak_cut_candidates=piece.weak_cut_candidates,
            )
        )
    return chunks
