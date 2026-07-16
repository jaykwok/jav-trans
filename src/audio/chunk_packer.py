from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from boundary.base import SpeechSegment


@dataclass(frozen=True)
class PackedChunk:
    """Final pre-ASR chunk anchored to the source audio timeline."""

    start: float
    end: float
    speech_segments: list[SpeechSegment]
    duration: float
    split_reason: str
    source_abs_start: float | None = None
    source_abs_end: float | None = None
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
    display_start: float | None = None
    display_end: float | None = None
    display_duration: float | None = None
    boundary_pipeline_version: int | None = None
    semantic_event_ids: list[str] | None = None
    semantic_event_probabilities: list[dict[str, float]] | None = None
    inner_edge_prediction: dict[str, Any] | None = None
    paired_inner_edges: dict[str, Any] | None = None
    removed_gap_spans: list[dict[str, Any]] | None = None
    removed_gap_duration_s: float = 0.0
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
    pre_asr_ptm_projection_digest: str = ""
