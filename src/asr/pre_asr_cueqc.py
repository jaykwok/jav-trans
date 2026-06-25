from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from asr.backends.qwen import (
    DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO,
    checkpoint_path_for_repo_env,
    current_qwen_asr_backend,
    validate_checkpoint_repo_id,
)
from boundary.sequence_features import (
    CHUNK_POOLED_PTM_SCHEMA,
    DEFAULT_CHUNK_POOLED_PTM_BINS,
    FrameSequenceFeatureConfig,
    chunk_pooled_ptm_feature_names,
)


PRE_ASR_CUEQC_SCHEMA = "cueqc_pre_asr_mamba_v9_binary"
PRE_ASR_CUEQC_MODEL_ARCH = "cueqc_pre_asr_mamba_v9"
PRE_ASR_CUEQC_DECISION_VERSION = "pre_asr_cueqc_v9_binary_v1"
PRE_ASR_CUEQC_FEATURE_SCHEMA = "pre_asr_cueqc_features_v5"
PRE_ASR_CUEQC_RUNTIME_ADAPTER = "pre_asr_planned_island_sequence_v1"
PRE_ASR_CUEQC_IGNORE_LABEL = -100
PRE_ASR_CUEQC_PTM_DIM = FrameSequenceFeatureConfig().max_ptm_dims
PRE_ASR_CUEQC_PTM_BINS = 8
PRE_ASR_CUEQC_DEFAULT_DROP_THRESHOLD = 0.95

PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES = (
    "duration_s",
    "raw_duration_s",
    "refined_duration_s",
    "prev_gap_s",
    "next_gap_s",
    "speech_segment_count",
    "internal_gap_count",
    "internal_gap_max_s",
    "boundary_score",
    "start_refine_delta_s",
    "end_refine_delta_s",
    "trim_left_s",
    "trim_right_s",
    "trim_total_s",
    "trim_ratio",
    "core_duration_ratio",
    "abs_start_refine_delta_s",
    "abs_end_refine_delta_s",
    "scorer_speech_mean",
    "scorer_speech_max",
    "scorer_speech_p90",
    "scorer_speech_p10",
    "scorer_speech_p50",
    "scorer_speech_std",
    "scorer_speech_active_ratio_05",
    "scorer_speech_active_ratio_07",
    "scorer_speech_active_ratio_09",
    "scorer_split_mean",
    "scorer_split_max",
    "scorer_split_p90",
    "scorer_split_std",
    "split_peak_count",
    "split_peak_density",
    "split_peak_top1",
    "split_peak_top2",
    "split_peak_top1_prominence",
    "split_peak_top2_prominence",
    "subtitle_min_duration_s",
    "below_subtitle_min_duration",
    "micro_chunk_candidate",
    "micro_action_none",
    "micro_action_preserve",
    "micro_action_merge_left",
    "micro_action_merge_right",
    "micro_action_unknown",
    "left_split_score",
    "right_split_score",
    "left_split_prominence",
    "right_split_prominence",
    "left_split_speech_valley",
    "right_split_speech_valley",
    "position_in_planned_island",
    "relative_position_in_planned_island",
    "num_chunks_in_planned_island",
    "is_first_chunk",
    "is_last_chunk",
    "prev_chunk_exists",
    "next_chunk_exists",
    "prev_duration_s",
    "next_duration_s",
    "prev_scorer_speech_mean",
    "prev_scorer_speech_p90",
    "prev_scorer_speech_active_ratio_05",
    "next_scorer_speech_mean",
    "next_scorer_speech_p90",
    "next_scorer_speech_active_ratio_05",
    "prev_below_subtitle_min_duration",
    "next_below_subtitle_min_duration",
    "prev_micro_chunk_candidate",
    "next_micro_chunk_candidate",
)
PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES = tuple(
    chunk_pooled_ptm_feature_names(
        ptm_dim=PRE_ASR_CUEQC_PTM_DIM,
        bins=PRE_ASR_CUEQC_PTM_BINS,
    )
)
PRE_ASR_CUEQC_PTM_BIN_FEATURE_NAMES = tuple(
    f"ptm_bin{bin_index:02d}_{dim_index:04d}"
    for bin_index in range(PRE_ASR_CUEQC_PTM_BINS)
    for dim_index in range(PRE_ASR_CUEQC_PTM_DIM)
)
PRE_ASR_CUEQC_FEATURE_NAMES = (
    PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES + PRE_ASR_CUEQC_PTM_BIN_FEATURE_NAMES
)
_BANNED_FEATURE_TOKENS = (
    "text",
    "raw_text",
    "token",
    "decoder",
    "asr_confidence",
    "subtitle_timing",
)


def _env_bool(name: str, default: str) -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw in {"1", "true", "yes", "on", "enabled"}


def enabled() -> bool:
    return _env_bool("PRE_ASR_CUEQC_ENABLED", "0")


def _checkpoint_path(repo_id: str | None = None) -> str:
    return checkpoint_path_for_repo_env(
        repo_id=repo_id or current_qwen_asr_backend(),
        mapping_env="PRE_ASR_CUEQC_MODEL_PATH_BY_REPO",
        default_mapping=DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO,
    )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _packed_value(chunk: Any, name: str, default: Any = None) -> Any:
    if hasattr(chunk, name):
        return getattr(chunk, name)
    if isinstance(chunk, Mapping):
        return chunk.get(name, default)
    return default


def _gap(spans: Sequence[Any], index: int, *, left: bool) -> float:
    current_start = _safe_float(_packed_value(spans[index], "start"))
    current_end = _safe_float(_packed_value(spans[index], "end"), current_start)
    if left:
        if index <= 0:
            return 5.0
        prev_end = _safe_float(_packed_value(spans[index - 1], "end"), current_start)
        return max(0.0, current_start - prev_end)
    if index + 1 >= len(spans):
        return 5.0
    next_start = _safe_float(_packed_value(spans[index + 1], "start"), current_end)
    return max(0.0, next_start - current_end)


def _sequence_count(value: Any) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return float(len(value))
    return 0.0


def _has_action(value: Any, token: str) -> float:
    actions = {item.strip() for item in str(value or "").split(",") if item.strip()}
    return 1.0 if token in actions else 0.0


def _bool_feature(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _cut_candidates(span: Any) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for key in ("primary_cut_candidates", "weak_cut_candidates"):
        raw = _packed_value(span, key, [])
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            continue
        for item in raw:
            if isinstance(item, Mapping):
                values.append(dict(item))
    return values


def _top_split_values(span: Any) -> tuple[float, float, float, float, int]:
    candidates = _cut_candidates(span)
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for item in candidates:
        score = _safe_float(item.get("score"))
        strength = _safe_float(item.get("strength"), score)
        scored.append((strength, score, item))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top1 = scored[0][1] if scored else 0.0
    top2 = scored[1][1] if len(scored) > 1 else 0.0
    prom1 = _safe_float(scored[0][2].get("prominence")) if scored else 0.0
    prom2 = _safe_float(scored[1][2].get("prominence")) if len(scored) > 1 else 0.0
    return top1, top2, prom1, prom2, len(scored)


def _planned_island_key(span: Any, index: int) -> str:
    for key in ("planned_island_id", "parent_chunk_id", "island_id"):
        value = _packed_value(span, key)
        if value is not None and str(value) != "":
            return str(value)
    return "sequence"


def _planned_island_position(spans: Sequence[Any], index: int) -> tuple[int, int]:
    key = _planned_island_key(spans[index], index)
    indexes = [
        item_index
        for item_index, span in enumerate(spans)
        if _planned_island_key(span, item_index) == key
    ]
    try:
        position = indexes.index(index)
    except ValueError:
        position = 0
    return position, max(1, len(indexes))


def _span_duration(span: Any) -> float:
    start = _safe_float(_packed_value(span, "start"))
    end = max(start, _safe_float(_packed_value(span, "end"), start))
    return max(0.0, end - start)


def _pooled_ptm_values(
    span: Any,
    *,
    require_ptm_pooling: bool,
) -> tuple[list[float], bool, str, int | None, int | None]:
    raw_values = _packed_value(span, "pre_asr_ptm_pooled_features", [])
    schema = str(_packed_value(span, "pre_asr_ptm_pooling_schema", "") or "")
    bins = _packed_value(span, "pre_asr_ptm_pooling_bins")
    dim = _packed_value(span, "pre_asr_ptm_pooling_dim")
    try:
        parsed_bins = None if bins is None else int(bins)
    except (TypeError, ValueError):
        parsed_bins = None
    try:
        parsed_dim = None if dim is None else int(dim)
    except (TypeError, ValueError):
        parsed_dim = None
    values: list[float] = []
    if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes, bytearray)):
        for item in raw_values:
            values.append(_safe_float(item))
    expected_dim = len(PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES)
    available = (
        schema == CHUNK_POOLED_PTM_SCHEMA
        and len(values) == expected_dim
        and (parsed_bins is None or parsed_bins == PRE_ASR_CUEQC_PTM_BINS)
        and (parsed_dim is None or parsed_dim == expected_dim)
    )
    if available:
        return values, True, schema, parsed_bins or PRE_ASR_CUEQC_PTM_BINS, expected_dim
    if require_ptm_pooling:
        raise ValueError("Pre-ASR CueQC v9 requires chunk-level pooled PTM features")
    return [0.0] * expected_dim, False, schema, parsed_bins, parsed_dim


def ptm_bin_matrix(candidate: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_pooled = candidate.get("pre_asr_ptm_pooled_features")
    pooled: list[float] = []
    if isinstance(raw_pooled, Sequence) and not isinstance(raw_pooled, (str, bytes, bytearray)):
        pooled = [_safe_float(item) for item in raw_pooled]
    expected = len(PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES)
    if len(pooled) != expected:
        pooled = [0.0] * expected
    dim = PRE_ASR_CUEQC_PTM_DIM
    offset = dim * 2
    values = np.asarray(pooled[offset : offset + PRE_ASR_CUEQC_PTM_BINS * dim], dtype=np.float32)
    if values.size != PRE_ASR_CUEQC_PTM_BINS * dim:
        values = np.zeros((PRE_ASR_CUEQC_PTM_BINS * dim,), dtype=np.float32)
    matrix = values.reshape(PRE_ASR_CUEQC_PTM_BINS, dim)
    available = bool(candidate.get("ptm_pooling_available"))
    mask = np.ones((PRE_ASR_CUEQC_PTM_BINS,), dtype=np.float32) if available else np.zeros(
        (PRE_ASR_CUEQC_PTM_BINS,), dtype=np.float32
    )
    return matrix, mask


def _neighbor_features(spans: Sequence[Any], index: int, offset: int) -> dict[str, float]:
    other = index + offset
    if other < 0 or other >= len(spans):
        prefix = "prev" if offset < 0 else "next"
        return {
            f"{prefix}_chunk_exists": 0.0,
            f"{prefix}_duration_s": 0.0,
            f"{prefix}_scorer_speech_mean": 0.0,
            f"{prefix}_scorer_speech_p90": 0.0,
            f"{prefix}_scorer_speech_active_ratio_05": 0.0,
            f"{prefix}_below_subtitle_min_duration": 0.0,
            f"{prefix}_micro_chunk_candidate": 0.0,
        }
    prefix = "prev" if offset < 0 else "next"
    span = spans[other]
    return {
        f"{prefix}_chunk_exists": 1.0,
        f"{prefix}_duration_s": _span_duration(span),
        f"{prefix}_scorer_speech_mean": _safe_float(_packed_value(span, "scorer_speech_mean")),
        f"{prefix}_scorer_speech_p90": _safe_float(_packed_value(span, "scorer_speech_p90")),
        f"{prefix}_scorer_speech_active_ratio_05": _safe_float(
            _packed_value(span, "scorer_speech_active_ratio_05")
        ),
        f"{prefix}_below_subtitle_min_duration": _bool_feature(
            _packed_value(span, "below_subtitle_min_duration", False)
        ),
        f"{prefix}_micro_chunk_candidate": _bool_feature(
            _packed_value(span, "micro_chunk_candidate", False)
        ),
    }


def candidate_from_span(
    spans: Sequence[Any],
    index: int,
    *,
    require_ptm_pooling: bool = False,
) -> dict[str, Any]:
    span = spans[index]
    start = _safe_float(_packed_value(span, "start"))
    end = max(start, _safe_float(_packed_value(span, "end"), start))
    duration = max(0.0, end - start)
    pooled_values, pooled_available, pooling_schema, pooling_bins, pooling_dim = _pooled_ptm_values(
        span,
        require_ptm_pooling=require_ptm_pooling,
    )
    start_delta = _safe_float(_packed_value(span, "boundary_start_refine_delta_s"))
    end_delta = _safe_float(_packed_value(span, "boundary_end_refine_delta_s"))
    trim_left = max(0.0, start_delta)
    trim_right = max(0.0, -end_delta)
    trim_total = trim_left + trim_right
    raw_duration = duration + trim_total
    top1, top2, prom1, prom2, split_peak_count = _top_split_values(span)
    position, group_count = _planned_island_position(spans, index)
    micro_action = str(_packed_value(span, "micro_resolve_action", "") or "")
    micro_none = 1.0 if not micro_action else 0.0
    known_action = max(
        _has_action(micro_action, "preserve_micro_candidate"),
        _has_action(micro_action, "preserve_edge_micro_candidate"),
        _has_action(micro_action, "merge_micro_into_left"),
        _has_action(micro_action, "merge_micro_into_right"),
    )
    neighbor_left = _neighbor_features(spans, index, -1)
    neighbor_right = _neighbor_features(spans, index, 1)
    features = {
        "duration_s": duration,
        "raw_duration_s": raw_duration,
        "refined_duration_s": duration,
        "prev_gap_s": _gap(spans, index, left=True),
        "next_gap_s": _gap(spans, index, left=False),
        "speech_segment_count": _sequence_count(_packed_value(span, "speech_segments", [])),
        "internal_gap_count": _safe_float(_packed_value(span, "internal_gap_count")),
        "internal_gap_max_s": _safe_float(_packed_value(span, "internal_gap_max_s")),
        "boundary_score": _safe_float(_packed_value(span, "boundary_score")),
        "start_refine_delta_s": start_delta,
        "end_refine_delta_s": end_delta,
        "trim_left_s": trim_left,
        "trim_right_s": trim_right,
        "trim_total_s": trim_total,
        "trim_ratio": trim_total / max(raw_duration, 1e-6),
        "core_duration_ratio": duration / max(raw_duration, 1e-6),
        "abs_start_refine_delta_s": abs(start_delta),
        "abs_end_refine_delta_s": abs(end_delta),
        "scorer_speech_mean": _safe_float(_packed_value(span, "scorer_speech_mean")),
        "scorer_speech_max": _safe_float(_packed_value(span, "scorer_speech_max")),
        "scorer_speech_p90": _safe_float(_packed_value(span, "scorer_speech_p90")),
        "scorer_speech_p10": _safe_float(_packed_value(span, "scorer_speech_p10")),
        "scorer_speech_p50": _safe_float(_packed_value(span, "scorer_speech_p50")),
        "scorer_speech_std": _safe_float(_packed_value(span, "scorer_speech_std")),
        "scorer_speech_active_ratio_05": _safe_float(
            _packed_value(span, "scorer_speech_active_ratio_05")
        ),
        "scorer_speech_active_ratio_07": _safe_float(
            _packed_value(span, "scorer_speech_active_ratio_07")
        ),
        "scorer_speech_active_ratio_09": _safe_float(
            _packed_value(span, "scorer_speech_active_ratio_09")
        ),
        "scorer_split_mean": _safe_float(_packed_value(span, "scorer_split_mean")),
        "scorer_split_max": _safe_float(_packed_value(span, "scorer_split_max")),
        "scorer_split_p90": _safe_float(_packed_value(span, "scorer_split_p90")),
        "scorer_split_std": _safe_float(_packed_value(span, "scorer_split_std")),
        "split_peak_count": float(split_peak_count),
        "split_peak_density": float(split_peak_count) / max(duration, 1e-6),
        "split_peak_top1": top1,
        "split_peak_top2": top2,
        "split_peak_top1_prominence": prom1,
        "split_peak_top2_prominence": prom2,
        "subtitle_min_duration_s": _safe_float(_packed_value(span, "subtitle_min_duration_s")),
        "below_subtitle_min_duration": _bool_feature(
            _packed_value(span, "below_subtitle_min_duration", False)
        ),
        "micro_chunk_candidate": _bool_feature(_packed_value(span, "micro_chunk_candidate", False)),
        "micro_action_none": micro_none,
        "micro_action_preserve": max(
            _has_action(micro_action, "preserve_micro_candidate"),
            _has_action(micro_action, "preserve_edge_micro_candidate"),
        ),
        "micro_action_merge_left": _has_action(micro_action, "merge_micro_into_left"),
        "micro_action_merge_right": _has_action(micro_action, "merge_micro_into_right"),
        "micro_action_unknown": 1.0 if micro_action and not known_action else 0.0,
        "left_split_score": _safe_float(_packed_value(span, "left_split_score")),
        "right_split_score": _safe_float(_packed_value(span, "right_split_score")),
        "left_split_prominence": _safe_float(_packed_value(span, "left_split_prominence")),
        "right_split_prominence": _safe_float(_packed_value(span, "right_split_prominence")),
        "left_split_speech_valley": _safe_float(_packed_value(span, "left_split_speech_valley")),
        "right_split_speech_valley": _safe_float(_packed_value(span, "right_split_speech_valley")),
        "position_in_planned_island": float(position),
        "relative_position_in_planned_island": (
            0.0 if group_count <= 1 else float(position) / float(group_count - 1)
        ),
        "num_chunks_in_planned_island": float(group_count),
        "is_first_chunk": 1.0 if position == 0 else 0.0,
        "is_last_chunk": 1.0 if position + 1 >= group_count else 0.0,
        **neighbor_left,
        **neighbor_right,
    }
    return {
        "schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
        "index": index,
        "start": round(start, 6),
        "end": round(end, 6),
        "planned_island_id": _planned_island_key(span, index),
        "position_in_planned_island": position,
        "num_chunks_in_planned_island": group_count,
        "subtitle_min_duration_s": round(features["subtitle_min_duration_s"], 6),
        "below_subtitle_min_duration": bool(features["below_subtitle_min_duration"]),
        "micro_chunk_candidate": bool(features["micro_chunk_candidate"]),
        "micro_resolve_action": micro_action,
        "micro_resolve_reason": str(_packed_value(span, "micro_resolve_reason", "") or ""),
        "left_split_score": features["left_split_score"],
        "right_split_score": features["right_split_score"],
        "features": features,
        "scalar_feature_names": list(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),
        "ptm_bin_feature_names": list(PRE_ASR_CUEQC_PTM_BIN_FEATURE_NAMES),
        "feature_names": list(PRE_ASR_CUEQC_FEATURE_NAMES),
        "ptm_pooling_schema": pooling_schema,
        "ptm_pooling_available": pooled_available,
        "ptm_pooling_bins": pooling_bins,
        "ptm_pooling_dim": pooling_dim,
        "pre_asr_ptm_pooled_features": pooled_values,
    }


def scalar_vector(candidate: Mapping[str, Any]) -> np.ndarray:
    features = candidate.get("features") if isinstance(candidate.get("features"), Mapping) else {}
    return np.asarray(
        [_safe_float(features.get(name)) for name in PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES],
        dtype=np.float32,
    )


def feature_vector(candidate: Mapping[str, Any]) -> np.ndarray:
    bins, _mask = ptm_bin_matrix(candidate)
    return np.concatenate([scalar_vector(candidate), bins.reshape(-1)], axis=0).astype(np.float32)


def planned_island_sequences(
    candidates: Sequence[Mapping[str, Any]],
) -> list[list[tuple[int, Mapping[str, Any]]]]:
    groups: list[list[tuple[int, Mapping[str, Any]]]] = []
    by_key: dict[tuple[str, str], int] = {}
    for order, candidate in enumerate(candidates):
        audio_id = str(candidate.get("audio_id") or candidate.get("video_id") or "")
        key = (audio_id, str(candidate.get("planned_island_id") or "sequence"))
        group_index = by_key.get(key)
        if group_index is None:
            group_index = len(groups)
            by_key[key] = group_index
            groups.append([])
        groups[group_index].append((order, candidate))
    return groups


def sequence_tensors(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray | list[tuple[int, int]]]:
    groups = planned_island_sequences(candidates)
    batch = len(groups)
    max_chunks = max((len(group) for group in groups), default=0)
    scalar = np.zeros(
        (batch, max_chunks, len(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES)),
        dtype=np.float32,
    )
    bins = np.zeros(
        (batch, max_chunks, PRE_ASR_CUEQC_PTM_BINS, PRE_ASR_CUEQC_PTM_DIM),
        dtype=np.float32,
    )
    bin_mask = np.zeros((batch, max_chunks, PRE_ASR_CUEQC_PTM_BINS), dtype=np.float32)
    chunk_mask = np.zeros((batch, max_chunks), dtype=np.float32)
    positions: list[tuple[int, int]] = [(-1, -1)] * len(candidates)
    for group_index, group in enumerate(groups):
        for chunk_index, (original_index, candidate) in enumerate(group):
            scalar[group_index, chunk_index] = scalar_vector(candidate)
            matrix, mask = ptm_bin_matrix(candidate)
            bins[group_index, chunk_index] = matrix
            bin_mask[group_index, chunk_index] = mask
            chunk_mask[group_index, chunk_index] = 1.0
            positions[original_index] = (group_index, chunk_index)
    return {
        "scalar_features": scalar,
        "ptm_bins": bins,
        "bin_mask": bin_mask,
        "chunk_mask": chunk_mask,
        "positions": positions,
    }


class PreAsrCueQCMambaV9:
    def __new__(cls, *args: Any, **kwargs: Any):
        import torch
        from torch import nn

        class _Model(nn.Module):
            def __init__(
                self,
                *,
                ptm_dim: int,
                scalar_dim: int,
                hidden_size: int = 128,
                bin_mamba_layers: int = 1,
                chunk_mamba_layers: int = 1,
                state_size: int = 32,
                num_heads: int = 4,
                head_dim: int = 64,
                n_groups: int = 2,
                chunk_size: int = 8,
                dropout: float = 0.1,
                num_classes: int = 2,
            ) -> None:
                super().__init__()
                from boundary.backbones import TinyMamba2BoundaryBackbone

                self.arch = PRE_ASR_CUEQC_MODEL_ARCH
                self.bin_backbone = TinyMamba2BoundaryBackbone(
                    input_dim=ptm_dim,
                    hidden_size=hidden_size,
                    num_layers=bin_mamba_layers,
                    state_size=state_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    n_groups=n_groups,
                    chunk_size=chunk_size,
                    bidirectional=True,
                )
                self.scalar_encoder = nn.Sequential(
                    nn.LayerNorm(scalar_dim),
                    nn.Linear(scalar_dim, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                )
                self.chunk_fuse = nn.Sequential(
                    nn.LayerNorm(self.bin_backbone.output_dim * 2 + hidden_size),
                    nn.Linear(self.bin_backbone.output_dim * 2 + hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.chunk_backbone = TinyMamba2BoundaryBackbone(
                    input_dim=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=chunk_mamba_layers,
                    state_size=state_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    n_groups=n_groups,
                    chunk_size=chunk_size,
                    bidirectional=True,
                )
                self.classifier = nn.Sequential(
                    nn.LayerNorm(self.chunk_backbone.output_dim),
                    nn.Linear(self.chunk_backbone.output_dim, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, num_classes),
                )

            @staticmethod
            def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
                if mask is None:
                    return x.mean(dim=1)
                m = mask.unsqueeze(-1).to(dtype=x.dtype)
                denom = m.sum(dim=1).clamp_min(1.0)
                return (x * m).sum(dim=1) / denom

            @staticmethod
            def _masked_max_pool(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
                if mask is None:
                    return x.max(dim=1).values
                valid = mask.unsqueeze(-1).bool()
                filled = x.masked_fill(~valid, -1.0e4)
                values = filled.max(dim=1).values
                any_valid = valid.any(dim=1)
                return torch.where(any_valid, values, torch.zeros_like(values))

            def forward(
                self,
                ptm_bins: torch.Tensor,
                scalar_features: torch.Tensor,
                chunk_mask: torch.Tensor | None = None,
                bin_mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
                if ptm_bins.ndim != 4:
                    raise ValueError("ptm_bins must have shape [batch, chunks, bins, dim]")
                if scalar_features.ndim != 3:
                    raise ValueError("scalar_features must have shape [batch, chunks, dim]")
                batch, chunks, bins, dim = ptm_bins.shape
                x = ptm_bins.reshape(batch * chunks, bins, dim)
                bm = None if bin_mask is None else bin_mask.reshape(batch * chunks, bins).long()
                bin_h = self.bin_backbone(x, attention_mask=bm)
                mean_pool = self._masked_mean_pool(bin_h, bm)
                max_pool = self._masked_max_pool(bin_h, bm)
                bin_repr = torch.cat([mean_pool, max_pool], dim=-1)
                scalar = scalar_features.reshape(batch * chunks, -1)
                scalar_repr = self.scalar_encoder(scalar)
                chunk_repr = self.chunk_fuse(torch.cat([bin_repr, scalar_repr], dim=-1))
                chunk_repr = chunk_repr.reshape(batch, chunks, -1)
                cm = None if chunk_mask is None else chunk_mask.long()
                ctx = self.chunk_backbone(chunk_repr, attention_mask=cm)
                logits = self.classifier(ctx)
                if chunk_mask is not None:
                    logits = logits * chunk_mask.unsqueeze(-1).to(dtype=logits.dtype)
                return logits

        return _Model(*args, **kwargs)


def make_model_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(config or {})
    return {
        "ptm_dim": int(raw.get("ptm_dim") or PRE_ASR_CUEQC_PTM_DIM),
        "scalar_dim": int(raw.get("scalar_dim") or len(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES)),
        "hidden_size": int(raw.get("hidden_size") or 128),
        "bin_mamba_layers": int(raw.get("bin_mamba_layers") or 1),
        "chunk_mamba_layers": int(raw.get("chunk_mamba_layers") or 1),
        "state_size": int(raw.get("state_size") or 32),
        "num_heads": int(raw.get("num_heads") or 4),
        "head_dim": int(raw.get("head_dim") or 64),
        "n_groups": int(raw.get("n_groups") or 2),
        "chunk_size": int(raw.get("chunk_size") or 8),
        "dropout": float(raw.get("dropout", 0.1)),
        "num_classes": int(raw.get("num_classes") or 2),
    }


class PreAsrCueQC:
    def __init__(
        self,
        *,
        checkpoint: Mapping[str, Any],
        path: Path,
        device: str = "auto",
        expected_asr_repo_id: str | None = None,
    ) -> None:
        import torch

        schema = str(checkpoint.get("schema") or "")
        if schema != PRE_ASR_CUEQC_SCHEMA:
            raise ValueError(f"unsupported Pre-ASR CueQC schema: {schema!r}")
        arch = str(checkpoint.get("arch") or "")
        if arch != PRE_ASR_CUEQC_MODEL_ARCH:
            raise ValueError(f"unsupported Pre-ASR CueQC arch: {arch!r}")
        if str(checkpoint.get("feature_schema") or "") != PRE_ASR_CUEQC_FEATURE_SCHEMA:
            raise ValueError("Pre-ASR CueQC feature_schema does not match runtime")
        if str(checkpoint.get("runtime_adapter") or "") != PRE_ASR_CUEQC_RUNTIME_ADAPTER:
            raise ValueError("Pre-ASR CueQC runtime_adapter does not match runtime")
        metadata = dict(checkpoint.get("metadata") or {})
        if expected_asr_repo_id is not None:
            validate_checkpoint_repo_id(
                str(metadata.get("asr_repo_id") or ""),
                expected_asr_repo_id,
                checkpoint_kind="Pre-ASR CueQC",
                metadata_key="metadata.asr_repo_id",
            )
        feature_names = tuple(str(item) for item in checkpoint.get("feature_names") or ())
        if feature_names != PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES:
            raise ValueError("Pre-ASR CueQC scalar feature_names do not match runtime schema")
        lowered = " ".join(PRE_ASR_CUEQC_FEATURE_NAMES).lower()
        if any(token in lowered for token in _BANNED_FEATURE_TOKENS):
            raise ValueError("Pre-ASR CueQC feature schema must not contain ASR text/token/decoder fields")

        config = make_model_config(checkpoint.get("model_config"))
        if config["ptm_dim"] != PRE_ASR_CUEQC_PTM_DIM:
            raise ValueError("Pre-ASR CueQC model_config.ptm_dim does not match runtime")
        if config["scalar_dim"] != len(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES):
            raise ValueError("Pre-ASR CueQC model_config.scalar_dim does not match runtime")
        self.model = PreAsrCueQCMambaV9(**config)
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Pre-ASR CueQC checkpoint missing model_state_dict")
        self.model.load_state_dict(state)

        normalized = (device or "auto").strip().lower()
        if normalized == "auto":
            normalized = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(normalized)
        self.model.to(self.device)
        self.model.eval()
        self.path = path
        self.sha256 = _file_sha256(path)
        self.metadata = metadata
        self.config = config
        self.mean = np.asarray(
            checkpoint.get("feature_mean", [0.0] * len(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES)),
            dtype=np.float32,
        )
        self.std = np.asarray(
            checkpoint.get("feature_std", [1.0] * len(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES)),
            dtype=np.float32,
        )
        if self.mean.shape[0] != len(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES) or self.std.shape[0] != len(
            PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES
        ):
            raise ValueError("Pre-ASR CueQC scalar normalization shape mismatch")
        decision = dict(checkpoint.get("decision_config") or {})
        self.drop_threshold = float(
            decision.get("drop_threshold", PRE_ASR_CUEQC_DEFAULT_DROP_THRESHOLD)
        )
        self.hard_keep_veto_enabled = bool(decision.get("hard_keep_veto", True))
        self.hard_drop_rule_enabled = bool(decision.get("hard_drop_rule", True))
        self.keep_veto_enabled = bool(decision.get("keep_veto", True))
        self.hard_keep_min_duration_s = float(decision.get("hard_keep_min_duration_s", 0.80))
        self.high_speech_p90 = float(decision.get("high_speech_p90", 0.85))
        self.high_active_ratio = float(decision.get("high_active_ratio", 0.50))
        self.very_low_speech_p90 = float(decision.get("very_low_speech_p90", 0.05))
        self.very_low_active_ratio = float(decision.get("very_low_active_ratio", 0.05))

    def signature(self) -> dict[str, Any]:
        return {
            "schema": PRE_ASR_CUEQC_SCHEMA,
            "arch": PRE_ASR_CUEQC_MODEL_ARCH,
            "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
            "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
            "path": str(self.path),
            "sha256": self.sha256,
            "drop_threshold": self.drop_threshold,
            "metadata": self.metadata,
        }

    @staticmethod
    def _feature(candidate: Mapping[str, Any], name: str) -> float:
        features = candidate.get("features") if isinstance(candidate.get("features"), Mapping) else {}
        return _safe_float(features.get(name))

    def _hard_keep_veto(self, candidate: Mapping[str, Any]) -> str:
        if not self.hard_keep_veto_enabled:
            return ""
        duration = self._feature(candidate, "refined_duration_s") or self._feature(candidate, "duration_s")
        if duration >= self.hard_keep_min_duration_s:
            return "duration_at_or_above_hard_keep_min"
        speech_p90 = self._feature(candidate, "scorer_speech_p90")
        active05 = self._feature(candidate, "scorer_speech_active_ratio_05")
        if speech_p90 >= self.high_speech_p90 and active05 >= self.high_active_ratio:
            return "high_stable_speech"
        if bool(candidate.get("micro_chunk_candidate")):
            left_score = self._feature(candidate, "left_split_score")
            right_score = self._feature(candidate, "right_split_score")
            if left_score >= 0.75 and right_score >= 0.75:
                return "strong_micro_split_evidence"
        prev_p90 = self._feature(candidate, "prev_scorer_speech_p90")
        next_p90 = self._feature(candidate, "next_scorer_speech_p90")
        if prev_p90 >= 0.70 and next_p90 >= 0.70:
            return "between_strong_speech_neighbors"
        return ""

    def _hard_drop_rule(self, candidate: Mapping[str, Any]) -> str:
        if not self.hard_drop_rule_enabled:
            return ""
        duration = self._feature(candidate, "refined_duration_s") or self._feature(candidate, "duration_s")
        speech_p90 = self._feature(candidate, "scorer_speech_p90")
        active05 = self._feature(candidate, "scorer_speech_active_ratio_05")
        if (
            duration < 0.12
            and speech_p90 < self.very_low_speech_p90
            and active05 < self.very_low_active_ratio
            and not bool(candidate.get("micro_chunk_candidate"))
        ):
            return "very_short_very_low_speech"
        return ""

    def _keep_veto(self, candidate: Mapping[str, Any]) -> str:
        if not self.keep_veto_enabled:
            return ""
        speech_p90 = self._feature(candidate, "scorer_speech_p90")
        active05 = self._feature(candidate, "scorer_speech_active_ratio_05")
        if speech_p90 >= self.high_speech_p90 and active05 >= self.high_active_ratio:
            return "high_stable_speech"
        duration = self._feature(candidate, "refined_duration_s") or self._feature(candidate, "duration_s")
        subtitle_min = self._feature(candidate, "subtitle_min_duration_s")
        speech_mean = self._feature(candidate, "scorer_speech_mean")
        if subtitle_min > 0.0 and duration >= subtitle_min and speech_mean >= 0.20:
            return "display_duration_with_speech"
        return ""

    def decide(self, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        import torch

        if not candidates:
            return []
        tensors = sequence_tensors(candidates)
        scalar = np.asarray(tensors["scalar_features"], dtype=np.float32)
        scalar = (scalar - self.mean.reshape(1, 1, -1)) / np.maximum(
            self.std.reshape(1, 1, -1),
            1e-6,
        )
        scalar = np.where(np.isfinite(scalar), scalar, 0.0).astype(np.float32)
        ptm_bins = np.asarray(tensors["ptm_bins"], dtype=np.float32)
        bin_mask = np.asarray(tensors["bin_mask"], dtype=np.float32)
        chunk_mask = np.asarray(tensors["chunk_mask"], dtype=np.float32)
        with torch.inference_mode():
            logits = self.model(
                torch.from_numpy(ptm_bins).to(self.device),
                torch.from_numpy(scalar).to(self.device),
                chunk_mask=torch.from_numpy(chunk_mask).to(self.device),
                bin_mask=torch.from_numpy(bin_mask).to(self.device),
            )
            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
        positions = list(tensors["positions"])
        decisions: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            group_index, chunk_index = positions[index]
            if group_index < 0 or chunk_index < 0:
                continue
            prob = probs[group_index, chunk_index]
            p_drop = float(prob[0])
            p_keep = float(prob[1])
            hard_keep = self._hard_keep_veto(candidate)
            hard_drop = "" if hard_keep else self._hard_drop_rule(candidate)
            keep_veto = "" if hard_keep or hard_drop else self._keep_veto(candidate)
            if hard_keep:
                drop = False
                reason = "hard_keep_veto"
            elif hard_drop:
                drop = True
                reason = "hard_drop_rule"
            elif p_drop >= self.drop_threshold and not keep_veto:
                drop = True
                reason = "model_drop_threshold"
            else:
                drop = False
                reason = "model_keep_default" if not keep_veto else "keep_veto"
            decisions.append(
                {
                    "schema": "pre_asr_cueqc_decision_v1",
                    "decision_version": PRE_ASR_CUEQC_DECISION_VERSION,
                    "model_schema": PRE_ASR_CUEQC_SCHEMA,
                    "model_arch": PRE_ASR_CUEQC_MODEL_ARCH,
                    "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
                    "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
                    "index": int(candidate.get("index", index)),
                    "route": "drop_before_asr" if drop else "keep_for_asr",
                    "confidence": round(p_drop if drop else p_keep, 4),
                    "prob_drop": round(p_drop, 4),
                    "prob_keep": round(p_keep, 4),
                    "drop_threshold": round(self.drop_threshold, 4),
                    "reason": reason,
                    "veto_reason": keep_veto or hard_keep,
                    "hard_rule_reason": hard_drop,
                }
            )
        return decisions


def load_checkpoint(
    path: str | Path,
    *,
    expected_asr_repo_id: str | None = None,
    device: str = "auto",
) -> PreAsrCueQC:
    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pre-ASR CueQC checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Pre-ASR CueQC checkpoint must be a mapping")
    return PreAsrCueQC(
        checkpoint=payload,
        path=checkpoint_path,
        device=device,
        expected_asr_repo_id=expected_asr_repo_id,
    )


def runtime_signature() -> dict[str, Any]:
    if not enabled():
        return {
            "schema": PRE_ASR_CUEQC_SCHEMA,
            "arch": PRE_ASR_CUEQC_MODEL_ARCH,
            "enabled": False,
            "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
            "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
        }
    checkpoint = _checkpoint_path()
    return {
        "schema": PRE_ASR_CUEQC_SCHEMA,
        "arch": PRE_ASR_CUEQC_MODEL_ARCH,
        "enabled": True,
        "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
        "model_path": checkpoint,
        "drop_threshold": os.getenv("PRE_ASR_CUEQC_DROP_THRESHOLD", ""),
    }


def load_active(*, expected_asr_repo_id: str | None = None) -> PreAsrCueQC:
    path = _checkpoint_path(expected_asr_repo_id)
    device = os.getenv("PRE_ASR_CUEQC_DEVICE", "auto").strip() or "auto"
    model = load_checkpoint(path, expected_asr_repo_id=expected_asr_repo_id, device=device)
    override = os.getenv("PRE_ASR_CUEQC_DROP_THRESHOLD", "").strip()
    if override:
        model.drop_threshold = float(override)
    return model
