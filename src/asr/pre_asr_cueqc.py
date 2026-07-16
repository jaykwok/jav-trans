from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from asr.backends.qwen import (
    DEFAULT_INNER_EDGE_REFINER_CHECKPOINT_BY_REPO,
    DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO,
    DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO,
    QWEN_ASR_06B_REPO_ID,
    checkpoint_path_for_repo_env,
    current_qwen_asr_backend,
    validate_checkpoint_repo_id,
)
from boundary.sequence_features import (
    CHUNK_LEARNED_PROJECTED_PTM_SCHEMA,
    CHUNK_PROJECTED_PTM_SCHEMA,
    CHUNK_POOLED_PTM_SCHEMA,
    chunk_pooled_ptm_feature_names,
)


PRE_ASR_CUEQC_SCHEMA = "cueqc_pre_asr_semantic_chunk_v13"
PRE_ASR_CUEQC_MODEL_ARCH = "cueqc_pre_asr_semantic_chunk_v13"
PRE_ASR_CUEQC_DECISION_VERSION = "pre_asr_cueqc_v13_argmax_v1"
PRE_ASR_CUEQC_FEATURE_SCHEMA = "pre_asr_cueqc_features_v10"
PRE_ASR_CUEQC_RUNTIME_ADAPTER = "pre_asr_provisional_subisland_sequence_v5"
PRE_ASR_CUEQC_ARTIFACT = {
    "name": "pre_asr_cueqc",
    "display_name": "Pre-ASR CueQC",
    "version": "v13",
    "pipeline_stage": 5,
    "pipeline_role": "provisional_subisland_keep_drop_unsure_routing",
}
PRE_ASR_CUEQC_LABELS = ("drop", "keep", "unsure")
PRE_ASR_CUEQC_LABEL_TO_INDEX = {
    label: index for index, label in enumerate(PRE_ASR_CUEQC_LABELS)
}
PRE_ASR_CUEQC_LEGACY_SCHEMA = "cueqc_pre_asr_semantic_chunk_v12_binary"
PRE_ASR_CUEQC_LEGACY_MODEL_ARCH = "cueqc_pre_asr_semantic_chunk_v12"
PRE_ASR_CUEQC_LEGACY_DECISION_VERSION = "pre_asr_cueqc_v12_binary_v1"
PRE_ASR_CUEQC_LEGACY_FEATURE_SCHEMA = "pre_asr_cueqc_features_v9"
PRE_ASR_CUEQC_LEGACY_RUNTIME_ADAPTER = "pre_asr_semantic_chunk_sequence_v4"
PRE_ASR_CUEQC_LEGACY_ARTIFACT = {
    "name": "pre_asr_cueqc",
    "display_name": "Pre-ASR CueQC",
    "version": "v12",
    "pipeline_stage": 5,
    "pipeline_role": "final_chunk_keep_drop_routing",
}
PRE_ASR_CUEQC_IGNORE_LABEL = -100
PRE_ASR_CUEQC_PTM_DIM = 128
PRE_ASR_CUEQC_PTM_BINS = 8
PRE_ASR_CUEQC_MODEL_PTM_TOKENS = PRE_ASR_CUEQC_PTM_BINS + 2
PRE_ASR_CUEQC_DEFAULT_DROP_THRESHOLD = 0.95

PRE_ASR_CUEQC_V8_SCALAR_FEATURE_NAMES = (
    "duration_s",
    "raw_start_s",
    "raw_end_s",
    "raw_duration_s",
    "acoustic_start_s",
    "acoustic_end_s",
    "acoustic_duration_s",
    "refined_duration_s",
    "raw_to_acoustic_start_shift_s",
    "raw_to_acoustic_end_shift_s",
    "prev_gap_s",
    "next_gap_s",
    "speech_segment_count",
    "internal_gap_count",
    "internal_gap_max_s",
    "boundary_score",
    "start_refine_delta_s",
    "end_refine_delta_s",
    "refiner_pred_start_delta_s",
    "refiner_pred_end_delta_s",
    "refiner_applied_start_delta_s",
    "refiner_applied_end_delta_s",
    "refiner_abs_pred_start_delta_s",
    "refiner_abs_pred_end_delta_s",
    "refiner_abs_applied_start_delta_s",
    "refiner_abs_applied_end_delta_s",
    "refiner_start_confidence",
    "refiner_end_confidence",
    "refiner_confidence_min",
    "refiner_confidence_mean",
    "refiner_effective_start_delta_max_s",
    "refiner_effective_end_delta_max_s",
    "refiner_start_source_model",
    "refiner_end_source_model",
    "refiner_start_source_model_scaled",
    "refiner_end_source_model_scaled",
    "refiner_start_source_noop_low_confidence",
    "refiner_end_source_noop_low_confidence",
    "refiner_start_source_acoustic_fallback",
    "refiner_end_source_acoustic_fallback",
    "refiner_start_source_unknown",
    "refiner_end_source_unknown",
    "refiner_safety_clamp",
    "refiner_safety_rollback",
    "refiner_fallback_used",
    "refiner_shared_boundary_adjusted",
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
    "primary_cut_count",
    "weak_cut_count",
    "primary_cut_density",
    "weak_cut_density",
    "semantic_p_cut_mean",
    "semantic_p_cut_max",
    "semantic_p_continue_mean",
    "semantic_p_unsure_mean",
    "semantic_unsure_ratio",
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
PRE_ASR_CUEQC_SPLIT_EDGE_ROLES = (
    "none",
    "speech_to_speech",
    "speech_to_noise",
    "noise_to_speech",
)
PRE_ASR_CUEQC_V9_SCALAR_FEATURE_NAMES = (
    "left_edge_is_split_cut",
    "right_edge_is_split_cut",
    "left_edge_is_island_edge",
    "right_edge_is_island_edge",
    "left_edge_noise_isolation_bracket",
    "right_edge_noise_isolation_bracket",
    "left_edge_p_cut",
    "right_edge_p_cut",
    "left_edge_p_continue",
    "right_edge_p_continue",
    "left_edge_p_unsure",
    "right_edge_p_unsure",
    "left_edge_p_role",
    "right_edge_p_role",
    "left_edge_role_none",
    "right_edge_role_none",
    "left_edge_role_speech_to_speech",
    "right_edge_role_speech_to_speech",
    "left_edge_role_speech_to_noise",
    "right_edge_role_speech_to_noise",
    "left_edge_role_noise_to_speech",
    "right_edge_role_noise_to_speech",
    "left_edge_role_unknown",
    "right_edge_role_unknown",
    "left_right_same_noise_pair",
    "left_right_split_edge_gap_s",
)
PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES = (
    PRE_ASR_CUEQC_V8_SCALAR_FEATURE_NAMES
    + PRE_ASR_CUEQC_V9_SCALAR_FEATURE_NAMES
)
PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES = tuple(
    chunk_pooled_ptm_feature_names(
        ptm_dim=PRE_ASR_CUEQC_PTM_DIM,
        bins=PRE_ASR_CUEQC_PTM_BINS,
    )
)
PRE_ASR_CUEQC_PTM_BIN_FEATURE_NAMES = tuple(
    f"ptm_{token_name}_{dim_index:04d}"
    for token_name in (
        "global_mean",
        "global_max",
        *(f"bin{bin_index:02d}" for bin_index in range(PRE_ASR_CUEQC_PTM_BINS)),
    )
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


def _semantic_split_checkpoint_path(repo_id: str | None = None) -> str:
    return checkpoint_path_for_repo_env(
        repo_id=repo_id or current_qwen_asr_backend(),
        mapping_env="SEMANTIC_SPLIT_MODEL_PATH_BY_REPO",
        default_mapping=DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO,
    )


def _inner_edge_refiner_checkpoint_path(repo_id: str | None = None) -> str:
    return checkpoint_path_for_repo_env(
        repo_id=repo_id or current_qwen_asr_backend(),
        mapping_env="INNER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        default_mapping=DEFAULT_INNER_EDGE_REFINER_CHECKPOINT_BY_REPO,
    )


def _runtime_contract(repo_id: str | None = None) -> dict[str, Any]:
    selected = repo_id or current_qwen_asr_backend()
    if selected == QWEN_ASR_06B_REPO_ID:
        return {
            "schema": PRE_ASR_CUEQC_LEGACY_SCHEMA,
            "arch": PRE_ASR_CUEQC_LEGACY_MODEL_ARCH,
            "decision_version": PRE_ASR_CUEQC_LEGACY_DECISION_VERSION,
            "feature_schema": PRE_ASR_CUEQC_LEGACY_FEATURE_SCHEMA,
            "runtime_adapter": PRE_ASR_CUEQC_LEGACY_RUNTIME_ADAPTER,
            "artifact": PRE_ASR_CUEQC_LEGACY_ARTIFACT,
            "num_classes": 2,
        }
    return {
        "schema": PRE_ASR_CUEQC_SCHEMA,
        "arch": PRE_ASR_CUEQC_MODEL_ARCH,
        "decision_version": PRE_ASR_CUEQC_DECISION_VERSION,
        "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
        "artifact": PRE_ASR_CUEQC_ARTIFACT,
        "num_classes": len(PRE_ASR_CUEQC_LABELS),
    }


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


def _source_feature(value: Any, token: str) -> float:
    return 1.0 if str(value or "").strip() == token else 0.0


def _source_unknown(value: Any, known: set[str]) -> float:
    raw = str(value or "").strip()
    return 1.0 if raw and raw not in known else 0.0


def _contains_token(value: Any, token: str) -> float:
    raw = str(value or "")
    return 1.0 if token in {item.strip() for item in raw.split(",") if item.strip()} else 0.0


def _cut_candidates_for_key(span: Any, key: str) -> list[dict[str, Any]]:
    raw = _packed_value(span, key, [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    values: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            values.append(dict(item))
    return values


def _cut_candidates(span: Any) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for key in ("primary_cut_candidates", "weak_cut_candidates"):
        values.extend(_cut_candidates_for_key(span, key))
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


def _boundary_primary_cut(span: Any, boundary_time: float) -> dict[str, Any] | None:
    candidates = _cut_candidates_for_key(span, "primary_cut_candidates")
    matches: list[dict[str, Any]] = []
    for item in candidates:
        try:
            time_s = float(item.get("time_s"))
        except (TypeError, ValueError):
            continue
        if abs(time_s - boundary_time) <= 1e-6:
            matches.append(item)
    if not matches:
        return None
    matches.sort(
        key=lambda item: (
            _safe_float(item.get("p_cut")),
            _safe_float(item.get("p_role")),
        ),
        reverse=True,
    )
    return dict(matches[0])


def _split_edge_payload(span: Any, boundary_time: float) -> dict[str, Any]:
    cut = _boundary_primary_cut(span, boundary_time)
    if cut is None:
        return {
            "kind": "island_edge",
            "time_s": round(boundary_time, 6),
            "proposal_time_s": round(boundary_time, 6),
            "p_cut": 0.0,
            "p_continue": 0.0,
            "p_unsure": 0.0,
            "role": "none",
            "p_role": 0.0,
            "noise_isolation_bracket": False,
            "bracket_pair_id": "",
        }
    return {
        "kind": "split_cut",
        "time_s": round(_safe_float(cut.get("time_s"), boundary_time), 6),
        "proposal_time_s": round(_safe_float(cut.get("proposal_time_s"), boundary_time), 6),
        "p_cut": _safe_float(cut.get("p_cut")),
        "p_continue": _safe_float(cut.get("p_continue")),
        "p_unsure": _safe_float(cut.get("p_unsure")),
        "role": str(cut.get("role") or "none"),
        "p_role": _safe_float(cut.get("p_role")),
        "noise_isolation_bracket": bool(cut.get("noise_isolation_bracket")),
        "bracket_pair_id": str(cut.get("bracket_pair_id") or ""),
    }


def _split_edge_features(
    *,
    prefix: str,
    edge: Mapping[str, Any],
) -> dict[str, float]:
    role = str(edge.get("role") or "none")
    values = {
        f"{prefix}_edge_is_split_cut": 1.0 if edge.get("kind") == "split_cut" else 0.0,
        f"{prefix}_edge_is_island_edge": 1.0 if edge.get("kind") == "island_edge" else 0.0,
        f"{prefix}_edge_noise_isolation_bracket": _bool_feature(
            edge.get("noise_isolation_bracket")
        ),
        f"{prefix}_edge_p_cut": _safe_float(edge.get("p_cut")),
        f"{prefix}_edge_p_continue": _safe_float(edge.get("p_continue")),
        f"{prefix}_edge_p_unsure": _safe_float(edge.get("p_unsure")),
        f"{prefix}_edge_p_role": _safe_float(edge.get("p_role")),
    }
    for known_role in PRE_ASR_CUEQC_SPLIT_EDGE_ROLES:
        values[f"{prefix}_edge_role_{known_role}"] = 1.0 if role == known_role else 0.0
    values[f"{prefix}_edge_role_unknown"] = (
        1.0 if role not in PRE_ASR_CUEQC_SPLIT_EDGE_ROLES else 0.0
    )
    return values


def _split_edge_pair_features(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    left_pair = str(left.get("bracket_pair_id") or "")
    right_pair = str(right.get("bracket_pair_id") or "")
    same_pair = bool(left_pair and right_pair and left_pair == right_pair)
    return {
        "left_right_same_noise_pair": 1.0 if same_pair else 0.0,
        "left_right_split_edge_gap_s": (
            abs(_safe_float(right.get("time_s")) - _safe_float(left.get("time_s")))
            if left.get("kind") == "split_cut" and right.get("kind") == "split_cut"
            else 0.0
        ),
    }


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
    schema = str(
        _packed_value(
            span,
            "pre_asr_ptm_pooling_schema",
            _packed_value(span, "ptm_pooling_schema", ""),
        )
        or ""
    )
    bins = _packed_value(
        span,
        "pre_asr_ptm_pooling_bins",
        _packed_value(span, "ptm_pooling_bins"),
    )
    dim = _packed_value(
        span,
        "pre_asr_ptm_pooling_dim",
        _packed_value(span, "ptm_pooling_dim"),
    )
    try:
        parsed_bins = None if bins is None else int(bins)
    except (TypeError, ValueError):
        parsed_bins = None
    try:
        parsed_dim = None if dim is None else int(dim)
    except (TypeError, ValueError):
        parsed_dim = None
    values = _numeric_list(raw_values)
    expected_dim = len(PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES)
    boundary_pipeline_version = int(
        _safe_float(_packed_value(span, "boundary_pipeline_version"), 0.0)
    )
    allowed_schemas = (
        {CHUNK_LEARNED_PROJECTED_PTM_SCHEMA}
        if boundary_pipeline_version == 10
        else {
            CHUNK_POOLED_PTM_SCHEMA,
            CHUNK_PROJECTED_PTM_SCHEMA,
            CHUNK_LEARNED_PROJECTED_PTM_SCHEMA,
        }
    )
    available = (
        schema in allowed_schemas
        and len(values) == expected_dim
        and (parsed_bins is None or parsed_bins == PRE_ASR_CUEQC_PTM_BINS)
        and (parsed_dim is None or parsed_dim == expected_dim)
    )
    if available:
        return values, True, schema, parsed_bins or PRE_ASR_CUEQC_PTM_BINS, expected_dim
    if require_ptm_pooling:
        raise ValueError("Pre-ASR CueQC requires chunk-level learned-projection PTM features")
    return [0.0] * expected_dim, False, schema, parsed_bins, parsed_dim


def ptm_bin_matrix(candidate: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_pooled = candidate.get("pre_asr_ptm_pooled_features")
    pooled = _numeric_list(raw_pooled)
    expected = len(PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES)
    if len(pooled) != expected:
        pooled = [0.0] * expected
    dim = PRE_ASR_CUEQC_PTM_DIM
    values = np.asarray(pooled, dtype=np.float32)
    if values.size != PRE_ASR_CUEQC_MODEL_PTM_TOKENS * dim:
        values = np.zeros((PRE_ASR_CUEQC_MODEL_PTM_TOKENS * dim,), dtype=np.float32)
    matrix = values.reshape(PRE_ASR_CUEQC_MODEL_PTM_TOKENS, dim)
    available = bool(candidate.get("ptm_pooling_available"))
    mask = np.ones((PRE_ASR_CUEQC_MODEL_PTM_TOKENS,), dtype=np.float32) if available else np.zeros(
        (PRE_ASR_CUEQC_MODEL_PTM_TOKENS,), dtype=np.float32
    )
    return matrix, mask


def _numeric_list(value: Any) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)):
        return []
    if isinstance(value, np.ndarray):
        return [_safe_float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, Sequence):
        return [_safe_float(item) for item in value]
    return []


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
    boundary_pipeline_version = int(
        _safe_float(_packed_value(span, "boundary_pipeline_version"), 0.0)
    )
    feature_schema = (
        PRE_ASR_CUEQC_FEATURE_SCHEMA
        if boundary_pipeline_version == 10
        else PRE_ASR_CUEQC_LEGACY_FEATURE_SCHEMA
    )
    runtime_adapter = (
        PRE_ASR_CUEQC_RUNTIME_ADAPTER
        if boundary_pipeline_version == 10
        else PRE_ASR_CUEQC_LEGACY_RUNTIME_ADAPTER
    )
    start = _safe_float(_packed_value(span, "start"))
    end = max(start, _safe_float(_packed_value(span, "end"), start))
    duration = max(0.0, end - start)
    raw_start = _safe_float(_packed_value(span, "raw_start"), start)
    raw_end = max(raw_start, _safe_float(_packed_value(span, "raw_end"), end))
    raw_duration = max(0.0, _safe_float(_packed_value(span, "raw_duration"), raw_end - raw_start))
    acoustic_start = _safe_float(_packed_value(span, "acoustic_start"), start)
    acoustic_end = max(acoustic_start, _safe_float(_packed_value(span, "acoustic_end"), end))
    acoustic_duration = max(
        0.0,
        _safe_float(_packed_value(span, "acoustic_duration"), acoustic_end - acoustic_start),
    )
    left_split_edge = _split_edge_payload(span, start)
    right_split_edge = _split_edge_payload(span, end)
    pooled_values, pooled_available, pooling_schema, pooling_bins, pooling_dim = _pooled_ptm_values(
        span,
        require_ptm_pooling=require_ptm_pooling,
    )
    start_delta = _safe_float(
        _packed_value(
            span,
            "boundary_start_refine_delta_s",
            _packed_value(span, "refiner_applied_start_delta_s"),
        )
    )
    end_delta = _safe_float(
        _packed_value(
            span,
            "boundary_end_refine_delta_s",
            _packed_value(span, "refiner_applied_end_delta_s"),
        )
    )
    pred_start_delta = _safe_float(_packed_value(span, "refiner_pred_start_delta_s"))
    pred_end_delta = _safe_float(_packed_value(span, "refiner_pred_end_delta_s"))
    applied_start_delta = _safe_float(_packed_value(span, "refiner_applied_start_delta_s"), start_delta)
    applied_end_delta = _safe_float(_packed_value(span, "refiner_applied_end_delta_s"), end_delta)
    start_confidence = _safe_float(_packed_value(span, "refiner_start_confidence"))
    end_confidence = _safe_float(_packed_value(span, "refiner_end_confidence"))
    start_source = str(_packed_value(span, "refiner_start_source", "") or "")
    end_source = str(_packed_value(span, "refiner_end_source", "") or "")
    safety_action = str(_packed_value(span, "refiner_safety_action", "") or "")
    trim_left = max(0.0, start_delta)
    trim_right = max(0.0, -end_delta)
    trim_total = trim_left + trim_right
    top1, top2, prom1, prom2, split_peak_count = _top_split_values(span)
    primary_cut_count = len(_cut_candidates_for_key(span, "primary_cut_candidates"))
    weak_cut_count = len(_cut_candidates_for_key(span, "weak_cut_candidates"))
    semantic_candidates = (
        _cut_candidates_for_key(span, "primary_cut_candidates")
        + _cut_candidates_for_key(span, "weak_cut_candidates")
    )
    p_cut_values = [_safe_float(item.get("p_cut")) for item in semantic_candidates]
    p_continue_values = [_safe_float(item.get("p_continue")) for item in semantic_candidates]
    p_unsure_values = [_safe_float(item.get("p_unsure")) for item in semantic_candidates]
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
        "raw_start_s": raw_start,
        "raw_end_s": raw_end,
        "raw_duration_s": raw_duration,
        "acoustic_start_s": acoustic_start,
        "acoustic_end_s": acoustic_end,
        "acoustic_duration_s": acoustic_duration,
        "refined_duration_s": duration,
        "raw_to_acoustic_start_shift_s": acoustic_start - raw_start,
        "raw_to_acoustic_end_shift_s": raw_end - acoustic_end,
        "prev_gap_s": _gap(spans, index, left=True),
        "next_gap_s": _gap(spans, index, left=False),
        "speech_segment_count": _sequence_count(_packed_value(span, "speech_segments", [])),
        "internal_gap_count": _safe_float(_packed_value(span, "internal_gap_count")),
        "internal_gap_max_s": _safe_float(_packed_value(span, "internal_gap_max_s")),
        "boundary_score": _safe_float(_packed_value(span, "boundary_score")),
        "start_refine_delta_s": start_delta,
        "end_refine_delta_s": end_delta,
        "refiner_pred_start_delta_s": pred_start_delta,
        "refiner_pred_end_delta_s": pred_end_delta,
        "refiner_applied_start_delta_s": applied_start_delta,
        "refiner_applied_end_delta_s": applied_end_delta,
        "refiner_abs_pred_start_delta_s": abs(pred_start_delta),
        "refiner_abs_pred_end_delta_s": abs(pred_end_delta),
        "refiner_abs_applied_start_delta_s": abs(applied_start_delta),
        "refiner_abs_applied_end_delta_s": abs(applied_end_delta),
        "refiner_start_confidence": start_confidence,
        "refiner_end_confidence": end_confidence,
        "refiner_confidence_min": min(start_confidence, end_confidence),
        "refiner_confidence_mean": (start_confidence + end_confidence) * 0.5,
        "refiner_effective_start_delta_max_s": _safe_float(
            _packed_value(span, "refiner_effective_start_delta_max_s")
        ),
        "refiner_effective_end_delta_max_s": _safe_float(
            _packed_value(span, "refiner_effective_end_delta_max_s")
        ),
        "refiner_start_source_model": _source_feature(start_source, "model"),
        "refiner_end_source_model": _source_feature(end_source, "model"),
        "refiner_start_source_model_scaled": _source_feature(start_source, "model_scaled"),
        "refiner_end_source_model_scaled": _source_feature(end_source, "model_scaled"),
        "refiner_start_source_noop_low_confidence": _source_feature(
            start_source,
            "noop_low_confidence",
        ),
        "refiner_end_source_noop_low_confidence": _source_feature(
            end_source,
            "noop_low_confidence",
        ),
        "refiner_start_source_acoustic_fallback": _source_feature(
            start_source,
            "acoustic_fallback",
        ),
        "refiner_end_source_acoustic_fallback": _source_feature(
            end_source,
            "acoustic_fallback",
        ),
        "refiner_start_source_unknown": _source_unknown(
            start_source,
            {"model", "model_scaled", "noop_low_confidence", "acoustic_fallback"},
        ),
        "refiner_end_source_unknown": _source_unknown(
            end_source,
            {"model", "model_scaled", "noop_low_confidence", "acoustic_fallback"},
        ),
        "refiner_safety_clamp": _contains_token(safety_action, "clamp"),
        "refiner_safety_rollback": _contains_token(safety_action, "rollback"),
        "refiner_fallback_used": _bool_feature(_packed_value(span, "refiner_fallback_used")),
        "refiner_shared_boundary_adjusted": _bool_feature(
            _packed_value(span, "refiner_shared_boundary_adjusted")
        ),
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
        "primary_cut_count": float(primary_cut_count),
        "weak_cut_count": float(weak_cut_count),
        "primary_cut_density": float(primary_cut_count) / max(duration, 1e-6),
        "weak_cut_density": float(weak_cut_count) / max(duration, 1e-6),
        "semantic_p_cut_mean": float(np.mean(p_cut_values)) if p_cut_values else 0.0,
        "semantic_p_cut_max": max(p_cut_values, default=0.0),
        "semantic_p_continue_mean": (
            float(np.mean(p_continue_values)) if p_continue_values else 0.0
        ),
        "semantic_p_unsure_mean": (
            float(np.mean(p_unsure_values)) if p_unsure_values else 0.0
        ),
        "semantic_unsure_ratio": (
            sum(
                1
                for item in semantic_candidates
                if str(item.get("label") or "") == "unsure"
            )
            / max(1, len(semantic_candidates))
        ),
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
        **_split_edge_features(prefix="left", edge=left_split_edge),
        **_split_edge_features(prefix="right", edge=right_split_edge),
        **_split_edge_pair_features(left_split_edge, right_split_edge),
        **neighbor_left,
        **neighbor_right,
    }
    return {
        "schema": feature_schema,
        "feature_schema": feature_schema,
        "runtime_adapter": runtime_adapter,
        "boundary_pipeline_version": boundary_pipeline_version,
        "index": index,
        "start": round(start, 6),
        "end": round(end, 6),
        "raw_start": round(raw_start, 6),
        "raw_end": round(raw_end, 6),
        "raw_duration": round(raw_duration, 6),
        "acoustic_start": round(acoustic_start, 6),
        "acoustic_end": round(acoustic_end, 6),
        "acoustic_duration": round(acoustic_duration, 6),
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
        "refiner_start_confidence": start_confidence,
        "refiner_end_confidence": end_confidence,
        "refiner_start_source": start_source,
        "refiner_end_source": end_source,
        "refiner_safety_action": safety_action,
        "refiner_safety_reason": str(_packed_value(span, "refiner_safety_reason", "") or ""),
        "refiner_fallback_used": bool(features["refiner_fallback_used"]),
        "refiner_shared_boundary_adjusted": bool(features["refiner_shared_boundary_adjusted"]),
        "pre_asr_split_edges": {
            "left": left_split_edge,
            "right": right_split_edge,
        },
        "pre_asr_split_edge_pair_id": (
            left_split_edge["bracket_pair_id"]
            if left_split_edge["bracket_pair_id"]
            and left_split_edge["bracket_pair_id"] == right_split_edge["bracket_pair_id"]
            else ""
        ),
        "features": features,
        "scalar_feature_names": list(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),
        "ptm_bin_feature_names": list(PRE_ASR_CUEQC_PTM_BIN_FEATURE_NAMES),
        "feature_names": list(PRE_ASR_CUEQC_FEATURE_NAMES),
        "ptm_pooling_schema": pooling_schema,
        "ptm_pooling_available": pooled_available,
        "ptm_pooling_bins": pooling_bins,
        "ptm_pooling_dim": pooling_dim,
        "ptm_projection_digest": str(
            _packed_value(span, "pre_asr_ptm_projection_digest", "") or ""
        ),
        "pre_asr_ptm_pooled_features": pooled_values,
    }


def scalar_vector(
    candidate: Mapping[str, Any],
    *,
    feature_names: Sequence[str] = PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES,
) -> np.ndarray:
    features = candidate.get("features") if isinstance(candidate.get("features"), Mapping) else {}
    return np.asarray(
        [_safe_float(features.get(name)) for name in feature_names],
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
    *,
    scalar_feature_names: Sequence[str] = PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES,
) -> dict[str, np.ndarray | list[tuple[int, int]]]:
    groups = planned_island_sequences(candidates)
    batch = len(groups)
    max_chunks = max((len(group) for group in groups), default=0)
    scalar = np.zeros(
        (batch, max_chunks, len(scalar_feature_names)),
        dtype=np.float32,
    )
    bins = np.zeros(
        (batch, max_chunks, PRE_ASR_CUEQC_MODEL_PTM_TOKENS, PRE_ASR_CUEQC_PTM_DIM),
        dtype=np.float32,
    )
    bin_mask = np.zeros((batch, max_chunks, PRE_ASR_CUEQC_MODEL_PTM_TOKENS), dtype=np.float32)
    chunk_mask = np.zeros((batch, max_chunks), dtype=np.float32)
    positions: list[tuple[int, int] | None] = [None] * len(candidates)
    for group_index, group in enumerate(groups):
        for chunk_index, (original_index, candidate) in enumerate(group):
            scalar[group_index, chunk_index] = scalar_vector(
                candidate,
                feature_names=scalar_feature_names,
            )
            matrix, mask = ptm_bin_matrix(candidate)
            bins[group_index, chunk_index] = matrix
            bin_mask[group_index, chunk_index] = mask
            chunk_mask[group_index, chunk_index] = 1.0
            positions[original_index] = (group_index, chunk_index)
    if any(position is None for position in positions):
        raise RuntimeError("Pre-ASR CueQC sequence planning left candidates unassigned")
    return {
        "scalar_features": scalar,
        "ptm_bins": bins,
        "bin_mask": bin_mask,
        "chunk_mask": chunk_mask,
        "positions": [position for position in positions if position is not None],
    }


class PreAsrCueQCNetwork:
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
                temporal_layers: int = 2,
                temporal_residual_scale: float = 1.0,
                valid_prefix_temporal: bool = False,
                ptm_encoder_mode: str = "summary_mlp",
                semantic_auxiliary: bool = False,
                late_fusion: bool = False,
                dropout: float = 0.1,
                num_classes: int = len(PRE_ASR_CUEQC_LABELS),
            ) -> None:
                super().__init__()
                from boundary.backbones import TinyMamba2BoundaryBackbone

                self.arch = PRE_ASR_CUEQC_MODEL_ARCH
                self.temporal_residual_scale = float(temporal_residual_scale)
                self.valid_prefix_temporal = bool(valid_prefix_temporal)
                self.ptm_encoder_mode = str(ptm_encoder_mode)
                self.semantic_auxiliary = bool(semantic_auxiliary)
                self.late_fusion = bool(late_fusion)
                if self.ptm_encoder_mode == "summary_mlp":
                    self.ptm_encoder = nn.Sequential(
                        nn.LayerNorm(ptm_dim * 4),
                        nn.Linear(ptm_dim * 4, hidden_size * 2),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size * 2, hidden_size),
                        nn.GELU(),
                    )
                    self.ptm_token_projection = None
                    self.ptm_token_encoder = None
                    self.ptm_token_pool = None
                    self.ptm_position_embedding = None
                elif self.ptm_encoder_mode == "token_attention":
                    self.ptm_encoder = None
                    self.ptm_token_projection = nn.Sequential(
                        nn.LayerNorm(ptm_dim),
                        nn.Linear(ptm_dim, hidden_size),
                        nn.GELU(),
                    )
                    token_layer = nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=4,
                        dim_feedforward=hidden_size * 2,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    self.ptm_token_encoder = nn.TransformerEncoder(
                        token_layer,
                        num_layers=1,
                        enable_nested_tensor=False,
                    )
                    self.ptm_token_pool = nn.Sequential(
                        nn.LayerNorm(hidden_size * 2),
                        nn.Linear(hidden_size * 2, hidden_size),
                        nn.GELU(),
                    )
                    self.ptm_position_embedding = nn.Parameter(
                        torch.zeros(1, PRE_ASR_CUEQC_MODEL_PTM_TOKENS, hidden_size)
                    )
                    nn.init.normal_(self.ptm_position_embedding, std=0.02)
                else:
                    raise ValueError(f"unsupported ptm_encoder_mode: {self.ptm_encoder_mode!r}")
                self.scalar_encoder = nn.Sequential(
                    nn.LayerNorm(scalar_dim),
                    nn.Linear(scalar_dim, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                )
                self.chunk_fuse = nn.Sequential(
                    nn.LayerNorm(hidden_size * 2),
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.delta_encoder = nn.Sequential(
                    nn.LayerNorm(hidden_size * 3),
                    nn.Linear(hidden_size * 3, hidden_size),
                    nn.GELU(),
                )
                self.temporal_backbone = TinyMamba2BoundaryBackbone(
                    input_dim=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=temporal_layers,
                    state_size=32,
                    num_heads=4,
                    head_dim=64,
                    n_groups=2,
                    chunk_size=8,
                    bidirectional=True,
                    valid_prefix_bidirectional=self.valid_prefix_temporal,
                )
                self.temporal_projection = nn.Linear(
                    self.temporal_backbone.output_dim,
                    hidden_size,
                )
                self.temporal_gate = nn.Sequential(
                    nn.LayerNorm(hidden_size * 2),
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.Sigmoid(),
                )
                self.classifier = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, num_classes),
                )
                if self.semantic_auxiliary or self.late_fusion:
                    self.semantic_classifier = nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Linear(hidden_size, hidden_size),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size, num_classes),
                    )
                else:
                    self.semantic_classifier = None
                if self.late_fusion:
                    self.semantic_fusion_gate = nn.Sequential(
                        nn.LayerNorm(hidden_size * 2),
                        nn.Linear(hidden_size * 2, 1),
                        nn.Sigmoid(),
                    )
                    nn.init.constant_(self.semantic_fusion_gate[1].bias, 1.0)
                else:
                    self.semantic_fusion_gate = None

            def _encode_ptm(
                self,
                ptm_bins: torch.Tensor,
                bin_mask: torch.Tensor | None,
            ) -> torch.Tensor:
                batch, chunks, bins, dim = ptm_bins.shape
                if self.ptm_encoder_mode == "summary_mlp":
                    global_mean = ptm_bins[:, :, 0]
                    global_max = ptm_bins[:, :, 1]
                    local = ptm_bins[:, :, 2:]
                    local_mean = local.mean(dim=2)
                    local_max = local.max(dim=2).values
                    ptm = torch.cat(
                        (global_mean, global_max, local_mean, local_max),
                        dim=-1,
                    ).reshape(batch * chunks, dim * 4)
                    return self.ptm_encoder(ptm)

                tokens = ptm_bins.reshape(batch * chunks, bins, dim)
                token_mask = (
                    torch.ones((batch * chunks, bins), dtype=torch.bool, device=ptm_bins.device)
                    if bin_mask is None
                    else bin_mask.reshape(batch * chunks, bins) > 0
                )
                safe_mask = token_mask.clone()
                empty = ~safe_mask.any(dim=1)
                safe_mask[empty, 0] = True
                encoded = self.ptm_token_projection(tokens)
                encoded = encoded + self.ptm_position_embedding[:, :bins]
                encoded = self.ptm_token_encoder(
                    encoded,
                    src_key_padding_mask=~safe_mask,
                )
                weights = safe_mask.unsqueeze(-1).to(encoded.dtype)
                pooled_mean = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                pooled_max = encoded.masked_fill(~safe_mask.unsqueeze(-1), -torch.inf).max(dim=1).values
                pooled_max = torch.where(empty.unsqueeze(-1), torch.zeros_like(pooled_max), pooled_max)
                return self.ptm_token_pool(torch.cat((pooled_mean, pooled_max), dim=-1))

            def forward(
                self,
                ptm_bins: torch.Tensor,
                scalar_features: torch.Tensor,
                chunk_mask: torch.Tensor | None = None,
                bin_mask: torch.Tensor | None = None,
                return_auxiliary: bool = False,
            ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
                if ptm_bins.ndim != 4:
                    raise ValueError("ptm_bins must have shape [batch, chunks, bins, dim]")
                if scalar_features.ndim != 3:
                    raise ValueError("scalar_features must have shape [batch, chunks, dim]")
                batch, chunks, _bins, _dim = ptm_bins.shape
                ptm_repr = self._encode_ptm(ptm_bins, bin_mask)
                scalar = scalar_features.reshape(batch * chunks, -1)
                scalar_repr = self.scalar_encoder(scalar)
                chunk_repr = self.chunk_fuse(
                    torch.cat([ptm_repr, scalar_repr], dim=-1)
                )
                local_repr = chunk_repr.reshape(batch, chunks, -1)
                if self.valid_prefix_temporal and chunk_mask is not None:
                    local_repr = local_repr * chunk_mask.unsqueeze(-1).to(local_repr.dtype)
                if self.temporal_residual_scale <= 0.0:
                    context_logits = self.classifier(local_repr)
                    semantic_logits = (
                        self.semantic_classifier(ptm_repr).reshape(batch, chunks, -1)
                        if self.semantic_classifier is not None
                        else context_logits
                    )
                    if self.late_fusion:
                        semantic_repr = ptm_repr.reshape(batch, chunks, -1)
                        gate = self.semantic_fusion_gate(
                            torch.cat((semantic_repr, local_repr), dim=-1)
                        )
                        logits = gate * semantic_logits + (1.0 - gate) * context_logits
                    else:
                        logits = context_logits
                    if chunk_mask is not None:
                        logits = logits * chunk_mask.unsqueeze(-1).to(dtype=logits.dtype)
                    if return_auxiliary:
                        return logits, {
                            "semantic_logits": semantic_logits,
                            "local_context_logits": context_logits,
                        }
                    return logits
                previous = torch.cat(
                    (torch.zeros_like(local_repr[:, :1]), local_repr[:, :-1]),
                    dim=1,
                )
                following = torch.cat(
                    (local_repr[:, 1:], torch.zeros_like(local_repr[:, :1])),
                    dim=1,
                )
                delta_repr = self.delta_encoder(
                    torch.cat(
                        (
                            local_repr,
                            local_repr - previous,
                            following - local_repr,
                        ),
                        dim=-1,
                    )
                )
                if self.valid_prefix_temporal and chunk_mask is not None:
                    delta_repr = delta_repr * chunk_mask.unsqueeze(-1).to(delta_repr.dtype)
                cm = None if chunk_mask is None else chunk_mask.long()
                temporal = self.temporal_backbone(
                    local_repr + delta_repr,
                    attention_mask=cm,
                )
                temporal = self.temporal_projection(temporal)
                gate = self.temporal_gate(
                    torch.cat((local_repr, delta_repr), dim=-1)
                )
                contextual_repr = local_repr + self.temporal_residual_scale * gate * temporal
                context_logits = self.classifier(contextual_repr)
                semantic_logits = (
                    self.semantic_classifier(ptm_repr).reshape(batch, chunks, -1)
                    if self.semantic_classifier is not None
                    else context_logits
                )
                if self.late_fusion:
                    semantic_repr = ptm_repr.reshape(batch, chunks, -1)
                    semantic_weight = self.semantic_fusion_gate(
                        torch.cat((semantic_repr, contextual_repr), dim=-1)
                    )
                    logits = semantic_weight * semantic_logits + (1.0 - semantic_weight) * context_logits
                else:
                    logits = context_logits
                if chunk_mask is not None:
                    logits = logits * chunk_mask.unsqueeze(-1).to(dtype=logits.dtype)
                    semantic_logits = semantic_logits * chunk_mask.unsqueeze(-1).to(dtype=logits.dtype)
                    context_logits = context_logits * chunk_mask.unsqueeze(-1).to(dtype=logits.dtype)
                if return_auxiliary:
                    return logits, {
                        "semantic_logits": semantic_logits,
                        "local_context_logits": self.classifier(local_repr),
                        "context_logits": context_logits,
                    }
                return logits

        return _Model(*args, **kwargs)


def make_model_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(config or {})
    return {
        "ptm_dim": int(raw.get("ptm_dim") or PRE_ASR_CUEQC_PTM_DIM),
        "scalar_dim": int(raw.get("scalar_dim") or len(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES)),
        "hidden_size": int(raw.get("hidden_size") or 128),
        "temporal_layers": int(raw.get("temporal_layers") or 2),
        "temporal_residual_scale": float(raw.get("temporal_residual_scale", 1.0)),
        "valid_prefix_temporal": bool(raw.get("valid_prefix_temporal", False)),
        "ptm_encoder_mode": str(raw.get("ptm_encoder_mode") or "summary_mlp"),
        "semantic_auxiliary": bool(raw.get("semantic_auxiliary", False)),
        "late_fusion": bool(raw.get("late_fusion", False)),
        "dropout": float(raw.get("dropout", 0.1)),
        "num_classes": int(raw.get("num_classes") or len(PRE_ASR_CUEQC_LABELS)),
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

        contract = _runtime_contract(expected_asr_repo_id)
        schema = str(checkpoint.get("schema") or "")
        if schema != contract["schema"]:
            raise ValueError(f"unsupported Pre-ASR CueQC schema: {schema!r}")
        arch = str(checkpoint.get("arch") or "")
        if arch != contract["arch"]:
            raise ValueError(f"unsupported Pre-ASR CueQC arch: {arch!r}")
        if str(checkpoint.get("feature_schema") or "") != contract["feature_schema"]:
            raise ValueError("Pre-ASR CueQC feature_schema does not match runtime")
        if str(checkpoint.get("runtime_adapter") or "") != contract["runtime_adapter"]:
            raise ValueError("Pre-ASR CueQC runtime_adapter does not match runtime")
        metadata = dict(checkpoint.get("metadata") or {})
        artifact = metadata.get("artifact")
        if not isinstance(artifact, dict):
            raise ValueError("Pre-ASR CueQC metadata.artifact is required")
        for key, expected in contract["artifact"].items():
            if artifact.get(key) != expected:
                raise ValueError(
                    f"Pre-ASR CueQC metadata.artifact.{key} must be {expected!r}"
                )
        if expected_asr_repo_id is not None:
            validate_checkpoint_repo_id(
                str(metadata.get("asr_repo_id") or ""),
                expected_asr_repo_id,
                checkpoint_kind="Pre-ASR CueQC",
                metadata_key="metadata.asr_repo_id",
            )
        feature_names = tuple(str(item) for item in checkpoint.get("feature_names") or ())
        if not feature_names:
            raise ValueError("Pre-ASR CueQC checkpoint feature_names must not be empty")
        unknown_feature_names = [
            name for name in feature_names if name not in PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES
        ]
        if unknown_feature_names:
            raise ValueError(
                "Pre-ASR CueQC checkpoint feature_names contain unknown fields: "
                + ", ".join(unknown_feature_names[:5])
            )
        lowered = " ".join(PRE_ASR_CUEQC_FEATURE_NAMES).lower()
        if any(token in lowered for token in _BANNED_FEATURE_TOKENS):
            raise ValueError("Pre-ASR CueQC feature schema must not contain ASR text/token/decoder fields")

        raw_model_config = dict(checkpoint.get("model_config") or {})
        raw_model_config.setdefault("num_classes", int(contract["num_classes"]))
        config = make_model_config(raw_model_config)
        if config["ptm_dim"] != PRE_ASR_CUEQC_PTM_DIM:
            raise ValueError("Pre-ASR CueQC model_config.ptm_dim does not match runtime")
        if config["scalar_dim"] != len(feature_names):
            raise ValueError("Pre-ASR CueQC model_config.scalar_dim does not match runtime")
        if config["num_classes"] != int(contract["num_classes"]):
            raise ValueError("Pre-ASR CueQC model_config.num_classes does not match runtime")
        self.model = PreAsrCueQCNetwork(**config)
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
        self.contract = contract
        self.scalar_feature_names = feature_names
        self.mean = np.asarray(
            checkpoint.get("feature_mean", [0.0] * len(feature_names)),
            dtype=np.float32,
        )
        self.std = np.asarray(
            checkpoint.get("feature_std", [1.0] * len(feature_names)),
            dtype=np.float32,
        )
        if self.mean.shape[0] != len(feature_names) or self.std.shape[0] != len(feature_names):
            raise ValueError("Pre-ASR CueQC scalar normalization shape mismatch")
        decision = dict(checkpoint.get("decision_config") or {})
        self.decision_mode = str(
            decision.get("decision_mode")
            or ("drop_threshold" if config["num_classes"] == 2 else "argmax")
        )
        if config["num_classes"] == 3 and self.decision_mode != "argmax":
            raise ValueError("Pre-ASR CueQC v13 decision_mode must be argmax")
        self.drop_threshold = float(
            decision.get("drop_threshold", PRE_ASR_CUEQC_DEFAULT_DROP_THRESHOLD)
        )
        self.inference_window_size = max(0, int(decision.get("inference_window_size") or 512))
        self.hard_keep_veto_enabled = bool(decision.get("hard_keep_veto", False))
        self.hard_drop_rule_enabled = bool(decision.get("hard_drop_rule", False))
        self.keep_veto_enabled = bool(decision.get("keep_veto", False))
        if (
            self.hard_keep_veto_enabled
            or self.hard_drop_rule_enabled
            or self.keep_veto_enabled
        ):
            raise ValueError("Pre-ASR CueQC checkpoints must disable hard rules")
        self.hard_keep_min_duration_s = float(decision.get("hard_keep_min_duration_s", 0.80))
        self.high_speech_p90 = float(decision.get("high_speech_p90", 0.85))
        self.high_active_ratio = float(decision.get("high_active_ratio", 0.50))
        self.very_low_speech_p90 = float(decision.get("very_low_speech_p90", 0.05))
        self.very_low_active_ratio = float(decision.get("very_low_active_ratio", 0.05))

    def signature(self) -> dict[str, Any]:
        signature = {
            "schema": self.contract["schema"],
            "arch": self.contract["arch"],
            "feature_schema": self.contract["feature_schema"],
            "runtime_adapter": self.contract["runtime_adapter"],
            "path": str(self.path),
            "sha256": self.sha256,
            "decision_mode": self.decision_mode,
            "inference_window_size": self.inference_window_size,
            "feature_names": list(self.scalar_feature_names),
            "metadata": self.metadata,
        }
        if self.config["num_classes"] == 2:
            signature["drop_threshold"] = self.drop_threshold
        return signature

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
        tensors = sequence_tensors(
            candidates,
            scalar_feature_names=self.scalar_feature_names,
        )
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
            ptm_t = torch.from_numpy(ptm_bins).to(self.device)
            scalar_t = torch.from_numpy(scalar).to(self.device)
            chunk_mask_t = torch.from_numpy(chunk_mask).to(self.device)
            bin_mask_t = torch.from_numpy(bin_mask).to(self.device)
            if self.inference_window_size > 0 and ptm_t.shape[1] > self.inference_window_size:
                batch, max_chunks = tuple(chunk_mask_t.shape)
                logits = torch.zeros(
                    (batch, max_chunks, int(self.config["num_classes"])),
                    dtype=torch.float32,
                    device=self.device,
                )
                counts = torch.zeros((batch, max_chunks, 1), dtype=torch.float32, device=self.device)
                window = min(int(self.inference_window_size), int(max_chunks))
                for group_index in range(int(batch)):
                    length = int(chunk_mask_t[group_index].sum().detach().cpu().item())
                    if length <= 0:
                        continue
                    for start in range(0, length, window):
                        end = min(length, start + window)
                        window_logits = self.model(
                            ptm_t[group_index : group_index + 1, start:end],
                            scalar_t[group_index : group_index + 1, start:end],
                            chunk_mask=chunk_mask_t[group_index : group_index + 1, start:end],
                            bin_mask=bin_mask_t[group_index : group_index + 1, start:end],
                        )
                        logits[group_index, start:end] += window_logits[0].float()
                        counts[group_index, start:end] += 1.0
                logits = logits / counts.clamp_min(1.0)
            else:
                logits = self.model(
                    ptm_t,
                    scalar_t,
                    chunk_mask=chunk_mask_t,
                    bin_mask=bin_mask_t,
                )
            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
        positions = list(tensors["positions"])
        decisions: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            group_index, chunk_index = positions[index]
            prob = probs[group_index, chunk_index]
            p_drop = float(prob[0])
            p_keep = float(prob[1])
            if self.config["num_classes"] == 3:
                p_unsure = float(prob[2])
                label_index = int(np.argmax(prob))
                label = PRE_ASR_CUEQC_LABELS[label_index]
                decisions.append(
                    {
                        "schema": "pre_asr_cueqc_decision_v3",
                        "decision_version": self.contract["decision_version"],
                        "model_schema": self.contract["schema"],
                        "model_arch": self.contract["arch"],
                        "feature_schema": self.contract["feature_schema"],
                        "runtime_adapter": self.contract["runtime_adapter"],
                        "index": int(candidate.get("index", index)),
                        "route": (
                            "drop_before_asr"
                            if label == "drop"
                            else "keep_for_asr"
                            if label == "keep"
                            else "unsure_for_asr"
                        ),
                        "label": label,
                        "confidence": round(float(prob[label_index]), 4),
                        "prob_drop": round(p_drop, 4),
                        "prob_keep": round(p_keep, 4),
                        "prob_unsure": round(p_unsure, 4),
                        "decision_mode": "argmax",
                        "reason": f"model_argmax_{label}",
                    }
                )
                continue
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
                    "schema": "pre_asr_cueqc_decision_v2",
                    "decision_version": self.contract["decision_version"],
                    "model_schema": self.contract["schema"],
                    "model_arch": self.contract["arch"],
                    "feature_schema": self.contract["feature_schema"],
                    "runtime_adapter": self.contract["runtime_adapter"],
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
    contract = _runtime_contract()
    if not enabled():
        return {
            "schema": contract["schema"],
            "arch": contract["arch"],
            "enabled": False,
            "feature_schema": contract["feature_schema"],
            "runtime_adapter": contract["runtime_adapter"],
        }
    checkpoint = _checkpoint_path()
    signature = {
        "schema": contract["schema"],
        "arch": contract["arch"],
        "enabled": True,
        "feature_schema": contract["feature_schema"],
        "runtime_adapter": contract["runtime_adapter"],
        "model_path": checkpoint,
    }
    if contract["num_classes"] == 2:
        signature["drop_threshold"] = os.getenv("PRE_ASR_CUEQC_DROP_THRESHOLD", "")
    else:
        signature["decision_mode"] = "argmax"
    return signature


def _drop_threshold_from_env(default: float) -> float:
    override = os.getenv("PRE_ASR_CUEQC_DROP_THRESHOLD", "").strip()
    if not override:
        return float(default)
    try:
        parsed = float(override)
    except ValueError as exc:
        raise ValueError(
            "PRE_ASR_CUEQC_DROP_THRESHOLD must be a float in [0, 1], "
            f"got {override!r}"
        ) from exc
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(
            "PRE_ASR_CUEQC_DROP_THRESHOLD must be a float in [0, 1], "
            f"got {override!r}"
        )
    return parsed


def load_active(*, expected_asr_repo_id: str | None = None) -> PreAsrCueQC:
    path = _checkpoint_path(expected_asr_repo_id)
    device = os.getenv("PRE_ASR_CUEQC_DEVICE", "auto").strip() or "auto"
    model = load_checkpoint(path, expected_asr_repo_id=expected_asr_repo_id, device=device)
    if model.config["num_classes"] == 2:
        model.drop_threshold = _drop_threshold_from_env(model.drop_threshold)
    split_sha = str(model.metadata.get("semantic_split_weights_sha256") or "")
    if not split_sha:
        raise ValueError("Pre-ASR CueQC metadata.semantic_split_weights_sha256 is required")
    split_path = Path(_semantic_split_checkpoint_path(expected_asr_repo_id))
    active_sha = _file_sha256(split_path)
    if active_sha != split_sha:
        raise ValueError(
            "Pre-ASR CueQC split checkpoint sha mismatch: "
            f"checkpoint expects {split_sha}, active split {split_path} is {active_sha}"
        )
    if model.config["num_classes"] == 3:
        inner_sha = str(model.metadata.get("inner_edge_refiner_weights_sha256") or "")
        if not inner_sha:
            raise ValueError(
                "Pre-ASR CueQC v13 metadata.inner_edge_refiner_weights_sha256 is required"
            )
        inner_path = Path(_inner_edge_refiner_checkpoint_path(expected_asr_repo_id))
        active_inner_sha = _file_sha256(inner_path)
        if active_inner_sha != inner_sha:
            raise ValueError(
                "Pre-ASR CueQC v13 inner checkpoint sha mismatch: "
                f"checkpoint expects {inner_sha}, active inner {inner_path} is {active_inner_sha}"
            )
    return model
