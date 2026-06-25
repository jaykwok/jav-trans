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


PRE_ASR_CUEQC_SCHEMA = "cueqc_pre_asr_mamba_v7_binary"
PRE_ASR_CUEQC_DECISION_VERSION = "pre_asr_cueqc_binary_v1"
PRE_ASR_CUEQC_FEATURE_SCHEMA = "pre_asr_cueqc_features_v3"
PRE_ASR_CUEQC_PTM_DIM = FrameSequenceFeatureConfig().max_ptm_dims
PRE_ASR_CUEQC_PTM_BINS = DEFAULT_CHUNK_POOLED_PTM_BINS
PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES = (
    "duration_s",
    "speech_segment_count",
    "internal_gap_count",
    "internal_gap_max_s",
    "boundary_score",
    "start_refine_delta_s",
    "end_refine_delta_s",
    "scorer_speech_mean",
    "scorer_speech_max",
    "scorer_speech_p90",
    "scorer_split_mean",
    "scorer_split_max",
    "scorer_split_p90",
    "prev_gap_s",
    "next_gap_s",
    "subtitle_min_duration_s",
    "below_subtitle_min_duration",
    "micro_chunk_candidate",
    "micro_action_preserve",
    "micro_action_merge_left",
    "micro_action_merge_right",
    "left_split_score",
    "right_split_score",
    "left_split_prominence",
    "right_split_prominence",
    "left_split_speech_valley",
    "right_split_speech_valley",
)
PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES = tuple(
    chunk_pooled_ptm_feature_names(
        ptm_dim=PRE_ASR_CUEQC_PTM_DIM,
        bins=PRE_ASR_CUEQC_PTM_BINS,
    )
)
PRE_ASR_CUEQC_FEATURE_NAMES = (
    PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES
    + PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES
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
        raise ValueError("Pre-ASR CueQC v7 requires chunk-level pooled PTM features")
    return [0.0] * expected_dim, False, schema, parsed_bins, parsed_dim


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
    features = {
        "duration_s": duration,
        "speech_segment_count": _sequence_count(_packed_value(span, "speech_segments", [])),
        "internal_gap_count": _safe_float(_packed_value(span, "internal_gap_count")),
        "internal_gap_max_s": _safe_float(_packed_value(span, "internal_gap_max_s")),
        "boundary_score": _safe_float(_packed_value(span, "boundary_score")),
        "start_refine_delta_s": _safe_float(_packed_value(span, "boundary_start_refine_delta_s")),
        "end_refine_delta_s": _safe_float(_packed_value(span, "boundary_end_refine_delta_s")),
        "scorer_speech_mean": _safe_float(_packed_value(span, "scorer_speech_mean")),
        "scorer_speech_max": _safe_float(_packed_value(span, "scorer_speech_max")),
        "scorer_speech_p90": _safe_float(_packed_value(span, "scorer_speech_p90")),
        "scorer_split_mean": _safe_float(_packed_value(span, "scorer_split_mean")),
        "scorer_split_max": _safe_float(_packed_value(span, "scorer_split_max")),
        "scorer_split_p90": _safe_float(_packed_value(span, "scorer_split_p90")),
        "prev_gap_s": _gap(spans, index, left=True),
        "next_gap_s": _gap(spans, index, left=False),
        "subtitle_min_duration_s": _safe_float(_packed_value(span, "subtitle_min_duration_s")),
        "below_subtitle_min_duration": 1.0
        if bool(_packed_value(span, "below_subtitle_min_duration", False))
        else 0.0,
        "micro_chunk_candidate": 1.0
        if bool(_packed_value(span, "micro_chunk_candidate", False))
        else 0.0,
        "micro_action_preserve": max(
            _has_action(_packed_value(span, "micro_resolve_action"), "preserve_micro_candidate"),
            _has_action(_packed_value(span, "micro_resolve_action"), "preserve_edge_micro_candidate"),
        ),
        "micro_action_merge_left": _has_action(
            _packed_value(span, "micro_resolve_action"),
            "merge_micro_into_left",
        ),
        "micro_action_merge_right": _has_action(
            _packed_value(span, "micro_resolve_action"),
            "merge_micro_into_right",
        ),
        "left_split_score": _safe_float(_packed_value(span, "left_split_score")),
        "right_split_score": _safe_float(_packed_value(span, "right_split_score")),
        "left_split_prominence": _safe_float(_packed_value(span, "left_split_prominence")),
        "right_split_prominence": _safe_float(_packed_value(span, "right_split_prominence")),
        "left_split_speech_valley": _safe_float(_packed_value(span, "left_split_speech_valley")),
        "right_split_speech_valley": _safe_float(_packed_value(span, "right_split_speech_valley")),
    }
    return {
        "schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "index": index,
        "start": round(start, 6),
        "end": round(end, 6),
        "subtitle_min_duration_s": round(features["subtitle_min_duration_s"], 6),
        "below_subtitle_min_duration": bool(features["below_subtitle_min_duration"]),
        "micro_chunk_candidate": bool(features["micro_chunk_candidate"]),
        "micro_resolve_action": str(_packed_value(span, "micro_resolve_action", "") or ""),
        "micro_resolve_reason": str(_packed_value(span, "micro_resolve_reason", "") or ""),
        "left_split_score": features["left_split_score"],
        "right_split_score": features["right_split_score"],
        "features": features,
        "ptm_pooling_schema": pooling_schema,
        "ptm_pooling_available": pooled_available,
        "ptm_pooling_bins": pooling_bins,
        "ptm_pooling_dim": pooling_dim,
        "pre_asr_ptm_pooled_features": pooled_values,
        "feature_names": list(PRE_ASR_CUEQC_FEATURE_NAMES),
    }


def feature_vector(candidate: Mapping[str, Any]) -> np.ndarray:
    features = candidate.get("features") if isinstance(candidate.get("features"), Mapping) else {}
    values = [_safe_float(features.get(name)) for name in PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES]
    raw_pooled = candidate.get("pre_asr_ptm_pooled_features")
    pooled: list[float] = []
    if isinstance(raw_pooled, Sequence) and not isinstance(raw_pooled, (str, bytes, bytearray)):
        pooled = [_safe_float(item) for item in raw_pooled]
    expected = len(PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES)
    if len(pooled) != expected:
        pooled = [_safe_float(features.get(name)) for name in PRE_ASR_CUEQC_POOLED_PTM_FEATURE_NAMES]
    if len(pooled) != expected:
        pooled = [0.0] * expected
    values.extend(pooled)
    return np.asarray(values, dtype=np.float32)


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
        from torch import nn

        schema = str(checkpoint.get("schema") or "")
        if schema != PRE_ASR_CUEQC_SCHEMA:
            raise ValueError(f"unsupported Pre-ASR CueQC schema: {schema!r}")
        metadata = dict(checkpoint.get("metadata") or {})
        if expected_asr_repo_id is not None:
            validate_checkpoint_repo_id(
                str(metadata.get("asr_repo_id") or ""),
                expected_asr_repo_id,
                checkpoint_kind="Pre-ASR CueQC",
                metadata_key="metadata.asr_repo_id",
            )
        feature_names = tuple(str(item) for item in checkpoint.get("feature_names") or ())
        if feature_names != PRE_ASR_CUEQC_FEATURE_NAMES:
            raise ValueError("Pre-ASR CueQC feature_names do not match runtime schema")
        lowered = " ".join(feature_names).lower()
        if any(token in lowered for token in _BANNED_FEATURE_TOKENS):
            raise ValueError("Pre-ASR CueQC feature schema must not contain ASR text/token/decoder fields")

        config = dict(checkpoint.get("model_config") or {})
        input_dim = int(config.get("input_dim") or len(PRE_ASR_CUEQC_FEATURE_NAMES))
        if input_dim != len(PRE_ASR_CUEQC_FEATURE_NAMES):
            raise ValueError("Pre-ASR CueQC model_config.input_dim does not match feature schema")
        hidden_size = int(config.get("hidden_size") or 64)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("Pre-ASR CueQC checkpoint missing state_dict")
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
        norm = dict(checkpoint.get("normalization") or {})
        self.mean = np.asarray(norm.get("mean", [0.0] * input_dim), dtype=np.float32)
        self.std = np.asarray(norm.get("std", [1.0] * input_dim), dtype=np.float32)
        if self.mean.shape[0] != input_dim or self.std.shape[0] != input_dim:
            raise ValueError("Pre-ASR CueQC normalization shape mismatch")
        decision = dict(checkpoint.get("decision_config") or {})
        self.drop_threshold = float(decision.get("drop_threshold", 0.90))

    def signature(self) -> dict[str, Any]:
        return {
            "schema": PRE_ASR_CUEQC_SCHEMA,
            "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
            "path": str(self.path),
            "sha256": self.sha256,
            "drop_threshold": self.drop_threshold,
            "metadata": self.metadata,
        }

    def decide(self, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        import torch

        if not candidates:
            return []
        matrix = np.stack([feature_vector(candidate) for candidate in candidates]).astype(np.float32)
        matrix = (matrix - self.mean) / np.maximum(self.std, 1e-6)
        matrix = np.where(np.isfinite(matrix), matrix, 0.0).astype(np.float32)
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(matrix).to(self.device))
            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
        decisions: list[dict[str, Any]] = []
        for index, prob in enumerate(probs):
            p_drop = float(prob[0])
            p_keep = float(prob[1])
            drop = p_drop >= self.drop_threshold
            decisions.append(
                {
                    "schema": "pre_asr_cueqc_decision_v1",
                    "decision_version": PRE_ASR_CUEQC_DECISION_VERSION,
                    "model_schema": PRE_ASR_CUEQC_SCHEMA,
                    "index": int(candidates[index].get("index", index)),
                    "route": "drop_before_asr" if drop else "keep_for_asr",
                    "confidence": round(p_drop if drop else p_keep, 4),
                    "prob_drop": round(p_drop, 4),
                    "prob_keep": round(p_keep, 4),
                    "drop_threshold": round(self.drop_threshold, 4),
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
            "enabled": False,
            "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        }
    checkpoint = _checkpoint_path()
    return {
        "schema": PRE_ASR_CUEQC_SCHEMA,
        "enabled": True,
        "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
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
