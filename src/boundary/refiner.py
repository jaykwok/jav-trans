from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch

from asr.backends.qwen import validate_checkpoint_repo_id
from boundary.backbones import (
    TRANSFORMERS_MAMBA2_BACKBONE,
    BoundarySequenceClassifier,
    normalize_boundary_backbone,
)
from boundary.sequence_features import (
    FRAME_SEQUENCE_FEATURE_SCHEMA,
    validate_sequence_features,
)

LEARNED_REFINER_SCHEMA = "boundary_edge_refiner_v8_safe_tight"
BOUNDARY_REFINER_MODEL_ARCH = "boundary_edge_refiner_v8_safe_tight"
EDGE_SEQUENCE_RUNTIME_ADAPTER = "edge_sequence_v2"
DEFAULT_REFINER_CHECKPOINT_PATH = Path(
    "src/boundary/checkpoints/boundary_edge_refiner_v8_safe_tight.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt"
)
BOUNDARY_REFINER_OUTPUT_DIM = 4
DEFAULT_BOUNDARY_DELTA_MAX_S = 0.5
DEFAULT_REFINER_MAX_DELTA_RATIO = 0.35
DEFAULT_REFINER_MIN_ALLOWED_DURATION_S = 0.25
DEFAULT_REFINER_SAFETY_MARGIN_S = 0.05
DEFAULT_REFINER_HIGH_CONFIDENCE = 0.70
DEFAULT_REFINER_MEDIUM_CONFIDENCE = 0.40
DEFAULT_REFINER_MEDIUM_SCALE = 0.50


@dataclass(frozen=True)
class BoundaryDecision:
    source: str = ""
    start_refine_delta_s: float | None = None
    end_refine_delta_s: float | None = None
    raw_start_refine_delta_s: float | None = None
    raw_end_refine_delta_s: float | None = None
    start_confidence: float | None = None
    end_confidence: float | None = None
    start_source: str = ""
    end_source: str = ""
    refiner_safety_action: str = ""
    refiner_safety_reason: str = ""
    effective_start_delta_max_s: float | None = None
    effective_end_delta_max_s: float | None = None
    fallback_used: bool = False
    shared_boundary_adjusted: bool = False
    shared_cut_id: str = ""


class SequenceBoundaryRefiner(Protocol):
    def decide_sequence(self, features: Sequence[Sequence[float]]) -> list[BoundaryDecision]: ...

    def signature(self) -> dict: ...


class LearnedBoundaryRefiner:
    """Runtime adapter for learned edge-only Boundary Refiner checkpoints."""

    def __init__(
        self,
        *,
        model: BoundarySequenceClassifier,
        checkpoint_path: Path,
        feature_names: tuple[str, ...],
        feature_mean: tuple[float, ...],
        feature_std: tuple[float, ...],
        backbone: str,
        metadata: dict | None = None,
        requested_device: str = "auto",
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        if len(feature_mean) != len(feature_names):
            raise ValueError("feature_mean length must match feature_names")
        if len(feature_std) != len(feature_names):
            raise ValueError("feature_std length must match feature_names")
        self.model = model.eval()
        self.checkpoint_path = checkpoint_path
        self.feature_names = feature_names
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.backbone = normalize_boundary_backbone(backbone)
        self.requested_device = str(requested_device or "auto")
        self.actual_device = _move_model_to_device(self.model, self.requested_device)
        self.metadata = dict(metadata or {})
        self._sha1 = file_sha1(checkpoint_path)
        self.boundary_delta_max_s = float(
            self.metadata.get("boundary_delta_max_s", DEFAULT_BOUNDARY_DELTA_MAX_S)
        )
        self.max_delta_ratio = float(
            self.metadata.get("max_delta_ratio", DEFAULT_REFINER_MAX_DELTA_RATIO)
        )
        self.min_allowed_duration_s = float(
            self.metadata.get(
                "min_allowed_duration_s",
                DEFAULT_REFINER_MIN_ALLOWED_DURATION_S,
            )
        )
        self.safety_margin_s = float(
            self.metadata.get("safety_margin_s", DEFAULT_REFINER_SAFETY_MARGIN_S)
        )
        confidence_policy = self.metadata.get("confidence")
        if not isinstance(confidence_policy, dict):
            confidence_policy = {}
        self.high_confidence_threshold = float(
            confidence_policy.get("high_threshold", DEFAULT_REFINER_HIGH_CONFIDENCE)
        )
        self.medium_confidence_threshold = float(
            confidence_policy.get("medium_threshold", DEFAULT_REFINER_MEDIUM_CONFIDENCE)
        )
        self.medium_confidence_scale = float(
            confidence_policy.get("medium_scale", DEFAULT_REFINER_MEDIUM_SCALE)
        )

    def signature(self) -> dict:
        return {
            "type": "learned_boundary_edge_refiner",
            "schema": LEARNED_REFINER_SCHEMA,
            "path": str(self.checkpoint_path),
            "sha1": self._sha1,
            "backbone": self.backbone,
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "feature_names": list(self.feature_names),
            "metadata": self.metadata,
        }


class EdgeSequenceBoundaryRefiner:
    """Learned edge-only refiner for scorer-produced island boundaries."""

    def __init__(self, learned: LearnedBoundaryRefiner) -> None:
        runtime_adapter = str(learned.metadata.get("runtime_adapter") or "")
        if runtime_adapter != EDGE_SEQUENCE_RUNTIME_ADAPTER:
            raise ValueError(
                "Boundary Refiner v8 checkpoint metadata.runtime_adapter must be "
                f"{EDGE_SEQUENCE_RUNTIME_ADAPTER!r}"
            )
        feature_schema = str(learned.metadata.get("feature_schema") or "")
        if feature_schema != FRAME_SEQUENCE_FEATURE_SCHEMA:
            raise ValueError(
                "Boundary Refiner v8 checkpoint metadata.feature_schema must be "
                f"{FRAME_SEQUENCE_FEATURE_SCHEMA!r}"
            )
        if not str(learned.metadata.get("feature_schema_hash") or ""):
            raise ValueError("Boundary Refiner v8 checkpoint missing feature_schema_hash")
        self.learned = learned

    def signature(self) -> dict:
        signature = self.learned.signature()
        signature["runtime_adapter"] = EDGE_SEQUENCE_RUNTIME_ADAPTER
        return signature

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.learned.feature_names

    @property
    def feature_schema_hash(self) -> str:
        return str(self.learned.metadata["feature_schema_hash"])

    def decide_sequence(self, features: Sequence[Sequence[float]]) -> list[BoundaryDecision]:
        if not features:
            return []
        raw_array = validate_sequence_features(
            features,
            feature_names=self.learned.feature_names,
        )
        tensor = torch.tensor(raw_array, dtype=torch.float32)
        mean = torch.tensor(self.learned.feature_mean, dtype=torch.float32)
        std = torch.tensor(self.learned.feature_std, dtype=torch.float32).clamp_min(1e-6)
        tensor = ((tensor - mean) / std).unsqueeze(0)
        device = next(self.learned.model.parameters()).device
        with torch.inference_mode():
            logits = self.learned.model(tensor.to(device)).detach().cpu()
        if logits.ndim == 3:
            delta_logits = logits[0]
        elif logits.ndim == 2:
            delta_logits = logits
        else:
            raise ValueError("edge sequence refiner logits must have shape [batch,time,heads]")
        decisions: list[BoundaryDecision] = []
        for index in range(delta_logits.shape[0]):
            decisions.append(
                _boundary_decision_from_logits(
                    delta_logits[index],
                    raw_array[index],
                    feature_names=self.learned.feature_names,
                    learned=self.learned,
                )
            )
        return decisions


def load_learned_refiner_checkpoint(
    checkpoint_path: str | Path,
    *,
    backbone_override: str | None = None,
    device: str = "auto",
    expected_ptm_repo_id: str | None = None,
) -> LearnedBoundaryRefiner:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Boundary Refiner checkpoint must be a dict")
    schema = payload.get("schema")
    if schema != LEARNED_REFINER_SCHEMA:
        raise ValueError(
            f"unsupported boundary refiner checkpoint schema: {schema!r}; "
            f"expected {LEARNED_REFINER_SCHEMA!r}"
        )
    feature_names = tuple(str(name) for name in payload.get("feature_names") or ())
    if not feature_names:
        raise ValueError("Boundary Refiner v8 checkpoint missing feature_names")
    model_config = dict(payload.get("model_config") or {})
    backbone = normalize_boundary_backbone(
        backbone_override
        or str(
            payload.get("backbone")
            or model_config.get("backbone")
            or TRANSFORMERS_MAMBA2_BACKBONE
        )
    )
    model_config["backbone"] = backbone
    model_config.setdefault("input_dim", len(feature_names))
    model_config.setdefault("output_dim", BOUNDARY_REFINER_OUTPUT_DIM)
    if int(model_config["input_dim"]) != len(feature_names):
        raise ValueError("checkpoint input_dim does not match feature_names length")
    if int(model_config.get("output_dim", 0)) != BOUNDARY_REFINER_OUTPUT_DIM:
        raise ValueError("Boundary Refiner v8 checkpoints must use output_dim=4")
    model = BoundarySequenceClassifier(**model_config)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Boundary Refiner checkpoint missing state_dict")
    model.load_state_dict(state_dict)
    metadata = payload.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    _validate_top_level_feature_metadata(payload, metadata_dict)
    if expected_ptm_repo_id is not None:
        validate_checkpoint_repo_id(
            metadata_dict.get("ptm_repo_id"),
            expected_ptm_repo_id,
            checkpoint_kind="Boundary Refiner",
            metadata_key="metadata.ptm_repo_id",
        )
    return LearnedBoundaryRefiner(
        model=model,
        checkpoint_path=checkpoint_path,
        feature_names=feature_names,
        feature_mean=_float_tuple(payload.get("feature_mean"), len(feature_names), 0.0),
        feature_std=_float_tuple(payload.get("feature_std"), len(feature_names), 1.0),
        backbone=backbone,
        requested_device=device,
        metadata=metadata_dict,
    )


def load_edge_sequence_refiner_v8_checkpoint(
    checkpoint_path: str | Path,
    *,
    backbone_override: str | None = None,
    device: str = "auto",
    expected_ptm_repo_id: str | None = None,
) -> EdgeSequenceBoundaryRefiner:
    return EdgeSequenceBoundaryRefiner(
        load_learned_refiner_checkpoint(
            checkpoint_path,
            backbone_override=backbone_override,
            device=device,
            expected_ptm_repo_id=expected_ptm_repo_id,
        )
    )


def _move_model_to_device(model: torch.nn.Module, requested_device: str) -> str:
    requested = str(requested_device or "auto").strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    device = torch.device(requested)
    model.to(device)
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return str(device)


def build_learned_refiner_checkpoint(
    *,
    model: BoundarySequenceClassifier,
    feature_names: tuple[str, ...],
    feature_mean: tuple[float, ...] | None = None,
    feature_std: tuple[float, ...] | None = None,
    model_config: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    config = dict(model.model_config)
    config.update(model_config or {})
    config["input_dim"] = len(feature_names)
    config["backbone"] = model.backbone_name
    config["output_dim"] = getattr(model, "output_dim", BOUNDARY_REFINER_OUTPUT_DIM)
    if int(config["output_dim"]) != BOUNDARY_REFINER_OUTPUT_DIM:
        raise ValueError("Boundary Refiner v8 checkpoints require output_dim=4")
    metadata_dict = dict(metadata or {})
    metadata_dict.setdefault("model_arch", BOUNDARY_REFINER_MODEL_ARCH)
    metadata_dict.setdefault("runtime_adapter", EDGE_SEQUENCE_RUNTIME_ADAPTER)
    if metadata_dict.get("runtime_adapter") != EDGE_SEQUENCE_RUNTIME_ADAPTER:
        raise ValueError(
            "Boundary Refiner v8 metadata.runtime_adapter must be "
            f"{EDGE_SEQUENCE_RUNTIME_ADAPTER!r}"
        )
    payload = {
        "schema": LEARNED_REFINER_SCHEMA,
        "backbone": model.backbone_name,
        "model_config": config,
        "feature_names": list(feature_names),
        "feature_mean": list(feature_mean or (0.0,) * len(feature_names)),
        "feature_std": list(feature_std or (1.0,) * len(feature_names)),
        "state_dict": model.state_dict(),
        "metadata": metadata_dict,
    }
    for key in ("feature_schema", "feature_schema_hash", "feature_signature"):
        if key in metadata_dict:
            payload[key] = metadata_dict[key]
    return payload


def _validate_top_level_feature_metadata(payload: dict, metadata: dict) -> None:
    for key in ("feature_schema", "feature_schema_hash", "feature_signature"):
        top_value = payload.get(key)
        metadata_value = metadata.get(key)
        if top_value is not None and metadata_value is not None and top_value != metadata_value:
            raise ValueError(f"checkpoint top-level {key} does not match metadata.{key}")


def file_sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _float_tuple(values: object, expected_len: int, default: float) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        return (float(default),) * expected_len
    result = tuple(float(value) for value in values)
    if len(result) != expected_len:
        raise ValueError("checkpoint normalization length does not match feature_names")
    return result


def _boundary_decision_from_logits(
    logits: torch.Tensor,
    raw_features: Sequence[float],
    *,
    feature_names: Sequence[str],
    learned: LearnedBoundaryRefiner,
) -> BoundaryDecision:
    if logits.numel() < BOUNDARY_REFINER_OUTPUT_DIM:
        return BoundaryDecision(source="edge_sequence_refiner_v8")
    left_duration_s = _feature_value(raw_features, feature_names, "left_duration_s")
    right_duration_s = _feature_value(raw_features, feature_names, "right_duration_s")
    effective_start_max_s = _effective_delta_max_s(
        duration_s=right_duration_s,
        boundary_delta_max_s=learned.boundary_delta_max_s,
        max_delta_ratio=learned.max_delta_ratio,
        min_allowed_duration_s=learned.min_allowed_duration_s,
        safety_margin_s=learned.safety_margin_s,
    )
    effective_end_max_s = _effective_delta_max_s(
        duration_s=left_duration_s,
        boundary_delta_max_s=learned.boundary_delta_max_s,
        max_delta_ratio=learned.max_delta_ratio,
        min_allowed_duration_s=learned.min_allowed_duration_s,
        safety_margin_s=learned.safety_margin_s,
    )
    raw_start = float(torch.tanh(logits[0]) * effective_start_max_s)
    raw_end = float(torch.tanh(logits[1]) * effective_end_max_s)
    start_confidence = float(torch.sigmoid(logits[2]))
    end_confidence = float(torch.sigmoid(logits[3]))
    start_delta, start_source = _apply_confidence_policy(
        raw_start,
        confidence=start_confidence,
        high_threshold=learned.high_confidence_threshold,
        medium_threshold=learned.medium_confidence_threshold,
        medium_scale=learned.medium_confidence_scale,
    )
    end_delta, end_source = _apply_confidence_policy(
        raw_end,
        confidence=end_confidence,
        high_threshold=learned.high_confidence_threshold,
        medium_threshold=learned.medium_confidence_threshold,
        medium_scale=learned.medium_confidence_scale,
    )
    return BoundaryDecision(
        source="edge_sequence_refiner_v8",
        start_refine_delta_s=start_delta,
        end_refine_delta_s=end_delta,
        raw_start_refine_delta_s=raw_start,
        raw_end_refine_delta_s=raw_end,
        start_confidence=start_confidence,
        end_confidence=end_confidence,
        start_source=start_source,
        end_source=end_source,
        effective_start_delta_max_s=effective_start_max_s,
        effective_end_delta_max_s=effective_end_max_s,
        fallback_used=False,
    )


def _feature_value(
    values: Sequence[float],
    feature_names: Sequence[str],
    name: str,
) -> float:
    try:
        index = tuple(feature_names).index(name)
    except ValueError:
        return 0.0
    try:
        return max(0.0, float(values[index]))
    except (IndexError, TypeError, ValueError):
        return 0.0


def _effective_delta_max_s(
    *,
    duration_s: float,
    boundary_delta_max_s: float,
    max_delta_ratio: float,
    min_allowed_duration_s: float,
    safety_margin_s: float,
) -> float:
    duration = max(0.0, float(duration_s))
    boundary_limit = max(0.0, float(boundary_delta_max_s))
    ratio_limit = duration * max(0.0, float(max_delta_ratio))
    duration_limit = max(0.0, duration - max(0.0, float(min_allowed_duration_s))) / 2.0
    duration_limit += max(0.0, float(safety_margin_s))
    return max(0.0, min(boundary_limit, ratio_limit, duration_limit))


def _apply_confidence_policy(
    raw_delta_s: float,
    *,
    confidence: float,
    high_threshold: float,
    medium_threshold: float,
    medium_scale: float,
) -> tuple[float, str]:
    conf = float(confidence)
    if conf >= float(high_threshold):
        return float(raw_delta_s), "model"
    if conf >= float(medium_threshold):
        return float(raw_delta_s) * max(0.0, float(medium_scale)), "model_scaled"
    return 0.0, "noop_low_confidence"
