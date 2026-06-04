from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from boundary.backbones import (
    TRANSFORMERS_MAMBA2_BACKBONE,
    BoundarySequenceClassifier,
    normalize_boundary_backbone,
)

LEARNED_REFINER_SCHEMA = "boundary_refiner_v1"

DEFAULT_REFINER_FEATURES = (
    "gap_s",
    "left_duration_s",
    "right_duration_s",
    "current_core_s",
    "proposed_core_s",
    "gap_merge_s",
    "gap_ratio",
    "proposed_over_target_s",
    "left_score",
    "right_score",
    "valley_score_min",
    "cut_score_max",
    "gap_boundary_score",
)


@dataclass(frozen=True)
class RefinerInput:
    gap_s: float
    left_start: float
    left_end: float
    right_start: float
    right_end: float
    current_core_s: float
    proposed_core_s: float
    gap_merge_s: float
    left_score: float | None = None
    right_score: float | None = None
    valley_score_min: float | None = None
    cut_score_max: float | None = None
    gap_boundary_score: float | None = None


@dataclass(frozen=True)
class BoundaryDecision:
    merge: bool
    score: float
    reason: str


class BoundaryRefiner(Protocol):
    def decide_gap(self, item: RefinerInput) -> BoundaryDecision: ...

    def signature(self) -> dict: ...


class HeuristicBoundaryRefiner:
    """Deterministic bootstrap refiner used until a learned checkpoint is available."""

    def __init__(
        self,
        *,
        merge_threshold: float = 0.5,
        max_merge_gap_s: float | None = None,
        target_core_s: float = 9.0,
        cut_score_threshold: float = 0.94,
        valley_score_threshold: float = 0.10,
    ) -> None:
        self.merge_threshold = float(merge_threshold)
        self.max_merge_gap_s = None if max_merge_gap_s is None else float(max_merge_gap_s)
        self.target_core_s = float(target_core_s)
        self.cut_score_threshold = float(cut_score_threshold)
        self.valley_score_threshold = float(valley_score_threshold)

    def signature(self) -> dict:
        return {
            "type": "heuristic_boundary_refiner",
            "merge_threshold": self.merge_threshold,
            "max_merge_gap_s": self.max_merge_gap_s,
            "target_core_s": self.target_core_s,
            "cut_score_threshold": self.cut_score_threshold,
            "valley_score_threshold": self.valley_score_threshold,
        }

    def decide_gap(self, item: RefinerInput) -> BoundaryDecision:
        if item.gap_s < 0.0:
            return BoundaryDecision(False, 0.0, "overlap_gap")
        if self.max_merge_gap_s is not None and item.gap_s > self.max_merge_gap_s:
            return BoundaryDecision(False, 0.0, "gap_above_refiner_limit")
        if item.cut_score_max is not None and item.cut_score_max >= self.cut_score_threshold:
            return BoundaryDecision(False, 0.0, "cut_score_high")
        if item.valley_score_min is not None and item.valley_score_min <= self.valley_score_threshold:
            return BoundaryDecision(False, 0.0, "valley_low")

        base = 1.0 - min(1.0, item.gap_s / max(item.gap_merge_s, 1e-6))
        length_penalty = min(0.35, max(0.0, item.proposed_core_s - self.target_core_s) / 30.0)
        score = max(0.0, min(1.0, base - length_penalty))
        return BoundaryDecision(score >= self.merge_threshold, score, "heuristic_score")


class LearnedBoundaryRefiner:
    """Runtime adapter for learned Boundary Refiner checkpoints."""

    def __init__(
        self,
        *,
        model: BoundarySequenceClassifier,
        checkpoint_path: Path,
        threshold: float,
        feature_names: tuple[str, ...],
        feature_mean: tuple[float, ...],
        feature_std: tuple[float, ...],
        backbone: str,
        metadata: dict | None = None,
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        if len(feature_mean) != len(feature_names):
            raise ValueError("feature_mean length must match feature_names")
        if len(feature_std) != len(feature_names):
            raise ValueError("feature_std length must match feature_names")
        self.model = model.eval()
        self.checkpoint_path = checkpoint_path
        self.threshold = float(threshold)
        self.feature_names = feature_names
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.backbone = normalize_boundary_backbone(backbone)
        self.metadata = dict(metadata or {})
        self._sha1 = file_sha1(checkpoint_path)

    def signature(self) -> dict:
        return {
            "type": "learned_boundary_refiner",
            "schema": LEARNED_REFINER_SCHEMA,
            "path": str(self.checkpoint_path),
            "sha1": self._sha1,
            "backbone": self.backbone,
            "threshold": self.threshold,
            "feature_names": list(self.feature_names),
            "metadata": self.metadata,
        }

    def decide_gap(self, item: RefinerInput) -> BoundaryDecision:
        if item.gap_s < 0.0:
            return BoundaryDecision(False, 0.0, "overlap_gap")
        vector = torch.tensor(
            [_feature_value(item, name) for name in self.feature_names],
            dtype=torch.float32,
        )
        mean = torch.tensor(self.feature_mean, dtype=torch.float32)
        std = torch.tensor(self.feature_std, dtype=torch.float32).clamp_min(1e-6)
        vector = ((vector - mean) / std).view(1, 1, -1)
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            logit = self.model(vector.to(device)).reshape(-1)[0]
            score = float(torch.sigmoid(logit).detach().cpu())
        return BoundaryDecision(
            score >= self.threshold,
            score,
            "learned_merge" if score >= self.threshold else "learned_split",
        )


def load_learned_refiner_checkpoint(
    checkpoint_path: Path,
    *,
    threshold: float,
    backbone_override: str | None = None,
) -> LearnedBoundaryRefiner:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Boundary refiner checkpoint must be a dict")
    schema = payload.get("schema")
    if schema != LEARNED_REFINER_SCHEMA:
        raise ValueError(
            f"unsupported boundary refiner checkpoint schema: {schema!r}; "
            f"expected {LEARNED_REFINER_SCHEMA!r}"
        )
    feature_names = tuple(str(name) for name in payload.get("feature_names") or ())
    if not feature_names:
        feature_names = DEFAULT_REFINER_FEATURES
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
    if int(model_config["input_dim"]) != len(feature_names):
        raise ValueError("checkpoint input_dim does not match feature_names length")
    model = BoundarySequenceClassifier(**model_config)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Boundary refiner checkpoint missing state_dict")
    model.load_state_dict(state_dict)
    metadata = payload.get("metadata")
    return LearnedBoundaryRefiner(
        model=model,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        feature_names=feature_names,
        feature_mean=_float_tuple(payload.get("feature_mean"), len(feature_names), 0.0),
        feature_std=_float_tuple(payload.get("feature_std"), len(feature_names), 1.0),
        backbone=backbone,
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def build_learned_refiner_checkpoint(
    *,
    model: BoundarySequenceClassifier,
    feature_names: tuple[str, ...] = DEFAULT_REFINER_FEATURES,
    feature_mean: tuple[float, ...] | None = None,
    feature_std: tuple[float, ...] | None = None,
    model_config: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    config = dict(model.model_config)
    config.update(model_config or {})
    config["input_dim"] = len(feature_names)
    config["backbone"] = model.backbone_name
    return {
        "schema": LEARNED_REFINER_SCHEMA,
        "backbone": model.backbone_name,
        "model_config": config,
        "feature_names": list(feature_names),
        "feature_mean": list(feature_mean or (0.0,) * len(feature_names)),
        "feature_std": list(feature_std or (1.0,) * len(feature_names)),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }


def load_boundary_refiner(
    *,
    enabled: bool,
    model_path: str = "",
    backbone: str = TRANSFORMERS_MAMBA2_BACKBONE,
    merge_threshold: float = 0.5,
    max_merge_gap_s: float | None = None,
    target_core_s: float = 9.0,
    cut_score_threshold: float = 0.94,
    valley_score_threshold: float = 0.10,
) -> BoundaryRefiner | None:
    if not enabled:
        return None
    backbone = normalize_boundary_backbone(backbone)
    path = Path(model_path).expanduser() if model_path else None
    if path is not None and str(path) and not path.exists():
        raise FileNotFoundError(f"Boundary refiner checkpoint not found: {path}")
    if path is not None and path.exists():
        return load_learned_refiner_checkpoint(
            path,
            threshold=merge_threshold,
            backbone_override=backbone,
        )
    return HeuristicBoundaryRefiner(
        merge_threshold=merge_threshold,
        max_merge_gap_s=max_merge_gap_s,
        target_core_s=target_core_s,
        cut_score_threshold=cut_score_threshold,
        valley_score_threshold=valley_score_threshold,
    )


def refiner_input_to_features(
    item: RefinerInput,
    feature_names: tuple[str, ...] = DEFAULT_REFINER_FEATURES,
) -> list[float]:
    return [_feature_value(item, name) for name in feature_names]


def file_sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _feature_value(item: RefinerInput, name: str) -> float:
    if name == "gap_s":
        return float(item.gap_s)
    if name == "left_duration_s":
        return float(max(0.0, item.left_end - item.left_start))
    if name == "right_duration_s":
        return float(max(0.0, item.right_end - item.right_start))
    if name == "current_core_s":
        return float(item.current_core_s)
    if name == "proposed_core_s":
        return float(item.proposed_core_s)
    if name == "gap_merge_s":
        return float(item.gap_merge_s)
    if name == "gap_ratio":
        return float(item.gap_s / max(item.gap_merge_s, 1e-6))
    if name == "proposed_over_target_s":
        return float(item.proposed_core_s - item.current_core_s)
    if name == "left_score":
        return _none_to_default(item.left_score, 0.0)
    if name == "right_score":
        return _none_to_default(item.right_score, 0.0)
    if name == "valley_score_min":
        return _none_to_default(item.valley_score_min, 1.0)
    if name == "cut_score_max":
        return _none_to_default(item.cut_score_max, 0.0)
    if name == "gap_boundary_score":
        return _none_to_default(item.gap_boundary_score, 0.0)
    raise ValueError(f"unknown boundary refiner feature: {name}")


def _none_to_default(value: float | None, default: float) -> float:
    return float(default if value is None else value)


def _float_tuple(values: object, expected_len: int, default: float) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        return (float(default),) * expected_len
    result = tuple(float(value) for value in values)
    if len(result) != expected_len:
        raise ValueError("checkpoint normalization length does not match feature_names")
    return result
