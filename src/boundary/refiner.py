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
from boundary.sequence_features import (
    FRAME_SEQUENCE_FEATURE_SCHEMA,
    validate_sequence_features,
)

LEARNED_REFINER_SCHEMA = "boundary_refiner_v2"
DEFAULT_REFINER_CHECKPOINT_PATH = Path("src/boundary/checkpoints/boundary_refiner.pt")
CONTEXT_OUTPUT_DIM = 3
DEFAULT_CONTEXT_MAX_PADDING_S = 1.5

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
    source: str = ""
    refine_delta_s: float | None = None
    left_context_s: float | None = None
    right_context_s: float | None = None
    context_source: str = ""


class BoundaryRefiner(Protocol):
    def decide_gap(self, item: RefinerInput) -> BoundaryDecision: ...

    def signature(self) -> dict: ...


class SequenceBoundaryRefiner(Protocol):
    def decide_sequence(self, features: Sequence[Sequence[float]]) -> list[BoundaryDecision]: ...

    def signature(self) -> dict: ...


class HeuristicBoundaryRefiner:
    """Deterministic bootstrap refiner used until a learned checkpoint is available."""

    def __init__(
        self,
        *,
        merge_threshold: float = 0.5,
        max_merge_gap_s: float | None = None,
        target_core_s: float = 3.0,
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
            return BoundaryDecision(False, 0.0, "overlap_gap", source="heuristic_refiner")
        if self.max_merge_gap_s is not None and item.gap_s > self.max_merge_gap_s:
            return BoundaryDecision(False, 0.0, "gap_above_refiner_limit", source="heuristic_refiner")
        if item.cut_score_max is not None and item.cut_score_max >= self.cut_score_threshold:
            return BoundaryDecision(False, 0.0, "cut_score_high", source="heuristic_refiner")
        if item.valley_score_min is not None and item.valley_score_min <= self.valley_score_threshold:
            return BoundaryDecision(False, 0.0, "valley_low", source="heuristic_refiner")

        base = 1.0 - min(1.0, item.gap_s / max(item.gap_merge_s, 1e-6))
        length_penalty = min(0.35, max(0.0, item.proposed_core_s - self.target_core_s) / 30.0)
        score = max(0.0, min(1.0, base - length_penalty))
        return BoundaryDecision(
            score >= self.merge_threshold,
            score,
            "heuristic_score",
            source="heuristic_refiner",
        )


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
        requested_device: str = "auto",
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
        self.requested_device = str(requested_device or "auto")
        self.actual_device = _move_model_to_device(self.model, self.requested_device)
        self.metadata = dict(metadata or {})
        self._sha1 = file_sha1(checkpoint_path)
        self.context_max_padding_s = float(
            self.metadata.get("context_max_padding_s", DEFAULT_CONTEXT_MAX_PADDING_S)
        )

    def signature(self) -> dict:
        return {
            "type": "learned_boundary_refiner",
            "schema": LEARNED_REFINER_SCHEMA,
            "path": str(self.checkpoint_path),
            "sha1": self._sha1,
            "backbone": self.backbone,
            "threshold": self.threshold,
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "feature_names": list(self.feature_names),
            "metadata": self.metadata,
        }

    def decide_gap(self, item: RefinerInput) -> BoundaryDecision:
        if item.gap_s < 0.0:
            return BoundaryDecision(False, 0.0, "overlap_gap", source="learned_refiner")
        vector = torch.tensor(
            [_feature_value(item, name) for name in self.feature_names],
            dtype=torch.float32,
        )
        mean = torch.tensor(self.feature_mean, dtype=torch.float32)
        std = torch.tensor(self.feature_std, dtype=torch.float32).clamp_min(1e-6)
        vector = ((vector - mean) / std).view(1, 1, -1)
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            logits = self.model(vector.to(device)).reshape(-1).detach().cpu()
            score = float(torch.sigmoid(logits[0]))
            left_context_s, right_context_s = _context_from_logits(
                logits,
                max_padding_s=self.context_max_padding_s,
            )
        return BoundaryDecision(
            score >= self.threshold,
            score,
            "learned_merge" if score >= self.threshold else "learned_split",
            source="learned_refiner",
            left_context_s=left_context_s,
            right_context_s=right_context_s,
            context_source="learned_refiner",
        )


class FrameSequenceBoundaryRefiner:
    """Learned refiner adapter for precomputed candidate/window feature sequences."""

    def __init__(self, learned: LearnedBoundaryRefiner) -> None:
        runtime_adapter = str(learned.metadata.get("runtime_adapter") or "")
        if runtime_adapter != "frame_sequence_v1":
            raise ValueError(
                "frame sequence refiner checkpoint metadata.runtime_adapter must be "
                "'frame_sequence_v1'"
            )
        feature_schema = str(learned.metadata.get("feature_schema") or "")
        if feature_schema != FRAME_SEQUENCE_FEATURE_SCHEMA:
            raise ValueError(
                "frame sequence refiner checkpoint metadata.feature_schema must be "
                f"{FRAME_SEQUENCE_FEATURE_SCHEMA!r}"
            )
        if not str(learned.metadata.get("feature_schema_hash") or ""):
            raise ValueError("frame sequence refiner checkpoint missing feature_schema_hash")
        self.learned = learned

    def signature(self) -> dict:
        signature = self.learned.signature()
        signature["runtime_adapter"] = "frame_sequence_v1"
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
        array = validate_sequence_features(
            features,
            feature_names=self.learned.feature_names,
        )
        tensor = torch.tensor(array, dtype=torch.float32)
        mean = torch.tensor(self.learned.feature_mean, dtype=torch.float32)
        std = torch.tensor(self.learned.feature_std, dtype=torch.float32).clamp_min(1e-6)
        tensor = ((tensor - mean) / std).unsqueeze(0)
        device = next(self.learned.model.parameters()).device
        with torch.inference_mode():
            logits = self.learned.model(tensor.to(device)).detach().cpu()
        if logits.ndim == 3:
            merge_scores = torch.sigmoid(logits[0, :, 0]).tolist()
            context_logits = logits[0]
        elif logits.ndim == 2:
            merge_scores = torch.sigmoid(logits.reshape(-1)).tolist()
            context_logits = logits
        else:
            raise ValueError("frame sequence refiner logits must have shape [batch,time,heads]")
        decisions: list[BoundaryDecision] = []
        for index, score in enumerate(merge_scores):
            left_context_s, right_context_s = _context_from_logits(
                context_logits[index],
                max_padding_s=self.learned.context_max_padding_s,
            )
            decisions.append(
                BoundaryDecision(
                    float(score) >= self.learned.threshold,
                    float(score),
                    "learned_sequence_merge"
                    if float(score) >= self.learned.threshold
                    else "learned_sequence_split",
                    source="frame_sequence_refiner",
                    left_context_s=left_context_s,
                    right_context_s=right_context_s,
                    context_source="frame_sequence_refiner",
                )
            )
        return decisions


def load_learned_refiner_checkpoint(
    checkpoint_path: str | Path,
    *,
    threshold: float,
    backbone_override: str | None = None,
    device: str = "auto",
) -> LearnedBoundaryRefiner:
    checkpoint_path = Path(checkpoint_path)
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
    model_config.setdefault("output_dim", CONTEXT_OUTPUT_DIM)
    if int(model_config["input_dim"]) != len(feature_names):
        raise ValueError("checkpoint input_dim does not match feature_names length")
    if int(model_config.get("output_dim", 0)) != CONTEXT_OUTPUT_DIM:
        raise ValueError("boundary refiner v2 checkpoints must use output_dim=3")
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
        requested_device=device,
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def load_frame_sequence_refiner_checkpoint(
    checkpoint_path: str | Path,
    *,
    threshold: float,
    backbone_override: str | None = None,
    device: str = "auto",
) -> FrameSequenceBoundaryRefiner:
    return FrameSequenceBoundaryRefiner(
        load_learned_refiner_checkpoint(
            checkpoint_path,
            threshold=threshold,
            backbone_override=backbone_override,
            device=device,
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
    config["output_dim"] = getattr(model, "output_dim", CONTEXT_OUTPUT_DIM)
    if int(config["output_dim"]) != CONTEXT_OUTPUT_DIM:
        raise ValueError("boundary refiner v2 checkpoints require output_dim=3")
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
    device: str = "auto",
    merge_threshold: float = 0.5,
    max_merge_gap_s: float | None = None,
    target_core_s: float = 3.0,
    cut_score_threshold: float = 0.94,
    valley_score_threshold: float = 0.10,
) -> BoundaryRefiner | None:
    if not enabled:
        return None
    backbone = normalize_boundary_backbone(backbone)
    path = Path(model_path).expanduser() if model_path else None
    if path is not None and str(path) and not path.exists():
        if _is_default_refiner_checkpoint_path(Path(model_path)):
            return HeuristicBoundaryRefiner(
                merge_threshold=merge_threshold,
                max_merge_gap_s=max_merge_gap_s,
                target_core_s=target_core_s,
                cut_score_threshold=cut_score_threshold,
                valley_score_threshold=valley_score_threshold,
            )
        raise FileNotFoundError(f"Boundary refiner checkpoint not found: {path}")
    if path is not None and path.exists():
        return load_learned_refiner_checkpoint(
            path,
            threshold=merge_threshold,
            backbone_override=backbone,
            device=device,
        )
    return HeuristicBoundaryRefiner(
        merge_threshold=merge_threshold,
        max_merge_gap_s=max_merge_gap_s,
        target_core_s=target_core_s,
        cut_score_threshold=cut_score_threshold,
        valley_score_threshold=valley_score_threshold,
    )


def _is_default_refiner_checkpoint_path(path: Path) -> bool:
    try:
        return path.expanduser().resolve(strict=False) == DEFAULT_REFINER_CHECKPOINT_PATH.resolve(strict=False)
    except Exception:
        return str(path).replace("\\", "/").lstrip("./") == str(DEFAULT_REFINER_CHECKPOINT_PATH).replace("\\", "/")


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


def _context_from_logits(
    logits: torch.Tensor,
    *,
    max_padding_s: float,
) -> tuple[float | None, float | None]:
    if logits.numel() < CONTEXT_OUTPUT_DIM:
        return None, None
    scale = max(0.0, float(max_padding_s))
    left = float(torch.sigmoid(logits[1]) * scale)
    right = float(torch.sigmoid(logits[2]) * scale)
    return left, right
