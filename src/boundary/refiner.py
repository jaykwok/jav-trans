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

LEARNED_REFINER_SCHEMA = "boundary_refiner_v4"
DEFAULT_REFINER_CHECKPOINT_PATH = Path("src/boundary/checkpoints/boundary_refiner.pt")
BOUNDARY_REFINER_OUTPUT_DIM = 3
DEFAULT_BOUNDARY_DELTA_MAX_S = 0.5


@dataclass(frozen=True)
class BoundaryDecision:
    merge: bool
    score: float
    reason: str
    source: str = ""
    start_refine_delta_s: float | None = None
    end_refine_delta_s: float | None = None


class SequenceBoundaryRefiner(Protocol):
    def decide_sequence(self, features: Sequence[Sequence[float]]) -> list[BoundaryDecision]: ...

    def signature(self) -> dict: ...


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
        self.boundary_delta_max_s = float(
            self.metadata.get("boundary_delta_max_s", DEFAULT_BOUNDARY_DELTA_MAX_S)
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
            merge_scores = torch.sigmoid(logits[:, 0]).tolist()
            context_logits = logits
        else:
            raise ValueError("frame sequence refiner logits must have shape [batch,time,heads]")
        decisions: list[BoundaryDecision] = []
        for index, score in enumerate(merge_scores):
            start_delta_s, end_delta_s = _boundary_delta_from_logits(
                context_logits[index],
                max_delta_s=self.learned.boundary_delta_max_s,
            )
            decisions.append(
                BoundaryDecision(
                    float(score) >= self.learned.threshold,
                    float(score),
                    "learned_sequence_merge"
                    if float(score) >= self.learned.threshold
                    else "learned_sequence_split",
                    source="frame_sequence_refiner",
                    start_refine_delta_s=start_delta_s,
                    end_refine_delta_s=end_delta_s,
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
        raise ValueError("boundary refiner v4 checkpoint missing feature_names")
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
        raise ValueError("boundary refiner v4 checkpoints must use output_dim=3")
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
        raise ValueError("boundary refiner v4 checkpoints require output_dim=3")
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


def _boundary_delta_from_logits(
    logits: torch.Tensor,
    *,
    max_delta_s: float,
) -> tuple[float | None, float | None]:
    if logits.numel() < BOUNDARY_REFINER_OUTPUT_DIM:
        return None, None
    scale = max(0.0, float(max_delta_s))
    start = float(torch.tanh(logits[1]) * scale)
    end = float(torch.tanh(logits[2]) * scale)
    return start, end
