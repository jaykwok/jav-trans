from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_FRAME_SCORER_SCHEMA = "speech_boundary_ja_feature_scorer_v1"


def count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class FeatureFrameScorer:
    """Runtime-loadable frame scorer over normalized PTM+MFCC features."""

    def __new__(
        cls,
        *,
        input_dim: int,
        hidden_size: int = 128,
        dropout: float = 0.05,
    ):
        from torch import nn

        input_dim = int(input_dim)
        hidden_size = int(hidden_size)
        dropout = float(dropout)
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")

        class _FeatureFrameScorer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = input_dim
                self.hidden_size = hidden_size
                self.dropout = dropout
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, max(16, hidden_size // 2)),
                    nn.GELU(),
                    nn.Linear(max(16, hidden_size // 2), 1),
                )

            def forward(self, features):
                if features.ndim != 3:
                    raise ValueError(f"features must be [batch, frames, dim], got shape={tuple(features.shape)}")
                if int(features.shape[-1]) != self.input_dim:
                    raise ValueError(f"expected input_dim={self.input_dim}, got {int(features.shape[-1])}")
                return self.net(features).squeeze(-1)

        return _FeatureFrameScorer()


class TinyFrameClassifier:
    """Small convolutional smoke model for validating label/audio plumbing."""

    def __new__(cls):
        from torch import nn

        class _TinyFrameClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(1, 8, kernel_size=31, stride=4, padding=15),
                    nn.ReLU(),
                    nn.Conv1d(8, 16, kernel_size=15, stride=4, padding=7),
                    nn.ReLU(),
                    nn.Conv1d(16, 1, kernel_size=5, padding=2),
                )

            def forward(self, audio, target_frames: int):
                import torch.nn.functional as F

                logits = self.net(audio).squeeze(1)
                return F.interpolate(
                    logits.unsqueeze(1),
                    size=target_frames,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)

        return _TinyFrameClassifier()


@dataclass(frozen=True)
class FeatureFrameScorerBundle:
    path: str
    sha256: str
    model: Any
    model_config: dict[str, Any]
    normalization: dict[str, Any]
    metadata: dict[str, Any]
    device: str

    @property
    def ptm_dim(self) -> int:
        return int(self.model_config["ptm_dim"])

    @property
    def mfcc_dim(self) -> int:
        return int(self.model_config["mfcc_dim"])

    @property
    def input_dim(self) -> int:
        return int(self.model_config["input_dim"])

    def signature(self) -> dict[str, Any]:
        return {
            "schema": FEATURE_FRAME_SCORER_SCHEMA,
            "path": self.path,
            "sha256": self.sha256,
            "model_config": {
                "ptm_dim": self.ptm_dim,
                "mfcc_dim": self.mfcc_dim,
                "input_dim": self.input_dim,
                "hidden_size": int(self.model_config["hidden_size"]),
                "dropout": float(self.model_config["dropout"]),
            },
            "metadata": {
                "operating_point": str(self.metadata.get("operating_point") or ""),
                "trained_steps": int(self.metadata.get("trained_steps") or 0),
                "labels": str(self.metadata.get("labels") or ""),
                "feature_manifest": str(self.metadata.get("feature_manifest") or ""),
            },
        }


def checkpoint_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_feature_frame_scorer_checkpoint(
    *,
    model: Any,
    model_config: dict[str, Any],
    normalization: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": FEATURE_FRAME_SCORER_SCHEMA,
        "model_type": "feature_frame_scorer",
        "model_config": dict(model_config),
        "normalization": dict(normalization),
        "metadata": dict(metadata or {}),
        "model_state_dict": model.state_dict(),
    }


def load_feature_frame_scorer_checkpoint(
    path: str | Path,
    *,
    device: str | Any = "cpu",
) -> FeatureFrameScorerBundle:
    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SpeechBoundary-JA scorer checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility
        payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid scorer checkpoint payload: {checkpoint_path}")
    if payload.get("schema") != FEATURE_FRAME_SCORER_SCHEMA:
        raise ValueError(
            f"unsupported scorer checkpoint schema: {payload.get('schema')!r}; "
            f"expected {FEATURE_FRAME_SCORER_SCHEMA!r}"
        )
    model_config = dict(payload.get("model_config") or {})
    for key in ("ptm_dim", "mfcc_dim", "input_dim", "hidden_size", "dropout"):
        if key not in model_config:
            raise ValueError(f"scorer checkpoint missing model_config.{key}")
    if int(model_config["input_dim"]) != int(model_config["ptm_dim"]) + int(model_config["mfcc_dim"]):
        raise ValueError("scorer checkpoint input_dim does not match ptm_dim + mfcc_dim")
    normalization = dict(payload.get("normalization") or {})
    mean = list(normalization.get("feature_mean") or [])
    std = list(normalization.get("feature_std") or [])
    if len(mean) != int(model_config["input_dim"]) or len(std) != int(model_config["input_dim"]):
        raise ValueError("scorer checkpoint normalization length does not match input_dim")
    model = FeatureFrameScorer(
        input_dim=int(model_config["input_dim"]),
        hidden_size=int(model_config["hidden_size"]),
        dropout=float(model_config["dropout"]),
    )
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("scorer checkpoint missing model_state_dict")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return FeatureFrameScorerBundle(
        path=str(checkpoint_path),
        sha256=checkpoint_sha256(checkpoint_path),
        model=model,
        model_config=model_config,
        normalization=normalization,
        metadata=dict(payload.get("metadata") or {}),
        device=str(device),
    )


def score_feature_frame_probabilities(
    bundle: FeatureFrameScorerBundle,
    *,
    ptm: np.ndarray,
    mfcc: np.ndarray,
) -> np.ndarray:
    import torch

    frame_total = min(int(ptm.shape[0]), int(mfcc.shape[0]))
    if frame_total <= 0:
        return np.zeros(0, dtype=np.float32)
    if int(ptm.shape[1]) != bundle.ptm_dim:
        raise ValueError(f"scorer expected ptm_dim={bundle.ptm_dim}, got {int(ptm.shape[1])}")
    if int(mfcc.shape[1]) != bundle.mfcc_dim:
        raise ValueError(f"scorer expected mfcc_dim={bundle.mfcc_dim}, got {int(mfcc.shape[1])}")
    features = np.concatenate(
        [
            np.asarray(ptm[:frame_total], dtype=np.float32),
            np.asarray(mfcc[:frame_total], dtype=np.float32),
        ],
        axis=1,
    )
    mean = np.asarray(bundle.normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(bundle.normalization["feature_std"], dtype=np.float32)
    features = np.ascontiguousarray((features - mean) / np.maximum(std, 1e-6), dtype=np.float32)
    with torch.inference_mode():
        tensor = torch.from_numpy(features).to(bundle.device).unsqueeze(0)
        logits = bundle.model(tensor).squeeze(0)
        return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
