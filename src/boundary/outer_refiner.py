from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asr.backends.qwen import validate_checkpoint_repo_id
from boundary.backbones import Mamba2TemporalEncoder


OUTER_EDGE_REFINER_SCHEMA = "outer_edge_refiner_v1"
OUTER_EDGE_REFINER_MODEL_ARCH = "speech_core_outer_edges_mamba_v1"
OUTER_EDGE_REFINER_RUNTIME_ADAPTER = "speech_island_outer_edges_v1"
OUTER_EDGE_FEATURE_SCHEMA = "speech_island_outer_edge_features_v1"
OUTER_EDGE_REFINER_ARTIFACT = {
    "name": "outer_edge_refiner",
    "display_name": "Outer Edge Refiner",
    "version": "v1",
    "pipeline_stage": 2,
    "pipeline_role": "speech_island_outer_edge_refinement",
}


class OuterEdgeRefinerNetwork:
    def __new__(
        cls,
        *,
        frame_dim: int,
        scalar_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        state_size: int = 32,
        num_heads: int = 4,
        head_dim: int = 64,
        n_groups: int = 2,
        conv_kernel: int = 4,
        chunk_size: int = 8,
        bidirectional: bool = True,
    ):
        import torch
        from torch import nn

        class _Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.frame_proj = nn.Linear(frame_dim, hidden_size)
                self.encoder = Mamba2TemporalEncoder(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    state_size=state_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    n_groups=n_groups,
                    conv_kernel=conv_kernel,
                    chunk_size=chunk_size,
                    bidirectional=bidirectional,
                )
                self.scalar_arm = nn.Sequential(
                    nn.LayerNorm(scalar_dim),
                    nn.Linear(scalar_dim, hidden_size),
                    nn.GELU(),
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(self.encoder.output_dim + hidden_size),
                    nn.Linear(self.encoder.output_dim + hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, 4),
                )

            def forward(self, frame_features, scalar_features):
                encoded = self.encoder(self.frame_proj(frame_features)).mean(dim=1)
                return self.head(
                    torch.cat((encoded, self.scalar_arm(scalar_features)), dim=-1)
                )

        return _Network()


@dataclass(frozen=True)
class OuterEdgePrediction:
    start_delta_s: float
    end_delta_s: float
    start_confidence: float
    end_confidence: float


@dataclass(frozen=True)
class OuterEdgeRefiner:
    path: str
    sha256: str
    model: Any
    model_config: dict[str, Any]
    feature_config: dict[str, Any]
    normalization: dict[str, Any]
    metadata: dict[str, Any]
    device: str

    def signature(self) -> dict[str, Any]:
        return {
            "schema": OUTER_EDGE_REFINER_SCHEMA,
            "model_arch": OUTER_EDGE_REFINER_MODEL_ARCH,
            "runtime_adapter": OUTER_EDGE_REFINER_RUNTIME_ADAPTER,
            "path": self.path,
            "sha256": self.sha256,
            "feature_config": self.feature_config,
            "metadata": self.metadata,
        }

    def predict(
        self,
        *,
        frame_features: np.ndarray,
        scalar_features: np.ndarray,
    ) -> list[OuterEdgePrediction]:
        import torch

        frame_mean = np.asarray(self.normalization["frame_mean"], dtype=np.float32)
        frame_std = np.asarray(self.normalization["frame_std"], dtype=np.float32)
        scalar_mean = np.asarray(self.normalization["scalar_mean"], dtype=np.float32)
        scalar_std = np.asarray(self.normalization["scalar_std"], dtype=np.float32)
        frames = (np.asarray(frame_features, dtype=np.float32) - frame_mean) / np.maximum(
            frame_std, 1e-6
        )
        scalars = (np.asarray(scalar_features, dtype=np.float32) - scalar_mean) / np.maximum(
            scalar_std, 1e-6
        )
        with torch.inference_mode():
            output = self.model(
                torch.from_numpy(frames).to(self.device),
                torch.from_numpy(scalars).to(self.device),
            ).detach().cpu().numpy()
        max_delta = float(self.metadata.get("max_delta_s", 0.5))
        return [
            OuterEdgePrediction(
                start_delta_s=float(np.clip(row[0], -max_delta, max_delta)),
                end_delta_s=float(np.clip(row[1], -max_delta, max_delta)),
                start_confidence=float(1.0 / (1.0 + np.exp(-row[2]))),
                end_confidence=float(1.0 / (1.0 + np.exp(-row[3]))),
            )
            for row in output
        ]


def build_outer_edge_refiner_checkpoint(
    *,
    model: Any,
    model_config: dict[str, Any],
    feature_config: dict[str, Any],
    normalization: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    artifact = {
        **OUTER_EDGE_REFINER_ARTIFACT,
        **dict(metadata.get("artifact") or {}),
    }
    return {
        "schema": OUTER_EDGE_REFINER_SCHEMA,
        "model_arch": OUTER_EDGE_REFINER_MODEL_ARCH,
        "model_config": dict(model_config),
        "feature_config": {**feature_config, "schema": OUTER_EDGE_FEATURE_SCHEMA},
        "normalization": dict(normalization),
        "metadata": {
            **metadata,
            "artifact": artifact,
            "runtime_adapter": OUTER_EDGE_REFINER_RUNTIME_ADAPTER,
        },
        "model_state_dict": model.state_dict(),
    }


def load_outer_edge_refiner(
    path: str | Path,
    *,
    device: str = "auto",
    expected_ptm_repo_id: str | None = None,
) -> OuterEdgeRefiner:
    import torch

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != OUTER_EDGE_REFINER_SCHEMA:
        raise ValueError(
            f"unsupported Outer Edge Refiner schema: {payload.get('schema')!r}; "
            f"expected {OUTER_EDGE_REFINER_SCHEMA!r}"
        )
    if payload.get("model_arch") != OUTER_EDGE_REFINER_MODEL_ARCH:
        raise ValueError(f"Outer Edge Refiner must use {OUTER_EDGE_REFINER_MODEL_ARCH!r}")
    metadata = dict(payload.get("metadata") or {})
    artifact = metadata.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Outer Edge Refiner metadata.artifact is required")
    for key, expected in OUTER_EDGE_REFINER_ARTIFACT.items():
        if artifact.get(key) != expected:
            raise ValueError(
                f"Outer Edge Refiner metadata.artifact.{key} must be {expected!r}"
            )
    if metadata.get("runtime_adapter") != OUTER_EDGE_REFINER_RUNTIME_ADAPTER:
        raise ValueError(
            f"Outer Edge Refiner runtime adapter must be "
            f"{OUTER_EDGE_REFINER_RUNTIME_ADAPTER!r}"
        )
    feature_config = dict(payload.get("feature_config") or {})
    if feature_config.get("schema") != OUTER_EDGE_FEATURE_SCHEMA:
        raise ValueError(
            f"Outer Edge Refiner feature schema must be {OUTER_EDGE_FEATURE_SCHEMA!r}"
        )
    model_config = dict(payload.get("model_config") or {})
    model = OuterEdgeRefinerNetwork(**model_config)
    model.load_state_dict(payload["model_state_dict"])
    actual_device = _device(device)
    model.to(actual_device).eval()
    if expected_ptm_repo_id is not None:
        validate_checkpoint_repo_id(
            metadata.get("ptm_repo_id"),
            expected_ptm_repo_id,
            checkpoint_kind="Outer Edge Refiner",
            metadata_key="metadata.ptm_repo_id",
        )
    return OuterEdgeRefiner(
        path=str(checkpoint_path),
        sha256=_sha256(checkpoint_path),
        model=model,
        model_config=model_config,
        feature_config=feature_config,
        normalization=dict(payload["normalization"]),
        metadata=metadata,
        device=str(actual_device),
    )


def _device(requested: str):
    import torch

    value = str(requested or "auto").lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        value = "cpu"
    return torch.device(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
