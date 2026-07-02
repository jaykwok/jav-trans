from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from asr.backends.qwen import validate_checkpoint_repo_id
from boundary.backbones import Mamba2TemporalEncoder


SEMANTIC_SPLIT_SCHEMA = "semantic_split_verifier_v1"
SEMANTIC_SPLIT_MODEL_ARCH = "candidate_mamba_v1"
SEMANTIC_SPLIT_RUNTIME_ADAPTER = "candidate_cut_continue_unsure_v1"
SEMANTIC_SPLIT_LABELS = ("cut", "continue", "unsure")
SEMANTIC_SPLIT_FEATURE_SCHEMA = "semantic_split_candidate_features_v1"
SEMANTIC_SPLIT_ARTIFACT = {
    "name": "semantic_split_model",
    "display_name": "Semantic Split Model",
    "version": "v1",
    "pipeline_stage": 3,
    "pipeline_role": "cut_continue_unsure_decision",
}


class SemanticSplitVerifierNetwork:
    """Factory wrapper that keeps torch imports local to model construction."""

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
        output_dim: int = 3,
    ):
        import torch
        from torch import nn

        if frame_dim <= 0 or scalar_dim <= 0:
            raise ValueError("frame_dim and scalar_dim must be positive")
        if output_dim != len(SEMANTIC_SPLIT_LABELS):
            raise ValueError("Semantic Split Model requires output_dim=3")

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
                    nn.Linear(hidden_size, output_dim),
                )

            def forward(self, frame_features, scalar_features):
                if frame_features.ndim != 3:
                    raise ValueError("frame_features must have shape [batch,bins,dim]")
                if scalar_features.ndim != 2:
                    raise ValueError("scalar_features must have shape [batch,dim]")
                encoded = self.encoder(self.frame_proj(frame_features))
                pooled = encoded.mean(dim=1)
                return self.head(torch.cat((pooled, self.scalar_arm(scalar_features)), dim=-1))

        return _Network()


@dataclass(frozen=True)
class SplitDecision:
    label: str
    p_cut: float
    p_continue: float
    p_unsure: float


@dataclass(frozen=True)
class SemanticSplitVerifier:
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
            "schema": SEMANTIC_SPLIT_SCHEMA,
            "model_arch": SEMANTIC_SPLIT_MODEL_ARCH,
            "runtime_adapter": SEMANTIC_SPLIT_RUNTIME_ADAPTER,
            "path": self.path,
            "sha256": self.sha256,
            "model_config": self.model_config,
            "feature_config": self.feature_config,
            "metadata": self.metadata,
        }

    def decide(
        self,
        *,
        frame_features: np.ndarray,
        scalar_features: np.ndarray,
    ) -> list[SplitDecision]:
        import torch

        frames = np.asarray(frame_features, dtype=np.float32)
        scalars = np.asarray(scalar_features, dtype=np.float32)
        frame_mean = np.asarray(self.normalization["frame_mean"], dtype=np.float32)
        frame_std = np.asarray(self.normalization["frame_std"], dtype=np.float32)
        scalar_mean = np.asarray(self.normalization["scalar_mean"], dtype=np.float32)
        scalar_std = np.asarray(self.normalization["scalar_std"], dtype=np.float32)
        frames = (frames - frame_mean) / np.maximum(frame_std, 1e-6)
        scalars = (scalars - scalar_mean) / np.maximum(scalar_std, 1e-6)
        with torch.inference_mode():
            logits = self.model(
                torch.from_numpy(frames).to(self.device),
                torch.from_numpy(scalars).to(self.device),
            )
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        decisions: list[SplitDecision] = []
        for row in probabilities:
            index = int(np.argmax(row))
            decisions.append(
                SplitDecision(
                    label=SEMANTIC_SPLIT_LABELS[index],
                    p_cut=float(row[0]),
                    p_continue=float(row[1]),
                    p_unsure=float(row[2]),
                )
            )
        return decisions


def load_semantic_split_verifier(
    path: str | Path,
    *,
    device: str = "auto",
    expected_ptm_repo_id: str | None = None,
) -> SemanticSplitVerifier:
    import torch

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SEMANTIC_SPLIT_SCHEMA:
        raise ValueError(
            f"unsupported Semantic Split Model schema: {payload.get('schema')!r}; "
            f"expected {SEMANTIC_SPLIT_SCHEMA!r}"
        )
    if payload.get("model_arch") != SEMANTIC_SPLIT_MODEL_ARCH:
        raise ValueError(f"Semantic Split Model must use {SEMANTIC_SPLIT_MODEL_ARCH!r}")
    metadata = dict(payload.get("metadata") or {})
    artifact = metadata.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Semantic Split Model metadata.artifact is required")
    for key, expected in SEMANTIC_SPLIT_ARTIFACT.items():
        if artifact.get(key) != expected:
            raise ValueError(
                f"Semantic Split Model metadata.artifact.{key} must be {expected!r}"
            )
    if metadata.get("runtime_adapter") != SEMANTIC_SPLIT_RUNTIME_ADAPTER:
        raise ValueError(
            f"Semantic Split Model metadata.runtime_adapter must be "
            f"{SEMANTIC_SPLIT_RUNTIME_ADAPTER!r}"
        )
    if tuple(metadata.get("labels") or ()) != SEMANTIC_SPLIT_LABELS:
        raise ValueError(f"Semantic Split Model labels must be {SEMANTIC_SPLIT_LABELS!r}")
    feature_config = dict(payload.get("feature_config") or {})
    if feature_config.get("schema") != SEMANTIC_SPLIT_FEATURE_SCHEMA:
        raise ValueError(
            f"Semantic Split Model feature schema must be {SEMANTIC_SPLIT_FEATURE_SCHEMA!r}"
        )
    model_config = dict(payload.get("model_config") or {})
    model = SemanticSplitVerifierNetwork(**model_config)
    model.load_state_dict(payload["model_state_dict"])
    actual_device = _model_device(device)
    model.to(actual_device).eval()
    if expected_ptm_repo_id is not None:
        validate_checkpoint_repo_id(
            metadata.get("ptm_repo_id"),
            expected_ptm_repo_id,
            checkpoint_kind="Semantic Split Model",
            metadata_key="metadata.ptm_repo_id",
        )
    return SemanticSplitVerifier(
        path=str(checkpoint_path),
        sha256=_sha256(checkpoint_path),
        model=model,
        model_config=model_config,
        feature_config=feature_config,
        normalization=dict(payload.get("normalization") or {}),
        metadata=metadata,
        device=str(actual_device),
    )


def build_semantic_split_checkpoint(
    *,
    model: Any,
    model_config: dict[str, Any],
    feature_config: dict[str, Any],
    normalization: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    artifact = {
        **SEMANTIC_SPLIT_ARTIFACT,
        **dict(metadata.get("artifact") or {}),
    }
    return {
        "schema": SEMANTIC_SPLIT_SCHEMA,
        "model_arch": SEMANTIC_SPLIT_MODEL_ARCH,
        "model_config": dict(model_config),
        "feature_config": {
            **feature_config,
            "schema": SEMANTIC_SPLIT_FEATURE_SCHEMA,
        },
        "normalization": dict(normalization),
        "metadata": {
            **metadata,
            "artifact": artifact,
            "runtime_adapter": SEMANTIC_SPLIT_RUNTIME_ADAPTER,
            "labels": list(SEMANTIC_SPLIT_LABELS),
        },
        "model_state_dict": model.state_dict(),
    }


def _model_device(requested: str):
    import torch

    value = str(requested or "auto").strip().lower()
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
