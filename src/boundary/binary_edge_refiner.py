from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from boundary.backbones import Mamba2TemporalEncoder


BINARY_EDGE_LABELS = ("background", "semantic_core")
BINARY_EDGE_IGNORE_INDEX = -100


@dataclass(frozen=True)
class BinaryEdgePrediction:
    """Shared result contract for binary Outer/Inner edge models."""

    raw_start_s: float
    raw_end_s: float
    start_s: float
    end_s: float
    start_action: str
    end_action: str
    abstain_reason: str
    start_probabilities: dict[str, float]
    end_probabilities: dict[str, float]
    class_probabilities: np.ndarray

    @property
    def start_delta_s(self) -> float:
        return float(self.start_s) - float(self.raw_start_s)

    @property
    def end_delta_s(self) -> float:
        return float(self.end_s) - float(self.raw_end_s)


class BinaryFrameEdgeNetwork:
    def __new__(
        cls,
        *,
        ptm_input_dim: int = 2048,
        ptm_projected_dim: int = 128,
        mfcc_dim: int = 40,
        position_dim: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        state_size: int = 32,
        num_heads: int = 4,
        head_dim: int = 64,
        n_groups: int = 2,
        conv_kernel: int = 4,
        chunk_size: int = 8,
        bidirectional: bool = True,
        output_dim: int = 2,
    ):
        from torch import nn

        if output_dim != 2:
            raise ValueError("binary frame edge network requires output_dim=2")

        class _Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ptm_input_dim = int(ptm_input_dim)
                auxiliary_dim = int(mfcc_dim) + int(position_dim)
                self.ptm_projector = nn.Linear(ptm_input_dim, ptm_projected_dim)
                self.frame_proj = nn.Linear(ptm_projected_dim + auxiliary_dim, hidden_size)
                self.encoder = Mamba2TemporalEncoder(
                    hidden_size=hidden_size, num_layers=num_layers,
                    state_size=state_size, num_heads=num_heads, head_dim=head_dim,
                    n_groups=n_groups, conv_kernel=conv_kernel,
                    chunk_size=chunk_size, bidirectional=bidirectional,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(self.encoder.output_dim), nn.Linear(self.encoder.output_dim, 2)
                )

            def forward(self, frame_features):
                import torch
                from torch import nn

                ptm = nn.functional.gelu(
                    self.ptm_projector(frame_features[..., : self.ptm_input_dim])
                )
                auxiliary = frame_features[..., self.ptm_input_dim :]
                hidden = self.frame_proj(torch.cat((ptm, auxiliary), dim=-1))
                return self.head(self.encoder(hidden))

        return _Network()


def canonical_to_binary_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if np.any(~np.isin(values, (0, 1, 2, BINARY_EDGE_IGNORE_INDEX))):
        raise ValueError("canonical edge labels must be discardable/semantic_target/unsure")
    return np.where(values == 2, BINARY_EDGE_IGNORE_INDEX, values).astype(np.int64)


def decode_binary_edge_logits(
    logits: np.ndarray,
    *,
    raw_start_s: float,
    raw_end_s: float,
    frame_hop_s: float,
) -> tuple[float, float]:
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("binary edge logits must have shape [frames,2]")
    target = np.flatnonzero(np.argmax(values, axis=1) == 1)
    if target.size == 0:
        raise ValueError("binary edge model emitted no semantic_core frame")
    start = float(raw_start_s) + int(target[0]) * float(frame_hop_s)
    end = float(raw_start_s) + (int(target[-1]) + 1) * float(frame_hop_s)
    start = min(float(raw_end_s), start)
    end = min(float(raw_end_s), end)
    if end <= start:
        raise ValueError("binary edge model emitted a non-positive span")
    return start, end


def binary_edge_checkpoint(
    *, schema: str, model_arch: str, runtime_adapter: str,
    artifact: dict[str, Any], model: Any, model_config: dict[str, Any],
    feature_config: dict[str, Any], normalization: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": schema, "model_arch": model_arch,
        "model_config": dict(model_config), "feature_config": dict(feature_config),
        "normalization": dict(normalization),
        "metadata": {
            **metadata, "training_labels": list(BINARY_EDGE_LABELS),
            "excluded_training_labels": ["unsure"], "decision_mode": "binary_frame_argmax",
            "runtime_adapter": runtime_adapter, "artifact": dict(artifact),
        },
        "model_state_dict": model.state_dict(),
    }
