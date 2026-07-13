from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from asr.backends.qwen import validate_checkpoint_repo_id
from boundary.backbones import Mamba2TemporalEncoder
from boundary.ja.model import SPEECH_ISLAND_SCORER_LABELS


OUTER_EDGE_REFINER_V2_SCHEMA = "outer_edge_refiner_v2"
OUTER_EDGE_REFINER_V2_MODEL_ARCH = "full_island_semantic_edges_mamba_v1"
OUTER_EDGE_REFINER_V2_RUNTIME_ADAPTER = "paired_outer_edges_v2"
OUTER_EDGE_REFINER_V2_FEATURE_SCHEMA = "full_island_semantic_edge_features_v2"
OUTER_EDGE_REFINER_V2_ARTIFACT = {
    "name": "outer_edge_refiner",
    "display_name": "Outer Edge Refiner",
    "version": "v2",
    "pipeline_stage": 2,
    "pipeline_role": "full_island_semantic_outer_edges",
}


class FullIslandOuterEdgeNetwork:
    def __new__(
        cls,
        *,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        state_size: int = 32,
        num_heads: int = 4,
        head_dim: int = 64,
        n_groups: int = 2,
        conv_kernel: int = 4,
        chunk_size: int = 8,
        bidirectional: bool = True,
        output_dim: int = len(SPEECH_ISLAND_SCORER_LABELS),
    ):
        from torch import nn

        if output_dim != len(SPEECH_ISLAND_SCORER_LABELS):
            raise ValueError(
                f"Outer Edge Refiner v2 requires output_dim={len(SPEECH_ISLAND_SCORER_LABELS)}"
            )

        class _Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.frame_proj = nn.Linear(input_dim, hidden_size)
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
                self.head = nn.Sequential(
                    nn.LayerNorm(self.encoder.output_dim),
                    nn.Linear(self.encoder.output_dim, output_dim),
                )

            def forward(self, frame_features):
                encoded = self.encoder(self.frame_proj(frame_features))
                return self.head(encoded)

        return _Network()


@dataclass(frozen=True)
class PairedOuterEdgePrediction:
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
        return self.start_s - self.raw_start_s

    @property
    def end_delta_s(self) -> float:
        return self.end_s - self.raw_end_s


def decode_outer_edge_probabilities(
    probabilities: np.ndarray,
    *,
    raw_start_s: float,
    raw_end_s: float,
    frame_hop_s: float,
) -> PairedOuterEdgePrediction:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(SPEECH_ISLAND_SCORER_LABELS):
        raise ValueError(
            "outer edge probabilities must have shape "
            f"[frames, {len(SPEECH_ISLAND_SCORER_LABELS)}]"
        )
    labels = np.argmax(values, axis=1)
    target_index = SPEECH_ISLAND_SCORER_LABELS.index("semantic_target")
    unsure_index = SPEECH_ISLAND_SCORER_LABELS.index("unsure")
    target_frames = np.flatnonzero(labels == target_index)
    empty_probabilities = {label: 0.0 for label in SPEECH_ISLAND_SCORER_LABELS}
    if target_frames.size == 0:
        return PairedOuterEdgePrediction(
            raw_start_s=float(raw_start_s),
            raw_end_s=float(raw_end_s),
            start_s=float(raw_start_s),
            end_s=float(raw_end_s),
            start_action="abstain",
            end_action="abstain",
            abstain_reason="no_semantic_target",
            start_probabilities=empty_probabilities,
            end_probabilities=empty_probabilities,
            class_probabilities=values,
        )
    first = int(target_frames[0])
    last = int(target_frames[-1])
    start_unsure = bool(np.any(labels[:first] == unsure_index))
    end_unsure = bool(np.any(labels[last + 1 :] == unsure_index))
    start = (
        float(raw_start_s)
        if start_unsure
        else min(float(raw_end_s), float(raw_start_s) + first * float(frame_hop_s))
    )
    end = (
        float(raw_end_s)
        if end_unsure
        else min(
            float(raw_end_s),
            float(raw_start_s) + (last + 1) * float(frame_hop_s),
        )
    )
    reasons: list[str] = []
    if start_unsure:
        reasons.append("unsure_before_target")
    if end_unsure:
        reasons.append("unsure_after_target")
    if end <= start:
        start = float(raw_start_s)
        end = float(raw_end_s)
        reasons.append("invalid_paired_edges")
        start_action = end_action = "abstain"
    else:
        start_action = "abstain" if start_unsure else "refined"
        end_action = "abstain" if end_unsure else "refined"
    return PairedOuterEdgePrediction(
        raw_start_s=float(raw_start_s),
        raw_end_s=float(raw_end_s),
        start_s=start,
        end_s=end,
        start_action=start_action,
        end_action=end_action,
        abstain_reason="+".join(reasons),
        start_probabilities={
            label: float(values[first, index])
            for index, label in enumerate(SPEECH_ISLAND_SCORER_LABELS)
        },
        end_probabilities={
            label: float(values[last, index])
            for index, label in enumerate(SPEECH_ISLAND_SCORER_LABELS)
        },
        class_probabilities=values,
    )


@dataclass(frozen=True)
class OuterEdgeRefinerV2:
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
            "schema": OUTER_EDGE_REFINER_V2_SCHEMA,
            "model_arch": OUTER_EDGE_REFINER_V2_MODEL_ARCH,
            "runtime_adapter": OUTER_EDGE_REFINER_V2_RUNTIME_ADAPTER,
            "path": self.path,
            "sha256": self.sha256,
            "feature_config": self.feature_config,
            "metadata": self.metadata,
        }

    def predict_islands(
        self,
        *,
        frame_feature_groups: Sequence[np.ndarray],
        raw_spans: Sequence[tuple[float, float]],
        frame_hop_s: float,
    ) -> list[PairedOuterEdgePrediction]:
        import torch

        mean = np.asarray(self.normalization["feature_mean"], dtype=np.float32)
        std = np.asarray(self.normalization["feature_std"], dtype=np.float32)
        predictions: list[PairedOuterEdgePrediction] = []
        with torch.inference_mode():
            for features, (raw_start, raw_end) in zip(frame_feature_groups, raw_spans):
                normalized = (
                    np.asarray(features, dtype=np.float32) - mean
                ) / np.maximum(std, 1e-6)
                logits = self.model(
                    torch.from_numpy(np.ascontiguousarray(normalized))
                    .unsqueeze(0)
                    .to(self.device)
                )[0]
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                predictions.append(
                    decode_outer_edge_probabilities(
                        probabilities,
                        raw_start_s=raw_start,
                        raw_end_s=raw_end,
                        frame_hop_s=frame_hop_s,
                    )
                )
        return predictions


def build_outer_edge_refiner_v2_checkpoint(
    *,
    model: Any,
    model_config: dict[str, Any],
    feature_config: dict[str, Any],
    normalization: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": OUTER_EDGE_REFINER_V2_SCHEMA,
        "model_arch": OUTER_EDGE_REFINER_V2_MODEL_ARCH,
        "model_config": dict(model_config),
        "feature_config": {
            **feature_config,
            "schema": OUTER_EDGE_REFINER_V2_FEATURE_SCHEMA,
        },
        "normalization": dict(normalization),
        "metadata": {
            **metadata,
            "labels": list(SPEECH_ISLAND_SCORER_LABELS),
            "decision_mode": "argmax",
            "artifact": {
                **OUTER_EDGE_REFINER_V2_ARTIFACT,
                **dict(metadata.get("artifact") or {}),
            },
            "runtime_adapter": OUTER_EDGE_REFINER_V2_RUNTIME_ADAPTER,
        },
        "model_state_dict": model.state_dict(),
    }


def load_outer_edge_refiner_v2(
    path: str | Path,
    *,
    device: str = "auto",
    expected_ptm_repo_id: str | None = None,
) -> OuterEdgeRefinerV2:
    import torch

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != OUTER_EDGE_REFINER_V2_SCHEMA:
        raise ValueError(
            f"unsupported Outer Edge Refiner schema: {payload.get('schema')!r}; "
            f"expected {OUTER_EDGE_REFINER_V2_SCHEMA!r}"
        )
    if payload.get("model_arch") != OUTER_EDGE_REFINER_V2_MODEL_ARCH:
        raise ValueError(
            f"Outer Edge Refiner v2 must use {OUTER_EDGE_REFINER_V2_MODEL_ARCH!r}"
        )
    feature_config = dict(payload.get("feature_config") or {})
    if feature_config.get("schema") != OUTER_EDGE_REFINER_V2_FEATURE_SCHEMA:
        raise ValueError(
            f"Outer Edge Refiner v2 feature schema must be "
            f"{OUTER_EDGE_REFINER_V2_FEATURE_SCHEMA!r}"
        )
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("runtime_adapter") != OUTER_EDGE_REFINER_V2_RUNTIME_ADAPTER:
        raise ValueError(
            f"Outer Edge Refiner v2 runtime adapter must be "
            f"{OUTER_EDGE_REFINER_V2_RUNTIME_ADAPTER!r}"
        )
    model_config = dict(payload.get("model_config") or {})
    model = FullIslandOuterEdgeNetwork(**model_config)
    model.load_state_dict(payload["model_state_dict"])
    actual_device = _device(device)
    model.to(actual_device).eval()
    if expected_ptm_repo_id is not None:
        validate_checkpoint_repo_id(
            metadata.get("ptm_repo_id"),
            expected_ptm_repo_id,
            checkpoint_kind="Outer Edge Refiner v2",
            metadata_key="metadata.ptm_repo_id",
        )
    return OuterEdgeRefinerV2(
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
