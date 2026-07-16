from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from asr.backends.qwen import validate_checkpoint_repo_id
from boundary.ja.model import SPEECH_ISLAND_SCORER_LABELS
from boundary.outer_refiner_v2 import (
    FullIslandOuterEdgeNetwork,
    PairedOuterEdgePrediction,
    decode_outer_edge_probabilities,
)


INNER_EDGE_REFINER_V1_SCHEMA = "inner_edge_refiner_v1"
INNER_EDGE_REFINER_V1_MODEL_ARCH = "subisland_learned_ptm_edges_mamba_v1"
INNER_EDGE_REFINER_V1_RUNTIME_ADAPTER = "paired_inner_edges_v1"
INNER_EDGE_REFINER_V1_FEATURE_SCHEMA = "subisland_raw_ptm_edge_features_v1"
INNER_EDGE_REFINER_V1_ARTIFACT = {
    "name": "inner_edge_refiner",
    "display_name": "Inner Edge Refiner",
    "version": "v1",
    "pipeline_stage": 5,
    "pipeline_role": "cueqc_retained_subisland_edges",
}


class FullSubislandInnerEdgeNetwork:
    def __new__(cls, **kwargs):
        return FullIslandOuterEdgeNetwork(**kwargs)


@dataclass(frozen=True)
class InnerEdgeRefinerV1:
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
            "schema": INNER_EDGE_REFINER_V1_SCHEMA,
            "model_arch": INNER_EDGE_REFINER_V1_MODEL_ARCH,
            "runtime_adapter": INNER_EDGE_REFINER_V1_RUNTIME_ADAPTER,
            "path": self.path,
            "sha256": self.sha256,
            "feature_config": self.feature_config,
            "metadata": self.metadata,
        }

    def predict_subislands(
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
                values = np.asarray(features, dtype=np.float32)
                if values.shape[0] == 0:
                    probabilities = {
                        label: 0.0 for label in SPEECH_ISLAND_SCORER_LABELS
                    }
                    predictions.append(
                        PairedOuterEdgePrediction(
                            raw_start_s=float(raw_start),
                            raw_end_s=float(raw_end),
                            start_s=float(raw_start),
                            end_s=float(raw_end),
                            start_action="abstain",
                            end_action="abstain",
                            abstain_reason="no_feature_frames",
                            start_probabilities=probabilities,
                            end_probabilities=probabilities,
                            class_probabilities=np.zeros(
                                (0, len(SPEECH_ISLAND_SCORER_LABELS)),
                                dtype=np.float32,
                            ),
                        )
                    )
                    continue
                normalized = (
                    values - mean
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


def build_inner_edge_refiner_v1_checkpoint(
    *,
    model: Any,
    model_config: dict[str, Any],
    feature_config: dict[str, Any],
    normalization: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": INNER_EDGE_REFINER_V1_SCHEMA,
        "model_arch": INNER_EDGE_REFINER_V1_MODEL_ARCH,
        "model_config": dict(model_config),
        "feature_config": {
            **feature_config,
            "schema": INNER_EDGE_REFINER_V1_FEATURE_SCHEMA,
        },
        "normalization": dict(normalization),
        "metadata": {
            **metadata,
            "labels": list(SPEECH_ISLAND_SCORER_LABELS),
            "decision_mode": "argmax",
            "artifact": {
                **INNER_EDGE_REFINER_V1_ARTIFACT,
                **dict(metadata.get("artifact") or {}),
            },
            "runtime_adapter": INNER_EDGE_REFINER_V1_RUNTIME_ADAPTER,
        },
        "model_state_dict": model.state_dict(),
    }


def load_inner_edge_refiner_v1(
    path: str | Path,
    *,
    device: str = "auto",
    expected_ptm_repo_id: str | None = None,
) -> InnerEdgeRefinerV1:
    import torch

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != INNER_EDGE_REFINER_V1_SCHEMA:
        raise ValueError(
            f"unsupported Inner Edge Refiner schema: {payload.get('schema')!r}; "
            f"expected {INNER_EDGE_REFINER_V1_SCHEMA!r}"
        )
    if payload.get("model_arch") != INNER_EDGE_REFINER_V1_MODEL_ARCH:
        raise ValueError(
            f"Inner Edge Refiner v1 must use {INNER_EDGE_REFINER_V1_MODEL_ARCH!r}"
        )
    feature_config = dict(payload.get("feature_config") or {})
    if feature_config.get("schema") != INNER_EDGE_REFINER_V1_FEATURE_SCHEMA:
        raise ValueError(
            f"Inner Edge Refiner v1 feature schema must be "
            f"{INNER_EDGE_REFINER_V1_FEATURE_SCHEMA!r}"
        )
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("runtime_adapter") != INNER_EDGE_REFINER_V1_RUNTIME_ADAPTER:
        raise ValueError(
            f"Inner Edge Refiner v1 runtime adapter must be "
            f"{INNER_EDGE_REFINER_V1_RUNTIME_ADAPTER!r}"
        )
    model_config = dict(payload.get("model_config") or {})
    model = FullSubislandInnerEdgeNetwork(**model_config)
    model.load_state_dict(payload["model_state_dict"])
    actual_device = _device(device)
    model.to(actual_device).eval()
    if expected_ptm_repo_id is not None:
        validate_checkpoint_repo_id(
            metadata.get("ptm_repo_id"),
            expected_ptm_repo_id,
            checkpoint_kind="Inner Edge Refiner v1",
            metadata_key="metadata.ptm_repo_id",
        )
    return InnerEdgeRefinerV1(
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
