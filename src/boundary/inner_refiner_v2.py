from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from asr.backends.qwen import validate_checkpoint_repo_id
from boundary.binary_edge_refiner import (
    BINARY_EDGE_LABELS,
    BinaryEdgePrediction,
    BinaryFrameEdgeNetwork,
    binary_edge_checkpoint,
)
from boundary.contracts import (
    ACOUSTIC_BINARY_V12_CONTRACT,
    require_boundary_contract_id,
)

INNER_EDGE_REFINER_V2_SCHEMA = "inner_edge_refiner_v2"
INNER_EDGE_REFINER_V2_MODEL_ARCH = "subisland_binary_ptm_edges_mamba_v2"
INNER_EDGE_REFINER_V2_FEATURE_SCHEMA = "subisland_raw_ptm_binary_edges_v2"
INNER_EDGE_REFINER_V2_RUNTIME_ADAPTER = "paired_acoustic_inner_edges_binary_v2"
INNER_EDGE_REFINER_V2_ARTIFACT = {
    "name": "inner_edge_refiner", "display_name": "Inner Edge Refiner",
    "version": "v2", "pipeline_stage": 5,
    "pipeline_role": "cueqc_retained_subisland_to_acoustic_speech_core",
}

FullSubislandBinaryInnerEdgeNetwork = BinaryFrameEdgeNetwork


@dataclass(frozen=True)
class InnerEdgeRefinerV2:
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
            "schema": INNER_EDGE_REFINER_V2_SCHEMA,
            "model_arch": INNER_EDGE_REFINER_V2_MODEL_ARCH,
            "runtime_adapter": INNER_EDGE_REFINER_V2_RUNTIME_ADAPTER,
            "boundary_serialization_contract_id": require_boundary_contract_id(
                self.metadata.get("boundary_serialization_contract_id")
            ),
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
    ) -> list[BinaryEdgePrediction]:
        import torch

        mean = np.asarray(self.normalization["feature_mean"], dtype=np.float32)
        std = np.asarray(self.normalization["feature_std"], dtype=np.float32)
        predictions: list[BinaryEdgePrediction] = []
        with torch.inference_mode():
            for features, (raw_start, raw_end) in zip(frame_feature_groups, raw_spans):
                values = np.asarray(features, dtype=np.float32)
                if values.ndim != 2 or values.shape[0] == 0:
                    raise ValueError("Inner Edge Refiner v2 requires non-empty frame features")
                normalized = (values - mean) / np.maximum(std, 1e-6)
                logits = self.model(
                    torch.from_numpy(np.ascontiguousarray(normalized))
                    .unsqueeze(0)
                    .to(self.device)
                )[0]
                probabilities = torch.softmax(logits, dim=-1).float().cpu().numpy()
                labels = np.argmax(probabilities, axis=1)
                semantic = np.flatnonzero(labels == 1)
                if not semantic.size:
                    summary = probabilities.mean(axis=0)
                    class_probabilities = {
                        label: float(summary[index])
                        for index, label in enumerate(BINARY_EDGE_LABELS)
                    }
                    predictions.append(BinaryEdgePrediction(
                        raw_start_s=float(raw_start), raw_end_s=float(raw_end),
                        start_s=float(raw_start), end_s=float(raw_end),
                        start_action="drop", end_action="drop",
                        abstain_reason="binary_all_background",
                        start_probabilities=class_probabilities,
                        end_probabilities=class_probabilities,
                        class_probabilities=probabilities,
                    ))
                    continue
                first, last = int(semantic[0]), int(semantic[-1])
                start = min(float(raw_end), float(raw_start) + first * float(frame_hop_s))
                end = min(float(raw_end), float(raw_start) + (last + 1) * float(frame_hop_s))
                predictions.append(BinaryEdgePrediction(
                    raw_start_s=float(raw_start), raw_end_s=float(raw_end),
                    start_s=start, end_s=end,
                    start_action="refined", end_action="refined", abstain_reason="",
                    start_probabilities={
                        label: float(probabilities[first, index])
                        for index, label in enumerate(BINARY_EDGE_LABELS)
                    },
                    end_probabilities={
                        label: float(probabilities[last, index])
                        for index, label in enumerate(BINARY_EDGE_LABELS)
                    },
                    class_probabilities=probabilities,
                ))
        return predictions


def build_inner_edge_refiner_v2_checkpoint(**kwargs):
    feature_config = {
        **dict(kwargs.pop("feature_config")), "schema": INNER_EDGE_REFINER_V2_FEATURE_SCHEMA
    }
    metadata = dict(kwargs.pop("metadata"))
    metadata.setdefault(
        "boundary_serialization_contract_id",
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
    )
    return binary_edge_checkpoint(
        schema=INNER_EDGE_REFINER_V2_SCHEMA,
        model_arch=INNER_EDGE_REFINER_V2_MODEL_ARCH,
        runtime_adapter=INNER_EDGE_REFINER_V2_RUNTIME_ADAPTER,
        artifact=INNER_EDGE_REFINER_V2_ARTIFACT,
        feature_config=feature_config,
        metadata=metadata,
        **kwargs,
    )


def load_inner_edge_refiner_v2(
    path: str | Path,
    *,
    device: str = "auto",
    expected_ptm_repo_id: str | None = None,
) -> InnerEdgeRefinerV2:
    import torch

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != INNER_EDGE_REFINER_V2_SCHEMA:
        raise ValueError(
            f"unsupported Inner Edge Refiner v2 schema: {payload.get('schema')!r}"
        )
    if payload.get("model_arch") != INNER_EDGE_REFINER_V2_MODEL_ARCH:
        raise ValueError("Inner Edge Refiner v2 model_arch mismatch")
    feature_config = dict(payload.get("feature_config") or {})
    if feature_config.get("schema") != INNER_EDGE_REFINER_V2_FEATURE_SCHEMA:
        raise ValueError("Inner Edge Refiner v2 feature schema mismatch")
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("runtime_adapter") != INNER_EDGE_REFINER_V2_RUNTIME_ADAPTER:
        raise ValueError("Inner Edge Refiner v2 runtime adapter mismatch")
    if metadata.get("training_labels") != list(BINARY_EDGE_LABELS):
        raise ValueError("Inner Edge Refiner v2 must train background/semantic_core only")
    if metadata.get("excluded_training_labels") != ["unsure"]:
        raise ValueError("Inner Edge Refiner v2 must exclude unsure from training")
    if metadata.get("decision_mode") != "binary_frame_argmax":
        raise ValueError("Inner Edge Refiner v2 must use binary frame argmax")
    require_boundary_contract_id(
        metadata.get("boundary_serialization_contract_id")
    )
    model_config = dict(payload.get("model_config") or {})
    if int(model_config.get("output_dim") or 0) != 2:
        raise ValueError("Inner Edge Refiner v2 requires a two-logit head")
    model = FullSubislandBinaryInnerEdgeNetwork(**model_config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    actual_device = _device(device)
    model.to(actual_device).eval()
    if expected_ptm_repo_id is not None:
        validate_checkpoint_repo_id(
            metadata.get("ptm_repo_id"), expected_ptm_repo_id,
            checkpoint_kind="Inner Edge Refiner v2",
            metadata_key="metadata.ptm_repo_id",
        )
    return InnerEdgeRefinerV2(
        path=str(checkpoint_path), sha256=_sha256(checkpoint_path), model=model,
        model_config=model_config, feature_config=feature_config,
        normalization=dict(payload["normalization"]), metadata=metadata,
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
