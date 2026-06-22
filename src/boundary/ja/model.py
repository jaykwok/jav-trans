from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


MAMBA2_FRAME_SCORER_SCHEMA = "speech_boundary_ja_mamba2_frame_boundary_scorer_v4"
MAMBA2_FRAME_SCORER_MODEL_TYPE = "mamba2_frame_boundary_scorer"
MAMBA2_FRAME_SCORER_OUTPUT_DIM = 3
MAMBA2_FRAME_SCORER_OUTPUT_HEADS = (
    "speech_prob",
    "split_boundary_prob",
    "drop_gap_prob",
)
MAMBA2_FRAME_SCORER_DECODER = "peak_split_v1"


def count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _bool_config(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _validate_metadata(metadata: dict[str, Any]) -> None:
    heads = tuple(str(item) for item in (metadata.get("output_heads") or ()))
    if heads != MAMBA2_FRAME_SCORER_OUTPUT_HEADS:
        raise ValueError(
            "Mamba2 boundary scorer checkpoint metadata.output_heads must be "
            f"{list(MAMBA2_FRAME_SCORER_OUTPUT_HEADS)!r}; got {list(heads)!r}"
        )
    decoder = str(metadata.get("decoder") or "")
    if decoder != MAMBA2_FRAME_SCORER_DECODER:
        raise ValueError(
            "Mamba2 boundary scorer checkpoint metadata.decoder must be "
            f"{MAMBA2_FRAME_SCORER_DECODER!r}; got {decoder!r}"
        )


def build_feature_frame_scorer_model(*, schema: str, model_config: dict[str, Any]):
    if schema == MAMBA2_FRAME_SCORER_SCHEMA:
        from boundary.backbones import TRANSFORMERS_MAMBA2_BACKBONE, BoundarySequenceClassifier

        for key in ("input_dim", "hidden_size", "num_layers"):
            if key not in model_config:
                raise ValueError(f"Mamba2 scorer checkpoint missing model_config.{key}")
        backbone_kwargs: dict[str, Any] = {
            "state_size": int(model_config.get("state_size", 32)),
            "num_heads": int(model_config.get("num_heads", 4)),
            "n_groups": int(model_config.get("n_groups", 2)),
            "chunk_size": int(model_config.get("chunk_size", 8)),
            "bidirectional": _bool_config(model_config.get("bidirectional", True)),
        }
        if "head_dim" in model_config:
            backbone_kwargs["head_dim"] = int(model_config["head_dim"])
        if "output_dim" not in model_config:
            raise ValueError("Mamba2 boundary scorer checkpoint missing model_config.output_dim")
        output_dim = int(model_config["output_dim"])
        if output_dim != MAMBA2_FRAME_SCORER_OUTPUT_DIM:
            raise ValueError(
                f"Mamba2 boundary scorer requires output_dim={MAMBA2_FRAME_SCORER_OUTPUT_DIM}, "
                f"got {output_dim}"
            )
        return BoundarySequenceClassifier(
            input_dim=int(model_config["input_dim"]),
            backbone=str(model_config.get("backbone") or TRANSFORMERS_MAMBA2_BACKBONE),
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            output_dim=output_dim,
            **backbone_kwargs,
        )
    raise ValueError(
        f"unsupported scorer checkpoint schema: {schema!r}; "
        f"expected {MAMBA2_FRAME_SCORER_SCHEMA!r}"
    )


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
class Mamba2FrameScorerBundle:
    path: str
    sha256: str
    model: Any
    model_config: dict[str, Any]
    normalization: dict[str, Any]
    metadata: dict[str, Any]
    device: str
    schema: str = MAMBA2_FRAME_SCORER_SCHEMA
    model_type: str = MAMBA2_FRAME_SCORER_MODEL_TYPE

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
        model_config: dict[str, Any] = {
            "ptm_dim": self.ptm_dim,
            "mfcc_dim": self.mfcc_dim,
            "input_dim": self.input_dim,
        }
        for key in (
            "hidden_size",
            "backbone",
            "num_layers",
            "state_size",
            "num_heads",
            "head_dim",
            "n_groups",
            "chunk_size",
            "bidirectional",
            "output_dim",
        ):
            if key in self.model_config:
                value = self.model_config[key]
                if isinstance(value, (bool, int, float, str)):
                    model_config[key] = value
        return {
            "schema": self.schema,
            "model_type": self.model_type,
            "path": self.path,
            "sha256": self.sha256,
            "model_config": model_config,
            "metadata": {
                "operating_point": str(self.metadata.get("operating_point") or ""),
                "ptm_repo_id": str(self.metadata.get("ptm_repo_id") or ""),
                "trained_steps": int(self.metadata.get("trained_steps") or 0),
                "labels": str(self.metadata.get("labels") or ""),
                "feature_manifest": str(self.metadata.get("feature_manifest") or ""),
                "output_heads": list(self.metadata.get("output_heads") or ()),
                "decoder": str(self.metadata.get("decoder") or ""),
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
    schema: str = MAMBA2_FRAME_SCORER_SCHEMA,
) -> dict[str, Any]:
    if schema != MAMBA2_FRAME_SCORER_SCHEMA:
        raise ValueError(
            f"unsupported scorer checkpoint schema: {schema!r}; "
            f"expected {MAMBA2_FRAME_SCORER_SCHEMA!r}"
        )
    if int(model_config.get("output_dim", 0)) != MAMBA2_FRAME_SCORER_OUTPUT_DIM:
        raise ValueError(
            f"Mamba2 boundary scorer checkpoint requires output_dim={MAMBA2_FRAME_SCORER_OUTPUT_DIM}"
        )
    metadata_dict = dict(metadata or {})
    metadata_dict.setdefault("output_heads", list(MAMBA2_FRAME_SCORER_OUTPUT_HEADS))
    metadata_dict.setdefault("decoder", MAMBA2_FRAME_SCORER_DECODER)
    _validate_metadata(metadata_dict)
    return {
        "schema": schema,
        "model_type": MAMBA2_FRAME_SCORER_MODEL_TYPE,
        "model_config": dict(model_config),
        "normalization": dict(normalization),
        "metadata": metadata_dict,
        "model_state_dict": model.state_dict(),
    }


def load_feature_frame_scorer_checkpoint(
    path: str | Path,
    *,
    device: str | Any = "cpu",
) -> Mamba2FrameScorerBundle:
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
    schema = str(payload.get("schema") or "")
    if schema != MAMBA2_FRAME_SCORER_SCHEMA:
        raise ValueError(
            f"unsupported scorer checkpoint schema: {payload.get('schema')!r}; "
            f"expected {MAMBA2_FRAME_SCORER_SCHEMA!r}"
        )
    model_type = str(payload.get("model_type") or "")
    if model_type != MAMBA2_FRAME_SCORER_MODEL_TYPE:
        raise ValueError(
            f"unsupported scorer checkpoint model_type: {payload.get('model_type')!r}; "
            f"expected {MAMBA2_FRAME_SCORER_MODEL_TYPE!r}"
        )
    model_config = dict(payload.get("model_config") or {})
    for key in ("ptm_dim", "mfcc_dim", "input_dim"):
        if key not in model_config:
            raise ValueError(f"scorer checkpoint missing model_config.{key}")
    if int(model_config["input_dim"]) != int(model_config["ptm_dim"]) + int(model_config["mfcc_dim"]):
        raise ValueError("scorer checkpoint input_dim does not match ptm_dim + mfcc_dim")
    metadata = dict(payload.get("metadata") or {})
    _validate_metadata(metadata)
    normalization = dict(payload.get("normalization") or {})
    mean = list(normalization.get("feature_mean") or [])
    std = list(normalization.get("feature_std") or [])
    if len(mean) != int(model_config["input_dim"]) or len(std) != int(model_config["input_dim"]):
        raise ValueError("scorer checkpoint normalization length does not match input_dim")
    model = build_feature_frame_scorer_model(schema=schema, model_config=model_config)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("scorer checkpoint missing model_state_dict")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return Mamba2FrameScorerBundle(
        path=str(checkpoint_path),
        sha256=checkpoint_sha256(checkpoint_path),
        model=model,
        model_config=model_config,
        normalization=normalization,
        metadata=metadata,
        device=str(device),
        schema=schema,
        model_type=model_type,
    )


def score_feature_frame_boundary_probabilities(
    bundle: Mamba2FrameScorerBundle,
    *,
    ptm: np.ndarray,
    mfcc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scored = score_feature_frame_boundary_probabilities_batch(
        bundle,
        feature_pairs=[(ptm, mfcc)],
    )
    if not scored:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty
    return scored[0]


def score_feature_frame_boundary_probabilities_batch(
    bundle: Mamba2FrameScorerBundle,
    *,
    feature_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    import torch

    pairs = list(feature_pairs)
    if not pairs:
        return []
    if int(bundle.model_config.get("output_dim", 0)) != MAMBA2_FRAME_SCORER_OUTPUT_DIM:
        raise ValueError(
            f"scorer expected output_dim={MAMBA2_FRAME_SCORER_OUTPUT_DIM}, "
            f"got {bundle.model_config.get('output_dim')}"
        )
    mean = np.asarray(bundle.normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(bundle.normalization["feature_std"], dtype=np.float32)
    lengths: list[int] = []
    normalized_rows: list[np.ndarray] = []
    for ptm, mfcc in pairs:
        frame_total = min(int(ptm.shape[0]), int(mfcc.shape[0]))
        if int(ptm.shape[1]) != bundle.ptm_dim:
            raise ValueError(f"scorer expected ptm_dim={bundle.ptm_dim}, got {int(ptm.shape[1])}")
        if int(mfcc.shape[1]) != bundle.mfcc_dim:
            raise ValueError(f"scorer expected mfcc_dim={bundle.mfcc_dim}, got {int(mfcc.shape[1])}")
        lengths.append(frame_total)
        if frame_total <= 0:
            normalized_rows.append(np.zeros((0, bundle.input_dim), dtype=np.float32))
            continue
        features = np.concatenate(
            [
                np.asarray(ptm[:frame_total], dtype=np.float32),
                np.asarray(mfcc[:frame_total], dtype=np.float32),
            ],
            axis=1,
        )
        normalized_rows.append(
            np.ascontiguousarray((features - mean) / np.maximum(std, 1e-6), dtype=np.float32)
        )
    max_len = max(lengths, default=0)
    if max_len <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return [(empty, empty, empty) for _length in lengths]
    batch = np.zeros((len(normalized_rows), max_len, bundle.input_dim), dtype=np.float32)
    mask = np.zeros((len(normalized_rows), max_len), dtype=np.int64)
    for index, row in enumerate(normalized_rows):
        length = int(lengths[index])
        if length <= 0:
            continue
        batch[index, :length, :] = row
        mask[index, :length] = 1
    with torch.inference_mode():
        tensor = torch.from_numpy(batch).to(bundle.device)
        attention_mask = torch.from_numpy(mask).to(bundle.device)
        logits = bundle.model(tensor, attention_mask=attention_mask)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    if probabilities.ndim != 3 or probabilities.shape[2] != MAMBA2_FRAME_SCORER_OUTPUT_DIM:
        raise ValueError(
            "boundary scorer probabilities must have shape "
            f"[batch, frames, {MAMBA2_FRAME_SCORER_OUTPUT_DIM}]"
        )
    outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index, length in enumerate(lengths):
        length = int(length)
        if length <= 0:
            empty = np.zeros(0, dtype=np.float32)
            outputs.append((empty, empty, empty))
            continue
        row = probabilities[index, :length, :]
        outputs.append(
            (
                np.ascontiguousarray(row[:, 0], dtype=np.float32),
                np.ascontiguousarray(row[:, 1], dtype=np.float32),
                np.ascontiguousarray(row[:, 2], dtype=np.float32),
            )
        )
    return outputs
