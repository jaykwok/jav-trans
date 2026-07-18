from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from boundary.contracts import (
    ACOUSTIC_BINARY_V12_CONTRACT,
    require_boundary_contract_id,
)


SPEECH_ISLAND_SCORER_V8_SCHEMA = "speech_boundary_ja_mamba2_speech_island_scorer_v8"
SPEECH_ISLAND_SCORER_V8_MODEL_TYPE = "mamba2_speech_island_scorer"
SPEECH_ISLAND_SCORER_V8_MODEL_ARCH = "v8-speech-island"
SPEECH_ISLAND_SCORER_V8_OUTPUT_HEADS = ("speech_prob",)
SPEECH_ISLAND_SCORER_V8_DECODER = "speech_hysteresis_islands_v1"

SPEECH_ISLAND_SCORER_SCHEMA = "semantic_speech_scorer_v9"
SPEECH_ISLAND_SCORER_MODEL_TYPE = "mamba2_semantic_speech_scorer"
SPEECH_ISLAND_SCORER_MODEL_ARCH = "semantic-source-membership-mamba-v1"
SPEECH_ISLAND_SCORER_LABELS = ("discardable", "semantic_target", "unsure")
SPEECH_ISLAND_MEMBERSHIP_LABELS = ("outside", "inside", "unsure")
SPEECH_ISLAND_SCORER_CONTENT_DIM = len(SPEECH_ISLAND_SCORER_LABELS)
SPEECH_ISLAND_SCORER_MEMBERSHIP_DIM = len(SPEECH_ISLAND_MEMBERSHIP_LABELS)
SPEECH_ISLAND_SCORER_OUTPUT_HEADS = tuple(
    f"content.{label}" for label in SPEECH_ISLAND_SCORER_LABELS
) + tuple(f"membership.{label}" for label in SPEECH_ISLAND_MEMBERSHIP_LABELS)
SPEECH_ISLAND_SCORER_OUTPUT_DIM = len(SPEECH_ISLAND_SCORER_OUTPUT_HEADS)
SPEECH_ISLAND_SCORER_DECODER = "argmax_source_membership_islands_v1"
SPEECH_ISLAND_SCORER_ARTIFACT = {
    "name": "semantic_speech_scorer",
    "display_name": "Semantic Speech Scorer",
    "version": "v9",
    "pipeline_stage": 1,
    "pipeline_role": "semantic_source_membership_detection",
}


def count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _bool_config(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _artifact_contract(schema: str) -> dict[str, Any]:
    if schema == SPEECH_ISLAND_SCORER_SCHEMA:
        return SPEECH_ISLAND_SCORER_ARTIFACT
    if schema == SPEECH_ISLAND_SCORER_V8_SCHEMA:
        return {
            "name": "speech_island_scorer",
            "display_name": "SpeechIslandScorer",
            "version": "v8",
            "pipeline_stage": 1,
            "pipeline_role": "speech_island_detection",
        }
    raise ValueError(f"unsupported scorer checkpoint schema: {schema!r}")


def _output_contract(schema: str) -> tuple[tuple[str, ...], str, int, str, str]:
    if schema == SPEECH_ISLAND_SCORER_SCHEMA:
        return (
            SPEECH_ISLAND_SCORER_OUTPUT_HEADS,
            SPEECH_ISLAND_SCORER_DECODER,
            SPEECH_ISLAND_SCORER_OUTPUT_DIM,
            SPEECH_ISLAND_SCORER_MODEL_ARCH,
            SPEECH_ISLAND_SCORER_MODEL_TYPE,
        )
    if schema == SPEECH_ISLAND_SCORER_V8_SCHEMA:
        return (
            SPEECH_ISLAND_SCORER_V8_OUTPUT_HEADS,
            SPEECH_ISLAND_SCORER_V8_DECODER,
            1,
            SPEECH_ISLAND_SCORER_V8_MODEL_ARCH,
            SPEECH_ISLAND_SCORER_V8_MODEL_TYPE,
        )
    raise ValueError(f"unsupported scorer checkpoint schema: {schema!r}")


def _validate_metadata(metadata: dict[str, Any], *, schema: str) -> None:
    require_boundary_contract_id(
        metadata.get("boundary_serialization_contract_id")
    )
    artifact = metadata.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Speech island scorer checkpoint metadata.artifact is required")
    for key, expected in _artifact_contract(schema).items():
        if artifact.get(key) != expected:
            raise ValueError(
                f"Speech island scorer metadata.artifact.{key} must be {expected!r}"
            )
    expected_heads, expected_decoder, _output_dim, _model_arch, _model_type = _output_contract(schema)
    heads = tuple(str(item) for item in (metadata.get("output_heads") or ()))
    if heads != expected_heads:
        raise ValueError(
            "Speech island scorer checkpoint metadata.output_heads must be "
            f"{list(expected_heads)!r}; got {list(heads)!r}"
        )
    decoder = str(metadata.get("decoder") or "")
    if decoder != expected_decoder:
        raise ValueError(
            "Speech island scorer checkpoint metadata.decoder must be "
            f"{expected_decoder!r}; got {decoder!r}"
        )


class SemanticSpeechScorerNetwork:
    """Task-aware full-PTM projector followed by a semantic frame encoder."""

    def __new__(
        cls,
        *,
        raw_ptm_dim: int,
        projected_ptm_dim: int,
        mfcc_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        state_size: int = 32,
        num_heads: int = 4,
        head_dim: int = 64,
        n_groups: int = 2,
        conv_kernel: int = 4,
        chunk_size: int = 8,
        bidirectional: bool = True,
        output_dim: int = SPEECH_ISLAND_SCORER_OUTPUT_DIM,
        mfcc_mean: Sequence[float] = (),
        mfcc_std: Sequence[float] = (),
        **_unused: Any,
    ):
        import torch
        from torch import nn

        from boundary.backbones import SpeechIslandSequenceClassifier

        if output_dim != SPEECH_ISLAND_SCORER_OUTPUT_DIM:
            raise ValueError(
                f"Semantic Speech Scorer requires output_dim={SPEECH_ISLAND_SCORER_OUTPUT_DIM}"
            )

        class _Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.raw_ptm_dim = int(raw_ptm_dim)
                self.projected_ptm_dim = int(projected_ptm_dim)
                self.mfcc_dim = int(mfcc_dim)
                self.ptm_projector = nn.Linear(
                    self.raw_ptm_dim,
                    self.projected_ptm_dim,
                    bias=True,
                )
                if len(mfcc_mean) != self.mfcc_dim or len(mfcc_std) != self.mfcc_dim:
                    raise ValueError("semantic scorer MFCC normalization length mismatch")
                self.register_buffer(
                    "mfcc_mean",
                    torch.tensor(mfcc_mean, dtype=torch.float32).reshape(1, 1, -1),
                )
                self.register_buffer(
                    "mfcc_std",
                    torch.tensor(mfcc_std, dtype=torch.float32).reshape(1, 1, -1),
                )
                self.encoder = SpeechIslandSequenceClassifier(
                    input_dim=self.projected_ptm_dim + self.mfcc_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_dim=output_dim,
                    state_size=state_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    n_groups=n_groups,
                    conv_kernel=conv_kernel,
                    chunk_size=chunk_size,
                    bidirectional=bidirectional,
                )

            def project_ptm(self, ptm: torch.Tensor) -> torch.Tensor:
                if ptm.ndim != 3 or int(ptm.shape[-1]) != self.raw_ptm_dim:
                    raise ValueError(
                        "semantic scorer PTM input must have shape "
                        f"[batch,frames,{self.raw_ptm_dim}]"
                    )
                return self.ptm_projector(ptm)

            def forward(
                self,
                ptm: torch.Tensor,
                mfcc: torch.Tensor,
                *,
                attention_mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
                if mfcc.ndim != 3 or int(mfcc.shape[-1]) != self.mfcc_dim:
                    raise ValueError(
                        "semantic scorer MFCC input must have shape "
                        f"[batch,frames,{self.mfcc_dim}]"
                    )
                if ptm.shape[:2] != mfcc.shape[:2]:
                    raise ValueError("semantic scorer PTM/MFCC frame shapes differ")
                features = torch.cat(
                    (
                        self.project_ptm(ptm),
                        (mfcc - self.mfcc_mean) / self.mfcc_std.clamp_min(1e-6),
                    ),
                    dim=-1,
                )
                return self.encoder(features, attention_mask=attention_mask)

        return _Network()


def build_speech_island_scorer_model(*, schema: str, model_config: dict[str, Any]):
    if schema == SPEECH_ISLAND_SCORER_SCHEMA:
        for key in (
            "raw_ptm_dim",
            "projected_ptm_dim",
            "mfcc_dim",
            "hidden_size",
            "num_layers",
        ):
            if key not in model_config:
                raise ValueError(f"Semantic scorer checkpoint missing model_config.{key}")
        if str(model_config.get("model_arch") or "") != SPEECH_ISLAND_SCORER_MODEL_ARCH:
            raise ValueError(
                "Semantic scorer model_arch must be "
                f"{SPEECH_ISLAND_SCORER_MODEL_ARCH!r}"
            )
        return SemanticSpeechScorerNetwork(**model_config)
    if schema == SPEECH_ISLAND_SCORER_V8_SCHEMA:
        from boundary.backbones import TRANSFORMERS_MAMBA2_BACKBONE, SpeechIslandSequenceClassifier

        for key in ("input_dim", "hidden_size", "num_layers"):
            if key not in model_config:
                raise ValueError(f"Mamba2 scorer checkpoint missing model_config.{key}")
        _heads, _decoder, expected_output_dim, expected_model_arch, _model_type = _output_contract(schema)
        model_arch = str(model_config.get("model_arch") or "")
        if model_arch != expected_model_arch:
            raise ValueError(
                "Speech island scorer checkpoint model_config.model_arch must be "
                f"{expected_model_arch!r}; got {model_arch!r}"
            )
        backbone_kwargs: dict[str, Any] = {
            "state_size": int(model_config.get("state_size", 32)),
            "num_heads": int(model_config.get("num_heads", 4)),
            "n_groups": int(model_config.get("n_groups", 2)),
            "chunk_size": int(model_config.get("chunk_size", 8)),
            "bidirectional": _bool_config(model_config.get("bidirectional", True)),
            "conv_kernel": int(model_config.get("conv_kernel", 4)),
        }
        if "head_dim" in model_config:
            backbone_kwargs["head_dim"] = int(model_config["head_dim"])
        if "output_dim" not in model_config:
            raise ValueError("Speech island scorer checkpoint missing model_config.output_dim")
        output_dim = int(model_config["output_dim"])
        if output_dim != expected_output_dim:
            raise ValueError(
                f"Speech island scorer requires output_dim={expected_output_dim}, "
                f"got {output_dim}"
            )
        return SpeechIslandSequenceClassifier(
            input_dim=int(model_config["input_dim"]),
            backbone=str(model_config.get("backbone") or TRANSFORMERS_MAMBA2_BACKBONE),
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            output_dim=output_dim,
            **backbone_kwargs,
        )
    raise ValueError(
        f"unsupported scorer checkpoint schema: {schema!r}; "
        f"expected {SPEECH_ISLAND_SCORER_SCHEMA!r} or {SPEECH_ISLAND_SCORER_V8_SCHEMA!r}"
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
class SpeechIslandScorerBundle:
    path: str
    sha256: str
    model: Any
    model_config: dict[str, Any]
    normalization: dict[str, Any]
    metadata: dict[str, Any]
    device: str
    schema: str = SPEECH_ISLAND_SCORER_SCHEMA
    model_type: str = SPEECH_ISLAND_SCORER_MODEL_TYPE

    @property
    def ptm_dim(self) -> int:
        if self.schema == SPEECH_ISLAND_SCORER_SCHEMA:
            return int(self.model_config["raw_ptm_dim"])
        return int(self.model_config["ptm_dim"])

    @property
    def projected_ptm_dim(self) -> int:
        if self.schema == SPEECH_ISLAND_SCORER_SCHEMA:
            return int(self.model_config["projected_ptm_dim"])
        return self.ptm_dim

    @property
    def mfcc_dim(self) -> int:
        return int(self.model_config["mfcc_dim"])

    @property
    def input_dim(self) -> int:
        if self.schema == SPEECH_ISLAND_SCORER_SCHEMA:
            return self.ptm_dim + self.mfcc_dim
        return int(self.model_config["input_dim"])

    @property
    def labels(self) -> tuple[str, ...]:
        return _output_contract(self.schema)[0]

    def signature(self) -> dict[str, Any]:
        model_config: dict[str, Any] = {
            "raw_ptm_dim": self.ptm_dim,
            "projected_ptm_dim": self.projected_ptm_dim,
            "mfcc_dim": self.mfcc_dim,
            "input_dim": self.input_dim,
        }
        if self.schema == SPEECH_ISLAND_SCORER_V8_SCHEMA:
            model_config["ptm_dim"] = self.ptm_dim
        for key in (
            "model_arch",
            "hidden_size",
            "backbone",
            "num_layers",
            "state_size",
            "num_heads",
            "head_dim",
            "n_groups",
            "chunk_size",
            "conv_kernel",
            "bidirectional",
            "output_dim",
            "projection_type",
        ):
            if key in self.model_config:
                value = self.model_config[key]
                if isinstance(value, (bool, int, float, str)):
                    model_config[key] = value
        return {
            "schema": self.schema,
            "model_type": self.model_type,
            "boundary_serialization_contract_id": require_boundary_contract_id(
                self.metadata.get("boundary_serialization_contract_id")
            ),
            "path": self.path,
            "sha256": self.sha256,
            "model_config": model_config,
            "metadata": {
                "operating_point": str(self.metadata.get("operating_point") or ""),
                "ptm_repo_id": str(self.metadata.get("ptm_repo_id") or ""),
                "trained_steps": int(self.metadata.get("trained_steps") or 0),
                "feature_manifest": str(self.metadata.get("feature_manifest") or ""),
                "output_heads": list(self.metadata.get("output_heads") or ()),
                "labels": (
                    list(self.metadata.get("labels") or ())
                    if isinstance(self.metadata.get("labels"), (list, tuple))
                    else str(self.metadata.get("labels") or "")
                ),
                "decoder": str(self.metadata.get("decoder") or ""),
                "artifact": dict(self.metadata.get("artifact") or {}),
            },
        }


def checkpoint_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_speech_island_scorer_checkpoint(
    *,
    model: Any,
    model_config: dict[str, Any],
    normalization: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    schema: str = SPEECH_ISLAND_SCORER_SCHEMA,
) -> dict[str, Any]:
    if schema not in {SPEECH_ISLAND_SCORER_SCHEMA, SPEECH_ISLAND_SCORER_V8_SCHEMA}:
        raise ValueError(
            f"unsupported scorer checkpoint schema: {schema!r}; "
            f"expected {SPEECH_ISLAND_SCORER_SCHEMA!r} or "
            f"{SPEECH_ISLAND_SCORER_V8_SCHEMA!r}"
        )
    heads, decoder, output_dim, _model_arch, model_type = _output_contract(schema)
    if int(model_config.get("output_dim", 0)) != output_dim:
        raise ValueError(
            f"Speech island scorer checkpoint requires output_dim={output_dim}"
        )
    metadata_dict = dict(metadata or {})
    metadata_dict.setdefault(
        "boundary_serialization_contract_id",
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
    )
    metadata_dict["artifact"] = {
        **_artifact_contract(schema),
        **dict(metadata_dict.get("artifact") or {}),
    }
    metadata_dict.setdefault("output_heads", list(heads))
    metadata_dict.setdefault("decoder", decoder)
    if schema == SPEECH_ISLAND_SCORER_SCHEMA:
        metadata_dict.setdefault("labels", list(SPEECH_ISLAND_SCORER_OUTPUT_HEADS))
        metadata_dict.setdefault("content_labels", list(SPEECH_ISLAND_SCORER_LABELS))
        metadata_dict.setdefault(
            "membership_labels", list(SPEECH_ISLAND_MEMBERSHIP_LABELS)
        )
        metadata_dict.setdefault("decision_mode", "argmax")
    _validate_metadata(metadata_dict, schema=schema)
    return {
        "schema": schema,
        "model_type": model_type,
        "model_config": dict(model_config),
        "normalization": dict(normalization),
        "metadata": metadata_dict,
        "model_state_dict": model.state_dict(),
    }


def load_speech_island_scorer_checkpoint(
    path: str | Path,
    *,
    device: str | Any = "cpu",
) -> SpeechIslandScorerBundle:
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
    if schema not in {SPEECH_ISLAND_SCORER_SCHEMA, SPEECH_ISLAND_SCORER_V8_SCHEMA}:
        raise ValueError(
            f"unsupported scorer checkpoint schema: {payload.get('schema')!r}; "
            f"expected {SPEECH_ISLAND_SCORER_SCHEMA!r} or "
            f"{SPEECH_ISLAND_SCORER_V8_SCHEMA!r}"
        )
    _heads, _decoder, _output_dim, _model_arch, expected_model_type = _output_contract(schema)
    model_type = str(payload.get("model_type") or "")
    if model_type != expected_model_type:
        raise ValueError(
            f"unsupported scorer checkpoint model_type: {payload.get('model_type')!r}; "
            f"expected {expected_model_type!r}"
        )
    model_config = dict(payload.get("model_config") or {})
    required_config = (
        ("raw_ptm_dim", "projected_ptm_dim", "mfcc_dim")
        if schema == SPEECH_ISLAND_SCORER_SCHEMA
        else ("ptm_dim", "mfcc_dim", "input_dim")
    )
    for key in required_config:
        if key not in model_config:
            raise ValueError(f"scorer checkpoint missing model_config.{key}")
    if schema == SPEECH_ISLAND_SCORER_V8_SCHEMA and int(model_config["input_dim"]) != int(model_config["ptm_dim"]) + int(model_config["mfcc_dim"]):
        raise ValueError("scorer checkpoint input_dim does not match ptm_dim + mfcc_dim")
    metadata = dict(payload.get("metadata") or {})
    _validate_metadata(metadata, schema=schema)
    normalization = dict(payload.get("normalization") or {})
    if schema == SPEECH_ISLAND_SCORER_V8_SCHEMA:
        mean = list(normalization.get("feature_mean") or [])
        std = list(normalization.get("feature_std") or [])
        if len(mean) != int(model_config["input_dim"]) or len(std) != int(model_config["input_dim"]):
            raise ValueError("scorer checkpoint normalization length does not match input_dim")
    model = build_speech_island_scorer_model(schema=schema, model_config=model_config)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("scorer checkpoint missing model_state_dict")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return SpeechIslandScorerBundle(
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


def score_speech_island_probabilities(
    bundle: SpeechIslandScorerBundle,
    *,
    ptm: np.ndarray,
    mfcc: np.ndarray,
) -> np.ndarray:
    scored = score_speech_island_probabilities_batch(
        bundle,
        feature_pairs=[(ptm, mfcc)],
    )
    if not scored:
        return np.zeros(0, dtype=np.float32)
    return scored[0]


def score_speech_island_probabilities_batch(
    bundle: SpeechIslandScorerBundle,
    *,
    feature_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[np.ndarray]:
    import torch

    pairs = list(feature_pairs)
    if not pairs:
        return []
    if int(bundle.model_config.get("output_dim", 0)) != 1:
        raise ValueError(
            "legacy speech/proposal scorer expected output_dim=1, "
            f"got {bundle.model_config.get('output_dim')}"
        )
    mean = np.asarray(bundle.normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(bundle.normalization["feature_std"], dtype=np.float32)
    lengths: list[int] = []
    normalized_rows: list[np.ndarray] = []
    for ptm, mfcc in pairs:
        frame_total = min(int(ptm.shape[0]), int(mfcc.shape[0]))
        if int(ptm.shape[1]) < bundle.ptm_dim:
            raise ValueError(
                f"scorer expected at least ptm_dim={bundle.ptm_dim}, "
                f"got {int(ptm.shape[1])}"
            )
        if int(mfcc.shape[1]) != bundle.mfcc_dim:
            raise ValueError(f"scorer expected mfcc_dim={bundle.mfcc_dim}, got {int(mfcc.shape[1])}")
        lengths.append(frame_total)
        if frame_total <= 0:
            normalized_rows.append(np.zeros((0, bundle.input_dim), dtype=np.float32))
            continue
        features = np.concatenate(
            [
                np.asarray(ptm[:frame_total, : bundle.ptm_dim], dtype=np.float32),
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
        return [empty for _length in lengths]
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
    if probabilities.ndim != 3 or probabilities.shape[2] != 1:
        raise ValueError(
            "speech island scorer probabilities must have shape "
            "[batch, frames, 1]"
        )
    outputs: list[np.ndarray] = []
    for index, length in enumerate(lengths):
        length = int(length)
        if length <= 0:
            empty = np.zeros(0, dtype=np.float32)
            outputs.append(empty)
            continue
        outputs.append(
            np.ascontiguousarray(probabilities[index, :length, 0], dtype=np.float32)
        )
    return outputs


def score_semantic_speech_class_probabilities(
    bundle: SpeechIslandScorerBundle,
    *,
    ptm: np.ndarray,
    mfcc: np.ndarray,
) -> np.ndarray:
    scored = score_semantic_speech_class_probabilities_batch(
        bundle,
        feature_pairs=[(ptm, mfcc)],
    )
    if not scored:
        return np.zeros((0, SPEECH_ISLAND_SCORER_CONTENT_DIM), dtype=np.float32)
    return scored[0]


def score_semantic_speech_outputs(
    bundle: SpeechIslandScorerBundle,
    *,
    ptm: np.ndarray,
    mfcc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scored = score_semantic_speech_outputs_batch(
        bundle,
        feature_pairs=[(ptm, mfcc)],
    )
    if not scored:
        return (
            np.zeros((0, SPEECH_ISLAND_SCORER_CONTENT_DIM), dtype=np.float32),
            np.zeros((0, SPEECH_ISLAND_SCORER_MEMBERSHIP_DIM), dtype=np.float32),
            np.zeros((0, bundle.projected_ptm_dim), dtype=np.float32),
        )
    return scored[0]


def score_semantic_speech_class_probabilities_batch(
    bundle: SpeechIslandScorerBundle,
    *,
    feature_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[np.ndarray]:
    return [
        probabilities
        for probabilities, _membership, _projected in score_semantic_speech_outputs_batch(
            bundle,
            feature_pairs=feature_pairs,
        )
    ]


def score_semantic_speech_outputs_batch(
    bundle: SpeechIslandScorerBundle,
    *,
    feature_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    import torch

    if bundle.schema != SPEECH_ISLAND_SCORER_SCHEMA:
        raise ValueError(
            f"semantic speech scoring requires {SPEECH_ISLAND_SCORER_SCHEMA!r}"
        )
    pairs = list(feature_pairs)
    if not pairs:
        return []
    lengths: list[int] = []
    ptm_rows: list[np.ndarray] = []
    mfcc_rows: list[np.ndarray] = []
    for ptm, mfcc in pairs:
        frame_total = min(int(ptm.shape[0]), int(mfcc.shape[0]))
        if int(ptm.shape[1]) != bundle.ptm_dim:
            raise ValueError(
                f"semantic scorer expected raw_ptm_dim={bundle.ptm_dim}, got {int(ptm.shape[1])}"
            )
        if int(mfcc.shape[1]) != bundle.mfcc_dim:
            raise ValueError(
                f"scorer expected mfcc_dim={bundle.mfcc_dim}, got {int(mfcc.shape[1])}"
            )
        lengths.append(frame_total)
        ptm_rows.append(np.asarray(ptm[:frame_total], dtype=np.float32))
        mfcc_rows.append(np.asarray(mfcc[:frame_total], dtype=np.float32))
    max_len = max(lengths, default=0)
    if max_len <= 0:
        return [
            (
                np.zeros((0, SPEECH_ISLAND_SCORER_CONTENT_DIM), dtype=np.float32),
                np.zeros((0, SPEECH_ISLAND_SCORER_MEMBERSHIP_DIM), dtype=np.float32),
                np.zeros((0, bundle.projected_ptm_dim), dtype=np.float32),
            )
            for _ in ptm_rows
        ]
    ptm_batch = np.zeros(
        (len(ptm_rows), max_len, bundle.ptm_dim), dtype=np.float32
    )
    mfcc_batch = np.zeros(
        (len(mfcc_rows), max_len, bundle.mfcc_dim), dtype=np.float32
    )
    mask = np.zeros((len(ptm_rows), max_len), dtype=np.int64)
    for index, (ptm, mfcc) in enumerate(zip(ptm_rows, mfcc_rows, strict=True)):
        ptm_batch[index, : ptm.shape[0]] = ptm
        mfcc_batch[index, : mfcc.shape[0]] = mfcc
        mask[index, : ptm.shape[0]] = 1
    with torch.inference_mode():
        ptm_tensor = torch.from_numpy(ptm_batch).to(bundle.device)
        mfcc_tensor = torch.from_numpy(mfcc_batch).to(bundle.device)
        attention_mask = torch.from_numpy(mask).to(bundle.device)
        projected = bundle.model.project_ptm(ptm_tensor)
        logits = bundle.model(
            ptm_tensor,
            mfcc_tensor,
            attention_mask=attention_mask,
        )
        content_logits = logits[..., :SPEECH_ISLAND_SCORER_CONTENT_DIM]
        membership_logits = logits[..., SPEECH_ISLAND_SCORER_CONTENT_DIM:]
        content_probabilities = (
            torch.softmax(content_logits, dim=-1).detach().cpu().numpy().astype(np.float32)
        )
        membership_probabilities = (
            torch.softmax(membership_logits, dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        projected_values = projected.detach().cpu().numpy().astype(np.float32)
    return [
        (
            np.ascontiguousarray(
                content_probabilities[index, :length], dtype=np.float32
            ),
            np.ascontiguousarray(
                membership_probabilities[index, :length], dtype=np.float32
            ),
            np.ascontiguousarray(projected_values[index, :length], dtype=np.float32),
        )
        for index, length in enumerate(lengths)
    ]
