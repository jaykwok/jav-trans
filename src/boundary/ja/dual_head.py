from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from boundary.backbones import SpeechIslandSequenceClassifier

SPEECH_PROPOSAL_DUAL_HEAD_SCHEMA = "speech_boundary_ja_speech_proposal_dual_head_v1"
SPEECH_PROPOSAL_DUAL_HEAD_MODEL_TYPE = "mamba2_speech_proposal_dual_head"
SPEECH_PROPOSAL_DUAL_HEAD_MODEL_ARCH = "v1-speech-proposal-dual-head"
SPEECH_PROPOSAL_DUAL_HEAD_NORMALIZATION_SCHEMA = (
    "speech_boundary_ja_speech_proposal_dual_head_normalization_v1"
)
SPEECH_PROPOSAL_DUAL_HEAD_OUTPUT_DIM = 2
SPEECH_PROPOSAL_DUAL_HEAD_OUTPUTS = ("speech_prob", "boundary_prob")
SPEECH_PROPOSAL_DUAL_HEAD_ARTIFACT = {
    "name": "speech_proposal_dual_head",
    "display_name": "Speech Island + Boundary Proposal Dual Head",
    "version": "v1",
    "pipeline_stage": 1,
    "pipeline_role": "speech_island_and_split_candidate_proposal",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_config(model_config: dict[str, Any]) -> None:
    if str(model_config.get("model_arch") or "") != SPEECH_PROPOSAL_DUAL_HEAD_MODEL_ARCH:
        raise ValueError("dual-head checkpoint model_arch mismatch")
    if int(model_config.get("output_dim") or 0) != SPEECH_PROPOSAL_DUAL_HEAD_OUTPUT_DIM:
        raise ValueError("dual-head checkpoint requires output_dim=2")
    raw_ptm_dim = int(model_config.get("raw_ptm_dim") or 0)
    projected_ptm_dim = int(model_config.get("projected_ptm_dim") or 0)
    mfcc_dim = int(model_config.get("mfcc_dim") or 0)
    if raw_ptm_dim <= 0 or projected_ptm_dim <= 0 or mfcc_dim <= 0:
        raise ValueError("dual-head raw/projected PTM and MFCC dimensions must be positive")
    if projected_ptm_dim > raw_ptm_dim:
        raise ValueError("dual-head projected_ptm_dim exceeds raw_ptm_dim")
    if int(model_config.get("input_dim") or 0) != raw_ptm_dim + mfcc_dim:
        raise ValueError("dual-head input_dim does not match raw_ptm_dim + mfcc_dim")
    if int(model_config.get("encoder_input_dim") or 0) != projected_ptm_dim + mfcc_dim:
        raise ValueError(
            "dual-head encoder_input_dim does not match projected_ptm_dim + mfcc_dim"
        )


def _validate_normalization(
    normalization: dict[str, Any], *, mfcc_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    if str(normalization.get("schema") or "") != SPEECH_PROPOSAL_DUAL_HEAD_NORMALIZATION_SCHEMA:
        raise ValueError("dual-head normalization schema mismatch")
    mean = np.asarray(normalization.get("mfcc_mean"), dtype=np.float32)
    std = np.asarray(normalization.get("mfcc_std"), dtype=np.float32)
    if mean.shape != (mfcc_dim,) or std.shape != (mfcc_dim,):
        raise ValueError("dual-head MFCC normalization dimension mismatch")
    return mean, np.maximum(std, 1e-6)


class SpeechProposalDualHeadClassifier(nn.Module):
    """Frame-level speech/proposal model with a learned full-PTM projection."""

    def __init__(
        self,
        *,
        model_config: dict[str, Any],
        normalization: dict[str, Any],
    ) -> None:
        super().__init__()
        _validate_config(model_config)
        raw_ptm_dim = int(model_config["raw_ptm_dim"])
        projected_ptm_dim = int(model_config["projected_ptm_dim"])
        mfcc_dim = int(model_config["mfcc_dim"])
        mfcc_mean, mfcc_std = _validate_normalization(
            normalization, mfcc_dim=mfcc_dim
        )
        self.raw_ptm_dim = raw_ptm_dim
        self.projected_ptm_dim = projected_ptm_dim
        self.mfcc_dim = mfcc_dim
        self.ptm_projector = nn.Linear(
            raw_ptm_dim, projected_ptm_dim, bias=True
        )
        self.register_buffer(
            "mfcc_mean", torch.from_numpy(mfcc_mean).reshape(1, 1, -1)
        )
        self.register_buffer(
            "mfcc_std", torch.from_numpy(mfcc_std).reshape(1, 1, -1)
        )
        self.encoder = SpeechIslandSequenceClassifier(
            input_dim=int(model_config["encoder_input_dim"]),
            hidden_size=int(model_config.get("hidden_size", 128)),
            num_layers=int(model_config.get("num_layers", 2)),
            output_dim=SPEECH_PROPOSAL_DUAL_HEAD_OUTPUT_DIM,
            state_size=int(model_config.get("state_size", 32)),
            num_heads=int(model_config.get("num_heads", 4)),
            n_groups=int(model_config.get("n_groups", 2)),
            chunk_size=int(model_config.get("chunk_size", 8)),
            conv_kernel=int(model_config.get("conv_kernel", 4)),
            bidirectional=bool(model_config.get("bidirectional", True)),
        )

    def forward(
        self,
        ptm: torch.Tensor,
        mfcc: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ptm.ndim != 3 or int(ptm.shape[-1]) != self.raw_ptm_dim:
            raise ValueError(
                f"dual-head PTM input must have shape [batch,frames,{self.raw_ptm_dim}]"
            )
        if mfcc.ndim != 3 or int(mfcc.shape[-1]) != self.mfcc_dim:
            raise ValueError(
                f"dual-head MFCC input must have shape [batch,frames,{self.mfcc_dim}]"
            )
        if ptm.shape[:2] != mfcc.shape[:2]:
            raise ValueError("dual-head PTM/MFCC frame shapes differ")
        features = torch.cat(
            (
                self.ptm_projector(ptm),
                (mfcc - self.mfcc_mean) / self.mfcc_std,
            ),
            dim=-1,
        )
        return self.encoder(features, attention_mask=attention_mask)


def build_dual_head_model(
    model_config: dict[str, Any], normalization: dict[str, Any]
) -> SpeechProposalDualHeadClassifier:
    return SpeechProposalDualHeadClassifier(
        model_config=model_config,
        normalization=normalization,
    )


@dataclass(frozen=True)
class SpeechProposalDualHeadBundle:
    path: str
    sha256: str
    model: SpeechProposalDualHeadClassifier
    model_config: dict[str, Any]
    normalization: dict[str, Any]
    metadata: dict[str, Any]
    device: str

    @property
    def raw_ptm_dim(self) -> int:
        return int(self.model_config["raw_ptm_dim"])

    @property
    def projected_ptm_dim(self) -> int:
        return int(self.model_config["projected_ptm_dim"])

    @property
    def mfcc_dim(self) -> int:
        return int(self.model_config["mfcc_dim"])

    @property
    def input_dim(self) -> int:
        return int(self.model_config["input_dim"])

    def signature(self) -> dict[str, Any]:
        return {
            "schema": SPEECH_PROPOSAL_DUAL_HEAD_SCHEMA,
            "model_type": SPEECH_PROPOSAL_DUAL_HEAD_MODEL_TYPE,
            "path": self.path,
            "sha256": self.sha256,
            "model_config": dict(self.model_config),
            "metadata": {
                "ptm_repo_id": str(self.metadata.get("ptm_repo_id") or ""),
                "trained_steps": int(self.metadata.get("trained_steps") or 0),
                "output_heads": list(self.metadata.get("output_heads") or ()),
                "ptm_projection_init": str(
                    self.metadata.get("ptm_projection_init") or ""
                ),
                "artifact": dict(self.metadata.get("artifact") or {}),
            },
        }


def build_dual_head_checkpoint(
    *,
    model: SpeechProposalDualHeadClassifier,
    model_config: dict[str, Any],
    normalization: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    _validate_config(model_config)
    _validate_normalization(
        normalization, mfcc_dim=int(model_config["mfcc_dim"])
    )
    metadata = {
        **dict(metadata),
        "artifact": {
            **SPEECH_PROPOSAL_DUAL_HEAD_ARTIFACT,
            **dict(dict(metadata).get("artifact") or {}),
        },
        "output_heads": list(SPEECH_PROPOSAL_DUAL_HEAD_OUTPUTS),
    }
    return {
        "schema": SPEECH_PROPOSAL_DUAL_HEAD_SCHEMA,
        "model_type": SPEECH_PROPOSAL_DUAL_HEAD_MODEL_TYPE,
        "model_config": dict(model_config),
        "normalization": dict(normalization),
        "metadata": metadata,
        "model_state_dict": model.state_dict(),
    }


def load_dual_head_checkpoint(
    path: str | Path, *, device: str | Any = "cpu"
) -> SpeechProposalDualHeadBundle:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if str(payload.get("schema") or "") != SPEECH_PROPOSAL_DUAL_HEAD_SCHEMA:
        raise ValueError("dual-head checkpoint schema mismatch")
    if str(payload.get("model_type") or "") != SPEECH_PROPOSAL_DUAL_HEAD_MODEL_TYPE:
        raise ValueError("dual-head checkpoint model_type mismatch")
    model_config = dict(payload.get("model_config") or {})
    normalization = dict(payload.get("normalization") or {})
    _validate_config(model_config)
    _validate_normalization(
        normalization, mfcc_dim=int(model_config["mfcc_dim"])
    )
    metadata = dict(payload.get("metadata") or {})
    if list(metadata.get("output_heads") or ()) != list(
        SPEECH_PROPOSAL_DUAL_HEAD_OUTPUTS
    ):
        raise ValueError("dual-head output_heads mismatch")
    model = build_dual_head_model(model_config, normalization)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return SpeechProposalDualHeadBundle(
        path=str(checkpoint_path),
        sha256=_sha256(checkpoint_path),
        model=model,
        model_config=model_config,
        normalization=normalization,
        metadata=metadata,
        device=str(device),
    )


def score_dual_head_probabilities_batch(
    bundle: SpeechProposalDualHeadBundle,
    *,
    feature_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = list(feature_pairs)
    if not pairs:
        return []
    lengths: list[int] = []
    ptm_rows: list[np.ndarray] = []
    mfcc_rows: list[np.ndarray] = []
    for ptm, mfcc in pairs:
        count = min(int(ptm.shape[0]), int(mfcc.shape[0]))
        if int(ptm.shape[1]) != bundle.raw_ptm_dim:
            raise ValueError("dual-head raw PTM dimension mismatch")
        if int(mfcc.shape[1]) != bundle.mfcc_dim:
            raise ValueError("dual-head MFCC dimension mismatch")
        lengths.append(count)
        ptm_rows.append(np.asarray(ptm[:count], dtype=np.float32))
        mfcc_rows.append(np.asarray(mfcc[:count], dtype=np.float32))
    max_length = max(lengths, default=0)
    ptm_batch = np.zeros(
        (len(ptm_rows), max_length, bundle.raw_ptm_dim), dtype=np.float32
    )
    mfcc_batch = np.zeros(
        (len(mfcc_rows), max_length, bundle.mfcc_dim), dtype=np.float32
    )
    mask = np.zeros((len(ptm_rows), max_length), dtype=np.int64)
    for index, (ptm, mfcc) in enumerate(zip(ptm_rows, mfcc_rows, strict=True)):
        ptm_batch[index, : ptm.shape[0]] = ptm
        mfcc_batch[index, : mfcc.shape[0]] = mfcc
        mask[index, : ptm.shape[0]] = 1
    with torch.inference_mode():
        logits = bundle.model(
            torch.from_numpy(ptm_batch).to(bundle.device),
            torch.from_numpy(mfcc_batch).to(bundle.device),
            attention_mask=torch.from_numpy(mask).to(bundle.device),
        )
        probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    return [
        (
            np.ascontiguousarray(probabilities[index, :length, 0]),
            np.ascontiguousarray(probabilities[index, :length, 1]),
        )
        for index, length in enumerate(lengths)
    ]
