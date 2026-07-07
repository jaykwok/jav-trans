from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from boundary.ja.model import checkpoint_sha256


BOUNDARY_PROPOSAL_SCORER_SCHEMA = "speech_boundary_ja_boundary_proposal_scorer_v1"
BOUNDARY_PROPOSAL_SCORER_MODEL_TYPE = "mamba2_boundary_proposal_scorer"
BOUNDARY_PROPOSAL_SCORER_MODEL_ARCH = "v1-boundary-proposal"
BOUNDARY_PROPOSAL_SCORER_OUTPUT_DIM = 1
BOUNDARY_PROPOSAL_SCORER_OUTPUT_HEADS = ("boundary_prob",)
BOUNDARY_PROPOSAL_SCORER_ARTIFACT = {
    "name": "boundary_proposal_scorer",
    "display_name": "BoundaryProposalScorer",
    "version": "v1",
    "pipeline_stage": 1,
    "pipeline_role": "split_candidate_proposal",
}


@dataclass(frozen=True)
class BoundaryProposalScorerBundle:
    """Learned dense boundary scorer feeding the split peak decoder.

    Exposes the same attribute surface as ``SpeechIslandScorerBundle`` so
    ``score_speech_island_probabilities_batch`` can run it unchanged. It only
    PROPOSES candidates (high recall); accept/reject stays with the Semantic
    Split verifier — unlike the retired v7 dual-head scorer whose split head
    made final cuts.
    """

    path: str
    sha256: str
    model: Any
    model_config: dict[str, Any]
    normalization: dict[str, Any]
    metadata: dict[str, Any]
    device: str
    schema: str = BOUNDARY_PROPOSAL_SCORER_SCHEMA
    model_type: str = BOUNDARY_PROPOSAL_SCORER_MODEL_TYPE

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
            "schema": self.schema,
            "model_type": self.model_type,
            "path": self.path,
            "sha256": self.sha256,
            "model_config": {
                key: value
                for key, value in self.model_config.items()
                if isinstance(value, (bool, int, float, str))
            },
            "metadata": {
                "ptm_repo_id": str(self.metadata.get("ptm_repo_id") or ""),
                "trained_steps": int(self.metadata.get("trained_steps") or 0),
                "artifact": dict(self.metadata.get("artifact") or {}),
            },
        }


def build_boundary_proposal_checkpoint(
    *,
    model: Any,
    model_config: dict[str, Any],
    normalization: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if int(model_config.get("output_dim", 0)) != BOUNDARY_PROPOSAL_SCORER_OUTPUT_DIM:
        raise ValueError("boundary proposal scorer requires output_dim=1")
    metadata_dict = dict(metadata or {})
    metadata_dict["artifact"] = {
        **BOUNDARY_PROPOSAL_SCORER_ARTIFACT,
        **dict(metadata_dict.get("artifact") or {}),
    }
    metadata_dict.setdefault(
        "output_heads", list(BOUNDARY_PROPOSAL_SCORER_OUTPUT_HEADS)
    )
    return {
        "schema": BOUNDARY_PROPOSAL_SCORER_SCHEMA,
        "model_type": BOUNDARY_PROPOSAL_SCORER_MODEL_TYPE,
        "model_config": dict(model_config),
        "normalization": dict(normalization),
        "metadata": metadata_dict,
        "model_state_dict": model.state_dict(),
    }


def load_boundary_proposal_checkpoint(
    path: str | Path,
    *,
    device: str | Any = "cpu",
) -> BoundaryProposalScorerBundle:
    import torch

    from boundary.backbones import SpeechIslandSequenceClassifier

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"boundary proposal scorer checkpoint not found: {checkpoint_path}"
        )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("schema") != BOUNDARY_PROPOSAL_SCORER_SCHEMA:
        raise ValueError(
            f"unsupported proposal checkpoint schema: {payload.get('schema')!r}; "
            f"expected {BOUNDARY_PROPOSAL_SCORER_SCHEMA!r}"
        )
    if payload.get("model_type") != BOUNDARY_PROPOSAL_SCORER_MODEL_TYPE:
        raise ValueError(
            f"unsupported proposal checkpoint model_type: {payload.get('model_type')!r}"
        )
    model_config = dict(payload.get("model_config") or {})
    for key in ("ptm_dim", "mfcc_dim", "input_dim", "hidden_size", "num_layers"):
        if key not in model_config:
            raise ValueError(f"proposal checkpoint missing model_config.{key}")
    if int(model_config["input_dim"]) != int(model_config["ptm_dim"]) + int(
        model_config["mfcc_dim"]
    ):
        raise ValueError("proposal checkpoint input_dim mismatch")
    if int(model_config.get("output_dim", 0)) != BOUNDARY_PROPOSAL_SCORER_OUTPUT_DIM:
        raise ValueError("proposal checkpoint requires output_dim=1")
    metadata = dict(payload.get("metadata") or {})
    artifact = metadata.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("proposal checkpoint metadata.artifact is required")
    for key, expected in BOUNDARY_PROPOSAL_SCORER_ARTIFACT.items():
        if artifact.get(key) != expected:
            raise ValueError(
                f"proposal checkpoint metadata.artifact.{key} must be {expected!r}"
            )
    normalization = dict(payload.get("normalization") or {})
    if len(list(normalization.get("feature_mean") or [])) != int(
        model_config["input_dim"]
    ):
        raise ValueError("proposal checkpoint normalization length mismatch")
    model = SpeechIslandSequenceClassifier(
        input_dim=int(model_config["input_dim"]),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        output_dim=BOUNDARY_PROPOSAL_SCORER_OUTPUT_DIM,
        state_size=int(model_config.get("state_size", 32)),
        num_heads=int(model_config.get("num_heads", 4)),
        n_groups=int(model_config.get("n_groups", 2)),
        chunk_size=int(model_config.get("chunk_size", 8)),
        conv_kernel=int(model_config.get("conv_kernel", 4)),
        bidirectional=bool(model_config.get("bidirectional", True)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return BoundaryProposalScorerBundle(
        path=str(checkpoint_path),
        sha256=checkpoint_sha256(checkpoint_path),
        model=model,
        model_config=model_config,
        normalization=normalization,
        metadata=metadata,
        device=str(device),
    )
