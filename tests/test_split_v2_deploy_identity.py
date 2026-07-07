from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boundary.sequence_features import PTM_PROJECTION_SCHEMA, ptm_projection_digest
from boundary.split_model import (
    SEMANTIC_SPLIT_FEATURE_SCHEMA,
    IslandCandidateSequenceNetwork,
    build_semantic_split_island_checkpoint,
    load_semantic_split_verifier,
)
from tools.boundary.ja.export_split_v2_deploy_checkpoint import (
    build_deploy_checkpoint,
)

# Tiny projection-trained v2 topology: 8-dim PTM -> 4-dim projector, 2-dim MFCC,
# so frame_dim=6 and the raw input is 10-dim. 5 bins (left 2 / gap 1 / right 2).
PTM_INPUT_DIM = 8
PTM_PROJECTED_DIM = 4
MFCC_DIM = 2
FRAME_DIM = PTM_PROJECTED_DIM + MFCC_DIM  # 6
RAW_DIM = PTM_INPUT_DIM + MFCC_DIM  # 10
SCALAR_DIM = 3


def _model_config() -> dict:
    return {
        "frame_dim": FRAME_DIM,
        "scalar_dim": SCALAR_DIM,
        "ptm_input_dim": PTM_INPUT_DIM,
        "ptm_projected_dim": PTM_PROJECTED_DIM,
        "left_bins": 2,
        "gap_bins": 1,
        "right_bins": 2,
        "extra_scale_bins": [],
        "candidate_layers": 1,
        "island_layers": 1,
    }


def _build_source_checkpoint(
    tmp_path: Path, *, seed: int
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Build a projection-trained v2 checkpoint (in-model projector, full-dim
    normalization, no embedded projection) mirroring how the trainer saves P."""

    import torch

    rng = np.random.default_rng(seed)
    model = IslandCandidateSequenceNetwork(**_model_config())
    projector_weight = rng.normal(size=(PTM_PROJECTED_DIM, PTM_INPUT_DIM)).astype(
        np.float32
    )
    with torch.no_grad():
        model.ptm_projector.weight.copy_(torch.from_numpy(projector_weight))
    frame_mean = rng.normal(size=RAW_DIM).astype(np.float32)
    frame_std = (rng.uniform(0.5, 2.0, size=RAW_DIM)).astype(np.float32)
    scalar_mean = rng.normal(size=SCALAR_DIM).astype(np.float32)
    scalar_std = (rng.uniform(0.5, 2.0, size=SCALAR_DIM)).astype(np.float32)
    source_path = tmp_path / "source.pt"
    payload = build_semantic_split_island_checkpoint(
        model=model,
        model_config=_model_config(),
        feature_config={
            "schema": SEMANTIC_SPLIT_FEATURE_SCHEMA,
            "ptm_dim": PTM_INPUT_DIM,
            "mfcc_dim": MFCC_DIM,
            "left_context_s": 0.6,
            "right_context_s": 0.6,
            "gap_context_s": 0.3,
            "left_bins": 2,
            "gap_bins": 1,
            "right_bins": 2,
            "extra_context_scales": [],
        },
        normalization={
            "frame_mean": frame_mean,
            "frame_std": frame_std,
            "scalar_mean": scalar_mean,
            "scalar_std": scalar_std,
        },
        metadata={"ptm_repo_id": "test/repo"},
        decision_config=None,
    )
    torch.save(payload, source_path)
    # Folded affine exactly as the trainer exports it.
    mean = frame_mean[:PTM_INPUT_DIM]
    components = (projector_weight / frame_std[:PTM_INPUT_DIM][None, :]).astype(
        np.float32
    )
    projection_npz = tmp_path / "learned_ptm_projection.npz"
    np.savez(
        projection_npz,
        schema=np.asarray(PTM_PROJECTION_SCHEMA),
        mean=mean,
        components=components,
    )
    return source_path, mean, components


def _decide(verifier, frames: np.ndarray, scalars: np.ndarray) -> list[list]:
    return verifier.decide_islands(
        island_frame_features=[np.asarray(f) for f in frames],
        island_scalar_features=[np.asarray(s) for s in scalars],
    )


def test_deploy_chain_matches_in_model_chain(tmp_path: Path) -> None:
    source_path, mean, components = _build_source_checkpoint(tmp_path, seed=7)
    deploy_path = tmp_path / "deploy.pt"
    build_deploy_checkpoint(source_path, tmp_path / "learned_ptm_projection.npz", deploy_path)

    source = load_semantic_split_verifier(source_path, device="cpu")
    deploy = load_semantic_split_verifier(deploy_path, device="cpu")

    # The stripped model must have no in-model projector and the PTM-block
    # normalization must be identity.
    assert deploy.model_config["ptm_input_dim"] == 0
    assert deploy.model_config["ptm_projected_dim"] == 0
    assert "ptm_projector.weight" not in deploy.model.state_dict()
    deploy_mean = np.asarray(deploy.normalization["frame_mean"])
    deploy_std = np.asarray(deploy.normalization["frame_std"])
    assert deploy_mean.shape == (FRAME_DIM,)
    assert np.all(deploy_mean[:PTM_PROJECTED_DIM] == 0.0)
    assert np.all(deploy_std[:PTM_PROJECTED_DIM] == 1.0)
    # MFCC block normalization preserved from the source.
    src_mean = np.asarray(source.normalization["frame_mean"])
    src_std = np.asarray(source.normalization["frame_std"])
    np.testing.assert_allclose(
        deploy_mean[PTM_PROJECTED_DIM:], src_mean[PTM_INPUT_DIM:], atol=1e-6
    )
    np.testing.assert_allclose(
        deploy_std[PTM_PROJECTED_DIM:], src_std[PTM_INPUT_DIM:], atol=1e-6
    )
    # The affine is embedded for the runtime pre-projection path.
    embedded = deploy.feature_config["ptm_projection"]
    assert embedded["digest"] == ptm_projection_digest(mean, components)

    rng = np.random.default_rng(123)
    islands, candidates, bins = 5, 4, 5
    raw = rng.normal(size=(islands, candidates, bins, RAW_DIM)).astype(np.float32)
    scalars = rng.normal(size=(islands, candidates, SCALAR_DIM)).astype(np.float32)

    # In-model chain: source consumes the raw 10-dim frame.
    src_decisions = _decide(source, raw, scalars)

    # Deploy chain: pre-project the PTM block with the folded affine, leave MFCC
    # raw (the deploy verifier normalizes the MFCC block itself).
    ptm_block = raw[..., :PTM_INPUT_DIM]
    mfcc_block = raw[..., PTM_INPUT_DIM:]
    projected = (ptm_block - mean) @ components.T
    deploy_frames = np.concatenate([projected, mfcc_block], axis=-1)
    deploy_decisions = _decide(deploy, deploy_frames, scalars)

    max_gate_drift = 0.0
    for src_row, deploy_row in zip(src_decisions, deploy_decisions):
        assert len(src_row) == len(deploy_row)
        for src_dec, deploy_dec in zip(src_row, deploy_row):
            assert src_dec.label == deploy_dec.label
            assert src_dec.role == deploy_dec.role
            max_gate_drift = max(max_gate_drift, abs(src_dec.p_cut - deploy_dec.p_cut))
    assert max_gate_drift < 1e-4, f"gate drift {max_gate_drift:.2e} exceeds 1e-4"


def test_deploy_rejects_stale_affine(tmp_path: Path) -> None:
    source_path, _mean, _components = _build_source_checkpoint(tmp_path, seed=11)
    # Hand-craft an affine whose components do NOT reproduce weight/std.
    stale = tmp_path / "stale.npz"
    np.savez(
        stale,
        schema=np.asarray(PTM_PROJECTION_SCHEMA),
        mean=np.zeros(PTM_INPUT_DIM, dtype=np.float32),
        components=np.full(
            (PTM_PROJECTED_DIM, PTM_INPUT_DIM), 0.123, dtype=np.float32
        ),
    )
    with pytest.raises(ValueError, match="do not reproduce"):
        build_deploy_checkpoint(source_path, stale, tmp_path / "deploy.pt")


def test_deploy_rejects_projector_free_checkpoint(tmp_path: Path) -> None:
    import torch

    # A checkpoint already without a projector has nothing to fold.
    model = IslandCandidateSequenceNetwork(
        frame_dim=PTM_PROJECTED_DIM + MFCC_DIM,
        scalar_dim=SCALAR_DIM,
        left_bins=2,
        gap_bins=1,
        right_bins=2,
        candidate_layers=1,
        island_layers=1,
    )
    flat_path = tmp_path / "flat.pt"
    payload = build_semantic_split_island_checkpoint(
        model=model,
        model_config={
            "frame_dim": PTM_PROJECTED_DIM + MFCC_DIM,
            "scalar_dim": SCALAR_DIM,
            "ptm_input_dim": 0,
            "ptm_projected_dim": 0,
            "left_bins": 2,
            "gap_bins": 1,
            "right_bins": 2,
            "extra_scale_bins": [],
            "candidate_layers": 1,
            "island_layers": 1,
        },
        feature_config={"schema": SEMANTIC_SPLIT_FEATURE_SCHEMA},
        normalization={
            "frame_mean": np.zeros(FRAME_DIM, dtype=np.float32),
            "frame_std": np.ones(FRAME_DIM, dtype=np.float32),
            "scalar_mean": np.zeros(SCALAR_DIM, dtype=np.float32),
            "scalar_std": np.ones(SCALAR_DIM, dtype=np.float32),
        },
        metadata={"ptm_repo_id": "test/repo"},
    )
    torch.save(payload, flat_path)
    affine = tmp_path / "affine.npz"
    np.savez(
        affine,
        schema=np.asarray(PTM_PROJECTION_SCHEMA),
        mean=np.zeros(PTM_INPUT_DIM, dtype=np.float32),
        components=np.eye(PTM_PROJECTED_DIM, PTM_INPUT_DIM, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="no in-model PTM projector"):
        build_deploy_checkpoint(flat_path, affine, tmp_path / "deploy.pt")
