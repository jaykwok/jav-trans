from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from boundary.backbones import SpeechIslandSequenceClassifier
from boundary.ja.model import score_speech_island_probabilities_batch
from boundary.ja.proposal import (
    BOUNDARY_PROPOSAL_SCORER_ARTIFACT,
    BOUNDARY_PROPOSAL_SCORER_SCHEMA,
    build_boundary_proposal_checkpoint,
    load_boundary_proposal_checkpoint,
)
from tools.boundary.ja.train_boundary_proposal_scorer import boundary_target_frames

PTM_DIM = 4
MFCC_DIM = 3
INPUT_DIM = PTM_DIM + MFCC_DIM


def _tiny_model() -> SpeechIslandSequenceClassifier:
    return SpeechIslandSequenceClassifier(
        input_dim=INPUT_DIM,
        hidden_size=8,
        num_layers=1,
        output_dim=1,
    )


def _tiny_model_config() -> dict:
    return {
        "ptm_dim": PTM_DIM,
        "mfcc_dim": MFCC_DIM,
        "input_dim": INPUT_DIM,
        "hidden_size": 8,
        "num_layers": 1,
        "state_size": 32,
        "num_heads": 4,
        "n_groups": 2,
        "chunk_size": 8,
        "conv_kernel": 4,
        "bidirectional": True,
        "model_arch": "v1-boundary-proposal",
        "output_dim": 1,
    }


def _tiny_normalization() -> dict:
    return {
        "feature_mean": [0.0] * INPUT_DIM,
        "feature_std": [1.0] * INPUT_DIM,
    }


def test_boundary_proposal_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(7)
    model = _tiny_model()
    checkpoint_path = tmp_path / "boundary_proposal_scorer_v1.test.pt"
    torch.save(
        build_boundary_proposal_checkpoint(
            model=model,
            model_config=_tiny_model_config(),
            normalization=_tiny_normalization(),
            metadata={"ptm_repo_id": "test/repo", "trained_steps": 3},
        ),
        checkpoint_path,
    )

    bundle = load_boundary_proposal_checkpoint(checkpoint_path, device="cpu")

    assert bundle.schema == BOUNDARY_PROPOSAL_SCORER_SCHEMA
    assert bundle.ptm_dim == PTM_DIM
    assert bundle.mfcc_dim == MFCC_DIM
    assert bundle.input_dim == INPUT_DIM
    artifact = bundle.metadata["artifact"]
    for key, expected in BOUNDARY_PROPOSAL_SCORER_ARTIFACT.items():
        assert artifact[key] == expected
    signature = bundle.signature()
    assert signature["schema"] == BOUNDARY_PROPOSAL_SCORER_SCHEMA
    assert signature["sha256"] == bundle.sha256

    # The bundle must run through the shared batch scorer unchanged: the
    # dataset builder and runtime backend both feed it (ptm, mfcc) pairs.
    rng = np.random.default_rng(11)
    pairs = [
        (
            rng.normal(size=(24, PTM_DIM)).astype(np.float32),
            rng.normal(size=(24, MFCC_DIM)).astype(np.float32),
        ),
        (
            rng.normal(size=(9, PTM_DIM)).astype(np.float32),
            rng.normal(size=(9, MFCC_DIM)).astype(np.float32),
        ),
    ]
    probabilities = score_speech_island_probabilities_batch(
        bundle, feature_pairs=pairs
    )
    assert [row.shape[0] for row in probabilities] == [24, 9]
    for row in probabilities:
        assert np.all((row >= 0.0) & (row <= 1.0))


def test_boundary_proposal_checkpoint_rejects_bad_contract(tmp_path) -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="output_dim=1"):
        build_boundary_proposal_checkpoint(
            model=model,
            model_config={**_tiny_model_config(), "output_dim": 3},
            normalization=_tiny_normalization(),
        )

    payload = build_boundary_proposal_checkpoint(
        model=model,
        model_config=_tiny_model_config(),
        normalization=_tiny_normalization(),
    )
    payload["schema"] = "speech_island_scorer_v8"
    bad_schema = tmp_path / "bad_schema.pt"
    torch.save(payload, bad_schema)
    with pytest.raises(ValueError, match="schema"):
        load_boundary_proposal_checkpoint(bad_schema, device="cpu")

    payload = build_boundary_proposal_checkpoint(
        model=model,
        model_config=_tiny_model_config(),
        normalization={"feature_mean": [0.0], "feature_std": [1.0]},
    )
    bad_norm = tmp_path / "bad_norm.pt"
    torch.save(payload, bad_norm)
    with pytest.raises(ValueError, match="normalization"):
        load_boundary_proposal_checkpoint(bad_norm, device="cpu")


def test_boundary_target_frames_marks_radius_and_clips_edges() -> None:
    targets = boundary_target_frames(
        boundary_times_s=[0.0, 0.10],
        frame_count=8,
        frame_hop_s=0.02,
        radius_frames=1,
    )

    assert targets.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
