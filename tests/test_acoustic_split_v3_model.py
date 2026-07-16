from __future__ import annotations

import numpy as np
import pytest
import torch

from boundary.split_model import (
    SEMANTIC_SPLIT_V3_MODEL_ARCH,
    SEMANTIC_SPLIT_V3_RUNTIME_ADAPTER,
    SEMANTIC_SPLIT_V3_SCHEMA,
    IslandCandidateSequenceNetwork,
    SplitDecision,
    aggregate_cut_event_runs,
    build_acoustic_split_v3_checkpoint,
    build_semantic_split_island_checkpoint,
    load_acoustic_split_v3_planner,
    load_semantic_split_verifier,
)


FRAME_DIM = 6
SCALAR_DIM = 3
BINS = 20


def _model_config() -> dict:
    return {
        "frame_dim": FRAME_DIM,
        "scalar_dim": SCALAR_DIM,
        "hidden_size": 16,
        "candidate_layers": 1,
        "island_layers": 1,
        "state_size": 4,
        "num_heads": 2,
        "head_dim": 16,
        "n_groups": 1,
        "conv_kernel": 2,
        "chunk_size": 4,
        "bidirectional": True,
        "dropout": 0.0,
    }


def _normalization() -> dict:
    return {
        "frame_mean": np.zeros(FRAME_DIM, dtype=np.float32).tolist(),
        "frame_std": np.ones(FRAME_DIM, dtype=np.float32).tolist(),
        "scalar_mean": np.zeros(SCALAR_DIM, dtype=np.float32).tolist(),
        "scalar_std": np.ones(SCALAR_DIM, dtype=np.float32).tolist(),
    }


def _feature_config() -> dict:
    return {
        "ptm_dim": 4,
        "mfcc_dim": 2,
        "left_bins": 8,
        "gap_bins": 4,
        "right_bins": 8,
    }


def test_v3_checkpoint_direct_argmax_contract(tmp_path) -> None:
    torch.manual_seed(17)
    model = IslandCandidateSequenceNetwork(**_model_config())
    path = tmp_path / "semantic_split_model_v3.test.pt"
    torch.save(
        build_acoustic_split_v3_checkpoint(
            model=model,
            model_config=_model_config(),
            feature_config=_feature_config(),
            normalization=_normalization(),
            metadata={"ptm_repo_id": "repo/1.7b"},
        ),
        path,
    )
    planner = load_acoustic_split_v3_planner(path, device="cpu")
    signature = planner.signature()
    assert signature["schema"] == SEMANTIC_SPLIT_V3_SCHEMA
    assert signature["model_arch"] == SEMANTIC_SPLIT_V3_MODEL_ARCH
    assert signature["runtime_adapter"] == SEMANTIC_SPLIT_V3_RUNTIME_ADAPTER
    assert signature["decision_config"] == {"decision_mode": "argmax_cut"}

    rng = np.random.default_rng(3)
    decisions = planner.decide_islands(
        island_frame_features=[
            rng.normal(size=(4, BINS, FRAME_DIM)).astype(np.float32)
        ],
        island_scalar_features=[
            rng.normal(size=(4, SCALAR_DIM)).astype(np.float32)
        ],
    )[0]
    assert len(decisions) == 4
    for decision in decisions:
        assert decision.label in ("cut", "continue", "unsure")
        assert decision.p_cut + decision.p_continue + decision.p_unsure == pytest.approx(
            1.0, abs=1e-5
        )
        probabilities = {
            "cut": decision.p_cut,
            "continue": decision.p_continue,
            "unsure": decision.p_unsure,
        }
        assert decision.label == max(probabilities, key=probabilities.get)


def test_cut_runs_emit_one_event_at_highest_probability() -> None:
    decisions = [
        SplitDecision("continue", 0.1, 0.8, 0.1),
        SplitDecision("cut", 0.70, 0.2, 0.1),
        SplitDecision("cut", 0.92, 0.05, 0.03),
        SplitDecision("unsure", 0.2, 0.3, 0.5),
        SplitDecision("cut", 0.81, 0.1, 0.09),
    ]
    events = aggregate_cut_event_runs(
        candidate_times_s=[1.0, 2.0, 2.2, 2.4, 4.0],
        decisions=decisions,
        event_id_prefix="island-a",
    )
    assert len(events) == 2
    assert events[0].candidate_start_index == 1
    assert events[0].candidate_end_index == 2
    assert events[0].representative_index == 2
    assert events[0].representative_time_s == pytest.approx(2.2)
    assert events[1].representative_index == 4


def test_v2_and_v3_loaders_do_not_alias_schemas(tmp_path) -> None:
    model = IslandCandidateSequenceNetwork(**_model_config())
    v3 = tmp_path / "v3.pt"
    torch.save(
        build_acoustic_split_v3_checkpoint(
            model=model,
            model_config=_model_config(),
            feature_config=_feature_config(),
            normalization=_normalization(),
            metadata={"ptm_repo_id": "repo/1.7b"},
        ),
        v3,
    )
    with pytest.raises(ValueError, match="semantic_split_verifier_v2"):
        load_semantic_split_verifier(v3, device="cpu")

    v2 = tmp_path / "v2.pt"
    torch.save(
        build_semantic_split_island_checkpoint(
            model=model,
            model_config=_model_config(),
            feature_config=_feature_config(),
            normalization=_normalization(),
            metadata={"ptm_repo_id": "repo/1.7b"},
        ),
        v2,
    )
    with pytest.raises(ValueError, match="semantic_split_model_v3"):
        load_acoustic_split_v3_planner(v2, device="cpu")
