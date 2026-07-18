from __future__ import annotations

import numpy as np
import pytest
import torch

from boundary.split_model import (
    SEMANTIC_SPLIT_V4_MODEL_ARCH,
    SEMANTIC_SPLIT_V4_RUNTIME_ADAPTER,
    SEMANTIC_SPLIT_V4_SCHEMA,
    IslandCandidateSequenceNetwork,
    SplitDecision,
    aggregate_cut_event_runs,
    build_acoustic_split_v4_checkpoint,
    load_acoustic_split_v4_planner,
)
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT


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


def test_cut_runs_emit_one_event_at_highest_probability() -> None:
    decisions = [
        SplitDecision("continue", 0.1, 0.8, 0.1),
        SplitDecision("cut", 0.70, 0.2, 0.1),
        SplitDecision("cut", 0.92, 0.05, 0.03),
        SplitDecision("continue", 0.2, 0.8, 0.0),
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


def test_v4_checkpoint_is_binary_argmax_only(tmp_path) -> None:
    config = {**_model_config(), "num_classes": 2}
    model = IslandCandidateSequenceNetwork(**config)
    v4 = tmp_path / "v4.pt"
    torch.save(
        build_acoustic_split_v4_checkpoint(
            model=model,
            model_config=config,
            feature_config=_feature_config(),
            normalization=_normalization(),
            metadata={"ptm_repo_id": "repo/1.7b", "excluded_training_label_count": 7},
        ),
        v4,
    )
    planner = load_acoustic_split_v4_planner(v4, device="cpu")
    assert not any(
        key.startswith(("gate_head.", "omni_head.", "offset_head."))
        for key in model.state_dict()
    )
    signature = planner.signature()
    assert signature["schema"] == SEMANTIC_SPLIT_V4_SCHEMA
    assert signature["model_arch"] == SEMANTIC_SPLIT_V4_MODEL_ARCH
    assert signature["runtime_adapter"] == SEMANTIC_SPLIT_V4_RUNTIME_ADAPTER
    assert signature["boundary_serialization_contract_id"] == (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    )
    assert signature["decision_config"] == {"decision_mode": "binary_argmax_cut"}
    decisions = planner.decide_islands(
        island_frame_features=[np.zeros((2, BINS, FRAME_DIM), dtype=np.float32)],
        island_scalar_features=[np.zeros((2, SCALAR_DIM), dtype=np.float32)],
    )[0]
    assert all(row.label in ("cut", "continue") for row in decisions)
    assert all(row.p_unsure == 0.0 for row in decisions)
    assert all(
        row.p_cut + row.p_continue == pytest.approx(1.0, abs=1e-5)
        for row in decisions
    )
    groups = [
        np.zeros((2, BINS, FRAME_DIM), dtype=np.float32),
        np.zeros((1, BINS, FRAME_DIM), dtype=np.float32),
        np.zeros((3, BINS, FRAME_DIM), dtype=np.float32),
    ]
    scalars = [np.zeros((len(group), SCALAR_DIM), dtype=np.float32) for group in groups]
    unbatched = planner.decide_islands(
        island_frame_features=groups,
        island_scalar_features=scalars,
        max_padded_candidates=100,
    )
    bounded = planner.decide_islands(
        island_frame_features=groups,
        island_scalar_features=scalars,
        max_padded_candidates=3,
    )
    assert [[row.label for row in group] for group in bounded] == [
        [row.label for row in group] for group in unbatched
    ]
    assert [
        probability
        for group in bounded
        for row in group
        for probability in (row.p_cut, row.p_continue)
    ] == pytest.approx(
        [
            probability
            for group in unbatched
            for row in group
            for probability in (row.p_cut, row.p_continue)
        ],
        abs=1e-6,
    )


def test_v4_loader_rejects_legacy_three_class_and_v2_schemas(tmp_path) -> None:
    model = IslandCandidateSequenceNetwork(**_model_config())
    v3 = tmp_path / "v3.pt"
    torch.save(
        {
            "schema": "semantic_split_model_v3",
            "model_arch": "acoustic_candidate_sequence_mamba_v1",
            "model_config": {**_model_config(), "num_classes": 3},
            "metadata": {"ptm_repo_id": "repo/1.7b"},
            "model_state_dict": model.state_dict(),
        },
        v3,
    )
    with pytest.raises(ValueError, match="semantic_split_model_v4"):
        load_acoustic_split_v4_planner(v3, device="cpu")

    v2 = tmp_path / "v2.pt"
    torch.save({"schema": "semantic_split_verifier_v2"}, v2)
    with pytest.raises(ValueError, match="semantic_split_model_v4"):
        load_acoustic_split_v4_planner(v2, device="cpu")

    retired_arch = tmp_path / "retired-arch.pt"
    torch.save(
        {
            "schema": SEMANTIC_SPLIT_V4_SCHEMA,
            "model_arch": "acoustic_candidate_sequence_mamba_binary_v1",
            "model_config": {**_model_config(), "num_classes": 2},
            "metadata": {
                "ptm_repo_id": "repo/1.7b",
                "training_labels": ["cut", "continue"],
                "excluded_training_labels": ["unsure"],
            },
            "model_state_dict": model.state_dict(),
        },
        retired_arch,
    )
    with pytest.raises(ValueError, match="model_arch"):
        load_acoustic_split_v4_planner(retired_arch, device="cpu")
