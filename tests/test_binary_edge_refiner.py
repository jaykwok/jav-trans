from __future__ import annotations

import sys

import numpy as np
import pytest

from boundary.binary_edge_refiner import (
    BINARY_EDGE_IGNORE_INDEX,
    BinaryFrameEdgeNetwork,
    canonical_to_binary_labels,
    decode_binary_edge_logits,
)
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.outer_refiner_v3 import (
    OUTER_EDGE_REFINER_V3_STATUS,
    OuterEdgeRefinerV3,
    build_outer_edge_refiner_v3_checkpoint,
    load_outer_edge_refiner_v3,
)
from boundary.inner_refiner_v2 import (
    INNER_EDGE_REFINER_V2_RUNTIME_ADAPTER,
    InnerEdgeRefinerV2,
    build_inner_edge_refiner_v2_checkpoint,
    load_inner_edge_refiner_v2,
)
from tools.boundary.ja.train_inner_edge_refiner_v2_binary import (
    load_binary as load_inner_binary,
    parse_args as parse_inner_train_args,
)


def test_canonical_unsure_is_ignored_not_mapped_to_background() -> None:
    assert canonical_to_binary_labels(np.asarray([0, 1, 2])).tolist() == [
        0, 1, BINARY_EDGE_IGNORE_INDEX
    ]


def test_binary_edge_network_and_argmax_contract() -> None:
    torch = pytest.importorskip("torch")
    config = {
        "ptm_input_dim": 8, "ptm_projected_dim": 4, "mfcc_dim": 2,
        "position_dim": 1, "hidden_size": 8, "num_layers": 1,
        "num_heads": 2, "head_dim": 8, "n_groups": 1, "output_dim": 2,
    }
    model = BinaryFrameEdgeNetwork(**config)
    assert model(torch.zeros(1, 5, 11)).shape == (1, 5, 2)
    assert decode_binary_edge_logits(
        np.asarray([[2, 0], [0, 2], [0, 3], [2, 0]], dtype=np.float32),
        raw_start_s=1.0, raw_end_s=1.08, frame_hop_s=0.02,
    ) == pytest.approx((1.02, 1.06))


def test_outer_v3_is_a_non_executable_placeholder() -> None:
    assert OUTER_EDGE_REFINER_V3_STATUS == "pending_outer_v3_audit"
    assert OuterEdgeRefinerV3().signature()[
        "boundary_serialization_contract_id"
    ] == ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    with pytest.raises(RuntimeError, match="pending_outer_v3_audit"):
        load_outer_edge_refiner_v3("outer-v3.pt", device="cpu")
    with pytest.raises(RuntimeError, match="pending_outer_v3_audit"):
        build_outer_edge_refiner_v3_checkpoint()


def test_inner_v2_loader_uses_manifest_frame_slice(tmp_path) -> None:
    source = tmp_path / "source.npz"
    labels = tmp_path / "labels.npz"
    np.savez(
        source,
        ptm=np.arange(6 * 4, dtype=np.float32).reshape(6, 4),
        mfcc=np.arange(6 * 2, dtype=np.float32).reshape(6, 2),
    )
    np.savez(labels, labels=np.asarray([0, 1, 1], dtype=np.int64))

    features, target, weights = load_inner_binary(
        {
            "row_id": "row",
            "source_feature_path": str(source),
            "label_path": str(labels),
            "start_frame": 2,
            "end_frame": 5,
            "acoustic_start_frame": 2,
            "acoustic_end_frame": 5,
        }
    )

    assert features.shape == (3, 8)
    assert features[:, :4].tolist() == np.arange(8, 20, dtype=np.float32).reshape(3, 4).tolist()
    assert target.tolist() == [0, 1, 1]
    assert weights.tolist() == [1.0, 1.0, 1.0]
    assert features[:, -1].tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_inner_v2_checkpoint_is_binary_acoustic_and_rejects_v1_schema(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = {
        "ptm_input_dim": 8, "ptm_projected_dim": 4, "mfcc_dim": 2,
        "position_dim": 1, "hidden_size": 8, "num_layers": 1,
        "num_heads": 2, "head_dim": 8, "n_groups": 1, "output_dim": 2,
    }
    payload = build_inner_edge_refiner_v2_checkpoint(
        model=BinaryFrameEdgeNetwork(**config), model_config=config,
        feature_config={"raw_ptm_dim": 8},
        normalization={"feature_mean": [0.0] * 11, "feature_std": [1.0] * 11},
        metadata={},
    )
    checkpoint = tmp_path / "inner-v2.pt"
    torch.save(payload, checkpoint)
    loaded = load_inner_edge_refiner_v2(checkpoint, device="cpu")
    assert loaded.metadata["runtime_adapter"] == INNER_EDGE_REFINER_V2_RUNTIME_ADAPTER
    assert loaded.metadata["training_labels"] == ["background", "semantic_core"]
    assert loaded.model_config["output_dim"] == 2

    payload["schema"] = "inner_edge_refiner_v1"
    old = tmp_path / "inner-v1.pt"
    torch.save(payload, old)
    with pytest.raises(ValueError, match="v2 schema"):
        load_inner_edge_refiner_v2(old, device="cpu")


def test_inner_v2_trainer_has_no_legacy_warm_start_surface(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train-inner", "--dataset-manifest", "manifest.jsonl", "--output-dir", "out"],
    )
    args = parse_inner_train_args()
    assert args.dataset_manifest == "manifest.jsonl"
    assert not hasattr(args, "warm_start")


def test_inner_v2_runtime_uses_binary_argmax_for_core_or_drop() -> None:
    torch = pytest.importorskip("torch")

    class _FixedModel:
        def __init__(self, logits) -> None:
            self.logits = torch.tensor(logits, dtype=torch.float32)

        def __call__(self, _features):
            return self.logits.unsqueeze(0)

    common = {
        "path": "inner-v2.pt", "sha256": "sha", "model_config": {"ptm_input_dim": 2},
        "feature_config": {"raw_ptm_dim": 2},
        "normalization": {"feature_mean": [0.0, 0.0], "feature_std": [1.0, 1.0]},
        "metadata": {}, "device": "cpu",
    }
    semantic = InnerEdgeRefinerV2(
        model=_FixedModel([[3.0, 0.0], [0.0, 3.0], [0.0, 4.0], [3.0, 0.0]]),
        **common,
    ).predict_subislands(
        frame_feature_groups=[np.zeros((4, 2), dtype=np.float32)],
        raw_spans=[(1.0, 1.08)], frame_hop_s=0.02,
    )[0]
    assert semantic.start_action == "refined"
    assert (semantic.start_s, semantic.end_s) == pytest.approx((1.02, 1.06))

    background = InnerEdgeRefinerV2(
        model=_FixedModel([[3.0, 0.0], [4.0, 0.0]]), **common,
    ).predict_subislands(
        frame_feature_groups=[np.zeros((2, 2), dtype=np.float32)],
        raw_spans=[(2.0, 2.04)], frame_hop_s=0.02,
    )[0]
    assert background.start_action == "drop"
    assert background.abstain_reason == "binary_all_background"
