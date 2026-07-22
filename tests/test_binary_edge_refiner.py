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
    OUTER_EDGE_REFINER_V3_DATASET_CONTRACT,
    OUTER_EDGE_REFINER_V3_RUNTIME_ADAPTER,
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
from tools.boundary.ja.train_outer_edge_refiner_v3_binary import (
    CANDIDATE_SCORER_V11_SCHEMA,
    _memory_snapshot,
    compute_normalization,
    evaluate,
    frame_budget_batches,
    parse_args as parse_outer_train_args,
    release_gate_fields,
    summarize_partition_label_presence,
    validate_dataset_rows,
)
import tools.boundary.ja.train_outer_edge_refiner_v3_binary as outer_trainer


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


def test_outer_v3_binary_checkpoint_loads_but_registry_remains_pending(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = {
        "ptm_input_dim": 8, "ptm_projected_dim": 4, "mfcc_dim": 2,
        "position_dim": 1, "hidden_size": 8, "num_layers": 1,
        "num_heads": 2, "head_dim": 8, "n_groups": 1,
        "chunk_size": 4, "valid_prefix_bidirectional": True,
        "output_dim": 2,
    }
    payload = build_outer_edge_refiner_v3_checkpoint(
        model=BinaryFrameEdgeNetwork(**config),
        model_config=config,
        feature_config={"raw_ptm_dim": 8},
        normalization={"feature_mean": [0.0] * 11, "feature_std": [1.0] * 11},
        metadata={"ptm_repo_id": "repo/1.7b"},
    )
    checkpoint = tmp_path / "outer-v3.pt"
    torch.save(payload, checkpoint)
    outer = load_outer_edge_refiner_v3(checkpoint, device="cpu")

    assert OUTER_EDGE_REFINER_V3_STATUS == "pending_outer_v3_audit"
    assert outer.signature()[
        "boundary_serialization_contract_id"
    ] == ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    assert outer.metadata["runtime_adapter"] == OUTER_EDGE_REFINER_V3_RUNTIME_ADAPTER
    assert outer.metadata["dataset_contract"] == OUTER_EDGE_REFINER_V3_DATASET_CONTRACT
    assert outer.metadata["training_initialization"] == "random"


def test_outer_v3_batching_matches_singletons_and_preserves_order() -> None:
    torch = pytest.importorskip("torch")
    config = {
        "ptm_input_dim": 8, "ptm_projected_dim": 4, "mfcc_dim": 2,
        "position_dim": 1, "hidden_size": 8, "num_layers": 1,
        "num_heads": 2, "head_dim": 8, "n_groups": 1,
        "chunk_size": 4, "valid_prefix_bidirectional": True,
        "output_dim": 2,
    }
    model = BinaryFrameEdgeNetwork(**config).eval()
    outer = OuterEdgeRefinerV3(
        path="outer-v3.pt", sha256="sha", model=model,
        model_config=config, feature_config={"raw_ptm_dim": 8},
        normalization={"feature_mean": [0.0] * 11, "feature_std": [1.0] * 11},
        metadata={
            "boundary_serialization_contract_id": (
                ACOUSTIC_BINARY_V12_CONTRACT.contract_id
            )
        },
        device="cpu",
    )
    rng = np.random.default_rng(51)
    groups = [rng.normal(size=(frames, 11)).astype(np.float32) for frames in (5, 9, 7)]
    spans = [(0.0, frames * 0.02) for frames in (5, 9, 7)]
    batched = outer.predict_islands(
        frame_feature_groups=groups, raw_spans=spans,
        frame_hop_s=0.02, max_padded_frames=100,
    )
    singleton = outer.predict_islands(
        frame_feature_groups=groups, raw_spans=spans,
        frame_hop_s=0.02, max_padded_frames=9,
    )
    assert [(row.raw_start_s, row.raw_end_s) for row in batched] == spans
    for left, right in zip(batched, singleton, strict=True):
        np.testing.assert_allclose(
            left.class_probabilities, right.class_probabilities,
            atol=1e-5, rtol=1e-5,
        )
        np.testing.assert_array_equal(
            np.argmax(left.class_probabilities, axis=1),
            np.argmax(right.class_probabilities, axis=1),
        )

    with pytest.raises(ValueError, match="feature width mismatch"):
        outer.predict_islands(
            frame_feature_groups=[np.zeros((4, 10), dtype=np.float32)],
            raw_spans=[(0.0, 0.08)], frame_hop_s=0.02,
        )


def _outer_row(source: str, core: str, partition: str, frames: int) -> dict:
    return {
        "source_id": source,
        "core_id": core,
        "partition": partition,
        "frame_count": frames,
        "input_distribution": "post_candidate_island_scorer_v11_islands",
        "scorer_schema": CANDIDATE_SCORER_V11_SCHEMA,
    }


def test_outer_v3_dataset_contract_freezes_source_core_partitions() -> None:
    rows = [
        _outer_row("s1", "c1", "train", 9),
        _outer_row("s2", "c2", "val", 5),
        _outer_row("s3", "c3", "test", 7),
    ]
    assert validate_dataset_rows(rows)["max_core_use_count"] == 1
    assert [
        [row["frame_count"] for row in batch]
        for batch in frame_budget_batches(rows, max_padded_frames=10)
    ] == [[5], [7], [9]]

    leaked = [*rows, _outer_row("s1", "c4", "val", 4)]
    with pytest.raises(ValueError, match="source identity crosses"):
        validate_dataset_rows(leaked)
    duplicate_core = [*rows, _outer_row("s4", "c1", "train", 4)]
    with pytest.raises(ValueError, match="max core use"):
        validate_dataset_rows(duplicate_core)


def test_outer_v3_trainer_has_no_warm_start_or_loss_tuning_surface() -> None:
    args = parse_outer_train_args([
        "--dataset-manifest", "manifest.jsonl",
        "--output-dir", "out",
    ])
    assert args.dataset_manifest == "manifest.jsonl"
    assert not hasattr(args, "warm_start")
    assert not hasattr(args, "focal_gamma")
    assert not hasattr(args, "boundary_radius_frames")


def test_outer_v3_every_partition_requires_semantic_and_background(
    monkeypatch,
) -> None:
    rows = [
        {"core_id": "train-sem", "partition": "train"},
        {"core_id": "train-bg", "partition": "train"},
        {"core_id": "val-sem", "partition": "val"},
        {"core_id": "val-bg", "partition": "val"},
        {"core_id": "test-sem", "partition": "test"},
        {"core_id": "test-bg", "partition": "test"},
    ]

    def fake_load(row):
        semantic = row["core_id"].endswith("sem")
        canonical = np.asarray([0, 1 if semantic else 0, 2], dtype=np.int64)
        return np.zeros((3, 1), dtype=np.float32), canonical, np.ones(3)

    monkeypatch.setattr(outer_trainer, "load_edge_row", fake_load)
    presence, counts = summarize_partition_label_presence(rows)
    assert all(values == {"semantic_rows": 1, "all_background_rows": 1}
               for values in presence.values())
    assert counts["unsure"] == 6

    with pytest.raises(ValueError, match="semantic and all-background"):
        summarize_partition_label_presence(rows[:-1])


def test_outer_v3_unsure_is_excluded_from_normalization_and_gate(
    monkeypatch,
) -> None:
    rows = [{"core_id": "row", "frame_count": 3}]
    features = np.asarray([[1.0], [1000.0], [3.0]], dtype=np.float32)
    labels = np.asarray([0, -100, 1], dtype=np.int64)
    weights = np.ones(3, dtype=np.float32)
    monkeypatch.setattr(
        outer_trainer, "load_binary", lambda _row: (features, labels, weights)
    )
    normalization = compute_normalization(rows)
    assert normalization["feature_mean"] == pytest.approx([2.0])
    assert normalization["feature_std"] == pytest.approx([1.0])

    torch = pytest.importorskip("torch")

    class UnsureOnlySemantic(torch.nn.Module):
        def forward(self, values, *, attention_mask=None):
            return torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32
            )

    metrics = evaluate(
        UnsureOnlySemantic(), rows, normalization, torch.device("cpu"),
        tolerance_frames=15, max_padded_frames=8,
    )
    assert metrics["true_speech_deletion_count"] == 1
    assert metrics["semantic_recall"] == 0.0


def test_outer_v3_numeric_gate_never_implies_promotion_ready() -> None:
    assert release_gate_fields(True) == {
        "numeric_gate_pass": True,
        "gate_pass": False,
        "promotion_ready": False,
        "manual_zero_clipping_gate": "required_before_promotion",
    }


def test_outer_v3_cpu_memory_snapshot_marks_shared_vram_not_applicable(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        outer_trainer,
        "runtime_memory_snapshot",
        lambda **_kwargs: {
            "physical_ram_used_mb": 10.0,
            "physical_ram_budget_mb": 95.0,
            "shared_vram_mb": 12.0,
            "shared_vram_raw_mb": 20.0,
            "shared_vram_baseline_mb": 8.0,
        },
    )
    snapshot = _memory_snapshot(torch.device("cpu"))
    assert snapshot["shared_vram_mb"] == 0.0
    assert snapshot["shared_vram_monitor"] == "not_applicable_cpu_stage"
    assert "shared_vram_raw_mb" not in snapshot


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
