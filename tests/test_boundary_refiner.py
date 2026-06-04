from __future__ import annotations

from pathlib import Path

import pytest

from boundary.refiner import (
    DEFAULT_REFINER_CHECKPOINT_PATH,
    DEFAULT_REFINER_FEATURES,
    HeuristicBoundaryRefiner,
    RefinerInput,
    build_learned_refiner_checkpoint,
    load_frame_sequence_refiner_checkpoint,
    load_boundary_refiner,
)
from boundary.sequence_features import (
    FRAME_SEQUENCE_FEATURE_SCHEMA,
    FrameSequenceFeatureConfig,
    feature_extraction_hash,
    feature_extraction_signature,
)


def _input(**overrides) -> RefinerInput:
    values = {
        "gap_s": 0.5,
        "left_start": 0.0,
        "left_end": 1.0,
        "right_start": 1.5,
        "right_end": 2.5,
        "current_core_s": 1.0,
        "proposed_core_s": 2.5,
        "gap_merge_s": 1.5,
    }
    values.update(overrides)
    return RefinerInput(**values)


def test_heuristic_boundary_refiner_blocks_high_cut_score():
    refiner = HeuristicBoundaryRefiner(cut_score_threshold=0.94)

    decision = refiner.decide_gap(_input(cut_score_max=0.98))

    assert decision.merge is False
    assert decision.reason == "cut_score_high"


def test_heuristic_boundary_refiner_can_merge_short_semantic_gap():
    refiner = HeuristicBoundaryRefiner(merge_threshold=0.4, target_core_s=9.0)

    decision = refiner.decide_gap(_input(gap_s=0.3, proposed_core_s=3.0))

    assert decision.merge is True
    assert decision.score > 0.4


def test_load_boundary_refiner_disabled_returns_none():
    assert load_boundary_refiner(enabled=False) is None


def test_load_boundary_refiner_rejects_unknown_checkpoint(tmp_path):
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError):
        load_boundary_refiner(enabled=True, model_path=str(missing))


def test_load_boundary_refiner_uses_bootstrap_when_default_checkpoint_missing():
    refiner = load_boundary_refiner(
        enabled=True,
        model_path=str(DEFAULT_REFINER_CHECKPOINT_PATH),
        backbone="transformers.Mamba2Model",
    )

    assert isinstance(refiner, HeuristicBoundaryRefiner)


def test_load_boundary_refiner_accepts_transformers_mamba2_without_checkpoint():
    refiner = load_boundary_refiner(
        enabled=True,
        backbone="transformers.Mamba2Model",
    )

    assert isinstance(refiner, HeuristicBoundaryRefiner)


def test_load_boundary_refiner_rejects_non_mamba2_backbone():
    with pytest.raises(ValueError, match="only transformers\\.Mamba2Model"):
        load_boundary_refiner(enabled=True, backbone="tcn")


def test_load_boundary_refiner_rejects_mamba2_alias():
    with pytest.raises(ValueError, match="only transformers\\.Mamba2Model"):
        load_boundary_refiner(enabled=True, backbone="mamba2")


def test_load_boundary_refiner_rejects_torch_mamba2_alias():
    with pytest.raises(ValueError, match="only transformers\\.Mamba2Model"):
        load_boundary_refiner(enabled=True, backbone="torch_mamba2")


def test_learned_mamba2_refiner_checkpoint_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        from boundary.backbones import BoundarySequenceClassifier
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"Mamba2Model is unavailable in this transformers build: {exc}")

    model = BoundarySequenceClassifier(
        input_dim=len(DEFAULT_REFINER_FEATURES),
        hidden_size=8,
        num_layers=1,
        state_size=4,
        num_heads=4,
        n_groups=2,
        chunk_size=4,
    )
    checkpoint = build_learned_refiner_checkpoint(
        model=model,
        metadata={"unit": "test"},
    )
    checkpoint_path = tmp_path / "boundary-refiner.pt"
    torch.save(checkpoint, checkpoint_path)

    refiner = load_boundary_refiner(
        enabled=True,
        model_path=str(checkpoint_path),
        backbone="transformers.Mamba2Model",
        merge_threshold=0.0,
    )
    decision = refiner.decide_gap(_input())

    assert decision.merge is True
    assert 0.0 <= decision.score <= 1.0
    assert refiner.signature()["type"] == "learned_boundary_refiner"
    assert refiner.signature()["backbone"] == "transformers.Mamba2Model"
    assert refiner.signature()["metadata"] == {"unit": "test"}


def test_tiny_mamba2_backbone_shape_or_skip():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        from boundary.backbones import TinyMamba2BoundaryBackbone
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"Mamba2Model is unavailable in this transformers build: {exc}")

    model = TinyMamba2BoundaryBackbone(
        input_dim=6,
        hidden_size=16,
        num_layers=1,
        state_size=8,
        num_heads=4,
        head_dim=8,
        n_groups=2,
        chunk_size=8,
        bidirectional=True,
    )
    features = torch.randn(2, 5, 6)

    output = model(features)

    assert output.shape == (2, 5, 32)


def test_tiny_mamba2_backbone_validates_head_shape():
    pytest.importorskip("torch")
    try:
        from boundary.backbones import TinyMamba2BoundaryBackbone
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"Mamba2Model is unavailable in this transformers build: {exc}")

    with pytest.raises(ValueError, match="hidden_size \\* expand"):
        TinyMamba2BoundaryBackbone(
            input_dim=6,
            hidden_size=16,
            num_heads=2,
            head_dim=8,
        )


def test_frame_sequence_refiner_checkpoint_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        from boundary.backbones import BoundarySequenceClassifier
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"Mamba2Model is unavailable in this transformers build: {exc}")

    feature_names = ("left_ptm_mean_000", "gap_ptm_mean_000", "right_ptm_mean_000")
    feature_config = FrameSequenceFeatureConfig(max_ptm_dims=1, include_mfcc=False)
    model = BoundarySequenceClassifier(
        input_dim=len(feature_names),
        hidden_size=8,
        num_layers=1,
        state_size=4,
        num_heads=4,
        n_groups=2,
        chunk_size=4,
    )
    checkpoint = build_learned_refiner_checkpoint(
        model=model,
        feature_names=feature_names,
        metadata={
            "runtime_adapter": "frame_sequence_v1",
            "feature_schema": FRAME_SEQUENCE_FEATURE_SCHEMA,
            "feature_schema_hash": feature_extraction_hash(
                config=feature_config,
                feature_names=feature_names,
            ),
            "feature_signature": feature_extraction_signature(
                config=feature_config,
                feature_names=feature_names,
            ),
        },
    )
    checkpoint_path = tmp_path / "frame-sequence-refiner.pt"
    torch.save(checkpoint, checkpoint_path)

    refiner = load_frame_sequence_refiner_checkpoint(
        checkpoint_path,
        threshold=0.0,
        backbone_override="transformers.Mamba2Model",
    )
    decisions = refiner.decide_sequence([[0.0, 0.1, 0.2], [0.2, 0.1, 0.0]])

    assert len(decisions) == 2
    assert all(decision.merge for decision in decisions)
    assert refiner.signature()["runtime_adapter"] == "frame_sequence_v1"
    assert refiner.signature()["metadata"]["feature_schema"] == FRAME_SEQUENCE_FEATURE_SCHEMA
    assert refiner.signature()["requested_device"] == "auto"
    assert refiner.signature()["actual_device"] in {"cpu", "cuda:0"}


def test_frame_sequence_refiner_rejects_missing_feature_schema(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        from boundary.backbones import BoundarySequenceClassifier
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"Mamba2Model is unavailable in this transformers build: {exc}")

    feature_names = ("left_ptm_mean_000", "gap_ptm_mean_000", "right_ptm_mean_000")
    model = BoundarySequenceClassifier(
        input_dim=len(feature_names),
        hidden_size=8,
        num_layers=1,
        state_size=4,
        num_heads=4,
        n_groups=2,
        chunk_size=4,
    )
    checkpoint = build_learned_refiner_checkpoint(
        model=model,
        feature_names=feature_names,
        metadata={"runtime_adapter": "frame_sequence_v1"},
    )
    checkpoint_path = tmp_path / "frame-sequence-refiner.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="metadata\\.feature_schema"):
        load_frame_sequence_refiner_checkpoint(
            checkpoint_path,
            threshold=0.0,
            backbone_override="transformers.Mamba2Model",
        )
