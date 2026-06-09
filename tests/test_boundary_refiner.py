from __future__ import annotations

import pytest

from boundary.refiner import (
    LEARNED_REFINER_SCHEMA,
    DEFAULT_REFINER_CHECKPOINT_PATH,
    build_learned_refiner_checkpoint,
    load_frame_sequence_refiner_checkpoint,
)
from boundary.sequence_features import (
    FRAME_SEQUENCE_FEATURE_SCHEMA,
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
    feature_extraction_hash,
    feature_extraction_signature,
)


def _tiny_sequence_classifier(feature_names: tuple[str, ...]):
    pytest.importorskip("transformers")
    try:
        from boundary.backbones import BoundarySequenceClassifier
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"Mamba2Model is unavailable in this transformers build: {exc}")

    return BoundarySequenceClassifier(
        input_dim=len(feature_names),
        hidden_size=8,
        num_layers=1,
        output_dim=3,
        state_size=4,
        num_heads=4,
        n_groups=2,
        chunk_size=4,
    )


def _frame_sequence_checkpoint(model, feature_names: tuple[str, ...]) -> dict:
    feature_config = FrameSequenceFeatureConfig(max_ptm_dims=1, include_mfcc=False)
    return build_learned_refiner_checkpoint(
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


def test_load_frame_sequence_refiner_loads_canonical_checkpoint_when_present():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")

    assert DEFAULT_REFINER_CHECKPOINT_PATH.exists()
    import torch
    payload = torch.load(DEFAULT_REFINER_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if payload.get("schema") != LEARNED_REFINER_SCHEMA:
        pytest.skip("default checkpoint has not been replaced with v4 yet")

    refiner = load_frame_sequence_refiner_checkpoint(
        DEFAULT_REFINER_CHECKPOINT_PATH,
        threshold=0.5,
        backbone_override="transformers.Mamba2Model",
    )

    assert refiner.signature()["type"] == "learned_boundary_refiner"
    assert refiner.signature()["metadata"]["runtime_adapter"] == "frame_sequence_v1"


def test_load_frame_sequence_refiner_rejects_non_mamba2_backbone(tmp_path):
    torch = pytest.importorskip("torch")
    feature_names = ("left_ptm_mean_000", "gap_ptm_mean_000", "right_ptm_mean_000")
    model = _tiny_sequence_classifier(feature_names)
    checkpoint_path = tmp_path / "frame-sequence-refiner.pt"
    torch.save(_frame_sequence_checkpoint(model, feature_names), checkpoint_path)

    with pytest.raises(ValueError, match="only transformers\\.Mamba2Model"):
        load_frame_sequence_refiner_checkpoint(checkpoint_path, threshold=0.0, backbone_override="tcn")


def test_load_frame_sequence_refiner_rejects_mamba2_alias(tmp_path):
    torch = pytest.importorskip("torch")
    feature_names = ("left_ptm_mean_000", "gap_ptm_mean_000", "right_ptm_mean_000")
    model = _tiny_sequence_classifier(feature_names)
    checkpoint_path = tmp_path / "frame-sequence-refiner.pt"
    torch.save(_frame_sequence_checkpoint(model, feature_names), checkpoint_path)

    with pytest.raises(ValueError, match="only transformers\\.Mamba2Model"):
        load_frame_sequence_refiner_checkpoint(checkpoint_path, threshold=0.0, backbone_override="mamba2")


def test_load_frame_sequence_refiner_rejects_torch_mamba2_alias(tmp_path):
    torch = pytest.importorskip("torch")
    feature_names = ("left_ptm_mean_000", "gap_ptm_mean_000", "right_ptm_mean_000")
    model = _tiny_sequence_classifier(feature_names)
    checkpoint_path = tmp_path / "frame-sequence-refiner.pt"
    torch.save(_frame_sequence_checkpoint(model, feature_names), checkpoint_path)

    with pytest.raises(ValueError, match="only transformers\\.Mamba2Model"):
        load_frame_sequence_refiner_checkpoint(
            checkpoint_path,
            threshold=0.0,
            backbone_override="torch_mamba2",
        )


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
        output_dim=3,
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
        str(checkpoint_path),
        threshold=0.0,
        backbone_override="transformers.Mamba2Model",
    )
    decisions = refiner.decide_sequence([[0.0, 0.1, 0.2], [0.2, 0.1, 0.0]])

    assert len(decisions) == 2
    assert all(decision.merge for decision in decisions)
    assert all(decision.start_refine_delta_s is not None for decision in decisions)
    assert all(decision.end_refine_delta_s is not None for decision in decisions)
    assert all(decision.start_refine_delta_s is not None for decision in decisions)
    assert refiner.signature()["runtime_adapter"] == "frame_sequence_v1"
    assert refiner.signature()["metadata"]["feature_schema"] == FRAME_SEQUENCE_FEATURE_SCHEMA
    assert refiner.signature()["requested_device"] == "auto"
    assert refiner.signature()["actual_device"] in {"cpu", "cuda:0"}


def test_frame_sequence_feature_provider_caches_frame_arrays(monkeypatch):
    import boundary.sequence_features as sequence_features

    calls: list[str] = []
    original = sequence_features._frame_array

    def counting_frame_array(values, *, name):
        calls.append(name)
        return original(values, name=name)

    monkeypatch.setattr(sequence_features, "_frame_array", counting_frame_array)
    provider = FrameSequenceFeatureProvider(
        duration_s=2.0,
        frame_hop_s=0.02,
        ptm=[[0.1, 0.2]] * 100,
        mfcc=[[0.3]] * 100,
        config=FrameSequenceFeatureConfig(max_ptm_dims=1, include_mfcc=True),
    )

    first = provider.features_for_gap(
        left_start_s=0.0,
        left_end_s=0.5,
        right_start_s=0.8,
        right_end_s=1.2,
    )
    second = provider.features_for_gap(
        left_start_s=1.0,
        left_end_s=1.2,
        right_start_s=1.4,
        right_end_s=1.8,
    )

    assert first
    assert second
    assert calls == ["ptm", "mfcc"]


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
        output_dim=3,
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
