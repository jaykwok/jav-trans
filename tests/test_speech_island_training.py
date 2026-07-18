from __future__ import annotations

import numpy as np
import pytest

from boundary.ja.speech_train import (
    SPEECH_ISLAND_TRAINING_OUTPUT_DIM,
    SPEECH_ISLAND_TRAINING_SCHEMA,
    _crop,
    _normalize,
)
from boundary.ja.dataset import LabelRecord
from boundary.ja.semantic_speech_train import _class_indexes, _membership_indexes
from boundary.ja.model import (
    SPEECH_ISLAND_SCORER_V10_DATASET_CONTRACT,
    SPEECH_ISLAND_SCORER_V10_MODEL_ARCH,
    SPEECH_ISLAND_SCORER_V10_SCHEMA,
    BinarySpeechIslandScorerNetwork,
    SemanticSpeechScorerNetwork,
    build_speech_island_scorer_checkpoint,
    load_speech_island_scorer_checkpoint,
    score_binary_speech_class_probabilities_batch,
)
import tools.boundary.ja.train_speech_island_scorer_v10_binary as scorer_v10_trainer
from tools.boundary.ja.train_speech_island_scorer_v10_binary import (
    compute_mfcc_normalization,
    parse_args as parse_scorer_v10_args,
    release_gate_fields as scorer_v10_release_gate_fields,
    summarize_partition_labels,
    validate_dataset_rows as validate_scorer_v10_rows,
)


def test_speech_island_trainer_is_bound_to_v8_single_logit_contract() -> None:
    assert SPEECH_ISLAND_TRAINING_SCHEMA.endswith("speech_island_scorer_v8")
    assert SPEECH_ISLAND_TRAINING_OUTPUT_DIM == 1


def test_speech_training_crop_keeps_features_labels_and_weights_aligned() -> None:
    features = np.arange(60, dtype=np.float32).reshape(10, 6)
    labels = np.arange(10, dtype=np.float32)
    weights = np.arange(10, dtype=np.float32) + 1.0
    cropped_features, cropped_labels, cropped_weights = _crop(
        features,
        labels,
        weights,
        max_frames=4,
        rng=np.random.default_rng(7),
        random=False,
    )

    assert cropped_features.shape == (4, 6)
    assert cropped_labels.tolist() == [3.0, 4.0, 5.0, 6.0]
    assert cropped_weights.tolist() == [4.0, 5.0, 6.0, 7.0]


def test_speech_training_normalization_uses_checkpoint_statistics() -> None:
    features = np.asarray([[2.0, 5.0]], dtype=np.float32)
    normalized = _normalize(
        features,
        {"feature_mean": [1.0, 1.0], "feature_std": [1.0, 2.0]},
    )

    assert normalized[0].tolist() == pytest.approx([1.0, 2.0])


def _record(*, semantic_frames=None, membership_frames=None) -> LabelRecord:
    metadata = {}
    if semantic_frames is not None:
        metadata["semantic_class_frames"] = semantic_frames
    if membership_frames is not None:
        metadata["semantic_membership_frames"] = membership_frames
    return LabelRecord(
        audio_id="sample",
        source="test",
        duration_s=0.06,
        text="",
        teacher_segments={},
        frame_hop_s=0.02,
        speech_frames=[0, 1, 1],
        label_quality="supervised",
        boundary_metadata=metadata,
    )


def test_semantic_speech_v9_requires_explicit_three_class_frames() -> None:
    with pytest.raises(ValueError, match="must not train from legacy binary"):
        _class_indexes(_record(), total=3)

    indexes = _class_indexes(
        _record(
            semantic_frames=["discardable", "semantic_target", "unsure"]
        ),
        total=3,
    )
    assert indexes.tolist() == [0, 1, 2]


def test_semantic_speech_v9_requires_separate_source_membership_frames() -> None:
    with pytest.raises(ValueError, match="must not be derived from content-class runs"):
        _membership_indexes(_record(), total=3)

    indexes = _membership_indexes(
        _record(membership_frames=["outside", "inside", "unsure"]),
        total=3,
    )
    assert indexes.tolist() == [0, 1, 2]


def test_semantic_speech_v9_uses_trainable_full_ptm_projection() -> None:
    torch = pytest.importorskip("torch")
    model = SemanticSpeechScorerNetwork(
        raw_ptm_dim=8,
        projected_ptm_dim=2,
        mfcc_dim=2,
        mfcc_mean=[0.0, 0.0],
        mfcc_std=[1.0, 1.0],
        hidden_size=8,
        num_layers=1,
        state_size=4,
        num_heads=2,
        head_dim=8,
        n_groups=1,
        conv_kernel=2,
        chunk_size=2,
    )
    assert tuple(model.ptm_projector.weight.shape) == (2, 8)
    assert model.ptm_projector.weight.requires_grad
    with torch.no_grad():
        model.ptm_projector.weight.zero_()
        model.ptm_projector.bias.zero_()
        model.ptm_projector.weight[0, 7] = 1.0
    ptm = torch.zeros((1, 1, 8))
    ptm[0, 0, 7] = 3.0
    projected = model.project_ptm(ptm)
    assert projected[0, 0, 0].item() == pytest.approx(3.0)
    logits = model(ptm, torch.zeros((1, 1, 2)))
    assert tuple(logits.shape) == (1, 1, 6)


def _binary_v10_config() -> dict:
    return {
        "raw_ptm_dim": 2048,
        "projected_ptm_dim": 128,
        "mfcc_dim": 40,
        "position_dim": 2,
        "mfcc_mean": [0.0] * 40,
        "mfcc_std": [1.0] * 40,
        "hidden_size": 8,
        "num_layers": 1,
        "state_size": 4,
        "num_heads": 2,
        "head_dim": 8,
        "n_groups": 1,
        "conv_kernel": 2,
        "chunk_size": 2,
        "bidirectional": True,
        "valid_prefix_bidirectional": True,
        "model_arch": SPEECH_ISLAND_SCORER_V10_MODEL_ARCH,
        "output_dim": 2,
    }


def test_binary_speech_v10_checkpoint_is_random_init_argmax_only(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = _binary_v10_config()
    model = BinarySpeechIslandScorerNetwork(**config)
    payload = build_speech_island_scorer_checkpoint(
        model=model,
        model_config=config,
        normalization={"mfcc_mean": [0.0] * 40, "mfcc_std": [1.0] * 40},
        metadata={"ptm_repo_id": "repo/1.7b"},
        schema=SPEECH_ISLAND_SCORER_V10_SCHEMA,
    )
    checkpoint = tmp_path / "scorer-v10.pt"
    torch.save(payload, checkpoint)
    loaded = load_speech_island_scorer_checkpoint(checkpoint, device="cpu")

    assert loaded.schema == SPEECH_ISLAND_SCORER_V10_SCHEMA
    assert loaded.metadata["decision_mode"] == "binary_frame_argmax"
    assert loaded.metadata["training_initialization"] == "random"
    assert loaded.metadata["excluded_training_labels"] == ["unsure"]
    assert loaded.metadata["dataset_contract"] == SPEECH_ISLAND_SCORER_V10_DATASET_CONTRACT

    with pytest.raises(ValueError, match="forbids warm-start"):
        build_speech_island_scorer_checkpoint(
            model=model,
            model_config=config,
            normalization={},
            metadata={"training_initialization": "warm_start"},
            schema=SPEECH_ISLAND_SCORER_V10_SCHEMA,
        )


def test_binary_speech_v10_batching_matches_singletons(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = _binary_v10_config()
    payload = build_speech_island_scorer_checkpoint(
        model=BinarySpeechIslandScorerNetwork(**config).eval(),
        model_config=config,
        normalization={"mfcc_mean": [0.0] * 40, "mfcc_std": [1.0] * 40},
        metadata={"ptm_repo_id": "repo/1.7b"},
        schema=SPEECH_ISLAND_SCORER_V10_SCHEMA,
    )
    checkpoint = tmp_path / "scorer-v10.pt"
    torch.save(payload, checkpoint)
    bundle = load_speech_island_scorer_checkpoint(checkpoint, device="cpu")
    rng = np.random.default_rng(71)
    pairs = [
        (
            rng.normal(size=(frames, 2048)).astype(np.float32),
            rng.normal(size=(frames, 40)).astype(np.float32),
        )
        for frames in (5, 9, 7)
    ]
    batched = score_binary_speech_class_probabilities_batch(
        bundle, feature_pairs=pairs
    )
    singleton = [
        score_binary_speech_class_probabilities_batch(bundle, feature_pairs=[pair])[0]
        for pair in pairs
    ]
    for left, right in zip(batched, singleton, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(np.argmax(left, axis=1), np.argmax(right, axis=1))


def _scorer_v10_row(
    source: str, core: str | None, partition: str, *, row_role: str = "speech"
) -> dict:
    return {
        "source_id": source,
        "core_ids": [] if core is None else [core],
        "background_id": source if core is None else "",
        "row_role": row_role,
        "partition": partition,
        "input_distribution": "full_source_windows",
        "canonical_label_schema": "speech_scorer_canonical_frames_v1",
        "ptm_repo_id": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
        "feature_path": f"{source}.features.npz",
        "label_path": f"{source}.labels.npz",
        "frame_count": 3,
    }


def test_binary_speech_v10_dataset_contract_freezes_source_and_core() -> None:
    rows = [
        _scorer_v10_row("s1", "c1", "train"),
        _scorer_v10_row("s2", "c2", "val"),
        _scorer_v10_row("s3", "c3", "test"),
    ]
    assert validate_scorer_v10_rows(rows)["max_core_use_count"] == 1
    with pytest.raises(ValueError, match="source is duplicated"):
        validate_scorer_v10_rows([*rows, _scorer_v10_row("s1", "c4", "train")])
    with pytest.raises(ValueError, match="max core use"):
        validate_scorer_v10_rows([*rows, _scorer_v10_row("s4", "c1", "train")])


def test_binary_speech_v10_unsure_is_excluded_from_normalization_and_presence(
    monkeypatch,
) -> None:
    rows = [
        _scorer_v10_row("train-speech", "c1", "train"),
        _scorer_v10_row("train-bg", None, "train", row_role="all_background"),
        _scorer_v10_row("val-speech", "c3", "val"),
        _scorer_v10_row("val-bg", None, "val", row_role="all_background"),
        _scorer_v10_row("test-speech", "c5", "test"),
        _scorer_v10_row("test-bg", None, "test", row_role="all_background"),
    ]

    def fake_load(row):
        speech = row["source_id"].endswith("speech")
        labels = np.asarray([0, -100, 1 if speech else 0], dtype=np.int64)
        mfcc = np.asarray([[1.0], [1000.0], [3.0]], dtype=np.float32)
        return np.zeros((3, 2), dtype=np.float32), mfcc, labels, np.ones(3)

    monkeypatch.setattr(scorer_v10_trainer, "load_binary_row", fake_load)
    normalization = compute_mfcc_normalization(rows[:2])
    assert normalization["mfcc_mean"] == pytest.approx([2.0])
    assert normalization["mfcc_std"] == pytest.approx([1.0])
    presence, counts = summarize_partition_labels(rows)
    assert all(
        value == {"speech_rows": 1, "all_background_rows": 1}
        for value in presence.values()
    )
    assert counts["unsure"] == 6


def test_binary_speech_v10_trainer_has_no_legacy_tuning_surface() -> None:
    args = parse_scorer_v10_args(
        ["--dataset-manifest", "manifest.jsonl", "--output-dir", "out"]
    )
    assert not hasattr(args, "warm_start_checkpoint")
    assert not hasattr(args, "focal_gamma")
    assert not hasattr(args, "threshold")
    assert not hasattr(args, "projected_ptm_dim")
    assert scorer_v10_release_gate_fields(True)["promotion_ready"] is False
