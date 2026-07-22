from __future__ import annotations

import numpy as np
import pytest

from boundary.ja.model import (
    SPEECH_ISLAND_SCORER_V10_DATASET_CONTRACT,
    SPEECH_ISLAND_SCORER_V10_MODEL_ARCH,
    SPEECH_ISLAND_SCORER_V10_SCHEMA,
    SPEECH_ISLAND_SCORER_V10_TRAINING_ROW_SCHEMA,
    BinarySpeechIslandScorerNetwork,
    SemanticSpeechScorerNetwork,
    build_speech_island_scorer_checkpoint,
    load_speech_island_scorer_checkpoint,
    score_binary_speech_class_probabilities_batch,
)
import tools.boundary.ja.train_speech_island_scorer_v10_binary as scorer_v10_trainer
from tools.boundary.ja.train_speech_island_scorer_v10_binary import (
    checkpoint_selection_score,
    compute_mfcc_normalization,
    frame_budget_batches as scorer_v10_frame_budget_batches,
    internal_background_run_structure,
    numeric_gate_pass as scorer_v10_numeric_gate_pass,
    parse_args as parse_scorer_v10_args,
    predicted_run_structure,
    release_gate_fields as scorer_v10_release_gate_fields,
    speech_continuity_auxiliary_loss,
    internal_background_run_worst_frame_auxiliary_loss,
    sequence_worst_frame_auxiliary_loss,
    summarize_partition_labels,
    validate_dataset_rows as validate_scorer_v10_rows,
)


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


def _binary_v10_metadata(**overrides) -> dict:
    metadata = {
        "ptm_repo_id": "repo/1.7b",
        "dataset_manifest": "training.jsonl",
        "dataset_manifest_sha256": "1" * 64,
        "feature_manifest": "signed-features.jsonl",
        "signed_feature_manifest_sha256": "2" * 64,
        "canonical_sources_sha256": "3" * 64,
        "feature_cache_gate": "cache-gate.json",
        "feature_cache_gate_sha256": "4" * 64,
        "feature_config_sha256": "5" * 64,
    }
    metadata.update(overrides)
    return metadata


def test_binary_speech_v10_checkpoint_is_random_init_argmax_only(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = _binary_v10_config()
    model = BinarySpeechIslandScorerNetwork(**config)
    payload = build_speech_island_scorer_checkpoint(
        model=model,
        model_config=config,
        normalization={"mfcc_mean": [0.0] * 40, "mfcc_std": [1.0] * 40},
        metadata=_binary_v10_metadata(),
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
            metadata=_binary_v10_metadata(training_initialization="warm_start"),
            schema=SPEECH_ISLAND_SCORER_V10_SCHEMA,
        )


def test_binary_speech_v10_batching_matches_singletons(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = _binary_v10_config()
    payload = build_speech_island_scorer_checkpoint(
        model=BinarySpeechIslandScorerNetwork(**config).eval(),
        model_config=config,
        normalization={"mfcc_mean": [0.0] * 40, "mfcc_std": [1.0] * 40},
        metadata=_binary_v10_metadata(),
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
        "schema": SPEECH_ISLAND_SCORER_V10_TRAINING_ROW_SCHEMA,
        "boundary_serialization_contract_id": "boundary_acoustic_binary_v12",
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
        "canonical_sources_sha256": "1" * 64,
        "signed_feature_manifest_sha256": "2" * 64,
        "feature_cache_gate": "cache-gate.json",
        "feature_cache_gate_sha256": "3" * 64,
        "feature_config_sha256": "4" * 64,
        "audio_sha256": "5" * 64,
        "feature_sha256": "6" * 64,
        "label_sha256": "7" * 64,
    }


def test_binary_speech_v10_dataset_contract_freezes_source_and_core() -> None:
    rows = [
        _scorer_v10_row("s1", "c1", "train"),
        _scorer_v10_row("s2", "c2", "val"),
        _scorer_v10_row("s3", "c3", "test"),
    ]
    assert validate_scorer_v10_rows(rows, verify_content=False)["max_core_use_count"] == 1
    with pytest.raises(ValueError, match="source is duplicated"):
        validate_scorer_v10_rows(
            [*rows, _scorer_v10_row("s1", "c4", "train")],
            verify_content=False,
        )
    with pytest.raises(ValueError, match="max core use"):
        validate_scorer_v10_rows(
            [*rows, _scorer_v10_row("s4", "c1", "train")],
            verify_content=False,
        )
    missing_contract = dict(rows[0])
    missing_contract.pop("boundary_serialization_contract_id")
    with pytest.raises(ValueError, match="central Boundary contract"):
        validate_scorer_v10_rows(
            [missing_contract, *rows[1:]], verify_content=False
        )
    diagnostic = {**rows[0], "schema": "speech_scorer_v10_binary_diagnostic_row_v1", "diagnostic_only": True}
    with pytest.raises(ValueError, match="rejects diagnostic"):
        validate_scorer_v10_rows([diagnostic, *rows[1:]], verify_content=False)


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
    assert args.seed == 17
    assert scorer_v10_release_gate_fields(True)["promotion_ready"] is False


def test_binary_speech_v10_batches_by_padded_frame_budget_without_truncation() -> None:
    rows = [
        {"source_id": "short-a", "frame_count": 3},
        {"source_id": "short-b", "frame_count": 4},
        {"source_id": "long", "frame_count": 11},
    ]
    batches = scorer_v10_frame_budget_batches(rows, max_padded_frames=10)
    assert [[row["source_id"] for row in batch] for batch in batches] == [
        ["short-a", "short-b"],
        ["long"],
    ]
    assert batches[-1][0]["frame_count"] == 11


def test_binary_speech_v10_frame_budget_can_limit_packed_row_count() -> None:
    rows = [{"frame_count": 2, "source_id": str(index)} for index in range(9)]
    batches = scorer_v10_frame_budget_batches(
        rows, max_padded_frames=1024, max_batch_rows=4
    )
    assert [len(batch) for batch in batches] == [4, 4, 1]


def test_binary_speech_v10_continuity_counts_internal_argmax_holes() -> None:
    torch = pytest.importorskip("torch")
    structure = predicted_run_structure(
        torch.tensor([1, 1, 0, 1, 0, 0, 1, 1], dtype=torch.bool)
    )
    assert structure == {
        "predicted_run_count": 3,
        "continuous": 0,
        "fragmented": 1,
        "internal_drop_gap_count": 2,
        "internal_drop_frame_count": 3,
    }


def test_binary_speech_v10_numeric_gate_requires_heldout_continuity() -> None:
    metrics = {
        "start_coverage": 0.99,
        "end_coverage": 0.99,
        "background_drop_recall": 0.96,
        "speech_run_continuity": 0.94,
        "true_speech_deletion_count": 0,
    }
    assert scorer_v10_numeric_gate_pass(metrics, metrics) is False
    assert scorer_v10_numeric_gate_pass(
        {**metrics, "speech_run_continuity": 0.95},
        {**metrics, "speech_run_continuity": 0.95},
    ) is True


def test_binary_speech_v10_evaluate_records_whole_run_deletion_identity(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")

    def fake_load(_row):
        return (
            np.zeros((3, 2), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
            np.asarray([1, 1, 0], dtype=np.int64),
            np.ones(3, dtype=np.float32),
        )

    class BackgroundModel(torch.nn.Module):
        def forward(self, ptm, mfcc, *, attention_mask=None):
            logits = torch.zeros((*ptm.shape[:2], 2), device=ptm.device)
            logits[..., 0] = 1.0
            return logits

    monkeypatch.setattr(scorer_v10_trainer, "load_binary_row", fake_load)
    metrics = scorer_v10_trainer.evaluate(
        BackgroundModel(),
        [
            {
                "source_id": "heldout-repair",
                "partition": "val",
                "frame_count": 3,
            }
        ],
        torch.device("cpu"),
        max_padded_frames=8,
        tolerance_frames=1,
    )

    assert metrics["true_speech_deletion_count"] == 1
    assert metrics["true_speech_deletion_items"] == [
        {
            "source_id": "heldout-repair",
            "partition": "val",
            "start_frame": 0,
            "end_frame": 2,
        }
    ]


def test_binary_speech_v10_checkpoint_selection_minimizes_false_keeps_after_safety() -> None:
    safe = {
        "start_coverage": 0.95,
        "end_coverage": 0.96,
        "background_drop_recall": 0.97,
        "speech_run_continuity": 0.98,
        "speech_recall": 0.99,
        "true_speech_deletion_count": 0,
        "independent_internal_background_false_keep_island_count": 4,
        "independent_internal_background_false_keep_frame_count": 10,
    }
    higher_safety_more_false_keeps = {
        **safe,
        "start_coverage": 0.99,
        "independent_internal_background_false_keep_island_count": 5,
    }
    unsafe_fewer_false_keeps = {
        **safe,
        "start_coverage": 0.94,
        "independent_internal_background_false_keep_island_count": 0,
    }

    assert checkpoint_selection_score(safe) > checkpoint_selection_score(
        higher_safety_more_false_keeps
    )
    assert checkpoint_selection_score(safe) > checkpoint_selection_score(
        unsafe_fewer_false_keeps
    )


def test_binary_speech_v10_internal_background_structure_is_topology_based() -> None:
    result = internal_background_run_structure(
        np.asarray([0, 1, 1, 0, 0, 1, 1, 0, -100]),
        np.asarray([1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=bool),
    )

    assert result == {
        "internal_background_run_count": 1,
        "fully_dropped_internal_background_run_count": 0,
        "internal_background_false_keep_frame_count": 1,
        "internal_background_false_keep_island_count": 1,
        "independent_internal_background_false_keep_frame_count": 0,
        "independent_internal_background_false_keep_island_count": 0,
    }


def test_binary_speech_v10_internal_background_structure_counts_independent_islands() -> None:
    result = internal_background_run_structure(
        np.asarray([1, 1, 0, 0, 0, 1, 1]),
        np.asarray([1, 1, 0, 1, 0, 1, 1], dtype=bool),
    )

    assert result["independent_internal_background_false_keep_frame_count"] == 1
    assert result["independent_internal_background_false_keep_island_count"] == 1


def test_binary_speech_v10_unsure_is_removed_from_background_bracketing() -> None:
    separated = internal_background_run_structure(
        np.asarray([1, 0, -100, 0, 1]),
        np.asarray([1, 1, 1, 1, 1], dtype=bool),
    )
    bracketed = internal_background_run_structure(
        np.asarray([1, -100, 0, -100, 1]),
        np.asarray([1, 1, 1, 1, 1], dtype=bool),
    )

    assert separated["internal_background_run_count"] == 0
    assert bracketed["internal_background_run_count"] == 1

    torch = pytest.importorskip("torch")
    labels = torch.tensor([[1, 0, -100, 0, 1]], dtype=torch.long)
    logits = torch.zeros((1, 5, 2), dtype=torch.float32)
    loss, term_count = internal_background_run_worst_frame_auxiliary_loss(
        logits, labels, labels
    )
    assert term_count == 0
    assert loss.item() == pytest.approx(0.0)


def test_binary_speech_v10_continuity_auxiliary_only_penalizes_inside_speech() -> None:
    torch = pytest.importorskip("torch")
    target = torch.tensor([[1, 1, 1, 0, -100, 1]], dtype=torch.long)
    stable = torch.tensor(
        [
            [
                [0.0, 2.0],
                [0.0, 2.0],
                [0.0, 2.0],
                [2.0, 0.0],
                [8.0, -8.0],
                [-8.0, 8.0],
            ]
        ]
    )
    unstable = stable.clone()
    unstable[0, 1] = torch.tensor([2.0, 0.0])
    stable_loss, stable_pairs = speech_continuity_auxiliary_loss(stable, target)
    unstable_loss, unstable_pairs = speech_continuity_auxiliary_loss(unstable, target)
    assert stable_pairs.item() == 2
    assert unstable_pairs.item() == 2
    assert stable_loss.item() == pytest.approx(0.0)
    assert unstable_loss.item() > 0.0


def test_binary_speech_v10_worst_frame_auxiliary_covers_runs_and_background() -> None:
    torch = pytest.importorskip("torch")
    labels = torch.tensor([[1, 1, 0, 1, 1], [0, 0, -100, 0, 0]], dtype=torch.long)
    logits = torch.tensor(
        [
            [[0.0, 2.0], [2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 2.0]],
            [[2.0, 0.0], [0.0, 2.0], [9.0, -9.0], [2.0, 0.0], [2.0, 0.0]],
        ]
    )
    loss, term_count = sequence_worst_frame_auxiliary_loss(logits, labels, labels)
    assert term_count == 3
    assert loss.item() > 0.0


def test_binary_speech_v10_internal_background_auxiliary_only_covers_bracketed_runs() -> None:
    torch = pytest.importorskip("torch")
    labels = torch.tensor(
        [
            [0, 1, 1, 0, 0, 1, 1, 0, -100],
            [0, 0, 0, 0, -100, -100, -100, -100, -100],
        ],
        dtype=torch.long,
    )
    logits = torch.zeros((2, 9, 2), dtype=torch.float32)
    logits[..., 0] = 2.0
    logits[..., 1] = 0.0
    logits[0, 3] = torch.tensor([0.0, 2.0])
    logits[0, 0] = torch.tensor([0.0, 8.0])
    logits[0, 7] = torch.tensor([0.0, 8.0])
    logits[1, 1] = torch.tensor([0.0, 8.0])

    loss, term_count = internal_background_run_worst_frame_auxiliary_loss(
        logits, labels, labels
    )

    assert term_count == 1
    assert loss.item() == pytest.approx(
        torch.nn.functional.cross_entropy(logits[0, 3:4], labels[0, 3:4]).item()
    )
