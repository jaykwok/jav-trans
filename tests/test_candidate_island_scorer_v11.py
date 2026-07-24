from __future__ import annotations

import numpy as np
import pytest

from boundary.linear_chain_crf import BinaryLinearChainCrf
from boundary.ja.backend import (
    SpeechBoundaryJaConfig,
    decode_candidate_island_segments,
    require_current_runtime_scorer,
)
from boundary.ja.candidate_windows import (
    CANDIDATE_CONTEXT_OVERLAP_FRAMES,
    CANDIDATE_CONTEXT_WINDOW_FRAMES,
    CANDIDATE_WINDOW_OWNERSHIP,
    plan_candidate_context_windows,
    stitch_candidate_window_outputs,
)
from boundary.ja.candidate_training import (
    candidate_boundary_heatmap_loss,
    candidate_boundary_heatmap_targets,
    gradient_alignment,
)
from boundary.ja.model import (
    CANDIDATE_ISLAND_SCORER_V11_CAPACITY_PROFILES,
    CANDIDATE_ISLAND_SCORER_V11_COMPACT_CAPACITY_PROFILE,
    CANDIDATE_ISLAND_SCORER_V11_COMPACT_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_CRF_DATASET_CONTRACT,
    CANDIDATE_ISLAND_SCORER_V11_CRF_AB_AXIS,
    CANDIDATE_ISLAND_SCORER_V11_CRF_MODEL_ARCH,
    CANDIDATE_ISLAND_SCORER_V11_CRF_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_DATASET_CONTRACT,
    CANDIDATE_ISLAND_SCORER_V11_FULL_CAPACITY_PROFILE,
    CANDIDATE_ISLAND_SCORER_V11_LABELS,
    CANDIDATE_ISLAND_SCORER_V11_HEATMAP_AUXILIARY,
    CANDIDATE_ISLAND_SCORER_V11_HEATMAP_DATASET_CONTRACT,
    CANDIDATE_ISLAND_SCORER_V11_HEATMAP_MODEL_ARCH,
    CANDIDATE_ISLAND_SCORER_V11_HEATMAP_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_HEATMAP_SIGMA_FRAMES,
    CANDIDATE_ISLAND_SCORER_V11_MODEL_ARCH,
    CANDIDATE_ISLAND_SCORER_V11_SCHEMA,
    CandidateIslandScorerNetwork,
    CandidateIslandCrfScorerNetwork,
    CandidateIslandHeatmapScorerNetwork,
    SpeechIslandScorerBundle,
    SPEECH_ISLAND_SCORER_V10_SCHEMA,
    build_speech_island_scorer_checkpoint,
    load_speech_island_scorer_checkpoint,
    score_binary_speech_class_probabilities_batch,
    score_candidate_island_class_probabilities_batch,
    score_candidate_island_crf_outputs_batch,
    score_candidate_island_heatmap_class_probabilities_batch,
    score_candidate_island_heatmap_source_probabilities,
    score_candidate_island_source_probabilities,
    score_candidate_island_source_outputs,
)


def _config(
    capacity_profile: str = CANDIDATE_ISLAND_SCORER_V11_FULL_CAPACITY_PROFILE,
) -> dict:
    capacity = CANDIDATE_ISLAND_SCORER_V11_CAPACITY_PROFILES[capacity_profile]
    return {
        "raw_ptm_dim": 2048,
        "projected_ptm_dim": capacity["projected_ptm_dim"],
        "mfcc_dim": 40,
        "capacity_profile": capacity_profile,
        "mfcc_mean": [0.0] * 40,
        "mfcc_std": [1.0] * 40,
        "hidden_size": capacity["hidden_size"],
        "num_layers": capacity["num_layers"],
        "state_size": capacity["state_size"],
        "num_heads": capacity["num_heads"],
        "head_dim": capacity["head_dim"],
        "n_groups": capacity["n_groups"],
        "conv_kernel": capacity["conv_kernel"],
        "chunk_size": capacity["chunk_size"],
        "bidirectional": True,
        "valid_prefix_bidirectional": True,
        "context_window_frames": CANDIDATE_CONTEXT_WINDOW_FRAMES,
        "context_overlap_frames": CANDIDATE_CONTEXT_OVERLAP_FRAMES,
        "window_ownership": CANDIDATE_WINDOW_OWNERSHIP,
        "model_arch": capacity["model_arch"],
        "output_dim": 2,
    }


def _metadata(**overrides) -> dict:
    metadata = {
        "ptm_repo_id": "repo/1.7b",
        "dataset_manifest": "candidate-training.jsonl",
        "dataset_manifest_sha256": "1" * 64,
        "feature_manifest": "candidate-features.jsonl",
        "signed_feature_manifest_sha256": "2" * 64,
        "canonical_sources_sha256": "3" * 64,
        "feature_cache_gate": "candidate-cache-gate.json",
        "feature_cache_gate_sha256": "4" * 64,
        "feature_config_sha256": "5" * 64,
    }
    metadata.update(overrides)
    return metadata


def _heatmap_config() -> dict:
    return {
        **_config(),
        "model_arch": CANDIDATE_ISLAND_SCORER_V11_HEATMAP_MODEL_ARCH,
        "boundary_heatmap_sigma_frames": (
            CANDIDATE_ISLAND_SCORER_V11_HEATMAP_SIGMA_FRAMES
        ),
        "boundary_auxiliary": CANDIDATE_ISLAND_SCORER_V11_HEATMAP_AUXILIARY,
    }


def _crf_config() -> dict:
    return {
        **_config(),
        "model_arch": CANDIDATE_ISLAND_SCORER_V11_CRF_MODEL_ARCH,
    }


@pytest.mark.parametrize("total", [0, 1, 999, 1000, 1001, 1601, 1700, 1801, 5000])
def test_candidate_windows_partition_every_frame_once(total: int) -> None:
    windows = plan_candidate_context_windows(total)
    owned = [
        frame
        for window in windows
        for frame in range(window.owner_start_frame, window.owner_end_frame)
    ]

    assert owned == list(range(total))
    assert all(window.frame_count <= CANDIDATE_CONTEXT_WINDOW_FRAMES for window in windows)
    if total > CANDIDATE_CONTEXT_WINDOW_FRAMES:
        assert all(
            window.frame_count == CANDIDATE_CONTEXT_WINDOW_FRAMES
            for window in windows
        )


def test_candidate_stitch_copies_owned_frames_without_averaging() -> None:
    total = 1801
    windows = plan_candidate_context_windows(total)
    outputs = []
    expected = np.empty((total, 2), dtype=np.float32)
    for index, window in enumerate(windows):
        output = np.full((window.frame_count, 2), index + 1, dtype=np.float32)
        outputs.append(output)
        expected[window.owner_start_frame : window.owner_end_frame] = index + 1

    stitched = stitch_candidate_window_outputs(windows, outputs, total_frames=total)

    np.testing.assert_array_equal(stitched, expected)


def test_candidate_v11_checkpoint_is_strict_random_init_argmax_only(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = _config()
    model = CandidateIslandScorerNetwork(**config)
    payload = build_speech_island_scorer_checkpoint(
        model=model,
        model_config=config,
        normalization={"mfcc_mean": [0.0] * 40, "mfcc_std": [1.0] * 40},
        metadata=_metadata(),
        schema=CANDIDATE_ISLAND_SCORER_V11_SCHEMA,
    )
    checkpoint = tmp_path / "candidate-scorer-v11.pt"
    torch.save(payload, checkpoint)
    loaded = load_speech_island_scorer_checkpoint(checkpoint, device="cpu")

    assert loaded.schema == CANDIDATE_ISLAND_SCORER_V11_SCHEMA
    assert loaded.metadata["training_labels"] == list(
        CANDIDATE_ISLAND_SCORER_V11_LABELS
    )
    assert loaded.metadata["excluded_training_labels"] == ["unsure"]
    assert loaded.metadata["dataset_contract"] == (
        CANDIDATE_ISLAND_SCORER_V11_DATASET_CONTRACT
    )
    assert "position_dim" not in loaded.model_config
    assert tuple(loaded.model.ptm_projector.weight.shape) == (2048, 2048)
    assert tuple(loaded.model.frame_proj.weight.shape) == (256, 2088)

    legacy_payload = dict(payload)
    legacy_payload["schema"] = "speech_boundary_ja_candidate_island_scorer_v11"
    legacy_checkpoint = tmp_path / "legacy-candidate-scorer-v11.pt"
    torch.save(legacy_payload, legacy_checkpoint)
    with pytest.raises(ValueError, match="unsupported scorer checkpoint schema"):
        load_speech_island_scorer_checkpoint(legacy_checkpoint, device="cpu")

    with pytest.raises(ValueError, match="forbids warm-start"):
        build_speech_island_scorer_checkpoint(
            model=model,
            model_config=config,
            normalization={"mfcc_mean": [0.0] * 40, "mfcc_std": [1.0] * 40},
            metadata=_metadata(training_initialization="warm_start"),
            schema=CANDIDATE_ISLAND_SCORER_V11_SCHEMA,
        )
    with pytest.raises(ValueError, match="position features"):
        build_speech_island_scorer_checkpoint(
            model=model,
            model_config={**config, "position_dim": 2},
            normalization={"mfcc_mean": [0.0] * 40, "mfcc_std": [1.0] * 40},
            metadata=_metadata(),
            schema=CANDIDATE_ISLAND_SCORER_V11_SCHEMA,
        )
    with pytest.raises(ValueError, match="labels mismatch"):
        build_speech_island_scorer_checkpoint(
            model=model,
            model_config=config,
            normalization={"mfcc_mean": [0.0] * 40, "mfcc_std": [1.0] * 40},
            metadata=_metadata(labels=["background", "speech"]),
            schema=CANDIDATE_ISLAND_SCORER_V11_SCHEMA,
        )


def test_candidate_v11_batching_matches_singletons_and_rejects_v10_api(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = _config(CANDIDATE_ISLAND_SCORER_V11_COMPACT_CAPACITY_PROFILE)
    payload = build_speech_island_scorer_checkpoint(
        model=CandidateIslandScorerNetwork(**config).eval(),
        model_config=config,
        normalization={"mfcc_mean": [0.0] * 40, "mfcc_std": [1.0] * 40},
        metadata=_metadata(),
        schema=CANDIDATE_ISLAND_SCORER_V11_COMPACT_SCHEMA,
    )
    checkpoint = tmp_path / "candidate-scorer-v11.pt"
    torch.save(payload, checkpoint)
    bundle = load_speech_island_scorer_checkpoint(checkpoint, device="cpu")
    rng = np.random.default_rng(117)
    pairs = [
        (
            rng.normal(size=(frames, 2048)).astype(np.float32),
            rng.normal(size=(frames, 40)).astype(np.float32),
        )
        for frames in (5, 9, 7)
    ]

    batched = score_candidate_island_class_probabilities_batch(
        bundle, feature_pairs=pairs
    )
    singletons = [
        score_candidate_island_class_probabilities_batch(
            bundle, feature_pairs=[pair]
        )[0]
        for pair in pairs
    ]
    for batched_row, singleton_row in zip(batched, singletons, strict=True):
        np.testing.assert_allclose(batched_row, singleton_row, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(
            np.argmax(batched_row, axis=1), np.argmax(singleton_row, axis=1)
        )

    with pytest.raises(ValueError, match=SPEECH_ISLAND_SCORER_V10_SCHEMA):
        score_binary_speech_class_probabilities_batch(bundle, feature_pairs=pairs)


def test_candidate_crf_checkpoint_and_batching_are_strict(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = _crf_config()
    model = CandidateIslandCrfScorerNetwork(**config).eval()
    with torch.no_grad():
        model.crf.transitions.copy_(torch.tensor([[0.4, -0.3], [-0.2, 0.5]]))
    payload = build_speech_island_scorer_checkpoint(
        model=model,
        model_config=config,
        normalization={"mfcc_mean": [0.0] * 40, "mfcc_std": [1.0] * 40},
        metadata=_metadata(),
        schema=CANDIDATE_ISLAND_SCORER_V11_CRF_SCHEMA,
    )
    checkpoint = tmp_path / "candidate-scorer-v11-crf.pt"
    torch.save(payload, checkpoint)
    bundle = load_speech_island_scorer_checkpoint(checkpoint, device="cpu")

    assert bundle.schema == CANDIDATE_ISLAND_SCORER_V11_CRF_SCHEMA
    assert bundle.metadata["dataset_contract"] == (
        CANDIDATE_ISLAND_SCORER_V11_CRF_DATASET_CONTRACT
    )
    assert bundle.metadata["decision_mode"] == (
        "learned_binary_sequence_viterbi_argmax"
    )
    assert bundle.metadata["capacity_ab_axis"] == (
        CANDIDATE_ISLAND_SCORER_V11_CRF_AB_AXIS
    )
    assert bundle.metadata["runtime_threshold"] is None
    assert bundle.metadata["runtime_duration_rule"] is None
    rng = np.random.default_rng(1911)
    pairs = [
        (
            rng.normal(size=(frames, 2048)).astype(np.float32),
            rng.normal(size=(frames, 40)).astype(np.float32),
        )
        for frames in (5, 9, 7)
    ]
    batched = score_candidate_island_crf_outputs_batch(
        bundle, feature_pairs=pairs
    )
    singleton = [
        score_candidate_island_crf_outputs_batch(bundle, feature_pairs=[pair])[0]
        for pair in pairs
    ]
    for left, right in zip(batched, singleton, strict=True):
        np.testing.assert_allclose(
            left.probabilities, right.probabilities, atol=1e-5, rtol=1e-5
        )
        np.testing.assert_array_equal(left.labels, right.labels)


def test_candidate_crf_keeps_frozen_baseline_initialization_order() -> None:
    torch = pytest.importorskip("torch")
    config = _config()
    torch.manual_seed(117)
    baseline = CandidateIslandScorerNetwork(**config)
    baseline_state = {
        key: value.detach().clone()
        for key, value in baseline.state_dict().items()
        if key.startswith(("ptm_projector.", "frame_proj.", "encoder.", "head."))
    }
    del baseline

    torch.manual_seed(117)
    crf = CandidateIslandCrfScorerNetwork(**_crf_config())
    crf_state = crf.state_dict()
    for key, expected in baseline_state.items():
        crf_key = key.replace("head.", "emission_head.", 1)
        torch.testing.assert_close(crf_state[crf_key], expected)


def test_candidate_crf_source_scoring_is_frame_budget_equivalent() -> None:
    torch = pytest.importorskip("torch")

    class _DeterministicCrfModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.crf = BinaryLinearChainCrf()
            with torch.no_grad():
                self.crf.transitions.copy_(
                    torch.tensor([[0.6, -0.4], [-0.3, 0.5]])
                )

        def forward(self, ptm, mfcc, *, attention_mask=None):
            del attention_mask
            return torch.stack(
                (
                    ptm[..., 0] + 0.25 * mfcc[..., 0],
                    ptm[..., 1] - 0.15 * mfcc[..., 0],
                ),
                dim=-1,
            )

        def decode(self, emissions, attention_mask):
            return self.crf.decode(emissions, attention_mask)

        def marginal_probabilities(self, emissions, attention_mask):
            return self.crf.marginal_probabilities(emissions, attention_mask)

    bundle = SpeechIslandScorerBundle(
        path="fixture",
        sha256="0" * 64,
        model=_DeterministicCrfModel().eval(),
        model_config={"raw_ptm_dim": 2, "projected_ptm_dim": 2, "mfcc_dim": 1},
        normalization={},
        metadata={},
        device="cpu",
        schema=CANDIDATE_ISLAND_SCORER_V11_CRF_SCHEMA,
    )
    rng = np.random.default_rng(2211)
    ptm = rng.normal(size=(1801, 2)).astype(np.float32)
    mfcc = rng.normal(size=(1801, 1)).astype(np.float32)

    one_batch = score_candidate_island_source_outputs(
        bundle, ptm=ptm, mfcc=mfcc, max_padded_frames=2000
    )
    split_batches = score_candidate_island_source_outputs(
        bundle, ptm=ptm, mfcc=mfcc, max_padded_frames=1000
    )

    np.testing.assert_allclose(
        one_batch.probabilities,
        split_batches.probabilities,
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_array_equal(one_batch.labels, split_batches.labels)


def test_candidate_source_scoring_is_frame_budget_equivalent(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = _config(CANDIDATE_ISLAND_SCORER_V11_COMPACT_CAPACITY_PROFILE)
    payload = build_speech_island_scorer_checkpoint(
        model=CandidateIslandScorerNetwork(**config).eval(),
        model_config=config,
        normalization={"mfcc_mean": [0.0] * 40, "mfcc_std": [1.0] * 40},
        metadata=_metadata(),
        schema=CANDIDATE_ISLAND_SCORER_V11_COMPACT_SCHEMA,
    )
    checkpoint = tmp_path / "candidate-scorer-v11.pt"
    torch.save(payload, checkpoint)
    bundle = load_speech_island_scorer_checkpoint(checkpoint, device="cpu")
    rng = np.random.default_rng(711)
    ptm = rng.normal(size=(1801, 2048)).astype(np.float32)
    mfcc = rng.normal(size=(1801, 40)).astype(np.float32)

    one_batch = score_candidate_island_source_probabilities(
        bundle,
        ptm=ptm,
        mfcc=mfcc,
        max_padded_frames=2000,
    )
    split_batches = score_candidate_island_source_probabilities(
        bundle,
        ptm=ptm,
        mfcc=mfcc,
        max_padded_frames=1000,
    )

    np.testing.assert_allclose(one_batch, split_batches, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(
        np.argmax(one_batch, axis=1), np.argmax(split_batches, axis=1)
    )
    with pytest.raises(ValueError, match="verified no-spill capacity"):
        score_candidate_island_source_probabilities(
            bundle,
            ptm=ptm,
            mfcc=mfcc,
            max_padded_frames=2001,
        )


def test_candidate_v11_decoder_is_argmax_only_and_runtime_remains_pending() -> None:
    probabilities = np.asarray(
        [[0.6, 0.4], [0.49, 0.51], [0.2, 0.8], [0.7, 0.3]],
        dtype=np.float32,
    )
    result = decode_candidate_island_segments(
        class_probabilities=probabilities,
        candidate_probabilities=np.zeros(4, dtype=np.float32),
        duration_s=0.08,
        config=SpeechBoundaryJaConfig(
            threshold=0.99,
            frame_dilation_s=10.0,
            min_segment_s=10.0,
            frame_hop_s=0.02,
        ),
    )

    assert result.decision_mode == "binary_frame_argmax_candidate_membership"
    assert result.speech_on_threshold is None
    assert result.speech_off_threshold is None
    assert result.raw_frames.tolist() == [0, 1, 1, 0]
    assert result.dilated_frames.tolist() == [0, 1, 1, 0]
    assert [(row.start, row.end) for row in result.segments] == [(0.02, 0.06)]

    class _V11:
        schema = CANDIDATE_ISLAND_SCORER_V11_SCHEMA

    with pytest.raises(RuntimeError, match="pending_binary_scorer_audit"):
        require_current_runtime_scorer(_V11())


def test_candidate_heatmap_targets_are_soft_and_do_not_invent_unsure_boundaries() -> None:
    targets = candidate_boundary_heatmap_targets(
        np.asarray([0, 1, 1, 0, -100, 1, 1, 0], dtype=np.int64)
    )

    assert targets.start_frames == (1,)
    assert targets.end_frames == (2, 6)
    assert targets.start[1] == pytest.approx(1.0)
    assert 0.0 < targets.start[0] < 1.0
    assert targets.end[2] == pytest.approx(1.0)
    assert targets.end[6] == pytest.approx(1.0)
    assert targets.valid.tolist() == [True, True, True, True, False, True, True, True]


def test_candidate_heatmap_network_keeps_auxiliary_out_of_runtime_head() -> None:
    torch = pytest.importorskip("torch")
    model = CandidateIslandHeatmapScorerNetwork(**_heatmap_config()).eval()
    ptm = torch.zeros((2, 7, 2048), dtype=torch.float32)
    mfcc = torch.zeros((2, 7, 40), dtype=torch.float32)
    mask = torch.ones((2, 7), dtype=torch.int64)

    outputs = model.forward_outputs(ptm, mfcc, attention_mask=mask)
    runtime_logits = model(ptm, mfcc, attention_mask=mask)

    assert tuple(outputs["class_logits"].shape) == (2, 7, 2)
    assert tuple(outputs["start_boundary_logits"].shape) == (2, 7, 1)
    assert tuple(outputs["end_boundary_logits"].shape) == (2, 7, 1)
    assert torch.equal(runtime_logits, outputs["class_logits"])


def test_candidate_heatmap_checkpoint_and_batching_are_strict(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = _heatmap_config()
    payload = build_speech_island_scorer_checkpoint(
        model=CandidateIslandHeatmapScorerNetwork(**config).eval(),
        model_config=config,
        normalization={"mfcc_mean": [0.0] * 40, "mfcc_std": [1.0] * 40},
        metadata=_metadata(),
        schema=CANDIDATE_ISLAND_SCORER_V11_HEATMAP_SCHEMA,
    )
    checkpoint = tmp_path / "candidate-scorer-v11-heatmap.pt"
    torch.save(payload, checkpoint)
    bundle = load_speech_island_scorer_checkpoint(checkpoint, device="cpu")

    assert bundle.metadata["dataset_contract"] == (
        CANDIDATE_ISLAND_SCORER_V11_HEATMAP_DATASET_CONTRACT
    )
    assert bundle.metadata["runtime_auxiliary_decoder"] == "disabled_ab_only"
    rng = np.random.default_rng(811)
    pairs = [
        (
            rng.normal(size=(frames, 2048)).astype(np.float32),
            rng.normal(size=(frames, 40)).astype(np.float32),
        )
        for frames in (5, 9, 7)
    ]
    batched = score_candidate_island_heatmap_class_probabilities_batch(
        bundle, feature_pairs=pairs
    )
    singleton = [
        score_candidate_island_heatmap_class_probabilities_batch(
            bundle, feature_pairs=[pair]
        )[0]
        for pair in pairs
    ]
    for left, right in zip(batched, singleton, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(np.argmax(left, axis=1), np.argmax(right, axis=1))

    ptm = rng.normal(size=(1801, 2048)).astype(np.float32)
    mfcc = rng.normal(size=(1801, 40)).astype(np.float32)
    one_batch = score_candidate_island_heatmap_source_probabilities(
        bundle, ptm=ptm, mfcc=mfcc, max_padded_frames=2000
    )
    split_batches = score_candidate_island_heatmap_source_probabilities(
        bundle, ptm=ptm, mfcc=mfcc, max_padded_frames=1000
    )
    np.testing.assert_allclose(one_batch, split_batches, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(
        np.argmax(one_batch, axis=1), np.argmax(split_batches, axis=1)
    )


def test_candidate_heatmap_loss_and_gradient_alignment() -> None:
    torch = pytest.importorskip("torch")
    start_logits = torch.zeros((1, 3, 1), requires_grad=True)
    end_logits = torch.zeros((1, 3, 1), requires_grad=True)
    targets = torch.tensor([[0.0, 1.0, 0.0]])
    valid = torch.tensor([[True, False, True]])
    loss = candidate_boundary_heatmap_loss(
        start_logits=start_logits,
        end_logits=end_logits,
        start_targets=targets,
        end_targets=targets,
        valid_mask=valid,
    )
    assert float(loss.detach()) == pytest.approx(np.log(2.0))

    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    main_loss = torch.sum(parameter**2)
    auxiliary_loss = torch.sum(parameter**2) * 0.5
    alignment = gradient_alignment(main_loss, auxiliary_loss, [parameter])
    assert alignment["main_gradient_norm"] > 0.0
    assert alignment["auxiliary_gradient_norm"] > 0.0
    assert alignment["gradient_cosine"] == pytest.approx(1.0)
