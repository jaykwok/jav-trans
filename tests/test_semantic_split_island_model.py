from __future__ import annotations

import numpy as np
import pytest
import torch

from boundary.base import SpeechSegment
from boundary.outer_refiner import OuterEdgePrediction
from boundary.runtime_pipeline import build_semantic_boundary_chunks
from boundary.sequence_features import (
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
)
from boundary.split_model import (
    SEMANTIC_SPLIT_V2_MODEL_ARCH,
    SEMANTIC_SPLIT_V2_SCHEMA,
    IslandCandidateSequenceNetwork,
    SplitDecision,
    build_semantic_split_island_checkpoint,
    load_semantic_split_verifier,
)


FRAME_DIM = 6
SCALAR_DIM = 13
BINS = 20


def _tiny_model_config() -> dict:
    return {
        "frame_dim": FRAME_DIM,
        "scalar_dim": SCALAR_DIM,
        "hidden_size": 16,
        "candidate_layers": 1,
        "island_layers": 1,
        "state_size": 8,
        "num_heads": 2,
        "head_dim": 16,
        "n_groups": 1,
        "conv_kernel": 4,
        "chunk_size": 8,
        "bidirectional": True,
        "dropout": 0.0,
    }


def _neutral_normalization() -> dict:
    return {
        "frame_mean": np.zeros(FRAME_DIM, dtype=np.float32).tolist(),
        "frame_std": np.ones(FRAME_DIM, dtype=np.float32).tolist(),
        "scalar_mean": np.zeros(SCALAR_DIM, dtype=np.float32).tolist(),
        "scalar_std": np.ones(SCALAR_DIM, dtype=np.float32).tolist(),
    }


def _island(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    frames = rng.normal(size=(count, BINS, FRAME_DIM)).astype(np.float32)
    scalars = rng.normal(size=(count, SCALAR_DIM)).astype(np.float32)
    return frames, scalars


def _save_checkpoint(tmp_path, decision_config=None):
    torch.manual_seed(7)
    model = IslandCandidateSequenceNetwork(**_tiny_model_config())
    payload = build_semantic_split_island_checkpoint(
        model=model,
        model_config=_tiny_model_config(),
        feature_config={
            "ptm_dim": 4,
            "mfcc_dim": 2,
            "left_context_s": 1.6,
            "right_context_s": 1.6,
            "gap_context_s": 0.3,
            "left_bins": 8,
            "gap_bins": 4,
            "right_bins": 8,
        },
        normalization=_neutral_normalization(),
        metadata={"ptm_repo_id": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"},
        decision_config=decision_config,
    )
    path = tmp_path / "semantic_split_model_v2.test.pt"
    torch.save(payload, path)
    return path


def test_island_network_output_shapes_and_mask():
    torch.manual_seed(3)
    model = IslandCandidateSequenceNetwork(**_tiny_model_config())
    model.eval()
    frames = torch.randn(2, 5, BINS, FRAME_DIM)
    scalars = torch.randn(2, 5, SCALAR_DIM)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.long)
    with torch.inference_mode():
        outputs = model(frames, scalars, mask)
    assert outputs["gate"].shape == (2, 5)
    assert outputs["label"].shape == (2, 5, 3)
    assert outputs["role"].shape == (2, 5, 4)
    assert outputs["omni"].shape == (2, 5, 3)


def test_decide_islands_batch_independent(tmp_path):
    path = _save_checkpoint(tmp_path)
    verifier = load_semantic_split_verifier(path, device="cpu")
    island_a = _island(3, seed=11)
    island_b = _island(7, seed=23)
    solo = verifier.decide_islands(
        island_frame_features=[island_a[0]],
        island_scalar_features=[island_a[1]],
    )
    batched = verifier.decide_islands(
        island_frame_features=[island_a[0], island_b[0]],
        island_scalar_features=[island_a[1], island_b[1]],
    )
    assert len(batched) == 2
    assert len(batched[0]) == 3
    assert len(batched[1]) == 7
    for lone, joint in zip(solo[0], batched[0]):
        assert lone.label == joint.label
        assert lone.p_cut == pytest.approx(joint.p_cut, abs=1e-4)
        assert lone.p_continue == pytest.approx(joint.p_continue, abs=1e-4)
    for decision in batched[1]:
        total = decision.p_cut + decision.p_continue + decision.p_unsure
        assert total == pytest.approx(1.0, abs=1e-4)


def test_checkpoint_round_trip_and_contract(tmp_path):
    path = _save_checkpoint(
        tmp_path,
        decision_config={"normal_cut_threshold": 0.6},
    )
    verifier = load_semantic_split_verifier(path, device="cpu")
    assert verifier.signature()["schema"] == SEMANTIC_SPLIT_V2_SCHEMA
    assert verifier.signature()["model_arch"] == SEMANTIC_SPLIT_V2_MODEL_ARCH
    assert verifier.decision_config["normal_cut_threshold"] == 0.6
    assert verifier.decision_config["short_core_cut_threshold"] == 0.90

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["metadata"]["artifact"]["version"] = "v1"
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(ValueError, match="artifact.version"):
        load_semantic_split_verifier(broken, device="cpu")

    with pytest.raises(ValueError):
        load_semantic_split_verifier(
            path,
            device="cpu",
            expected_ptm_repo_id="jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf",
        )


class _OuterRefiner:
    feature_config = {"context_s": 0.6, "ptm_dim": 4}

    def predict(self, *, frame_features, scalar_features):
        return [
            OuterEdgePrediction(0.0, 0.0, 0.9, 0.9)
            for _ in range(frame_features.shape[0])
        ]


class _CutRefiner:
    feature_config = {
        "context_s": 1.6,
        "gap_context_s": 0.3,
        "bins": [8, 4, 8],
        "ptm_dim": 4,
    }

    def refine(
        self,
        *,
        proposal_times_s,
        frame_features,
        scalar_features,
        core_start_s,
        core_end_s,
    ):
        return proposal_times_s


class _IslandVerifier:
    feature_config = {
        "left_context_s": 1.6,
        "right_context_s": 1.6,
        "gap_context_s": 0.3,
        "left_bins": 8,
        "gap_bins": 4,
        "right_bins": 8,
        "ptm_dim": 4,
    }
    decision_config = {
        "short_core_max_s": 6.0,
        "short_core_cut_threshold": 0.9,
        "normal_cut_threshold": 0.6,
        "min_chunk_after_split_s": 1.2,
    }

    def __init__(self):
        self.calls: list[list[int]] = []

    def decide_islands(self, *, island_frame_features, island_scalar_features):
        self.calls.append([int(f.shape[0]) for f in island_frame_features])
        decisions = []
        for frames in island_frame_features:
            rows = []
            for position in range(frames.shape[0]):
                if position == 0:
                    rows.append(SplitDecision("cut", 0.65, 0.30, 0.05))
                else:
                    rows.append(SplitDecision("continue", 0.05, 0.90, 0.05))
            decisions.append(rows)
        return decisions


def _proposal(time_s: float, frame: int) -> dict:
    return {
        "kind": "proposal",
        "time_s": time_s,
        "frame": frame,
        "score": 0.8,
        "prominence": 0.2,
        "speech_valley": 0.5,
        "strength": 1.5,
    }


def test_runtime_island_path_uses_checkpoint_thresholds():
    provider = FrameSequenceFeatureProvider(
        duration_s=20.0,
        frame_hop_s=0.02,
        ptm=np.zeros((1000, 4), dtype=np.float32),
        mfcc=np.zeros((1000, 2), dtype=np.float32),
        config=FrameSequenceFeatureConfig(max_ptm_dims=4),
    )
    verifier = _IslandVerifier()
    segments = [
        SpeechSegment(
            start=0.0,
            end=8.0,
            weak_cut_candidates=[_proposal(4.0, 200), _proposal(6.0, 300)],
        ),
        SpeechSegment(
            start=10.0,
            end=18.0,
            weak_cut_candidates=[_proposal(14.0, 700)],
        ),
    ]
    chunks = build_semantic_boundary_chunks(
        segments,
        duration_s=20.0,
        speech_probabilities=np.ones(1000, dtype=np.float32),
        feature_provider=provider,
        outer_refiner=_OuterRefiner(),
        split_verifier=verifier,
        cut_refiner=_CutRefiner(),
    )
    # One batched call carrying both islands, ordered candidate counts intact.
    assert verifier.calls == [[2, 1]]
    # p_cut=0.65 passes the checkpoint-calibrated 0.6 gate on both islands.
    assert [round(chunk.start, 2) for chunk in chunks] == [0.0, 4.0, 10.0, 14.0]
    assert chunks[0].primary_cut_candidates[0]["p_cut"] == pytest.approx(0.65)


def test_runtime_island_path_default_thresholds_reject_low_gate():
    provider = FrameSequenceFeatureProvider(
        duration_s=20.0,
        frame_hop_s=0.02,
        ptm=np.zeros((1000, 4), dtype=np.float32),
        mfcc=np.zeros((1000, 2), dtype=np.float32),
        config=FrameSequenceFeatureConfig(max_ptm_dims=4),
    )

    class DefaultVerifier(_IslandVerifier):
        decision_config: dict = {}

    verifier = DefaultVerifier()
    segments = [
        SpeechSegment(
            start=0.0,
            end=8.0,
            weak_cut_candidates=[_proposal(4.0, 200)],
        )
    ]
    chunks = build_semantic_boundary_chunks(
        segments,
        duration_s=20.0,
        speech_probabilities=np.ones(1000, dtype=np.float32),
        feature_provider=provider,
        outer_refiner=_OuterRefiner(),
        split_verifier=verifier,
        cut_refiner=_CutRefiner(),
    )
    # Without checkpoint calibration the default 0.75 gate rejects p_cut=0.65.
    assert len(chunks) == 1


class _BracketVerifier(_IslandVerifier):
    """First two candidates form a speech_to_noise -> noise_to_speech bracket."""

    def __init__(self, roles: tuple[str, str]):
        super().__init__()
        self._roles = roles

    def decide_islands(self, *, island_frame_features, island_scalar_features):
        self.calls.append([int(f.shape[0]) for f in island_frame_features])
        decisions = []
        for frames in island_frame_features:
            rows = []
            for position in range(frames.shape[0]):
                if position < 2:
                    rows.append(
                        SplitDecision(
                            "cut",
                            0.95,
                            0.03,
                            0.02,
                            role=self._roles[position],
                            p_role=0.9,
                        )
                    )
                else:
                    rows.append(SplitDecision("continue", 0.05, 0.90, 0.05))
            decisions.append(rows)
        return decisions


def _run_bracket(verifier) -> list:
    provider = FrameSequenceFeatureProvider(
        duration_s=12.0,
        frame_hop_s=0.02,
        ptm=np.zeros((600, 4), dtype=np.float32),
        mfcc=np.zeros((600, 2), dtype=np.float32),
        config=FrameSequenceFeatureConfig(max_ptm_dims=4),
    )
    segments = [
        SpeechSegment(
            start=0.0,
            end=10.0,
            weak_cut_candidates=[
                _proposal(4.0, 200),
                _proposal(4.6, 230),  # 0.6s run, below min_chunk 1.2s
            ],
        )
    ]
    return build_semantic_boundary_chunks(
        segments,
        duration_s=12.0,
        speech_probabilities=np.ones(600, dtype=np.float32),
        feature_provider=provider,
        outer_refiner=_OuterRefiner(),
        split_verifier=verifier,
        cut_refiner=_CutRefiner(),
    )


def test_noise_isolation_bracket_exempts_min_spacing():
    chunks = _run_bracket(
        _BracketVerifier(("speech_to_noise", "noise_to_speech"))
    )
    # Both cuts accepted: speech | 0.6s noise run | speech.
    assert [round(chunk.start, 2) for chunk in chunks] == [0.0, 4.0, 4.6]
    middle = chunks[1]
    assert round(middle.end - middle.start, 2) == 0.6
    assert middle.primary_cut_candidates[0]["noise_isolation_bracket"] is True
    assert middle.primary_cut_candidates[0]["bracket_pair_id"]
    assert {
        candidate["role"]
        for chunk in chunks
        for candidate in chunk.primary_cut_candidates
    } == {"speech_to_noise", "noise_to_speech"}
    pair_ids = {
        candidate["bracket_pair_id"]
        for chunk in chunks
        for candidate in chunk.primary_cut_candidates
        if candidate.get("noise_isolation_bracket")
    }
    assert len(pair_ids) == 1


def test_close_cuts_without_bracket_roles_keep_min_spacing():
    chunks = _run_bracket(
        _BracketVerifier(("speech_to_speech", "speech_to_speech"))
    )
    # Only one cut survives the 1.2s spacing rule.
    assert [round(chunk.start, 2) for chunk in chunks] == [0.0, 4.0]
