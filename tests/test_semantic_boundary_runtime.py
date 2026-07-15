from __future__ import annotations

import numpy as np

from boundary.base import SpeechSegment
from boundary.outer_refiner import OuterEdgePrediction
from boundary.outer_refiner_v2 import OuterEdgeRefinerV2
from boundary.runtime_pipeline import (
    SemanticBoundaryConfig,
    build_semantic_boundary_chunks,
    effective_semantic_config,
    semantic_config_payload,
)
from boundary.sequence_features import FrameSequenceFeatureConfig, FrameSequenceFeatureProvider
from boundary.split_model import SplitDecision


class _OuterRefiner:
    feature_config = {"context_s": 0.6, "ptm_dim": 4}

    def predict(self, *, frame_features, scalar_features):
        assert frame_features.shape[0] == 1
        return [OuterEdgePrediction(0.1, -0.1, 0.9, 0.9)]


class _SplitVerifier:
    feature_config = {
        "left_context_s": 1.6,
        "right_context_s": 1.6,
        "gap_context_s": 0.3,
        "left_bins": 8,
        "gap_bins": 4,
        "right_bins": 8,
        "ptm_dim": 4,
    }

    def decide(self, *, frame_features, scalar_features):
        return [SplitDecision("cut", 0.95, 0.03, 0.02)]

    def decide_islands(self, *, island_frame_features, island_scalar_features):
        counts = [int(frames.shape[0]) for frames in island_frame_features]
        if not any(counts):
            return [[] for _count in counts]
        flat = self.decide(
            frame_features=np.concatenate(island_frame_features, axis=0),
            scalar_features=np.concatenate(island_scalar_features, axis=0),
        )
        groups = []
        offset = 0
        for count in counts:
            groups.append(flat[offset : offset + count])
            offset += count
        return groups


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
        return proposal_times_s + 0.08


class _OuterV2Logits:
    def __call__(self, frames):
        import torch

        logits = torch.full((frames.shape[0], frames.shape[1], 3), -6.0)
        logits[..., 0] = 6.0
        logits[:, 1:3, 0] = -6.0
        logits[:, 1:3, 1] = 6.0
        return logits


def test_effective_semantic_config_reports_checkpoint_decision() -> None:
    verifier = _SplitVerifier()
    verifier.decision_config = {
        "normal_cut_threshold": 0.75,
        "short_core_cut_threshold": 0.775,
        "duration_pressure_enabled": True,
        "duration_pressure_log_median": -0.1,
        "duration_pressure_log_mad": 1.05,
        "duration_pressure_z": 1.7,
        "duration_pressure_floor": 0.5,
    }

    resolved = effective_semantic_config(verifier, SemanticBoundaryConfig())
    payload = semantic_config_payload(resolved)

    assert payload["decision_mode"] == "threshold"
    assert payload["duration_pressure_enabled"] is True
    assert payload["duration_pressure_z"] == 1.7
    assert payload["short_core_cut_threshold"] == 0.775


def test_internal_cut_uses_one_shared_absolute_timestamp():
    provider = FrameSequenceFeatureProvider(
        duration_s=8.0,
        frame_hop_s=0.02,
        ptm=np.zeros((400, 4), dtype=np.float32),
        mfcc=np.zeros((400, 2), dtype=np.float32),
        config=FrameSequenceFeatureConfig(max_ptm_dims=4),
    )
    segment = SpeechSegment(
        start=0.5,
        end=7.5,
        weak_cut_candidates=[
            {
                "kind": "proposal",
                "time_s": 4.0,
                "frame": 200,
                "score": 0.8,
                "prominence": 0.2,
                "speech_valley": 0.5,
                "strength": 1.5,
            }
        ],
    )
    stages: list[str] = []
    chunks = build_semantic_boundary_chunks(
        [segment],
        duration_s=8.0,
        speech_probabilities=np.ones(400, dtype=np.float32),
        feature_provider=provider,
        outer_refiner=_OuterRefiner(),
        split_verifier=_SplitVerifier(),
        cut_refiner=_CutRefiner(),
        on_stage=stages.append,
    )

    assert len(chunks) == 2
    assert chunks[0].start == 0.6
    assert chunks[0].end == chunks[1].start == 4.08
    assert chunks[1].end == 7.4
    assert chunks[0].source_abs_end == chunks[1].source_abs_start
    assert chunks[0].primary_cut_candidates[0]["shared_absolute_timestamp"] is True
    assert stages == [
        "外边界精修 0/1",
        "外边界精修 1/1",
        "语义切分判断 0/1",
        "语义切分判断 1/1",
        "内部切点精修 0/1",
        "内部切点精修 1/1",
    ]


def test_outer_v2_materializes_paired_edges_and_probability_metadata() -> None:
    provider = FrameSequenceFeatureProvider(
        duration_s=0.08,
        frame_hop_s=0.02,
        ptm=np.zeros((4, 2048), dtype=np.float32),
        mfcc=np.zeros((4, 40), dtype=np.float32),
        config=FrameSequenceFeatureConfig(max_ptm_dims=2048),
    )
    outer = OuterEdgeRefinerV2(
        path="outer-v2.pt",
        sha256="sha",
        model=_OuterV2Logits(),
        model_config={},
        feature_config={"raw_ptm_dim": 2048},
        normalization={
            "feature_mean": [0.0] * 2089,
            "feature_std": [1.0] * 2089,
        },
        metadata={},
        device="cpu",
    )

    chunks = build_semantic_boundary_chunks(
        [SpeechSegment(start=0.0, end=0.08)],
        duration_s=0.08,
        speech_probabilities=np.ones(4, dtype=np.float32),
        feature_provider=provider,
        outer_refiner=outer,
        split_verifier=_SplitVerifier(),
        cut_refiner=_CutRefiner(),
    )

    assert len(chunks) == 1
    assert chunks[0].start == 0.02
    assert chunks[0].end == 0.06
    assert chunks[0].boundary_decision_source == "outer_edge_refiner_v2"
    assert chunks[0].boundary_start_refine_delta_s == 0.02
    assert np.isclose(chunks[0].boundary_end_refine_delta_s, -0.02)
    assert chunks[0].refiner_start_confidence > 0.99
    assert chunks[0].refiner_end_confidence > 0.99


def test_short_core_prefers_continue_below_high_threshold():
    class ShortSplit(_SplitVerifier):
        def decide(self, *, frame_features, scalar_features):
            return [SplitDecision("cut", 0.89, 0.08, 0.03)]

    provider = FrameSequenceFeatureProvider(
        duration_s=5.0,
        frame_hop_s=0.02,
        ptm=np.zeros((250, 4), dtype=np.float32),
        mfcc=np.zeros((250, 2), dtype=np.float32),
        config=FrameSequenceFeatureConfig(max_ptm_dims=4),
    )
    segment = SpeechSegment(
        start=0.0,
        end=5.0,
        weak_cut_candidates=[
            {
                "kind": "proposal",
                "time_s": 2.5,
                "frame": 125,
                "score": 0.8,
                "prominence": 0.2,
                "speech_valley": 0.5,
                "strength": 1.5,
            }
        ],
    )
    chunks = build_semantic_boundary_chunks(
        [segment],
        duration_s=5.0,
        speech_probabilities=np.ones(250, dtype=np.float32),
        feature_provider=provider,
        outer_refiner=_OuterRefiner(),
        split_verifier=ShortSplit(),
        cut_refiner=_CutRefiner(),
    )
    assert len(chunks) == 1


def test_duration_pressure_accepts_best_internal_cut_for_overlong_chunk():
    class PressureSplit(_SplitVerifier):
        def decide(self, *, frame_features, scalar_features):
            return [SplitDecision("cut", 0.62, 0.30, 0.08)]

    provider = FrameSequenceFeatureProvider(
        duration_s=22.0,
        frame_hop_s=0.02,
        ptm=np.zeros((1100, 4), dtype=np.float32),
        mfcc=np.zeros((1100, 2), dtype=np.float32),
        config=FrameSequenceFeatureConfig(max_ptm_dims=4),
    )
    segment = SpeechSegment(
        start=0.0,
        end=20.0,
        weak_cut_candidates=[
            {
                "kind": "proposal",
                "time_s": 10.0,
                "frame": 500,
                "score": 0.8,
                "prominence": 0.2,
                "speech_valley": 0.5,
                "strength": 1.5,
            }
        ],
    )

    chunks = build_semantic_boundary_chunks(
        [segment],
        duration_s=22.0,
        speech_probabilities=np.ones(1100, dtype=np.float32),
        feature_provider=provider,
        outer_refiner=_OuterRefiner(),
        split_verifier=PressureSplit(),
        cut_refiner=_CutRefiner(),
        config=SemanticBoundaryConfig(
            duration_pressure_enabled=True,
            duration_pressure_log_median=np.log(10.0),
            duration_pressure_log_mad=np.log(14.0 / 10.0) / (1.4826 * 2.0),
            duration_pressure_z=2.0,
            duration_pressure_floor=0.50,
        ),
    )

    assert len(chunks) == 2
    assert chunks[0].end == chunks[1].start == 10.08
    assert chunks[0].primary_cut_candidates[0]["duration_pressure_acceptance"] is True
    assert chunks[0].primary_cut_candidates[0]["p_cut"] == 0.62


def test_duration_pressure_does_not_accept_below_trigger():
    class PressureSplit(_SplitVerifier):
        def decide(self, *, frame_features, scalar_features):
            return [SplitDecision("cut", 0.62, 0.30, 0.08)]

    provider = FrameSequenceFeatureProvider(
        duration_s=14.0,
        frame_hop_s=0.02,
        ptm=np.zeros((700, 4), dtype=np.float32),
        mfcc=np.zeros((700, 2), dtype=np.float32),
        config=FrameSequenceFeatureConfig(max_ptm_dims=4),
    )
    segment = SpeechSegment(
        start=0.0,
        end=12.0,
        weak_cut_candidates=[
            {
                "kind": "proposal",
                "time_s": 6.0,
                "frame": 300,
                "score": 0.8,
                "prominence": 0.2,
                "speech_valley": 0.5,
                "strength": 1.5,
            }
        ],
    )

    chunks = build_semantic_boundary_chunks(
        [segment],
        duration_s=14.0,
        speech_probabilities=np.ones(700, dtype=np.float32),
        feature_provider=provider,
        outer_refiner=_OuterRefiner(),
        split_verifier=PressureSplit(),
        cut_refiner=_CutRefiner(),
        config=SemanticBoundaryConfig(
            duration_pressure_enabled=True,
            duration_pressure_log_median=np.log(10.0),
            duration_pressure_log_mad=np.log(14.0 / 10.0) / (1.4826 * 2.0),
            duration_pressure_z=2.0,
            duration_pressure_floor=0.50,
        ),
    )

    assert len(chunks) == 1


def test_split_inference_batches_candidates_without_cross_island_decision_leakage(monkeypatch):
    class MultiOuterRefiner(_OuterRefiner):
        def predict(self, *, frame_features, scalar_features):
            assert frame_features.shape[0] == 2
            return [
                OuterEdgePrediction(0.0, 0.0, 0.9, 0.9),
                OuterEdgePrediction(0.0, 0.0, 0.9, 0.9),
            ]

    class RecordingSplitVerifier(_SplitVerifier):
        def __init__(self):
            self.call_shapes = []

        def decide(self, *, frame_features, scalar_features):
            self.call_shapes.append((frame_features.shape, scalar_features.shape))
            assert frame_features.shape[0] == 2
            return [
                SplitDecision("cut", 0.95, 0.03, 0.02),
                SplitDecision("continue", 0.03, 0.95, 0.02),
            ]

    monkeypatch.setenv("SEMANTIC_SPLIT_INFERENCE_BATCH_SIZE", "128")
    verifier = RecordingSplitVerifier()
    provider = FrameSequenceFeatureProvider(
        duration_s=20.0,
        frame_hop_s=0.02,
        ptm=np.zeros((1000, 4), dtype=np.float32),
        mfcc=np.zeros((1000, 2), dtype=np.float32),
        config=FrameSequenceFeatureConfig(max_ptm_dims=4),
    )
    segments = [
        SpeechSegment(
            start=0.0,
            end=8.0,
            weak_cut_candidates=[
                {
                    "kind": "proposal",
                    "time_s": 4.0,
                    "frame": 200,
                    "score": 0.8,
                    "prominence": 0.2,
                    "speech_valley": 0.5,
                    "strength": 1.5,
                }
            ],
        ),
        SpeechSegment(
            start=10.0,
            end=18.0,
            weak_cut_candidates=[
                {
                    "kind": "proposal",
                    "time_s": 14.0,
                    "frame": 700,
                    "score": 0.8,
                    "prominence": 0.2,
                    "speech_valley": 0.5,
                    "strength": 1.5,
                }
            ],
        ),
    ]

    chunks = build_semantic_boundary_chunks(
        segments,
        duration_s=20.0,
        speech_probabilities=np.ones(1000, dtype=np.float32),
        feature_provider=provider,
        outer_refiner=MultiOuterRefiner(),
        split_verifier=verifier,
        cut_refiner=_CutRefiner(),
    )

    assert len(verifier.call_shapes) == 1
    assert [round(chunk.start, 2) for chunk in chunks] == [0.0, 4.08, 10.0]
    assert [round(chunk.end, 2) for chunk in chunks] == [4.08, 8.0, 18.0]
    assert len(chunks[0].primary_cut_candidates) == 1
    assert chunks[2].primary_cut_candidates == []
