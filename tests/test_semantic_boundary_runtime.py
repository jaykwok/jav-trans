from __future__ import annotations

import numpy as np

from boundary.base import SpeechSegment
from boundary.outer_refiner import OuterEdgePrediction
from boundary.runtime_pipeline import build_semantic_boundary_chunks
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
