from __future__ import annotations

import numpy as np

import boundary.runtime_pipeline as runtime_pipeline
from boundary.base import SpeechSegment
from boundary.runtime_pipeline import (
    _MAX_ISLAND_BATCH_CANDIDATES,
    _batched_island_split_decisions,
)
from boundary.sequence_features import FrameSequenceFeatureConfig, FrameSequenceFeatureProvider
from boundary.split_model import SplitDecision


class _OrderTaggingVerifier:
    """Tags each returned decision with a monotonic id via offset_s so the
    caller can verify slab decisions merge back in candidate order. Records the
    candidate count of every island forwarded to decide_islands."""

    def __init__(self) -> None:
        self.counter = 0
        self.forwarded_sizes: list[int] = []

    def decide_islands(self, *, island_frame_features, island_scalar_features):
        batch_out = []
        for frames in island_frame_features:
            n = int(frames.shape[0])
            self.forwarded_sizes.append(n)
            row = []
            for _ in range(n):
                row.append(
                    SplitDecision(
                        label="continue",
                        p_cut=0.0,
                        p_continue=1.0,
                        p_unsure=0.0,
                        offset_s=float(self.counter),
                    )
                )
                self.counter += 1
            batch_out.append(row)
        return batch_out


def test_oversized_island_is_slabbed_under_the_candidate_cap() -> None:
    cap = _MAX_ISLAND_BATCH_CANDIDATES
    verifier = _OrderTaggingVerifier()
    big = int(cap * 2.5)
    frame_groups = [
        np.zeros((10, 3, 5), dtype=np.float32),
        np.zeros((big, 3, 5), dtype=np.float32),
        np.zeros((5, 3, 5), dtype=np.float32),
    ]
    scalar_groups = [
        np.zeros((10, 2), dtype=np.float32),
        np.zeros((big, 2), dtype=np.float32),
        np.zeros((5, 2), dtype=np.float32),
    ]
    results = _batched_island_split_decisions(
        verifier,
        frame_feature_groups=frame_groups,
        scalar_feature_groups=scalar_groups,
    )

    # One merged decision list per input island, covering every candidate.
    assert len(results) == 3
    assert [len(r) for r in results] == [10, big, 5]
    # No single forward exceeded the candidate cap.
    assert max(verifier.forwarded_sizes) <= cap
    # The oversized island's merged decisions are in strict candidate order
    # (monotonic offset_s) and contiguous, proving slabs merged correctly.
    offsets = [d.offset_s for d in results[1]]
    assert offsets == sorted(offsets) and len(set(offsets)) == big


def test_small_islands_batch_together_under_cap() -> None:
    cap = _MAX_ISLAND_BATCH_CANDIDATES
    verifier = _OrderTaggingVerifier()
    # Many small islands whose total stays under the cap should be forwarded
    # together in a single decide_islands batch.
    frame_groups = [np.zeros((4, 3, 5), dtype=np.float32) for _ in range(8)]
    scalar_groups = [np.zeros((4, 2), dtype=np.float32) for _ in range(8)]
    results = _batched_island_split_decisions(
        verifier,
        frame_feature_groups=frame_groups,
        scalar_feature_groups=scalar_groups,
    )
    assert len(results) == 8
    assert all(len(r) == 4 for r in results)
    # All 8 islands forwarded in one batch (one decide_islands call -> 8 entries).
    assert verifier.forwarded_sizes == [4] * 8
    assert max(verifier.forwarded_sizes) <= cap


class _OuterPrediction:
    start_delta_s = 0.0
    end_delta_s = 0.0
    start_confidence = 0.0
    end_confidence = 0.0


class _OuterRefiner:
    feature_config = {"context_s": 0.1, "ptm_dim": 2}

    def predict(self, *, frame_features, scalar_features):
        return [_OuterPrediction() for _ in range(frame_features.shape[0])]


class _CutRefiner:
    feature_config = {
        "context_s": 0.1,
        "gap_context_s": 0.02,
        "bins": (1, 1, 1),
        "ptm_dim": 2,
    }

    def refine(self, **kwargs):  # pragma: no cover - no cuts are accepted here
        return kwargs["proposal_times_s"]


class _ContinueIslandVerifier:
    feature_config = {
        "left_context_s": 0.1,
        "right_context_s": 0.1,
        "gap_context_s": 0.02,
        "left_bins": 1,
        "gap_bins": 1,
        "right_bins": 1,
        "ptm_dim": 2,
    }
    decision_config = {
        "normal_cut_threshold": 0.75,
        "short_core_cut_threshold": 0.9,
        "short_core_max_s": 6.0,
        "min_chunk_after_split_s": 1.2,
    }

    def __init__(self) -> None:
        self.forwarded_sizes: list[int] = []

    def decide_islands(self, *, island_frame_features, island_scalar_features):
        out = []
        for frames in island_frame_features:
            count = int(frames.shape[0])
            self.forwarded_sizes.append(count)
            out.append(
                [
                    SplitDecision("continue", 0.0, 1.0, 0.0)
                    for _ in range(count)
                ]
            )
        return out


def test_build_chunks_streams_oversized_island_features(monkeypatch) -> None:
    cap = _MAX_ISLAND_BATCH_CANDIDATES
    proposal_count = cap * 2 + 17
    seen_feature_counts: list[int] = []

    def fake_split_features(proposals, **_kwargs):
        count = len(proposals)
        seen_feature_counts.append(count)
        return (
            np.zeros((count, 3, 5), dtype=np.float32),
            np.zeros((count, 2), dtype=np.float32),
        )

    monkeypatch.setattr(runtime_pipeline, "_split_features", fake_split_features)
    provider = FrameSequenceFeatureProvider(
        duration_s=40.0,
        frame_hop_s=0.02,
        ptm=np.zeros((2000, 2), dtype=np.float32),
        mfcc=np.zeros((2000, 1), dtype=np.float32),
        config=FrameSequenceFeatureConfig(max_ptm_dims=2),
    )
    segment = SpeechSegment(
        start=0.0,
        end=35.0,
        weak_cut_candidates=[
            {
                "time_s": 1.0 + index * 0.01,
                "score": 0.8,
                "prominence": 0.2,
                "speech_valley": 0.5,
                "strength": 1.0,
            }
            for index in range(proposal_count)
        ],
    )
    verifier = _ContinueIslandVerifier()

    chunks = runtime_pipeline.build_semantic_boundary_chunks(
        [segment],
        duration_s=40.0,
        speech_probabilities=np.ones(2000, dtype=np.float32),
        feature_provider=provider,
        outer_refiner=_OuterRefiner(),
        split_verifier=verifier,
        cut_refiner=_CutRefiner(),
    )

    assert len(chunks) == 1
    assert sum(seen_feature_counts) == proposal_count
    assert max(seen_feature_counts) <= cap
    assert max(verifier.forwarded_sizes) <= cap
