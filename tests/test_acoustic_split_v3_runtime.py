from __future__ import annotations

import numpy as np

from boundary.base import SpeechSegment
from boundary.outer_refiner_v2 import PairedOuterEdgePrediction
from audio.chunk_packer import PackedChunk
from boundary.runtime_pipeline import (
    apply_paired_inner_edges_after_cueqc,
    build_acoustic_split_v3_provisional_chunks,
)
from boundary.sequence_features import FrameSequenceFeatureConfig, FrameSequenceFeatureProvider
from boundary.split_model import SplitDecision


class _Planner:
    feature_config = {
        "left_context_s": 1.6,
        "right_context_s": 1.6,
        "gap_context_s": 0.3,
        "left_bins": 8,
        "gap_bins": 4,
        "right_bins": 8,
        "ptm_dim": 4,
        "ptm_projection": {
            "kind": "learned_linear_in_checkpoint",
            "input_dim": 4,
            "output_dim": 2,
        },
        "extra_context_scales": [],
    }

    def decide_islands(self, *, island_frame_features, island_scalar_features):
        assert island_frame_features[0].shape[0] == 5
        return [[
            SplitDecision("continue", 0.1, 0.8, 0.1),
            SplitDecision("cut", 0.7, 0.2, 0.1),
            SplitDecision("cut", 0.9, 0.05, 0.05),
            SplitDecision("continue", 0.1, 0.8, 0.1),
            SplitDecision("cut", 0.8, 0.1, 0.1),
        ]]


def _proposal(time_s: float) -> dict:
    return {
        "kind": "proposal",
        "time_s": time_s,
        "frame": int(round(time_s / 0.02)),
        "score": 0.8,
        "prominence": 0.2,
        "speech_valley": 0.5,
        "strength": 1.5,
    }


def test_v3_runtime_builds_events_without_duration_thresholds(monkeypatch) -> None:
    prediction = PairedOuterEdgePrediction(
        raw_start_s=0.0,
        raw_end_s=8.0,
        start_s=0.2,
        end_s=7.8,
        start_action="refined",
        end_action="refined",
        abstain_reason="",
        start_probabilities={},
        end_probabilities={},
        class_probabilities=np.zeros((1, 3), dtype=np.float32),
    )
    segment = SpeechSegment(
        start=0.0,
        end=8.0,
        weak_cut_candidates=[
            _proposal(1.0),
            _proposal(2.0),
            _proposal(2.2),
            _proposal(4.0),
            _proposal(6.0),
        ],
    )
    monkeypatch.setattr(
        "boundary.runtime_pipeline._refine_outer_edges",
        lambda *args, **kwargs: [(segment, 0.2, 7.8, prediction)],
    )
    provider = FrameSequenceFeatureProvider(
        duration_s=8.0,
        frame_hop_s=0.02,
        ptm=np.zeros((400, 4), dtype=np.float32),
        mfcc=np.zeros((400, 2), dtype=np.float32),
        config=FrameSequenceFeatureConfig(max_ptm_dims=4),
    )
    chunks = build_acoustic_split_v3_provisional_chunks(
        [segment],
        duration_s=8.0,
        speech_probabilities=np.ones(400, dtype=np.float32),
        feature_provider=provider,
        outer_refiner=object(),
        split_planner=_Planner(),
    )

    assert [(round(chunk.start, 1), round(chunk.end, 1)) for chunk in chunks] == [
        (0.2, 2.2),
        (2.2, 6.0),
        (6.0, 7.8),
    ]
    assert chunks[0].semantic_event_ids == ["island-0000-event-000"]
    assert chunks[1].semantic_event_ids == [
        "island-0000-event-000",
        "island-0000-event-001",
    ]
    assert chunks[0].boundary_pipeline_version == 10
    assert chunks[0].boundary_decision_source == "acoustic_candidate_event_runs_v1"
    assert chunks[0].display_start == chunks[0].acoustic_start
    assert chunks[0].primary_cut_candidates[0]["p_cut"] == 0.9
    assert all(
        "shared_absolute_timestamp" not in candidate
        for chunk in chunks
        for candidate in (chunk.primary_cut_candidates or [])
    )


def _inner_chunk(
    start: float,
    end: float,
    island_id: int,
    prediction: dict,
) -> PackedChunk:
    return PackedChunk(
        start=start,
        end=end,
        duration=end - start,
        speech_segments=[],
        split_reason="acoustic_split_v3",
        parent_chunk_id=0,
        island_id=island_id,
        island_count=2,
        source_abs_start=start,
        source_abs_end=end,
        acoustic_start=start,
        acoustic_end=end,
        acoustic_duration=end - start,
        display_start=start,
        display_end=end,
        display_duration=end - start,
        semantic_event_ids=["event-0"],
        inner_edge_prediction=prediction,
    )


def test_paired_inner_edges_remove_only_explicit_safe_gap() -> None:
    chunks = apply_paired_inner_edges_after_cueqc(
        [
            _inner_chunk(
                0.0,
                3.0,
                0,
                {"end_s": 2.7, "end_action": "refined"},
            ),
            _inner_chunk(
                3.0,
                6.0,
                1,
                {"start_s": 3.2, "start_action": "refined"},
            ),
        ]
    )
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0.0, 2.7), (3.2, 6.0)]
    assert chunks[0].removed_gap_spans == [
        {"event_ids": ["event-0"], "start": 2.7, "end": 3.2, "duration": 0.5}
    ]
    assert chunks[0].paired_inner_edges["action"] == "safe"


def test_paired_inner_edge_abstain_merges_without_rule_fallback() -> None:
    chunks = apply_paired_inner_edges_after_cueqc(
        [
            _inner_chunk(
                0.0,
                3.0,
                0,
                {"end_s": 2.8, "end_action": "abstain", "abstain_reason": "unsure"},
            ),
            _inner_chunk(
                3.0,
                6.0,
                1,
                {"start_s": 3.1, "start_action": "refined"},
            ),
        ]
    )
    assert len(chunks) == 1
    assert (chunks[0].start, chunks[0].end) == (0.0, 6.0)
    assert chunks[0].paired_inner_edges["action"] == "abstain_merge"
    assert chunks[0].removed_gap_spans in (None, [])
