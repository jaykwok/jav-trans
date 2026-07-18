from __future__ import annotations

import numpy as np

from boundary.base import SpeechSegment
from boundary.binary_edge_refiner import BinaryEdgePrediction
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from audio.chunk_packer import PackedChunk
from boundary.runtime_pipeline import (
    apply_binary_inner_edges_after_cueqc,
    build_acoustic_split_v4_provisional_chunks,
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
        "extra_context_scales": [],
    }

    def signature(self):
        return {
            "schema": "semantic_split_model_v4",
            "runtime_adapter": "acoustic_candidate_binary_event_runs_v2",
        }

    def decide_islands(self, *, island_frame_features, island_scalar_features):
        assert island_frame_features[0].shape[0] == 5
        return [[
            SplitDecision("continue", 0.1, 0.9, 0.0),
            SplitDecision("cut", 0.7, 0.3, 0.0),
            SplitDecision("cut", 0.9, 0.1, 0.0),
            SplitDecision("continue", 0.1, 0.9, 0.0),
            SplitDecision("cut", 0.8, 0.2, 0.0),
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


def test_v4_runtime_builds_events_without_duration_thresholds(monkeypatch) -> None:
    prediction = BinaryEdgePrediction(
        raw_start_s=0.0,
        raw_end_s=8.0,
        start_s=0.2,
        end_s=7.8,
        start_action="refined",
        end_action="refined",
        abstain_reason="",
        start_probabilities={},
        end_probabilities={},
        class_probabilities=np.zeros((1, 2), dtype=np.float32),
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
    chunks = build_acoustic_split_v4_provisional_chunks(
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
    assert chunks[0].boundary_contract_id == ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    assert chunks[0].boundary_decision_source == "acoustic_candidate_binary_event_runs_v2"
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
        split_reason="acoustic_split_v4",
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
        boundary_contract_id=ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        semantic_event_ids=["event-0"],
        inner_edge_prediction=prediction,
    )


def test_binary_inner_v2_applies_acoustic_core_and_drops_all_background() -> None:
    chunks = apply_binary_inner_edges_after_cueqc([
        _inner_chunk(
            0.0, 3.0, 0,
            {
                "schema": "binary_acoustic_inner_edges_v2_prediction",
                "action": "refined", "start_s": 0.4, "end_s": 2.6,
            },
        ),
        _inner_chunk(
            3.0, 6.0, 1,
            {
                "schema": "binary_acoustic_inner_edges_v2_prediction",
                "action": "drop", "start_s": 3.0, "end_s": 6.0,
            },
        ),
    ])
    assert len(chunks) == 1
    assert (chunks[0].start, chunks[0].end) == (0.4, 2.6)
    assert (chunks[0].acoustic_start, chunks[0].acoustic_end) == (0.4, 2.6)
    assert chunks[0].paired_inner_edges["action"] == "binary_core"
