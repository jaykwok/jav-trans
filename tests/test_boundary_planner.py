from __future__ import annotations

import pytest

from boundary.features import make_feature_bundle
from boundary.planner import BoundaryPlannerConfig, plan_boundary_chunks
from boundary.base import SpeechSegment


def test_boundary_planner_uses_candidate_split_before_packing():
    features = make_feature_bundle(
        frame_hop_s=1.0,
        cut_scores=[0.05] * 5 + [0.98] * 2 + [0.05] * 5,
    )

    chunks = plan_boundary_chunks(
        [SpeechSegment(0.0, 12.0)],
        features=features,
        config=BoundaryPlannerConfig(
            frame_hop_s=1.0,
            max_chunk_s=30.0,
            target_chunk_s=5.0,
            min_chunk_s=3.0,
            target_padding_s=1.0,
        ),
    )

    assert len(chunks) == 2
    assert chunks[0].islands[0].end == pytest.approx(6.0)
    assert chunks[1].islands[0].start == pytest.approx(6.0)
    assert chunks[0].split_reason == "cut_candidate"


def test_boundary_planner_validates_config():
    features = make_feature_bundle(frame_hop_s=1.0)

    with pytest.raises(ValueError, match="max_chunk_s"):
        plan_boundary_chunks(
            [SpeechSegment(0.0, 1.0)],
            features=features,
            config=BoundaryPlannerConfig(max_chunk_s=0.0),
        )
