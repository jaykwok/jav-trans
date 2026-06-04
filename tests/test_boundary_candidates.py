from __future__ import annotations

import pytest

from boundary.candidates import CandidateExtractionConfig, extract_boundary_candidates
from boundary.features import make_feature_bundle


def test_extract_boundary_candidates_from_cut_scores():
    features = make_feature_bundle(
        frame_hop_s=1.0,
        cut_scores=[0.05] * 5 + [0.98] * 2 + [0.05] * 5,
    )

    candidates = extract_boundary_candidates(
        start_s=0.0,
        end_s=12.0,
        features=features,
        config=CandidateExtractionConfig(min_chunk_s=3.0, cut_score_threshold=0.94),
    )

    assert len(candidates) == 1
    assert candidates[0].time_s == pytest.approx(6.0)
    assert candidates[0].source == "cut"
    assert candidates[0].score == pytest.approx(0.98)


def test_extract_boundary_candidates_from_vad_valley_scores():
    features = make_feature_bundle(
        frame_hop_s=1.0,
        speech_scores=[0.9] * 5 + [0.05] * 2 + [0.9] * 5,
    )

    candidates = extract_boundary_candidates(
        start_s=0.0,
        end_s=12.0,
        features=features,
        config=CandidateExtractionConfig(min_chunk_s=3.0, valley_score_threshold=0.10),
    )

    assert len(candidates) == 1
    assert candidates[0].time_s == pytest.approx(6.0)
    assert candidates[0].source == "valley"
