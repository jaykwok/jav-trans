from __future__ import annotations

import pytest

from boundary.candidates import (
    CandidateExtractionConfig,
    extract_boundary_candidates,
    soft_candidate_near_target,
)
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


def test_soft_candidate_near_target_uses_best_soft_valley():
    features = make_feature_bundle(
        frame_hop_s=1.0,
        speech_scores=[0.9] * 7 + [0.6, 0.4, 0.7, 0.8] + [0.9] * 5,
    )

    candidate = soft_candidate_near_target(
        start_s=0.0,
        end_s=16.0,
        target_s=8.0,
        features=features,
        config=CandidateExtractionConfig(
            min_chunk_s=3.0,
            soft_candidate_search_radius_s=2.0,
        ),
    )

    assert candidate is not None
    assert candidate.time_s == pytest.approx(8.5)
    assert candidate.source == "soft_valley"
    assert candidate.score == pytest.approx(0.6)


def test_soft_candidate_near_target_prefers_target_when_scores_tie():
    features = make_feature_bundle(
        frame_hop_s=1.0,
        speech_scores=[0.8] * 20,
    )

    candidate = soft_candidate_near_target(
        start_s=0.0,
        end_s=20.0,
        target_s=9.0,
        features=features,
        config=CandidateExtractionConfig(
            min_chunk_s=3.0,
            soft_candidate_search_radius_s=2.0,
        ),
    )

    assert candidate is not None
    assert candidate.time_s == pytest.approx(8.5)
    assert candidate.source == "soft_valley"
