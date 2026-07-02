from __future__ import annotations

import numpy as np

from boundary.ja.backend import SpeechBoundaryJaConfig, decode_speech_island_segments
from boundary.ja.model import (
    SPEECH_ISLAND_SCORER_DECODER,
    SPEECH_ISLAND_SCORER_OUTPUT_DIM,
    SPEECH_ISLAND_SCORER_OUTPUT_HEADS,
    SPEECH_ISLAND_SCORER_SCHEMA,
)


def test_speech_island_scorer_has_one_speech_head() -> None:
    assert SPEECH_ISLAND_SCORER_SCHEMA.endswith("speech_island_scorer_v8")
    assert SPEECH_ISLAND_SCORER_OUTPUT_DIM == 1
    assert SPEECH_ISLAND_SCORER_OUTPUT_HEADS == ("speech_prob",)
    assert SPEECH_ISLAND_SCORER_DECODER == "speech_hysteresis_islands_v1"


def test_decoder_attaches_proposals_without_splitting_speech_island() -> None:
    speech = np.full(120, 0.9, dtype=np.float32)
    candidate = np.full(120, 0.05, dtype=np.float32)
    candidate[58:63] = 0.99
    result = decode_speech_island_segments(
        speech_probabilities=speech,
        candidate_probabilities=candidate,
        duration_s=2.4,
        config=SpeechBoundaryJaConfig(
            threshold=0.15,
            frame_dilation_s=0.0,
            frame_hop_s=0.02,
            min_segment_s=0.05,
        ),
    )

    assert len(result.segments) == 1
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 2.4
    assert result.segments[0].primary_cut_candidates == []
    assert result.segments[0].weak_cut_candidates


def test_decoder_uses_high_recall_default_threshold() -> None:
    config = SpeechBoundaryJaConfig()
    assert config.threshold == 0.15
    assert config.sequence_feature_max_ptm_dims == 128
