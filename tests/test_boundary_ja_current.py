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


def test_proposal_checkpoint_without_mapping_keeps_bootstrap(monkeypatch) -> None:
    from boundary.ja.backend import _proposal_checkpoint_from_env

    monkeypatch.delenv(
        "SPEECH_BOUNDARY_JA_PROPOSAL_CHECKPOINT_BY_REPO", raising=False
    )
    # Pin the registry default mapping to empty: with neither env nor registry
    # entry the repo stays on bootstrap candidates (split-v1 chains only).
    monkeypatch.setattr(
        "boundary.ja.backend.DEFAULT_SPEECH_BOUNDARY_PROPOSAL_CHECKPOINT_BY_REPO",
        {},
    )
    assert (
        _proposal_checkpoint_from_env("jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf")
        == ""
    )
    assert SpeechBoundaryJaConfig().proposal_checkpoint == ""


def test_proposal_checkpoint_env_maps_by_repo(monkeypatch, tmp_path) -> None:
    from boundary.ja.backend import _proposal_checkpoint_from_env

    repo_id = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    checkpoint = tmp_path / "boundary_proposal_scorer_v1.pt"
    checkpoint.write_bytes(b"")
    monkeypatch.setenv(
        "SPEECH_BOUNDARY_JA_PROPOSAL_CHECKPOINT_BY_REPO",
        f"{repo_id}={checkpoint}",
    )
    assert _proposal_checkpoint_from_env(repo_id) == str(checkpoint)


def test_signature_candidate_source_tracks_proposal_checkpoint() -> None:
    from boundary.ja.backend import SpeechBoundaryJaBackend

    bootstrap = SpeechBoundaryJaBackend(
        config=SpeechBoundaryJaConfig(proposal_checkpoint="")
    )
    assert bootstrap.signature()["candidate_source"] == "bootstrap_energy_ptm_mfcc_v1"

    learned = SpeechBoundaryJaBackend(
        config=SpeechBoundaryJaConfig(proposal_checkpoint="proposal.pt")
    )
    assert learned.signature()["candidate_source"] == "learned_boundary_proposal_v1"
    assert learned.signature()["split_decision"] == "external_semantic_split_model_v2"
