from __future__ import annotations

import numpy as np
import pytest

from boundary.ja.backend import (
    SpeechBoundaryJaBackend,
    SpeechBoundaryJaConfig,
    decode_semantic_speech_island_segments,
    decode_speech_island_segments,
)
from boundary.ja.model import (
    SPEECH_ISLAND_SCORER_V8_SCHEMA,
    SPEECH_ISLAND_MEMBERSHIP_LABELS,
    SPEECH_ISLAND_SCORER_DECODER,
    SPEECH_ISLAND_SCORER_LABELS,
    SPEECH_ISLAND_SCORER_OUTPUT_DIM,
    SPEECH_ISLAND_SCORER_OUTPUT_HEADS,
    SPEECH_ISLAND_SCORER_SCHEMA,
)


def test_17b_semantic_speech_scorer_separates_content_and_membership() -> None:
    assert SPEECH_ISLAND_SCORER_SCHEMA == "semantic_speech_scorer_v9"
    assert SPEECH_ISLAND_SCORER_OUTPUT_DIM == 6
    assert SPEECH_ISLAND_SCORER_OUTPUT_HEADS == (
        "content.discardable",
        "content.semantic_target",
        "content.unsure",
        "membership.outside",
        "membership.inside",
        "membership.unsure",
    )
    assert SPEECH_ISLAND_SCORER_LABELS == (
        "discardable",
        "semantic_target",
        "unsure",
    )
    assert SPEECH_ISLAND_MEMBERSHIP_LABELS == ("outside", "inside", "unsure")
    assert SPEECH_ISLAND_SCORER_DECODER == "argmax_source_membership_islands_v1"
    assert SPEECH_ISLAND_SCORER_V8_SCHEMA.endswith("speech_island_scorer_v8")


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


def test_semantic_decoder_does_not_split_on_internal_discardable_content() -> None:
    content_probabilities = np.asarray(
        [
            [0.1, 0.8, 0.1],
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float32,
    )
    membership_probabilities = np.asarray(
        [[0.05, 0.9, 0.05]] * 3,
        dtype=np.float32,
    )
    result = decode_semantic_speech_island_segments(
        content_class_probabilities=content_probabilities,
        membership_class_probabilities=membership_probabilities,
        candidate_probabilities=np.zeros(3, dtype=np.float32),
        duration_s=0.06,
        config=SpeechBoundaryJaConfig(frame_hop_s=0.02),
    )

    assert result.decision_mode == "argmax_source_membership"
    assert result.speech_on_threshold is None
    assert result.raw_frames.tolist() == [1, 1, 1]
    assert result.dilated_frames.tolist() == [1, 1, 1]
    assert [(item.start, item.end) for item in result.segments] == [(0.0, 0.06)]


def test_17b_signature_has_no_fixed_speech_threshold_and_06b_is_retired() -> None:
    config_17b = SpeechBoundaryJaConfig(
        scorer_checkpoint=(
            "semantic_speech_scorer_v9."
            "jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt"
        )
    )
    signature_17b = SpeechBoundaryJaBackend(config=config_17b).signature()
    assert signature_17b["speech_threshold_mode"] == "argmax_source_membership"
    assert "threshold" not in signature_17b
    assert "frame_dilation_s" not in signature_17b

    config_06b = SpeechBoundaryJaConfig(
        ptm="jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"
    )
    with pytest.raises(RuntimeError, match="pending_binary_retrain"):
        SpeechBoundaryJaBackend(config=config_06b)


def test_proposal_checkpoint_without_mapping_keeps_bootstrap(monkeypatch) -> None:
    from boundary.ja.backend import _proposal_checkpoint_from_env

    monkeypatch.delenv(
        "SPEECH_BOUNDARY_JA_PROPOSAL_CHECKPOINT_BY_REPO", raising=False
    )
    # Pin the registry default mapping to empty for offline bootstrap coverage.
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
    assert learned.signature()["split_decision"] == (
        "external_repo_bound_semantic_split_model"
    )
