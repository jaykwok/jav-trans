from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import boundary.ja as boundary_ja

from boundary.ja.backend import (
    SpeechBoundaryJaBackend,
    SpeechBoundaryJaConfig,
    decode_binary_speech_island_segments,
    decode_semantic_speech_island_segments,
    decode_speech_island_segments,
    require_current_runtime_scorer,
)
from boundary.ja.model import (
    CANDIDATE_ISLAND_SCORER_V11_DECODER,
    CANDIDATE_ISLAND_SCORER_V11_LABELS,
    CANDIDATE_ISLAND_SCORER_V11_SCHEMA,
    SPEECH_ISLAND_SCORER_V8_SCHEMA,
    SPEECH_ISLAND_SCORER_V10_SCHEMA,
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


def test_binary_v10_decoder_uses_argmax_without_threshold_or_dilation() -> None:
    probabilities = np.asarray(
        [[0.6, 0.4], [0.49, 0.51], [0.2, 0.8], [0.7, 0.3]],
        dtype=np.float32,
    )
    result = decode_binary_speech_island_segments(
        class_probabilities=probabilities,
        candidate_probabilities=np.zeros(4, dtype=np.float32),
        duration_s=0.08,
        config=SpeechBoundaryJaConfig(
            threshold=0.99,
            frame_dilation_s=10.0,
            min_segment_s=10.0,
            frame_hop_s=0.02,
        ),
    )

    assert result.decision_mode == "binary_frame_argmax"
    assert result.speech_on_threshold is None
    assert result.speech_off_threshold is None
    assert result.raw_frames.tolist() == [0, 1, 1, 0]
    assert result.dilated_frames.tolist() == [0, 1, 1, 0]
    assert [(row.start, row.end) for row in result.segments] == [(0.02, 0.06)]


def test_17b_signature_exposes_only_pending_v11_contract_and_06b_is_retired() -> None:
    signature_17b = SpeechBoundaryJaBackend().signature()
    assert signature_17b["schema"] == CANDIDATE_ISLAND_SCORER_V11_SCHEMA
    assert signature_17b["status"] == "pending_binary_scorer_audit"
    assert signature_17b["labels"] == list(CANDIDATE_ISLAND_SCORER_V11_LABELS)
    assert signature_17b["decoder"] == CANDIDATE_ISLAND_SCORER_V11_DECODER
    assert signature_17b["decision_mode"] == "two_logit_softmax_argmax"
    assert signature_17b["runtime_threshold"] is None
    assert signature_17b["runtime_auxiliary_decoder"] == "disabled_ab_only"
    assert signature_17b["boundary_serialization_contract_id"] == (
        "boundary_acoustic_binary_v12"
    )
    assert "threshold" not in signature_17b
    assert "frame_dilation_s" not in signature_17b
    assert "candidate_source" not in signature_17b
    assert "split_nms_s" not in signature_17b

    config_06b = SpeechBoundaryJaConfig(
        ptm="jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"
    )
    with pytest.raises(RuntimeError, match="pending_binary_retrain"):
        SpeechBoundaryJaBackend(config=config_06b)


def test_v8_threshold_scorer_is_audit_only_for_current_runtime() -> None:
    class _V8:
        schema = SPEECH_ISLAND_SCORER_V8_SCHEMA

    with pytest.raises(RuntimeError, match="pending_binary_scorer_audit"):
        require_current_runtime_scorer(_V8())

    class _V9:
        schema = SPEECH_ISLAND_SCORER_SCHEMA

    with pytest.raises(RuntimeError, match="pending_binary_scorer_audit"):
        require_current_runtime_scorer(_V9())

    class _V10Candidate:
        schema = SPEECH_ISLAND_SCORER_V10_SCHEMA

    with pytest.raises(RuntimeError, match="pending_binary_scorer_audit"):
        require_current_runtime_scorer(_V10Candidate())


def test_boundary_ja_package_does_not_export_retired_v8_v9_training_surface() -> None:
    retired = {
        "SPEECH_ISLAND_SCORER_V8_SCHEMA",
        "SPEECH_ISLAND_SCORER_SCHEMA",
        "SpeechIslandTrainConfig",
        "SpeechIslandTrainMetrics",
        "TinyFrameClassifier",
        "score_semantic_speech_outputs",
        "score_speech_island_probabilities",
        "train_speech_island_scorer",
    }
    assert not (retired & set(boundary_ja.__all__))
    assert all(not hasattr(boundary_ja, name) for name in retired)


def test_retired_dual_head_and_threshold_metric_tools_are_absent() -> None:
    root = Path(__file__).resolve().parents[1]
    retired_paths = (
        root / "src" / "boundary" / "ja" / "dual_head.py",
        root / "tools" / "boundary" / "ja" / "gate_speech_proposal_dual_head.py",
        root / "tools" / "boundary" / "ja" / "speech_recall_metrics.py",
        root / "src" / "audio" / "audio_metrics.py",
        root / "tests" / "test_gate_speech_proposal_dual_head.py",
    )
    assert all(not path.exists() for path in retired_paths)


def test_boundary_explicit_cuda_request_never_falls_back_to_cpu(monkeypatch) -> None:
    import torch
    from boundary.ja.backend import _model_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        _model_device("cuda")
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        _model_device("auto")
    assert str(_model_device("cpu")) == "cpu"


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


def test_signature_blocks_proposal_until_candidate_scorer_is_promoted() -> None:
    from boundary.ja.backend import SpeechBoundaryJaBackend

    learned = SpeechBoundaryJaBackend(
        config=SpeechBoundaryJaConfig(proposal_checkpoint="proposal.pt")
    )
    signature = learned.signature()
    assert signature["proposal_status"] == (
        "blocked_until_candidate_scorer_promotion"
    )
    assert signature["scorer_checkpoint"] == ""
    assert "proposal_checkpoint" not in signature


def test_production_segment_fails_before_loading_retired_runtime(tmp_path) -> None:
    audio_path = tmp_path / "unused.wav"
    with pytest.raises(RuntimeError, match="pending_binary_scorer_audit"):
        SpeechBoundaryJaBackend().segment(str(audio_path))
