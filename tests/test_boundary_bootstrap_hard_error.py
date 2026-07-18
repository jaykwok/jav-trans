from __future__ import annotations

from pathlib import Path

import pytest

from asr.pipeline import _require_learned_split_candidates
from boundary.ja.backend import _proposal_checkpoint_from_env

REPO_17B = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
REPO_06B = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"


def test_island_split_with_bootstrap_candidates_raises() -> None:
    with pytest.raises(RuntimeError, match="bootstrap"):
        _require_learned_split_candidates({})
    with pytest.raises(RuntimeError, match="bootstrap"):
        _require_learned_split_candidates({"proposal_checkpoint": ""})


def test_island_split_with_learned_candidates_passes() -> None:
    _require_learned_split_candidates(
        {"proposal_checkpoint": {"path": "x.pt", "sha256": "abc"}},
    )


def test_proposal_checkpoint_empty_default_keeps_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_PROPOSAL_CHECKPOINT_BY_REPO", raising=False)
    monkeypatch.setattr(
        "boundary.ja.backend.DEFAULT_SPEECH_BOUNDARY_PROPOSAL_CHECKPOINT_BY_REPO",
        {},
    )
    assert _proposal_checkpoint_from_env(REPO_17B) == ""


def test_proposal_checkpoint_default_mapping_resolves_per_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "boundary_proposal_scorer_v1.pt"
    checkpoint.write_bytes(b"stub")
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_PROPOSAL_CHECKPOINT_BY_REPO", raising=False)
    monkeypatch.setattr(
        "boundary.ja.backend.DEFAULT_SPEECH_BOUNDARY_PROPOSAL_CHECKPOINT_BY_REPO",
        {REPO_17B: str(checkpoint)},
    )
    resolved = _proposal_checkpoint_from_env(REPO_17B)
    assert Path(resolved) == checkpoint
    # A repo without a promoted proposer remains unavailable to Split v4.
    assert _proposal_checkpoint_from_env(REPO_06B) == ""


def test_proposal_checkpoint_env_mapping_is_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "proposal_06b.pt"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setenv(
        "SPEECH_BOUNDARY_JA_PROPOSAL_CHECKPOINT_BY_REPO",
        f"{REPO_06B}={checkpoint}",
    )
    with pytest.raises(RuntimeError, match="no checkpoint for ASR repo"):
        _proposal_checkpoint_from_env(REPO_17B)
