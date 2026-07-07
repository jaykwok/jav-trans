from __future__ import annotations

import importlib

import numpy as np

from helpers import ASR_06B_BACKEND, ASR_17B_BACKEND


def _checkpoint_name(
    monkeypatch,
    *,
    asr_backend: str,
) -> str:
    monkeypatch.setenv("ASR_BACKEND", asr_backend)

    from asr import pipeline as asr
    asr = importlib.reload(asr)
    asr._LAST_BOUNDARY_SIGNATURE = {"backend": "speech_boundary_ja", "threshold": 0.2}
    return asr._get_asr_checkpoint_path("sample.wav").name


def test_checkpoint_key_changes_between_qwen_repo_backends(monkeypatch):
    default_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_06B_BACKEND,
    )
    large_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_17B_BACKEND,
    )

    assert default_key != large_key


def test_checkpoint_key_changes_with_asr_model_id_override(monkeypatch):
    default_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_06B_BACKEND,
    )

    monkeypatch.setenv("ASR_MODEL_ID", ASR_17B_BACKEND)
    tuned_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_06B_BACKEND,
    )

    assert default_key != tuned_key


def test_checkpoint_key_ignores_asr_after_cueqc_drop_threshold(monkeypatch):
    default_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_06B_BACKEND,
    )

    monkeypatch.setenv("CUEQC_DROP_THRESHOLD", "0.90")
    tuned_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_06B_BACKEND,
    )

    assert default_key == tuned_key


def test_checkpoint_key_ignores_removed_asr_context_env(monkeypatch):
    monkeypatch.setenv("ASR_CONTEXT", "actor-a")
    actor_a_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_17B_BACKEND,
    )
    monkeypatch.setenv("ASR_CONTEXT", "actor-b")
    actor_b_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_17B_BACKEND,
    )

    assert actor_a_key == actor_b_key


def test_checkpoint_key_changes_with_qwen_generation_inputs(monkeypatch):
    default_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_17B_BACKEND,
    )

    monkeypatch.setenv("ASR_LANGUAGE", "Japanese")
    monkeypatch.setenv("ASR_MAX_NEW_TOKENS", "256")
    tuned_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_17B_BACKEND,
    )

    assert default_key != tuned_key


def test_checkpoint_key_accepts_numpy_boundary_signature(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", ASR_17B_BACKEND)

    from asr import pipeline as asr
    asr = importlib.reload(asr)
    asr._set_last_boundary_signature(
        {
            "backend": "speech_boundary_ja",
            "threshold": np.float32(0.2),
            "diagnostic_scores": np.asarray([0.1, 0.2], dtype=np.float32),
        }
    )

    checkpoint_name = asr._get_asr_checkpoint_path("sample.wav").name

    assert checkpoint_name.startswith("asr_checkpoint_")
    assert checkpoint_name.endswith(".json")


def test_checkpoint_excludes_quarantined_results():
    from asr.checkpoint import _checkpointable_text_results

    text_results = {
        0: {"text": "ok", "log": [], "asr_generation": {"policy": "ok"}},
        1: {
            "text": "",
            "log": ["QUARANTINED: kind=timeout, respawn_count=3"],
            "asr_generation": {"policy": "quarantined_result"},
        },
        2: {"text": "ok2", "log": ["TIMEOUT: 180s"], "asr_generation": {}},
    }
    filtered = _checkpointable_text_results(text_results)
    # Quarantined (1) and timed-out (2) are excluded; only the clean result (0)
    # persists, so quarantined chunks get re-transcribed on resume instead of
    # being silently restored as empty completed results.
    assert set(filtered.keys()) == {0}
