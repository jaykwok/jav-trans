from __future__ import annotations

import importlib

from helpers import ASR_06B_BACKEND, ASR_17B_BACKEND


def _checkpoint_name(
    monkeypatch,
    *,
    asr_backend: str,
) -> str:
    monkeypatch.setenv("ASR_BACKEND", asr_backend)
    monkeypatch.setenv("ASR_WORKER_MODE", "inproc")

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

