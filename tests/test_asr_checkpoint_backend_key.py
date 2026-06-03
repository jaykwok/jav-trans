from __future__ import annotations

import importlib

from helpers import ASR_06B_BACKEND, ASR_17B_BACKEND


def _checkpoint_name(
    monkeypatch,
    *,
    asr_backend: str,
    alignment_mode: str = "forced",
    asr_context: str = "",
) -> str:
    monkeypatch.setenv("ASR_BACKEND", asr_backend)
    monkeypatch.setenv("ALIGNMENT_TIMESTAMP_MODE", alignment_mode)
    monkeypatch.setenv("ASR_WORKER_MODE", "inproc")
    monkeypatch.setenv("ASR_CONTEXT", asr_context)

    from asr import pipeline as asr
    asr = importlib.reload(asr)
    asr._LAST_VAD_SIGNATURE = {"backend": "fusionvad_ja", "threshold": 0.2}
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


def test_checkpoint_key_changes_with_alignment_timestamp_mode(monkeypatch):
    forced_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_06B_BACKEND,
        alignment_mode="forced",
    )
    native_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_06B_BACKEND,
        alignment_mode="native",
    )

    assert forced_key != native_key


def test_checkpoint_key_changes_with_prompt_token_budget(monkeypatch):
    default_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_06B_BACKEND,
    )

    monkeypatch.setenv("ASR_MIN_EFFECTIVE_NEW_TOKENS", "96")
    tuned_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_06B_BACKEND,
    )

    assert default_key != tuned_key


def test_checkpoint_key_changes_with_prompt_char_cap(monkeypatch):
    default_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_06B_BACKEND,
    )

    monkeypatch.setenv("ASR_INITIAL_PROMPT_MAX_CHARS", "160")
    tuned_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_06B_BACKEND,
    )

    assert default_key != tuned_key


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


def test_checkpoint_key_changes_with_adaptive_precision_threshold(monkeypatch):
    default_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_06B_BACKEND,
    )

    monkeypatch.setenv("ASR_QC_ADAPTIVE_MIN_LOGPROB", "-0.90")
    tuned_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_06B_BACKEND,
    )

    assert default_key != tuned_key


def test_checkpoint_key_changes_with_asr_context(monkeypatch):
    actor_a_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_17B_BACKEND,
        asr_context="actor-a",
    )
    actor_b_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_17B_BACKEND,
        asr_context="actor-b",
    )

    assert actor_a_key != actor_b_key


def test_checkpoint_key_changes_with_qwen_generation_inputs(monkeypatch):
    default_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_17B_BACKEND,
    )

    monkeypatch.setenv("ASR_LANGUAGE", "Japanese")
    monkeypatch.setenv("TRANSCRIPTION_MAX_NEW_TOKENS", "256")
    tuned_key = _checkpoint_name(
        monkeypatch,
        asr_backend=ASR_17B_BACKEND,
    )

    assert default_key != tuned_key

