from __future__ import annotations

import importlib


def _checkpoint_name(
    monkeypatch,
    *,
    asr_backend: str,
    whisper_mode: str,
    asr_context: str = "",
) -> str:
    monkeypatch.setenv("ASR_BACKEND", asr_backend)
    monkeypatch.setenv("WHISPER_TIMESTAMP_MODE", whisper_mode)
    monkeypatch.setenv("ASR_WORKER_MODE", "inproc")
    monkeypatch.setenv("ASR_CONTEXT", asr_context)

    from whisper import pipeline as asr
    asr = importlib.reload(asr)
    asr._LAST_VAD_SIGNATURE = {"backend": "whisperseg_v1", "threshold": 0.35}
    return asr._get_asr_checkpoint_path("sample.wav").name


def test_checkpoint_key_changes_between_qwen_and_anime_backend(monkeypatch):
    qwen_key = _checkpoint_name(
        monkeypatch,
        asr_backend="qwen3-asr-1.7b",
        whisper_mode="forced",
    )
    anime_key = _checkpoint_name(
        monkeypatch,
        asr_backend="anime-whisper",
        whisper_mode="forced",
    )

    assert qwen_key != anime_key


def test_checkpoint_key_changes_with_anime_timestamp_mode(monkeypatch):
    forced_key = _checkpoint_name(
        monkeypatch,
        asr_backend="anime-whisper",
        whisper_mode="forced",
    )
    vad_key = _checkpoint_name(
        monkeypatch,
        asr_backend="anime-whisper",
        whisper_mode="vad_ratio",
    )

    assert forced_key != vad_key


def test_checkpoint_key_changes_with_whisper_generation_kwargs(monkeypatch):
    default_key = _checkpoint_name(
        monkeypatch,
        asr_backend="anime-whisper",
        whisper_mode="forced",
    )

    monkeypatch.setenv("WHISPER_BEAMS", "5")
    tuned_key = _checkpoint_name(
        monkeypatch,
        asr_backend="anime-whisper",
        whisper_mode="forced",
    )

    assert default_key != tuned_key


def test_checkpoint_key_changes_with_whisper_budget_inputs(monkeypatch):
    default_key = _checkpoint_name(
        monkeypatch,
        asr_backend="anime-whisper",
        whisper_mode="forced",
    )

    monkeypatch.setenv("ASR_MIN_EFFECTIVE_NEW_TOKENS", "96")
    tuned_key = _checkpoint_name(
        monkeypatch,
        asr_backend="anime-whisper",
        whisper_mode="forced",
    )

    assert default_key != tuned_key


def test_checkpoint_key_changes_with_whisper_prompt_char_cap(monkeypatch):
    default_key = _checkpoint_name(
        monkeypatch,
        asr_backend="anime-whisper",
        whisper_mode="forced",
    )

    monkeypatch.setenv("ASR_INITIAL_PROMPT_MAX_CHARS", "160")
    tuned_key = _checkpoint_name(
        monkeypatch,
        asr_backend="anime-whisper",
        whisper_mode="forced",
    )

    assert default_key != tuned_key


def test_checkpoint_key_changes_with_asr_context(monkeypatch):
    actor_a_key = _checkpoint_name(
        monkeypatch,
        asr_backend="qwen3-asr-1.7b",
        whisper_mode="forced",
        asr_context="actor-a",
    )
    actor_b_key = _checkpoint_name(
        monkeypatch,
        asr_backend="qwen3-asr-1.7b",
        whisper_mode="forced",
        asr_context="actor-b",
    )

    assert actor_a_key != actor_b_key


def test_checkpoint_key_changes_with_qwen_generation_inputs(monkeypatch):
    default_key = _checkpoint_name(
        monkeypatch,
        asr_backend="qwen3-asr-1.7b",
        whisper_mode="forced",
    )

    monkeypatch.setenv("ASR_LANGUAGE", "Japanese")
    monkeypatch.setenv("TRANSCRIPTION_MAX_NEW_TOKENS", "256")
    tuned_key = _checkpoint_name(
        monkeypatch,
        asr_backend="qwen3-asr-1.7b",
        whisper_mode="forced",
    )

    assert default_key != tuned_key

