from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_fallback_constants_default_off():
    import importlib
    import whisper.transcribe as transcribe_mod
    importlib.reload(transcribe_mod)
    assert transcribe_mod._ASR_TEMPERATURE_FALLBACK is False


def test_fallback_constants_on(monkeypatch):
    monkeypatch.setenv("ASR_TEMPERATURE_FALLBACK", "1")
    import importlib
    import whisper.transcribe as transcribe_mod
    importlib.reload(transcribe_mod)
    assert transcribe_mod._ASR_TEMPERATURE_FALLBACK is True


def test_fallback_temperatures_parsed(monkeypatch):
    monkeypatch.setenv("ASR_FALLBACK_TEMPERATURES", "0.2,0.4")
    import importlib
    import whisper.transcribe as transcribe_mod
    importlib.reload(transcribe_mod)
    assert transcribe_mod._ASR_FALLBACK_TEMPERATURES == [0.2, 0.4]


def test_check_logprob_drives_retry_decision():
    """verify that check_logprob_quality with reject verdict is what AA-4 checks."""
    from whisper.qc import check_logprob_quality
    reject = {"avg_logprob": -2.0, "no_speech_prob": 0.9, "compression_ratio": 1.0}
    ok = {"avg_logprob": -0.2, "no_speech_prob": 0.05, "compression_ratio": 1.0}
    assert check_logprob_quality(reject)["verdict"] == "reject"
    assert check_logprob_quality(ok)["verdict"] == "ok"


def test_model_backend_transcribe_texts_accepts_temperature():
    """WhisperModelBackend.transcribe_texts must declare temperature param."""
    import inspect
    from whisper.model_backend import WhisperModelBackend
    sig = inspect.signature(WhisperModelBackend.transcribe_texts)
    assert "temperature" in sig.parameters
    assert sig.parameters["temperature"].default == 0.0


def test_base_asr_backend_protocol_declares_prompt_and_temperature():
    import inspect
    from whisper.backends.base import BaseAsrBackend

    sig = inspect.signature(BaseAsrBackend.transcribe_texts)

    assert "supports_temperature" in BaseAsrBackend.__annotations__
    assert "initial_prompts" in sig.parameters
    assert "temperature" in sig.parameters
    assert sig.parameters["initial_prompts"].default is None
    assert sig.parameters["temperature"].default == 0.0


def test_temperature_support_is_explicit_on_backends():
    from whisper.local_backend import LocalAsrBackend, SubprocessAsrBackend
    from whisper.model_backend import WhisperModelBackend

    assert WhisperModelBackend.supports_temperature is True
    assert LocalAsrBackend.supports_temperature is False
    assert SubprocessAsrBackend.supports_temperature is False
