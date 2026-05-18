from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_check_logprob_drives_adaptive_drop_decision():
    """Reject verdicts are used for adaptive precision drops, not ASR retries."""
    from whisper.qc import check_logprob_quality
    reject = {"avg_logprob": -2.0, "no_speech_prob": 0.9, "compression_ratio": 1.0}
    ok = {"avg_logprob": -0.2, "no_speech_prob": 0.05, "compression_ratio": 1.0}
    assert check_logprob_quality(reject)["verdict"] == "reject"
    assert check_logprob_quality(ok)["verdict"] == "ok"


def test_model_backend_transcribe_texts_has_no_temperature_retry_knob():
    import inspect
    from whisper.model_backend import WhisperModelBackend

    sig = inspect.signature(WhisperModelBackend.transcribe_texts)

    assert "temperature" not in sig.parameters


def test_base_asr_backend_protocol_declares_prompt_without_temperature():
    import inspect
    from whisper.backends.base import BaseAsrBackend

    sig = inspect.signature(BaseAsrBackend.transcribe_texts)

    assert "initial_prompts" in sig.parameters
    assert "supports_temperature" not in BaseAsrBackend.__annotations__
    assert "temperature" not in sig.parameters
    assert sig.parameters["initial_prompts"].default is None
