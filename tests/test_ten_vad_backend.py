from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile

from whisper import timestamp_fallback


def _torchcodec_loadable() -> bool:
    try:
        import torchcodec  # noqa: F401
        return True
    except (ImportError, OSError, RuntimeError):
        return False


@pytest.mark.skipif(not _torchcodec_loadable(), reason="torchcodec/libavutil not available in this environment")
def test_detect_speech_spans_ten_vad_silent_wav(tmp_path):
    wav_path = tmp_path / "silent.wav"
    wavfile.write(wav_path, 16000, np.zeros(16000, dtype=np.int16))

    spans, error = timestamp_fallback._detect_speech_spans_ten_vad(str(wav_path))

    assert spans == []
    assert error == ""


def test_detect_speech_spans_ten_vad_skips_when_libcxx_missing(monkeypatch):
    monkeypatch.setattr(timestamp_fallback.platform, "system", lambda: "Linux")
    monkeypatch.setattr(timestamp_fallback.ctypes.util, "find_library", lambda _name: None)

    spans, error = timestamp_fallback._detect_speech_spans_ten_vad("audio.wav")

    assert spans == []
    assert error == "libc++.so.1 not available for TEN VAD"


def test_detect_speech_spans_ten_vad_does_not_check_libcxx_on_windows(monkeypatch):
    def fake_import(name, *_args, **_kwargs):
        if name == "ten_vad":
            raise ImportError("attempted ten_vad import")
        return original_import(name, *_args, **_kwargs)

    original_import = __import__
    monkeypatch.setattr(timestamp_fallback.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        timestamp_fallback.ctypes.util,
        "find_library",
        lambda _name: (_ for _ in ()).throw(AssertionError("libc++ should not be checked on Windows")),
    )
    monkeypatch.setattr("builtins.__import__", fake_import)

    spans, error = timestamp_fallback._detect_speech_spans_ten_vad("audio.wav")

    assert spans == []
    assert error == "attempted ten_vad import"


def test_detect_speech_spans_uses_ten_vad_when_enabled(monkeypatch):
    calls = {"ten": 0, "silero": 0}

    def fake_ten(audio_path):
        calls["ten"] += 1
        assert audio_path == "audio.wav"
        return [(0.0, 0.5)], ""

    def fail_silero():
        calls["silero"] += 1
        raise AssertionError("silero should not be used when TEN succeeds")

    monkeypatch.setenv("TEN_VAD_BACKEND", "1")
    monkeypatch.setattr(timestamp_fallback, "_TEN_VAD_ENABLED", True)
    monkeypatch.setattr(timestamp_fallback, "_detect_speech_spans_ten_vad", fake_ten)
    monkeypatch.setattr(timestamp_fallback, "_load_silero_vad", fail_silero)

    spans, error = timestamp_fallback.detect_speech_spans("audio.wav")

    assert spans == [(0.0, 0.5)]
    assert error == ""
    assert calls == {"ten": 1, "silero": 0}


def test_detect_speech_spans_falls_back_to_silero_when_ten_errors(monkeypatch):
    class FakeWaveform:
        def numel(self):
            return 16000

    def fake_get_speech_timestamps(*_args, **_kwargs):
        return [{"start": 0.2, "end": 0.6}]

    monkeypatch.setenv("TEN_VAD_BACKEND", "1")
    monkeypatch.setattr(timestamp_fallback, "_TEN_VAD_ENABLED", True)
    monkeypatch.setattr(
        timestamp_fallback,
        "_detect_speech_spans_ten_vad",
        lambda _audio_path: ([], "ten dll missing"),
    )
    monkeypatch.setattr(
        timestamp_fallback,
        "_load_audio_for_vad",
        lambda _audio_path: (FakeWaveform(), 16000),
    )
    monkeypatch.setattr(
        timestamp_fallback,
        "_load_silero_vad",
        lambda: (object(), fake_get_speech_timestamps),
    )

    spans, error = timestamp_fallback.detect_speech_spans("audio.wav")

    assert spans == [(0.2, 0.6)]
    assert error == "ten_vad: ten dll missing; fallback=silero_vad"


def test_detect_speech_spans_keeps_empty_ten_result_without_silero(monkeypatch):
    def fail_silero():
        raise AssertionError("Silero should not be used when TEN returns a valid empty result")

    monkeypatch.setenv("TEN_VAD_BACKEND", "1")
    monkeypatch.setattr(timestamp_fallback, "_TEN_VAD_ENABLED", True)
    monkeypatch.setattr(
        timestamp_fallback,
        "_detect_speech_spans_ten_vad",
        lambda _audio_path: ([], ""),
    )
    monkeypatch.setattr(timestamp_fallback, "_load_silero_vad", fail_silero)

    spans, error = timestamp_fallback.detect_speech_spans("audio.wav")

    assert spans == []
    assert error == ""


def test_detect_speech_spans_reports_ten_and_silero_errors(monkeypatch):
    monkeypatch.setenv("TEN_VAD_BACKEND", "1")
    monkeypatch.setattr(timestamp_fallback, "_TEN_VAD_ENABLED", True)
    monkeypatch.setattr(
        timestamp_fallback,
        "_detect_speech_spans_ten_vad",
        lambda _audio_path: ([], "ten failed"),
    )
    monkeypatch.setattr(
        timestamp_fallback,
        "_detect_speech_spans_silero_vad",
        lambda _audio_path: ([], "silero failed"),
    )

    spans, error = timestamp_fallback.detect_speech_spans("audio.wav")

    assert spans == []
    assert error == "ten_vad: ten failed; silero_vad: silero failed"


def test_detect_speech_spans_uses_silero_when_ten_disabled(monkeypatch):
    class FakeWaveform:
        def numel(self):
            return 16000

    def fake_get_speech_timestamps(*_args, **_kwargs):
        return [{"start": 0.1, "end": 0.4}]

    def fake_silero():
        return object(), fake_get_speech_timestamps

    def fail_ten(_audio_path):
        raise AssertionError("TEN should not be used when disabled")

    monkeypatch.setenv("TEN_VAD_BACKEND", "0")
    monkeypatch.setattr(timestamp_fallback, "_TEN_VAD_ENABLED", False)
    monkeypatch.setattr(timestamp_fallback, "_detect_speech_spans_ten_vad", fail_ten)
    monkeypatch.setattr(
        timestamp_fallback,
        "_load_audio_for_vad",
        lambda _audio_path: (FakeWaveform(), 16000),
    )
    monkeypatch.setattr(timestamp_fallback, "_load_silero_vad", fake_silero)

    spans, error = timestamp_fallback.detect_speech_spans("audio.wav")

    assert spans == [(0.1, 0.4)]
    assert error == ""
