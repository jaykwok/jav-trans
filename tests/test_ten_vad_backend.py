from __future__ import annotations

import numpy as np
from scipy.io import wavfile

from whisper import timestamp_fallback
def test_detect_speech_spans_ten_vad_silent_wav(tmp_path):
    wav_path = tmp_path / "silent.wav"
    wavfile.write(wav_path, 16000, np.zeros(16000, dtype=np.int16))

    spans, error = timestamp_fallback._detect_speech_spans_ten_vad(str(wav_path))

    assert spans == []
    assert error == ""


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

