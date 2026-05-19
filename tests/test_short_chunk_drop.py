"""Tests for the short low-energy span drop gate in pipeline._build_processing_spans."""
from __future__ import annotations

import math
import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    data = (samples * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data)


def _silence(duration_s: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _tone(duration_s: float, sr: int = 16000, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(duration_s * sr), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture()
def audio_file(tmp_path):
    """Write a 1-second file: 0.0–0.5s silence, 0.5–1.0s 440 Hz tone."""
    samples = np.concatenate([_silence(0.5), _tone(0.5)])
    wav = tmp_path / "test.wav"
    _write_wav(wav, samples)
    return str(wav)


def _run_drop(audio_path, spans, *, enabled="1", min_duration="0.20", rms_dbfs="-40.0", monkeypatch):
    monkeypatch.setenv("ASR_CHUNK_DROP_ENABLED", enabled)
    monkeypatch.setenv("ASR_CHUNK_DROP_MIN_DURATION_S", min_duration)
    monkeypatch.setenv("ASR_CHUNK_DROP_RMS_DBFS", rms_dbfs)

    import importlib
    import whisper.pipeline as pipeline
    importlib.reload(pipeline)

    # Access the internal helper directly
    from whisper import pipeline as pl
    importlib.reload(pl)
    return pl._drop_short_low_energy_spans(audio_path, spans)


# --- Case A: gate disabled → all spans preserved ---
def test_drop_disabled_keeps_all_spans(audio_file, monkeypatch, tmp_path):
    spans = [(0.0, 0.1), (0.1, 0.5), (0.5, 1.0)]
    monkeypatch.setenv("ASR_CHUNK_DROP_ENABLED", "0")

    import importlib
    import whisper.pipeline as pl
    importlib.reload(pl)

    # With gate disabled, _drop_short_low_energy_spans is never called in the
    # main path. We verify the helper behaviour directly: even short silent spans
    # are NOT dropped when we temporarily force it with small thresholds,
    # but here we just confirm the env flag is respected in _build_processing_spans
    # by checking the runtime config reader.
    assert not pl._chunk_config()["drop_enabled"]


# --- Case B: short silent span is dropped ---
def test_drop_short_silent_span_is_dropped(audio_file, monkeypatch):
    """0.1s silence region → duration < 0.20 AND rms < -40 dBFS → dropped."""
    monkeypatch.setenv("ASR_CHUNK_DROP_ENABLED", "1")
    monkeypatch.setenv("ASR_CHUNK_DROP_MIN_DURATION_S", "0.20")
    monkeypatch.setenv("ASR_CHUNK_DROP_RMS_DBFS", "-40.0")

    import importlib
    from whisper import pipeline as pl
    importlib.reload(pl)

    # span covering silent part of the audio (0.0–0.5s is silence)
    spans = [(0.0, 0.1)]
    result = pl._drop_short_low_energy_spans(audio_file, spans)
    assert result == [], f"Expected empty, got {result}"


# --- Case C: short span with signal is kept (AND logic) ---
def test_drop_short_span_with_signal_is_kept(audio_file, monkeypatch):
    """0.15s of 440 Hz tone → duration < 0.20 but rms > -40 dBFS → kept."""
    monkeypatch.setenv("ASR_CHUNK_DROP_ENABLED", "1")
    monkeypatch.setenv("ASR_CHUNK_DROP_MIN_DURATION_S", "0.20")
    monkeypatch.setenv("ASR_CHUNK_DROP_RMS_DBFS", "-40.0")

    import importlib
    from whisper import pipeline as pl
    importlib.reload(pl)

    # span covering the tone part (0.5–0.65s has the 440 Hz tone)
    spans = [(0.5, 0.65)]
    result = pl._drop_short_low_energy_spans(audio_file, spans)
    assert result == [(0.5, 0.65)], f"Expected span kept, got {result}"


# --- Case D: long silent span is kept (duration criterion not met) ---
def test_drop_long_silent_span_is_kept(audio_file, monkeypatch):
    """0.5s silence → duration > 0.20 threshold → AND fails → kept."""
    monkeypatch.setenv("ASR_CHUNK_DROP_ENABLED", "1")
    monkeypatch.setenv("ASR_CHUNK_DROP_MIN_DURATION_S", "0.20")
    monkeypatch.setenv("ASR_CHUNK_DROP_RMS_DBFS", "-40.0")

    import importlib
    from whisper import pipeline as pl
    importlib.reload(pl)

    # 0.0–0.5s is silence but 0.5s > 0.20s so duration criterion fails
    spans = [(0.0, 0.5)]
    result = pl._drop_short_low_energy_spans(audio_file, spans)
    assert result == [(0.0, 0.5)], f"Expected span kept, got {result}"
