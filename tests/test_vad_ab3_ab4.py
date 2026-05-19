from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_ab3_neg_threshold_default_015(monkeypatch):
    monkeypatch.delenv("WHISPERSEG_NEG_THRESHOLD_OFFSET", raising=False)
    from vad.whisperseg.whisperseg_core import WhisperSegSpeechSegmenter
    seg = WhisperSegSpeechSegmenter(threshold=0.35)
    assert seg.neg_threshold_offset == pytest.approx(0.15)


def test_ab3_neg_threshold_env_override(monkeypatch):
    monkeypatch.setenv("WHISPERSEG_NEG_THRESHOLD_OFFSET", "0.22")
    from vad.whisperseg.whisperseg_core import WhisperSegSpeechSegmenter
    seg = WhisperSegSpeechSegmenter()
    assert seg.neg_threshold_offset == pytest.approx(0.22)


def test_ab3_neg_threshold_in_signature(monkeypatch):
    monkeypatch.delenv("WHISPERSEG_NEG_THRESHOLD_OFFSET", raising=False)
    from vad.whisperseg_backend import WhisperSegVadBackend
    backend = WhisperSegVadBackend()
    sig = backend.signature()
    assert "neg_offset" in sig
    assert sig["neg_offset"] == pytest.approx(0.15)


def test_ab3_threshold_override_passed_to_segment(tmp_path, monkeypatch):
    """AB-3: vad_refine passes threshold_override to backend.segment()."""
    import wave
    from audio.vad_refine import refine_chunks_via_vad
    from vad.base import SegmentationResult, SpeechSegment

    received = {}

    class RecordingBackend:
        name = "recording"

        def segment(self, audio_path, *, target_sr=16000, threshold_override=None):
            received["threshold_override"] = threshold_override
            segs = [SpeechSegment(0.0, 0.5), SpeechSegment(0.8, 1.5)]
            return SegmentationResult(
                segments=segs, groups=[[s] for s in segs],
                method="test", audio_duration_sec=1.5, parameters={}, processing_time_sec=0.0,
            )

        def signature(self):
            return {"backend": self.name}

    wav = tmp_path / "chunk.wav"
    n = int(1.5 * 16000)
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * n)

    chunk = {"index": 0, "start": 0.0, "end": 1.5, "path": str(wav), "source_audio_path": "a.wav"}
    refine_chunks_via_vad([chunk], vad_backend=RecordingBackend(), threshold_override=0.42)
    assert received.get("threshold_override") == pytest.approx(0.42)


def test_ab4_adaptive_off_by_default(monkeypatch):
    """AB-4: ASR_VAD_ADAPTIVE default is off."""
    monkeypatch.delenv("ASR_VAD_ADAPTIVE", raising=False)
    from vad.whisperseg_backend import _env_bool
    assert not _env_bool("ASR_VAD_ADAPTIVE", "0")


def test_ab4_adaptive_threshold_high_speech_ratio(monkeypatch):
    """AB-4: speech_ratio > 0.85 raises threshold."""
    from vad.whisperseg_backend import _adaptive_threshold
    result = _adaptive_threshold(0.35, speech_ratio=0.90)
    assert result == pytest.approx(0.45)


def test_ab4_adaptive_threshold_low_speech_ratio(monkeypatch):
    """AB-4: speech_ratio < 0.05 lowers threshold."""
    from vad.whisperseg_backend import _adaptive_threshold
    result = _adaptive_threshold(0.35, speech_ratio=0.02)
    assert result == pytest.approx(0.30)


def test_ab4_adaptive_threshold_normal_ratio(monkeypatch):
    """AB-4: speech_ratio in [0.05, 0.85] leaves threshold unchanged."""
    from vad.whisperseg_backend import _adaptive_threshold
    assert _adaptive_threshold(0.35, speech_ratio=0.50) == pytest.approx(0.35)


def test_ab4_adaptive_enabled_reruns_with_high_ratio(monkeypatch):
    """AB-4: when ASR_VAD_ADAPTIVE=1, high speech_ratio triggers second run."""
    monkeypatch.setenv("ASR_VAD_ADAPTIVE", "1")
    import numpy as np
    from vad.base import SegmentationResult, SpeechSegment
    from vad.whisperseg_backend import WhisperSegVadBackend

    call_thresholds = []

    class FakeSegmenter:
        threshold = 0.35
        neg_threshold_offset = 0.15

        def segment(self, audio_path, sample_rate=16000):
            call_thresholds.append(self.threshold)
            # first call → speech_ratio=0.92 (high) triggers re-run
            # second call → speech_ratio=0.5 (normal)
            ratio = 0.92 if len(call_thresholds) == 1 else 0.5
            segs = [SpeechSegment(0.0, 1.0, score=0.9)]
            return SegmentationResult(
                segments=segs,
                groups=[segs],
                method="test",
                audio_duration_sec=1.0,
                parameters={
                    "audio_stats": {"mean_prob": 0.9, "speech_ratio": ratio},
                    "neg_offset": 0.15,
                    "threshold": self.threshold,
                },
                processing_time_sec=0.0,
            )

    backend = WhisperSegVadBackend()
    backend._segmenter = FakeSegmenter()

    import tempfile, wave
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    n = 16000
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * n)

    try:
        result = backend.segment(wav_path)
        assert len(call_thresholds) == 2
        assert call_thresholds[1] > call_thresholds[0]
        assert result.parameters["adaptive"]["threshold_adjusted"] is True
    finally:
        os.unlink(wav_path)


def test_whisperseg_empty_groups_are_valid_skip_result(monkeypatch):
    from vad.base import SegmentationResult
    from vad.whisperseg_backend import WhisperSegVadBackend

    class EmptySegmenter:
        threshold = 0.35

        def segment(self, audio_path, sample_rate=16000):
            del audio_path, sample_rate
            return SegmentationResult(
                segments=[],
                groups=[],
                method="test",
                audio_duration_sec=1.0,
                parameters={
                    "audio_stats": {"mean_prob": 0.0, "speech_ratio": 0.0},
                    "neg_offset": 0.15,
                    "threshold": self.threshold,
                },
                processing_time_sec=0.0,
            )

    monkeypatch.setenv("ASR_VAD_ADAPTIVE", "0")
    backend = WhisperSegVadBackend()
    backend._segmenter = EmptySegmenter()

    result = backend.segment("silent.wav")

    assert result.groups == []
    assert result.segments == []
    assert result.parameters["audio_stats"]["speech_ratio"] == 0.0
