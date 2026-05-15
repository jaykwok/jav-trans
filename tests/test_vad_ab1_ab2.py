from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_ab1_default_threshold_is_035():
    """AB-1: whisperseg default threshold changed from 0.25 to 0.35."""
    # Module-level default in whisperseg_core
    assert os.getenv("WHISPERSEG_THRESHOLD", "0.35") == "0.35"
    # Verify backend signature default
    from vad.whisperseg_backend import WhisperSegVadBackend
    backend = WhisperSegVadBackend()
    sig = backend.signature()
    assert sig["threshold"] == pytest.approx(0.35)


def test_ab1_threshold_env_override(monkeypatch):
    """AB-1: threshold can be overridden via env."""
    monkeypatch.setenv("WHISPERSEG_THRESHOLD", "0.5")
    from vad.whisperseg_backend import WhisperSegVadBackend
    backend = WhisperSegVadBackend()
    sig = backend.signature()
    assert sig["threshold"] == pytest.approx(0.5)


def test_ab2_speech_segment_has_score_field():
    """AB-2: SpeechSegment has score: float | None = None."""
    from vad.base import SpeechSegment
    seg = SpeechSegment(start=0.0, end=1.0)
    assert seg.score is None

    seg_with_score = SpeechSegment(start=0.0, end=1.0, score=0.85)
    assert seg_with_score.score == pytest.approx(0.85)


def test_ab2_speech_segment_score_default_none():
    """AB-2: SpeechSegment.score defaults to None when not provided."""
    from vad.base import SpeechSegment
    seg = SpeechSegment(start=1.0, end=2.5)
    assert hasattr(seg, "score")
    assert seg.score is None


def test_ab2_segmentation_result_audio_stats_structure():
    """AB-2: SegmentationResult.parameters can hold audio_stats dict."""
    from vad.base import SegmentationResult, SpeechSegment
    segs = [SpeechSegment(start=0.0, end=1.0, score=0.9)]
    result = SegmentationResult(
        segments=segs,
        groups=[segs],
        method="test",
        audio_duration_sec=5.0,
        parameters={"audio_stats": {"mean_prob": 0.9, "speech_ratio": 0.2}},
        processing_time_sec=0.1,
    )
    stats = result.parameters["audio_stats"]
    assert "mean_prob" in stats
    assert "speech_ratio" in stats
    assert stats["mean_prob"] == pytest.approx(0.9)
    assert stats["speech_ratio"] == pytest.approx(0.2)


def test_ab2_whisperseg_core_probs_to_segments_fills_score():
    """AB-2: _probs_to_segments fills SpeechSegment.score from avg_prob."""
    from vad.whisperseg.whisperseg_core import WhisperSegSpeechSegmenter
    import numpy as np

    segmenter = WhisperSegSpeechSegmenter.__new__(WhisperSegSpeechSegmenter)
    segmenter.threshold = 0.35
    segmenter.min_speech_duration_ms = 80
    segmenter.min_silence_duration_ms = 80
    segmenter.speech_pad_ms = 0
    segmenter.max_speech_duration_s = 10.0
    segmenter._frame_duration_ms = 20

    # Construct a minimal speech_probs array with a clear speech region
    probs = np.array([0.0] * 10 + [0.9] * 20 + [0.0] * 10, dtype=np.float32)
    segments = segmenter._probs_to_segments(probs, audio_duration_sec=0.8)

    assert len(segments) >= 1
    speech_segs = [s for s in segments if s.score is not None]
    assert len(speech_segs) >= 1
    assert all(0.0 <= s.score <= 1.0 for s in speech_segs)
