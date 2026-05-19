from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_whisperseg_backend_defaults_match_runtime_config(monkeypatch):
    for key in (
        "WHISPERSEG_MIN_SPEECH_MS",
        "WHISPERSEG_MIN_SILENCE_MS",
        "WHISPERSEG_PAD_MS",
        "WHISPERSEG_MAX_SPEECH_S",
        "WHISPERSEG_MAX_GROUP_S",
    ):
        monkeypatch.delenv(key, raising=False)

    from vad.whisperseg.whisperseg_core import WhisperSegSpeechSegmenter
    from vad.whisperseg_backend import WhisperSegVadBackend

    segmenter = WhisperSegSpeechSegmenter()
    signature = WhisperSegVadBackend().signature()

    assert segmenter.min_speech_duration_ms == 100
    assert segmenter.min_silence_duration_ms == 100
    assert segmenter.speech_pad_ms == 300
    assert segmenter.max_speech_duration_s == pytest.approx(5.0)
    assert segmenter.max_group_duration_s == pytest.approx(6.0)
    assert signature["min_speech_ms"] == 100
    assert signature["min_silence_ms"] == 100
    assert signature["pad_ms"] == 300
    assert signature["max_speech_s"] == pytest.approx(5.0)
    assert signature["max_group_s"] == pytest.approx(6.0)


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
    segmenter.neg_threshold_offset = 0.15

    # Construct a minimal speech_probs array with a clear speech region
    probs = np.array([0.0] * 10 + [0.9] * 20 + [0.0] * 10, dtype=np.float32)
    segments = segmenter._probs_to_segments(probs, audio_duration_sec=0.8)

    assert len(segments) >= 1
    speech_segs = [s for s in segments if s.score is not None]
    assert len(speech_segs) >= 1
    assert all(0.0 <= s.score <= 1.0 for s in speech_segs)


def test_ab2_whisperseg_segment_result_includes_score_and_audio_stats(monkeypatch):
    """AB-2: fake WhisperSeg output populates score plus parameters.audio_stats."""
    import numpy as np

    from vad.whisperseg.whisperseg_core import WhisperSegSpeechSegmenter

    segmenter = WhisperSegSpeechSegmenter(
        threshold=0.35,
        min_speech_duration_ms=80,
        min_silence_duration_ms=80,
        speech_pad_ms=0,
    )
    segmenter._frame_duration_ms = 20
    segmenter._actual_device = "CPU"
    monkeypatch.setattr(segmenter, "_ensure_model", lambda: None)
    monkeypatch.setattr(
        segmenter,
        "_audio_forward",
        lambda _audio: np.array(
            [0.0] * 10 + [0.9] * 20 + [0.0] * 10,
            dtype=np.float32,
        ),
    )

    result = segmenter.segment(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    assert result.segments
    assert result.segments[0].score is not None
    assert result.segments[0].score == pytest.approx(0.9, rel=1e-6)
    stats = result.parameters["audio_stats"]
    assert stats["mean_prob"] == pytest.approx(0.9, rel=1e-6)
    assert stats["speech_ratio"] == pytest.approx(0.4, rel=1e-6)


def test_ab3_neg_offset_defaults_to_015(monkeypatch):
    monkeypatch.delenv("WHISPERSEG_NEG_THRESHOLD_OFFSET", raising=False)

    from vad.whisperseg.whisperseg_core import WhisperSegSpeechSegmenter
    from vad.whisperseg_backend import WhisperSegVadBackend

    segmenter = WhisperSegSpeechSegmenter()
    assert segmenter.neg_threshold_offset == pytest.approx(0.15)
    assert segmenter._get_parameters()["neg_offset"] == pytest.approx(0.15)
    assert WhisperSegVadBackend().signature()["neg_offset"] == pytest.approx(0.15)


def test_whisperseg_cuda_provider_requires_gpu_device():
    from vad.whisperseg import whisperseg_core

    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        @staticmethod
        def get_device():
            return "CPU"

    assert whisperseg_core._cuda_provider_usable(FakeOrt) is False


def test_whisperseg_preloads_onnx_cuda_runtime():
    from vad.whisperseg import whisperseg_core

    calls: list[dict] = []

    class FakeOrt:
        @staticmethod
        def preload_dlls(**kwargs):
            calls.append(kwargs)

    whisperseg_core._preload_onnx_cuda_runtime(FakeOrt)

    assert calls == [{"cuda": True, "cudnn": True, "msvc": False, "directory": ""}]


def test_ab3_neg_offset_env_override(monkeypatch):
    monkeypatch.setenv("WHISPERSEG_NEG_THRESHOLD_OFFSET", "0.22")

    from vad.whisperseg.whisperseg_core import WhisperSegSpeechSegmenter
    from vad.whisperseg_backend import WhisperSegVadBackend

    segmenter = WhisperSegSpeechSegmenter()
    assert segmenter.neg_threshold_offset == pytest.approx(0.22)
    assert segmenter._get_parameters()["neg_offset"] == pytest.approx(0.22)
    assert WhisperSegVadBackend().signature()["neg_offset"] == pytest.approx(0.22)


class _FakeSegmenter:
    def __init__(self, speech_ratios: list[float]):
        self.threshold = 0.35
        self.calls: list[float] = []
        self._speech_ratios = list(speech_ratios)

    def segment(self, audio_path: str, sample_rate: int = 16000):
        del audio_path, sample_rate
        self.calls.append(self.threshold)
        speech_ratio = self._speech_ratios.pop(0) if self._speech_ratios else 0.5
        segment = SimpleNamespace(start=0.0, end=1.0, score=0.8)
        return SimpleNamespace(
            groups=[[segment]],
            audio_duration_sec=1.0,
            parameters={
                "threshold": self.threshold,
                "neg_offset": 0.15,
                "audio_stats": {"mean_prob": 0.8, "speech_ratio": speech_ratio},
            },
        )


def test_ab4_adaptive_threshold_default_off(monkeypatch):
    from vad.whisperseg_backend import WhisperSegVadBackend

    monkeypatch.delenv("ASR_VAD_ADAPTIVE", raising=False)
    backend = WhisperSegVadBackend()
    fake_segmenter = _FakeSegmenter([0.9])
    backend._segmenter = fake_segmenter

    result = backend.segment("audio.wav")

    assert fake_segmenter.calls == [pytest.approx(0.35)]
    assert fake_segmenter.threshold == pytest.approx(0.35)
    assert result.parameters["adaptive"]["enabled"] is False
    assert result.parameters["adaptive"]["threshold_adjusted"] is False


def test_ab4_adaptive_threshold_raises_for_dense_audio(monkeypatch):
    from vad.whisperseg_backend import WhisperSegVadBackend

    monkeypatch.setenv("ASR_VAD_ADAPTIVE", "1")
    backend = WhisperSegVadBackend()
    fake_segmenter = _FakeSegmenter([0.9, 0.6])
    backend._segmenter = fake_segmenter

    result = backend.segment("audio.wav")

    assert fake_segmenter.calls == [pytest.approx(0.35), pytest.approx(0.45)]
    assert fake_segmenter.threshold == pytest.approx(0.35)
    assert result.parameters["threshold"] == pytest.approx(0.45)
    assert result.parameters["adaptive"]["threshold_adjusted"] is True
    assert result.parameters["adaptive"]["final_threshold"] == pytest.approx(0.45)


def test_ab4_adaptive_threshold_lowers_for_sparse_audio(monkeypatch):
    from vad.whisperseg_backend import WhisperSegVadBackend

    monkeypatch.setenv("ASR_VAD_ADAPTIVE", "1")
    backend = WhisperSegVadBackend()
    fake_segmenter = _FakeSegmenter([0.01, 0.2])
    backend._segmenter = fake_segmenter

    result = backend.segment("audio.wav")

    assert fake_segmenter.calls == [pytest.approx(0.35), pytest.approx(0.3)]
    assert fake_segmenter.threshold == pytest.approx(0.35)
    assert result.parameters["threshold"] == pytest.approx(0.3)
    assert result.parameters["adaptive"]["threshold_adjusted"] is True
    assert result.parameters["adaptive"]["final_threshold"] == pytest.approx(0.3)
