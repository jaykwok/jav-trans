from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_vad_registry_exposes_only_current_user_modes(monkeypatch):
    import vad

    monkeypatch.setattr(
        "vad.fusion_lite_backend.FusionLiteVadBackend.__init__",
        lambda self: None,
        raising=False,
    )
    adaptive = vad.get_vad_backend("whisperseg-adaptive")
    assert adaptive.name == "whisperseg_v1"
    fusion = vad.get_vad_backend("fusion_lite")
    assert fusion.name == "fusion_lite_v1"
    with pytest.raises(ValueError, match="fusion_lite_boost"):
        vad.get_vad_backend("fusion_lite_boost")
    with pytest.raises(ValueError, match="whisperseg"):
        vad.get_vad_backend("whisperseg")
    with pytest.raises(ValueError, match="hybrid_precision"):
        vad.get_vad_backend("hybrid_precision")
    with pytest.raises(ValueError, match="silero"):
        vad.get_vad_backend("silero")


def test_silero_timestamps_to_segments_skips_invalid_items():
    from vad.silero_backend import _timestamps_to_segments

    segments = _timestamps_to_segments(
        [
            {"start": 0.0, "end": 0.5},
            {"start": 1.0, "end": 0.5},
            {"start": "bad", "end": 2.0},
        ]
    )

    assert [(segment.start, segment.end) for segment in segments] == [(0.0, 0.5)]


def test_fusion_lite_scoring_keeps_acoustic_speech_without_gate_overlap():
    from vad.base import SpeechSegment
    from vad.fusion_lite_backend import _filter_grouped_segments

    speech_like = SpeechSegment(0.0, 1.0, 0.70)
    noise_like = SpeechSegment(2.0, 2.2, 0.15)
    params = {
        "primary_weight": 0.45,
        "gate_weight": 0.25,
        "rms_weight": 0.15,
        "spectral_flux_weight": 0.10,
        "duration_weight": 0.05,
        "min_score": 0.45,
        "min_gate_overlap_ratio": 0.05,
        "gate_pad_s": 0.0,
        "default_primary_score": 0.50,
    }

    def features(segment):
        if segment is speech_like:
            return {
                "rms_dbfs": -25.0,
                "rms_score": 0.9,
                "spectral_flux_score": 0.7,
                "duration_score": 1.0,
            }
        return {
            "rms_dbfs": -60.0,
            "rms_score": 0.0,
            "spectral_flux_score": 0.0,
            "duration_score": 0.2,
        }

    groups, kept, dropped, decisions = _filter_grouped_segments(
        [[speech_like, noise_like]],
        [],
        feature_lookup=features,
        params=params,
    )

    assert groups == [[speech_like]]
    assert kept == [speech_like]
    assert dropped == [noise_like]
    assert decisions[0]["speech_score"] > decisions[1]["speech_score"]
    assert decisions[0]["raw_score"] == decisions[0]["speech_score"]
    assert decisions[0]["score_enhancement"] == pytest.approx(1.0)


def test_fusion_lite_signature_contains_weights_and_feature_thresholds(monkeypatch):
    from vad.base import SegmentationResult, SpeechSegment
    from vad.fusion_lite_backend import FusionLiteVadBackend

    class StubBackend:
        def __init__(self, name: str):
            self.name = name

        def signature(self):
            return {"backend": self.name}

        def segment(self, audio_path: str, *, target_sr: int = 16000, threshold_override=None):
            del audio_path, target_sr, threshold_override
            segment = SpeechSegment(0.0, 1.0, 0.8)
            return SegmentationResult(
                segments=[segment],
                groups=[[segment]],
                method=self.name,
                audio_duration_sec=2.0,
                parameters={"backend": self.name},
                processing_time_sec=0.01,
            )

    monkeypatch.setattr(
        "vad.fusion_lite_backend._load_component_backend",
        lambda name: StubBackend(str(name)),
    )
    backend = FusionLiteVadBackend()
    sig = backend.signature()

    assert sig["backend"] == "fusion_lite_v1"
    assert sig["primary"] == {"backend": "whisperseg"}
    assert sig["gate"] == {"backend": "silero"}
    assert sig["primary_weight"] == pytest.approx(0.45)
    assert sig["rms_floor_dbfs"] == pytest.approx(-50.0)
    assert sig["spectral_flux_full"] == pytest.approx(0.0060)
    assert sig["scoring_variant"] == "linear"
    assert sig["dynamic_gate_pad"] is False


def test_vad_cache_key_changes_with_fusion_weight(monkeypatch, tmp_path):
    import wave

    from whisper import vad_chunk_cache

    wav_path = tmp_path / "audio.wav"
    with wave.open(str(wav_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 16000)

    monkeypatch.setenv("VAD_CHUNK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ASR_VAD_BACKEND", "fusion_lite")
    monkeypatch.setenv("FUSION_VAD_RMS_WEIGHT", "0.15")
    first = vad_chunk_cache.build_cache_lookup(
        str(wav_path),
        vad_signature={"backend": "fusion_lite_v1"},
        chunk_config={},
    )
    monkeypatch.setenv("FUSION_VAD_RMS_WEIGHT", "0.25")
    second = vad_chunk_cache.build_cache_lookup(
        str(wav_path),
        vad_signature={"backend": "fusion_lite_v1"},
        chunk_config={},
    )

    assert first["digest"] != second["digest"]
