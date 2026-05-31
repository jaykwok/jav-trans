from __future__ import annotations

import os
import time
from typing import Any

from audio.loading import load_audio_16k_mono
from vad import get_vad_backend
from vad.base import SegmentationResult, SpeechSegment, VadBackend
from vad.whisperseg.postprocess import group_segments


class TimestampSpanVadBackend:
    """Research-only adapter for timestamp fallback span detectors."""

    name = "timestamp_span_vad"

    def __init__(self, *, name: str, use_ten_fallback: bool) -> None:
        self.name = name
        self.use_ten_fallback = use_ten_fallback

    def signature(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "revision": "research-v1",
            "source": "whisper.timestamp_fallback",
            "use_ten_fallback": self.use_ten_fallback,
            "timestamp_vad_onset": os.getenv("TIMESTAMP_VAD_ONSET", "0.5"),
            "vad_min_off": os.getenv("VAD_MIN_OFF", "0.1"),
            "vad_pad": os.getenv("VAD_PAD", "0.15"),
        }

    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        del target_sr, threshold_override
        started = time.monotonic()
        audio, sample_rate = load_audio_16k_mono(audio_path)
        duration_s = len(audio) / sample_rate if sample_rate else 0.0

        if self.use_ten_fallback:
            from whisper.timestamp_fallback import detect_speech_spans

            spans, error = detect_speech_spans(audio_path)
        else:
            from whisper.timestamp_fallback import _detect_speech_spans_ten_vad

            spans, error = _detect_speech_spans_ten_vad(audio_path)
        if error and not spans:
            raise RuntimeError(f"{self.name} failed: {error}")

        segments = [
            SpeechSegment(start=float(start), end=float(end))
            for start, end in spans
            if float(end) > float(start)
        ]
        groups = group_segments(segments)
        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=duration_s,
            parameters={**self.signature(), "error": error},
            processing_time_sec=time.monotonic() - started,
        )


def get_research_vad_backend(name: str | None) -> VadBackend:
    normalized = (name or "fusion_lite").strip().lower().replace("-", "_")
    if normalized in {"whisperseg_adaptive", "whisperseg"}:
        return get_vad_backend("whisperseg-adaptive")
    if normalized in {"fusion", "fusion_lite"}:
        return get_vad_backend("fusion_lite")
    if normalized in {"silero", "silero_vad"}:
        from vad.silero_backend import SileroVadBackend

        return SileroVadBackend()
    if normalized in {"ten", "ten_vad"}:
        return TimestampSpanVadBackend(name="ten_vad", use_ten_fallback=False)
    if normalized in {"ten_silero", "timestamp_fallback"}:
        return TimestampSpanVadBackend(name="ten_silero", use_ten_fallback=True)
    raise ValueError(
        "Unknown research VAD backend: "
        f"{name!r}. Choose: whisperseg-adaptive, fusion_lite, silero, ten_vad, ten_silero"
    )
