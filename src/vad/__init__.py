from __future__ import annotations

import os

from vad.base import SegmentationResult, SpeechSegment, VadBackend
from vad.ffmpeg_backend import FfmpegSilencedetectBackend


def get_vad_backend(name: str | None = None) -> VadBackend:
    name = (name or os.getenv("ASR_VAD_BACKEND", "whisperseg")).lower()
    if name == "whisperseg":
        from vad.whisperseg_backend import WhisperSegVadBackend

        return WhisperSegVadBackend()
    if name in {"fusion", "fusion_lite"}:
        from vad.fusion_lite_backend import FusionLiteVadBackend

        return FusionLiteVadBackend()
    raise ValueError(
        f"Unknown VAD backend: {name!r}. Choose: whisperseg, fusion_lite"
    )


__all__ = [
    "FfmpegSilencedetectBackend",
    "SegmentationResult",
    "SpeechSegment",
    "VadBackend",
    "get_vad_backend",
]
