from __future__ import annotations

import os

from vad.base import SegmentationResult, SpeechSegment, VadBackend
from vad.ffmpeg_backend import FfmpegSilencedetectBackend


def get_vad_backend(name: str | None = None) -> VadBackend:
    name = (name or os.getenv("ASR_VAD_BACKEND", "ffmpeg")).lower()
    if name == "ffmpeg":
        return FfmpegSilencedetectBackend()
    if name == "whisperseg":
        from vad.whisperseg_backend import WhisperSegVadBackend

        return WhisperSegVadBackend()
    raise ValueError(f"Unknown VAD backend: {name!r}. Choose: ffmpeg, whisperseg")


__all__ = [
    "FfmpegSilencedetectBackend",
    "SegmentationResult",
    "SpeechSegment",
    "VadBackend",
    "get_vad_backend",
]
