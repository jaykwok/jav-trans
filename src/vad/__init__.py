from __future__ import annotations

import os

from vad.base import SegmentationResult, SpeechSegment, VadBackend


def get_vad_backend(name: str | None = None) -> VadBackend:
    name = (name or os.getenv("ASR_VAD_BACKEND", "fusionvad_ja")).lower()
    if name == "whisperseg-adaptive":
        from vad.whisperseg_backend import WhisperSegVadBackend

        return WhisperSegVadBackend()
    if name in {"fusion", "fusion_lite"}:
        from vad.fusion_lite_backend import FusionLiteVadBackend

        return FusionLiteVadBackend()
    if name in {"fusionvad_ja", "fusionvad-ja", "fusionvad"}:
        from vad.fusionvad_ja.backend import FusionVadJaBackend

        return FusionVadJaBackend()
    raise ValueError(
        "Unknown VAD backend: "
        f"{name!r}. Choose: whisperseg-adaptive, fusion_lite, fusionvad_ja"
    )


__all__ = [
    "SegmentationResult",
    "SpeechSegment",
    "VadBackend",
    "get_vad_backend",
]
