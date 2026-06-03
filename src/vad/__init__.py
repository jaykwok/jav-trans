from __future__ import annotations

import os

from vad.base import SegmentationResult, SpeechSegment, VadBackend


def get_vad_backend(name: str | None = None) -> VadBackend:
    name = (name or os.getenv("ASR_VAD_BACKEND", "fusionvad_ja")).strip()
    if name == "fusionvad_ja":
        from vad.fusionvad_ja.backend import FusionVadJaBackend

        return FusionVadJaBackend()
    raise ValueError(
        "Unknown VAD backend: "
        f"{name!r}. Choose: fusionvad_ja"
    )


__all__ = [
    "SegmentationResult",
    "SpeechSegment",
    "VadBackend",
    "get_vad_backend",
]
