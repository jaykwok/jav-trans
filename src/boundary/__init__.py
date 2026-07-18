from __future__ import annotations

import os

from boundary.backbones import (
    Mamba2TemporalEncoder,
    SpeechIslandSequenceClassifier,
)
from boundary.base import SegmentationResult, SpeechBoundaryBackend, SpeechSegment
from boundary.split_model import (
    AcousticSplitV4Planner,
    load_acoustic_split_v4_planner,
)
from boundary.sequence_features import (
    FRAME_SEQUENCE_FEATURE_SCHEMA,
    FRAME_SEQUENCE_FRAMES_SCHEMA,
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
    boundary_window_sequence_features,
    feature_extraction_hash,
    feature_extraction_signature,
    frame_sequence_feature_names,
    get_default_config,
    get_feature_dim,
    validate_sequence_features,
)


def get_boundary_backend(name: str | None = None) -> SpeechBoundaryBackend:
    backend_name = (name or os.getenv("ASR_BOUNDARY_BACKEND", "speech_boundary_ja")).strip()
    if backend_name == "speech_boundary_ja":
        from boundary.ja.backend import SpeechBoundaryJaBackend

        return SpeechBoundaryJaBackend()
    raise ValueError(
        "Unknown boundary backend: "
        f"{backend_name!r}. Choose: speech_boundary_ja"
    )

__all__ = [
    "AcousticSplitV4Planner",
    "Mamba2TemporalEncoder",
    "SegmentationResult",
    "SpeechIslandSequenceClassifier",
    "SpeechBoundaryBackend",
    "SpeechSegment",
    "FRAME_SEQUENCE_FEATURE_SCHEMA",
    "FRAME_SEQUENCE_FRAMES_SCHEMA",
    "FrameSequenceFeatureConfig",
    "FrameSequenceFeatureProvider",
    "boundary_window_sequence_features",
    "feature_extraction_hash",
    "feature_extraction_signature",
    "get_boundary_backend",
    "get_default_config",
    "get_feature_dim",
    "frame_sequence_feature_names",
    "load_acoustic_split_v4_planner",
    "validate_sequence_features",
]
