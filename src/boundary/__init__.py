from __future__ import annotations

import os

from boundary.backbones import (
    Mamba2TemporalEncoder,
    SpeechIslandSequenceClassifier,
)
from boundary.base import SegmentationResult, SpeechBoundaryBackend, SpeechSegment
from boundary.cut_refiner import CutEdgeRefiner, load_cut_edge_refiner
from boundary.outer_refiner import OuterEdgeRefiner, load_outer_edge_refiner
from boundary.split_model import SemanticSplitVerifier, load_semantic_split_verifier
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
    "CutEdgeRefiner",
    "Mamba2TemporalEncoder",
    "OuterEdgeRefiner",
    "SegmentationResult",
    "SemanticSplitVerifier",
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
    "load_cut_edge_refiner",
    "load_outer_edge_refiner",
    "load_semantic_split_verifier",
    "validate_sequence_features",
]
