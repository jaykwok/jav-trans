from __future__ import annotations

import os

from boundary.backbones import (
    BoundarySequenceClassifier,
    DualBranchDiffBoundarySequenceClassifier,
    SplitBoundaryAdapter,
    TinyMamba2BoundaryBackbone,
    compute_temporal_diff_features,
    normalize_boundary_backbone,
)
from boundary.base import SegmentationResult, SpeechBoundaryBackend, SpeechSegment
from boundary.refiner import (
    BoundaryDecision,
    DEFAULT_REFINER_CHECKPOINT_PATH,
    EdgeSequenceBoundaryRefiner,
    LEARNED_REFINER_SCHEMA,
    LearnedBoundaryRefiner,
    build_learned_refiner_checkpoint,
    load_edge_sequence_refiner_v8_checkpoint,
    load_learned_refiner_checkpoint,
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
    "BoundaryDecision",
    "BoundarySequenceClassifier",
    "DualBranchDiffBoundarySequenceClassifier",
    "SegmentationResult",
    "SpeechBoundaryBackend",
    "SpeechSegment",
    "DEFAULT_REFINER_CHECKPOINT_PATH",
    "EdgeSequenceBoundaryRefiner",
    "FRAME_SEQUENCE_FEATURE_SCHEMA",
    "FRAME_SEQUENCE_FRAMES_SCHEMA",
    "FrameSequenceFeatureConfig",
    "FrameSequenceFeatureProvider",
    "LEARNED_REFINER_SCHEMA",
    "LearnedBoundaryRefiner",
    "SplitBoundaryAdapter",
    "TinyMamba2BoundaryBackbone",
    "build_learned_refiner_checkpoint",
    "boundary_window_sequence_features",
    "feature_extraction_hash",
    "feature_extraction_signature",
    "compute_temporal_diff_features",
    "get_boundary_backend",
    "get_default_config",
    "get_feature_dim",
    "frame_sequence_feature_names",
    "load_edge_sequence_refiner_v8_checkpoint",
    "load_learned_refiner_checkpoint",
    "normalize_boundary_backbone",
    "validate_sequence_features",
]
