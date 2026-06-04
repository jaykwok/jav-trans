from __future__ import annotations

import os

from boundary.backbones import (
    BoundarySequenceClassifier,
    TinyMamba2BoundaryBackbone,
    normalize_boundary_backbone,
)
from boundary.base import SegmentationResult, SpeechBoundaryBackend, SpeechSegment
from boundary.candidates import (
    BoundaryCandidate,
    CandidateExtractionConfig,
    extract_boundary_candidates,
)
from boundary.features import BoundaryFeatureBundle, make_feature_bundle, summarize_gap_features
from boundary.planner import BoundaryPlannerConfig, PlannedChunk, PlannedIsland, plan_boundary_chunks
from boundary.refiner import (
    BoundaryDecision,
    BoundaryRefiner,
    DEFAULT_REFINER_FEATURES,
    HeuristicBoundaryRefiner,
    LEARNED_REFINER_SCHEMA,
    LearnedBoundaryRefiner,
    RefinerInput,
    build_learned_refiner_checkpoint,
    load_learned_refiner_checkpoint,
    load_boundary_refiner,
    refiner_input_to_features,
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
    "BoundaryCandidate",
    "BoundaryDecision",
    "BoundaryFeatureBundle",
    "BoundaryRefiner",
    "BoundaryPlannerConfig",
    "BoundarySequenceClassifier",
    "SegmentationResult",
    "SpeechBoundaryBackend",
    "SpeechSegment",
    "CandidateExtractionConfig",
    "DEFAULT_REFINER_FEATURES",
    "HeuristicBoundaryRefiner",
    "LEARNED_REFINER_SCHEMA",
    "LearnedBoundaryRefiner",
    "PlannedChunk",
    "PlannedIsland",
    "RefinerInput",
    "TinyMamba2BoundaryBackbone",
    "build_learned_refiner_checkpoint",
    "extract_boundary_candidates",
    "get_boundary_backend",
    "load_learned_refiner_checkpoint",
    "load_boundary_refiner",
    "make_feature_bundle",
    "normalize_boundary_backbone",
    "plan_boundary_chunks",
    "refiner_input_to_features",
    "summarize_gap_features",
]
