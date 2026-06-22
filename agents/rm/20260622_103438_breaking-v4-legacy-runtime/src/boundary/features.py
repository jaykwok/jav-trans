from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BoundaryFeatureBundle:
    frame_hop_s: float
    speech_scores: Sequence[float] | None = None
    cut_scores: Sequence[float] | None = None

    def signature(self) -> dict:
        return {
            "feature_bundle": "boundary_feature_bundle_v1",
            "frame_hop_s": self.frame_hop_s,
            "speech_scores": self.speech_scores is not None,
            "cut_scores": self.cut_scores is not None,
        }


@dataclass(frozen=True)
class GapFeatureSummary:
    valley_score_min: float | None = None
    cut_score_max: float | None = None


def make_feature_bundle(
    *,
    frame_hop_s: float,
    speech_scores: Sequence[float] | None = None,
    cut_scores: Sequence[float] | None = None,
) -> BoundaryFeatureBundle:
    if frame_hop_s <= 0:
        raise ValueError("frame_hop_s must be positive")
    return BoundaryFeatureBundle(
        frame_hop_s=frame_hop_s,
        speech_scores=speech_scores,
        cut_scores=cut_scores,
    )


def summarize_gap_features(
    bundle: BoundaryFeatureBundle,
    *,
    start_s: float,
    end_s: float,
) -> GapFeatureSummary:
    return GapFeatureSummary(
        valley_score_min=_range_min(bundle.speech_scores, bundle.frame_hop_s, start_s, end_s),
        cut_score_max=_range_max(bundle.cut_scores, bundle.frame_hop_s, start_s, end_s),
    )


def _range_min(
    scores: Sequence[float] | None,
    frame_hop_s: float,
    start_s: float,
    end_s: float,
) -> float | None:
    values = _range_values(scores, frame_hop_s, start_s, end_s)
    return min(values) if values else None


def _range_max(
    scores: Sequence[float] | None,
    frame_hop_s: float,
    start_s: float,
    end_s: float,
) -> float | None:
    values = _range_values(scores, frame_hop_s, start_s, end_s)
    return max(values) if values else None


def _range_values(
    scores: Sequence[float] | None,
    frame_hop_s: float,
    start_s: float,
    end_s: float,
) -> list[float]:
    if scores is None or len(scores) == 0:
        return []
    lower = max(0, int(round(max(0.0, start_s) / frame_hop_s)))
    upper = min(len(scores), int(round(max(start_s, end_s) / frame_hop_s)))
    if upper <= lower:
        return []
    return [float(value) for value in scores[lower:upper]]
