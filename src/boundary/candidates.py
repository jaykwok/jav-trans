from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from boundary.features import BoundaryFeatureBundle

CandidateSource = Literal["cut", "valley", "gap_midpoint"]

CANDIDATE_EXTRACTOR_VERSION = 1


@dataclass(frozen=True)
class BoundaryCandidate:
    time_s: float
    score: float
    reason: str
    source: CandidateSource


@dataclass(frozen=True)
class CandidateExtractionConfig:
    min_chunk_s: float = 0.4
    target_chunk_s: float = 9.0
    cut_score_threshold: float = 0.94
    valley_score_threshold: float = 0.10

    def signature(self) -> dict:
        return {
            "candidate_extractor_version": CANDIDATE_EXTRACTOR_VERSION,
            "min_chunk_s": self.min_chunk_s,
            "target_chunk_s": self.target_chunk_s,
            "cut_score_threshold": self.cut_score_threshold,
            "valley_score_threshold": self.valley_score_threshold,
        }


def extract_boundary_candidates(
    *,
    start_s: float,
    end_s: float,
    features: BoundaryFeatureBundle,
    config: CandidateExtractionConfig,
) -> list[BoundaryCandidate]:
    if end_s <= start_s:
        return []
    lower_s = start_s + max(0.0, config.min_chunk_s)
    upper_s = end_s - max(0.0, config.min_chunk_s)
    if upper_s <= lower_s:
        return []
    candidates: list[BoundaryCandidate] = []
    candidates.extend(
        _score_run_candidates(
            features.cut_scores,
            score_frame_hop_s=features.frame_hop_s,
            lower_s=lower_s,
            upper_s=upper_s,
            threshold=config.cut_score_threshold,
            mode="high",
            source="cut",
        )
    )
    candidates.extend(
        _score_run_candidates(
            features.speech_scores,
            score_frame_hop_s=features.frame_hop_s,
            lower_s=lower_s,
            upper_s=upper_s,
            threshold=config.valley_score_threshold,
            mode="low",
            source="valley",
        )
    )
    return candidates


def gap_midpoint_candidate(
    *,
    left_end_s: float,
    right_start_s: float,
    target_chunk_s: float,
) -> BoundaryCandidate | None:
    gap_s = right_start_s - left_end_s
    if gap_s <= 0.0:
        return None
    scale = max(0.2, min(1.5, target_chunk_s / 6.0))
    score = max(0.0, min(1.0, gap_s / scale))
    return BoundaryCandidate(
        time_s=(left_end_s + right_start_s) / 2.0,
        score=score,
        reason="gap_midpoint",
        source="gap_midpoint",
    )


def best_candidate_near_target(
    candidates: Sequence[BoundaryCandidate],
    *,
    target_s: float,
) -> BoundaryCandidate | None:
    if not candidates:
        return None
    return min(candidates, key=lambda item: (abs(item.time_s - target_s), -item.score))


def _score_run_candidates(
    scores: Sequence[float] | None,
    *,
    score_frame_hop_s: float,
    lower_s: float,
    upper_s: float,
    threshold: float,
    mode: str,
    source: CandidateSource,
) -> list[BoundaryCandidate]:
    if scores is None or len(scores) == 0:
        return []
    lower = max(0, int(round(lower_s / score_frame_hop_s)))
    upper = min(len(scores), int(round(upper_s / score_frame_hop_s)))
    if upper <= lower:
        return []

    candidates: list[BoundaryCandidate] = []
    run_start: int | None = None
    for frame in range(lower, upper):
        value = float(scores[frame])
        hit = value >= threshold if mode == "high" else value <= threshold
        if hit:
            if run_start is None:
                run_start = frame
            continue
        if run_start is not None:
            candidates.append(
                _run_candidate(scores, run_start, frame, score_frame_hop_s, mode, source)
            )
        run_start = None
    if run_start is not None:
        candidates.append(
            _run_candidate(scores, run_start, upper, score_frame_hop_s, mode, source)
        )
    return candidates


def _run_candidate(
    scores: Sequence[float],
    start_frame: int,
    end_frame: int,
    score_frame_hop_s: float,
    mode: str,
    source: CandidateSource,
) -> BoundaryCandidate:
    run = [float(value) for value in scores[start_frame:end_frame]]
    score = max(run) if mode == "high" else 1.0 - min(run)
    boundary_s = ((start_frame + end_frame) / 2.0) * score_frame_hop_s
    return BoundaryCandidate(
        time_s=boundary_s,
        score=score,
        reason=f"{source}_candidate",
        source=source,
    )
