from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from asr.backends.qwen import (
    DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO,
    checkpoint_path_for_repo_env,
    current_qwen_asr_backend,
    qwen_asr_default_model_path,
    qwen_asr_repo_id,
    validate_checkpoint_repo_id,
)
from audio.loading import load_audio_16k_mono
from boundary.base import SegmentationResult, SpeechSegment
from boundary.ja.dataset import frame_count
from boundary.ja.features import (
    FeatureConfig,
    align_feature_frames,
    build_ptm_feature_extractor,
    extract_mfcc,
    is_low_frame_rate_ptm,
)
from boundary.ja.model import (
    load_speech_island_scorer_checkpoint,
    score_speech_island_probabilities,
)
from subtitles.options import BASE_FPS


DEFAULT_PTM = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
DEFAULT_MODEL_PATH = qwen_asr_default_model_path(DEFAULT_PTM)
DEFAULT_OPERATING_POINT = "qwen-mamba2-speech-island-scorer-v8"


@dataclass(frozen=True)
class _SplitPeakCandidate:
    frame: int
    score: float
    prominence: float
    speech_valley: float = 0.0

    @property
    def strength(self) -> float:
        return float(self.score) + float(self.prominence) + float(self.speech_valley)


_MAX_WEAK_CUT_CANDIDATES_PER_SEGMENT = 64


def _cut_candidate_payload(
    candidate: _SplitPeakCandidate,
    *,
    frame_hop_s: float,
    kind: str,
) -> dict[str, Any]:
    frame = int(candidate.frame)
    time_s = float(frame) * float(frame_hop_s)
    return {
        "kind": str(kind),
        "time_s": round(time_s, 6),
        "frame": frame,
        "score": float(candidate.score),
        "prominence": float(candidate.prominence),
        "speech_valley": float(candidate.speech_valley),
        "strength": float(candidate.strength),
    }


def _dedupe_cut_candidate_payloads(
    candidates: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            time_s = float(candidate["time_s"])
        except (KeyError, TypeError, ValueError):
            continue
        kind = str(candidate.get("kind") or "")
        key = (kind, int(round(time_s * 1000.0)))
        strength = float(candidate.get("strength") or 0.0)
        existing = by_key.get(key)
        existing_strength = float(existing.get("strength") or 0.0) if existing else -1.0
        if existing is None or strength > existing_strength:
            by_key[key] = dict(candidate)
    return [
        by_key[key]
        for key in sorted(
            by_key,
            key=lambda item: (float(by_key[item].get("time_s") or 0.0), item[0]),
        )
    ]


def _candidate_payloads_in_window(
    candidates: Iterable[dict[str, Any]],
    *,
    start: float,
    end: float,
    include_edges: bool = False,
) -> list[dict[str, Any]]:
    eps = 1e-6
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            time_s = float(candidate["time_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if include_edges:
            keep = float(start) - eps <= time_s <= float(end) + eps
        else:
            keep = float(start) + eps < time_s < float(end) - eps
        if keep:
            selected.append(dict(candidate))
    return _dedupe_cut_candidate_payloads(selected)


def _as_weak_cut_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    payload = dict(candidate)
    payload["kind"] = "weak"
    payload["downgraded_from"] = str(candidate.get("kind") or "primary")
    return payload


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_optional_float(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _scorer_checkpoint_from_env(ptm: str) -> str:
    raw_mapping = os.getenv("SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO", "").strip()
    return checkpoint_path_for_repo_env(
        repo_id=ptm,
        mapping_env="SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO",
        default_mapping=DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO,
        required=bool(raw_mapping),
    )


def _validate_scorer_checkpoint_repo(scorer, expected_ptm: str) -> None:
    validate_checkpoint_repo_id(
        scorer.metadata.get("ptm_repo_id"),
        expected_ptm,
        checkpoint_kind="SpeechBoundary-JA scorer",
        metadata_key="metadata.ptm_repo_id",
    )


def _model_device(requested: str):
    import torch

    value = requested.strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        value = "cpu"
    return torch.device(value)


def _first_parameter_device_dtype(model) -> tuple[str, str]:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return "unknown", "unknown"
    try:
        parameter = next(parameters())
    except StopIteration:
        return "none", "none"
    except Exception as exc:  # pragma: no cover - defensive logging only
        return "error", type(exc).__name__
    return str(parameter.device), str(parameter.dtype)


def _dilated_frames(values: np.ndarray, *, dilation_frames: int) -> np.ndarray:
    mask = np.asarray(values, dtype=bool)
    if dilation_frames <= 0 or mask.size == 0:
        return mask.astype(np.int8, copy=False)
    out = mask.copy()
    active = np.flatnonzero(mask)
    for index in active:
        start = max(0, int(index) - dilation_frames)
        end = min(out.size, int(index) + dilation_frames + 1)
        out[start:end] = True
    return out.astype(np.int8, copy=False)


def _hysteresis_frames(
    probabilities: np.ndarray,
    *,
    on_threshold: float,
    off_threshold: float,
) -> np.ndarray:
    if on_threshold < 0.0:
        raise ValueError("SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD must be non-negative")
    if off_threshold < 0.0:
        raise ValueError("SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD must be non-negative")
    if on_threshold < off_threshold:
        raise ValueError(
            "SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD must be greater than or equal to "
            "SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD"
        )
    values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    frames = np.zeros(values.size, dtype=np.int8)
    active = False
    for index, value in enumerate(values):
        if active:
            if float(value) < off_threshold:
                active = False
        elif float(value) >= on_threshold:
            active = True
        if active:
            frames[index] = 1
    return frames


def _smooth_scores(values: np.ndarray, *, window_frames: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    window = max(1, int(window_frames))
    if data.size == 0 or window <= 1:
        return data.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(data, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _peak_prominence(values: np.ndarray, index: int, *, lower: int, upper: int, window_frames: int) -> float:
    radius = max(1, int(window_frames))
    left = values[max(lower, index - radius) : index + 1]
    right = values[index : min(upper, index + radius + 1)]
    if left.size == 0 or right.size == 0:
        return 0.0
    base = max(float(np.min(left)), float(np.min(right)))
    return float(values[index]) - base


def _quantile(values: Iterable[float], quantile: float) -> float:
    data = np.asarray(list(values), dtype=np.float32)
    if data.size == 0:
        return 0.0
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0
    return float(np.quantile(finite, min(1.0, max(0.0, float(quantile)))))


def _snap_split_frame(
    peak_frame: int,
    *,
    start_frame: int,
    end_frame: int,
    speech_probs: np.ndarray,
    snap_frames: int,
) -> int:
    lower = max(start_frame + 1, int(peak_frame) - max(0, int(snap_frames)))
    upper = min(end_frame - 1, int(peak_frame) + max(0, int(snap_frames)))
    if upper < lower:
        return int(peak_frame)
    speech = np.asarray(speech_probs, dtype=np.float32)
    total = int(speech.size)
    lower = min(lower, max(0, total - 1))
    upper = min(upper, max(0, total - 1))
    if upper < lower:
        return int(peak_frame)
    candidates = list(range(lower, upper + 1))
    return min(
        candidates,
        key=lambda frame: (
            float(speech[frame]),
            abs(int(frame) - int(peak_frame)),
        ),
    )


def _split_peak_candidates_for_segment(
    segment: SpeechSegment,
    *,
    speech_probs: np.ndarray,
    split_probs: np.ndarray,
    frame_hop_s: float,
    split_smooth_s: float,
    split_snap_s: float,
    min_split_segment_s: float,
) -> list[_SplitPeakCandidate]:
    total = min(int(speech_probs.size), int(split_probs.size))
    if total <= 0:
        return []
    start_frame = max(0, min(total, int(np.floor(max(0.0, segment.start) / frame_hop_s))))
    end_frame = max(start_frame, min(total, int(np.ceil(max(segment.start, segment.end) / frame_hop_s))))
    min_frames = max(1, int(round(max(0.0, min_split_segment_s) / frame_hop_s)))
    lower = start_frame + min_frames
    upper = end_frame - min_frames
    if upper <= lower:
        return []
    smooth_window = max(1, int(round(max(0.0, split_smooth_s) / frame_hop_s)))
    smoothed = _smooth_scores(split_probs[:total], window_frames=smooth_window)
    candidates: list[_SplitPeakCandidate] = []
    for frame in range(lower, upper):
        value = float(smoothed[frame])
        prev_value = float(smoothed[frame - 1]) if frame > start_frame else value
        next_value = float(smoothed[frame + 1]) if frame + 1 < end_frame else value
        if value < prev_value or value < next_value:
            continue
        prominence = _peak_prominence(
            smoothed,
            frame,
            lower=start_frame,
            upper=end_frame,
            window_frames=max(1, smooth_window),
        )
        if prominence <= 0.0:
            continue
        snapped = _snap_split_frame(
            frame,
            start_frame=start_frame,
            end_frame=end_frame,
            speech_probs=speech_probs[:total],
            snap_frames=max(0, int(round(max(0.0, split_snap_s) / frame_hop_s))),
        )
        speech_valley = 1.0 - float(np.clip(speech_probs[min(max(snapped, 0), total - 1)], 0.0, 1.0))
        candidates.append(
            _SplitPeakCandidate(
                frame=snapped,
                score=value,
                prominence=prominence,
                speech_valley=speech_valley,
            )
        )
    return candidates


def _select_peak_candidates(
    candidates: Iterable[_SplitPeakCandidate],
    *,
    nms_frames: int,
    max_count: int | None = None,
    rank_by_prominence: bool = False,
) -> list[_SplitPeakCandidate]:
    selected_frames: list[int] = []
    selected: list[_SplitPeakCandidate] = []
    ranked = sorted(
        candidates,
        key=(
            (lambda item: (item.prominence, item.score))
            if rank_by_prominence
            else (lambda item: (item.score, item.prominence))
        ),
        reverse=True,
    )
    for candidate in ranked:
        if max_count is not None and len(selected) >= max_count:
            break
        frame = int(candidate.frame)
        if all(abs(frame - existing) >= nms_frames for existing in selected_frames):
            selected_frames.append(frame)
            selected.append(candidate)
    return selected


def _select_peak_frames(
    candidates: Iterable[_SplitPeakCandidate],
    *,
    nms_frames: int,
    max_count: int | None = None,
    rank_by_prominence: bool = False,
) -> list[int]:
    return [
        int(candidate.frame)
        for candidate in _select_peak_candidates(
            candidates,
            nms_frames=nms_frames,
            max_count=max_count,
            rank_by_prominence=rank_by_prominence,
        )
    ]


def _weak_split_peak_candidates_for_segment(
    candidates: Iterable[_SplitPeakCandidate],
    primary_candidates: Iterable[_SplitPeakCandidate],
    *,
    nms_frames: int,
) -> list[_SplitPeakCandidate]:
    primary_frames = [int(candidate.frame) for candidate in primary_candidates]
    weak_pool = [
        candidate
        for candidate in candidates
        if all(abs(int(candidate.frame) - frame) >= nms_frames for frame in primary_frames)
    ]
    if not weak_pool:
        return []
    return sorted(
        _select_peak_candidates(
            weak_pool,
            nms_frames=nms_frames,
            max_count=_MAX_WEAK_CUT_CANDIDATES_PER_SEGMENT,
            rank_by_prominence=True,
        ),
        key=lambda item: int(item.frame),
    )


def _split_segment_at_candidates(
    segment: SpeechSegment,
    split_candidates: Iterable[_SplitPeakCandidate],
    *,
    frame_hop_s: float,
    min_split_segment_s: float,
    subtitle_min_duration_s: float,
    weak_candidates: Iterable[_SplitPeakCandidate] = (),
) -> list[SpeechSegment]:
    by_frame: dict[int, _SplitPeakCandidate] = {}
    for candidate in split_candidates:
        frame = int(candidate.frame)
        current = by_frame.get(frame)
        if current is None or candidate.strength > current.strength:
            by_frame[frame] = candidate
    candidates = [by_frame[frame] for frame in sorted(by_frame)]
    weak_payloads = [
        _cut_candidate_payload(candidate, frame_hop_s=frame_hop_s, kind="weak")
        for candidate in weak_candidates
    ]
    if not candidates:
        base_segment = _segment_with_split_metadata(
            start=segment.start,
            end=segment.end,
            score=segment.score,
            subtitle_min_duration_s=subtitle_min_duration_s,
            left=None,
            right=None,
            weak_cut_candidates=_candidate_payloads_in_window(
                weak_payloads,
                start=segment.start,
                end=segment.end,
            ),
        )
        return [base_segment]
    parts: list[SpeechSegment] = []
    cursor = float(segment.start)
    left_candidate: _SplitPeakCandidate | None = None
    left_payload: dict[str, Any] | None = None
    min_duration = max(0.0, float(min_split_segment_s))
    for candidate in candidates:
        boundary = float(candidate.frame) * frame_hop_s
        boundary = max(segment.start, min(float(boundary), segment.end))
        if boundary - cursor < min_duration:
            continue
        if segment.end - boundary < min_duration:
            continue
        right_payload = _cut_candidate_payload(candidate, frame_hop_s=frame_hop_s, kind="primary")
        parts.append(
            _segment_with_split_metadata(
                start=cursor,
                end=boundary,
                score=segment.score,
                subtitle_min_duration_s=subtitle_min_duration_s,
                left=left_candidate,
                right=candidate,
                primary_cut_candidates=[
                    payload for payload in (left_payload, right_payload) if payload is not None
                ],
                weak_cut_candidates=_candidate_payloads_in_window(
                    weak_payloads,
                    start=cursor,
                    end=boundary,
                ),
            )
        )
        cursor = boundary
        left_candidate = candidate
        left_payload = right_payload
    if segment.end > cursor:
        parts.append(
            _segment_with_split_metadata(
                start=cursor,
                end=segment.end,
                score=segment.score,
                subtitle_min_duration_s=subtitle_min_duration_s,
                left=left_candidate,
                right=None,
                primary_cut_candidates=(
                    [] if left_payload is None else [left_payload]
                ),
                weak_cut_candidates=_candidate_payloads_in_window(
                    weak_payloads,
                    start=cursor,
                    end=segment.end,
                ),
            )
        )
    return _resolve_micro_segments(parts or [segment], subtitle_min_duration_s=subtitle_min_duration_s)


def _segment_with_split_metadata(
    *,
    start: float,
    end: float,
    score: float | None,
    subtitle_min_duration_s: float,
    left: _SplitPeakCandidate | None,
    right: _SplitPeakCandidate | None,
    micro_chunk_candidate: bool = False,
    micro_resolve_action: str = "",
    micro_resolve_reason: str = "",
    primary_cut_candidates: Iterable[dict[str, Any]] = (),
    weak_cut_candidates: Iterable[dict[str, Any]] = (),
) -> SpeechSegment:
    duration = max(0.0, float(end) - float(start))
    return SpeechSegment(
        start=float(start),
        end=float(end),
        score=score,
        subtitle_min_duration_s=float(subtitle_min_duration_s),
        below_subtitle_min_duration=duration < float(subtitle_min_duration_s),
        micro_chunk_candidate=bool(micro_chunk_candidate),
        micro_resolve_action=micro_resolve_action,
        micro_resolve_reason=micro_resolve_reason,
        left_split_score=None if left is None else float(left.score),
        right_split_score=None if right is None else float(right.score),
        left_split_prominence=None if left is None else float(left.prominence),
        right_split_prominence=None if right is None else float(right.prominence),
        left_split_speech_valley=None if left is None else float(left.speech_valley),
        right_split_speech_valley=None if right is None else float(right.speech_valley),
        primary_cut_candidates=_dedupe_cut_candidate_payloads(primary_cut_candidates),
        weak_cut_candidates=_dedupe_cut_candidate_payloads(weak_cut_candidates),
    )


def _split_strength(
    *,
    score: float | None,
    prominence: float | None,
    speech_valley: float | None,
) -> float | None:
    if score is None or prominence is None or speech_valley is None:
        return None
    return float(score) + float(prominence) + float(speech_valley)


def _metadata_candidate_from_side(segment: SpeechSegment, *, right: bool) -> _SplitPeakCandidate | None:
    if right:
        strength = _split_strength(
            score=segment.right_split_score,
            prominence=segment.right_split_prominence,
            speech_valley=segment.right_split_speech_valley,
        )
        if strength is None:
            return None
        return _SplitPeakCandidate(
            frame=0,
            score=float(segment.right_split_score),
            prominence=float(segment.right_split_prominence),
            speech_valley=float(segment.right_split_speech_valley),
        )
    strength = _split_strength(
        score=segment.left_split_score,
        prominence=segment.left_split_prominence,
        speech_valley=segment.left_split_speech_valley,
    )
    if strength is None:
        return None
    return _SplitPeakCandidate(
        frame=0,
        score=float(segment.left_split_score),
        prominence=float(segment.left_split_prominence),
        speech_valley=float(segment.left_split_speech_valley),
    )


def _split_strength_for_side(segment: SpeechSegment, *, right: bool) -> float | None:
    if right:
        return _split_strength(
            score=segment.right_split_score,
            prominence=segment.right_split_prominence,
            speech_valley=segment.right_split_speech_valley,
        )
    return _split_strength(
        score=segment.left_split_score,
        prominence=segment.left_split_prominence,
        speech_valley=segment.left_split_speech_valley,
    )


def _micro_strengths_are_balanced(left: float, right: float) -> bool:
    stronger = max(abs(left), abs(right), 1e-6)
    return abs(left - right) <= 0.15 * stronger


def _merge_segments(
    left: SpeechSegment,
    right: SpeechSegment,
    *,
    subtitle_min_duration_s: float,
    action: str,
    reason: str,
) -> SpeechSegment:
    left_boundary = _metadata_candidate_from_side(left, right=False)
    right_boundary = _metadata_candidate_from_side(right, right=True)
    start = float(left.start)
    end = float(right.end)
    primary_candidates: list[dict[str, Any]] = []
    weak_candidates: list[dict[str, Any]] = []
    for candidate in list(left.primary_cut_candidates or []) + list(right.primary_cut_candidates or []):
        try:
            time_s = float(candidate["time_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if start + 1e-6 < time_s < end - 1e-6:
            weak_candidates.append(_as_weak_cut_candidate(candidate))
        else:
            primary_candidates.append(dict(candidate))
    weak_candidates.extend(left.weak_cut_candidates or [])
    weak_candidates.extend(right.weak_cut_candidates or [])
    return _segment_with_split_metadata(
        start=start,
        end=end,
        score=max(
            [value for value in (left.score, right.score) if value is not None],
            default=None,
        ),
        subtitle_min_duration_s=subtitle_min_duration_s,
        left=left_boundary,
        right=right_boundary,
        micro_chunk_candidate=left.micro_chunk_candidate or right.micro_chunk_candidate,
        micro_resolve_action=action,
        micro_resolve_reason=reason,
        primary_cut_candidates=primary_candidates,
        weak_cut_candidates=weak_candidates,
    )


def _mark_micro_candidate(
    segment: SpeechSegment,
    *,
    subtitle_min_duration_s: float,
    action: str,
    reason: str,
) -> SpeechSegment:
    return _segment_with_split_metadata(
        start=segment.start,
        end=segment.end,
        score=segment.score,
        subtitle_min_duration_s=subtitle_min_duration_s,
        left=_metadata_candidate_from_side(segment, right=False),
        right=_metadata_candidate_from_side(segment, right=True),
        micro_chunk_candidate=True,
        micro_resolve_action=action,
        micro_resolve_reason=reason,
        primary_cut_candidates=segment.primary_cut_candidates,
        weak_cut_candidates=segment.weak_cut_candidates,
    )


def _with_subtitle_metadata(
    segment: SpeechSegment,
    *,
    subtitle_min_duration_s: float,
) -> SpeechSegment:
    return _segment_with_split_metadata(
        start=segment.start,
        end=segment.end,
        score=segment.score,
        subtitle_min_duration_s=subtitle_min_duration_s,
        left=_metadata_candidate_from_side(segment, right=False),
        right=_metadata_candidate_from_side(segment, right=True),
        micro_chunk_candidate=segment.micro_chunk_candidate,
        micro_resolve_action=segment.micro_resolve_action,
        micro_resolve_reason=segment.micro_resolve_reason,
        primary_cut_candidates=segment.primary_cut_candidates,
        weak_cut_candidates=segment.weak_cut_candidates,
    )


def _resolve_micro_segments(
    parts: list[SpeechSegment],
    *,
    subtitle_min_duration_s: float,
) -> list[SpeechSegment]:
    resolved = list(parts)
    index = 0
    while index < len(resolved):
        segment = resolved[index]
        duration = max(0.0, float(segment.end) - float(segment.start))
        if duration >= subtitle_min_duration_s:
            index += 1
            continue
        if segment.micro_resolve_action == "preserve_micro_candidate":
            index += 1
            continue
        left_strength = _split_strength_for_side(segment, right=False)
        right_strength = _split_strength_for_side(segment, right=True)
        if index <= 0 or index + 1 >= len(resolved) or left_strength is None or right_strength is None:
            resolved[index] = _mark_micro_candidate(
                segment,
                subtitle_min_duration_s=subtitle_min_duration_s,
                action="preserve_edge_micro_candidate",
                reason="micro_chunk_without_two_split_boundaries",
            )
            index += 1
            continue
        if _micro_strengths_are_balanced(left_strength, right_strength):
            resolved[index] = _mark_micro_candidate(
                segment,
                subtitle_min_duration_s=subtitle_min_duration_s,
                action="preserve_micro_candidate",
                reason="balanced_split_evidence",
            )
            index += 1
            continue
        if left_strength < right_strength:
            merged = _merge_segments(
                resolved[index - 1],
                segment,
                subtitle_min_duration_s=subtitle_min_duration_s,
                action="merge_micro_into_left",
                reason="left_split_weaker",
            )
            resolved[index - 1 : index + 1] = [merged]
            index = max(0, index - 1)
            continue
        merged = _merge_segments(
            segment,
            resolved[index + 1],
            subtitle_min_duration_s=subtitle_min_duration_s,
            action="merge_micro_into_right",
            reason="right_split_weaker",
        )
        resolved[index : index + 2] = [merged]
    return [
        segment
        if segment.subtitle_min_duration_s is not None
        else _with_subtitle_metadata(segment, subtitle_min_duration_s=subtitle_min_duration_s)
        for segment in resolved
    ]


def _attach_split_proposals(
    segments: Iterable[SpeechSegment],
    *,
    speech_probs: np.ndarray,
    split_probs: np.ndarray,
    config: "SpeechBoundaryJaConfig",
) -> list[SpeechSegment]:
    result: list[SpeechSegment] = []
    subtitle_min_duration_s = _subtitle_min_duration_s_for_config(config)
    for segment in segments:
        all_candidates = _split_peak_candidates_for_segment(
            segment,
            speech_probs=speech_probs,
            split_probs=split_probs,
            frame_hop_s=config.frame_hop_s,
            split_smooth_s=config.split_smooth_s,
            split_snap_s=0.0,
            min_split_segment_s=config.min_split_segment_s,
        )
        nms_frames = max(1, int(round(max(0.0, config.split_nms_s) / config.frame_hop_s)))
        score_floor = _quantile(
            (candidate.score for candidate in all_candidates),
            config.split_score_quantile,
        )
        prominence_floor = _quantile(
            (candidate.prominence for candidate in all_candidates),
            config.split_prominence_quantile,
        )
        proposal_pool = [
            candidate
            for candidate in all_candidates
            if candidate.score >= score_floor - 1e-6
            and candidate.prominence >= prominence_floor - 1e-6
        ]
        proposals = sorted(
            _select_peak_candidates(
                proposal_pool,
                nms_frames=nms_frames,
                rank_by_prominence=True,
            ),
            key=lambda item: int(item.frame),
        )
        result.append(
            _segment_with_split_metadata(
                start=segment.start,
                end=segment.end,
                score=segment.score,
                subtitle_min_duration_s=subtitle_min_duration_s,
                left=None,
                right=None,
                weak_cut_candidates=[
                    {
                        **_cut_candidate_payload(
                            candidate,
                            frame_hop_s=config.frame_hop_s,
                            kind="proposal",
                        ),
                        "proposal_time_s": round(
                            float(candidate.frame) * config.frame_hop_s,
                            6,
                        ),
                    }
                    for candidate in proposals
                ],
            )
        )
    return result


def _subtitle_min_duration_s_for_config(config: "SpeechBoundaryJaConfig") -> float:
    del config
    return 20.0 / BASE_FPS


def _range_normalize(values: np.ndarray, *, lower_pct: float = 20.0, upper_pct: float = 95.0) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return data
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lower = float(np.percentile(finite, lower_pct))
    upper = float(np.percentile(finite, upper_pct))
    if upper <= lower + 1e-6:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def _frame_rms_db(
    audio: np.ndarray,
    *,
    sample_rate: int,
    frame_count: int,
    frame_hop_s: float,
    n_fft: int,
) -> np.ndarray:
    hop = max(1, int(round(frame_hop_s * sample_rate)))
    window = max(1, int(n_fft))
    samples = np.asarray(audio, dtype=np.float32)
    out = np.zeros(frame_count, dtype=np.float32)
    for index in range(frame_count):
        start = index * hop
        end = min(samples.shape[0], start + window)
        if end <= start:
            out[index] = -80.0
            continue
        rms = float(np.sqrt(np.mean(np.square(samples[start:end]), dtype=np.float64)))
        out[index] = 20.0 * np.log10(max(rms, 1e-8))
    return out


def _bootstrap_frame_scores(
    *,
    audio: np.ndarray,
    sample_rate: int,
    ptm: np.ndarray,
    mfcc: np.ndarray,
    config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    frame_total = min(int(ptm.shape[0]), int(mfcc.shape[0]))
    if frame_total <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty

    energy = _range_normalize(
        _frame_rms_db(
            audio,
            sample_rate=sample_rate,
            frame_count=frame_total,
            frame_hop_s=config.frame_hop_s,
            n_fft=config.n_fft,
        )
    )
    ptm_norm = _range_normalize(np.linalg.norm(ptm[:frame_total], axis=1))
    if mfcc.shape[0] > 1:
        delta = np.zeros(frame_total, dtype=np.float32)
        delta[1:] = np.mean(np.abs(np.diff(mfcc[:frame_total], axis=0)), axis=1)
        mfcc_delta = _range_normalize(delta)
    else:
        mfcc_delta = np.zeros(frame_total, dtype=np.float32)

    speech = (0.70 * energy + 0.20 * ptm_norm + 0.10 * mfcc_delta).astype(np.float32)
    if speech.size >= 3:
        smoothed = np.pad(speech, (1, 1), mode="edge")
        speech = ((smoothed[:-2] + smoothed[1:-1] + smoothed[2:]) / 3.0).astype(np.float32)
    candidate = (
        0.45 * (1.0 - energy)
        + 0.25 * (1.0 - speech)
        + 0.30 * mfcc_delta
    ).astype(np.float32)
    return np.clip(speech, 0.0, 1.0), np.clip(candidate, 0.0, 1.0)


def frames_to_segments(
    frames: Iterable[int],
    *,
    frame_hop_s: float,
    duration_s: float,
    scores: np.ndarray | None = None,
) -> list[SpeechSegment]:
    values = [1 if int(value) else 0 for value in frames]
    segments: list[SpeechSegment] = []
    start_index: int | None = None
    for index, value in enumerate(values + [0]):
        if value and start_index is None:
            start_index = index
        if not value and start_index is not None:
            start = max(0.0, min(float(start_index) * frame_hop_s, duration_s))
            end = max(0.0, min(float(index) * frame_hop_s, duration_s))
            score = None
            if scores is not None and index > start_index:
                score = float(np.max(scores[start_index:index]))
            if end > start:
                segments.append(SpeechSegment(start=start, end=end, score=score))
            start_index = None
    return segments


def filter_segments(
    segments: Iterable[SpeechSegment],
    *,
    duration_s: float,
    min_segment_s: float,
) -> list[SpeechSegment]:
    normalized: list[SpeechSegment] = []
    for segment in segments:
        start = max(0.0, min(float(segment.start), duration_s))
        end = max(0.0, min(float(segment.end), duration_s))
        if end - start < min_segment_s:
            continue
        below_subtitle_min_duration = segment.below_subtitle_min_duration
        if segment.subtitle_min_duration_s is not None:
            below_subtitle_min_duration = end - start < float(segment.subtitle_min_duration_s)
        normalized.append(
            replace(
                segment,
                start=start,
                end=end,
                below_subtitle_min_duration=below_subtitle_min_duration,
            )
        )
    return sorted(normalized, key=lambda item: (item.start, item.end))


@dataclass(frozen=True)
class FrameBoundaryDecodeResult:
    segments: list[SpeechSegment]
    coarse_segments: list[SpeechSegment]
    raw_frames: np.ndarray
    dilated_frames: np.ndarray
    speech_on_threshold: float
    speech_off_threshold: float


def _speech_thresholds_for_config(
    config: "SpeechBoundaryJaConfig",
    *,
    threshold_override: float | None = None,
) -> tuple[float, float]:
    if threshold_override is not None:
        threshold = float(threshold_override)
        return threshold, threshold
    fallback = float(config.threshold)
    speech_on_threshold = (
        fallback
        if config.speech_on_threshold is None
        else float(config.speech_on_threshold)
    )
    speech_off_threshold = (
        fallback
        if config.speech_off_threshold is None
        else float(config.speech_off_threshold)
    )
    if speech_on_threshold < 0.0:
        raise ValueError("SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD must be non-negative")
    if speech_off_threshold < 0.0:
        raise ValueError("SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD must be non-negative")
    if speech_on_threshold < speech_off_threshold:
        raise ValueError(
            "SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD must be greater than or equal to "
            "SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD"
        )
    return speech_on_threshold, speech_off_threshold


def decode_speech_island_segments(
    *,
    speech_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    duration_s: float,
    config: "SpeechBoundaryJaConfig",
    threshold_override: float | None = None,
) -> FrameBoundaryDecodeResult:
    """Decode speech-only frame scores and attach non-binding cut proposals."""

    speech_on_threshold, speech_off_threshold = _speech_thresholds_for_config(
        config,
        threshold_override=threshold_override,
    )
    probabilities = np.asarray(speech_probabilities, dtype=np.float32).reshape(-1)
    candidate_probs = np.asarray(candidate_probabilities, dtype=np.float32).reshape(-1)
    raw_frames = _hysteresis_frames(
        probabilities,
        on_threshold=speech_on_threshold,
        off_threshold=speech_off_threshold,
    )
    dilated = _dilated_frames(
        raw_frames,
        dilation_frames=max(0, int(round(config.frame_dilation_s / config.frame_hop_s))),
    )
    coarse_segments = frames_to_segments(
        dilated,
        frame_hop_s=config.frame_hop_s,
        duration_s=duration_s,
        scores=probabilities,
    )
    coarse_segments = filter_segments(
        coarse_segments,
        duration_s=duration_s,
        min_segment_s=config.min_segment_s,
    )
    segments = _attach_split_proposals(
        coarse_segments,
        speech_probs=probabilities,
        split_probs=candidate_probs,
        config=config,
    )
    segments = filter_segments(
        segments,
        duration_s=duration_s,
        min_segment_s=config.min_segment_s,
    )
    return FrameBoundaryDecodeResult(
        segments=segments,
        coarse_segments=coarse_segments,
        raw_frames=raw_frames,
        dilated_frames=dilated,
        speech_on_threshold=speech_on_threshold,
        speech_off_threshold=speech_off_threshold,
    )


@dataclass(frozen=True)
class SpeechBoundaryJaConfig:
    threshold: float = 0.15
    speech_on_threshold: float | None = None
    speech_off_threshold: float | None = None
    frame_dilation_s: float = 0.2
    frame_hop_s: float = 0.02
    ptm: str = DEFAULT_PTM
    model_path: str = DEFAULT_MODEL_PATH
    device: str = "auto"
    dtype: str = "bfloat16"
    attention: str = "sdpa"
    window_s: float = 30.0
    overlap_s: float = 5.0
    min_segment_s: float = 0.05
    split_smooth_s: float = 0.08
    split_nms_s: float = 0.20
    split_snap_s: float = 0.10
    min_split_segment_s: float = 0.08
    split_score_quantile: float = 0.50
    split_prominence_quantile: float = 0.50
    export_sequence_features: bool = False
    sequence_feature_max_ptm_dims: int = 128
    no_download: bool = False
    scorer_checkpoint: str = ""
    scorer_checkpoint_repo_id: str = ""
    scorer_device: str = "auto"

    @classmethod
    def from_env(cls) -> "SpeechBoundaryJaConfig":
        ptm = os.getenv("SPEECH_BOUNDARY_JA_PTM", "").strip() or current_qwen_asr_backend()
        ptm = qwen_asr_repo_id(ptm)
        model_path = os.getenv("SPEECH_BOUNDARY_JA_MODEL_PATH", "").strip() or qwen_asr_default_model_path(ptm)
        scorer_checkpoint = _scorer_checkpoint_from_env(ptm)
        return cls(
            threshold=_env_float("SPEECH_BOUNDARY_JA_THRESHOLD", "0.15"),
            speech_on_threshold=_env_optional_float("SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD"),
            speech_off_threshold=_env_optional_float("SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD"),
            frame_dilation_s=_env_float("SPEECH_BOUNDARY_JA_FRAME_DILATION_S", "0.2"),
            frame_hop_s=_env_float("SPEECH_BOUNDARY_JA_FRAME_HOP_S", "0.02"),
            ptm=ptm,
            model_path=model_path,
            device=os.getenv("SPEECH_BOUNDARY_JA_DEVICE", "auto").strip() or "auto",
            dtype=os.getenv("SPEECH_BOUNDARY_JA_DTYPE", "bfloat16").strip() or "bfloat16",
            attention=os.getenv("SPEECH_BOUNDARY_JA_ATTENTION", "sdpa").strip() or "sdpa",
            window_s=_env_float("SPEECH_BOUNDARY_JA_WINDOW_S", "30.0"),
            overlap_s=_env_float("SPEECH_BOUNDARY_JA_OVERLAP_S", "5.0"),
            min_segment_s=_env_float("SPEECH_BOUNDARY_JA_MIN_SEGMENT_S", "0.05"),
            split_smooth_s=_env_float("SPEECH_BOUNDARY_JA_SPLIT_SMOOTH_S", "0.08"),
            split_nms_s=_env_float("SPEECH_BOUNDARY_JA_SPLIT_NMS_S", "0.20"),
            split_snap_s=_env_float("SPEECH_BOUNDARY_JA_SPLIT_SNAP_S", "0.10"),
            min_split_segment_s=_env_float("SPEECH_BOUNDARY_JA_MIN_SPLIT_SEGMENT_S", "0.08"),
            split_score_quantile=_env_float(
                "SPEECH_BOUNDARY_JA_SPLIT_SCORE_QUANTILE",
                "0.50",
            ),
            split_prominence_quantile=_env_float(
                "SPEECH_BOUNDARY_JA_SPLIT_PROMINENCE_QUANTILE",
                "0.50",
            ),
            export_sequence_features=_env_bool("SPEECH_BOUNDARY_JA_EXPORT_SEQUENCE_FEATURES", "0"),
            sequence_feature_max_ptm_dims=max(
                1,
                int(_env_float("BOUNDARY_FRAME_SEQUENCE_MAX_PTM_DIMS", "128")),
            ),
            no_download=_env_bool("SPEECH_BOUNDARY_JA_NO_DOWNLOAD", "0"),
            scorer_checkpoint=scorer_checkpoint,
            scorer_checkpoint_repo_id=ptm if scorer_checkpoint else "",
            scorer_device=os.getenv("SPEECH_BOUNDARY_JA_SCORER_DEVICE", "auto").strip() or "auto",
        )


class SpeechBoundaryJaBackend:
    name = "speech_boundary_ja_mamba2_speech_island_scorer_v8"

    def __init__(self, config: SpeechBoundaryJaConfig | None = None) -> None:
        self.config = config or SpeechBoundaryJaConfig.from_env()

    @staticmethod
    def _speech_thresholds(
        config: SpeechBoundaryJaConfig,
        *,
        threshold_override: float | None = None,
    ) -> tuple[float, float]:
        return _speech_thresholds_for_config(config, threshold_override=threshold_override)

    def signature(self) -> dict:
        cfg = self.config
        speech_on_threshold, speech_off_threshold = self._speech_thresholds(cfg)
        signature = {
            "backend": self.name,
            "schema": "speech_boundary_ja_mamba2_speech_island_scorer_v8",
            "threshold": float(cfg.threshold),
            "speech_threshold_mode": "hysteresis",
            "speech_on_threshold": float(speech_on_threshold),
            "speech_off_threshold": float(speech_off_threshold),
            "frame_dilation_s": float(cfg.frame_dilation_s),
            "frame_hop_s": float(cfg.frame_hop_s),
            "ptm": cfg.ptm,
            "model_path": cfg.model_path,
            "device": cfg.device,
            "dtype": cfg.dtype,
            "attention": cfg.attention,
            "window_s": float(cfg.window_s),
            "overlap_s": float(cfg.overlap_s),
            "min_segment_s": float(cfg.min_segment_s),
            "split_strategy": "candidate_proposal_only",
            "split_smooth_s": float(cfg.split_smooth_s),
            "split_nms_s": float(cfg.split_nms_s),
            "split_snap_s": float(cfg.split_snap_s),
            "min_split_segment_s": float(cfg.min_split_segment_s),
            "split_score_quantile": float(cfg.split_score_quantile),
            "split_prominence_quantile": float(cfg.split_prominence_quantile),
            "split_decision": "external_semantic_split_model_v1",
            "base_fps": float(BASE_FPS),
            "micro_chunk_min_duration_s": float(_subtitle_min_duration_s_for_config(cfg)),
            "export_sequence_features": bool(cfg.export_sequence_features),
            "sequence_feature_max_ptm_dims": int(cfg.sequence_feature_max_ptm_dims),
            "scorer_checkpoint": "",
            "operating_point": DEFAULT_OPERATING_POINT,
            "allow_empty": True,
        }
        scorer_checkpoint = cfg.scorer_checkpoint.strip()
        if scorer_checkpoint:
            signature["scorer_checkpoint"] = scorer_checkpoint
            signature["scorer_checkpoint_repo_id"] = cfg.scorer_checkpoint_repo_id or cfg.ptm
            signature["scorer_device"] = cfg.scorer_device
        return signature

    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        del target_sr
        started = time.perf_counter()
        cfg = self.config
        if cfg.window_s <= 0.0:
            raise ValueError("SPEECH_BOUNDARY_JA_WINDOW_S must be positive")
        if cfg.overlap_s < 0.0:
            raise ValueError("SPEECH_BOUNDARY_JA_OVERLAP_S must be non-negative")
        if cfg.overlap_s >= cfg.window_s:
            raise ValueError("SPEECH_BOUNDARY_JA_OVERLAP_S must be smaller than SPEECH_BOUNDARY_JA_WINDOW_S")

        import torch

        device = _model_device(cfg.device)
        scorer_device = _model_device(cfg.scorer_device)
        scorer = (
            load_speech_island_scorer_checkpoint(cfg.scorer_checkpoint, device=scorer_device)
            if cfg.scorer_checkpoint.strip()
            else None
        )
        if scorer is not None:
            _validate_scorer_checkpoint_repo(scorer, cfg.scorer_checkpoint_repo_id or cfg.ptm)
        scorer_signature = scorer.signature() if scorer is not None else None
        speech_on_threshold, speech_off_threshold = self._speech_thresholds(
            cfg,
            threshold_override=threshold_override,
        )
        feature_config = FeatureConfig(
            ptm=cfg.ptm,
            frame_hop_s=cfg.frame_hop_s,
            n_mfcc=40,
            n_fft=400,
            device=str(device),
            dtype=cfg.dtype,
            model_path=cfg.model_path,
            download=not cfg.no_download,
            attention=cfg.attention,
            language="Japanese",
        )
        ptm_extractor = build_ptm_feature_extractor(feature_config)
        ptm_param_device, ptm_param_dtype = _first_parameter_device_dtype(
            getattr(ptm_extractor, "model", None)
        )
        runtime_device = {
            "requested_device": cfg.device,
            "actual_device": str(device),
            "dtype": cfg.dtype,
            "ptm_param_device": ptm_param_device,
            "ptm_param_dtype": ptm_param_dtype,
            "score_model": (
                "mamba2_speech_island_scorer_v8" if scorer is not None else "bootstrap_energy_ptm_mfcc"
            ),
            "scorer_device": str(scorer_device) if scorer is not None else "",
        }
        print(
            "[boundary] speech_boundary_ja device "
            f"requested_device={runtime_device['requested_device']} "
            f"actual_device={runtime_device['actual_device']} "
            f"dtype={runtime_device['dtype']} "
            f"ptm_param_device={runtime_device['ptm_param_device']} "
            f"ptm_param_dtype={runtime_device['ptm_param_dtype']} "
            f"score_model={runtime_device['score_model']}",
            flush=True,
        )
        try:
            audio, sample_rate = load_audio_16k_mono(audio_path)
            duration_s = float(len(audio) / sample_rate) if sample_rate else 0.0
            total_frames = frame_count(duration_s, cfg.frame_hop_s)
            probability_sum = np.zeros(total_frames, dtype=np.float64)
            probability_count = np.zeros(total_frames, dtype=np.float32)
            split_probability_sum = np.zeros(total_frames, dtype=np.float64)
            split_probability_count = np.zeros(total_frames, dtype=np.float32)
            sequence_ptm_sum: np.ndarray | None = None
            sequence_mfcc_sum: np.ndarray | None = None
            sequence_feature_count: np.ndarray | None = None
            window_samples = max(1, int(round(cfg.window_s * sample_rate)))
            stride_samples = max(1, int(round((cfg.window_s - cfg.overlap_s) * sample_rate)))
            starts = list(range(0, max(1, len(audio)), stride_samples))

            for window_index, start_sample in enumerate(starts):
                end_sample = min(len(audio), start_sample + window_samples)
                if start_sample >= end_sample:
                    continue
                chunk = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
                mfcc = extract_mfcc(chunk, sample_rate=sample_rate, config=feature_config)
                ptm = ptm_extractor.extract(chunk, sample_rate=sample_rate)
                ptm, mfcc = align_feature_frames(
                    ptm,
                    mfcc,
                    resize_ptm=is_low_frame_rate_ptm(cfg.ptm),
                )
                if scorer is None:
                    probs, candidate_probs = _bootstrap_frame_scores(
                        audio=chunk,
                        sample_rate=sample_rate,
                        ptm=ptm,
                        mfcc=mfcc,
                        config=feature_config,
                    )
                else:
                    probs = score_speech_island_probabilities(
                        scorer,
                        ptm=ptm,
                        mfcc=mfcc,
                    )
                    _, candidate_probs = _bootstrap_frame_scores(
                        audio=chunk,
                        sample_rate=sample_rate,
                        ptm=ptm,
                        mfcc=mfcc,
                        config=feature_config,
                    )
                window_start_s = start_sample / sample_rate
                global_start = max(0, int(round(window_start_s / cfg.frame_hop_s)))
                global_end = min(total_frames, global_start + probs.size)
                local_end = max(0, global_end - global_start)
                if local_end <= 0:
                    continue
                probability_sum[global_start:global_end] += probs[:local_end]
                probability_count[global_start:global_end] += 1.0
                split_probability_sum[global_start:global_end] += candidate_probs[:local_end]
                split_probability_count[global_start:global_end] += 1.0
                if cfg.export_sequence_features:
                    ptm_dim = min(int(ptm.shape[1]), int(cfg.sequence_feature_max_ptm_dims))
                    mfcc_dim = int(mfcc.shape[1])
                    if sequence_ptm_sum is None:
                        sequence_ptm_sum = np.zeros((total_frames, ptm_dim), dtype=np.float64)
                        sequence_mfcc_sum = np.zeros((total_frames, mfcc_dim), dtype=np.float64)
                        sequence_feature_count = np.zeros(total_frames, dtype=np.float32)
                    sequence_ptm_sum[global_start:global_end] += ptm[:local_end, :ptm_dim]
                    sequence_mfcc_sum[global_start:global_end] += mfcc[:local_end, :mfcc_dim]
                    sequence_feature_count[global_start:global_end] += 1.0
                print(
                    "[boundary] speech_boundary_ja window "
                    f"{window_index + 1}/{len(starts)} start={window_start_s:.1f}s "
                    f"frames={local_end}",
                    flush=True,
                )

            probabilities = np.divide(
                probability_sum,
                np.maximum(probability_count, 1.0),
                out=np.zeros_like(probability_sum, dtype=np.float64),
                where=probability_count > 0,
            ).astype(np.float32)
            candidate_probabilities = np.divide(
                split_probability_sum,
                np.maximum(split_probability_count, 1.0),
                out=np.zeros_like(split_probability_sum, dtype=np.float64),
                where=split_probability_count > 0,
            ).astype(np.float32)
            decode = decode_speech_island_segments(
                speech_probabilities=probabilities,
                candidate_probabilities=candidate_probabilities,
                duration_s=duration_s,
                config=cfg,
                threshold_override=threshold_override,
            )
            raw_frames = decode.raw_frames
            dilated = decode.dilated_frames
            coarse_segments = decode.coarse_segments
            segments = decode.segments
            groups = [[segment] for segment in segments]
            params = self.signature()
            params.update(
                {
                    "audio_stats": {
                        "duration_s": duration_s,
                        "frames": int(total_frames),
                        "windows": len(starts),
                        "speech_threshold_mode": "hysteresis",
                        "speech_on_threshold": float(speech_on_threshold),
                        "speech_off_threshold": float(speech_off_threshold),
                        "probability_mean": float(probabilities.mean()) if probabilities.size else 0.0,
                        "probability_max": float(probabilities.max()) if probabilities.size else 0.0,
                        "candidate_probability_mean": (
                            float(candidate_probabilities.mean()) if candidate_probabilities.size else 0.0
                        ),
                        "candidate_probability_max": (
                            float(candidate_probabilities.max()) if candidate_probabilities.size else 0.0
                        ),
                        "raw_speech_ratio": float(raw_frames.mean()) if raw_frames.size else 0.0,
                        "dilated_speech_ratio": float(dilated.mean()) if dilated.size else 0.0,
                        "coarse_segment_count": len(coarse_segments),
                        "speech_island_count": len(segments),
                        "uncovered_frame_ratio": float((probability_count <= 0).mean())
                        if probability_count.size
                        else 0.0,
                    },
                    "runtime_device": runtime_device,
                }
            )
            if scorer_signature is not None:
                params["scorer_checkpoint"] = scorer_signature
            if (
                cfg.export_sequence_features
                and sequence_ptm_sum is not None
                and sequence_mfcc_sum is not None
                and sequence_feature_count is not None
            ):
                counts = np.maximum(sequence_feature_count.reshape(-1, 1), 1.0)
                sequence_ptm = (sequence_ptm_sum / counts).astype(np.float32)
                sequence_mfcc = (sequence_mfcc_sum / counts).astype(np.float32)
                params["sequence_feature_frames"] = {
                    "schema": "speech_boundary_ja_sequence_feature_frames_v1",
                    "frame_hop_s": float(cfg.frame_hop_s),
                    "ptm": sequence_ptm.tolist(),
                    "mfcc": sequence_mfcc.tolist(),
                    "ptm_dim": int(sequence_ptm.shape[1]),
                    "mfcc_dim": int(sequence_mfcc.shape[1]),
                }
            if _env_bool("SPEECH_BOUNDARY_JA_EXPORT_FRAME_SCORES", "0") or cfg.export_sequence_features:
                params["frame_scores"] = [float(value) for value in probabilities]
                params["candidate_frame_scores"] = [float(value) for value in candidate_probabilities]
            return SegmentationResult(
                segments=segments,
                groups=groups,
                method=self.name,
                audio_duration_sec=duration_s,
                parameters=params,
                processing_time_sec=time.perf_counter() - started,
            )
        finally:
            close = getattr(ptm_extractor, "close", None)
            if callable(close):
                close()
            if device.type == "cuda":
                torch.cuda.empty_cache()
