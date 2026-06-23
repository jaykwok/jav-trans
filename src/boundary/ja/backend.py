from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable

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
    load_feature_frame_scorer_checkpoint,
    score_feature_frame_boundary_probabilities,
)


DEFAULT_PTM = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame"
DEFAULT_MODEL_PATH = qwen_asr_default_model_path(DEFAULT_PTM)
DEFAULT_OPERATING_POINT = "qwen-mamba2-frame-boundary-scorer-v4"


@dataclass(frozen=True)
class _SplitPeakCandidate:
    frame: int
    score: float
    prominence: float


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
    legacy = os.getenv("SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT", "").strip()
    if legacy:
        raise RuntimeError(
            "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT is no longer supported. "
            "Use SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO="
            f"{ptm}=path/to/speech_boundary_ja_frame_boundary_scorer_v4.pt"
        )
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


def _apply_drop_gap_mask(
    speech_frames: np.ndarray,
    drop_gap_probs: np.ndarray,
    *,
    drop_gap_threshold: float,
) -> np.ndarray:
    frames = np.asarray(speech_frames, dtype=np.int8).copy()
    if frames.size == 0 or drop_gap_probs.size == 0:
        return frames
    total = min(int(frames.size), int(drop_gap_probs.size))
    active = frames[:total]
    active[np.asarray(drop_gap_probs[:total], dtype=np.float32) >= drop_gap_threshold] = 0
    return frames


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
    drop_gap_probs: np.ndarray,
    snap_frames: int,
) -> int:
    lower = max(start_frame + 1, int(peak_frame) - max(0, int(snap_frames)))
    upper = min(end_frame - 1, int(peak_frame) + max(0, int(snap_frames)))
    if upper < lower:
        return int(peak_frame)
    speech = np.asarray(speech_probs, dtype=np.float32)
    drop = np.asarray(drop_gap_probs, dtype=np.float32)
    total = min(int(speech.size), int(drop.size))
    lower = min(lower, max(0, total - 1))
    upper = min(upper, max(0, total - 1))
    if upper < lower:
        return int(peak_frame)
    candidates = list(range(lower, upper + 1))
    return min(
        candidates,
        key=lambda frame: (
            float(speech[frame]) - float(drop[frame]),
            abs(int(frame) - int(peak_frame)),
        ),
    )


def _split_peak_candidates_for_segment(
    segment: SpeechSegment,
    *,
    speech_probs: np.ndarray,
    split_probs: np.ndarray,
    drop_gap_probs: np.ndarray,
    frame_hop_s: float,
    split_smooth_s: float,
    split_snap_s: float,
    min_split_segment_s: float,
) -> list[_SplitPeakCandidate]:
    total = min(int(speech_probs.size), int(split_probs.size), int(drop_gap_probs.size))
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
            drop_gap_probs=drop_gap_probs[:total],
            snap_frames=max(0, int(round(max(0.0, split_snap_s) / frame_hop_s))),
        )
        candidates.append(_SplitPeakCandidate(frame=snapped, score=value, prominence=prominence))
    return candidates


def _select_peak_frames(
    candidates: Iterable[_SplitPeakCandidate],
    *,
    nms_frames: int,
    max_count: int | None = None,
    rank_by_prominence: bool = False,
) -> list[int]:
    selected: list[int] = []
    ranked = sorted(
        candidates,
        key=(
            (lambda item: (item.prominence, item.score))
            if rank_by_prominence
            else (lambda item: (item.score, item.prominence))
        ),
        reverse=True,
    )
    added: list[int] = []
    for candidate in ranked:
        if max_count is not None and len(added) >= max_count:
            break
        frame = int(candidate.frame)
        if all(abs(frame - existing) >= nms_frames for existing in selected):
            selected.append(frame)
            added.append(frame)
    return added


def _split_peak_frames_for_segment(
    segment: SpeechSegment,
    *,
    speech_probs: np.ndarray,
    split_probs: np.ndarray,
    drop_gap_probs: np.ndarray,
    frame_hop_s: float,
    split_smooth_s: float,
    split_nms_s: float,
    split_snap_s: float,
    min_split_segment_s: float,
    split_target_s: float,
    split_score_quantile: float,
    split_prominence_quantile: float,
) -> list[int]:
    candidates = _split_peak_candidates_for_segment(
        segment,
        speech_probs=speech_probs,
        split_probs=split_probs,
        drop_gap_probs=drop_gap_probs,
        frame_hop_s=frame_hop_s,
        split_smooth_s=split_smooth_s,
        split_snap_s=split_snap_s,
        min_split_segment_s=min_split_segment_s,
    )
    if not candidates:
        return []
    target_s = max(0.0, float(split_target_s))
    if target_s <= 0.0:
        return []
    segment_duration = max(0.0, float(segment.end) - float(segment.start))
    budget = max(0, int(np.ceil(segment_duration / target_s)) - 1)
    if budget <= 0:
        return []
    nms_frames = max(1, int(round(max(0.0, split_nms_s) / frame_hop_s)))
    score_floor = _quantile((candidate.score for candidate in candidates), split_score_quantile)
    prominence_floor = _quantile(
        (candidate.prominence for candidate in candidates),
        split_prominence_quantile,
    )
    adaptive_candidates = [
        candidate
        for candidate in candidates
        if candidate.score >= score_floor and candidate.prominence >= prominence_floor
    ]
    return sorted(
        _select_peak_frames(
            adaptive_candidates,
            nms_frames=nms_frames,
            max_count=budget,
            rank_by_prominence=True,
        )
    )


def _split_segment_at_frames(
    segment: SpeechSegment,
    split_frames: Iterable[int],
    *,
    frame_hop_s: float,
    min_split_segment_s: float,
) -> list[SpeechSegment]:
    boundaries = [float(frame) * frame_hop_s for frame in sorted(set(int(frame) for frame in split_frames))]
    parts: list[SpeechSegment] = []
    cursor = float(segment.start)
    min_duration = max(0.0, float(min_split_segment_s))
    for boundary in boundaries:
        boundary = max(segment.start, min(float(boundary), segment.end))
        if boundary - cursor < min_duration:
            continue
        if segment.end - boundary < min_duration:
            continue
        parts.append(SpeechSegment(start=cursor, end=boundary, score=segment.score))
        cursor = boundary
    if segment.end > cursor:
        parts.append(SpeechSegment(start=cursor, end=segment.end, score=segment.score))
    return parts or [segment]


def _split_segments_by_peaks(
    segments: Iterable[SpeechSegment],
    *,
    speech_probs: np.ndarray,
    split_probs: np.ndarray,
    drop_gap_probs: np.ndarray,
    config: "SpeechBoundaryJaConfig",
) -> list[SpeechSegment]:
    result: list[SpeechSegment] = []
    for segment in segments:
        peaks = _split_peak_frames_for_segment(
            segment,
            speech_probs=speech_probs,
            split_probs=split_probs,
            drop_gap_probs=drop_gap_probs,
            frame_hop_s=config.frame_hop_s,
            split_smooth_s=config.split_smooth_s,
            split_nms_s=config.split_nms_s,
            split_snap_s=config.split_snap_s,
            min_split_segment_s=config.min_split_segment_s,
            split_target_s=config.split_target_s,
            split_score_quantile=config.split_score_quantile,
            split_prominence_quantile=config.split_prominence_quantile,
        )
        result.extend(
            _split_segment_at_frames(
                segment,
                peaks,
                frame_hop_s=config.frame_hop_s,
                min_split_segment_s=config.min_split_segment_s,
            )
        )
    return result


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_total = min(int(ptm.shape[0]), int(mfcc.shape[0]))
    if frame_total <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty

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
    split = np.zeros_like(speech, dtype=np.float32)
    drop_gap = (1.0 - speech).astype(np.float32)
    return np.clip(speech, 0.0, 1.0), split, np.clip(drop_gap, 0.0, 1.0)


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
    return [
        segment
        for segment in sorted(
            (
                SpeechSegment(
                    start=max(0.0, min(float(segment.start), duration_s)),
                    end=max(0.0, min(float(segment.end), duration_s)),
                    score=segment.score,
                )
                for segment in segments
            ),
            key=lambda item: (item.start, item.end),
        )
        if segment.end - segment.start >= min_segment_s
    ]


@dataclass(frozen=True)
class FrameBoundaryDecodeResult:
    segments: list[SpeechSegment]
    coarse_segments: list[SpeechSegment]
    raw_frames: np.ndarray
    dilated_frames: np.ndarray
    gap_masked_frames: np.ndarray
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


def decode_frame_boundary_segments(
    *,
    speech_probabilities: np.ndarray,
    split_probabilities: np.ndarray,
    drop_gap_probabilities: np.ndarray,
    duration_s: float,
    config: "SpeechBoundaryJaConfig",
    threshold_override: float | None = None,
) -> FrameBoundaryDecodeResult:
    """Decode scorer v4 frame heads into speech islands.

    This is intentionally shared by runtime and Boundary Refiner v6 dataset
    export so the refiner learns from exactly the same island contract it sees
    during inference.
    """

    speech_on_threshold, speech_off_threshold = _speech_thresholds_for_config(
        config,
        threshold_override=threshold_override,
    )
    probabilities = np.asarray(speech_probabilities, dtype=np.float32).reshape(-1)
    split_probs = np.asarray(split_probabilities, dtype=np.float32).reshape(-1)
    drop_gap_probs = np.asarray(drop_gap_probabilities, dtype=np.float32).reshape(-1)
    raw_frames = _hysteresis_frames(
        probabilities,
        on_threshold=speech_on_threshold,
        off_threshold=speech_off_threshold,
    )
    dilated = _dilated_frames(
        raw_frames,
        dilation_frames=max(0, int(round(config.frame_dilation_s / config.frame_hop_s))),
    )
    gap_masked = _apply_drop_gap_mask(
        dilated,
        drop_gap_probs,
        drop_gap_threshold=config.drop_gap_threshold,
    )
    coarse_segments = frames_to_segments(
        gap_masked,
        frame_hop_s=config.frame_hop_s,
        duration_s=duration_s,
        scores=probabilities,
    )
    coarse_segments = filter_segments(
        coarse_segments,
        duration_s=duration_s,
        min_segment_s=config.min_segment_s,
    )
    split_segments = _split_segments_by_peaks(
        coarse_segments,
        speech_probs=probabilities,
        split_probs=split_probs,
        drop_gap_probs=drop_gap_probs,
        config=config,
    )
    segments = filter_segments(
        split_segments,
        duration_s=duration_s,
        min_segment_s=config.min_segment_s,
    )
    return FrameBoundaryDecodeResult(
        segments=segments,
        coarse_segments=coarse_segments,
        raw_frames=raw_frames,
        dilated_frames=dilated,
        gap_masked_frames=gap_masked,
        speech_on_threshold=speech_on_threshold,
        speech_off_threshold=speech_off_threshold,
    )


@dataclass(frozen=True)
class SpeechBoundaryJaConfig:
    threshold: float = 0.5
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
    overlap_s: float = 1.0
    min_segment_s: float = 0.05
    drop_gap_threshold: float = 0.5
    split_target_s: float = 5.0
    split_smooth_s: float = 0.08
    split_nms_s: float = 0.20
    split_snap_s: float = 0.10
    min_split_segment_s: float = 0.08
    split_score_quantile: float = 0.50
    split_prominence_quantile: float = 0.50
    export_sequence_features: bool = False
    sequence_feature_max_ptm_dims: int = 64
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
            threshold=_env_float("SPEECH_BOUNDARY_JA_THRESHOLD", "0.5"),
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
            overlap_s=_env_float("SPEECH_BOUNDARY_JA_OVERLAP_S", "1.0"),
            min_segment_s=_env_float("SPEECH_BOUNDARY_JA_MIN_SEGMENT_S", "0.05"),
            drop_gap_threshold=_env_float("SPEECH_BOUNDARY_JA_DROP_GAP_THRESHOLD", "0.5"),
            split_target_s=_env_float("SPEECH_BOUNDARY_JA_SPLIT_TARGET_S", "5.0"),
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
                int(_env_float("BOUNDARY_FRAME_SEQUENCE_MAX_PTM_DIMS", "64")),
            ),
            no_download=_env_bool("SPEECH_BOUNDARY_JA_NO_DOWNLOAD", "0"),
            scorer_checkpoint=scorer_checkpoint,
            scorer_checkpoint_repo_id=ptm if scorer_checkpoint else "",
            scorer_device=os.getenv("SPEECH_BOUNDARY_JA_SCORER_DEVICE", "auto").strip() or "auto",
        )


class SpeechBoundaryJaBackend:
    name = "speech_boundary_ja_mamba2_frame_boundary_scorer_v4"

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
            "schema": "speech_boundary_ja_mamba2_frame_boundary_scorer_v4",
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
            "drop_gap_threshold": float(cfg.drop_gap_threshold),
            "split_strategy": "adaptive_topk_peak",
            "split_target_s": float(cfg.split_target_s),
            "split_smooth_s": float(cfg.split_smooth_s),
            "split_nms_s": float(cfg.split_nms_s),
            "split_snap_s": float(cfg.split_snap_s),
            "min_split_segment_s": float(cfg.min_split_segment_s),
            "split_score_quantile": float(cfg.split_score_quantile),
            "split_prominence_quantile": float(cfg.split_prominence_quantile),
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
            load_feature_frame_scorer_checkpoint(cfg.scorer_checkpoint, device=scorer_device)
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
                "mamba2_frame_boundary_scorer_v4" if scorer is not None else "bootstrap_energy_ptm_mfcc"
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
            drop_gap_probability_sum = np.zeros(total_frames, dtype=np.float64)
            drop_gap_probability_count = np.zeros(total_frames, dtype=np.float32)
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
                    probs, split_probs, drop_gap_probs = _bootstrap_frame_scores(
                        audio=chunk,
                        sample_rate=sample_rate,
                        ptm=ptm,
                        mfcc=mfcc,
                        config=feature_config,
                    )
                else:
                    probs, split_probs, drop_gap_probs = score_feature_frame_boundary_probabilities(
                        scorer,
                        ptm=ptm,
                        mfcc=mfcc,
                    )
                window_start_s = start_sample / sample_rate
                global_start = max(0, int(round(window_start_s / cfg.frame_hop_s)))
                global_end = min(total_frames, global_start + probs.size)
                local_end = max(0, global_end - global_start)
                if local_end <= 0:
                    continue
                probability_sum[global_start:global_end] += probs[:local_end]
                probability_count[global_start:global_end] += 1.0
                split_probability_sum[global_start:global_end] += split_probs[:local_end]
                split_probability_count[global_start:global_end] += 1.0
                drop_gap_probability_sum[global_start:global_end] += drop_gap_probs[:local_end]
                drop_gap_probability_count[global_start:global_end] += 1.0
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
            split_probabilities = np.divide(
                split_probability_sum,
                np.maximum(split_probability_count, 1.0),
                out=np.zeros_like(split_probability_sum, dtype=np.float64),
                where=split_probability_count > 0,
            ).astype(np.float32)
            drop_gap_probabilities = np.divide(
                drop_gap_probability_sum,
                np.maximum(drop_gap_probability_count, 1.0),
                out=np.zeros_like(drop_gap_probability_sum, dtype=np.float64),
                where=drop_gap_probability_count > 0,
            ).astype(np.float32)
            decode = decode_frame_boundary_segments(
                speech_probabilities=probabilities,
                split_probabilities=split_probabilities,
                drop_gap_probabilities=drop_gap_probabilities,
                duration_s=duration_s,
                config=cfg,
                threshold_override=threshold_override,
            )
            raw_frames = decode.raw_frames
            dilated = decode.dilated_frames
            gap_masked = decode.gap_masked_frames
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
                        "split_boundary_probability_mean": (
                            float(split_probabilities.mean()) if split_probabilities.size else 0.0
                        ),
                        "split_boundary_probability_max": (
                            float(split_probabilities.max()) if split_probabilities.size else 0.0
                        ),
                        "drop_gap_probability_mean": (
                            float(drop_gap_probabilities.mean()) if drop_gap_probabilities.size else 0.0
                        ),
                        "drop_gap_probability_max": (
                            float(drop_gap_probabilities.max()) if drop_gap_probabilities.size else 0.0
                        ),
                        "raw_speech_ratio": float(raw_frames.mean()) if raw_frames.size else 0.0,
                        "dilated_speech_ratio": float(dilated.mean()) if dilated.size else 0.0,
                        "drop_gap_masked_speech_ratio": float(gap_masked.mean()) if gap_masked.size else 0.0,
                        "coarse_segment_count": len(coarse_segments),
                        "split_segment_count": len(segments),
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
                params["split_boundary_frame_scores"] = [float(value) for value in split_probabilities]
                params["drop_gap_frame_scores"] = [float(value) for value in drop_gap_probabilities]
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
