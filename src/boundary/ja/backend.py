from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

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
from boundary.ja.postprocess import group_segments


DEFAULT_MODEL_PATH = "models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame"
DEFAULT_OPERATING_POINT = "qwen-feature-energy-bootstrap-v1"
DEFAULT_PTM = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame"


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


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


def _padded_frames(values: np.ndarray, *, pad_frames: int) -> np.ndarray:
    mask = np.asarray(values, dtype=bool)
    if pad_frames <= 0 or mask.size == 0:
        return mask.astype(np.int8, copy=False)
    out = mask.copy()
    active = np.flatnonzero(mask)
    for index in active:
        start = max(0, int(index) - pad_frames)
        end = min(out.size, int(index) + pad_frames + 1)
        out[start:end] = True
    return out.astype(np.int8, copy=False)


def _apply_cut_gate(
    speech_probs: np.ndarray,
    cut_probs: np.ndarray | None,
    *,
    cut_threshold: float,
    apply_cut: bool,
) -> np.ndarray:
    if not apply_cut or cut_probs is None:
        return speech_probs
    frame_total = min(int(speech_probs.size), int(cut_probs.size))
    if frame_total <= 0:
        return speech_probs
    gated = speech_probs.copy()
    active = gated[:frame_total]
    active[cut_probs[:frame_total] >= cut_threshold] = 0.0
    return gated


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
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

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
        padded = np.pad(speech, (1, 1), mode="edge")
        speech = ((padded[:-2] + padded[1:-1] + padded[2:]) / 3.0).astype(np.float32)
    cut = (1.0 - speech).astype(np.float32)
    return np.clip(speech, 0.0, 1.0), np.clip(cut, 0.0, 1.0)


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


def merge_segments(
    segments: Iterable[SpeechSegment],
    *,
    duration_s: float,
    merge_gap_s: float,
    min_segment_s: float,
) -> list[SpeechSegment]:
    ordered = sorted(
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
    merged: list[SpeechSegment] = []
    for segment in ordered:
        if segment.end - segment.start < min_segment_s:
            continue
        if not merged or segment.start - merged[-1].end > merge_gap_s:
            merged.append(segment)
            continue
        merged[-1].end = max(merged[-1].end, segment.end)
        if merged[-1].score is None:
            merged[-1].score = segment.score
        elif segment.score is not None:
            merged[-1].score = max(float(merged[-1].score), float(segment.score))
    return [segment for segment in merged if segment.end - segment.start >= min_segment_s]


@dataclass(frozen=True)
class SpeechBoundaryJaConfig:
    threshold: float = 0.200
    pad_s: float = 0.2
    frame_hop_s: float = 0.02
    ptm: str = DEFAULT_PTM
    model_path: str = DEFAULT_MODEL_PATH
    device: str = "auto"
    dtype: str = "bfloat16"
    attention: str = "sdpa"
    window_s: float = 30.0
    overlap_s: float = 1.0
    min_segment_s: float = 0.05
    merge_gap_s: float = 0.0
    max_group_s: float = 6.0
    chunk_threshold_s: float = 1.0
    cut_threshold: float = 0.500
    apply_cut_to_speech: bool = True
    export_sequence_features: bool = False
    sequence_feature_max_ptm_dims: int = 64
    no_download: bool = False

    @classmethod
    def from_env(cls) -> "SpeechBoundaryJaConfig":
        return cls(
            threshold=_env_float("SPEECH_BOUNDARY_JA_THRESHOLD", "0.200"),
            pad_s=_env_float("SPEECH_BOUNDARY_JA_PAD_S", "0.2"),
            frame_hop_s=_env_float("SPEECH_BOUNDARY_JA_FRAME_HOP_S", "0.02"),
            ptm=os.getenv("SPEECH_BOUNDARY_JA_PTM", DEFAULT_PTM).strip() or DEFAULT_PTM,
            model_path=os.getenv("SPEECH_BOUNDARY_JA_MODEL_PATH", DEFAULT_MODEL_PATH).strip(),
            device=os.getenv("SPEECH_BOUNDARY_JA_DEVICE", "auto").strip() or "auto",
            dtype=os.getenv("SPEECH_BOUNDARY_JA_DTYPE", "bfloat16").strip() or "bfloat16",
            attention=os.getenv("SPEECH_BOUNDARY_JA_ATTENTION", "sdpa").strip() or "sdpa",
            window_s=_env_float("SPEECH_BOUNDARY_JA_WINDOW_S", "30.0"),
            overlap_s=_env_float("SPEECH_BOUNDARY_JA_OVERLAP_S", "1.0"),
            min_segment_s=_env_float("SPEECH_BOUNDARY_JA_MIN_SEGMENT_S", "0.05"),
            merge_gap_s=_env_float("SPEECH_BOUNDARY_JA_MERGE_GAP_S", "0.0"),
            max_group_s=_env_float("SPEECH_BOUNDARY_JA_MAX_GROUP_S", "6.0"),
            chunk_threshold_s=_env_float("SPEECH_BOUNDARY_JA_CHUNK_THRESHOLD_S", "1.0"),
            cut_threshold=_env_float("SPEECH_BOUNDARY_JA_CUT_THRESHOLD", "0.500"),
            apply_cut_to_speech=_env_bool("SPEECH_BOUNDARY_JA_APPLY_CUT_TO_SPEECH", "1"),
            export_sequence_features=_env_bool("SPEECH_BOUNDARY_JA_EXPORT_SEQUENCE_FEATURES", "0"),
            sequence_feature_max_ptm_dims=max(
                1,
                int(_env_float("BOUNDARY_FRAME_SEQUENCE_MAX_PTM_DIMS", "64")),
            ),
            no_download=_env_bool("SPEECH_BOUNDARY_JA_NO_DOWNLOAD", "0"),
        )


class SpeechBoundaryJaBackend:
    name = "speech_boundary_ja_qwen_feature_bootstrap"

    def __init__(self, config: SpeechBoundaryJaConfig | None = None) -> None:
        self.config = config or SpeechBoundaryJaConfig.from_env()

    def signature(self) -> dict:
        cfg = self.config
        return {
            "backend": self.name,
            "threshold": float(cfg.threshold),
            "pad_s": float(cfg.pad_s),
            "frame_hop_s": float(cfg.frame_hop_s),
            "ptm": cfg.ptm,
            "model_path": cfg.model_path,
            "device": cfg.device,
            "dtype": cfg.dtype,
            "attention": cfg.attention,
            "window_s": float(cfg.window_s),
            "overlap_s": float(cfg.overlap_s),
            "min_segment_s": float(cfg.min_segment_s),
            "merge_gap_s": float(cfg.merge_gap_s),
            "max_group_s": float(cfg.max_group_s),
            "chunk_threshold_s": float(cfg.chunk_threshold_s),
            "cut_threshold": float(cfg.cut_threshold),
            "apply_cut_to_speech": bool(cfg.apply_cut_to_speech),
            "export_sequence_features": bool(cfg.export_sequence_features),
            "sequence_feature_max_ptm_dims": int(cfg.sequence_feature_max_ptm_dims),
            "operating_point": DEFAULT_OPERATING_POINT,
            "allow_empty": True,
        }

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
        threshold = float(cfg.threshold if threshold_override is None else threshold_override)
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
            "score_model": "bootstrap_energy_ptm_mfcc",
        }
        print(
            "[boundary] speech_boundary_ja device "
            f"requested_device={runtime_device['requested_device']} "
            f"actual_device={runtime_device['actual_device']} "
            f"dtype={runtime_device['dtype']} "
            f"ptm_param_device={runtime_device['ptm_param_device']} "
            f"ptm_param_dtype={runtime_device['ptm_param_dtype']} "
            "score_model=bootstrap_energy_ptm_mfcc",
            flush=True,
        )
        try:
            audio, sample_rate = load_audio_16k_mono(audio_path)
            duration_s = float(len(audio) / sample_rate) if sample_rate else 0.0
            total_frames = frame_count(duration_s, cfg.frame_hop_s)
            probability_sum = np.zeros(total_frames, dtype=np.float64)
            probability_count = np.zeros(total_frames, dtype=np.float32)
            cut_probability_sum = np.zeros(total_frames, dtype=np.float64)
            cut_probability_count = np.zeros(total_frames, dtype=np.float32)
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
                probs, cut_probs = _bootstrap_frame_scores(
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
                cut_probability_sum[global_start:global_end] += cut_probs[:local_end]
                cut_probability_count[global_start:global_end] += 1.0
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
            cut_probabilities = np.divide(
                cut_probability_sum,
                np.maximum(cut_probability_count, 1.0),
                out=np.zeros_like(cut_probability_sum, dtype=np.float64),
                where=cut_probability_count > 0,
            ).astype(np.float32)
            effective_probabilities = _apply_cut_gate(
                probabilities,
                cut_probabilities,
                cut_threshold=cfg.cut_threshold,
                apply_cut=cfg.apply_cut_to_speech,
            )
            raw_frames = effective_probabilities >= threshold
            padded = _padded_frames(
                raw_frames,
                pad_frames=max(0, int(round(cfg.pad_s / cfg.frame_hop_s))),
            )
            segments = frames_to_segments(
                padded,
                frame_hop_s=cfg.frame_hop_s,
                duration_s=duration_s,
                scores=probabilities,
            )
            segments = merge_segments(
                segments,
                duration_s=duration_s,
                merge_gap_s=cfg.merge_gap_s,
                min_segment_s=cfg.min_segment_s,
            )
            groups = group_segments(
                segments,
                max_group_duration_s=cfg.max_group_s,
                chunk_threshold_s=cfg.chunk_threshold_s,
            )
            params = self.signature()
            params.update(
                {
                    "audio_stats": {
                        "duration_s": duration_s,
                        "frames": int(total_frames),
                        "windows": len(starts),
                        "probability_mean": float(probabilities.mean()) if probabilities.size else 0.0,
                        "probability_max": float(probabilities.max()) if probabilities.size else 0.0,
                        "effective_probability_mean": (
                            float(effective_probabilities.mean())
                            if effective_probabilities.size
                            else 0.0
                        ),
                        "effective_probability_max": (
                            float(effective_probabilities.max())
                            if effective_probabilities.size
                            else 0.0
                        ),
                        "cut_probability_mean": (
                            float(cut_probabilities.mean()) if cut_probabilities.size else 0.0
                        ),
                        "cut_probability_max": (
                            float(cut_probabilities.max()) if cut_probabilities.size else 0.0
                        ),
                        "raw_speech_ratio": float(raw_frames.mean()) if raw_frames.size else 0.0,
                        "padded_speech_ratio": float(padded.mean()) if padded.size else 0.0,
                        "uncovered_frame_ratio": float((probability_count <= 0).mean())
                        if probability_count.size
                        else 0.0,
                    },
                    "runtime_device": runtime_device,
                }
            )
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
            if _env_bool("SPEECH_BOUNDARY_JA_EXPORT_FRAME_SCORES", "0") or _env_bool(
                "BOUNDARY_REFINER_ENABLED", "0"
            ):
                params["frame_scores"] = [float(value) for value in probabilities]
                params["cut_frame_scores"] = [float(value) for value in cut_probabilities]
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
