from __future__ import annotations

import logging
import os
import time
from dataclasses import replace
from typing import Callable

import numpy as np

from audio.loading import load_audio_16k_mono
from vad.base import SegmentationResult, SpeechSegment
from vad.whisperseg.postprocess import group_segments

log = logging.getLogger(__name__)

_REVISION = "1"


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _load_component_backend(name: str):
    if name == "silero":
        from vad.silero_backend import SileroVadBackend

        return SileroVadBackend()

    from vad import get_vad_backend

    return get_vad_backend(name)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _segment_duration(segment: SpeechSegment) -> float:
    return max(0.0, segment.end - segment.start)


def _overlap_s(left: SpeechSegment, right: SpeechSegment, *, pad_s: float = 0.0) -> float:
    start = max(left.start, right.start - pad_s)
    end = min(left.end, right.end + pad_s)
    return max(0.0, end - start)


def _segment_gate_overlap(
    segment: SpeechSegment,
    gate_segments: list[SpeechSegment],
    *,
    pad_s: float,
) -> float:
    return sum(_overlap_s(segment, gate, pad_s=pad_s) for gate in gate_segments)


def _score_linear(value: float, *, floor: float, full: float) -> float:
    if full <= floor:
        return 0.0
    return _clamp((value - floor) / (full - floor))


def _rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64, copy=False)))))
    return float(20.0 * np.log10(max(rms, 1e-12)))


def _spectral_flux_score(
    samples: np.ndarray,
    sample_rate: int,
    *,
    floor: float,
    full: float,
) -> float:
    if samples.size < max(64, int(sample_rate * 0.03)):
        return 0.0
    frame_size = max(128, int(sample_rate * 0.025))
    hop = max(64, int(sample_rate * 0.010))
    if samples.size < frame_size:
        samples = np.pad(samples, (0, frame_size - samples.size))
    window = np.hanning(frame_size).astype(np.float32, copy=False)
    previous: np.ndarray | None = None
    flux_values: list[float] = []
    for start in range(0, max(1, samples.size - frame_size + 1), hop):
        frame = samples[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        magnitude = np.abs(np.fft.rfft(frame * window)).astype(np.float32, copy=False)
        total = float(np.sum(magnitude))
        if total > 1e-8:
            magnitude = magnitude / total
        if previous is not None:
            flux_values.append(float(np.mean(np.maximum(0.0, magnitude - previous))))
        previous = magnitude
    if not flux_values:
        return 0.0
    return _score_linear(float(np.mean(flux_values)), floor=floor, full=full)


def _duration_score(duration_s: float, *, min_s: float, full_s: float, max_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    if duration_s < min_s:
        return 0.5 * _clamp(duration_s / max(min_s, 1e-6))
    if duration_s < full_s:
        return 0.5 + 0.5 * _clamp((duration_s - min_s) / max(full_s - min_s, 1e-6))
    if duration_s <= max_s:
        return 1.0
    return max(0.25, 1.0 - ((duration_s - max_s) / max(max_s, 1e-6)))


def _extract_segment_features(
    audio: np.ndarray,
    sample_rate: int,
    segment: SpeechSegment,
    *,
    rms_floor_dbfs: float,
    rms_full_dbfs: float,
    flux_floor: float,
    flux_full: float,
    duration_min_s: float,
    duration_full_s: float,
    duration_max_s: float,
) -> dict[str, float]:
    start = max(0, int(segment.start * sample_rate))
    end = min(len(audio), int(segment.end * sample_rate))
    samples = audio[start:end]
    duration = _segment_duration(segment)
    rms = _rms_dbfs(samples)
    return {
        "rms_dbfs": rms,
        "rms_score": _score_linear(rms, floor=rms_floor_dbfs, full=rms_full_dbfs),
        "spectral_flux_score": _spectral_flux_score(
            samples,
            sample_rate,
            floor=flux_floor,
            full=flux_full,
        ),
        "duration_score": _duration_score(
            duration,
            min_s=duration_min_s,
            full_s=duration_full_s,
            max_s=duration_max_s,
        ),
    }


def _neutral_features(segment: SpeechSegment, params: dict) -> dict[str, float]:
    duration = _segment_duration(segment)
    return {
        "rms_dbfs": 0.0,
        "rms_score": 0.5,
        "spectral_flux_score": 0.5,
        "duration_score": _duration_score(
            duration,
            min_s=float(params["duration_min_s"]),
            full_s=float(params["duration_full_s"]),
            max_s=float(params["duration_max_s"]),
        ),
    }


def _build_feature_lookup(
    audio_path: str,
    *,
    params: dict,
) -> Callable[[SpeechSegment], dict[str, float]]:
    try:
        audio, sample_rate = load_audio_16k_mono(audio_path)
    except Exception as exc:
        log.warning("[vad] fusion_lite feature extraction failed (%s), using neutral features", exc)
        return lambda segment: _neutral_features(segment, params)

    return lambda segment: _extract_segment_features(
        audio,
        sample_rate,
        segment,
        rms_floor_dbfs=float(params["rms_floor_dbfs"]),
        rms_full_dbfs=float(params["rms_full_dbfs"]),
        flux_floor=float(params["spectral_flux_floor"]),
        flux_full=float(params["spectral_flux_full"]),
        duration_min_s=float(params["duration_min_s"]),
        duration_full_s=float(params["duration_full_s"]),
        duration_max_s=float(params["duration_max_s"]),
    )


def _score_segment(
    segment: SpeechSegment,
    gate_segments: list[SpeechSegment],
    *,
    feature_lookup: Callable[[SpeechSegment], dict[str, float]],
    params: dict,
) -> dict[str, float | bool]:
    duration = _segment_duration(segment)
    overlap = _segment_gate_overlap(
        segment,
        gate_segments,
        pad_s=float(params["gate_pad_s"]),
    )
    overlap_ratio = _clamp(overlap / duration) if duration > 0 else 0.0
    primary_score = _clamp(
        float(segment.score)
        if segment.score is not None
        else float(params["default_primary_score"])
    )
    features = feature_lookup(segment)
    speech_score = (
        float(params["primary_weight"]) * primary_score
        + float(params["gate_weight"]) * overlap_ratio
        + float(params["rms_weight"]) * float(features["rms_score"])
        + float(params["spectral_flux_weight"]) * float(features["spectral_flux_score"])
        + float(params["duration_weight"]) * float(features["duration_score"])
    )
    keep = not (
        speech_score < float(params["min_score"])
        and overlap_ratio < float(params["min_gate_overlap_ratio"])
    )
    return {
        "keep": keep,
        "speech_score": speech_score,
        "primary_score": primary_score,
        "gate_overlap_s": overlap,
        "gate_overlap_ratio": overlap_ratio,
        **features,
    }


def _filter_grouped_segments(
    primary_groups: list[list[SpeechSegment]],
    gate_segments: list[SpeechSegment],
    *,
    feature_lookup: Callable[[SpeechSegment], dict[str, float]],
    params: dict,
) -> tuple[list[list[SpeechSegment]], list[SpeechSegment], list[SpeechSegment], list[dict]]:
    kept_groups: list[list[SpeechSegment]] = []
    kept: list[SpeechSegment] = []
    dropped: list[SpeechSegment] = []
    decisions: list[dict] = []
    for group in primary_groups:
        kept_group: list[SpeechSegment] = []
        for segment in group:
            decision = _score_segment(
                segment,
                gate_segments,
                feature_lookup=feature_lookup,
                params=params,
            )
            row = {
                "start": segment.start,
                "end": segment.end,
                "duration_s": _segment_duration(segment),
                "kept": bool(decision["keep"]),
                "speech_score": round(float(decision["speech_score"]), 6),
                "primary_score": round(float(decision["primary_score"]), 6),
                "gate_overlap_ratio": round(float(decision["gate_overlap_ratio"]), 6),
                "rms_dbfs": round(float(decision["rms_dbfs"]), 3),
                "rms_score": round(float(decision["rms_score"]), 6),
                "spectral_flux_score": round(float(decision["spectral_flux_score"]), 6),
                "duration_score": round(float(decision["duration_score"]), 6),
            }
            decisions.append(row)
            if bool(decision["keep"]):
                kept_group.append(segment)
                kept.append(segment)
            else:
                dropped.append(segment)
        if kept_group:
            kept_groups.append(kept_group)
    return kept_groups, kept, dropped, decisions


class FusionLiteVadBackend:
    name = "fusion_lite_v1"

    def __init__(self) -> None:
        primary_name = os.getenv("ASR_VAD_PRIMARY", "whisperseg").strip().lower()
        gate_name = os.getenv("ASR_VAD_GATE", "silero").strip().lower()
        if primary_name in {"fusion_lite", "fusion"}:
            raise ValueError("ASR_VAD_PRIMARY cannot be fusion_lite")
        if gate_name in {"fusion_lite", "fusion"}:
            raise ValueError("ASR_VAD_GATE cannot be fusion_lite")
        self.primary_name = primary_name
        self.gate_name = gate_name
        self.primary = _load_component_backend(primary_name)
        self.gate = _load_component_backend(gate_name)

    def signature(self) -> dict:
        return {
            "backend": self.name,
            "revision": _REVISION,
            "primary": self.primary.signature(),
            "gate": self.gate.signature(),
            "primary_weight": _env_float("FUSION_VAD_PRIMARY_WEIGHT", "0.45"),
            "gate_weight": _env_float("FUSION_VAD_GATE_WEIGHT", "0.25"),
            "rms_weight": _env_float("FUSION_VAD_RMS_WEIGHT", "0.15"),
            "spectral_flux_weight": _env_float("FUSION_VAD_SPECTRAL_FLUX_WEIGHT", "0.10"),
            "duration_weight": _env_float("FUSION_VAD_DURATION_WEIGHT", "0.05"),
            "min_score": _env_float("FUSION_VAD_MIN_SCORE", "0.45"),
            "min_gate_overlap_ratio": _env_float("FUSION_VAD_MIN_GATE_OVERLAP_RATIO", "0.05"),
            "gate_pad_s": _env_float("FUSION_VAD_GATE_PAD_S", "0.30"),
            "default_primary_score": _env_float("FUSION_VAD_DEFAULT_PRIMARY_SCORE", "0.50"),
            "rms_floor_dbfs": _env_float("FUSION_VAD_RMS_FLOOR_DBFS", "-50.0"),
            "rms_full_dbfs": _env_float("FUSION_VAD_RMS_FULL_DBFS", "-24.0"),
            "spectral_flux_floor": _env_float("FUSION_VAD_SPECTRAL_FLUX_FLOOR", "0.0002"),
            "spectral_flux_full": _env_float("FUSION_VAD_SPECTRAL_FLUX_FULL", "0.0060"),
            "duration_min_s": _env_float("FUSION_VAD_DURATION_MIN_S", "0.20"),
            "duration_full_s": _env_float("FUSION_VAD_DURATION_FULL_S", "0.50"),
            "duration_max_s": _env_float("FUSION_VAD_DURATION_MAX_S", "12.0"),
            "allow_empty": True,
        }

    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        started = time.monotonic()
        primary = self.primary.segment(
            audio_path,
            target_sr=target_sr,
            threshold_override=threshold_override,
        )
        gate = self.gate.segment(audio_path, target_sr=target_sr)
        gate_segments = list(gate.segments)
        sig = self.signature()
        feature_lookup = _build_feature_lookup(audio_path, params=sig)
        primary_groups = primary.groups or [[segment] for segment in primary.segments]
        kept_groups, kept, dropped, decisions = _filter_grouped_segments(
            primary_groups,
            gate_segments,
            feature_lookup=feature_lookup,
            params=sig,
        )
        if not primary.groups and primary.segments:
            kept_groups = group_segments(
                kept,
                max_group_duration_s=float(
                    primary.parameters.get("max_group_s")
                    or primary.parameters.get("max_group_duration_s")
                    or 6.0
                ),
                chunk_threshold_s=float(
                    primary.parameters.get("chunk_threshold_s")
                    or primary.parameters.get("chunk_threshold")
                    or 1.0
                ),
            )
        audio_duration = max(primary.audio_duration_sec, gate.audio_duration_sec)
        kept_duration = sum(_segment_duration(segment) for segment in kept)
        dropped_duration = sum(_segment_duration(segment) for segment in dropped)
        params = {
            **sig,
            "primary_runtime": primary.parameters,
            "gate_runtime": gate.parameters,
            "stats": {
                "primary_segments": len(primary.segments),
                "gate_segments": len(gate.segments),
                "kept_segments": len(kept),
                "dropped_segments": len(dropped),
                "kept_duration_s": kept_duration,
                "dropped_duration_s": dropped_duration,
                "kept_speech_ratio": kept_duration / audio_duration if audio_duration > 0 else 0.0,
            },
            "dropped_examples": [row for row in decisions if not row["kept"]][:20],
            "kept_examples": [row for row in decisions if row["kept"]][:20],
        }
        log.info(
            (
                "[vad] backend=fusion_lite primary=%s gate=%s "
                "kept=%d dropped=%d"
            ),
            self.primary_name,
            self.gate_name,
            len(kept),
            len(dropped),
        )
        return SegmentationResult(
            segments=[replace(segment) for segment in kept],
            groups=[[replace(segment) for segment in group] for group in kept_groups],
            method=self.name,
            audio_duration_sec=audio_duration,
            parameters=params,
            processing_time_sec=time.monotonic() - started,
        )
