from __future__ import annotations

import logging
import math
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
_FUSION_BACKEND_NAMES = {"fusion", "fusion_lite", "fusion_lite_boost", "fusion_lite_sigmoid"}


_BOOST_RECIPE = {
    "scoring_variant": "boost",
    "dynamic_gate_pad": True,
    "primary_weight": 0.50,
    "gate_weight": 0.25,
    "rms_weight": 0.15,
    "spectral_flux_weight": 0.08,
    "duration_weight": 0.02,
    "min_score": 0.45,
    "min_gate_overlap_ratio": 0.08,
    "boost_overlap_scale": 0.30,
    "boost_rms_scale": 0.20,
    "boost_keep_min_rms_score": 0.60,
}

_SIGMOID_RECIPE = {
    "scoring_variant": "sigmoid",
    "dynamic_gate_pad": True,
    "primary_weight": 0.40,
    "gate_weight": 0.30,
    "rms_weight": 0.15,
    "spectral_flux_weight": 0.10,
    "duration_weight": 0.05,
    "min_score": 0.42,
    "min_gate_overlap_ratio": 0.12,
    "sigmoid_keep_min_primary_score": 0.55,
    "sigmoid_midpoint": 0.50,
    "sigmoid_steepness": 8.0,
    "duration_curve": "window",
    "duration_sigmoid_mid_s": 0.60,
    "duration_sigmoid_steepness": 6.0,
    "duration_long_mid_s": 8.0,
    "duration_long_steepness": 1.4,
}


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _load_component_backend(name: str):
    if name == "whisperseg":
        from vad.whisperseg_backend import WhisperSegVadBackend

        return WhisperSegVadBackend()
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


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


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


def _duration_window_score(
    duration_s: float,
    *,
    short_mid_s: float,
    short_steepness: float,
    long_mid_s: float,
    long_steepness: float,
) -> float:
    if duration_s <= 0:
        return 0.0
    short_gate = _sigmoid(short_steepness * (duration_s - short_mid_s))
    long_gate = _sigmoid(long_steepness * (long_mid_s - duration_s))
    return _clamp(short_gate * long_gate)


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
    duration_curve: str = "linear",
    duration_sigmoid_mid_s: float = 0.60,
    duration_sigmoid_steepness: float = 6.0,
    duration_long_mid_s: float = 8.0,
    duration_long_steepness: float = 1.4,
) -> dict[str, float]:
    start = max(0, int(segment.start * sample_rate))
    end = min(len(audio), int(segment.end * sample_rate))
    samples = audio[start:end]
    duration = _segment_duration(segment)
    rms = _rms_dbfs(samples)
    if duration_curve == "window":
        duration_score = _duration_window_score(
            duration,
            short_mid_s=duration_sigmoid_mid_s,
            short_steepness=duration_sigmoid_steepness,
            long_mid_s=duration_long_mid_s,
            long_steepness=duration_long_steepness,
        )
    else:
        duration_score = _duration_score(
            duration,
            min_s=duration_min_s,
            full_s=duration_full_s,
            max_s=duration_max_s,
        )
    return {
        "rms_dbfs": rms,
        "rms_score": _score_linear(rms, floor=rms_floor_dbfs, full=rms_full_dbfs),
        "spectral_flux_score": _spectral_flux_score(
            samples,
            sample_rate,
            floor=flux_floor,
            full=flux_full,
        ),
        "duration_score": duration_score,
    }


def _neutral_features(segment: SpeechSegment, params: dict) -> dict[str, float]:
    duration = _segment_duration(segment)
    if params.get("duration_curve") == "window":
        duration_score = _duration_window_score(
            duration,
            short_mid_s=float(params["duration_sigmoid_mid_s"]),
            short_steepness=float(params["duration_sigmoid_steepness"]),
            long_mid_s=float(params["duration_long_mid_s"]),
            long_steepness=float(params["duration_long_steepness"]),
        )
    else:
        duration_score = _duration_score(
            duration,
            min_s=float(params["duration_min_s"]),
            full_s=float(params["duration_full_s"]),
            max_s=float(params["duration_max_s"]),
        )
    return {
        "rms_dbfs": 0.0,
        "rms_score": 0.5,
        "spectral_flux_score": 0.5,
        "duration_score": duration_score,
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
        duration_curve=str(params.get("duration_curve", "linear")),
        duration_sigmoid_mid_s=float(params.get("duration_sigmoid_mid_s", 0.60)),
        duration_sigmoid_steepness=float(params.get("duration_sigmoid_steepness", 6.0)),
        duration_long_mid_s=float(params.get("duration_long_mid_s", 8.0)),
        duration_long_steepness=float(params.get("duration_long_steepness", 1.4)),
    )


def _effective_gate_pad_s(duration_s: float, *, params: dict) -> float:
    base_pad_s = float(params["gate_pad_s"])
    if not bool(params.get("dynamic_gate_pad")):
        return base_pad_s
    if duration_s <= 0:
        return 0.0
    return min(base_pad_s, max(0.05, duration_s * 0.35))


def _score_segment(
    segment: SpeechSegment,
    gate_segments: list[SpeechSegment],
    *,
    feature_lookup: Callable[[SpeechSegment], dict[str, float]],
    params: dict,
) -> dict[str, float | bool]:
    duration = _segment_duration(segment)
    gate_pad_s = _effective_gate_pad_s(duration, params=params)
    overlap = _segment_gate_overlap(
        segment,
        gate_segments,
        pad_s=gate_pad_s,
    )
    overlap_ratio = _clamp(overlap / duration) if duration > 0 else 0.0
    primary_score = _clamp(
        float(segment.score)
        if segment.score is not None
        else float(params["default_primary_score"])
    )
    features = feature_lookup(segment)
    raw_score = (
        float(params["primary_weight"]) * primary_score
        + float(params["gate_weight"]) * overlap_ratio
        + float(params["rms_weight"]) * float(features["rms_score"])
        + float(params["spectral_flux_weight"]) * float(features["spectral_flux_score"])
        + float(params["duration_weight"]) * float(features["duration_score"])
    )
    variant = str(params.get("scoring_variant", "linear"))
    enhancement = 1.0
    if variant == "boost":
        enhancement = (
            1.0 + float(params["boost_overlap_scale"]) * overlap_ratio
        ) * (
            1.0 + float(params["boost_rms_scale"]) * float(features["rms_score"])
        )
        speech_score = min(1.0, raw_score * enhancement)
        keep = (
            speech_score > float(params["min_score"])
            or (
                overlap_ratio > float(params["min_gate_overlap_ratio"])
                and float(features["rms_score"]) > float(params["boost_keep_min_rms_score"])
            )
        )
    elif variant == "sigmoid":
        speech_score = _sigmoid(
            float(params["sigmoid_steepness"]) * (raw_score - float(params["sigmoid_midpoint"]))
        )
        keep = (
            speech_score > float(params["min_score"])
            or (
                overlap_ratio > float(params["min_gate_overlap_ratio"])
                and primary_score > float(params["sigmoid_keep_min_primary_score"])
            )
        )
    else:
        speech_score = raw_score
        keep = not (
            speech_score < float(params["min_score"])
            and overlap_ratio < float(params["min_gate_overlap_ratio"])
        )
    return {
        "keep": keep,
        "speech_score": speech_score,
        "raw_score": raw_score,
        "score_enhancement": enhancement,
        "primary_score": primary_score,
        "gate_pad_s": gate_pad_s,
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
                "raw_score": round(float(decision["raw_score"]), 6),
                "score_enhancement": round(float(decision["score_enhancement"]), 6),
                "primary_score": round(float(decision["primary_score"]), 6),
                "gate_pad_s": round(float(decision["gate_pad_s"]), 6),
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


def _score_distribution(decisions: list[dict]) -> dict:
    scores = [float(row["speech_score"]) for row in decisions]
    if not scores:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "bins": ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
            "histogram": [0, 0, 0, 0, 0],
        }
    arr = np.asarray(scores, dtype=np.float64)
    histogram = [0, 0, 0, 0, 0]
    for score in scores:
        idx = min(4, max(0, int(_clamp(score) / 0.2)))
        histogram[idx] += 1
    return {
        "count": len(scores),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "p10": round(float(np.percentile(arr, 10)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "bins": ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        "histogram": histogram,
    }


class FusionLiteVadBackend:
    name = "fusion_lite_v1"
    scoring_variant = "linear"

    def __init__(self) -> None:
        primary_name = os.getenv("ASR_VAD_PRIMARY", "whisperseg").strip().lower()
        gate_name = os.getenv("ASR_VAD_GATE", "silero").strip().lower()
        if primary_name in _FUSION_BACKEND_NAMES:
            raise ValueError("ASR_VAD_PRIMARY cannot be fusion_lite")
        if gate_name in _FUSION_BACKEND_NAMES:
            raise ValueError("ASR_VAD_GATE cannot be fusion_lite")
        self.primary_name = primary_name
        self.gate_name = gate_name
        self.primary = _load_component_backend(primary_name)
        self.gate = _load_component_backend(gate_name)

    def _variant_overrides(self) -> dict:
        if self.scoring_variant == "boost":
            return dict(_BOOST_RECIPE)
        if self.scoring_variant == "sigmoid":
            return dict(_SIGMOID_RECIPE)
        return {"scoring_variant": "linear", "dynamic_gate_pad": False}

    def signature(self) -> dict:
        sig = {
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
        sig.update(self._variant_overrides())
        return sig

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
                "score_distribution": _score_distribution(decisions),
            },
            "dropped_examples": [row for row in decisions if not row["kept"]][:20],
            "kept_examples": [row for row in decisions if row["kept"]][:20],
        }
        log.info(
            (
                "[vad] backend=%s primary=%s gate=%s "
                "kept=%d dropped=%d"
            ),
            self.name,
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


class FusionLiteBoostVadBackend(FusionLiteVadBackend):
    name = "fusion_lite_boost_v1"
    scoring_variant = "boost"


class FusionLiteSigmoidVadBackend(FusionLiteVadBackend):
    name = "fusion_lite_sigmoid_v1"
    scoring_variant = "sigmoid"
