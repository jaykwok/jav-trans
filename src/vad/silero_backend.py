from __future__ import annotations

import contextlib
import io
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

from audio.loading import load_audio_16k_mono
from vad.base import SegmentationResult, SpeechSegment
from vad.ffmpeg_backend import FfmpegSilencedetectBackend
from vad.whisperseg.postprocess import group_segments

log = logging.getLogger(__name__)

_REVISION = "1"
_MODEL_CACHE: tuple[Any, Any] | None = None
_MODEL_LOCK = Lock()


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _torch_home() -> Path:
    from utils.model_paths import PROJECT_ROOT

    torch_home = Path(os.getenv("TORCH_HOME", "./temp/torch")).expanduser()
    if not torch_home.is_absolute():
        torch_home = PROJECT_ROOT / torch_home
    torch_home = torch_home.resolve()
    os.environ["TORCH_HOME"] = str(torch_home)
    return torch_home


def _load_silero_model() -> tuple[Any, Any]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    with _MODEL_LOCK:
        if _MODEL_CACHE is not None:
            return _MODEL_CACHE

        import torch

        hub_dir = _torch_home() / "hub"
        cached_repo = hub_dir / "snakers4_silero-vad_master"
        load_kwargs: dict[str, Any] = {
            "repo_or_dir": str(cached_repo) if cached_repo.exists() else "snakers4/silero-vad",
            "model": "silero_vad",
            "force_reload": False,
            "onnx": _env_bool("SILERO_VAD_ONNX", "1"),
            "trust_repo": True,
        }
        if cached_repo.exists():
            load_kwargs["source"] = "local"

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                vad_model, vad_utils = torch.hub.load(**load_kwargs)
        except Exception:
            if not load_kwargs["onnx"]:
                raise
            load_kwargs["onnx"] = False
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                vad_model, vad_utils = torch.hub.load(**load_kwargs)
        _MODEL_CACHE = (vad_model, vad_utils[0])
        return _MODEL_CACHE


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _timestamps_to_segments(items: list[dict[str, Any]]) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    for item in items:
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        raw_score = item.get("score")
        try:
            score = None if raw_score is None else float(raw_score)
        except (TypeError, ValueError):
            score = None
        segments.append(SpeechSegment(start=start, end=end, score=score))
    return segments


class SileroVadBackend:
    name = "silero_vad"

    def signature(self) -> dict:
        return {
            "backend": self.name,
            "revision": _REVISION,
            "threshold": _env_float("SILERO_VAD_THRESHOLD", "0.50"),
            "min_speech_ms": _env_int("SILERO_VAD_MIN_SPEECH_MS", "180"),
            "min_silence_ms": _env_int("SILERO_VAD_MIN_SILENCE_MS", "120"),
            "pad_ms": _env_int("SILERO_VAD_PAD_MS", "120"),
            "max_speech_s": _env_float("SILERO_VAD_MAX_SPEECH_S", "30.0"),
            "max_group_s": _env_float("SILERO_VAD_MAX_GROUP_S", "6.0"),
            "chunk_threshold_s": _env_float("SILERO_VAD_CHUNK_THRESHOLD_S", "1.0"),
            "onnx": _env_bool("SILERO_VAD_ONNX", "1"),
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
        started = time.monotonic()
        params = self.signature()
        threshold = float(threshold_override) if threshold_override is not None else float(params["threshold"])
        try:
            import torch

            audio, sample_rate = load_audio_16k_mono(audio_path)
            waveform = torch.from_numpy(audio).contiguous()
            duration_s = len(audio) / sample_rate if sample_rate else 0.0
            vad_model, get_speech_timestamps = _load_silero_model()
            kwargs: dict[str, Any] = {
                "model": vad_model,
                "sampling_rate": sample_rate,
                "threshold": threshold,
                "min_speech_duration_ms": int(params["min_speech_ms"]),
                "min_silence_duration_ms": int(params["min_silence_ms"]),
                "speech_pad_ms": int(params["pad_ms"]),
                "return_seconds": True,
            }
            max_speech_s = float(params["max_speech_s"])
            if max_speech_s > 0:
                kwargs["max_speech_duration_s"] = max_speech_s
            try:
                timestamps = get_speech_timestamps(waveform, **kwargs)
            except TypeError:
                kwargs.pop("max_speech_duration_s", None)
                timestamps = get_speech_timestamps(waveform, **kwargs)
        except Exception as exc:
            log.warning("[vad] Silero VAD failed (%s), falling back to ffmpeg", exc)
            fallback = FfmpegSilencedetectBackend().segment(audio_path)
            fallback.parameters = {
                **fallback.parameters,
                "backend": self.name,
                "fallback": "ffmpeg",
                "error": str(exc),
                "allow_empty": False,
            }
            return fallback

        segments = _timestamps_to_segments(list(timestamps or []))
        spans = _merge_spans([(segment.start, segment.end) for segment in segments])
        segments = [SpeechSegment(start=start, end=end) for start, end in spans]
        groups = group_segments(
            segments,
            max_group_duration_s=float(params["max_group_s"]),
            chunk_threshold_s=float(params["chunk_threshold_s"]),
        )
        speech_dur = sum(max(0.0, segment.end - segment.start) for segment in segments)
        stats = {
            "speech_ratio": float(speech_dur / duration_s) if duration_s > 0 else 0.0,
            "segment_count": len(segments),
            "chunks_per_min": float(len(segments) / (duration_s / 60.0)) if duration_s > 0 else 0.0,
        }
        params = {
            **params,
            "threshold": threshold,
            "audio_stats": stats,
        }
        log.info(
            "[vad] backend=silero_vad chunks=%d speech_ratio=%.3f threshold=%.3f",
            len(groups),
            stats["speech_ratio"],
            threshold,
        )
        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=duration_s,
            parameters=params,
            processing_time_sec=time.monotonic() - started,
        )
