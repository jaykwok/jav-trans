from __future__ import annotations

import logging
import os
import time

from vad.base import SegmentationResult, SpeechSegment

log = logging.getLogger(__name__)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
log.setLevel(logging.INFO)
log.propagate = False

_REVISION = "6ac29e2c"


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


class WhisperSegVadBackend:
    name = "whisperseg_v1"

    def __init__(self) -> None:
        self._segmenter = None

    def _ensure(self) -> None:
        if self._segmenter is not None:
            return

        from vad.whisperseg.whisperseg_core import WhisperSegSpeechSegmenter

        self._segmenter = WhisperSegSpeechSegmenter(
            threshold=float(os.getenv("WHISPERSEG_THRESHOLD", "0.35")),
            min_speech_duration_ms=int(os.getenv("WHISPERSEG_MIN_SPEECH_MS", "80")),
            min_silence_duration_ms=int(os.getenv("WHISPERSEG_MIN_SILENCE_MS", "80")),
            speech_pad_ms=int(os.getenv("WHISPERSEG_PAD_MS", "400")),
            max_speech_duration_s=float(os.getenv("WHISPERSEG_MAX_SPEECH_S", "4.0")),
            max_group_duration_s=float(os.getenv("WHISPERSEG_MAX_GROUP_S", "5.0")),
            chunk_threshold_s=float(os.getenv("WHISPERSEG_CHUNK_THRESHOLD_S", "1.0")),
            force_cpu=_env_bool("WHISPERSEG_FORCE_CPU", "0"),
            neg_threshold_offset=_env_float("WHISPERSEG_NEG_THRESHOLD_OFFSET", "0.15"),
        )

    def signature(self) -> dict:
        return {
            "backend": self.name,
            "revision": _REVISION,
            "threshold": float(os.getenv("WHISPERSEG_THRESHOLD", "0.35")),
            "min_speech_ms": int(os.getenv("WHISPERSEG_MIN_SPEECH_MS", "80")),
            "min_silence_ms": int(os.getenv("WHISPERSEG_MIN_SILENCE_MS", "80")),
            "pad_ms": int(os.getenv("WHISPERSEG_PAD_MS", "400")),
            "max_speech_s": float(os.getenv("WHISPERSEG_MAX_SPEECH_S", "4.0")),
            "max_group_s": float(os.getenv("WHISPERSEG_MAX_GROUP_S", "5.0")),
            "chunk_threshold_s": float(os.getenv("WHISPERSEG_CHUNK_THRESHOLD_S", "1.0")),
            "neg_offset": _env_float("WHISPERSEG_NEG_THRESHOLD_OFFSET", "0.15"),
        }

    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        try:
            self._ensure()
        except Exception as exc:
            raise RuntimeError(f"WhisperSeg init failed: {exc}") from exc

        t0 = time.monotonic()
        original_threshold = float(getattr(self._segmenter, "threshold"))
        requested_threshold = (
            float(threshold_override)
            if threshold_override is not None
            else original_threshold
        )
        final_threshold = requested_threshold
        adaptive_enabled = _env_bool("ASR_VAD_ADAPTIVE", "0")
        threshold_adjusted = False
        try:
            self._segmenter.threshold = requested_threshold
            raw = self._segmenter.segment(audio_path, sample_rate=target_sr)
            speech_ratio = _audio_stats(raw).get("speech_ratio", 0.0)
            adjusted_threshold = _adaptive_threshold(requested_threshold, speech_ratio)
            if adaptive_enabled and adjusted_threshold != requested_threshold:
                threshold_adjusted = True
                final_threshold = adjusted_threshold
                self._segmenter.threshold = adjusted_threshold
                raw = self._segmenter.segment(audio_path, sample_rate=target_sr)
        except Exception as exc:
            raise RuntimeError(f"WhisperSeg segment failed: {exc}") from exc
        finally:
            self._segmenter.threshold = original_threshold
        elapsed = time.monotonic() - t0

        groups = [
            [
                SpeechSegment(start=segment.start, end=segment.end, score=segment.score)
                for segment in group
            ]
            for group in raw.groups
        ]
        segments = [segment for group in groups for segment in group]
        if not groups:
            log.warning("[vad] WhisperSeg returned 0 groups; ASR will skip this audio")

        provider = (
            "CPUExecutionProvider"
            if _env_bool("WHISPERSEG_FORCE_CPU", "0")
            else "CUDAExecutionProvider"
        )
        mean_dur = sum(group[-1].end - group[0].start for group in groups) / len(groups)
        raw_params = raw.parameters if isinstance(raw.parameters, dict) else {}
        stats = _audio_stats(raw)
        mean_score = stats.get("mean_prob", 0.0)
        speech_ratio = stats.get("speech_ratio", 0.0)
        neg_offset = float(raw_params.get("neg_offset", self.signature()["neg_offset"]))
        log.info(
            (
                "[vad] backend=whisperseg_v1 onnx_provider=%s chunks=%d "
                "mean_dur=%.2fs mean_score=%.3f speech_ratio=%.3f neg_offset=%.3f"
            ),
            provider,
            len(groups),
            mean_dur,
            mean_score,
            speech_ratio,
            neg_offset,
        )

        params = self.signature()
        params["threshold"] = float(raw_params.get("threshold", final_threshold))
        params["neg_offset"] = neg_offset
        params["audio_stats"] = stats
        params["adaptive"] = {
            "enabled": adaptive_enabled,
            "threshold_adjusted": threshold_adjusted,
            "initial_threshold": requested_threshold,
            "final_threshold": final_threshold,
        }

        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=raw.audio_duration_sec,
            parameters=params,
            processing_time_sec=elapsed,
        )


def _audio_stats(result: SegmentationResult) -> dict[str, float]:
    parameters = result.parameters if isinstance(result.parameters, dict) else {}
    stats = parameters.get("audio_stats")
    if not isinstance(stats, dict):
        return {"mean_prob": 0.0, "speech_ratio": 0.0}
    return {
        "mean_prob": float(stats.get("mean_prob") or 0.0),
        "speech_ratio": float(stats.get("speech_ratio") or 0.0),
    }


def _adaptive_threshold(threshold: float, speech_ratio: float) -> float:
    if speech_ratio > 0.85:
        return min(threshold + 0.10, 0.95)
    if speech_ratio < 0.05:
        return max(threshold - 0.05, 0.05)
    return threshold
