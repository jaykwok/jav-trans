from __future__ import annotations

import logging
import os
import time

from vad.base import SegmentationResult, SpeechSegment
from vad.ffmpeg_backend import FfmpegSilencedetectBackend

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


class WhisperSegVadBackend:
    name = "whisperseg_v1"

    def __init__(self) -> None:
        self._segmenter = None

    def _ensure(self) -> None:
        if self._segmenter is not None:
            return

        from vad.whisperseg.whisperseg_core import WhisperSegSpeechSegmenter

        self._segmenter = WhisperSegSpeechSegmenter(
            threshold=float(os.getenv("WHISPERSEG_THRESHOLD", "0.25")),
            min_speech_duration_ms=int(os.getenv("WHISPERSEG_MIN_SPEECH_MS", "80")),
            min_silence_duration_ms=int(os.getenv("WHISPERSEG_MIN_SILENCE_MS", "80")),
            speech_pad_ms=int(os.getenv("WHISPERSEG_PAD_MS", "400")),
            max_speech_duration_s=float(os.getenv("WHISPERSEG_MAX_SPEECH_S", "4.0")),
            max_group_duration_s=float(os.getenv("WHISPERSEG_MAX_GROUP_S", "5.0")),
            chunk_threshold_s=float(os.getenv("WHISPERSEG_CHUNK_THRESHOLD_S", "1.0")),
            force_cpu=_env_bool("WHISPERSEG_FORCE_CPU", "0"),
        )

    def signature(self) -> dict:
        return {
            "backend": self.name,
            "revision": _REVISION,
            "threshold": float(os.getenv("WHISPERSEG_THRESHOLD", "0.25")),
            "min_speech_ms": int(os.getenv("WHISPERSEG_MIN_SPEECH_MS", "80")),
            "min_silence_ms": int(os.getenv("WHISPERSEG_MIN_SILENCE_MS", "80")),
            "pad_ms": int(os.getenv("WHISPERSEG_PAD_MS", "400")),
            "max_speech_s": float(os.getenv("WHISPERSEG_MAX_SPEECH_S", "4.0")),
            "max_group_s": float(os.getenv("WHISPERSEG_MAX_GROUP_S", "5.0")),
            "chunk_threshold_s": float(os.getenv("WHISPERSEG_CHUNK_THRESHOLD_S", "1.0")),
        }

    def segment(self, audio_path: str, *, target_sr: int = 16000) -> SegmentationResult:
        try:
            self._ensure()
        except Exception as exc:
            log.warning("[vad] WhisperSeg init failed (%s), falling back to ffmpeg", exc)
            return FfmpegSilencedetectBackend().segment(audio_path, target_sr=target_sr)

        t0 = time.monotonic()
        try:
            raw = self._segmenter.segment(audio_path, sample_rate=target_sr)
        except Exception as exc:
            log.warning("[vad] WhisperSeg segment failed (%s), falling back to ffmpeg", exc)
            return FfmpegSilencedetectBackend().segment(audio_path, target_sr=target_sr)
        elapsed = time.monotonic() - t0

        groups = [
            [SpeechSegment(start=segment.start, end=segment.end) for segment in group]
            for group in raw.groups
        ]
        segments = [segment for group in groups for segment in group]
        if not groups:
            log.warning("[vad] WhisperSeg returned 0 groups, falling back to ffmpeg")
            return FfmpegSilencedetectBackend().segment(audio_path, target_sr=target_sr)

        provider = (
            "CPUExecutionProvider"
            if _env_bool("WHISPERSEG_FORCE_CPU", "0")
            else "CUDAExecutionProvider"
        )
        mean_dur = sum(group[-1].end - group[0].start for group in groups) / len(groups)
        log.info(
            "[vad] backend=whisperseg_v1 onnx_provider=%s chunks=%d mean_dur=%.2fs",
            provider,
            len(groups),
            mean_dur,
        )

        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=raw.audio_duration_sec,
            parameters=self.signature(),
            processing_time_sec=elapsed,
        )

