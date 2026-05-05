from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class SpeechSegment:
    """A speech span in seconds, independent of any backend-specific format."""

    start: float
    end: float


@dataclass
class SegmentationResult:
    """VAD output consumed by ASR chunking.

    `segments` keeps the flat detected speech spans. `groups` is the chunked
    form passed to ASR, where short nearby spans can share one inference call.
    """

    segments: list[SpeechSegment]
    groups: list[list[SpeechSegment]]
    method: str
    audio_duration_sec: float
    parameters: dict
    processing_time_sec: float


@runtime_checkable
class VadBackend(Protocol):
    """Common contract for ffmpeg, WhisperSeg, and future VAD backends."""

    name: str

    def segment(self, audio_path: str, *, target_sr: int = 16000) -> SegmentationResult: ...

    def signature(self) -> dict: ...

