from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SpeechSegment:
    """A speech span in seconds, independent of any backend-specific format."""

    start: float
    end: float
    score: float | None = None
    subtitle_min_duration_s: float | None = None
    below_subtitle_min_duration: bool = False
    micro_chunk_candidate: bool = False
    micro_resolve_action: str = ""
    micro_resolve_reason: str = ""
    left_split_score: float | None = None
    right_split_score: float | None = None
    left_split_prominence: float | None = None
    right_split_prominence: float | None = None
    left_split_speech_valley: float | None = None
    right_split_speech_valley: float | None = None
    primary_cut_candidates: list[dict[str, Any]] = field(default_factory=list)
    weak_cut_candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SegmentationResult:
    """Speech-boundary output consumed by ASR chunking.

    `segments` keeps the flat detected speech spans. `groups` is the chunked
    form passed to ASR, where nearby spans can share one inference call.
    """

    segments: list[SpeechSegment]
    groups: list[list[SpeechSegment]]
    method: str
    audio_duration_sec: float
    parameters: dict
    processing_time_sec: float


@runtime_checkable
class SpeechBoundaryBackend(Protocol):
    """Common contract for speech-boundary backends."""

    name: str

    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult: ...

    def signature(self) -> dict: ...
