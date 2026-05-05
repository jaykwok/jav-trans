from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class BaseAsrBackend(Protocol):
    """Interface shared by in-process, subprocess, and Whisper-family ASR backends.

    The pipeline separates text generation from timestamp finalization so large
    ASR models can be unloaded before the forced aligner is loaded.
    """

    is_subprocess: bool
    accepts_contexts: bool
    timestamp_mode: str
    request_batch_size: int
    align_batch_size: int

    def load(self, on_stage: Callable[[str], None] | None = None) -> None: ...

    def unload_model(self, on_stage: Callable[[str], None] | None = None) -> None: ...

    def unload_forced_aligner(
        self, on_stage: Callable[[str], None] | None = None
    ) -> None: ...

    def transcribe_texts(
        self,
        audio_paths: list[str],
        contexts: list[str] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> list[dict[str, Any]]: ...

    def finalize_text_results(
        self,
        text_results: list[dict[str, Any]],
        on_stage: Callable[[str], None] | None = None,
    ) -> list[tuple[dict[str, Any], list[str]]]: ...

