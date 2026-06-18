from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class BaseAsrBackend(Protocol):
    """Interface shared by in-process and subprocess ASR backends."""

    is_subprocess: bool
    accepts_contexts: bool
    request_batch_size: int

    def load(self, on_stage: Callable[[str], None] | None = None) -> None: ...

    def unload_model(self, on_stage: Callable[[str], None] | None = None) -> None: ...

    def transcribe_texts(
        self,
        audio_paths: list[str],
        contexts: list[str] | None = None,
        initial_prompts: list[str | None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> list[dict[str, Any]]: ...

    def finalize_text_results(
        self,
        text_results: list[dict[str, Any]],
        on_stage: Callable[[str], None] | None = None,
    ) -> list[tuple[dict[str, Any], list[str]]]: ...

