from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class BaseAsrBackend(Protocol):
    """Interface for ASR model execution inside the unified GPU worker."""

    is_subprocess: bool
    request_batch_size: int

    def load(self, on_stage: Callable[[str], None] | None = None) -> None: ...

    def unload_model(self, on_stage: Callable[[str], None] | None = None) -> None: ...

    def transcribe_texts(
        self,
        audio_paths: list[str],
        on_stage: Callable[[str], None] | None = None,
    ) -> list[dict[str, Any]]: ...

    def finalize_text_results(
        self,
        text_results: list[dict[str, Any]],
        on_stage: Callable[[str], None] | None = None,
    ) -> list[tuple[dict[str, Any], list[str]]]: ...

