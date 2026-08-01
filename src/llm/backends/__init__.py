"""Translation backend registry and process-wide instance lifecycle."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Protocol


class TranslationBackend(Protocol):
    def chat_completion(self, messages: list[dict], **kwargs) -> str: ...
    def cache_identity(self) -> str: ...
    def name(self) -> str: ...


BackendFactory = Callable[[], TranslationBackend]

_BACKEND_REGISTRY: dict[str, BackendFactory] = {}
_BACKEND_INSTANCES: dict[str, TranslationBackend] = {}
_REGISTRY_LOCK = threading.RLock()


def _normalize_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    if not normalized:
        raise ValueError("Translation backend name must not be empty")
    return normalized


def register_backend(
    name: str,
    factory: BackendFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a backend factory.

    Instances are shared process-wide. Re-registering requires ``replace=True``
    and invalidates the old instance, making extension mistakes visible rather
    than silently overriding a backend.
    """
    normalized = _normalize_name(name)
    if not callable(factory):
        raise TypeError("Backend factory must be callable")
    with _REGISTRY_LOCK:
        if normalized in _BACKEND_REGISTRY and not replace:
            raise ValueError(f"Translation backend already registered: {normalized}")
        previous = _BACKEND_INSTANCES.pop(normalized, None)
        if previous is not None:
            close = getattr(previous, "close", None)
            if callable(close):
                close()
        _BACKEND_REGISTRY[normalized] = factory


def selected_backend_name(name: str | None = None) -> str:
    return _normalize_name(
        name if name is not None else os.getenv("TRANSLATION_BACKEND", "openai")
    )


def get_backend(name: str | None = None) -> TranslationBackend:
    """Return the shared backend instance for ``name``."""
    normalized = selected_backend_name(name)
    with _REGISTRY_LOCK:
        factory = _BACKEND_REGISTRY.get(normalized)
        if factory is None:
            available = ", ".join(sorted(_BACKEND_REGISTRY)) or "<none>"
            raise ValueError(
                f"Unknown translation backend: {normalized}. Available: {available}"
            )
        instance = _BACKEND_INSTANCES.get(normalized)
        if instance is None:
            instance = factory()
            _BACKEND_INSTANCES[normalized] = instance
        return instance


def reset_backend(name: str | None = None) -> None:
    """Drop cached instances, primarily for configuration changes and tests."""
    with _REGISTRY_LOCK:
        if name is None:
            instances = list(_BACKEND_INSTANCES.values())
            _BACKEND_INSTANCES.clear()
        else:
            instance = _BACKEND_INSTANCES.pop(selected_backend_name(name), None)
            instances = [instance] if instance is not None else []
        # Close while the registry lock is held so another caller cannot load a
        # replacement GPU model before the previous instance releases memory.
        for instance in instances:
            close = getattr(instance, "close", None)
            if callable(close):
                close()


def list_backends() -> list[str]:
    with _REGISTRY_LOCK:
        return sorted(_BACKEND_REGISTRY)


def _register_builtin_backends() -> None:
    from llm.backends.llamacpp_server import LlamaCppServerBackend
    from llm.backends.local_model import LocalModelBackend
    from llm.backends.openai_compat import OpenAICompatBackend

    register_backend("openai", OpenAICompatBackend)
    register_backend("local", LocalModelBackend)
    register_backend("llamacpp", LlamaCppServerBackend)


_register_builtin_backends()


__all__ = [
    "TranslationBackend",
    "get_backend",
    "list_backends",
    "register_backend",
    "reset_backend",
    "selected_backend_name",
]
