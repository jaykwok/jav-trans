import os

from asr.backends.base import BaseAsrBackend
from asr.backends.qwen import (
    DEFAULT_QWEN_ASR_BACKEND,
    QWEN_ASR_BACKEND_REPOS,
)
from asr.local_backend import LocalAsrBackend


DEFAULT_ASR_BACKEND = DEFAULT_QWEN_ASR_BACKEND
ASR_RUNTIME_MODE = "gpu_worker"

_QWEN_BACKENDS = set(QWEN_ASR_BACKEND_REPOS)
_VALID_ASR_BACKENDS = set(_QWEN_BACKENDS)


def current_asr_backend() -> str:
    return os.getenv("ASR_BACKEND", DEFAULT_ASR_BACKEND).strip() or DEFAULT_ASR_BACKEND


def current_asr_worker_mode() -> str:
    return ASR_RUNTIME_MODE


def get_backend_label() -> str:
    backend_name = current_asr_backend()
    return backend_name


def _resolve_asr_backend(device: str) -> BaseAsrBackend:
    backend_name = current_asr_backend()
    if backend_name not in _VALID_ASR_BACKENDS:
        raise ValueError(
            f"Unsupported ASR_BACKEND={backend_name!r}; "
            f"expected one of {sorted(_VALID_ASR_BACKENDS)}"
        )
    return LocalAsrBackend(device)


def _create_asr_backend(device: str) -> BaseAsrBackend:
    return _resolve_asr_backend(device)
