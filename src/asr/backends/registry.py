import os

from asr.backends.base import BaseAsrBackend
from asr.backends.qwen import DEFAULT_QWEN_ASR_BACKEND, QWEN_ASR_BACKEND_REPOS
from asr.local_backend import LocalAsrBackend, SubprocessAsrBackend


DEFAULT_ASR_BACKEND = DEFAULT_QWEN_ASR_BACKEND

_QWEN_BACKENDS = set(QWEN_ASR_BACKEND_REPOS)
_VALID_ASR_BACKENDS = set(_QWEN_BACKENDS)
_VALID_ASR_WORKER_MODES = {"inproc", "subprocess"}


def current_asr_backend() -> str:
    return os.getenv("ASR_BACKEND", DEFAULT_ASR_BACKEND).strip() or DEFAULT_ASR_BACKEND


def current_asr_worker_mode() -> str:
    return os.getenv("ASR_WORKER_MODE", "subprocess").strip().lower()


def get_backend_label() -> str:
    backend_name = current_asr_backend()
    worker_mode = current_asr_worker_mode()
    if backend_name in _QWEN_BACKENDS:
        if worker_mode == "subprocess":
            return f"{backend_name} (subprocess worker)"
        if worker_mode == "inproc":
            return f"{backend_name} (inproc)"
        return f"{backend_name} ({worker_mode})"
    return backend_name


def _resolve_asr_backend(device: str) -> BaseAsrBackend:
    backend_name = current_asr_backend()
    worker_mode = current_asr_worker_mode()
    if backend_name not in _VALID_ASR_BACKENDS:
        raise ValueError(
            f"Unsupported ASR_BACKEND={backend_name!r}; "
            f"expected one of {sorted(_VALID_ASR_BACKENDS)}"
        )
    if worker_mode not in _VALID_ASR_WORKER_MODES:
        raise ValueError(
            f"Unsupported ASR_WORKER_MODE={worker_mode!r}; "
            f"expected one of {sorted(_VALID_ASR_WORKER_MODES)}"
        )
    if worker_mode == "inproc":
        return LocalAsrBackend(device)
    return SubprocessAsrBackend(device)


def _create_asr_backend(device: str) -> BaseAsrBackend:
    return _resolve_asr_backend(device)


def _is_subprocess_backend(backend: BaseAsrBackend) -> bool:
    return bool(getattr(backend, "is_subprocess", False))
