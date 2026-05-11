import os

from whisper.backends.base import BaseAsrBackend
from whisper.local_backend import LocalAsrBackend, SubprocessAsrBackend


ASR_BACKEND = os.getenv("ASR_BACKEND", "anime-whisper").strip().lower()
_ASR_WORKER_MODE = os.getenv("ASR_WORKER_MODE", "subprocess").strip().lower()

_WHISPER_BACKENDS = {
    "anime-whisper",
    "whisper-ja-anime-v0.3",
    "whisper-ja-1.5b",
}
_QWEN_BACKENDS = {"qwen3-asr-1.7b"}
_VALID_ASR_BACKENDS = _WHISPER_BACKENDS | _QWEN_BACKENDS
_VALID_ASR_WORKER_MODES = {"inproc", "subprocess"}


def current_asr_backend() -> str:
    return os.getenv("ASR_BACKEND", "anime-whisper").strip().lower()


def current_asr_worker_mode() -> str:
    return os.getenv("ASR_WORKER_MODE", "subprocess").strip().lower()


def get_backend_label() -> str:
    backend_name = current_asr_backend()
    worker_mode = current_asr_worker_mode()
    if backend_name == "qwen3-asr-1.7b":
        if worker_mode == "subprocess":
            return "qwen3-asr-1.7b (subprocess worker)"
        if worker_mode == "inproc":
            return "qwen3-asr-1.7b (inproc)"
        return f"qwen3-asr-1.7b ({worker_mode})"
    return backend_name


def _create_whisper_backend(device: str) -> BaseAsrBackend:
    from whisper.model_backend import create_whisper_model_backend

    return create_whisper_model_backend(current_asr_backend(), device)


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
    if backend_name in _WHISPER_BACKENDS:
        return _create_whisper_backend(device)
    if worker_mode == "inproc":
        return LocalAsrBackend(device)
    return SubprocessAsrBackend(device)


def _create_asr_backend(device: str) -> BaseAsrBackend:
    return _resolve_asr_backend(device)


def _is_subprocess_backend(backend: BaseAsrBackend) -> bool:
    return bool(getattr(backend, "is_subprocess", False))
