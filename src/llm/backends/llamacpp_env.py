"""Low-level environment and accelerator probing helpers for ``llama-server``.

Factored out of ``llamacpp_server.py`` so ``llamacpp_runtime.py`` and
``llamacpp_server.py`` share one implementation without a circular import.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from core.diagnostics import format_exit_code
from utils.subprocess_tools import no_window_subprocess_kwargs

logger = logging.getLogger(__name__)

_ACCELERATOR_PREFIXES = ("cuda", "vulkan", "rocm", "hip", "sycl", "metal", "opencl")

# Child server processes only serve local GGUF inference on 127.0.0.1 and have
# no need for API credentials or auth tokens present in the parent environment.
_SENSITIVE_ENV_EXACT = frozenset(
    {
        "API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    }
)
_SENSITIVE_ENV_SUFFIXES = (
    "_API_KEY",
    "_SECRET",
    "_SECRET_KEY",
    "_TOKEN",
    "_PASSWORD",
    "_PASSWD",
    "_PRIVATE_KEY",
)


def _is_sensitive_env_key(key: str) -> bool:
    upper = str(key or "").upper()
    return upper in _SENSITIVE_ENV_EXACT or upper.endswith(_SENSITIVE_ENV_SUFFIXES)


def _torch_library_dir() -> Path | None:
    """``torch/lib`` without importing torch (which costs seconds and VRAM)."""
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    for location in getattr(spec, "submodule_search_locations", None) or []:
        candidate = Path(location) / "lib"
        if candidate.is_dir():
            return candidate
    return None


def cuda_library_dirs(
    *,
    torch_lib_dir_fn: Callable[[], Path | None] = _torch_library_dir,
) -> tuple[str, ...]:
    """Directories to put on the server's PATH so ``ggml-cuda.dll`` can load.

    The CUDA llama.cpp zip links cuBLAS dynamically but ships no CUDA runtime;
    its own release publishes that as a separate ~400MB `cudart-*` download.
    Without `cublas64_*.dll` on the search path the CUDA backend fails to
    register *silently* and the server runs on the CPU.

    torch already installs the matching CUDA libraries into the venv this app
    runs from, so point the server at those instead of asking for a second
    download of the same DLLs.
    """
    lib_dir = torch_lib_dir_fn()
    if lib_dir is None or not any(lib_dir.glob("cublas64_*.dll")):
        return ()
    return (str(lib_dir),)


def server_environment(
    *,
    cuda_dirs_fn: Callable[[], tuple[str, ...]] = cuda_library_dirs,
) -> dict[str, str]:
    """Environment for ``llama-server`` with CUDA DLL search path and stripped secrets."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not _is_sensitive_env_key(key)
    }
    # `--fit` (on by default in current builds) may only change arguments left
    # unset, and `_build_command` sets them all. Turning it off avoids noisy
    # "failed to fit params ... abort" lines on 8 GB cards.
    env["LLAMA_ARG_FIT"] = "off"
    extra = cuda_dirs_fn()
    if extra:
        env["PATH"] = os.pathsep.join([*extra, env.get("PATH", "")])
    return env


def probe_compute_devices(
    exe: str, env: dict[str, str] | None = None
) -> list[str]:
    """Accelerators the server can actually see, via its own ``--list-devices``.

    Cheap (no model load) and the only honest answer: whether ``-ngl`` does
    anything depends on which ggml backend DLLs loaded, not on what the binary
    was named or built for.
    """
    try:
        result = subprocess.run(
            [exe, "--list-devices"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=env,
            **no_window_subprocess_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("llama-server --list-devices failed to run (%s): %s", exe, exc)
        return []
    returncode = getattr(result, "returncode", 0)
    if returncode not in (0, None):
        stderr_tail = (getattr(result, "stderr", "") or "").strip()[-400:]
        logger.warning(
            "llama-server --list-devices exited with %s: %s",
            format_exit_code(returncode),
            stderr_tail or "(no stderr)",
        )
    devices: list[str] = []
    for line in (result.stdout or "").splitlines():
        entry = line.strip()
        if entry.split(":", 1)[0].lower().startswith(_ACCELERATOR_PREFIXES):
            devices.append(entry)
    return devices


__all__ = [
    "_ACCELERATOR_PREFIXES",
    "_is_sensitive_env_key",
    "_torch_library_dir",
    "cuda_library_dirs",
    "probe_compute_devices",
    "server_environment",
]
