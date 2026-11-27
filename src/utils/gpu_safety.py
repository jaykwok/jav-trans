"""CUDA admission and VRAM caps for training and dataset-build tools.

The runtime ASR worker enforces its own VRAM and RAM budgets. Host memory
admission for the application lives in pipeline.memory_safety.
"""
from __future__ import annotations

import re

from core.typed_config import env_float

DEFAULT_VRAM_SAFETY_RATIO = 0.95
_CUDA_DEVICE_RE = re.compile(r"^cuda(?::(\d+))?$")


def resolve_inference_device(requested: str | None, *, stage: str):
    """Require CUDA for model workloads, including strict syntax and device-count checks."""
    import torch

    value = str(requested or "auto").strip().lower()
    if value == "auto":
        value = "cuda"
    match = _CUDA_DEVICE_RE.fullmatch(value)
    if match is None:
        raise RuntimeError(
            f"{stage} requires CUDA ('cuda' or 'cuda:<index>'); "
            f"device {value!r} is unsupported; CPU fallback is disabled"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{stage} requested CUDA but CUDA is unavailable; CPU fallback is disabled"
        )
    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        raise RuntimeError(
            f"{stage} requested CUDA but no CUDA devices are available "
            f"(device_count={device_count}); CPU fallback is disabled"
        )
    index_text = match.group(1)
    if index_text is not None:
        index = int(index_text)
        if index >= device_count:
            raise RuntimeError(
                f"{stage} requested CUDA device {value!r}, but only {device_count} "
                f"CUDA device(s) are available (valid indices: 0..{device_count - 1})"
            )
    return torch.device(value)


def _resolve_ratio(ratio: float | None, env_key: str, default: float) -> float:
    """Explicit value, else env override, else default - clamped to a sane band."""
    if ratio is None:
        ratio = env_float(env_key, default, minimum=0.1, maximum=1.0)
    return min(1.0, max(0.1, float(ratio)))


def apply_vram_safety_cap(ratio: float | None = None) -> float | None:
    """Cap the CUDA caching allocator at ratio x physical VRAM on all devices.

    ``ratio`` defaults to env ``VRAM_SAFETY_RATIO``, then 0.95. Returns the
    applied ratio, or None when CUDA is unavailable.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    ratio = _resolve_ratio(ratio, "VRAM_SAFETY_RATIO", DEFAULT_VRAM_SAFETY_RATIO)
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(ratio, device_index)
    return ratio
