"""CUDA allocator safety cap for training and dataset-build tools.

On Windows WDDM the driver silently spills allocations beyond physical VRAM
into shared system memory: jobs slow to a crawl long before they fail. Capping
the PyTorch caching allocator makes overflow surface immediately as a regular
CUDA OOM so batch sizes get fixed instead of silently degrading.

The runtime ASR worker has its own budget (``ASR_STAGE_WORKER_VRAM_RATIO``);
this module covers everything outside that worker.
"""
from __future__ import annotations

import os

DEFAULT_VRAM_SAFETY_RATIO = 0.95


def apply_vram_safety_cap(ratio: float | None = None) -> float | None:
    """Cap the CUDA caching allocator at ratio x physical VRAM on all devices.

    ``ratio`` defaults to env ``VRAM_SAFETY_RATIO``, then 0.95. Returns the
    applied ratio, or None when CUDA is unavailable.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    if ratio is None:
        raw = os.getenv("VRAM_SAFETY_RATIO", "").strip()
        try:
            ratio = float(raw) if raw else DEFAULT_VRAM_SAFETY_RATIO
        except ValueError:
            ratio = DEFAULT_VRAM_SAFETY_RATIO
    ratio = min(1.0, max(0.1, float(ratio)))
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(ratio, device_index)
    return ratio
