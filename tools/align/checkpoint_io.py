"""Reading a checkpoint, and choosing a device, on purpose rather than by luck.

Two habits ran through the offline alignment tools and both of them decided
something important silently.

**`torch.load(..., weights_only=False)`** unpickles whatever the archive says,
which is arbitrary code executed before a single tensor is read. That is fine
for a file this project produced and nothing else, and "a file this project
produced" was never checked - it was assumed, in scripts that also take a
`--checkpoint` path from the command line. So the default here is the strict
loader, and the permissive one is something the operator asks for by name,
having decided the file is theirs.

**`torch.device("cuda" if torch.cuda.is_available() else "cpu")`** turns a
missing driver into a training run that is a hundred times slower and finishes
looking exactly like a successful one. Training, feature extraction and batch
evaluation require CUDA. Tiny tensor tests can use CPU without making it an
option in the real workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

TRUST_HELP = (
    "load the checkpoint with the permissive pickle loader. Only for files this "
    "project produced: it executes whatever the archive contains."
)

DEVICE_HELP = (
    "auto (default) and cuda both require CUDA; CPU execution is unsupported"
)


def add_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trust-checkpoint", action="store_true", help=TRUST_HELP)


def add_device_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", choices=("auto", "cuda"), default="auto", help=DEVICE_HELP
    )


def load_checkpoint(path: str | Path, *, trusted: bool = False) -> Any:
    """Read a checkpoint, strictly unless the caller vouched for the file."""
    import torch

    resolved = Path(path)
    if trusted:
        return torch.load(resolved, map_location="cpu", weights_only=False)
    try:
        return torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SystemExit(
            f"{resolved} 无法在安全模式下读取：它包含张量和基本类型以外的对象。"
            "如果这个文件确实是本项目训练/晋升流程产出的，请加 --trust-checkpoint；"
            f"否则不要加载它。（原始错误：{exc}）"
        ) from exc


def resolve_device(requested: str = "auto"):
    """Require CUDA before a tool reads data, downloads weights or starts work."""
    from utils.gpu_safety import resolve_inference_device

    try:
        return resolve_inference_device(requested, stage="alignment tools")
    except RuntimeError as exc:
        raise SystemExit(
            "对齐训练与评估必须使用 CUDA，不能使用 CPU。"
            "请检查 NVIDIA 驱动和项目 PyTorch CUDA 环境后重试。"
            f"（{exc}）"
        ) from exc


__all__ = [
    "add_checkpoint_arguments",
    "add_device_arguments",
    "load_checkpoint",
    "resolve_device",
]
