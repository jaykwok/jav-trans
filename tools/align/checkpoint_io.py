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
looking exactly like a successful one. A CPU run is a legitimate thing to want -
a smoke test on a laptop - but it has to be the thing that was asked for.
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
    "auto (default) uses CUDA and fails when it is unavailable; cuda requires "
    "it; cpu asks for it deliberately"
)


def add_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trust-checkpoint", action="store_true", help=TRUST_HELP)


def add_device_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto", help=DEVICE_HELP
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit falling back to CPU when CUDA is unavailable; without it "
        "an unavailable GPU is an error rather than a very slow run",
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


def resolve_device(requested: str = "auto", *, allow_cpu: bool = False):
    """The device to train on, or an error saying why there is not one.

    `auto` means "the GPU this workflow is written for". It does not mean "CPU
    if that is all there is": a silent fallback is how a run that was supposed to
    take two hours takes a week and is only noticed at the end.
    """
    import torch

    choice = str(requested or "auto").strip().lower()
    if choice == "cpu":
        return torch.device("cpu")
    available = torch.cuda.is_available()
    if available:
        return torch.device("cuda")
    if choice == "auto" and allow_cpu:
        print("[device] CUDA 不可用，按 --allow-cpu 在 CPU 上运行（会非常慢）")
        return torch.device("cpu")
    raise SystemExit(
        "CUDA 不可用。这个流程是为 GPU 写的，静默退回 CPU 会把两小时的训练变成"
        "几天而不报错。确认驱动/环境后重试，或显式使用 --device cpu / --allow-cpu。"
    )


__all__ = [
    "add_checkpoint_arguments",
    "add_device_arguments",
    "load_checkpoint",
    "resolve_device",
]
