from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def runtime_root() -> Path:
    override = os.getenv("JAVTRANS_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resource_root() -> Path:
    override = os.getenv("JAVTRANS_RESOURCE_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", runtime_root())).resolve()
    return Path(__file__).resolve().parents[2]


def runtime_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (runtime_root() / candidate).resolve()


def resource_path(*parts: str | Path) -> Path:
    return resource_root().joinpath(*(Path(part) for part in parts)).resolve()


def prepend_to_path(path: str | Path) -> None:
    candidate = Path(path)
    if not candidate.exists():
        return
    current = os.environ.get("PATH", "")
    prefix = str(candidate.resolve())
    paths = current.split(os.pathsep) if current else []
    if prefix not in paths:
        os.environ["PATH"] = prefix + (os.pathsep + current if current else "")
