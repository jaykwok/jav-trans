from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

_PATH_LOCK = threading.Lock()


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def runtime_root() -> Path:
    override = os.getenv("JAV_SUB_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resource_root() -> Path:
    override = os.getenv("JAV_SUB_RESOURCE_ROOT", "").strip()
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


def _norm_path_entry(raw: str) -> str:
    cleaned = raw.strip().strip('"')
    if not cleaned:
        return ""
    return os.path.normcase(os.path.normpath(cleaned))


def prepend_to_path(path: str | Path) -> None:
    """Move or insert ``path`` at the front of ``PATH``, deduplicating by normcase/normpath."""
    candidate = Path(path)
    if not candidate.exists():
        return
    prefix = str(candidate.resolve())
    target_key = _norm_path_entry(prefix)
    with _PATH_LOCK:
        current = os.environ.get("PATH", "")
        existing = [entry for entry in current.split(os.pathsep) if entry.strip()]
        deduped = [entry for entry in existing if _norm_path_entry(entry) != target_key]
        os.environ["PATH"] = os.pathsep.join([prefix, *deduped])
