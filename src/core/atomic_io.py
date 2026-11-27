"""Shared atomic file I/O helpers and launcher workspace lock primitives.

Stdlib-only so every core, pipeline, utils, and web module can rely on one
durable write path (same-directory temp file + flush + fsync + Windows-retry
replace) and one cross-process workspace mutex matching ``launcher.py``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_INSTANCE_DIR_NAME = "launcher"
_GATE_NAME = "workspace.gate"
_DEFAULT_GATE_WAIT_S = 5.0
_DEFAULT_GATE_POLL_S = 0.05


def replace_atomically(
    src: str | Path,
    dst: str | Path,
    *,
    retries: int = 6,
    base_delay_s: float = 0.02,
) -> None:
    """Replace ``dst`` with ``src``, retrying transient Windows ``PermissionError``."""
    src_path = Path(src)
    dst_path = Path(dst)
    attempts = max(1, int(retries))
    delay = max(0.0, float(base_delay_s))
    for attempt in range(attempts):
        try:
            os.replace(src_path, dst_path)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(delay * (2**attempt))


def _temp_sibling_path(target: Path) -> Path:
    return target.parent / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"


def write_text_atomic(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write text to ``path`` via a same-directory temp file + fsync."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _temp_sibling_path(target)
    try:
        with tmp_path.open("w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        replace_atomically(tmp_path, target)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def write_json_atomic(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    trailing_newline: bool = True,
    default: Callable[[Any], Any] | None = None,
) -> Path:
    """Stream-serialize JSON to a same-directory temp file, fsync, and replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _temp_sibling_path(target)
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=ensure_ascii,
                indent=indent,
                default=default,
            )
            if trailing_newline:
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_atomically(tmp_path, target)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def _instance_dir(root: str | Path) -> Path:
    return Path(root) / "tmp" / _INSTANCE_DIR_NAME


def _try_lock_file(handle: Any) -> bool:
    """Non-blocking exclusive OS lock compatible with ``launcher.py``."""
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock_and_close(handle: Any) -> None:
    if handle is None:
        return
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        try:
            handle.close()
        except OSError:
            pass


@contextmanager
def workspace_gate(
    root: str | Path,
    *,
    wait_s: float = _DEFAULT_GATE_WAIT_S,
    poll_s: float = _DEFAULT_GATE_POLL_S,
) -> Iterator[bool]:
    """Acquire the shared OS lock ``tmp/launcher/workspace.gate``.

    Yields ``True`` when the lock is held for the duration of the ``with``
    block, or ``False`` if the lock stayed busy past ``wait_s`` (or could not
    be opened). Matches ``launcher.py``'s ``workspace.gate`` protocol so
    instance registration, exit sweeps, and self-update slot scans never race.
    """
    try:
        directory = _instance_dir(root)
        directory.mkdir(parents=True, exist_ok=True)
        handle = (directory / _GATE_NAME).open("a+", encoding="utf-8")
    except OSError:
        yield False
        return

    deadline = time.monotonic() + max(0.0, float(wait_s))
    sleep_s = max(0.005, float(poll_s))
    acquired = False
    while True:
        if _try_lock_file(handle):
            acquired = True
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(sleep_s, remaining))

    if not acquired:
        try:
            handle.close()
        except OSError:
            pass
        yield False
        return

    try:
        yield True
    finally:
        _unlock_and_close(handle)


def _slot_is_held(slot: Path) -> bool:
    """Return True if a live process still holds ``slot``, else remove stale slot."""
    if os.name == "nt":
        try:
            slot.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            return True
        return False

    try:
        import fcntl

        with slot.open("r+", encoding="utf-8") as probe:
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        slot.unlink(missing_ok=True)
    except OSError:
        pass
    return False


def other_launcher_instances_running(
    root: str | Path,
    *,
    current_pid: int | None = None,
    wait_s: float = _DEFAULT_GATE_WAIT_S,
) -> bool:
    """Detect whether another ``launcher.py`` window holds a slot under ``root``.

    Acquires ``workspace.gate`` before probing or pruning stale
    ``instance-*.lock`` files so a new window publishing its slot cannot race
    with stale-slot cleanup. If ``workspace.gate`` remains locked past
    ``wait_s``, conservatively returns ``True`` because another instance owns
    the workspace transition.
    """
    instance_dir = _instance_dir(root)
    if not instance_dir.is_dir():
        return False

    pid = os.getpid() if current_pid is None else int(current_pid)
    own_name = f"instance-{pid}.lock"

    with workspace_gate(root, wait_s=wait_s) as acquired:
        if not acquired:
            return True
        try:
            slots = sorted(instance_dir.glob("instance-*.lock"))
        except OSError:
            return True
        for slot in slots:
            if slot.name == own_name:
                continue
            if _slot_is_held(slot):
                return True
    return False


__all__ = [
    "other_launcher_instances_running",
    "replace_atomically",
    "workspace_gate",
    "write_json_atomic",
    "write_text_atomic",
]
