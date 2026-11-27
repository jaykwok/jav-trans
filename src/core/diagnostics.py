"""Process-wide diagnostics, rotating log setup, fault handling and error formatting.

Stdlib-only by design so `launcher.py`, `core/*`, `pipeline/*`, `llm/*`, `asr/*`
and `web/*` can all share one implementation without pulling in heavy runtime
dependencies or creating import cycles.
"""

from __future__ import annotations

import asyncio
import atexit
import faulthandler
import logging
import os
import re
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlsplit, urlunsplit

from utils.runtime_paths import runtime_path

_LOGGER = logging.getLogger(__name__)
_URL_CREDENTIALS_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^/\s?#\"'<>]+)@")

_NTSTATUS_NAMES: dict[int, str] = {
    0x80000003: "STATUS_BREAKPOINT",
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
    0xC00000FD: "STATUS_STACK_OVERFLOW",
    0xC0000135: "STATUS_DLL_NOT_FOUND",
    0xC0000139: "STATUS_ENTRYPOINT_NOT_FOUND",
    0xC0000142: "STATUS_DLL_INIT_FAILED",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
}

_PROCESS_LOG_LOCK = threading.Lock()
_INSTALLED_HANDLERS: dict[Path, logging.Handler] = {}
_FAULT_STREAMS: dict[Path, IO[str]] = {}
_ACTIVE_LOG_PATH: Path | None = None
_HOOKS_INSTALLED = False
_PREV_SYS_EXCEPTHOOK = None
_PREV_THREAD_EXCEPTHOOK = None


def redact_url_credentials(text: str) -> str:
    """Scrub `user[:password]@` from URLs and free-form exception strings."""
    raw = str(text or "")
    if "@" not in raw:
        return raw
    # Fast-path standalone URLs via urlsplit so host/port/IPv6 formatting stays intact.
    if "://" in raw and not any(ch.isspace() for ch in raw):
        try:
            parsed = urlsplit(raw)
            if parsed.username is not None or parsed.password is not None:
                host = parsed.hostname or ""
                if ":" in host and not host.startswith("["):
                    host = f"[{host}]"
                netloc = f"***@{host}"
                if parsed.port is not None:
                    netloc = f"{netloc}:{parsed.port}"
                return urlunsplit(
                    (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
                )
        except ValueError:
            pass
    return _URL_CREDENTIALS_RE.sub(r"\1***@", raw)


def is_fatal_cuda_error(exc: BaseException) -> bool:
    """Unified detector for unrecoverable CUDA / GPU / OOM / memory-budget failures.

    Shared across ASR chunking, alignment, and local backend layers so an
    illegal memory access or CUDA OOM is never downgraded into synthetic timing
    and instead propagates to the GPU worker for clean process recovery.
    """
    name = type(exc).__name__.lower()
    if "outofmemoryerror" in name:
        return True
    detail = f"{name}: {exc!r}".lower()
    return (
        ("out of memory" in detail and ("cuda" in detail or "gpu" in detail or "torch" in detail))
        or "cuda error" in detail
        or "illegal memory access" in detail
        or "device-side assert" in detail
        or "cublas_status" in detail
        or "cudnn_status" in detail
        or "vram budget exceeded" in detail
        or "shared vram spill" in detail
        or "physical ram budget exceeded" in detail
    )


def current_process_log_path() -> Path | None:
    """Return the active rotating log file path configured for this process."""
    with _PROCESS_LOG_LOCK:
        return _ACTIVE_LOG_PATH


def current_process_log_hint(default: str = "logs/app-<PID>.log") -> str:
    """Human-readable relative/file hint for the current process log."""
    active = current_process_log_path()
    if active is None:
        return default
    return f"logs/{active.name}"


def format_exit_code(code: int | None) -> str:
    """Format a process exit code, surfacing Windows NTSTATUS codes in hex."""
    if code is None:
        return "None"
    try:
        value = int(code)
    except (TypeError, ValueError):
        return str(code)
    if os.name == "nt":
        unsigned: int | None = None
        if value >= 0x80000000:
            unsigned = value & 0xFFFFFFFF
        elif value < 0:
            candidate = value & 0xFFFFFFFF
            if candidate >= 0x80000000:
                unsigned = candidate
        if unsigned is not None:
            label = _NTSTATUS_NAMES.get(unsigned)
            if label:
                return f"{value} (0x{unsigned:08X} {label})"
            return f"{value} (0x{unsigned:08X})"
    return str(value)


def format_exception_brief(exc: BaseException) -> str:
    """Single-line `TypeName: message` (or `TypeName` when message is empty)."""
    name = type(exc).__name__
    message = str(exc).strip()
    if not message:
        return name
    if message.startswith(f"{name}:"):
        return message
    return f"{name}: {message}"


def format_exception_chain(
    exc: BaseException,
    *,
    max_depth: int = 4,
    max_chars: int = 600,
) -> str:
    """Walk `__cause__` / non-suppressed `__context__` so wrapped errors keep root cause."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(parts) < max_depth:
        seen.add(id(current))
        brief = format_exception_brief(current)
        if not parts or brief != parts[-1]:
            parts.append(brief)
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__ and current.__context__ is not None:
            current = current.__context__
        else:
            break
    joined = " <- ".join(parts)
    if max_chars > 0 and len(joined) > max_chars:
        return joined[: max_chars - 3].rstrip() + "..."
    return joined


def _close_fault_streams() -> None:
    try:
        if faulthandler.is_enabled():
            faulthandler.disable()
    except Exception:
        pass
    with _PROCESS_LOG_LOCK:
        streams = list(_FAULT_STREAMS.values())
        _FAULT_STREAMS.clear()
    for stream in streams:
        try:
            stream.flush()
            stream.close()
        except OSError:
            pass


atexit.register(_close_fault_streams)


def _install_exception_hooks() -> None:
    global _HOOKS_INSTALLED, _PREV_SYS_EXCEPTHOOK, _PREV_THREAD_EXCEPTHOOK
    if _HOOKS_INSTALLED:
        return
    _PREV_SYS_EXCEPTHOOK = sys.excepthook
    _PREV_THREAD_EXCEPTHOOK = getattr(threading, "excepthook", None)

    def _sys_hook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            if _PREV_SYS_EXCEPTHOOK is not None:
                _PREV_SYS_EXCEPTHOOK(exc_type, exc_value, exc_tb)
            return
        logging.getLogger("jav_sub.unhandled").critical(
            "Uncaught exception in main thread",
            exc_info=(exc_type, exc_value, exc_tb),
        )
        if _PREV_SYS_EXCEPTHOOK is not None:
            _PREV_SYS_EXCEPTHOOK(exc_type, exc_value, exc_tb)

    def _thread_hook(args) -> None:
        if issubclass(args.exc_type, SystemExit):
            return
        thread_name = getattr(getattr(args, "thread", None), "name", "unknown")
        logging.getLogger("jav_sub.unhandled").critical(
            "Uncaught exception in thread %s",
            thread_name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        if _PREV_THREAD_EXCEPTHOOK is not None:
            _PREV_THREAD_EXCEPTHOOK(args)

    sys.excepthook = _sys_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook
    _HOOKS_INSTALLED = True


def attach_asyncio_exception_handler(loop: asyncio.AbstractEventLoop) -> None:
    """Route unhandled asyncio task/callback exceptions into the process log."""
    previous = loop.get_exception_handler()

    def _handler(active_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        message = str(context.get("message") or "Unhandled asyncio exception")
        exc = context.get("exception")
        task = context.get("future") or context.get("task")
        task_name = task.get_name() if hasattr(task, "get_name") else ""
        suffix = f" (task={task_name})" if task_name else ""
        if isinstance(exc, BaseException):
            logging.getLogger("jav_sub.asyncio").error(
                "%s%s",
                message,
                suffix,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        else:
            logging.getLogger("jav_sub.asyncio").error(
                "%s%s context=%r",
                message,
                suffix,
                {k: v for k, v in context.items() if k != "message"},
            )
        if previous is not None:
            previous(active_loop, context)
        else:
            active_loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)


def _prune_old_process_logs(
    base_dir: Path,
    *,
    stem: str,
    suffix: str,
    keep: int,
    exclude: set[Path],
) -> None:
    """Remove oldest per-process logs beyond `keep`, skipping any locked active files.

    Matches only generated `<stem>-<pid><suffix>` files (never arbitrary files
    like `app-investigation.log`). Unlinks the primary log first so if another
    live Windows window holds an open handle to it, the entire process group
    (primary log, `-fault` sibling, and `.1`/`.2` rotated backups) is left intact.
    """
    if keep <= 0:
        return
    primary_re = re.compile(rf"^{re.escape(stem)}-\d+{re.escape(suffix)}$")
    try:
        entries = list(base_dir.iterdir())
    except OSError:
        return

    all_files: list[Path] = []
    primary_entries: list[tuple[float, Path]] = []
    for entry in entries:
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue
        all_files.append(entry)
        if primary_re.match(entry.name) is None:
            continue
        try:
            resolved = entry.resolve()
            if resolved in exclude:
                continue
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        primary_entries.append((mtime, entry))

    if len(primary_entries) <= keep:
        return
    primary_entries.sort(key=lambda item: item[0], reverse=True)
    for _mtime, old_log in primary_entries[keep:]:
        try:
            old_log.unlink()
        except OSError:
            # Active sibling windows hold an open handle to their primary log on
            # Windows; skip the entire process group rather than pruning its backups.
            continue
        backup_re = re.compile(rf"^{re.escape(old_log.name)}\.\d+$")
        fault_name = f"{old_log.stem}-fault{suffix}"
        for target in all_files:
            if target.name != fault_name and backup_re.match(target.name) is None:
                continue
            try:
                if target.resolve() not in exclude:
                    target.unlink()
            except OSError:
                pass


def configure_process_logging(
    log_name: str = "app.log",
    *,
    log_dir: Path | None = None,
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    enable_faulthandler: bool = True,
    install_hooks: bool = True,
    per_process: bool = True,
    max_process_logs: int = 20,
) -> Path | None:
    """Attach a rotating UTF-8 log file and native crash logger to the root logger.

    Safe to call repeatedly: one handler is kept per resolved log file path.
    By default (`per_process=True`), each process writes to its own `<stem>-<pid><suffix>`
    file (e.g. `app-12345.log`) so concurrent Windows instances sharing one
    workspace directory never collide with `PermissionError` during rollover.
    `faulthandler` writes to a dedicated unrotated sibling handle (`<stem>-fault.log`)
    so log rotation can never close the file descriptor underneath native crash dumps.
    """
    global _ACTIVE_LOG_PATH
    base_dir = Path(log_dir) if log_dir is not None else runtime_path("logs")
    requested = Path(log_name)
    effective_name = (
        f"{requested.stem}-{os.getpid()}{requested.suffix}"
        if per_process
        else requested.name
    )
    log_path = (base_dir / effective_name).resolve()
    with _PROCESS_LOG_LOCK:
        existing = _INSTALLED_HANDLERS.get(log_path)
        root = logging.getLogger()
        if existing is not None and existing in root.handlers:
            if root.level == logging.NOTSET or root.level > level:
                root.setLevel(level)
            _ACTIVE_LOG_PATH = log_path
            if install_hooks:
                _install_exception_hooks()
            return log_path
        if existing is not None:
            _INSTALLED_HANDLERS.pop(log_path, None)
            try:
                existing.close()
            except OSError:
                pass

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                log_path,
                maxBytes=max(64 * 1024, int(max_bytes)),
                backupCount=max(1, int(backup_count)),
                encoding="utf-8",
            )
        except OSError:
            _LOGGER.warning("Could not open process log %s", log_path, exc_info=True)
            return None

        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] [pid=%(process)d %(threadName)s] %(name)s: %(message)s"
            )
        )
        root.addHandler(handler)
        if root.level == logging.NOTSET or root.level > level:
            root.setLevel(level)
        _INSTALLED_HANDLERS[log_path] = handler
        _ACTIVE_LOG_PATH = log_path

        fault_path = log_path.with_name(f"{log_path.stem}-fault{log_path.suffix}")
        if enable_faulthandler:
            if fault_path not in _FAULT_STREAMS:
                try:
                    stream = fault_path.open("a", encoding="utf-8", errors="replace")
                    faulthandler.enable(file=stream, all_threads=True)
                    _FAULT_STREAMS[fault_path] = stream
                except (OSError, RuntimeError):
                    _LOGGER.warning(
                        "Could not enable faulthandler on %s",
                        fault_path,
                        exc_info=True,
                    )

        if per_process:
            _prune_old_process_logs(
                base_dir,
                stem=requested.stem,
                suffix=requested.suffix,
                keep=max_process_logs,
                exclude={log_path, fault_path.resolve()},
            )

        if install_hooks:
            _install_exception_hooks()
        return log_path


def redirect_process_stdio_to_file(stream: IO[str]) -> bool:
    """Redirect OS-level stdout/stderr (fd 1 and 2) to `stream` in an isolated child.

    Native C/C++/CUDA/cuDNN/FFmpeg errors write directly to file descriptor 2 and
    bypass Python's `logging` module. Duplicating the log stream fd onto fd 1/2
    ensures native diagnostics land inside the worker log file on hidden-console
    Windows builds.
    """
    try:
        target_fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return False
    try:
        stream.flush()
        os.dup2(target_fd, 2)
        os.dup2(target_fd, 1)
    except OSError:
        return False
    sys.stderr = stream
    sys.stdout = stream
    return True
