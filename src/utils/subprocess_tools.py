from __future__ import annotations

import logging
import math
import os
import subprocess
import time
from collections.abc import Callable
from typing import Any

from core import resources

logger = logging.getLogger(__name__)


class SubprocessCancelledError(RuntimeError):
    """A managed child was stopped because the caller asked to cancel."""


def no_window_subprocess_kwargs() -> dict[str, int]:
    """Hide console windows for child console programs from a windowed app.

    Windows console executables such as ffmpeg.exe, ffprobe.exe and
    nvidia-smi.exe can create a transient black console window when launched
    from a PyInstaller windowed executable. CREATE_NO_WINDOW is ignored on
    non-Windows platforms by not passing it there.
    """
    if os.name != "nt":
        return {}
    flag = int(getattr(subprocess, "CREATE_NO_WINDOW", 0) or 0)
    return {"creationflags": flag} if flag else {}


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    def _win32_kernel32():
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        return kernel32

    _KERNEL32 = _win32_kernel32()

    def create_kill_on_close_job() -> int:
        """Create a Windows Job Object configured with KILL_ON_JOB_CLOSE.

        Explicit 64-bit HANDLE restype/argtypes prevent pointer truncation on
        x64 Windows, and `use_last_error=True` captures accurate Win32 errors.
        """
        job = _KERNEL32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = _KERNEL32.SetInformationJobObject(
            wintypes.HANDLE(job),
            _JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            err = ctypes.get_last_error()
            _KERNEL32.CloseHandle(wintypes.HANDLE(job))
            raise ctypes.WinError(err)
        return int(job)

    def assign_handle_to_job(job: int, win32_handle: int) -> None:
        if not win32_handle:
            raise RuntimeError("subprocess has no win32 handle to assign")
        if not _KERNEL32.AssignProcessToJobObject(
            wintypes.HANDLE(int(job)),
            wintypes.HANDLE(int(win32_handle)),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def bind_process_to_kill_on_close_job(job_handle: int | None, process_or_pid: Any) -> int:
        """Assign a ``multiprocessing.Process``, ``Popen``, or raw PID to a kill-on-close Job.

        When ``job_handle`` is ``None``, creates a fresh kill-on-close Job Object
        (and closes it if assignment fails). When ``job_handle`` is provided,
        assigns to that existing Job Object and returns ``int(job_handle)``.
        """
        owns_job = job_handle is None
        job = create_kill_on_close_job() if owns_job else int(job_handle)
        opened_handle: int | None = None
        try:
            win32_handle: Any = None
            pid: int | None = None
            if isinstance(process_or_pid, int) and not isinstance(process_or_pid, bool):
                pid = process_or_pid
            else:
                win32_handle = getattr(process_or_pid, "_handle", None) or getattr(
                    getattr(process_or_pid, "_popen", None), "_handle", None
                )
                if not win32_handle:
                    raw_pid = getattr(process_or_pid, "pid", None)
                    if isinstance(raw_pid, int) and not isinstance(raw_pid, bool):
                        pid = raw_pid
            if win32_handle:
                assign_handle_to_job(job, int(win32_handle))
            elif pid is not None and pid > 0:
                opened = _KERNEL32.OpenProcess(
                    _PROCESS_SET_QUOTA | _PROCESS_TERMINATE,
                    False,
                    wintypes.DWORD(pid),
                )
                if not opened:
                    raise ctypes.WinError(ctypes.get_last_error())
                opened_handle = int(opened)
                assign_handle_to_job(job, opened_handle)
            else:
                raise RuntimeError("process has no win32 handle or valid pid to assign")
            return int(job)
        except BaseException:
            if owns_job:
                close_kill_on_close_job(job)
            raise
        finally:
            if opened_handle is not None:
                close_kill_on_close_job(opened_handle)

    def bind_kill_on_close_job(win32_handle) -> int | None:
        """Put a running child in a kill-on-close Job Object.

        Two things follow from the job: closing the returned handle kills every
        process still inside it (the child's own grandchildren included, which
        `terminate()` never reaches), and a hard death of this process closes it
        for us. Returns None when the OS refuses - the caller then only has
        terminate/kill, which is still better than nothing.
        """
        if not win32_handle:
            logger.warning("Skipping Job Object binding: child has no Win32 handle")
            return None
        job: int | None = None
        try:
            job = create_kill_on_close_job()
            assign_handle_to_job(job, int(win32_handle))
            return job
        except Exception as exc:
            logger.warning(
                "Could not bind child handle=%r to kill-on-close Job Object: %s",
                win32_handle,
                exc,
            )
            if job is not None:
                close_kill_on_close_job(job)
            return None

    def close_kill_on_close_job(handle) -> bool:
        """Close a job handle, killing whatever is still in the job.

        A raw job handle is an integer: dropping the reference leaks it and the
        processes inside survive. Returns True when the handle was closed.
        """
        if not handle:
            return False
        try:
            closed = bool(_KERNEL32.CloseHandle(wintypes.HANDLE(int(handle))))
            if not closed:
                err = ctypes.get_last_error()
                logger.warning(
                    "CloseHandle(%r) failed: %s",
                    handle,
                    ctypes.WinError(err),
                )
            return closed
        except Exception as exc:
            logger.warning("CloseHandle(%r) raised: %s", handle, exc)
            return False

else:  # pragma: no cover - project targets Windows

    def create_kill_on_close_job() -> int:
        raise NotImplementedError("Windows Job Objects are only available on Windows")

    def assign_handle_to_job(job: int, win32_handle: int) -> None:
        del job, win32_handle
        raise NotImplementedError("Windows Job Objects are only available on Windows")

    def bind_process_to_kill_on_close_job(job_handle: int | None, process_or_pid: Any) -> int:
        del job_handle, process_or_pid
        raise NotImplementedError("Windows Job Objects are only available on Windows")

    def bind_kill_on_close_job(win32_handle) -> int | None:
        return None

    def close_kill_on_close_job(handle) -> bool:
        return False


def _popen_win32_handle(proc: subprocess.Popen):
    return getattr(proc, "_handle", None)


def child_state(proc: subprocess.Popen) -> str:
    """"gone", "alive" or "unknown".

    An error while asking is "unknown", never "gone": treating a failed check as
    a dead child is how a running process stops being anyone's responsibility.
    """
    try:
        return "gone" if proc.poll() is not None else "alive"
    except Exception:
        return "unknown"


def _track_unreleased(
    proc: subprocess.Popen,
    command: list[str],
    state: str,
    job_handle,
    reason: str,
) -> None:
    """Hand a child we could not confirm we released to the resource registry.

    `run_cancellable` returns or raises and its locals go away with it; without
    this, a child that survived kill() - or a job handle that would not close -
    simply stopped existing as far as this process was concerned. The Popen
    object is kept on purpose: it owns the OS handle on Windows, so the pid
    cannot be recycled underneath us and a later retry acts on the same child
    rather than on whatever now answers to that number.
    """
    if state == "gone" and not job_handle:
        # Nothing to own: it exited and left no handle behind.
        return
    # One call per child (every path that reaps also stops tracking), so the id
    # is allocated here rather than looked up by object identity - `id(proc)` is
    # reused once the object is freed, which would file a new child under a
    # previous child's name.
    resource_id = resources.new_resource_id("media-child")
    resources.register(
        resource_id=resource_id,
        generation=1,  # a Popen is never restarted; one child, one identity
        kind="media_child",
        # Only the program name: the arguments carry media paths.
        description=os.path.basename(str(command[0])) if command else "",
        process=proc,
        job_handle=job_handle,
        stopper=_reap_stopper,
        holds_gpu=False,
        state="stopping",
        process_state=state,
        detail=reason,
    )


def _reap_stopper(proc: subprocess.Popen, job_handle) -> resources.StopOutcome:
    state, remaining = _reap(proc, job_handle)
    return resources.StopOutcome(
        process_state=state,
        job_handle=remaining,
        detail="" if state == "gone" else f"child is {state} after kill()",
    )


def unreleased_children() -> list[dict[str, Any]]:
    """Children this process started and could not confirm it released."""
    return [
        {
            "name": record["description"],
            "pid": record["pid"],
            "state": record["process_state"],
            "reason": record["last_error"],
            "handle_open": record["handle_open"],
            "since": record["since"],
        }
        for record in resources.pending("media_child")
    ]


def retry_unreleased_cleanup() -> dict[str, int]:
    """Re-check (and retry) every child left behind. Safe to call repeatedly."""
    summary = resources.retry_pending("media_child")
    return {
        "pending": len(resources.pending("media_child")),
        "released": summary["released"],
    }


def _close_pipes(proc: subprocess.Popen) -> None:
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is None:
            continue
        try:
            if not getattr(pipe, "closed", False):
                pipe.close()
        except OSError:
            pass


def _reap_with_output(proc: subprocess.Popen, job_handle) -> tuple[str, Any, Any, Any]:
    """Take the child tree down and collect any partial captured output.

    On Windows with a ``KILL_ON_JOB_CLOSE`` Job Object, closing the handle
    first terminates the entire process tree (including grandchildren) in one
    OS operation before draining pipes with a short timeout. Returns
    ``(process_state, remaining_job_handle, stdout, stderr)`` where
    ``remaining_job_handle`` is ``None`` only when the handle actually closed.
    """
    if job_handle and close_kill_on_close_job(job_handle):
        job_handle = None
    try:
        if child_state(proc) != "gone":
            proc.kill()
    except Exception:
        pass
    out: Any = None
    err: Any = None
    try:
        out, err = proc.communicate(timeout=2.0)
    except subprocess.TimeoutExpired as exc:
        out, err = exc.output, exc.stderr
    except Exception:
        try:
            proc.wait(timeout=2.0)
        except Exception:
            pass
    _close_pipes(proc)
    return child_state(proc), job_handle, out, err


def _reap(proc: subprocess.Popen, job_handle) -> tuple[str, Any]:
    """Take the child down and wait for it, escalating until it is gone."""
    state, remaining, _out, _err = _reap_with_output(proc, job_handle)
    return state, remaining


def run_cancellable(
    command: list[str],
    *,
    timeout_s: float | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    poll_interval_s: float = 0.25,
    capture_output: bool = False,
    encoding: str | None = None,
    errors: str | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run a child process that a caller can time out *and* cancel.

    `subprocess.run(timeout=...)` only answers the first: a cancelled job kept
    a two-hour ffmpeg (and an unbounded ffprobe) running to completion while the
    task bar already said 取消, holding the worker thread with it. This keeps the
    process handle, polls the cancel channel between short waits, and kills the
    tree on cancel, timeout or error.

    Raises `SubprocessCancelledError` on cancel and `subprocess.TimeoutExpired`
    on timeout; with `check=True`, `subprocess.CalledProcessError` on failure.
    """
    if not command:
        raise ValueError("command must not be empty")
    poll_val = float(poll_interval_s)
    if not math.isfinite(poll_val) or poll_val <= 0:
        raise ValueError(f"poll_interval_s must be a positive finite number, got {poll_interval_s!r}")
    slice_s = max(0.01, poll_val)

    timeout_val: float | None = None
    if timeout_s is not None:
        timeout_val = float(timeout_s)
        if not math.isfinite(timeout_val):
            raise ValueError(f"timeout_s must be finite, got {timeout_s!r}")

    popen_kwargs: dict = dict(no_window_subprocess_kwargs())
    popen_kwargs["stdin"] = subprocess.DEVNULL
    if capture_output:
        popen_kwargs["stdout"] = subprocess.PIPE
        popen_kwargs["stderr"] = subprocess.PIPE
    if encoding is not None or errors is not None:
        popen_kwargs["text"] = True
        popen_kwargs["encoding"] = encoding
        popen_kwargs["errors"] = errors

    proc = subprocess.Popen(command, **popen_kwargs)
    job_handle = None
    reaped = False
    try:
        job_handle = bind_kill_on_close_job(_popen_win32_handle(proc))
        deadline = (
            time.monotonic() + timeout_val
            if timeout_val is not None and timeout_val > 0
            else None
        )
        while True:
            wait_s = slice_s
            if deadline is not None:
                wait_s = max(0.01, min(wait_s, deadline - time.monotonic()))
            try:
                stdout, stderr = proc.communicate(timeout=wait_s)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel_requested is not None and cancel_requested():
                state, job_handle, _out, _err = _reap_with_output(proc, job_handle)
                _track_unreleased(proc, command, state, job_handle, "cancel")
                job_handle = None
                reaped = True
                raise SubprocessCancelledError(
                    f"cancelled while running: {os.path.basename(str(command[0]))}"
                )
            if deadline is not None and time.monotonic() >= deadline:
                state, job_handle, partial_out, partial_err = _reap_with_output(
                    proc, job_handle
                )
                _track_unreleased(proc, command, state, job_handle, "timeout")
                job_handle = None
                reaped = True
                raise subprocess.TimeoutExpired(
                    command,
                    float(timeout_val or 0.0),
                    output=partial_out,
                    stderr=partial_err,
                )
    except BaseException:
        # Cancel and timeout already reaped and recorded why; re-reaping here
        # would only overwrite the specific reason with a generic one.
        if not reaped and (job_handle is not None or child_state(proc) != "gone"):
            state, job_handle, _out, _err = _reap_with_output(proc, job_handle)
            _track_unreleased(proc, command, state, job_handle, "error")
            job_handle = None
        raise
    finally:
        if job_handle is not None and not close_kill_on_close_job(job_handle):
            # The child finished normally, but the job object did not close.
            # Keep the handle so a retry can close it instead of dropping the
            # only reference to it.
            _track_unreleased(
                proc,
                command,
                child_state(proc),
                job_handle,
                "handle_close_failed",
            )

    completed = subprocess.CompletedProcess(
        command,
        proc.returncode,
        stdout,
        stderr,
    )
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return completed
