from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from typing import Any

from core import resources


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

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9

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

    def bind_kill_on_close_job(win32_handle) -> int | None:
        """Put a running child in a kill-on-close Job Object.

        Two things follow from the job: closing the returned handle kills every
        process still inside it (the child's own grandchildren included, which
        `terminate()` never reaches), and a hard death of this process closes it
        for us. Returns None when the OS refuses - the caller then only has
        terminate/kill, which is still better than nothing.
        """
        try:
            kernel32 = ctypes.windll.kernel32
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return None
            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                job,
                _JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            ):
                kernel32.CloseHandle(job)
                return None
            if not kernel32.AssignProcessToJobObject(job, int(win32_handle)):
                kernel32.CloseHandle(job)
                return None
            return job
        except Exception:
            return None

    def close_kill_on_close_job(handle) -> bool:
        """Close a job handle, killing whatever is still in the job.

        A raw job handle is an integer: dropping the reference leaks it and the
        processes inside survive. Returns True when the handle was closed.
        """
        if not handle:
            return False
        try:
            return bool(ctypes.windll.kernel32.CloseHandle(int(handle)))
        except Exception:
            return False

else:  # pragma: no cover - project targets Windows

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


def _reap(proc: subprocess.Popen, job_handle) -> tuple[str, Any]:
    """Take the child down and wait for it, escalating until it is gone.

    Returns the child's state and the job handle *still open* - None only when
    the handle was actually closed. Reporting a failed close as "released" would
    leak the handle, and because the job is kill-on-close, the tree inside it
    would then outlive every attempt to stop it.
    """
    try:
        if child_state(proc) != "gone":
            proc.kill()
    except Exception:
        pass
    try:
        proc.communicate(timeout=10)
    except Exception:
        pass
    # Last resort for a child that ignored kill() or left grandchildren: the
    # job object takes the whole tree with it when the handle closes.
    if job_handle and close_kill_on_close_job(job_handle):
        job_handle = None
    return child_state(proc), job_handle


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
    popen_kwargs: dict = dict(no_window_subprocess_kwargs())
    if capture_output:
        popen_kwargs["stdout"] = subprocess.PIPE
        popen_kwargs["stderr"] = subprocess.PIPE
    if encoding is not None or errors is not None:
        popen_kwargs["text"] = True
        popen_kwargs["encoding"] = encoding
        popen_kwargs["errors"] = errors

    proc = subprocess.Popen(command, **popen_kwargs)
    job_handle = bind_kill_on_close_job(_popen_win32_handle(proc))
    deadline = (
        time.monotonic() + timeout_s
        if timeout_s is not None and timeout_s > 0
        else None
    )
    slice_s = max(0.01, float(poll_interval_s))
    reaped = False
    try:
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
                state, job_handle = _reap(proc, job_handle)
                _track_unreleased(proc, command, state, job_handle, "cancel")
                job_handle = None
                reaped = True
                raise SubprocessCancelledError(
                    f"cancelled while running: {os.path.basename(str(command[0]))}"
                )
            if deadline is not None and time.monotonic() >= deadline:
                state, job_handle = _reap(proc, job_handle)
                _track_unreleased(proc, command, state, job_handle, "timeout")
                job_handle = None
                reaped = True
                raise subprocess.TimeoutExpired(command, float(timeout_s or 0.0))
    except BaseException:
        # Cancel and timeout already reaped and recorded why; re-reaping here
        # would only overwrite the specific reason with a generic one.
        if not reaped and (job_handle is not None or child_state(proc) != "gone"):
            state, job_handle = _reap(proc, job_handle)
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
