"""Parent-process client managing the persistent ASR GPU worker subprocess."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core import resources
from core.diagnostics import format_exit_code
from core.typed_config import env_float, env_int
from pipeline.gpu_oom_policy import (
    _budget_exceeded_warning,
    _effective_stage_batch_size,
    _oom_downshift,
    _oom_setting_for_error,
    _terminal_oom_detail,
)
from pipeline.gpu_worker_runtime import (
    _prune_old_worker_logs,
    _worker_log_path,
    worker_main,
)
from pipeline.stage_log import ASR_STAGE_HEARTBEAT_PREFIX, ASR_WORKER_DIAGNOSTIC_PREFIX
from utils.subprocess_tools import (
    bind_process_to_kill_on_close_job,
    close_kill_on_close_job,
    create_kill_on_close_job,
)

logger = logging.getLogger(__name__)
_EOF_EXIT_GRACE_S = 0.5
_MAX_IDLE_EXIT_RESTARTS = 2

# Env read by CUDA only at runtime/driver init, so changing it on a *running*
# persistent worker has no effect. The client restarts the worker when one of
# these differs from the values it was started under.
_CUDA_INIT_ENV_KEYS = ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")
_ASR_KIND = "asr_worker"

_env_float = env_float
_env_int = env_int


class GpuWorkerError(RuntimeError):
    def __init__(
        self,
        kind: str,
        detail: str,
        *,
        stage: str = "",
        runtime_tuning: dict[str, Any] | None = None,
    ):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail
        self.stage = str(stage or "")
        self.runtime_tuning = dict(runtime_tuning or {})


class GpuWorkerTimeoutError(GpuWorkerError):
    def __init__(self, detail: str):
        super().__init__("timeout", detail)


class _GpuWorkerDisconnectedError(GpuWorkerError):
    def __init__(self, detail: str):
        super().__init__("crash", detail)


class _GpuWorkerIdleExitError(GpuWorkerError):
    """The worker exited cleanly on idle timeout right as a request arrived."""

    def __init__(self, detail: str):
        super().__init__("idle_exit", detail)


def _facade_attr(name: str, default: Any) -> Any:
    facade = sys.modules.get("pipeline.gpu_worker")
    if facade is not None and hasattr(facade, name):
        return getattr(facade, name)
    return default


def _override_float(
    env_overrides: dict[str, str] | None,
    key: str,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
    if env_overrides and key in env_overrides:
        raw = str(env_overrides.get(key) or "").strip()
        if raw:
            try:
                value = float(raw)
                if minimum is not None:
                    value = max(minimum, value)
                return value
            except ValueError:
                pass
    return _env_float(key, default, minimum=minimum)


def _override_int(
    env_overrides: dict[str, str] | None,
    key: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    if env_overrides and key in env_overrides:
        raw = str(env_overrides.get(key) or "").strip()
        if raw:
            try:
                value = int(float(raw))
                if minimum is not None:
                    value = max(minimum, value)
                return value
            except ValueError:
                pass
    return _env_int(key, default, minimum=minimum)


def _worker_exit_diagnostics(process, *, wait_for_exit: bool = False) -> str:
    if process is None:
        return "pid=None exitcode=None"
    if wait_for_exit and process.exitcode is None:
        process.join(_EOF_EXIT_GRACE_S)
    exitcode = process.exitcode
    status = format_exit_code(exitcode)
    pid = process.pid
    log_path = _facade_attr("_worker_log_path", _worker_log_path)(pid)
    log_ref = f"logs/{log_path.name}"
    if pid is not None and not log_path.exists():
        log_ref += " (not created)"
    return f"pid={pid} exitcode={status} worker_log={log_ref}"


def _bind_worker_job_object(process: Any) -> int | None:
    """Create and bind a Windows kill-on-close Job Object without importing ``asr.local_backend``."""
    if os.name != "nt":
        return None
    create_fn = _facade_attr("create_kill_on_close_job", create_kill_on_close_job)
    bind_fn = _facade_attr(
        "bind_process_to_kill_on_close_job", bind_process_to_kill_on_close_job
    )
    job: int | None = None
    try:
        job = create_fn()
        bind_fn(job, process)
        return job
    except Exception as exc:
        logger.warning(
            "Could not assign ASR worker pid=%s to kill-on-close Job Object: %s",
            getattr(process, "pid", None),
            exc,
        )
        if job is not None:
            _facade_attr("close_kill_on_close_job", close_kill_on_close_job)(job)
        return None


def unclosed_job_handles() -> list[dict[str, Any]]:
    """Job Object handles from ASR children that would not close."""
    return [
        record
        for record in resources.open_handles()
        if record["kind"] == _ASR_KIND
    ]


def asr_gpu_cleanup_status() -> dict[str, Any]:
    """What still holds the GPU, if anything.

    Read by the web layer on every poll, so it must not take the worker lock -
    that one is held for the whole length of an ASR run.
    """
    records = resources.pending(_ASR_KIND)
    blocked = any(record["blocked"] for record in records)
    handles = unclosed_job_handles()
    worst = max(records, key=lambda record: record["attempts"], default=None)
    next_retry = [r["next_retry_in_s"] for r in records if r["next_retry_in_s"]]
    return {
        "state": (
            "cleanup_failed" if blocked else "cleaning" if records else "released"
        ),
        "blocked": blocked,
        "pid": worst["pid"] if worst else None,
        "attempts": worst["attempts"] if worst else 0,
        "last_error": worst["last_error"] if worst else "",
        "since": worst["since"] if worst else 0.0,
        "generation": worst["generation"] if worst else 0,
        "next_retry_in_s": min(next_retry) if next_retry else 0.0,
        "handle_open": bool(handles),
        "open_job_handles": len(handles),
    }


def asr_gpu_blocked() -> bool:
    """True while an ASR child we could not stop still owns the GPU."""
    return any(record["blocked"] for record in resources.pending(_ASR_KIND))


def retry_asr_gpu_cleanup() -> dict[str, Any]:
    """Manual 重新检查 / 重试清理 for ASR resources, then the current status."""
    resources.retry_pending(_ASR_KIND)
    return asr_gpu_cleanup_status()


class _GpuWorkerClient:
    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._conn = None
        self._job_handle = None
        # _kill_child runs from the request path, from the watchdog and from the
        # manual button; they must not tear down the same child concurrently.
        self._child_lock = threading.RLock()
        self._in_request = False
        # The id comes from the registry, so no two clients can ever name the
        # same resource - not even after this one is garbage collected.
        self._resource_token = resources.new_resource_id("asr-gpu-worker")
        self._child_generation = 0
        self.kill_grace_s = _env_float("ASR_STAGE_WORKER_KILL_GRACE_S", 5.0)
        # Snapshot of the CUDA-init-time env the live worker was started under,
        # so we can restart it when a later job needs different alloc/device.
        self._cuda_init_env: dict[str, str] | None = None

    def is_alive(self) -> bool:
        return (
            self._process is not None
            and self._process.is_alive()
            and self._conn is not None
        )

    def _close_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None

    def _resource_id(self) -> str:
        return self._resource_token

    def _ensure_registered_locked(self, process, generation: int) -> None:
        if resources.is_registered(self._resource_id(), generation):
            return
        resources.register(
            resource_id=self._resource_id(),
            generation=generation,
            kind=_ASR_KIND,
            description="asr-gpu-worker",
            process=process,
            job_handle=self._job_handle,
            stopper=self._stop_child,
            holds_gpu=True,
        )

    def _stop_child(self, process, job_handle) -> resources.StopOutcome:
        """The blocking part of stopping this client's child.

        Called by the resource registry, which owns the identity, the claim and
        the retry schedule. Everything here is about *this* process object: it
        never looks up "the current child", so a retry can only ever act on the
        one it was registered for.
        """
        close_job_fn = _facade_attr("close_kill_on_close_job", close_kill_on_close_job)
        detail = ""
        for step, grace_s in (("terminate", self.kill_grace_s), ("kill", 5.0)):
            if self._child_liveness(process) == "gone":
                break
            try:
                getattr(process, step)()
            except Exception as exc:
                # Keep escalating: the next step is stronger, and giving up here
                # is what left the child running.
                detail = f"{step}() failed: {exc!r}"
                logger.warning("ASR stage worker %s() failed: %r", step, exc)
            try:
                process.join(grace_s)
            except Exception:
                pass

        # A Job Object handle is a raw integer: assigning None to it leaks the
        # handle instead of closing it. Closing it is also the last escalation
        # available - the job is kill-on-close, so the whole tree goes with it,
        # including children terminate()/kill() never reach.
        if job_handle and close_job_fn(job_handle):
            job_handle = None
        elif job_handle:
            detail = detail or "job object handle could not be closed"
            logger.warning("ASR stage worker job object handle could not be closed")

        if self._child_liveness(process) != "gone":
            try:
                process.join(1)
            except Exception:
                pass
        liveness = self._child_liveness(process)
        if liveness != "gone":
            # The pid is diagnostics only - the process object is the identity,
            # and reusing a bare pid after it exits is how you kill an unrelated
            # process.
            pid = getattr(process, "pid", None)
            logger.warning("ASR stage worker did not exit; pid=%s %s", pid, liveness)
            detail = detail or f"child process is {liveness} after terminate and kill"
        with self._child_lock:
            # Only touch this client's own state when the record being stopped
            # *is* its current child. A retry for an older generation must not
            # reach in and clear the handle or the connection of the worker that
            # replaced it - the record owns what it was registered with.
            if self._process is process:
                self._close_conn()
                self._job_handle = job_handle
                if liveness == "gone":
                    self._process = None
        return resources.StopOutcome(
            process_state=liveness,
            job_handle=job_handle,
            detail=detail,
        )

    def _child_liveness(self, process) -> str:
        """"gone", "alive" or "unknown" - an error is never "gone".

        `is_alive()` raising (a process from another parent, a torn-down
        multiprocessing state) used to read as "already exited", which is the
        one answer that lets a caller start a second worker on the same GPU.
        """
        if process is None:
            return "gone"
        try:
            return "alive" if process.is_alive() else "gone"
        except Exception as exc:
            logger.warning("ASR worker liveness check failed: %r", exc)
            return "unknown"

    def _process_is_alive(self, process) -> bool:
        # "Not confirmed gone" is what every caller of this actually means:
        # unknown must keep the child tracked, not release it.
        return self._child_liveness(process) != "gone"

    def _kill_child(self) -> bool:
        """Stop this client's current child. True when no child remains.

        A thin request to the registry: it owns the identity, so a cleanup that
        was scheduled for an earlier child cannot reach this one, and two
        cleanups cannot run against the same child at once. What is left here is
        only "which child is mine right now".
        """
        with self._child_lock:
            generation = self._child_generation
            process = self._process
            if process is None:
                self._close_conn()
                return True
            # Invariant: a live child is always a registered resource. Restoring
            # it here rather than assuming it keeps "stop" reachable even if a
            # child was adopted outside `_start_worker`.
            self._ensure_registered_locked(process, generation)
        result = resources.request_stop(self._resource_id(), generation)
        if not result.stopped:
            # Keep `_process` so the next attempt retries this same child
            # instead of a caller treating the GPU as free and starting a
            # second worker.
            return False
        # Two separate facts, deliberately not conflated: the child is confirmed
        # gone, so the GPU is free; the job handle may still be open, which is a
        # tracked resource, not a reason to hold the queue.
        return True

    def has_unreleased_child(self) -> bool:
        """A child we failed to kill is still ours - do not replace the client."""
        return self._process_is_alive(self._process)

    def close(self) -> bool:
        """Stop the child. Returns whether it is confirmed gone."""
        process_to_join = None
        with self._child_lock:
            conn = self._conn
            process = self._process
            busy = self._in_request
            # When the child is mid-request or already dead, skip the graceful
            # shutdown message + 5s wait and go straight to `_kill_child()`.
            if (
                not busy
                and conn is not None
                and process is not None
                and self._child_liveness(process) == "alive"
            ):
                try:
                    conn.send({"op": "shutdown"})
                    process_to_join = process
                except Exception:
                    pass
        if process_to_join is not None:
            try:
                process_to_join.join(5)
            except Exception:
                pass
        return self._kill_child()

    def _effective_cuda_init_env(
        self,
        env_overrides: dict[str, str] | None,
    ) -> dict[str, str]:
        env_overrides = env_overrides or {}
        result: dict[str, str] = {}
        for key in _CUDA_INIT_ENV_KEYS:
            if key in env_overrides:
                result[key] = str(env_overrides[key])
            else:
                result[key] = os.environ.get(key, "")
        return result

    def _start_worker(
        self,
        cancel_requested: Callable[[], bool] | None = None,
        *,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        # Prune stale historical worker logs on startup (best-effort).
        try:
            _prune_old_worker_logs()
        except Exception:
            pass

        self.kill_grace_s = _override_float(
            env_overrides,
            "ASR_STAGE_WORKER_KILL_GRACE_S",
            self.kill_grace_s,
            minimum=0.1,
        )

        # Everything that replaces this client's child runs under the same lock
        # a cleanup takes, so "is the generation still the one I was scheduled
        # for" cannot go stale between the check and the kill. The lock is
        # released before the ready wait: cancelling a slow startup has to stay
        # reachable.
        with self._child_lock:
            if not self._kill_child():
                status = asr_gpu_cleanup_status()
                raise GpuWorkerError(
                    "cleanup_failed",
                    "上一个 ASR GPU 子进程仍未退出，无法启动新的 GPU 进程；"
                    "已暂停派发新的 GPU 任务，并在后台按退避间隔继续重试清理"
                    f"（pid={status.get('pid')}，已重试 {status.get('attempts')} 次，"
                    f"原因：{status.get('last_error') or '未知'}）。"
                    "可在任务页点击「重试清理」或「重新检查」。",
                )
            parent_conn, child_conn = self._ctx.Pipe(duplex=True)
            # daemon=True off-Windows so a SIGKILL'd/orphaned parent still reaps
            # the GPU child (no Job Object there). Windows keeps daemon=False and
            # relies on the kill-on-close Job Object assigned below.
            process = self._ctx.Process(
                target=worker_main,
                args=(child_conn,),
                daemon=(os.name != "nt"),
            )
            process.start()
            self._child_generation += 1
            # The previous child's handle, if it would not close, stayed with
            # its own registry record - a record this client can no longer
            # reach. Binding a new one here therefore cannot overwrite it.
            self._job_handle = None
            # Registered before the handle is bound: from this moment the child
            # has an identity, and every later stop request names it.
            resources.register(
                resource_id=self._resource_id(),
                generation=self._child_generation,
                kind=_ASR_KIND,
                description="asr-gpu-worker",
                process=process,
                job_handle=None,
                stopper=self._stop_child,
                holds_gpu=True,
            )
            if os.name == "nt":
                job = _bind_worker_job_object(process)
                if job is not None:
                    attached = resources.attach_handle(
                        self._resource_id(), self._child_generation, job
                    )
                    if attached:
                        self._job_handle = job
                    else:
                        # Record was already retired concurrently; close the
                        # handle immediately instead of leaking it.
                        _facade_attr(
                            "close_kill_on_close_job", close_kill_on_close_job
                        )(job)
                        self._job_handle = None
            child_conn.close()
            self._process = process
            self._conn = parent_conn

        ready_timeout_s = _override_float(
            env_overrides,
            "ASR_STAGE_WORKER_READY_TIMEOUT_S",
            60.0,
            minimum=0.1,
        )
        deadline = time.monotonic() + ready_timeout_s
        while True:
            # Startup is part of the request's cancellable lifetime: model load
            # can take tens of seconds, and without this the user's 取消 was only
            # noticed after the worker came up and the job had been sent.
            if cancel_requested is not None and cancel_requested():
                self._kill_child()
                raise GpuWorkerError("cancelled", "ASR stage worker cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_child()
                raise GpuWorkerTimeoutError(
                    f"ASR stage worker ready timeout after {ready_timeout_s:.1f}s"
                )
            try:
                if parent_conn.poll(min(0.5, remaining)):
                    message = parent_conn.recv()
                    break
            except (EOFError, OSError) as exc:
                diagnostics = _worker_exit_diagnostics(process, wait_for_exit=True)
                self._kill_child()
                raise GpuWorkerError(
                    "crash",
                    f"ASR stage worker pipe closed before ready: {diagnostics}",
                ) from exc
            if process.exitcode is not None:
                diagnostics = _worker_exit_diagnostics(process)
                self._kill_child()
                raise GpuWorkerError(
                    "crash",
                    f"ASR stage worker exited before ready: {diagnostics}",
                )

        if not isinstance(message, dict):
            self._kill_child()
            raise GpuWorkerError("protocol_error", "ready message is not a dict")
        if message.get("op") != "ready":
            kind = str(message.get("kind") or "protocol_error")
            detail = str(message.get("detail") or f"unexpected ready message: {message!r}")
            self._kill_child()
            raise GpuWorkerError(kind, detail)

    def _ensure_worker(
        self,
        cancel_requested: Callable[[], bool] | None = None,
        *,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        if self.is_alive():
            return
        self._start_worker(cancel_requested, env_overrides=env_overrides)

    def _drain_pending_idle_exit(self) -> bool:
        """Return True if the live worker already sent ``idle_exit`` while idle."""
        conn = self._conn
        if conn is None:
            return False
        try:
            if not conn.poll(0):
                return False
            msg = conn.recv()
        except (EOFError, OSError):
            return True
        return isinstance(msg, dict) and msg.get("op") == "idle_exit"

    def _send_request(
        self,
        payload: dict[str, Any],
        env_overrides: dict[str, str] | None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        """Send a request, transparently restarting once if the pipe is dead.

        A persistent worker can self-exit between jobs (idle timeout) or be
        killed by an OOM retry; rather than surfacing that as a crash, restart
        and resend once before giving up. Also records the CUDA-init-time env
        the (re)started worker is running under.
        """
        for attempt in range(2):
            self._ensure_worker(cancel_requested, env_overrides=env_overrides)
            if self._drain_pending_idle_exit():
                self._kill_child()
                self._ensure_worker(cancel_requested, env_overrides=env_overrides)
            conn = self._conn
            if conn is None:
                continue
            try:
                conn.send(payload)
                self._cuda_init_env = self._effective_cuda_init_env(env_overrides)
                return
            except (EOFError, OSError) as exc:
                diagnostics = _worker_exit_diagnostics(self._process, wait_for_exit=True)
                logger.warning("ASR worker request send failed: %s; error=%r", diagnostics, exc)
                stopped = self._kill_child()
                if not stopped or attempt == 1:
                    raise GpuWorkerError(
                        "crash",
                        f"ASR stage worker send failed: {diagnostics}; error={exc!r}",
                    ) from exc
                # Loop: _ensure_worker starts a fresh worker and we resend.

    def _transcribe_and_align_once(
        self,
        audio_path: str,
        *,
        device: str = "auto",
        env_overrides: dict[str, str] | None = None,
        job_id: str = "",
        run_id: str = "",
        on_stage: Callable[[str], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> tuple[list[dict], list[str], dict]:
        # If CUDA-init-time env changed since the worker started (e.g. a new
        # PYTORCH_CUDA_ALLOC_CONF), restart it: those keys are only read at CUDA
        # runtime init and would otherwise be frozen for the worker's lifetime.
        if self.is_alive() and self._cuda_init_env is not None:
            if self._effective_cuda_init_env(env_overrides) != self._cuda_init_env:
                self._kill_child()

        request_id = str(job_id or uuid.uuid4().hex[:8])
        payload = {
            "op": "transcribe_and_align",
            "job_id": request_id,
            "run_id": str(run_id or ""),
            "audio_path": str(Path(audio_path).resolve()),
            "device": str(device or "auto"),
            "env": dict(env_overrides or {}),
        }
        self._send_request(payload, env_overrides, cancel_requested)
        conn = self._conn
        process = self._process
        if conn is None:
            raise GpuWorkerError("crash", "ASR stage worker connection is unavailable")

        timeout_s = _override_float(
            env_overrides,
            "ASR_STAGE_WORKER_TIMEOUT_S",
            0.0,
            minimum=0.0,
        )
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        heartbeat_s = _override_float(
            env_overrides,
            "ASR_STAGE_WORKER_HEARTBEAT_S",
            10.0,
            minimum=0.0,
        )
        request_started = time.monotonic()
        last_stage_at = request_started
        last_heartbeat_at = request_started
        last_stage_message = "GPU worker request accepted"

        # One cleanup site for the whole wait. Every branch below used to have
        # to remember to kill the child before raising, and the heartbeat
        # callback did not: a cancel that landed between the top-of-loop check
        # and the heartbeat raised out of `on_stage` and left the GPU worker
        # transcribing with nobody listening.
        self._in_request = True
        try:
            while True:
                if cancel_requested is not None and cancel_requested():
                    raise GpuWorkerError("cancelled", "ASR stage worker cancelled")
                if deadline is not None and time.monotonic() >= deadline:
                    raise GpuWorkerTimeoutError(
                        f"ASR stage worker timeout after {timeout_s:.1f}s"
                    )

                wait_s = 0.25
                if deadline is not None:
                    wait_s = max(0.05, min(wait_s, deadline - time.monotonic()))
                try:
                    ready = conn.poll(wait_s)
                    message = conn.recv() if ready else None
                except (EOFError, OSError) as exc:
                    diagnostics = _worker_exit_diagnostics(process, wait_for_exit=True)
                    raise _GpuWorkerDisconnectedError(
                        f"ASR stage worker pipe closed before result: {diagnostics}; "
                        f"last_step={last_stage_message!r}; pipe_error={exc!r}"
                    ) from exc

                if not ready:
                    if process is not None and process.exitcode is not None:
                        diagnostics = _worker_exit_diagnostics(process)
                        raise _GpuWorkerDisconnectedError(
                            f"ASR stage worker exited before result: {diagnostics}; "
                            f"last_step={last_stage_message!r}"
                        )
                    now = time.monotonic()
                    if (
                        heartbeat_s > 0.0
                        and on_stage is not None
                        and now - last_heartbeat_at >= heartbeat_s
                    ):
                        on_stage(
                            f"{ASR_STAGE_HEARTBEAT_PREFIX} "
                            f"current={last_stage_message} "
                            f"elapsed={now - request_started:.1f}s "
                            f"idle={now - last_stage_at:.1f}s"
                        )
                        last_heartbeat_at = now
                    continue

                if not isinstance(message, dict):
                    raise GpuWorkerError("protocol_error", "worker message is not a dict")

                op = message.get("op")
                if op == "idle_exit":
                    raise _GpuWorkerIdleExitError(
                        "ASR stage worker exited cleanly on idle timeout"
                    )

                if str(message.get("job_id") or "") != request_id:
                    continue

                if op == "stage":
                    last_stage_message = str(message.get("message") or "")
                    last_stage_at = time.monotonic()
                    last_heartbeat_at = last_stage_at
                    if on_stage is not None:
                        on_stage(last_stage_message)
                    continue

                if op == "result":
                    segments = message.get("segments")
                    asr_log = message.get("asr_log")
                    asr_details = message.get("asr_details")
                    if not isinstance(segments, list):
                        raise GpuWorkerError("protocol_error", "segments must be a list")
                    if not isinstance(asr_log, list):
                        raise GpuWorkerError("protocol_error", "asr_log must be a list")
                    if not isinstance(asr_details, dict):
                        raise GpuWorkerError("protocol_error", "asr_details must be a dict")
                    return segments, [str(item) for item in asr_log], dict(asr_details)

                if op == "error":
                    kind = str(message.get("kind") or "crash")
                    detail = str(message.get("detail") or "ASR stage worker error")
                    stage = str(message.get("stage") or "")
                    runtime_tuning = message.get("runtime_tuning")
                    raise GpuWorkerError(
                        kind,
                        detail,
                        stage=stage,
                        runtime_tuning=(
                            runtime_tuning if isinstance(runtime_tuning, dict) else {}
                        ),
                    )

                raise GpuWorkerError("protocol_error", f"unexpected worker op: {op}")
        except BaseException:
            self._kill_child()
            raise
        finally:
            self._in_request = False

    def transcribe_and_align(
        self,
        audio_path: str,
        *,
        device: str = "auto",
        env_overrides: dict[str, str] | None = None,
        job_id: str = "",
        run_id: str = "",
        on_stage: Callable[[str], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> tuple[list[dict], list[str], dict]:
        current_env = dict(env_overrides or {})
        retry_limit = _override_int(
            current_env,
            "ASR_STAGE_WORKER_OOM_RETRY_LIMIT",
            6,
            minimum=0,
        )
        attempts_by_setting: dict[str, int] = {}
        retry_records: list[dict[str, Any]] = []
        retried_disconnect = False
        idle_exit_restarts = 0
        oom_downshift_fn = _facade_attr("_oom_downshift", _oom_downshift)
        terminal_oom_fn = _facade_attr("_terminal_oom_detail", _terminal_oom_detail)
        budget_warn_fn = _facade_attr("_budget_exceeded_warning", _budget_exceeded_warning)

        while True:
            try:
                segments, asr_log, asr_details = self._transcribe_and_align_once(
                    audio_path,
                    device=device,
                    env_overrides=current_env,
                    job_id=job_id,
                    run_id=run_id,
                    on_stage=on_stage,
                    cancel_requested=cancel_requested,
                )
                # A successful result is final -- never discard it for a VRAM
                # budget. Surface a non-fatal note if the reserved pool spiked,
                # but keep the segments. Hard OOM is caught below / mid-pipeline.
                warning = budget_warn_fn(asr_details, current_env)
                if warning and on_stage is not None:
                    try:
                        on_stage(warning)
                    except Exception:
                        logger.warning("Failed to report VRAM budget warning via on_stage")
                if retry_records:
                    stage_worker = asr_details.setdefault("stage_worker", {})
                    if isinstance(stage_worker, dict):
                        stage_worker["oom_retries"] = list(retry_records)
                return segments, asr_log, asr_details
            except _GpuWorkerIdleExitError as exc:
                if idle_exit_restarts >= _MAX_IDLE_EXIT_RESTARTS or self.has_unreleased_child():
                    raise GpuWorkerError("crash", exc.detail) from exc
                if cancel_requested is not None and cancel_requested():
                    raise GpuWorkerError("cancelled", "ASR stage worker cancelled") from exc
                idle_exit_restarts += 1
                logger.info(
                    "ASR worker exited cleanly on idle timeout while accepting job_id=%s; "
                    "restarting without consuming disconnect retry budget",
                    job_id,
                )
                continue
            except _GpuWorkerDisconnectedError as exc:
                if retried_disconnect or self.has_unreleased_child():
                    raise
                if cancel_requested is not None and cancel_requested():
                    raise GpuWorkerError("cancelled", "ASR stage worker cancelled") from exc
                retried_disconnect = True
                logger.warning("ASR worker disconnected, restarting once: %s", exc.detail)
                if on_stage is not None:
                    on_stage(
                        f"{ASR_WORKER_DIAGNOSTIC_PREFIX} · 连接中断，"
                        f"重启并尝试从缓存续跑一次。{exc.detail}"
                    )
            except GpuWorkerError as exc:
                if str(getattr(exc, "kind", "")) != "oom":
                    raise
                lowered = oom_downshift_fn(current_env, exc)
                if lowered is None:
                    failed_setting, tuning_key = _oom_setting_for_error(
                        str(exc.stage or ""),
                        str(exc.detail or ""),
                    )
                    batch_size = int(exc.runtime_tuning.get(tuning_key) or 0) or (
                        _effective_stage_batch_size(current_env, failed_setting)
                    )
                    raise GpuWorkerError(
                        "oom",
                        terminal_oom_fn(
                            env=current_env,
                            detail=str(getattr(exc, "detail", exc)),
                            batch_size=batch_size,
                            retry_records=retry_records,
                            failed_setting=failed_setting,
                        ),
                    ) from exc
                next_env, setting, previous_value, next_value = lowered
                setting_attempt = attempts_by_setting.get(setting, 0)
                if setting_attempt >= retry_limit:
                    _, tuning_key = _oom_setting_for_error(
                        str(exc.stage or ""),
                        str(exc.detail or ""),
                    )
                    batch_size = int(exc.runtime_tuning.get(tuning_key) or 0) or (
                        _effective_stage_batch_size(current_env, setting)
                    )
                    raise GpuWorkerError(
                        "oom",
                        terminal_oom_fn(
                            env=current_env,
                            detail=str(getattr(exc, "detail", exc)),
                            batch_size=batch_size,
                            retry_records=retry_records,
                            retry_limit_exhausted=True,
                            failed_setting=setting,
                        ),
                    ) from exc
                setting_attempt += 1
                attempts_by_setting[setting] = setting_attempt
                record = {
                    "attempt": len(retry_records) + 1,
                    "stage": str(exc.stage or ""),
                    "setting": setting,
                    "previous_value": previous_value,
                    "next_value": next_value,
                    "detail": str(getattr(exc, "detail", exc)),
                }
                if setting == "ASR_BATCH_SIZE":
                    record["previous_batch_size"] = int(previous_value)
                    record["next_batch_size"] = int(next_value)
                retry_records.append(record)
                current_env = next_env
                if on_stage is not None:
                    on_stage(
                        "GPU worker OOM; restarting worker and resuming from "
                        f"ASR result cache with {setting} "
                        f"{previous_value:g}->{next_value:g}"
                    )
                continue
