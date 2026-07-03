from __future__ import annotations

import atexit
import gc
import multiprocessing as mp
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable


class GpuWorkerError(RuntimeError):
    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


class GpuWorkerTimeoutError(GpuWorkerError):
    def __init__(self, detail: str):
        super().__init__("timeout", detail)


LOW_VRAM_ASR_BACKEND = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _safe_send(conn: Connection, message: dict[str, Any]) -> bool:
    try:
        conn.send(message)
        return True
    except Exception:
        return False


def _is_oom_error(exc: BaseException, torch_module: Any | None) -> bool:
    if torch_module is not None:
        try:
            if isinstance(exc, torch_module.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass

    detail = repr(exc).lower()
    return (
        ("out of memory" in detail and ("cuda" in detail or "gpu" in detail))
        or "gpu vram budget exceeded" in detail
        or "vram budget exceeded" in detail
        or "cumemalloc" in detail
    )


@contextmanager
def _temporary_env(overrides: dict[str, str]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[str(key)] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _env_payload(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in raw.items()
        if str(key).strip() and value is not None
    }


def _resolve_device(device: str) -> str:
    requested = str(device or "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _clear_worker_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_asr_pipeline_for_request():
    # No per-request reload. The settings that vary per job (ASR_BACKEND,
    # ASR_BATCH_SIZE, dtype/attention, boundary/pre-asr-cueqc flags) are all read
    # from env at CALL time -- current_asr_backend(), _resolve_asr_batch_size(),
    # _detect_dtype(), _boundary_config(), pre_asr_cueqc.enabled(), ... -- so a
    # persistent worker picks up each job's _temporary_env without re-importing.
    # The only env frozen at CUDA init (PYTORCH_CUDA_ALLOC_CONF,
    # CUDA_VISIBLE_DEVICES) is handled by the client's restart-on-change.
    from asr import pipeline as asr_pipeline

    return asr_pipeline


class GpuModelManager:
    """Owns GPU lifecycle boundaries inside the unified worker process."""

    def __init__(
        self,
        *,
        pid: int,
        on_stage: Callable[[str], None] | None = None,
    ) -> None:
        self.pid = int(pid)
        self.on_stage = on_stage
        self.events: list[dict[str, Any]] = []

    def _record(self, *, stage: str, action: str, **extra: Any) -> None:
        event = {
            "ts": round(time.time(), 3),
            "stage": str(stage),
            "action": str(action),
        }
        event.update(extra)
        self.events.append(event)

    def reset_cuda_state(self, *, reason: str) -> dict[str, Any]:
        gc.collect()
        snapshot: dict[str, Any] = {"reason": reason, "cuda_available": False}
        try:
            import torch

            snapshot["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                device_index = torch.cuda.current_device()
                scale = 1024 * 1024
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                snapshot.update(
                    {
                        "device_index": int(device_index),
                        "allocated_mb": round(
                            torch.cuda.memory_allocated(device_index) / scale,
                            1,
                        ),
                        "reserved_mb": round(
                            torch.cuda.memory_reserved(device_index) / scale,
                            1,
                        ),
                        "free_mb": round(free_bytes / scale, 1),
                        "total_mb": round(total_bytes / scale, 1),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            snapshot["error"] = f"{type(exc).__name__}: {exc}"
        self._record(stage="cuda", action="reset", **snapshot)
        return snapshot

    def lifecycle_event(self, *, stage: str, action: str) -> None:
        stage = str(stage)
        action = str(action)
        if stage == "asr" and action == "load_exclusive":
            self.reset_cuda_state(reason="before_asr_exclusive_load")
        self._record(stage=stage, action=action)
        if action == "unload":
            self.reset_cuda_state(reason=f"after_{stage}_unload")

    def run_transcribe_and_align(
        self,
        *,
        audio_path: str,
        requested_device: str,
        mock: bool,
    ) -> tuple[list[dict], list[str], dict]:
        self.reset_cuda_state(reason="before_request")
        device = _resolve_device(requested_device)
        try:
            import torch

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        if mock:
            if self.on_stage is not None:
                self.on_stage("ASR stage worker mock")
            segments, asr_log, asr_details = _mock_result(
                audio_path,
                device,
                pid=self.pid,
            )
        else:
            asr_pipeline = _load_asr_pipeline_for_request()
            segments, asr_log, asr_details = asr_pipeline.transcribe_and_align(
                audio_path,
                device,
                on_stage=self.on_stage,
                include_details=True,
                model_manager=self,
            )
            asr_details = dict(asr_details or {})

        self.reset_cuda_state(reason="after_request")
        asr_details.setdefault("device", device)
        asr_details["stage_worker"] = {
            "mode": "gpu_worker",
            "process_model": "persistent_subprocess",
            "pid": self.pid,
            "mock": bool(mock),
            "gpu_owner": True,
            "model_manager": {
                "policy": "sequential_exclusive",
                "events": list(self.events),
            },
        }
        return segments, asr_log, asr_details


def _mock_result(audio_path: str, device: str, *, pid: int) -> tuple[list[dict], list[str], dict]:
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "mock",
            "words": [{"start": 0.0, "end": 1.0, "word": "mock"}],
        }
    ]
    log = ["ASR stage worker mock result"]
    details = {
        "backend": "mock",
        "audio_path": audio_path,
        "device": device,
        "chunk_count": 1,
        "transcript_chunks": [{"text": "mock"}],
        "stage_timings": {},
        "cuda_memory": [],
        "word_count": 1,
        "segment_count": 1,
        "stage_worker": {
            "mode": "gpu_worker",
            "process_model": "persistent_subprocess",
            "pid": pid,
            "mock": True,
        },
    }
    return segments, log, details


def _parse_batch_mapping(raw: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        repo_id, value = item.rsplit("=", 1)
        try:
            mapping[repo_id.strip()] = max(1, int(float(value.strip())))
        except (TypeError, ValueError):
            continue
    return mapping


def _effective_asr_batch_size(env: dict[str, str]) -> int | None:
    raw = str(env.get("ASR_BATCH_SIZE") or os.getenv("ASR_BATCH_SIZE", "auto")).strip()
    if raw and raw.lower() != "auto":
        try:
            return max(1, int(float(raw)))
        except (TypeError, ValueError):
            return None
    try:
        from asr.backends.qwen import (
            DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO,
            qwen_asr_repo_id,
        )

        backend = str(env.get("ASR_BACKEND") or os.getenv("ASR_BACKEND", "")).strip()
        repo_id = qwen_asr_repo_id(backend or None)
        mapping = dict(DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO)
        mapping.update(
            _parse_batch_mapping(
                str(
                    env.get("ASR_BATCH_SIZE_BY_REPO")
                    or os.getenv("ASR_BATCH_SIZE_BY_REPO", "")
                )
            )
        )
        return max(1, int(mapping[repo_id]))
    except Exception:
        return None


def _downshift_asr_batch_env(env: dict[str, str]) -> tuple[dict[str, str], int, int] | None:
    current = _effective_asr_batch_size(env)
    if current is None or current <= 1:
        return None
    lowered = max(1, current // 2)
    if lowered >= current:
        return None
    updated = dict(env)
    updated["ASR_BATCH_SIZE"] = str(lowered)
    return updated, current, lowered


def _terminal_oom_detail(
    *,
    env: dict[str, str],
    detail: str,
    batch_size: int | None,
    retry_records: list[dict[str, Any]],
    retry_limit_exhausted: bool = False,
) -> str:
    backend = str(env.get("ASR_BACKEND") or os.getenv("ASR_BACKEND", "")).strip()
    budget_mb = _vram_budget_for_env(env)
    retry_summary = ""
    if retry_records:
        steps = [
            f"{record.get('previous_batch_size')}->{record.get('next_batch_size')}"
            for record in retry_records
        ]
        retry_summary = f" 已尝试降 batch：{', '.join(steps)}。"
    budget_text = f" 当前显存预算为 {budget_mb:.0f}MB。" if budget_mb > 0 else ""
    if batch_size is not None and batch_size <= 1:
        headline = "GPU 显存不足：ASR_BATCH_SIZE 已降到 1 后仍然 OOM，任务已停止。"
    elif retry_limit_exhausted:
        headline = "GPU 显存不足：OOM 自动降 batch 重试次数已用尽，任务已停止。"
    else:
        headline = "GPU 显存不足：无法继续降低 ASR_BATCH_SIZE，任务已停止。"

    if LOW_VRAM_ASR_BACKEND not in backend:
        action = (
            "建议在前端将 ASR 后端切换为 0.6B 低显存档 "
            f"({LOW_VRAM_ASR_BACKEND}) 后重试；也可以关闭其他占用 GPU 的程序。"
        )
    else:
        action = (
            "当前已经是 0.6B 低显存档；建议关闭其他占用 GPU 的程序，"
            "或降低显存预算/检查是否有异常共享显存占用后重试。"
        )
    return (
        f"{headline}{retry_summary}{budget_text}{action} "
        f"原始错误：{detail}"
    )


def _vram_budget_for_env(env: dict[str, str]) -> float:
    raw = str(
        env.get("ASR_STAGE_WORKER_VRAM_BUDGET_MB")
        or os.getenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "0")
    ).strip().lower()
    if raw in {"", "0", "false", "no", "off", "none"}:
        return 0.0
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _cuda_peak_mb_from_details(
    asr_details: dict[str, Any],
    *,
    metric: str,
) -> float | None:
    """Peak VRAM (MB) across worker-reported snapshots for one metric.

    metric="reserved"  -> max of max_reserved_mb / reserved_mb (allocator pool).
    metric="allocated" -> max of max_allocated_mb / allocated_mb (working set).
    """
    keys = {
        "reserved": ("max_reserved_mb", "reserved_mb"),
        "allocated": ("max_allocated_mb", "allocated_mb"),
    }[metric]
    values: list[float] = []
    snapshots = asr_details.get("cuda_memory")
    if isinstance(snapshots, list):
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            for key in keys:
                try:
                    values.append(float(snapshot.get(key)))
                except (TypeError, ValueError):
                    continue
    return round(max(values), 1) if values else None


def _budget_exceeded_warning(
    asr_details: dict[str, Any],
    env: dict[str, str],
) -> str:
    """Non-fatal note when the allocator's *reserved* pool exceeded the budget.

    A job that completed successfully is never retroactively turned into an OOM:
    the reserved pool routinely fills dedicated VRAM on a 6GB card without any
    spill to shared memory, and reserved does not respond to batch downshift.
    Real enforcement is allocated-based and happens mid-pipeline
    (asr/pipeline.py). This only surfaces the peak so it can be logged.
    """
    budget_mb = _vram_budget_for_env(env)
    if budget_mb <= 0.0:
        return ""
    peak_reserved = _cuda_peak_mb_from_details(asr_details, metric="reserved")
    if peak_reserved is None or peak_reserved <= budget_mb:
        return ""
    peak_allocated = _cuda_peak_mb_from_details(asr_details, metric="allocated")
    alloc_text = f"{peak_allocated:.0f}MB" if peak_allocated is not None else "n/a"
    return (
        f"GPU reserved VRAM peaked at {peak_reserved:.0f}MB "
        f"(budget {budget_mb:.0f}MB, allocated peak {alloc_text}); "
        "job completed successfully, no retry triggered."
    )


def worker_main(parent_conn: Connection) -> None:
    sys.stdout = sys.stderr
    pid = os.getpid()
    if not _safe_send(parent_conn, {"op": "ready", "pid": pid}):
        return

    # Idle self-exit (read from the inherited process env, not per-job
    # overrides -- this is a process-lifecycle knob, see config.py). When the
    # worker has had no inbound request for MAX_IDLE_S seconds, exit so the next
    # job starts from a clean CUDA state. The client detects the exit via
    # is_alive() and restarts transparently on the next request. A per-job
    # restart cadence is intentionally NOT provided: every job already does
    # gc.collect() + empty_cache() on completion, so VRAM does not accumulate
    # across jobs and a job counter would only add cold-start cost.
    idle_timeout_s = _env_float("ASR_STAGE_WORKER_MAX_IDLE_S", 300.0)

    while True:
        try:
            if idle_timeout_s > 0.0 and not parent_conn.poll(idle_timeout_s):
                _clear_worker_cuda()
                return
            msg = parent_conn.recv()
        except (EOFError, OSError):
            return

        if not isinstance(msg, dict):
            if not _safe_send(
                parent_conn,
                {
                    "op": "error",
                    "job_id": "",
                    "kind": "protocol_error",
                    "detail": "message must be an object",
                },
            ):
                return
            continue

        op = msg.get("op")
        if op == "shutdown":
            _clear_worker_cuda()
            raise SystemExit(0)

        job_id = str(msg.get("job_id") or "")
        if op != "transcribe_and_align":
            _safe_send(
                parent_conn,
                {
                    "op": "error",
                    "job_id": job_id,
                    "kind": "protocol_error",
                    "detail": f"unknown op: {op}",
                },
            )
            continue

        audio_path = str(msg.get("audio_path") or "")
        if not audio_path:
            _safe_send(
                parent_conn,
                {
                    "op": "error",
                    "job_id": job_id,
                    "kind": "protocol_error",
                    "detail": "missing audio_path",
                },
            )
            continue

        env = _env_payload(msg.get("env"))
        requested_device = str(msg.get("device") or "auto")
        torch_module = None
        try:
            with _temporary_env(env):
                try:
                    import torch as imported_torch

                    torch_module = imported_torch
                except Exception:
                    torch_module = None

                def _on_stage(message: str) -> None:
                    _safe_send(
                        parent_conn,
                        {
                            "op": "stage",
                            "job_id": job_id,
                            "message": str(message),
                        },
                    )

                mock = os.getenv("ASR_STAGE_WORKER_MOCK", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                model_manager = GpuModelManager(pid=pid, on_stage=_on_stage)
                segments, asr_log, asr_details = model_manager.run_transcribe_and_align(
                    audio_path=audio_path,
                    requested_device=requested_device,
                    mock=mock,
                )

                _clear_worker_cuda()
                if not _safe_send(
                    parent_conn,
                    {
                        "op": "result",
                        "job_id": job_id,
                        "segments": segments,
                        "asr_log": asr_log,
                        "asr_details": asr_details,
                        "device": str(asr_details.get("device") or "auto"),
                    },
                ):
                    return
        except Exception as exc:
            kind = "oom" if _is_oom_error(exc, torch_module) else "crash"
            _safe_send(
                parent_conn,
                {
                    "op": "error",
                    "job_id": job_id,
                    "kind": kind,
                    "detail": repr(exc),
                },
            )
            _clear_worker_cuda()
            raise SystemExit(0)


# Env read by CUDA only at runtime/driver init, so changing it on a *running*
# persistent worker has no effect. The client restarts the worker when one of
# these differs from the values it was started under.
_CUDA_INIT_ENV_KEYS = ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")


class _GpuWorkerClient:
    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._conn = None
        self._job_handle = None
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

    def _kill_child(self) -> None:
        process = self._process
        self._process = None
        self._job_handle = None
        if process is not None:
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(self.kill_grace_s)
                if process.is_alive():
                    process.kill()
                    process.join(5)
                if process.is_alive():
                    process.join(1)
            except Exception:
                pass
        self._close_conn()

    def close(self) -> None:
        conn = self._conn
        process = self._process
        try:
            if conn is not None and process is not None and process.is_alive():
                conn.send({"op": "shutdown"})
                process.join(5)
        except Exception:
            pass
        finally:
            if process is not None and process.is_alive():
                self._kill_child()
            else:
                self._process = None
                self._job_handle = None
                self._close_conn()

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

    def _start_worker(self) -> None:
        self._kill_child()
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        # daemon=True off-Windows so a SIGKILL'd/orphaned parent still reaps the
        # GPU child (no Job Object there). Windows keeps daemon=False and relies
        # on the kill-on-close Job Object assigned below.
        process = self._ctx.Process(
            target=worker_main,
            args=(child_conn,),
            daemon=(os.name != "nt"),
        )
        process.start()
        self._job_handle = None
        if os.name == "nt":
            try:
                from asr.local_backend import (
                    _assign_process_to_job_object,
                    _create_kill_on_close_job_object,
                )

                job = _create_kill_on_close_job_object()
                _assign_process_to_job_object(job, process)
                self._job_handle = job
            except Exception:
                self._job_handle = None
        child_conn.close()
        self._process = process
        self._conn = parent_conn

        ready_timeout_s = _env_float("ASR_STAGE_WORKER_READY_TIMEOUT_S", 60.0)
        deadline = time.monotonic() + ready_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_child()
                raise GpuWorkerTimeoutError(
                    f"ASR stage worker ready timeout after {ready_timeout_s:.1f}s"
                )
            if parent_conn.poll(min(0.5, remaining)):
                break
            if process.exitcode is not None:
                exitcode = process.exitcode
                self._kill_child()
                raise GpuWorkerError(
                    "crash",
                    f"ASR stage worker exited before ready: {exitcode}",
                )

        try:
            message = parent_conn.recv()
        except EOFError as exc:
            exitcode = process.exitcode
            self._kill_child()
            raise GpuWorkerError(
                "crash",
                f"ASR stage worker pipe closed before ready exitcode={exitcode}",
            ) from exc

        if not isinstance(message, dict):
            self._kill_child()
            raise GpuWorkerError("protocol_error", "ready message is not a dict")
        if message.get("op") != "ready":
            kind = str(message.get("kind") or "protocol_error")
            detail = str(message.get("detail") or f"unexpected ready message: {message!r}")
            self._kill_child()
            raise GpuWorkerError(kind, detail)

    def _ensure_worker(self) -> None:
        if self.is_alive():
            return
        self._start_worker()

    def _send_request(self, payload: dict[str, Any], env_overrides: dict[str, str] | None) -> None:
        """Send a request, transparently restarting once if the pipe is dead.

        A persistent worker can self-exit between jobs (idle timeout / max-jobs
        cadence) or be killed by an OOM retry; rather than surfacing that as a
        crash, restart and resend once before giving up. Also records the
        CUDA-init-time env the (re)started worker is running under.
        """
        for attempt in range(2):
            self._ensure_worker()
            conn = self._conn
            if conn is None:
                continue
            try:
                conn.send(payload)
                self._cuda_init_env = self._effective_cuda_init_env(env_overrides)
                return
            except (BrokenPipeError, EOFError, OSError) as exc:
                exitcode = self._process.exitcode if self._process is not None else None
                self._kill_child()
                if attempt == 1:
                    raise GpuWorkerError(
                        "crash",
                        f"ASR stage worker send failed exitcode={exitcode}: {exc!r}",
                    ) from exc
                # Loop: _ensure_worker starts a fresh worker and we resend.

    def _transcribe_and_align_once(
        self,
        audio_path: str,
        *,
        device: str = "auto",
        env_overrides: dict[str, str] | None = None,
        job_id: str = "",
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
            "audio_path": str(Path(audio_path).resolve()),
            "device": str(device or "auto"),
            "env": dict(env_overrides or {}),
        }
        self._send_request(payload, env_overrides)
        assert self._conn is not None

        timeout_s = _env_float("ASR_STAGE_WORKER_TIMEOUT_S", 0.0)
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None

        while True:
            if cancel_requested is not None and cancel_requested():
                self._kill_child()
                raise GpuWorkerError("cancelled", "ASR stage worker cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                self._kill_child()
                raise GpuWorkerTimeoutError(
                    f"ASR stage worker timeout after {timeout_s:.1f}s"
                )

            wait_s = 0.25
            if deadline is not None:
                wait_s = max(0.05, min(wait_s, deadline - time.monotonic()))
            if not self._conn.poll(wait_s):
                exitcode = self._process.exitcode if self._process is not None else None
                if exitcode is not None:
                    self._kill_child()
                    raise GpuWorkerError(
                        "crash",
                        f"ASR stage worker exited before result exitcode={exitcode}",
                    )
                continue

            try:
                message = self._conn.recv()
            except EOFError as exc:
                exitcode = self._process.exitcode if self._process is not None else None
                self._kill_child()
                raise GpuWorkerError(
                    "crash",
                    f"ASR stage worker pipe closed exitcode={exitcode}",
                ) from exc

            if not isinstance(message, dict):
                self._kill_child()
                raise GpuWorkerError("protocol_error", "worker message is not a dict")
            if str(message.get("job_id") or "") != request_id:
                continue

            op = message.get("op")
            if op == "stage":
                if on_stage is not None:
                    try:
                        on_stage(str(message.get("message") or ""))
                    except BaseException:
                        self._kill_child()
                        raise
                continue

            if op == "result":
                segments = message.get("segments")
                asr_log = message.get("asr_log")
                asr_details = message.get("asr_details")
                if not isinstance(segments, list):
                    self._kill_child()
                    raise GpuWorkerError("protocol_error", "segments must be a list")
                if not isinstance(asr_log, list):
                    self._kill_child()
                    raise GpuWorkerError("protocol_error", "asr_log must be a list")
                if not isinstance(asr_details, dict):
                    self._kill_child()
                    raise GpuWorkerError("protocol_error", "asr_details must be a dict")
                return segments, [str(item) for item in asr_log], dict(asr_details)

            if op == "error":
                kind = str(message.get("kind") or "crash")
                detail = str(message.get("detail") or "ASR stage worker error")
                self._kill_child()
                raise GpuWorkerError(kind, detail)

            self._kill_child()
            raise GpuWorkerError("protocol_error", f"unexpected worker op: {op}")

    def transcribe_and_align(
        self,
        audio_path: str,
        *,
        device: str = "auto",
        env_overrides: dict[str, str] | None = None,
        job_id: str = "",
        on_stage: Callable[[str], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> tuple[list[dict], list[str], dict]:
        retry_limit = max(0, _env_int("ASR_STAGE_WORKER_OOM_RETRY_LIMIT", 3))
        attempt = 0
        retry_records: list[dict[str, Any]] = []
        current_env = dict(env_overrides or {})
        while True:
            try:
                segments, asr_log, asr_details = self._transcribe_and_align_once(
                    audio_path,
                    device=device,
                    env_overrides=current_env,
                    job_id=job_id,
                    on_stage=on_stage,
                    cancel_requested=cancel_requested,
                )
                # A successful result is final -- never discard it for a VRAM
                # budget. Surface a non-fatal note if the reserved pool spiked,
                # but keep the segments. Hard OOM is caught below / mid-pipeline.
                warning = _budget_exceeded_warning(asr_details, current_env)
                if warning and on_stage is not None:
                    try:
                        on_stage(warning)
                    except BaseException:
                        pass
                if retry_records:
                    stage_worker = asr_details.setdefault("stage_worker", {})
                    if isinstance(stage_worker, dict):
                        stage_worker["oom_retries"] = list(retry_records)
                return segments, asr_log, asr_details
            except GpuWorkerError as exc:
                if str(getattr(exc, "kind", "")) != "oom":
                    raise
                lowered = _downshift_asr_batch_env(current_env)
                if lowered is None:
                    batch_size = _effective_asr_batch_size(current_env)
                    raise GpuWorkerError(
                        "oom",
                        _terminal_oom_detail(
                            env=current_env,
                            detail=str(getattr(exc, "detail", exc)),
                            batch_size=batch_size,
                            retry_records=retry_records,
                        ),
                    ) from exc
                if attempt >= retry_limit:
                    batch_size = _effective_asr_batch_size(current_env)
                    raise GpuWorkerError(
                        "oom",
                        _terminal_oom_detail(
                            env=current_env,
                            detail=str(getattr(exc, "detail", exc)),
                            batch_size=batch_size,
                            retry_records=retry_records,
                            retry_limit_exhausted=True,
                        ),
                    ) from exc
                next_env, previous_batch, next_batch = lowered
                attempt += 1
                retry_records.append(
                    {
                        "attempt": attempt,
                        "previous_batch_size": previous_batch,
                        "next_batch_size": next_batch,
                        "detail": str(getattr(exc, "detail", exc)),
                    }
                )
                current_env = next_env
                if on_stage is not None:
                    on_stage(
                        "GPU worker OOM; restarting worker and lowering "
                        f"ASR_BATCH_SIZE {previous_batch}->{next_batch}"
                    )
                continue


_GLOBAL_WORKER: _GpuWorkerClient | None = None
_GLOBAL_WORKER_LOCK = threading.RLock()


def _get_global_worker() -> _GpuWorkerClient:
    global _GLOBAL_WORKER
    if _GLOBAL_WORKER is None or not _GLOBAL_WORKER.is_alive():
        _GLOBAL_WORKER = _GpuWorkerClient()
    return _GLOBAL_WORKER


def transcribe_and_align(
    audio_path: str,
    *,
    device: str = "auto",
    env_overrides: dict[str, str] | None = None,
    job_id: str = "",
    on_stage: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[list[dict], list[str], dict]:
    global _GLOBAL_WORKER
    with _GLOBAL_WORKER_LOCK:
        worker = _get_global_worker()
        try:
            return worker.transcribe_and_align(
                audio_path,
                device=device,
                env_overrides=env_overrides,
                job_id=job_id,
                on_stage=on_stage,
                cancel_requested=cancel_requested,
            )
        except Exception:
            if not worker.is_alive():
                _GLOBAL_WORKER = None
            raise


def shutdown_global_worker() -> None:
    global _GLOBAL_WORKER
    with _GLOBAL_WORKER_LOCK:
        if _GLOBAL_WORKER is not None:
            _GLOBAL_WORKER.close()
        _GLOBAL_WORKER = None


atexit.register(shutdown_global_worker)
