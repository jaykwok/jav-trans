from __future__ import annotations

import atexit
import gc
import importlib
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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    if not raw:
        return default
    try:
        return float(raw)
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
    # These modules read ASR settings into module-level constants. Reload them
    # after applying per-request env so a long-lived worker can handle changed
    # model/batch/checkpoint settings without inheriting stale globals.
    module_order = (
        "asr.backends.qwen",
        "asr.local_backend",
        "asr.backends.registry",
        "asr.pipeline",
    )
    for module_name in module_order:
        module = sys.modules.get(module_name)
        if module is not None:
            importlib.reload(module)
    from asr import pipeline as asr_pipeline

    return importlib.reload(asr_pipeline)


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
            "mode": "subprocess",
            "pid": pid,
            "mock": True,
        },
    }
    return segments, log, details


def worker_main(parent_conn: Connection) -> None:
    sys.stdout = sys.stderr
    pid = os.getpid()
    if not _safe_send(parent_conn, {"op": "ready", "pid": pid}):
        return

    while True:
        try:
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
                device = _resolve_device(requested_device)
                try:
                    import torch as imported_torch

                    torch_module = imported_torch
                    try:
                        if device.startswith("cuda") and torch_module.cuda.is_available():
                            torch_module.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
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
                if mock:
                    _on_stage("ASR stage worker mock")
                    segments, asr_log, asr_details = _mock_result(
                        audio_path,
                        device,
                        pid=pid,
                    )
                else:
                    asr_pipeline = _load_asr_pipeline_for_request()
                    segments, asr_log, asr_details = asr_pipeline.transcribe_and_align(
                        audio_path,
                        device,
                        on_stage=_on_stage,
                        include_details=True,
                    )
                    asr_details = dict(asr_details or {})
                    asr_details["stage_worker"] = {
                        "mode": "subprocess",
                        "pid": pid,
                        "mock": False,
                    }
                    asr_details.setdefault("device", device)

                _clear_worker_cuda()
                if not _safe_send(
                    parent_conn,
                    {
                        "op": "result",
                        "job_id": job_id,
                        "segments": segments,
                        "asr_log": asr_log,
                        "asr_details": asr_details,
                        "device": str(asr_details.get("device") or device),
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


class _GpuWorkerClient:
    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._conn = None
        self._job_handle = None
        self.kill_grace_s = _env_float("ASR_STAGE_WORKER_KILL_GRACE_S", 5.0)

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

    def _start_worker(self) -> None:
        self._kill_child()
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=worker_main,
            args=(child_conn,),
            daemon=False,
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
        self._ensure_worker()
        assert self._conn is not None
        request_id = str(job_id or uuid.uuid4().hex[:8])
        payload = {
            "op": "transcribe_and_align",
            "job_id": request_id,
            "audio_path": str(Path(audio_path).resolve()),
            "device": str(device or "auto"),
            "env": dict(env_overrides or {}),
        }
        try:
            self._conn.send(payload)
        except (BrokenPipeError, EOFError, OSError) as exc:
            exitcode = self._process.exitcode if self._process is not None else None
            self._kill_child()
            raise GpuWorkerError(
                "crash",
                f"ASR stage worker send failed exitcode={exitcode}: {exc!r}",
            ) from exc

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
