"""Child-process runtime for the persistent ASR GPU worker."""

from __future__ import annotations

import faulthandler
import gc
import logging
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from core.diagnostics import (
    format_exception_chain,
    is_fatal_cuda_error,
    redirect_process_stdio_to_file,
)
from core.typed_config import env_float
from pipeline import memory_safety
from pipeline.gpu_oom_policy import (
    _adaptive_runtime_tuning,
    _is_oom_error,
    _is_ram_oom_error,
    _oom_stage_from_message,
)
from utils.ffmpeg_runtime import configure_ffmpeg_shared_runtime
from utils.runtime_paths import runtime_path

logger = logging.getLogger(__name__)
_env_float = env_float

# Keys that `_adaptive_runtime_tuning` mutates inside `os.environ` even when
# the caller's `overrides` dict omitted them. Saving/restoring them explicitly
# prevents one job's resolved batch or budget from leaking into the next job.
_ALWAYS_RESTORED_ENV_KEYS = (
    "ASR_BATCH_SIZE",
    "ASR_FEATURE_BATCH_SIZE",
    "ASR_ALIGN_BATCH_SIZE",
    "ASR_STAGE_WORKER_VRAM_BUDGET_MB",
)

_WORKER_LOG_MAX_FILES = 20
_WORKER_LOG_MAX_AGE_S = 7 * 24 * 3600.0


def _facade_attr(name: str, default: Any) -> Any:
    facade = sys.modules.get("pipeline.gpu_worker")
    if facade is not None and hasattr(facade, name):
        return getattr(facade, name)
    return default


def _safe_send(conn: Connection, message: dict[str, Any]) -> bool:
    try:
        conn.send(message)
        return True
    except Exception:
        logger.exception("ASR worker could not send %s message", message.get("op"))
        return False


def _worker_log_path(pid: int) -> Path:
    return runtime_path(Path("logs") / f"asr-worker-{pid}.log")


def _prune_old_worker_logs(
    *,
    log_dir: Path | None = None,
    max_files: int = _WORKER_LOG_MAX_FILES,
    max_age_s: float = _WORKER_LOG_MAX_AGE_S,
    exclude_pid: int | None = None,
) -> int:
    """Prune stale ``logs/asr-worker-*.log`` files by age and total count."""
    target_dir = log_dir if log_dir is not None else runtime_path(Path("logs"))
    if not target_dir.is_dir():
        return 0
    keep_name = f"asr-worker-{exclude_pid}.log" if exclude_pid is not None else ""
    now = time.time()
    entries: list[tuple[float, Path]] = []
    try:
        for path in target_dir.glob("asr-worker-*.log"):
            if not path.is_file() or (keep_name and path.name == keep_name):
                continue
            try:
                entries.append((path.stat().st_mtime, path))
            except OSError:
                continue
    except OSError:
        return 0

    # Newest first so we keep the most recent `max_files` logs within `max_age_s`.
    entries.sort(key=lambda item: item[0], reverse=True)
    removed = 0
    keep_limit = max(0, int(max_files))
    age_limit = max(0.0, float(max_age_s))
    for index, (mtime, path) in enumerate(entries):
        too_many = index >= keep_limit
        too_old = age_limit > 0.0 and (now - mtime) >= age_limit
        if not (too_many or too_old):
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            # Active worker logs stay locked on Windows; skip silently.
            continue
    return removed


@contextmanager
def _temporary_env(overrides: dict[str, str]):
    keys = set(overrides) | set(_ALWAYS_RESTORED_ENV_KEYS)
    previous = {key: os.environ.get(key) for key in keys}
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


def _resolve_device(device: str, *, mock: bool = False) -> str:
    requested = str(device or "auto").strip().lower()
    if requested == "cpu":
        if mock:
            return "cpu"
        raise RuntimeError(
            "正式 ASR 流程已禁用 CPU 回退，无法使用 device='cpu'。请使用可用的 NVIDIA CUDA 设备。"
        )
    if requested and requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if mock:
        return "mock"
    raise RuntimeError(
        "CUDA 不可用，正式 ASR 流程已禁用 CPU 回退。请检查 NVIDIA 驱动与 CUDA 环境。"
    )


def _clear_worker_cuda() -> None:
    outer_exc_active = sys.exc_info()[0] is not None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        if not outer_exc_active and is_fatal_cuda_error(exc):
            raise
        logger.warning("Worker CUDA cleanup failed: %s", exc)


def _load_asr_pipeline_for_request():
    # No per-request reload. The settings that vary per job (ASR_BACKEND,
    # ASR_BATCH_SIZE, dtype/attention, chunking and alignment-head paths) are
    # all read from env at CALL time -- current_asr_backend(),
    # _resolve_asr_batch_size(), _detect_dtype(), _chunking_config(),
    # AlignmentHead.from_env() -- so a persistent worker picks up each job's
    # _temporary_env without re-importing. The only env frozen at CUDA init
    # (PYTORCH_CUDA_ALLOC_CONF, CUDA_VISIBLE_DEVICES) is handled by the
    # client's restart-on-change.
    from asr import pipeline as asr_pipeline

    return asr_pipeline


def _ensure_torchcodec_runtime(shared_directories: tuple[str, ...]) -> None:
    try:
        import torchcodec  # noqa: F401
    except Exception as exc:
        directories = ", ".join(shared_directories) or "未找到 FFmpeg Shared 目录"
        raise RuntimeError(
            "TorchCodec 无法加载 FFmpeg 共享运行库，ASR 任务已在边界分析前停止。"
            "Windows 源码环境请卸载 Gyan.FFmpeg 静态版并安装 "
            f"Gyan.FFmpeg.Shared。已注册目录：{directories}。原始错误：{exc}"
        ) from exc


class _CudaStageMemoryTracker:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.current_stage = ""
        self.peaks_mb: dict[str, float] = {}

    def _cuda_ready(self) -> bool:
        try:
            return bool(self.torch is not None and self.torch.cuda.is_available())
        except Exception:
            return False

    def _finish_current(self) -> None:
        if not self.current_stage or not self._cuda_ready():
            return
        outer_exc_active = sys.exc_info()[0] is not None
        try:
            device_index = self.torch.cuda.current_device()
            peak_bytes = max(
                self.torch.cuda.max_memory_allocated(device_index),
                self.torch.cuda.memory_allocated(device_index),
            )
            self.peaks_mb[self.current_stage] = max(
                self.peaks_mb.get(self.current_stage, 0.0),
                float(peak_bytes) / (1024 * 1024),
            )
        except Exception as exc:
            if not outer_exc_active and is_fatal_cuda_error(exc):
                raise

    def observe(self, message: str) -> str:
        stage = _oom_stage_from_message(message)
        if not stage or stage == self.current_stage:
            return self.current_stage
        self._finish_current()
        self.current_stage = stage
        if self._cuda_ready():
            outer_exc_active = sys.exc_info()[0] is not None
            try:
                self.torch.cuda.reset_peak_memory_stats()
            except Exception as exc:
                if not outer_exc_active and is_fatal_cuda_error(exc):
                    raise
        return self.current_stage

    def finish(self) -> dict[str, float]:
        self._finish_current()
        return {stage: round(value, 1) for stage, value in self.peaks_mb.items()}


def _mock_result(
    audio_path: str,
    device: str,
    *,
    pid: int,
) -> tuple[list[dict], list[str], dict]:
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


class GpuModelManager:
    """Owns GPU lifecycle boundaries inside the unified worker process."""

    def __init__(
        self,
        *,
        pid: int,
        on_stage: Callable[[str], None] | None = None,
        runtime_tuning: dict[str, Any] | None = None,
    ) -> None:
        self.pid = int(pid)
        self.on_stage = on_stage
        self.runtime_tuning = dict(runtime_tuning or {})
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
        outer_exc_active = sys.exc_info()[0] is not None
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
            if not outer_exc_active and is_fatal_cuda_error(exc):
                raise
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
        defer_cleanup: Callable[[Callable[[], None]], object],
    ) -> tuple[list[dict], list[str], dict]:
        self.reset_cuda_state(reason="before_request")
        memory_safety.reset_shared_vram_baseline()
        device = _resolve_device(requested_device, mock=mock)
        try:
            import torch

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception as exc:
            if is_fatal_cuda_error(exc):
                raise
            logger.warning("Could not reset CUDA peak memory stats: %s", exc)

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
            if self.on_stage is not None:
                self.on_stage("ASR worker: running ASR pipeline")
            segments, asr_log, asr_details = asr_pipeline.transcribe_and_align(
                audio_path,
                device,
                on_stage=self.on_stage,
                include_details=True,
                model_manager=self,
                defer_cleanup=defer_cleanup,
            )
            asr_details = dict(asr_details or {})

        if self.on_stage is not None:
            self.on_stage("ASR worker: post-ASR CUDA reset")
        self.reset_cuda_state(reason="after_request")
        asr_details.setdefault("device", device)
        asr_details["stage_worker"] = {
            "mode": "gpu_worker",
            "process_model": "persistent_subprocess",
            "pid": self.pid,
            "mock": bool(mock),
            "gpu_owner": True,
            "runtime_tuning": dict(self.runtime_tuning),
            "model_manager": {
                "policy": "sequential_exclusive",
                "events": list(self.events),
            },
        }
        return segments, asr_log, asr_details


def _run_chunk_cleanup(cleanup: Callable[[], None], *, job_id: str) -> None:
    logger.info("ASR worker job_id=%s temporary WAV cleanup started", job_id)
    try:
        cleanup()
    except SystemExit as exc:
        logger.warning(
            "ASR worker job_id=%s temporary WAV cleanup interrupted by SystemExit(%r) "
            "(possibly an external sandbox or safe-delete hook); skipping remaining cleanup",
            job_id,
            exc.code,
        )
        return
    except Exception:
        logger.exception("ASR worker job_id=%s temporary WAV cleanup failed", job_id)
        return
    logger.info("ASR worker job_id=%s temporary WAV cleanup finished", job_id)


def worker_main(parent_conn: Connection) -> None:
    sys.stdout = sys.stderr
    with ExitStack() as diagnostics:
        try:
            pid = os.getpid()
            log_path = _facade_attr("_worker_log_path", _worker_log_path)(pid)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _prune_old_worker_logs(log_dir=log_path.parent, exclude_pid=pid)
            handler = logging.FileHandler(log_path, encoding="utf-8")
        except OSError:
            logger.exception("ASR worker could not open its diagnostic log")
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            root = logging.getLogger()
            root.setLevel(logging.INFO)
            root.addHandler(handler)
            diagnostics.callback(handler.close)
            diagnostics.callback(root.removeHandler, handler)
            logger.setLevel(logging.INFO)
            if mp.current_process().name != "MainProcess":
                redirect_process_stdio_to_file(handler.stream)
            try:
                faulthandler.enable(file=handler.stream, all_threads=True)
            except (OSError, RuntimeError):
                logger.exception("ASR worker could not enable native fault logging")
            else:
                diagnostics.callback(faulthandler.disable)
            sitecustomize = sys.modules.get("sitecustomize")
            if sitecustomize is not None:
                logger.info(
                    "ASR worker loaded sitecustomize=%s",
                    getattr(sitecustomize, "__file__", repr(sitecustomize)),
                )
        try:
            _serve_worker(parent_conn)
        except SystemExit as exc:
            if exc.code not in (0, None):
                logger.exception("ASR worker terminated with SystemExit(%r)", exc.code)
            raise
        except BaseException:
            logger.exception("ASR worker terminated unexpectedly")
            raise


def _serve_worker(parent_conn: Connection) -> None:
    pid = os.getpid()
    ffmpeg_runtime_fn = _facade_attr(
        "configure_ffmpeg_shared_runtime", configure_ffmpeg_shared_runtime
    )
    ffmpeg_shared_directories = ffmpeg_runtime_fn()

    # Model downloads (the ASR model, on first run) happen inside this
    # subprocess, so hf_progress's tqdm-hooked events need their own route to
    # the SSE stream: the main process already runs a TCP listener for this
    # (see web/app.py) and merely points its own core.events sink at it.
    try:
        from core import events as _events

        events_port = os.environ.get("JAV_SUB_EVENTS_PORT", "2234")
        _events.configure_sink(f"tcp:127.0.0.1:{events_port}")
    except Exception:
        pass

    if not _safe_send(parent_conn, {"op": "ready", "pid": pid}):
        return

    configured_idle_s: float | None = None

    while True:
        idle_timeout_s = (
            configured_idle_s
            if configured_idle_s is not None
            else _env_float("ASR_STAGE_WORKER_MAX_IDLE_S", 300.0)
        )
        try:
            if idle_timeout_s > 0.0 and not parent_conn.poll(idle_timeout_s):
                # Re-check non-blockingly so a request arriving right at the
                # idle deadline is served instead of triggering a restart.
                if not parent_conn.poll(0):
                    _safe_send(parent_conn, {"op": "idle_exit", "pid": pid})
                    try:
                        _facade_attr("_clear_worker_cuda", _clear_worker_cuda)()
                    except Exception as exc:
                        logger.warning("Idle shutdown CUDA cleanup failed: %s", exc)
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
            try:
                _facade_attr("_clear_worker_cuda", _clear_worker_cuda)()
            except Exception as exc:
                logger.warning("Shutdown CUDA cleanup failed: %s", exc)
            return

        job_id = str(msg.get("job_id") or "")
        run_id = str(msg.get("run_id") or "")
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

        raw_env = msg.get("env")
        env = {}
        if isinstance(raw_env, dict):
            env = {
                str(key): str(value)
                for key, value in raw_env.items()
                if str(key).strip() and value is not None
            }
        requested_device = str(msg.get("device") or "auto")
        logger.info("ASR worker job_id=%s started", job_id)
        torch_module = None
        runtime_tuning: dict[str, Any] = {}
        active_stage = ""
        last_step = "initialization"
        sending_result = False
        stage_memory: _CudaStageMemoryTracker | None = None
        cleanup = ExitStack()
        clear_cuda_fn = _facade_attr("_clear_worker_cuda", _clear_worker_cuda)
        adaptive_tuning_fn = _facade_attr("_adaptive_runtime_tuning", _adaptive_runtime_tuning)
        manager_cls = _facade_attr("GpuModelManager", GpuModelManager)
        log_path_fn = _facade_attr("_worker_log_path", _worker_log_path)

        try:
            try:
                from core import events as _events
                from utils import hf_progress as _hf_progress

                # Stamped from the request, not looked up: this process outlives a
                # single run, so events must carry the run that asked for them.
                _events.set_current_run(job_id, run_id)
                _hf_progress.set_current_job_id(job_id, run_id)
            except Exception:
                pass
            with _temporary_env(env):
                if "ASR_STAGE_WORKER_MAX_IDLE_S" in env:
                    configured_idle_s = _env_float("ASR_STAGE_WORKER_MAX_IDLE_S", 300.0)
                else:
                    configured_idle_s = None

                mock = os.getenv("ASR_STAGE_WORKER_MOCK", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if not mock:
                    _ensure_torchcodec_runtime(ffmpeg_shared_directories)
                try:
                    import torch as imported_torch
                except ImportError:
                    imported_torch = None
                if imported_torch is not None:
                    torch_module = imported_torch
                    runtime_tuning = adaptive_tuning_fn(
                        torch_module,
                        env,
                        enforce_minimum=not mock,
                    )
                    stage_memory = _CudaStageMemoryTracker(torch_module)

                def _on_stage(
                    message: str,
                    *,
                    tracker=stage_memory,
                    request_job_id=job_id,
                ) -> None:
                    nonlocal active_stage, last_step
                    stage_hint = (
                        tracker.observe(message)
                        if tracker is not None
                        else _oom_stage_from_message(message)
                    )
                    if stage_hint:
                        active_stage = stage_hint
                    last_step = message
                    logger.info("ASR worker job_id=%s stage=%s", request_job_id, message)
                    parent_conn.send(
                        {
                            "op": "stage",
                            "job_id": request_job_id,
                            "message": message,
                        },
                    )

                _on_stage(
                    "GPU batch tuning "
                    f"asr={runtime_tuning.get('asr_batch_size')} "
                    f"feature={runtime_tuning.get('asr_feature_batch_size')} "
                    f"align={runtime_tuning.get('asr_align_batch_size')} "
                    f"asr_source={runtime_tuning.get('asr_batch_source')} "
                    f"budget_mb={runtime_tuning.get('vram_budget_mb')}"
                )
                model_manager = manager_cls(
                    pid=pid,
                    on_stage=_on_stage,
                    runtime_tuning=runtime_tuning,
                )
                segments, asr_log, asr_details = model_manager.run_transcribe_and_align(
                    audio_path=audio_path,
                    requested_device=requested_device,
                    mock=mock,
                    defer_cleanup=lambda fn, _cleanup=cleanup, _job_id=job_id: _cleanup.callback(
                        _run_chunk_cleanup,
                        fn,
                        job_id=_job_id,
                    ),
                )
                stage_peaks = stage_memory.finish() if stage_memory is not None else {}
                stage_worker = asr_details.get("stage_worker")
                if isinstance(stage_worker, dict):
                    stage_worker["stage_peak_allocated_mb"] = stage_peaks

                # `GpuModelManager.run_transcribe_and_align` already performs a
                # post-request `reset_cuda_state(reason="after_request")`
                # (`gc.collect()` + `torch.cuda.empty_cache()`). Skip the
                # redundant back-to-back cache flush when it already ran, while
                # keeping the final flush for stubbed managers in tests.
                already_reset = any(
                    isinstance(ev, dict)
                    and ev.get("stage") == "cuda"
                    and ev.get("reason") == "after_request"
                    for ev in getattr(model_manager, "events", ())
                )
                if not already_reset:
                    _on_stage("ASR worker: final CUDA cleanup")
                    clear_cuda_fn()
                sending_result = True
                _on_stage("ASR worker: sending result")
                parent_conn.send(
                    {
                        "op": "result",
                        "job_id": job_id,
                        "segments": segments,
                        "asr_log": asr_log,
                        "asr_details": asr_details,
                        "device": str(asr_details.get("device") or "auto"),
                    },
                )
        except (Exception, SystemExit) as exc:
            kind = (
                "result_send_failed"
                if sending_result
                else "ram_oom"
                if _is_ram_oom_error(exc)
                else "oom"
                if _is_oom_error(exc, torch_module)
                else "crash"
            )
            logger.exception(
                "ASR worker job_id=%s pid=%s kind=%s last_step=%r",
                job_id,
                pid,
                kind,
                last_step,
            )
            if not sending_result and stage_memory is not None:
                stage_memory.finish()
            error_summary = (
                f"SystemExit({exc.code!r})"
                if isinstance(exc, SystemExit)
                else format_exception_chain(exc)
            )
            sent = _safe_send(
                parent_conn,
                {
                    "op": "error",
                    "job_id": job_id,
                    "kind": kind,
                    "detail": (
                        f"{last_step}: {error_summary}; "
                        f"pid={pid} worker_log=logs/{log_path_fn(pid).name}"
                    ),
                    "stage": "result_send" if sending_result else active_stage,
                    "runtime_tuning": runtime_tuning,
                },
            )
            clear_cuda_fn()
            raise SystemExit(0 if sent else 1) from None
        finally:
            # Publish the completed ASR result before best-effort filesystem cleanup.
            cleanup.close()
            try:
                from core import events as _events
                from utils import hf_progress as _hf_progress

                _events.set_current_run("", "")
                _hf_progress.set_current_job_id("", "")
            except Exception:
                pass
