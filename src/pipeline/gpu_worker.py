from __future__ import annotations

import atexit
import faulthandler
import gc
import hashlib
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable

from asr import vram_sizing as asr_vram_sizing
from pipeline.stage_log import ASR_STAGE_HEARTBEAT_PREFIX, ASR_WORKER_DIAGNOSTIC_PREFIX
from utils.ffmpeg_runtime import configure_ffmpeg_shared_runtime
from core import gpu_admission, resources
from core.diagnostics import (
    format_exception_chain,
    format_exit_code,
    is_fatal_cuda_error,
    redirect_process_stdio_to_file,
)
from core.typed_config import env_float, env_int
from utils.runtime_paths import runtime_path
from utils.subprocess_tools import close_kill_on_close_job


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


logger = logging.getLogger(__name__)
_EOF_EXIT_GRACE_S = 0.5


# Thin names kept because this module reads a dozen worker knobs and the call
# sites read better without the module prefix; the rules are the shared ones.
_env_float = env_float
_env_int = env_int


def _safe_send(conn: Connection, message: dict[str, Any]) -> bool:
    try:
        conn.send(message)
        return True
    except Exception:
        logger.exception("ASR worker could not send %s message", message.get("op"))
        return False


def _worker_log_path(pid: int) -> Path:
    return runtime_path(Path("logs") / f"asr-worker-{pid}.log")


def _worker_exit_diagnostics(process, *, wait_for_exit: bool = False) -> str:
    if process is None:
        return "pid=None exitcode=None"
    if wait_for_exit and process.exitcode is None:
        process.join(_EOF_EXIT_GRACE_S)
    exitcode = process.exitcode
    status = format_exit_code(exitcode)
    pid = process.pid
    log_path = _worker_log_path(pid)
    log_ref = f"logs/{log_path.name}"
    if pid is not None and not log_path.exists():
        log_ref += " (not created)"
    return f"pid={pid} exitcode={status} worker_log={log_ref}"


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
        or "shared vram spill" in detail
    )


def _is_ram_oom_error(exc: BaseException) -> bool:
    return "physical ram budget exceeded" in repr(exc).lower()


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




def _resolve_device(device: str) -> str:
    requested = str(device or "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


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
    # AlignmentHead.from_env() -- so a
    # persistent worker picks up each job's _temporary_env without re-importing.
    # The only env frozen at CUDA init (PYTORCH_CUDA_ALLOC_CONF,
    # CUDA_VISIBLE_DEVICES) is handled by the client's restart-on-change.
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
        device = _resolve_device(requested_device)
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


def _effective_asr_batch_size(env: dict[str, str]) -> int | None:
    raw = str(env.get("ASR_BATCH_SIZE") or os.getenv("ASR_BATCH_SIZE", "auto")).strip()
    if raw and raw.lower() != "auto":
        try:
            return max(1, int(float(raw)))
        except (TypeError, ValueError):
            return None
    return None


def _physical_vram_mb(torch_module: Any) -> float:
    try:
        if not torch_module.cuda.is_available():
            return 0.0
        device_index = torch_module.cuda.current_device()
        properties = torch_module.cuda.get_device_properties(device_index)
        return float(properties.total_memory) / (1024 * 1024)
    except Exception as exc:
        if is_fatal_cuda_error(exc):
            raise
        return 0.0


def _gpu_device_name(torch_module: Any) -> str:
    try:
        if not torch_module.cuda.is_available():
            return ""
        return str(torch_module.cuda.get_device_name(torch_module.cuda.current_device()))
    except Exception as exc:
        if is_fatal_cuda_error(exc):
            raise
        return ""


def _gpu_hardware_identity(
    torch_module: Any,
    *,
    device_name: str,
    total_mb: float,
) -> dict[str, Any]:
    """Return a privacy-preserving identity for the physical CUDA device.

    CUDA ordinals are deliberately excluded: ``cuda:0`` changes when device
    visibility or enumeration order changes.  A CUDA UUID identifies the card
    itself, while PCI coordinates are the best identity older torch builds can
    expose.  Only a truncated hash is persisted, never the raw UUID or PCI
    address.
    """

    properties = None
    try:
        if torch_module.cuda.is_available():
            device_index = torch_module.cuda.current_device()
            properties = torch_module.cuda.get_device_properties(device_index)
    except Exception as exc:
        if is_fatal_cuda_error(exc):
            raise
        properties = None

    major = getattr(properties, "major", None)
    minor = getattr(properties, "minor", None)
    compute_capability = (
        f"{int(major)}.{int(minor)}"
        if major is not None and minor is not None
        else ""
    )

    raw_uuid = str(getattr(properties, "uuid", "") or "").strip().lower()
    if raw_uuid and raw_uuid not in {"none", "unknown"}:
        fingerprint_source = "cuda_uuid"
        fingerprint_material = f"nvidia:uuid:{raw_uuid}"
    else:
        pci_parts = (
            getattr(properties, "pci_domain_id", None),
            getattr(properties, "pci_bus_id", None),
            getattr(properties, "pci_device_id", None),
        )
        if all(part is not None for part in pci_parts):
            fingerprint_source = "pci_location"
            fingerprint_material = "nvidia:pci:" + ":".join(
                str(int(part)) for part in pci_parts
            )
        else:
            fingerprint_source = "model_vram_compute"
            fingerprint_material = (
                "nvidia:fallback:"
                f"{str(device_name).strip().lower()}:"
                f"{round(float(total_mb))}:{compute_capability}"
            )

    digest = hashlib.sha256(fingerprint_material.encode("utf-8")).hexdigest()[:24]
    return {
        "vendor": "nvidia",
        "fingerprint": f"sha256:{digest}",
        "fingerprint_source": fingerprint_source,
        "name": str(device_name or "").strip(),
        "physical_vram_mb": round(float(total_mb)),
        "compute_capability": compute_capability,
    }


def _enforce_min_physical_vram(
    *,
    total_mb: float,
    env: dict[str, str],
) -> dict[str, Any]:
    from asr.backends.qwen import qwen_asr_min_physical_vram_mb, qwen_asr_repo_id

    backend = str(env.get("ASR_BACKEND") or os.getenv("ASR_BACKEND", "")).strip()
    repo_id = qwen_asr_repo_id(backend or None)
    minimum_mb = float(qwen_asr_min_physical_vram_mb(repo_id))
    if total_mb < minimum_mb:
        raise RuntimeError(
            "Physical dedicated VRAM below supported minimum: "
            f"repo={repo_id} detected_mb={total_mb:.0f} required_mb={minimum_mb:.0f}. "
            "Shared VRAM and an explicit worker budget do not count; CPU fallback is disabled."
        )
    return {"repo_id": repo_id, "minimum_physical_vram_mb": round(minimum_mb, 1)}


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
            pass

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
                pass
        return self.current_stage

    def finish(self) -> dict[str, float]:
        self._finish_current()
        return {
            stage: round(value, 1)
            for stage, value in self.peaks_mb.items()
        }


def _live_vram_snapshot_mb(torch_module: Any) -> tuple[float | None, float]:
    """Return ``(free_mb, own_reserved_mb)`` when the CUDA module exposes them."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return None, 0.0
    try:
        if not cuda.is_available():
            return None, 0.0
        mem_info = getattr(cuda, "mem_get_info", None)
        if not callable(mem_info):
            return None, 0.0
        device_index = cuda.current_device()
        free_bytes, _total_bytes = mem_info(device_index)
        reserved_fn = getattr(cuda, "memory_reserved", None)
        own_reserved_bytes = reserved_fn(device_index) if callable(reserved_fn) else 0
        scale = 1024 * 1024
        return float(free_bytes) / scale, float(own_reserved_bytes) / scale
    except Exception as exc:
        if is_fatal_cuda_error(exc):
            raise
        return None, 0.0


def _apply_worker_vram_allocator_cap(
    torch_module: Any,
    *,
    total_mb: float,
    budget_mb: float,
) -> float | None:
    """Keep the CUDA worker below the physical-VRAM soft budget.

    Windows WDDM can satisfy CUDA allocations from shared system memory after
    dedicated VRAM is exhausted. The stage snapshot budget catches that after a
    stage, but the allocator fraction makes the offending allocation fail at
    the boundary instead of running slowly in shared memory.
    """

    if total_mb <= 0.0 or budget_mb <= 0.0:
        return None
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return None
    try:
        if not cuda.is_available():
            return None
    except Exception as exc:
        if is_fatal_cuda_error(exc):
            raise
        return None
    setter = getattr(cuda, "set_per_process_memory_fraction", None)
    if not callable(setter):
        return None
    fraction = min(1.0, max(0.1, float(budget_mb) / float(total_mb)))
    try:
        device_count_fn = getattr(cuda, "device_count", None)
        if callable(device_count_fn):
            device_indexes = range(max(1, int(device_count_fn())))
        else:
            current_device_fn = getattr(cuda, "current_device", None)
            current = int(current_device_fn()) if callable(current_device_fn) else 0
            device_indexes = (current,)
        for device_index in device_indexes:
            setter(fraction, device_index)
    except Exception as exc:
        if is_fatal_cuda_error(exc):
            raise
        return None
    return fraction


def _adaptive_runtime_tuning(
    torch_module: Any,
    env: dict[str, str],
    *,
    enforce_minimum: bool = True,
) -> dict[str, Any]:
    """Resolve auto VRAM budget and ASR batch inside the CUDA owner process."""
    total_mb = _physical_vram_mb(torch_module)
    device_name = _gpu_device_name(torch_module)
    hardware = _gpu_hardware_identity(
        torch_module,
        device_name=device_name,
        total_mb=total_mb,
    )
    minimum_contract = (
        _enforce_min_physical_vram(total_mb=total_mb, env=env)
        if enforce_minimum
        else {}
    )
    ratio = min(1.0, max(0.1, _env_float("ASR_STAGE_WORKER_VRAM_RATIO", 0.95)))
    raw_budget = str(
        env.get("ASR_STAGE_WORKER_VRAM_BUDGET_MB")
        or os.getenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "auto")
    ).strip().lower()
    if raw_budget in {"", "auto", "physical"}:
        budget_mb = total_mb * ratio if total_mb > 0.0 else 0.0
        os.environ["ASR_STAGE_WORKER_VRAM_BUDGET_MB"] = (
            f"{budget_mb:.1f}" if budget_mb > 0.0 else "0"
        )
        budget_source = "physical_vram_ratio"
    else:
        try:
            budget_mb = max(0.0, float(raw_budget))
        except (TypeError, ValueError):
            budget_mb = 0.0
        budget_source = "explicit"
    allocator_fraction = _apply_worker_vram_allocator_cap(
        torch_module,
        total_mb=total_mb,
        budget_mb=budget_mb,
    )

    raw_batch = str(
        env.get("ASR_BATCH_SIZE") or os.getenv("ASR_BATCH_SIZE", "auto")
    ).strip().lower()
    free_mb, own_reserved_mb = _live_vram_snapshot_mb(torch_module)

    def _read_float_opt(name: str, default: float) -> float:
        raw = str(env.get(name) or os.getenv(name, "")).strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    raw_max_new = str(
        env.get("ASR_MAX_NEW_TOKENS") or os.getenv("ASR_MAX_NEW_TOKENS", "")
    ).strip()
    try:
        parsed_max_new = int(float(raw_max_new)) if raw_max_new else None
    except ValueError:
        parsed_max_new = None

    plan = asr_vram_sizing.select_asr_batch_plan(
        total_vram_mb=total_mb,
        budget_mb=budget_mb,
        free_vram_mb=free_mb,
        own_reserved_mb=own_reserved_mb,
        explicit_budget=(budget_source == "explicit"),
        dtype=str(env.get("ASR_DTYPE") or os.getenv("ASR_DTYPE", "auto")),
        chunk_max_s=_read_float_opt("ASR_CHUNK_MAX_S", 30.0),
        attention=str(env.get("ASR_ATTENTION") or os.getenv("ASR_ATTENTION", "auto")),
        max_new_tokens=parsed_max_new,
        tokens_per_second=_read_float_opt("ASR_DECODE_TOKENS_PER_SECOND", 10.0),
    )

    if raw_batch in {"", "auto"}:
        effective_batch = plan.batch_size
        batch_source = plan.source
    else:
        try:
            effective_batch = max(1, int(float(raw_batch)))
            batch_source = "explicit"
        except (TypeError, ValueError):
            effective_batch = plan.batch_size
            batch_source = plan.source
    os.environ["ASR_BATCH_SIZE"] = str(effective_batch)

    return {
        "device_name": device_name,
        "hardware": hardware,
        "physical_vram_mb": round(total_mb, 1),
        "minimum_physical_vram_mb": minimum_contract.get(
            "minimum_physical_vram_mb"
        ),
        "vram_ratio": round(ratio, 4),
        "vram_budget_mb": round(budget_mb, 1),
        "vram_budget_source": budget_source,
        "vram_allocator_fraction": (
            round(float(allocator_fraction), 4)
            if allocator_fraction is not None
            else None
        ),
        "asr_batch_size": effective_batch,
        "asr_batch_source": batch_source,
        "asr_vram_plan": {
            "batch_size": plan.batch_size,
            "estimated_allocated_mb": plan.estimated_allocated_mb,
            "estimated_reserved_mb": plan.estimated_reserved_mb,
            "weights_mb": plan.weights_mb,
            "per_sample_mb": plan.per_sample_mb,
            "usable_vram_mb": plan.usable_vram_mb,
            "free_vram_mb": plan.free_vram_mb,
            "own_reserved_mb": plan.own_reserved_mb,
        },
    }


def _downshift_asr_batch_env(
    env: dict[str, str],
    *,
    current_override: int | None = None,
) -> tuple[dict[str, str], int, int] | None:
    current = current_override or _effective_asr_batch_size(env)
    if current is None or current <= 1:
        return None
    lowered = max(1, current // 2)
    if lowered >= current:
        return None
    updated = dict(env)
    updated["ASR_BATCH_SIZE"] = str(lowered)
    return updated, current, lowered


def _oom_stage_from_message(message: str) -> str:
    text = str(message or "")
    if (
        "ASR 文本转写" in text
        or "asr_text_transcribe" in text
        or "GPU model manager asr" in text
        or "加载本地 ASR 模型" in text
    ):
        return "asr_text_transcribe"
    if "字幕时间轴" in text or "subtitle_timing" in text:
        return "subtitle_timing"
    # Chunking runs one encoder forward per 30s window and holds no batch knob;
    # naming it here would only produce a downshift that cannot be applied.
    if "切分" in text or "音频切块" in text:
        return "audio_chunking"
    return ""


def _oom_downshift(
    env: dict[str, str],
    exc: GpuWorkerError,
) -> tuple[dict[str, str], str, float, float] | None:
    stage = str(exc.stage or "")
    detail = str(exc.detail or "").lower()
    tuning = exc.runtime_tuning
    if stage == "audio_chunking" or "stage=split_done" in detail:
        # Chunking encodes fixed 30-second windows one at a time. There is no
        # batch to halve, and shortening the window would change the features
        # the blank runs are read from.
        return None

    lowered = _downshift_asr_batch_env(
        env,
        current_override=int(tuning.get("asr_batch_size") or 0) or None,
    )
    if lowered is None:
        return None
    updated, previous, next_value = lowered
    return updated, "ASR_BATCH_SIZE", float(previous), float(next_value)


def _terminal_oom_detail(
    *,
    env: dict[str, str],
    detail: str,
    batch_size: int | None,
    retry_records: list[dict[str, Any]],
    retry_limit_exhausted: bool = False,
    failed_setting: str = "ASR_BATCH_SIZE",
) -> str:
    budget_mb = _vram_budget_for_env(env)
    retry_summary = ""
    if retry_records:
        steps = [
            "{setting} {previous}->{next}".format(
                setting=record.get("setting") or "ASR_BATCH_SIZE",
                previous=record.get("previous_value", record.get("previous_batch_size")),
                next=record.get("next_value", record.get("next_batch_size")),
            )
            for record in retry_records
        ]
        retry_summary = f" 已尝试降 batch：{', '.join(steps)}。"
    budget_text = f" 当前显存预算为 {budget_mb:.0f}MB。" if budget_mb > 0 else ""
    if failed_setting == "ASR_CHUNK_ENCODER":
        headline = (
            "GPU 显存不足：OOM 出现在切分用的 encoder 前向；"
            "该阶段逐个编码固定 30 秒窗口，没有可降的 batch，任务已停止。"
        )
    elif batch_size is not None and batch_size <= 1:
        headline = "GPU 显存不足：ASR_BATCH_SIZE 已降到 1 后仍然 OOM，任务已停止。"
    elif retry_limit_exhausted:
        headline = "GPU 显存不足：OOM 自动降 batch 重试次数已用尽，任务已停止。"
    else:
        headline = "GPU 显存不足：无法继续降低 ASR_BATCH_SIZE，任务已停止。"

    # The 0.6B tier was dropped on 2026-07-31, so there is no smaller model to
    # suggest. Pointing at one that no longer ships would be worse than saying
    # plainly that this hardware cannot run the job.
    action = (
        "只有 1.7B 一档，没有更小的模型可切换；请关闭其他占用 GPU 的程序后重试。"
        "在当前硬件/可用显存下仍然不行时任务只能停止，"
        "本程序不会改用 CPU 或缩短时序窗口继续推理。"
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
        stage_worker = asr_details.get("stage_worker")
        if isinstance(stage_worker, dict):
            tuning = stage_worker.get("runtime_tuning")
            if isinstance(tuning, dict):
                try:
                    budget_mb = float(tuning.get("vram_budget_mb") or 0.0)
                except (TypeError, ValueError):
                    budget_mb = 0.0
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
    with ExitStack() as diagnostics:
        try:
            log_path = _worker_log_path(os.getpid())
            log_path.parent.mkdir(parents=True, exist_ok=True)
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


def _serve_worker(parent_conn: Connection) -> None:
    pid = os.getpid()
    ffmpeg_shared_directories = configure_ffmpeg_shared_runtime()

    # Model downloads (the ASR model, on first run) happen inside this
    # subprocess, so hf_progress's tqdm-hooked events need their own route to
    # the SSE stream: the main process already runs a TCP listener for this
    # (see web/app.py) and merely points its own core.events sink at it, so
    # pointing this process's sink at the same port makes downloads show up
    # in the job card exactly like any other stage event. Best-effort: a
    # stuck/slow port never blocks ASR, it only silences the progress bar.
    try:
        from core import events as _events

        events_port = os.environ.get("JAV_SUB_EVENTS_PORT", "2234")
        _events.configure_sink(f"tcp:127.0.0.1:{events_port}")
    except Exception:
        pass

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
                try:
                    _clear_worker_cuda()
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
                _clear_worker_cuda()
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
                    runtime_tuning = _adaptive_runtime_tuning(
                        torch_module,
                        env,
                        enforce_minimum=not mock,
                    )
                    stage_memory = _CudaStageMemoryTracker(torch_module)

                def _on_stage(message: str, *, tracker=stage_memory, request_job_id=job_id) -> None:
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
                    f"asr_source={runtime_tuning.get('asr_batch_source')} "
                    f"budget_mb={runtime_tuning.get('vram_budget_mb')}"
                )
                model_manager = GpuModelManager(
                    pid=pid,
                    on_stage=_on_stage,
                    runtime_tuning=runtime_tuning,
                )
                segments, asr_log, asr_details = model_manager.run_transcribe_and_align(
                    audio_path=audio_path,
                    requested_device=requested_device,
                    mock=mock,
                    defer_cleanup=lambda fn, _cleanup=cleanup, _job_id=job_id: _cleanup.callback(
                        _run_chunk_cleanup, fn, job_id=_job_id,
                    ),
                )
                stage_peaks = stage_memory.finish() if stage_memory is not None else {}
                stage_worker = asr_details.get("stage_worker")
                if isinstance(stage_worker, dict):
                    stage_worker["stage_peak_allocated_mb"] = stage_peaks

                _on_stage("ASR worker: final CUDA cleanup")
                _clear_worker_cuda()
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
                job_id, pid, kind, last_step,
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
                        f"pid={pid} worker_log=logs/{_worker_log_path(pid).name}"
                    ),
                    "stage": "result_send" if sending_result else active_stage,
                    "runtime_tuning": runtime_tuning,
                },
            )
            _clear_worker_cuda()
            raise SystemExit(0 if sent else 1)
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


# Env read by CUDA only at runtime/driver init, so changing it on a *running*
# persistent worker has no effect. The client restarts the worker when one of
# these differs from the values it was started under.
_CUDA_INIT_ENV_KEYS = ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")


_CLIENT_ID_SEQ = 0
_CLIENT_ID_LOCK = threading.RLock()
_ASR_KIND = "asr_worker"


def _next_client_id() -> int:
    global _CLIENT_ID_SEQ
    with _CLIENT_ID_LOCK:
        _CLIENT_ID_SEQ += 1
        return _CLIENT_ID_SEQ


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
        self._cleanup_error = ""
        # Identity of this client, and of the child it is currently responsible
        # for. Both travel with every cleanup-state update so a *different*
        # client can never release a child it does not own.
        self._client_id = _next_client_id()
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
                print(f"[WARN] ASR stage worker {step}() failed: {exc!r}")
            try:
                process.join(grace_s)
            except Exception:
                pass

        # A Job Object handle is a raw integer: assigning None to it leaks the
        # handle instead of closing it. Closing it is also the last escalation
        # available - the job is kill-on-close, so the whole tree goes with it,
        # including children terminate()/kill() never reach.
        if job_handle and close_kill_on_close_job(job_handle):
            job_handle = None
        elif job_handle:
            detail = detail or "job object handle could not be closed"
            print("[WARN] ASR stage worker job object handle could not be closed")

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
            print(f"[WARN] ASR stage worker did not exit; pid={pid} {liveness}")
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
            process_state=liveness, job_handle=job_handle, detail=detail
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
            self._cleanup_error = f"liveness check failed: {exc!r}"
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
        conn = self._conn
        process = self._process
        try:
            if conn is not None and process is not None and process.is_alive():
                conn.send({"op": "shutdown"})
                process.join(5)
        except Exception:
            pass
        finally:
            # Always through _kill_child: on the clean-exit path it only clears
            # the references and closes the Job Object handle, and it is the one
            # place that reports a child it could not stop.
            stopped = self._kill_child()
        return stopped

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

    def _start_worker(self, cancel_requested: Callable[[], bool] | None = None) -> None:
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
                job = None
                try:
                    from asr.local_backend import (
                        _assign_process_to_job_object,
                        _create_kill_on_close_job_object,
                    )

                    job = _create_kill_on_close_job_object()
                    _assign_process_to_job_object(job, process)
                    self._job_handle = job
                    resources.attach_handle(
                        self._resource_id(), self._child_generation, job
                    )
                except Exception as exc:
                    # A job that was created but not assigned still holds a
                    # handle; dropping the reference leaks it for the life of
                    # the process.
                    logger.warning(
                        "Could not assign ASR worker pid=%s to kill-on-close Job Object: %s",
                        getattr(process, "pid", None),
                        exc,
                    )
                    close_kill_on_close_job(job)
                    self._job_handle = None
            child_conn.close()
            self._process = process
            self._conn = parent_conn

        ready_timeout_s = _env_float("ASR_STAGE_WORKER_READY_TIMEOUT_S", 60.0)
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

    def _ensure_worker(self, cancel_requested: Callable[[], bool] | None = None) -> None:
        if self.is_alive():
            return
        self._start_worker(cancel_requested)

    def _send_request(
        self,
        payload: dict[str, Any],
        env_overrides: dict[str, str] | None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        """Send a request, transparently restarting once if the pipe is dead.

        A persistent worker can self-exit between jobs (idle timeout / max-jobs
        cadence) or be killed by an OOM retry; rather than surfacing that as a
        crash, restart and resend once before giving up. Also records the
        CUDA-init-time env the (re)started worker is running under.
        """
        for attempt in range(2):
            self._ensure_worker(cancel_requested)
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
        assert conn is not None

        timeout_s = _env_float("ASR_STAGE_WORKER_TIMEOUT_S", 0.0)
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        heartbeat_s = max(0.0, _env_float("ASR_STAGE_WORKER_HEARTBEAT_S", 10.0))
        request_started = time.monotonic()
        last_stage_at = request_started
        last_heartbeat_at = request_started
        last_stage_message = "GPU worker request accepted"

        # One cleanup site for the whole wait. Every branch below used to have
        # to remember to kill the child before raising, and the heartbeat
        # callback did not: a cancel that landed between the top-of-loop check
        # and the heartbeat raised out of `on_stage` and left the GPU worker
        # transcribing with nobody listening.
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
                if str(message.get("job_id") or "") != request_id:
                    continue

                op = message.get("op")
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
        retry_limit = max(0, _env_int("ASR_STAGE_WORKER_OOM_RETRY_LIMIT", 6))
        attempts_by_setting: dict[str, int] = {}
        retry_records: list[dict[str, Any]] = []
        current_env = dict(env_overrides or {})
        retried_disconnect = False
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
                lowered = _oom_downshift(current_env, exc)
                if lowered is None:
                    batch_size = int(exc.runtime_tuning.get("asr_batch_size") or 0) or (
                        _effective_asr_batch_size(current_env)
                    )
                    failed_setting = (
                        "ASR_CHUNK_ENCODER"
                        if exc.stage == "audio_chunking"
                        or "stage=split_done" in str(exc.detail or "").lower()
                        else "ASR_BATCH_SIZE"
                    )
                    raise GpuWorkerError(
                        "oom",
                        _terminal_oom_detail(
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
                    batch_size = int(exc.runtime_tuning.get("asr_batch_size") or 0) or (
                        _effective_asr_batch_size(current_env)
                    )
                    raise GpuWorkerError(
                        "oom",
                        _terminal_oom_detail(
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


_GLOBAL_WORKER: _GpuWorkerClient | None = None
_GLOBAL_WORKER_LOCK = threading.RLock()


def _get_global_worker() -> _GpuWorkerClient:
    global _GLOBAL_WORKER
    # Only the child decides this now. A handle that would not close has already
    # been handed to the pending table, which outlives any client - keeping the
    # client for its sake would just give a retry a second, moving target.
    if _GLOBAL_WORKER is None or not (
        _GLOBAL_WORKER.is_alive() or _GLOBAL_WORKER.has_unreleased_child()
    ):
        # A client whose child survived the kill keeps the client: dropping it
        # would forget the only reference to a live GPU process and start a
        # second one next to it. The next _start_worker retries the kill and
        # refuses to spawn while it fails.
        _GLOBAL_WORKER = _GpuWorkerClient()
    return _GLOBAL_WORKER


def transcribe_and_align(
    audio_path: str,
    *,
    device: str = "auto",
    env_overrides: dict[str, str] | None = None,
    job_id: str = "",
    run_id: str = "",
    on_stage: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[list[dict], list[str], dict]:
    global _GLOBAL_WORKER
    # The permit, not `_GLOBAL_WORKER_LOCK`, is what makes one transcription at
    # a time true. Holding this lock for the whole request made it the thing a
    # local-model start blocked on inside `shutdown_global_worker()` - an
    # uncancellable wait for another job's entire ASR stage, on a lock whose
    # actual job is a two-line state change. The registry lock is short again;
    # exclusion and the cancellable wait belong to admission.
    try:
        permit = gpu_admission.acquire(
            gpu_admission.ASR,
            cancel=cancel_requested,
            description="asr-gpu-worker",
        )
    except gpu_admission.GpuAdmissionCancelled as exc:
        raise GpuWorkerError("cancelled", "ASR stage worker cancelled") from exc
    try:
        with _GLOBAL_WORKER_LOCK:
            worker = _get_global_worker()
        try:
            return worker.transcribe_and_align(
                audio_path,
                device=device,
                env_overrides=env_overrides,
                job_id=job_id,
                run_id=run_id,
                on_stage=on_stage,
                cancel_requested=cancel_requested,
            )
        except Exception:
            with _GLOBAL_WORKER_LOCK:
                # Only if it is still the shared one: the lock is no longer held
                # across the request, so clearing "whatever is filed there now"
                # could drop a replacement somebody else is already using.
                if _GLOBAL_WORKER is worker and not (
                    worker.is_alive() or worker.has_unreleased_child()
                ):
                    _GLOBAL_WORKER = None
            raise
    finally:
        gpu_admission.release(permit)


def shutdown_global_worker() -> bool:
    """Stop the shared worker. Returns whether it is confirmed stopped.

    Dropping the reference regardless of the outcome was the whole bug: the
    undead child stayed on the card, the next client had no child of its own,
    and its empty cleanup then cleared the blocked state - so the queue resumed
    and a second GPU process started beside the first. A failed close keeps the
    client, keeps the blocked state, and says so.
    """
    global _GLOBAL_WORKER
    with _GLOBAL_WORKER_LOCK:
        worker = _GLOBAL_WORKER
        if worker is None:
            return True
        stopped = bool(worker.close())
        if stopped:
            _GLOBAL_WORKER = None
        return stopped


atexit.register(shutdown_global_worker)
