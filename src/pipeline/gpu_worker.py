"""Facade and shared singleton lifecycle for the persistent ASR GPU worker.

Implementation is split by responsibility across:
- ``pipeline.gpu_oom_policy``: VRAM sizing, allocator cap, and per-stage OOM downshift.
- ``pipeline.gpu_worker_runtime``: Child-process loop, CUDA lifecycle, and worker logs.
- ``pipeline.gpu_worker_client``: Parent-process IPC client, Job Object binding, and cleanup status.
"""

from __future__ import annotations

import atexit
import os
import threading
from collections.abc import Callable

from core import gpu_admission
from pipeline.gpu_oom_policy import (
    _adaptive_runtime_tuning,
    _apply_worker_vram_allocator_cap,
    _budget_exceeded_warning,
    _cuda_peak_mb_from_details,
    _downshift_asr_batch_env,
    _downshift_stage_batch_env,
    _effective_asr_batch_size,
    _effective_stage_batch_size,
    _enforce_min_physical_vram,
    _gpu_device_name,
    _gpu_hardware_identity,
    _is_oom_error,
    _is_ram_oom_error,
    _live_vram_snapshot_mb,
    _oom_downshift,
    _oom_setting_for_error,
    _oom_stage_from_message,
    _physical_vram_mb,
    _terminal_oom_detail,
    _vram_budget_for_env,
)
from pipeline.gpu_worker_client import (
    _ASR_KIND,
    _CUDA_INIT_ENV_KEYS,
    _EOF_EXIT_GRACE_S,
    GpuWorkerError,
    GpuWorkerTimeoutError,
    _GpuWorkerClient,
    _GpuWorkerDisconnectedError,
    _GpuWorkerIdleExitError,
    _worker_exit_diagnostics,
    asr_gpu_blocked,
    asr_gpu_cleanup_status,
    retry_asr_gpu_cleanup,
    unclosed_job_handles,
)
from pipeline.gpu_worker_runtime import (
    GpuModelManager,
    _clear_worker_cuda,
    _CudaStageMemoryTracker,
    _ensure_torchcodec_runtime,
    _load_asr_pipeline_for_request,
    _mock_result,
    _prune_old_worker_logs,
    _resolve_device,
    _run_chunk_cleanup,
    _safe_send,
    _serve_worker,
    _temporary_env,
    _worker_log_path,
    worker_main,
)
from utils.ffmpeg_runtime import configure_ffmpeg_shared_runtime
from utils.subprocess_tools import (
    bind_process_to_kill_on_close_job,
    close_kill_on_close_job,
    create_kill_on_close_job,
)

_GLOBAL_WORKER: _GpuWorkerClient | None = None
_GLOBAL_WORKER_LOCK = threading.RLock()


def _get_global_worker() -> _GpuWorkerClient:
    global _GLOBAL_WORKER
    if _GLOBAL_WORKER is None:
        _GLOBAL_WORKER = _GpuWorkerClient()
        return _GLOBAL_WORKER
    # A client whose child survived a kill keeps the singleton slot: dropping it
    # would forget the only reference to a live GPU process and start a second
    # one next to it. The next _start_worker retries the kill and refuses to
    # spawn while it fails.
    if _GLOBAL_WORKER.is_alive() or _GLOBAL_WORKER.has_unreleased_child():
        return _GLOBAL_WORKER
    # The previous worker exited (e.g. idle timeout or crash): close it before
    # replacing the singleton so its Job Object handle, pipe, and `core.resources`
    # record are released cleanly instead of leaking.
    old = _GLOBAL_WORKER
    old.close()
    if old.has_unreleased_child():
        return old
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
    # uncancellable wait for another job's entire ASR stage.
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
                    worker.close()
                    if not worker.has_unreleased_child():
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


__all__ = [
    "GpuModelManager",
    "GpuWorkerError",
    "GpuWorkerTimeoutError",
    "_ASR_KIND",
    "_CUDA_INIT_ENV_KEYS",
    "_EOF_EXIT_GRACE_S",
    "_GLOBAL_WORKER",
    "_GLOBAL_WORKER_LOCK",
    "_CudaStageMemoryTracker",
    "_GpuWorkerClient",
    "_GpuWorkerDisconnectedError",
    "_GpuWorkerIdleExitError",
    "_adaptive_runtime_tuning",
    "_apply_worker_vram_allocator_cap",
    "_budget_exceeded_warning",
    "_clear_worker_cuda",
    "_cuda_peak_mb_from_details",
    "_downshift_asr_batch_env",
    "_downshift_stage_batch_env",
    "_effective_asr_batch_size",
    "_effective_stage_batch_size",
    "_enforce_min_physical_vram",
    "_ensure_torchcodec_runtime",
    "_get_global_worker",
    "_gpu_device_name",
    "_gpu_hardware_identity",
    "_is_oom_error",
    "_is_ram_oom_error",
    "_live_vram_snapshot_mb",
    "_load_asr_pipeline_for_request",
    "_mock_result",
    "_oom_downshift",
    "_oom_setting_for_error",
    "_oom_stage_from_message",
    "_physical_vram_mb",
    "_prune_old_worker_logs",
    "_resolve_device",
    "_run_chunk_cleanup",
    "_safe_send",
    "_serve_worker",
    "_temporary_env",
    "_terminal_oom_detail",
    "_vram_budget_for_env",
    "_worker_exit_diagnostics",
    "_worker_log_path",
    "asr_gpu_blocked",
    "asr_gpu_cleanup_status",
    "bind_process_to_kill_on_close_job",
    "close_kill_on_close_job",
    "configure_ffmpeg_shared_runtime",
    "create_kill_on_close_job",
    "os",
    "retry_asr_gpu_cleanup",
    "shutdown_global_worker",
    "transcribe_and_align",
    "unclosed_job_handles",
    "worker_main",
]
