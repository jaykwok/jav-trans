"""Background GPU/ASR, translation, eviction workers and GPU cleanup coordination."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from core.typed_config import env_int
from pipeline import artifact_store
from web.models import JobState

log = logging.getLogger(__name__)


def close_asr_artifacts_logger(artifacts: Any, pipeline_main_mod: Any) -> None:
    """Safely close any open ``.run.log`` logger attached to ``AsrArtifacts``."""
    if artifacts is None or not hasattr(artifacts, "logger"):
        return
    if not isinstance(getattr(artifacts, "logger", None), logging.Logger):
        return
    closer = getattr(pipeline_main_mod, "_close_artifacts_logger", None)
    if callable(closer):
        try:
            closer(artifacts)
        except Exception:
            log.warning("Failed to close run logger on ASR artifacts", exc_info=True)


def gpu_cleanup_status(pm: Any) -> dict[str, Any]:
    """GPU ownership as the page needs to show it: 正在清理 / 清理失败 / 已释放."""
    status = dict(pm.resources.snapshot())
    status["asr"] = pm.asr_gpu.asr_gpu_cleanup_status()
    status["orphan_media_children"] = pm.subprocess_tools.unreleased_children()
    local = dict(pm.llm_backends.local_backend_drain_status())
    local["cleanup"] = pm.llm_backends.local_backend_cleanup_status()
    status["local_translation"] = local
    status["gpu_blocked"] = status["blocked"]
    status["pending_resources"] = status.pop("pending")
    status["gpu_permit"] = pm.gpu_admission.snapshot()
    return status


def gpu_admission_blocked(pm: Any) -> bool:
    """May an ASR stage start right now? Asked of every owner, not just ASR."""
    return pm.gpu_admission.blocked_for(pm.gpu_admission.ASR)


async def retry_gpu_cleanup(pm: Any) -> dict[str, Any]:
    """Manual 重试清理 / 重新检查. Runs off the loop: a kill attempt blocks."""
    await asyncio.to_thread(pm.resources.retry_pending)
    await asyncio.to_thread(pm.llm_backends.retry_local_backend_cleanup)
    return gpu_cleanup_status(pm)


async def wait_for_gpu_release(
    pm: Any,
    job: JobState,
    cancel_event: threading.Event,
    run_id: str,
) -> bool:
    """Hold a job back while a child we could not stop still owns the GPU."""
    announced = False
    while pm.gpu_admission_blocked():
        if cancel_event.is_set() or job.status in pm._CANCELLING_STATUSES:
            await pm._worker_set_job(
                job,
                status="cancelled",
                current_stage="cancelled",
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
            return False
        if not announced:
            await pm._worker_set_job(
                job,
                status="queued",
                current_stage="gpu_blocked",
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
            announced = True
        await asyncio.sleep(pm._GPU_BLOCKED_POLL_S)
    return True


async def gpu_worker(pm: Any) -> None:
    while True:
        job = await pm.gpu_queue.get()
        cancel_event: threading.Event | None = None
        run_id = ""
        asr_artifacts: Any = None
        enqueued_for_translation = False
        try:
            async with pm._state_lock:
                if pm._jobs.get(job.id) is not job:
                    continue
                cancel_event = pm._cancel_events.setdefault(job.id, pm._new_cancel_event())
                run_id = job.run_id
            if cancel_event.is_set() or job.status in pm._CANCELLING_STATUSES:
                await pm._worker_set_job(
                    job,
                    status="cancelled",
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
                continue
            if not await pm._wait_for_gpu_release(job, cancel_event, run_id):
                continue
            await pm._worker_set_job(
                job,
                status="asr",
                current_stage="asr",
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
            loop = asyncio.get_running_loop()
            asr_artifacts = await loop.run_in_executor(
                pm._get_executor(),
                pm._run_asr_alignment,
                job,
                cancel_event,
                run_id,
            )
            # Check cancel again after executor returns: if cancel happened inside the
            # executor (between its last checkpoint and return), the artifacts are
            # complete but stale. Close their logger and do not enqueue them.
            if cancel_event.is_set():
                close_asr_artifacts_logger(asr_artifacts, pm.pipeline_main)
                asr_artifacts = None
                await pm._worker_set_job(
                    job,
                    status="cancelled",
                    current_stage="cancelled",
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
                continue
            next_status = "writing" if job.spec.skip_translation else "translating"
            next_stage = "write_output" if job.spec.skip_translation else "translation"
            await pm._worker_set_job(
                job,
                status=next_status,
                current_stage=next_stage,
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
            await pm.trans_queue.put((job, asr_artifacts))
            enqueued_for_translation = True
        except Exception as exc:
            if (
                pm._is_pipeline_cancelled(exc)
                or job.status in pm._CANCELLING_STATUSES
                or (cancel_event is not None and cancel_event.is_set())
            ):
                await pm._worker_set_job(
                    job,
                    status="cancelled",
                    current_stage="cancelled",
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
            else:
                log.exception("ASR job failed: job_id=%s", job.id)
                await pm._worker_set_job(
                    job,
                    status="failed",
                    error=pm._public_error_message(exc),
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
        finally:
            if asr_artifacts is not None and not enqueued_for_translation:
                close_asr_artifacts_logger(asr_artifacts, pm.pipeline_main)
            pm.gpu_queue.task_done()


async def translation_worker(pm: Any) -> None:
    while True:
        job, asr_artifacts = await pm.trans_queue.get()
        cancel_event: threading.Event | None = None
        run_id = ""
        handed_to_translation = False
        try:
            async with pm._state_lock:
                if pm._jobs.get(job.id) is not job:
                    continue
                cancel_event = pm._cancel_events.setdefault(job.id, pm._new_cancel_event())
                run_id = job.run_id
            if cancel_event.is_set() or job.status in pm._CANCELLING_STATUSES:
                await pm._worker_set_job(
                    job,
                    status="cancelled",
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
                continue
            stage_status = "writing" if job.spec.skip_translation else "translating"
            stage_name = "write_output" if job.spec.skip_translation else "translation_context"
            await pm._worker_set_job(
                job,
                status=stage_status,
                current_stage=stage_name,
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
            loop = asyncio.get_running_loop()
            handed_to_translation = True
            output_paths = await loop.run_in_executor(
                pm._get_executor(),
                pm._run_translation_and_write,
                job,
                asr_artifacts,
                cancel_event,
                run_id,
            )
            close_asr_artifacts_logger(asr_artifacts, pm.pipeline_main)
            cancelled = cancel_event.is_set()
            _, committed = await asyncio.to_thread(
                pm._committed_job_outputs, job, verify=cancelled
            )
            if cancelled and committed is None:
                await pm._worker_set_job(
                    job,
                    status="cancelled",
                    current_stage="cancelled",
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
                continue
            artifacts = pm._relative_artifacts(output_paths, job.spec.output_dir)
            await pm._worker_set_job(
                job,
                status="done",
                current_stage="done",
                artifacts=artifacts,
                subtitle_cue_count=committed.get("subtitle_cue_count") if committed else None,
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
        except (artifact_store.OutputCommitError, artifact_store.OutputIntegrityError) as exc:
            close_asr_artifacts_logger(asr_artifacts, pm.pipeline_main)
            log.exception("Output commit/integrity failure for job_id=%s", job.id)
            await pm._worker_set_job(
                job,
                status="publish_failed",
                current_stage="publish_failed",
                error=str(exc),
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
        except artifact_store.OutputExportError as exc:
            close_asr_artifacts_logger(asr_artifacts, pm.pipeline_main)
            log.exception("Subtitle export failed for job_id=%s", job.id)
            try:
                committed = await asyncio.to_thread(
                    artifact_store.read_run_manifest, exc.lease, verify=True
                )
                if committed is None:
                    raise artifact_store.OutputIntegrityError()
                paths = artifact_store.manifest_paths(committed, project_root=pm.PROJECT_ROOT)
            except Exception as integrity_error:
                log.exception(
                    "Could not verify committed manifest after export failure for job_id=%s",
                    job.id,
                )
                error_text = (
                    str(integrity_error)
                    if isinstance(integrity_error, artifact_store.OutputIntegrityError)
                    else str(artifact_store.OutputIntegrityError())
                )
                await pm._worker_set_job(
                    job,
                    status="publish_failed",
                    current_stage="publish_failed",
                    error=error_text,
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
                continue
            await pm._worker_set_job(
                job,
                status="export_failed",
                current_stage="export_failed",
                error=str(exc),
                artifacts=pm._relative_artifacts(paths, job.spec.output_dir),
                subtitle_cue_count=committed.get("subtitle_cue_count"),
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
        except Exception as exc:
            close_asr_artifacts_logger(asr_artifacts, pm.pipeline_main)
            if (
                pm._is_pipeline_cancelled(exc)
                or job.status in pm._CANCELLING_STATUSES
                or (cancel_event is not None and cancel_event.is_set())
            ):
                await pm._worker_set_job(
                    job,
                    status="cancelled",
                    current_stage="cancelled",
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
            else:
                log.exception("translation job failed: job_id=%s", job.id)
                await pm._worker_set_job(
                    job,
                    status="failed",
                    error=pm._public_error_message(exc),
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
        finally:
            if not handed_to_translation:
                close_asr_artifacts_logger(asr_artifacts, pm.pipeline_main)
            pm.trans_queue.task_done()


async def eviction_loop(pm: Any) -> None:
    while True:
        await asyncio.sleep(3600)
        try:
            await pm.evict_old_jobs()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Periodic job eviction failed")


def _on_background_worker_done(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.critical(
            "Background worker %s exited unexpectedly",
            task.get_name(),
            exc_info=exc,
        )


async def start_workers(
    pm: Any,
    translation_workers: int | None = None,
) -> list[asyncio.Task]:
    count = translation_workers
    if count is None:
        count = env_int("TRANSLATION_PARALLEL_VIDEOS", 2, minimum=1)
    tasks = [asyncio.create_task(pm.gpu_worker(), name="web-gpu-worker")]
    tasks.extend(
        asyncio.create_task(
            pm.translation_worker(),
            name=f"web-translation-worker-{index + 1}",
        )
        for index in range(count)
    )
    tasks.append(asyncio.create_task(pm._eviction_loop(), name="web-eviction-loop"))
    for task in tasks:
        task.add_done_callback(_on_background_worker_done)
    return tasks
