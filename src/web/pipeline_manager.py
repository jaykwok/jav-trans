from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import threading
import time as _time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core import events
from core import resources
from core.config import DEFAULT_SETTINGS
from core.job_context import JobContext
from core.stage_errors import describe_stage_failure
from pipeline.artifacts import (
    AsrArtifacts,
    load_translation_artifacts_snapshot,
    write_translation_artifacts_snapshot,
)
from pipeline.aligned_cache import try_load_aligned_segments
from pipeline.audio import get_audio_cache_key, reclaim_stale_partials
from pipeline import gpu_worker as asr_gpu
from pipeline.ids import sanitize_job_id
from llm import backends as llm_backends
from utils import subprocess_tools
import main as pipeline_main
from main import run_asr_alignment, run_translation_and_write
from utils.model_paths import PROJECT_ROOT
from web.models import (
    JobSpec,
    JobState,
    normalize_llm_reasoning_effort as _normalize_llm_reasoning_effort,
)


log = logging.getLogger(__name__)
gpu_queue: asyncio.Queue[JobState] = asyncio.Queue()
trans_queue: asyncio.Queue[tuple[JobState, Any]] = asyncio.Queue()


def _executor_max_workers() -> int:
    # TRANSLATION_PARALLEL_VIDEOS drives how many translation_worker coroutines
    # start; translation jobs plus the lone ASR worker all run_in_executor into
    # this same pool, so size it to follow that env instead of silently capping
    # it at 4. The +1 gives the ASR path a slot alongside translation workers.
    try:
        parallel = max(1, int(os.getenv("TRANSLATION_PARALLEL_VIDEOS", "2")))
    except (TypeError, ValueError):
        parallel = 2
    return max(4, 1 + parallel)


_EXECUTOR_MAX_WORKERS = _executor_max_workers()
_executor = ThreadPoolExecutor(max_workers=_EXECUTOR_MAX_WORKERS)

_jobs: dict[str, JobState] = {}
_cancel_events: dict[str, threading.Event] = {}
_state_lock = asyncio.Lock()
_jobs_path = PROJECT_ROOT / "tmp" / "web" / "jobs.json"
_FINISHED_STATUSES = {"done", "failed", "cancelled"}
_RETRYABLE_STATUSES = {"failed", "cancelled"}
# "cancelling" is 已请求取消: the stop signal is out but the run has not reached
# a checkpoint yet. It is deliberately not retryable - a retry while the old run
# is still working is the situation run ids exist to survive, not to invite.
_CANCELLING_STATUSES = {"cancelled", "cancelling"}
# How often a job held back by a blocked GPU re-checks. Short enough that the
# queue resumes on its own once cleanup confirms.
_GPU_BLOCKED_POLL_S = 2.0
_last_progress_write_ts: dict[str, float] = {}
_jobs_write_lock = threading.Lock()

_EVENT_STAGE_STATUS = {
    "translation": "translating",
    "write_output": "writing",
}
_TRANSLATION_RETRY_STAGES = {"translation_context", "translation", "write_output"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_jobs_snapshot(snapshot: list[JobState]) -> None:
    # Persist a pre-captured snapshot. Callers capture the snapshot inside
    # _state_lock and call this outside the lock so the blocking dumps+replace
    # does not extend the critical section.
    payload = [job.model_dump() for job in snapshot]
    with _jobs_write_lock:
        _jobs_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _jobs_path.with_name(
            f"{_jobs_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for attempt in range(5):
            try:
                tmp_path.replace(_jobs_path)
                break
            except PermissionError:
                if attempt >= 4:
                    raise
                _time.sleep(0.02 * (attempt + 1))


def _write_jobs_unlocked() -> None:
    _write_jobs_snapshot(list(_jobs.values()))


_ACTIVE_STATUSES = {"queued", "asr", "translating", "writing", "cancelling"}


async def load_jobs() -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    if not _jobs_path.exists():
        return
    try:
        payload = json.loads(_jobs_path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Failed to read jobs state from %s", _jobs_path)
        return
    if not isinstance(payload, list):
        return
    loaded: dict[str, JobState] = {}
    rejected = 0
    for item in payload:
        if not isinstance(item, dict):
            rejected += 1
            continue
        try:
            job = JobState.model_validate(item)
        except Exception as exc:
            rejected += 1
            log.warning(
                "Skipping malformed job record in %s: %s",
                _jobs_path,
                exc,
            )
            continue
        loaded[job.id] = job
    if rejected:
        # Skipping a record only hides it; the next write rewrites the file from
        # what loaded and the record is gone for good. That is a real path, not
        # a hypothetical: the models take bare `Literal`s, so retiring an enum
        # value (`llm_reasoning_effort="none"`, 2026-08-14) makes every older
        # record unparseable at once. Keep a copy before anything can overwrite
        # it - the file is the only record of what the user ran.
        backup = _jobs_path.with_name(
            f"{_jobs_path.name}.rejected-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            if not backup.exists():
                backup.write_bytes(_jobs_path.read_bytes())
            log.warning(
                "Kept %d unreadable job record(s) in %s", rejected, backup
            )
        except OSError:
            log.exception("Failed to back up unreadable job records")
    async with _state_lock:
        _jobs.clear()
        _jobs.update(loaded)
        changed = False
        for job in _jobs.values():
            if job.status in _ACTIVE_STATUSES:
                job.status = "failed"
                job.error = "进程异常退出（上次运行被中断）"
                changed = True
        if changed:
            _write_jobs_unlocked()


async def _set_job(
    job: JobState,
    *,
    status: str | None = None,
    current_stage: str | None = None,
    progress: dict[str, Any] | None = None,
    artifacts: list[str] | None = None,
    error: str | None = None,
    expected_cancel_event: threading.Event | None = None,
    expected_run_id: str | None = None,
) -> JobState:
    persist_snapshot: list[JobState] | None = None
    async with _state_lock:
        current = _jobs.get(job.id)
        if current is None:
            return job
        # Both checks are the same question - "is the run reporting this still
        # the current one?" - asked of the two things a run is identified by.
        # The run id is the durable one (it survives a restart and travels with
        # every event); the event object also covers records written before run
        # ids existed.
        if (
            expected_cancel_event is not None
            and _cancel_events.get(job.id) is not expected_cancel_event
        ):
            return current
        if expected_run_id is not None and current.run_id != expected_run_id:
            return current
        if status is not None:
            current.status = status  # type: ignore[assignment]
        if current_stage is not None:
            current.current_stage = current_stage
        if progress is not None:
            current.progress = dict(progress)
        if artifacts is not None:
            current.artifacts = list(artifacts)
        if error is not None:
            current.error = error
        _jobs[job.id] = current
        persist_snapshot = list(_jobs.values())
    _write_jobs_snapshot(persist_snapshot)
    return current


def _new_cancel_event() -> threading.Event:
    return threading.Event()


def _new_run_id() -> str:
    return uuid.uuid4().hex


def active_run_id(job_id: str) -> str:
    """The run a job is executing right now, for filtering events by hand.

    Deliberately lock-free: the SSE fan-out runs on the event loop for every
    line and only needs the latest published value. A read that races a retry
    can only drop or admit one event at the moment of the swap, which the
    state-lock check on the write path then settles.
    """
    job = _jobs.get(job_id)
    return str(getattr(job, "run_id", "") or "") if job is not None else ""


def _discard_cancel_event(job_id: str) -> None:
    """Drop a job's cancel event, setting it on the way out.

    "Finished" is the job's view, not the worker threads'. A failed translation
    can leave batch workers mid-request, and they hold this event object
    directly - dropping it without setting it strands them with no stop signal
    at all, including the one `shutdown_executor` would have sent. Setting a
    finished job's event costs nothing: nobody reads it again.
    """
    event = _cancel_events.pop(job_id, None)
    if event is not None:
        event.set()


def _is_pipeline_cancelled(exc: BaseException) -> bool:
    cancelled_type = getattr(pipeline_main, "PipelineCancelledError", None)
    return bool(cancelled_type is not None and isinstance(exc, cancelled_type))


def _public_error_message(exc: BaseException) -> str:
    return describe_stage_failure(exc)


def _resume_cache_job_id(job: JobState) -> str:
    resume_from = getattr(job.spec, "resume_from_job_id", "").strip()
    if not resume_from:
        return job.id
    try:
        cache_job_id = sanitize_job_id(resume_from)
        video_path = job.spec.video_paths[0]
        aligned_path = (
            Path(_job_temp_dir(cache_job_id))
            / f"{Path(video_path).stem}.aligned_segments.json"
        )
        if not aligned_path.exists():
            return job.id
        cache_ctx = JobContext.from_spec(
            job.spec,
            cache_job_id,
            _job_temp_dir(cache_job_id),
            _translation_cache_path(cache_job_id),
        )
        backend_label, cache_signature = pipeline_main.aligned_cache_expectations_for_ctx(
            cache_ctx,
        )
        cached = try_load_aligned_segments(
            str(aligned_path),
            get_audio_cache_key(video_path),
            backend_label,
            cache_signature,
        )
        if cached is not None:
            return cache_job_id
    except Exception:
        return job.id
    return job.id


def _jobs_temp_root() -> Path:
    return PROJECT_ROOT / "tmp" / "web" / "jobs"


def _job_temp_dir(job_id: str) -> str:
    return str((_jobs_temp_root() / sanitize_job_id(job_id)).resolve())


async def reclaim_stale_audio_partials() -> int:
    """Drop extraction partials that no live run can still own.

    Publishing audio atomically means an interrupted extraction leaves a
    `.partial` instead of a poisoned cache entry; nothing reads those, so this
    is only about disk. Age-based on purpose - a CLI run beside the app may be
    writing one right now.
    """
    try:
        return await asyncio.to_thread(reclaim_stale_partials, _jobs_temp_root())
    except Exception:
        log.exception("Failed to reclaim stale audio partials")
        return 0


def _remove_job_temp_dir(job_id: str) -> None:
    safe_id = sanitize_job_id(job_id)
    path = Path(_job_temp_dir(safe_id)).resolve()
    if path.name != safe_id or not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _translation_cache_path(job_id: str) -> str:
    return str(Path(_job_temp_dir(job_id)) / "translation_cache.jsonl")


def _runtime_setting(key: str, fallback: str = "") -> str:
    if key in os.environ:
        return os.environ.get(key, "").strip()
    return str(DEFAULT_SETTINGS.get(key, fallback)).strip()


def _current_translation_settings() -> dict[str, str]:
    return {
        "target_lang": _runtime_setting("TARGET_LANG", "简体中文") or "简体中文",
        "translation_glossary": _runtime_setting("TRANSLATION_GLOSSARY", ""),
        "llm_reasoning_effort": _normalize_llm_reasoning_effort(
            _runtime_setting("LLM_REASONING_EFFORT", "medium")
        ),
    }


def _snapshot_translation_settings(spec: JobSpec) -> JobSpec:
    """Fill in whatever the caller left out, from the settings in effect now."""
    current = _current_translation_settings()
    updates: dict[str, str] = {}
    for name, value in current.items():
        given = getattr(spec, name, None)
        # An empty glossary is a real choice; an empty language or format is not.
        blank = given is None or (
            name != "translation_glossary" and not str(given).strip()
        )
        if blank:
            updates[name] = value
    return spec.model_copy(update=updates) if updates else spec


def _refresh_translation_settings(spec: JobSpec) -> JobSpec:
    """Re-read the translation settings that a retry should honour.

    The spec is a snapshot taken when the job was queued, which is what keeps a
    draining queue consistent while the panel is being edited. A retry is a new
    decision, though - usually made *because* a setting was just changed - so it
    runs with the current values instead of replaying the ones that failed.
    Without this, turning 推理强度 down to none and hitting 重试 re-ran the job
    with the effort it was queued with, and the thinking tokens kept coming.
    """
    return spec.model_copy(update=_current_translation_settings())


def _job_context(job: JobState) -> JobContext:
    return JobContext.from_spec(
        job.spec,
        job.id,
        _job_temp_dir(job.id),
        _translation_cache_path(job.id),
    )


def _run_asr_alignment(job: JobState, cancel_event: threading.Event, run_id: str):
    # Identity and stop signal both arrive as arguments and are never looked up
    # here. `_cancel_events[job.id]` and `_jobs[job.id].run_id` are the *current*
    # run's; a retry replaces both, so a thread resolving them after it was
    # handed to the executor would pick up the new run's (unset) event and stamp
    # its own late events with the new run's id.
    events.set_current_run(job.id, run_id)
    from utils import hf_progress as _hf_progress

    _hf_progress.set_current_job_id(job.id, run_id)
    try:
        artifacts = run_asr_alignment(
            job.spec.video_paths[0],
            ctx=_job_context(job),
            job_id=job.id,
            run_id=run_id,
            cache_job_id=_resume_cache_job_id(job),
            cancel_event=cancel_event,
        )
        if isinstance(artifacts, AsrArtifacts):
            snapshot_started = _time.perf_counter()
            pipeline_main._log_stage(
                artifacts.logger,
                "stage_start translation_handoff_snapshot",
            )
            write_translation_artifacts_snapshot(artifacts)
            snapshot_elapsed = _time.perf_counter() - snapshot_started
            artifacts.pipeline_timings["translation_handoff_snapshot_s"] = (
                snapshot_elapsed
            )
            pipeline_main._log_stage(
                artifacts.logger,
                "stage_done translation_handoff_snapshot elapsed=%.2fs"
                % snapshot_elapsed,
            )
        return artifacts
    finally:
        _hf_progress.set_current_job_id("")


def _run_translation_and_write(
    job: JobState,
    asr_artifacts,
    cancel_event: threading.Event,
    run_id: str,
) -> list[str]:
    # Same rule as the ASR path: use the event and run the worker captured for
    # *this* run. Re-reading `_cancel_events[job.id]` here handed a cancelled-
    # then-retried job's old translation the new run's event, and the old batch
    # kept spending tokens because its own cancel flag was no longer reachable.
    events.set_current_run(job.id, run_id)
    return run_translation_and_write(
        job.spec.video_paths[0],
        asr_artifacts,
        ctx=_job_context(job),
        job_id=job.id,
        run_id=run_id,
        cancel_event=cancel_event,
    )


def _job_stage_names(job: JobState) -> set[str]:
    stages = {str(job.current_stage or "")}
    progress = job.progress if isinstance(job.progress, dict) else {}
    stages.add(str(progress.get("stage") or ""))
    extra = progress.get("extra")
    if isinstance(extra, dict):
        stages.add(str(extra.get("stage") or ""))
    return {stage for stage in stages if stage}


def _load_translation_retry_artifacts(job: JobState):
    if not (_job_stage_names(job) & _TRANSLATION_RETRY_STAGES):
        return None
    try:
        return load_translation_artifacts_snapshot(_job_temp_dir(job.id))
    except Exception:
        return None


def _relative_artifacts(paths: list[str], output_dir: str | None) -> list[str]:
    base = Path(output_dir).expanduser() if output_dir else PROJECT_ROOT
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    base = base.resolve()
    result: list[str] = []
    for raw_path in paths:
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            continue
        if not resolved.is_file():
            continue
        try:
            result.append(resolved.relative_to(base).as_posix())
        except ValueError:
            try:
                result.append(resolved.relative_to(PROJECT_ROOT).as_posix())
            except ValueError:
                result.append(str(resolved))
    return result


def gpu_cleanup_status() -> dict[str, Any]:
    """GPU ownership as the page needs to show it: 正在清理 / 清理失败 / 已释放.

    Also carries the media children `run_cancellable` could not confirm it
    released, so "查看诊断信息" has one place to look instead of two.
    """
    status = dict(resources.snapshot())
    # The ASR worker's own view stays available for the parts of the page that
    # only speak about it, but it is no longer what "is the GPU free?" reads.
    status["asr"] = asr_gpu.asr_gpu_cleanup_status()
    status["orphan_media_children"] = subprocess_tools.unreleased_children()
    # A cancelled local request releases this process, not necessarily the
    # server's slot. Shown, never acted on: the server is shared.
    local = dict(llm_backends.local_backend_drain_status())
    # A local server we could not confirm we stopped, on the other hand, does
    # block: it still holds its VRAM and refuses to load a replacement.
    local["cleanup"] = llm_backends.local_backend_cleanup_status()
    status["local_translation"] = local
    # One answer for "may a GPU stage start?", covering every owner of the card.
    # A page (or a queue) reading only the ASR worker's state would call the GPU
    # free while a llama-server nobody could stop still had the model resident.
    status["gpu_blocked"] = status["blocked"]
    # Diagnostics: every pending record, whoever owns it.
    status["pending_resources"] = status.pop("pending")
    return status


def gpu_admission_blocked() -> bool:
    """Does anything still hold the GPU? Asked of every owner, not just ASR.

    A leaked Job Object handle deliberately does not count: the process inside
    it is confirmed gone, so it holds no VRAM. It is tracked and retried on its
    own, which is a different problem from "the card is occupied".
    """
    return resources.gpu_blocked()


async def retry_gpu_cleanup() -> dict[str, Any]:
    """Manual 重试清理 / 重新检查. Runs off the loop: a kill attempt blocks.

    One request against the resource registry, which retries only records that
    are already waiting for cleanup. It cannot escalate into "stop what is
    running now" - that used to be possible through the local backend, where the
    button shut down a healthy server another job was translating through.
    """
    await asyncio.to_thread(resources.retry_pending)
    # The local backend keeps one extra bookkeeping step of its own: a confirmed
    # stop lets the instance leave the backend registry.
    await asyncio.to_thread(llm_backends.retry_local_backend_cleanup)
    return gpu_cleanup_status()


async def _wait_for_gpu_release(
    job: JobState,
    cancel_event: threading.Event,
    run_id: str,
) -> bool:
    """Hold a job back while a child we could not stop still owns the GPU.

    Refusing to start a second worker is only half an answer: without this the
    job would fail one by one against a busy GPU, and the user would be left
    retrying by hand. Dispatch pauses instead, and resumes by itself as soon as
    the cleanup confirms. Returns False when the job was cancelled while waiting.
    """
    announced = False
    while gpu_admission_blocked():
        if cancel_event.is_set() or job.status in _CANCELLING_STATUSES:
            await _set_job(
                job,
                status="cancelled",
                current_stage="cancelled",
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
            return False
        if not announced:
            await _set_job(
                job,
                status="queued",
                current_stage="gpu_blocked",
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
            announced = True
        await asyncio.sleep(_GPU_BLOCKED_POLL_S)
    return True


async def gpu_worker() -> None:
    while True:
        job = await gpu_queue.get()
        cancel_event: threading.Event | None = None
        run_id = ""
        try:
            async with _state_lock:
                if _jobs.get(job.id) is not job:
                    continue
                cancel_event = _cancel_events.setdefault(job.id, _new_cancel_event())
                run_id = job.run_id
            if cancel_event.is_set() or job.status in _CANCELLING_STATUSES:
                await _set_job(
                    job,
                    status="cancelled",
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
                continue
            if not await _wait_for_gpu_release(job, cancel_event, run_id):
                continue
            await _set_job(
                job,
                status="asr",
                current_stage="asr",
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
            loop = asyncio.get_running_loop()
            asr_artifacts = await loop.run_in_executor(
                _executor,
                _run_asr_alignment,
                job,
                cancel_event,
                run_id,
            )
            # Check cancel again after executor returns: if cancel happened inside the
            # executor (between its last checkpoint and return), the artifacts are
            # complete but stale. Do not enqueue them for translation.
            if cancel_event.is_set():
                await _set_job(
                    job,
                    status="cancelled",
                    current_stage="cancelled",
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
                continue
            await _set_job(
                job,
                status="translating",
                current_stage="translation",
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
            await trans_queue.put((job, asr_artifacts))
        except Exception as exc:
            if (
                _is_pipeline_cancelled(exc)
                or job.status in _CANCELLING_STATUSES
                or (cancel_event is not None and cancel_event.is_set())
            ):
                await _set_job(
                    job,
                    status="cancelled",
                    current_stage="cancelled",
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
            else:
                log.exception("ASR job failed: job_id=%s", job.id)
                await _set_job(
                    job,
                    status="failed",
                    error=_public_error_message(exc),
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
        finally:
            gpu_queue.task_done()


async def translation_worker() -> None:
    while True:
        job, asr_artifacts = await trans_queue.get()
        cancel_event: threading.Event | None = None
        run_id = ""
        try:
            async with _state_lock:
                if _jobs.get(job.id) is not job:
                    continue
                cancel_event = _cancel_events.setdefault(job.id, _new_cancel_event())
                run_id = job.run_id
            if cancel_event.is_set() or job.status in _CANCELLING_STATUSES:
                await _set_job(
                    job,
                    status="cancelled",
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
                continue
            await _set_job(
                job,
                status="translating",
                current_stage="translation_context",
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
            loop = asyncio.get_running_loop()
            output_paths = await loop.run_in_executor(
                _executor,
                _run_translation_and_write,
                job,
                asr_artifacts,
                cancel_event,
                run_id,
            )
            if cancel_event.is_set():
                await _set_job(
                    job,
                    status="cancelled",
                    current_stage="cancelled",
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
                continue
            artifacts = _relative_artifacts(output_paths, job.spec.output_dir)
            await _set_job(
                job,
                status="done",
                current_stage="done",
                artifacts=artifacts,
                expected_cancel_event=cancel_event,
                expected_run_id=run_id,
            )
        except Exception as exc:
            if (
                _is_pipeline_cancelled(exc)
                or job.status in _CANCELLING_STATUSES
                or (cancel_event is not None and cancel_event.is_set())
            ):
                await _set_job(
                    job,
                    status="cancelled",
                    current_stage="cancelled",
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
            else:
                log.exception("translation job failed: job_id=%s", job.id)
                await _set_job(
                    job,
                    status="failed",
                    error=_public_error_message(exc),
                    expected_cancel_event=cancel_event,
                    expected_run_id=run_id,
                )
        finally:
            trans_queue.task_done()


async def create_job(spec: JobSpec) -> list[JobState]:
    jobs: list[JobState] = []
    parent_spec = _snapshot_translation_settings(spec)
    for video_path in spec.video_paths:
        child_spec = parent_spec.model_copy(update={"video_paths": [video_path]})
        job_id = uuid.uuid4().hex
        job = JobState(
            id=job_id,
            spec=child_spec,
            created_at=_utc_now(),
            run_id=_new_run_id(),
            status="queued",
        )
        async with _state_lock:
            _jobs[job.id] = job
            _cancel_events[job.id] = _new_cancel_event()
            _write_jobs_unlocked()
        await gpu_queue.put(job)
        jobs.append(job)
    return jobs


async def get_job(job_id: str) -> JobState | None:
    async with _state_lock:
        return _jobs.get(job_id)


async def list_jobs(status: str | None = None) -> list[JobState]:
    async with _state_lock:
        jobs = list(_jobs.values())
    if status:
        jobs = [job for job in jobs if job.status == status]
    return sorted(jobs, key=lambda item: item.created_at)


async def cancel_job(job_id: str) -> bool:
    async with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        event = _cancel_events.setdefault(job_id, _new_cancel_event())
        event.set()
        if job.status == "queued":
            # Nothing is executing it yet, so there is nothing to wind down.
            job.status = "cancelled"
            job.current_stage = "cancelled"
        elif job.status not in _FINISHED_STATUSES:
            # 已请求取消, not 已取消: a worker thread owns this run and is still
            # between checkpoints. Calling it "cancelled" here is what let the
            # page offer 重试 while the previous run was still spending tokens.
            # Whichever worker owns it flips it to "cancelled" once it stops.
            job.status = "cancelling"
            job.current_stage = "cancelling"
        _write_jobs_unlocked()
        return True


async def retry_job(job_id: str) -> JobState | None:
    retry_artifacts = None
    async with _state_lock:
        job = _jobs.get(job_id)
        if job is None or job.status not in _RETRYABLE_STATUSES:
            return None
        retry_artifacts = _load_translation_retry_artifacts(job)
        retried = job.model_copy(
            update={
                "spec": _refresh_translation_settings(job.spec),
                # A retry is a new execution of the same job: new run id, so
                # anything the previous run still emits can be told apart from
                # this one's instead of overwriting its progress.
                "run_id": _new_run_id(),
                "status": "queued",
                "current_stage": None,
                "progress": {},
                "artifacts": [],
                "error": None,
            },
        )
        # Same reason removal sets it: whatever was still holding the previous
        # run's event must be told to stop before the retry replaces it, or the
        # old worker keeps running under the same job id as the new one.
        _discard_cancel_event(job_id)
        _cancel_events[job_id] = _new_cancel_event()
        _jobs[job_id] = retried
        _write_jobs_unlocked()
    if retry_artifacts is not None:
        await trans_queue.put((retried, retry_artifacts))
    else:
        await gpu_queue.put(retried)
    return retried


async def remove_job(job_id: str) -> bool:
    removed_job: JobState | None = None
    async with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if job.status in _FINISHED_STATUSES:
            _jobs.pop(job_id, None)
            _discard_cancel_event(job_id)
            _last_progress_write_ts.pop(job_id, None)
            _write_jobs_unlocked()
            removed_job = job

    if removed_job is not None:
        _remove_job_temp_dir(removed_job.id)
        return True

    return await cancel_job(job_id)


async def remove_finished_jobs() -> int:
    removed_jobs: list[JobState] = []
    async with _state_lock:
        removed_jobs = [
            job
            for job in _jobs.values()
            if job.status in _FINISHED_STATUSES
        ]
        for job in removed_jobs:
            _jobs.pop(job.id, None)
            _discard_cancel_event(job.id)
            _last_progress_write_ts.pop(job.id, None)
        if removed_jobs:
            _write_jobs_unlocked()
    for job in removed_jobs:
        _remove_job_temp_dir(job.id)
    return len(removed_jobs)


async def evict_old_jobs(max_age_hours: int = 48) -> int:
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    cutoff_str = cutoff.isoformat(timespec="milliseconds")
    removed_jobs: list[JobState] = []
    async with _state_lock:
        removed_jobs = [
            job
            for job in _jobs.values()
            if job.status in _FINISHED_STATUSES and job.created_at < cutoff_str
        ]
        for job in removed_jobs:
            _jobs.pop(job.id, None)
            _discard_cancel_event(job.id)
            _last_progress_write_ts.pop(job.id, None)
        if removed_jobs:
            _write_jobs_unlocked()
    for job in removed_jobs:
        _remove_job_temp_dir(job.id)
    return len(removed_jobs)


async def _eviction_loop() -> None:
    while True:
        await asyncio.sleep(3600)
        await evict_old_jobs()


async def update_job_progress(
    job_id: str,
    progress: dict[str, Any],
    *,
    run_id: str = "",
) -> None:
    persist_snapshot: list[JobState] | None = None
    async with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        # Inside the state lock, so the run cannot be swapped between the check
        # and the write. An event with no run id (the CLI, or anything written
        # before run ids) is not filtered - there is nothing to compare.
        if run_id and job.run_id and run_id != job.run_id:
            return
        job.progress = dict(progress)
        stage = progress.get("stage")
        # A cancelled run keeps emitting until it reaches a checkpoint, and
        # those events are its *own* - the run id matches, so no filter catches
        # them. Counters may still move (they are what the run was doing), but
        # the cancel stage is not theirs to undo: letting a translation event
        # write "translating" back over 停止中 put the card straight back to
        # looking like a job that had restarted itself. Only the worker that
        # owns the run moves it out of cancelling, and only into a final state.
        cancel_pending = job.status == "cancelling"
        if isinstance(stage, str) and stage and not cancel_pending:
            job.current_stage = stage
            status = _EVENT_STAGE_STATUS.get(stage)
            if status and job.status not in _FINISHED_STATUSES:
                job.status = status  # type: ignore[assignment]
        now = _time.monotonic()
        # Throttle disk persistence per job so one busy job cannot starve
        # another job's progress from being written within the 2s window.
        if now - _last_progress_write_ts.get(job_id, 0.0) >= 2.0:
            persist_snapshot = list(_jobs.values())
            _last_progress_write_ts[job_id] = now
    if persist_snapshot is not None:
        _write_jobs_snapshot(persist_snapshot)


async def shutdown_executor() -> None:
    global _executor
    # Signal in-flight ASR/translation tasks to bail at their next checkpoint.
    # generate() itself is uninterruptible, so we don't block on the thread:
    # the NVIDIA driver reclaims CUDA when the process exits, and these
    # daemon=False threads are awaited by threading._shutdown on interpreter
    # exit anyway. Setting cancel events just lets long batches finish sooner.
    for event in _cancel_events.values():
        event.set()
    _executor.shutdown(wait=False, cancel_futures=True)
    _executor = ThreadPoolExecutor(max_workers=_EXECUTOR_MAX_WORKERS)


async def start_workers(translation_workers: int | None = None) -> list[asyncio.Task]:
    count = translation_workers
    if count is None:
        count = max(1, int(os.getenv("TRANSLATION_PARALLEL_VIDEOS", "2")))
    tasks = [asyncio.create_task(gpu_worker())]
    tasks.extend(asyncio.create_task(translation_worker()) for _ in range(count))
    tasks.append(asyncio.create_task(_eviction_loop()))
    return tasks
