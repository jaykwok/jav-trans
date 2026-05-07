from __future__ import annotations

import asyncio
import json
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
from core.config import DEFAULT_SETTINGS
from core.job_context import JobContext
import main as pipeline_main
from main import run_asr_alignment_f0, run_translation_and_write
from utils.model_paths import PROJECT_ROOT
from web.models import JobSpec, JobState


gpu_queue: asyncio.Queue[JobState] = asyncio.Queue()
trans_queue: asyncio.Queue[tuple[JobState, Any]] = asyncio.Queue()
_executor = ThreadPoolExecutor(max_workers=4)

_jobs: dict[str, JobState] = {}
_cancel_events: dict[str, threading.Event] = {}
_state_lock = asyncio.Lock()
_jobs_path = PROJECT_ROOT / "temp" / "web" / "jobs.json"
_FINISHED_STATUSES = {"done", "failed", "cancelled"}
_RETRYABLE_STATUSES = {"failed", "cancelled"}
_last_progress_write_ts: float = 0.0

_EVENT_STAGE_STATUS = {
    "translation": "translating",
    "write_output": "writing",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_jobs_unlocked() -> None:
    payload = [job.model_dump() for job in _jobs.values()]
    tmp_path = _jobs_path.with_name(f"{_jobs_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(_jobs_path)


async def _persist_jobs() -> None:
    async with _state_lock:
        _write_jobs_unlocked()


_ACTIVE_STATUSES = {"queued", "asr", "translating", "writing"}


async def load_jobs() -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    if not _jobs_path.exists():
        return
    try:
        payload = json.loads(_jobs_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return
        loaded = {
            job.id: job
            for item in payload
            if isinstance(item, dict)
            for job in [JobState.model_validate(item)]
        }
    except Exception:
        return
    async with _state_lock:
        _jobs.clear()
        _jobs.update(loaded)
        changed = False
        for job in _jobs.values():
            if job.status in _ACTIVE_STATUSES:
                job.status = "failed"
                job.current_stage = "failed"
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
) -> JobState:
    async with _state_lock:
        current = _jobs.get(job.id)
        if current is None:
            return job
        if (
            expected_cancel_event is not None
            and _cancel_events.get(job.id) is not expected_cancel_event
        ):
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
        _write_jobs_unlocked()
        return current


def _new_cancel_event() -> threading.Event:
    return threading.Event()


def _is_pipeline_cancelled(exc: BaseException) -> bool:
    cancelled_type = getattr(pipeline_main, "PipelineCancelledError", None)
    return bool(cancelled_type is not None and isinstance(exc, cancelled_type))


def _resume_cache_job_id(job: JobState) -> str:
    resume_from = getattr(job.spec, "resume_from_job_id", "").strip()
    if not resume_from:
        return job.id
    try:
        cache_job_id = pipeline_main._sanitize_job_id(resume_from)
        video_path = job.spec.video_paths[0]
        aligned_path = (
            Path(_job_temp_dir(cache_job_id))
            / f"{Path(video_path).stem}.aligned_segments.json"
        )
        if not aligned_path.exists():
            return job.id
        cached = pipeline_main._try_load_aligned_segments(
            str(aligned_path),
            pipeline_main._get_audio_cache_key(video_path),
            job.spec.asr_backend,
        )
        if cached is not None:
            return cache_job_id
    except Exception:
        return job.id
    return job.id


def _job_temp_dir(job_id: str) -> str:
    return str((PROJECT_ROOT / "temp" / "web" / "jobs" / pipeline_main._sanitize_job_id(job_id)).resolve())


def _remove_job_temp_dir(job_id: str) -> None:
    safe_id = pipeline_main._sanitize_job_id(job_id)
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


def _normalize_llm_api_format(value: str) -> str:
    normalized = (value or "chat").strip().lower()
    return normalized if normalized in {"chat", "responses"} else "chat"


def _normalize_llm_reasoning_effort(value: str) -> str:
    normalized = (value or "max").strip().lower()
    return normalized if normalized in {"low", "medium", "max"} else "max"


def _snapshot_translation_settings(spec: JobSpec) -> JobSpec:
    updates: dict[str, str] = {}

    target_lang = getattr(spec, "target_lang", None)
    if target_lang is None or not str(target_lang).strip():
        updates["target_lang"] = _runtime_setting("TARGET_LANG", "简体中文") or "简体中文"

    if getattr(spec, "translation_glossary", None) is None:
        updates["translation_glossary"] = _runtime_setting("TRANSLATION_GLOSSARY", "")

    if getattr(spec, "llm_api_format", None) is None:
        updates["llm_api_format"] = _normalize_llm_api_format(
            _runtime_setting("LLM_API_FORMAT", "chat")
        )

    reasoning_effort = getattr(spec, "llm_reasoning_effort", None)
    if reasoning_effort is None or not str(reasoning_effort).strip():
        updates["llm_reasoning_effort"] = _normalize_llm_reasoning_effort(
            _runtime_setting("LLM_REASONING_EFFORT", "max")
        )

    return spec.model_copy(update=updates) if updates else spec


def _job_context(job: JobState) -> JobContext:
    return JobContext.from_spec(
        job.spec,
        job.id,
        _job_temp_dir(job.id),
        _translation_cache_path(job.id),
    )


def _run_asr_alignment_f0(job: JobState, cancel_event=None):
    events.set_current_job_id(job.id)
    from utils import hf_progress as _hf_progress

    _hf_progress.set_current_job_id(job.id)
    try:
        cancel_event = cancel_event or _cancel_events.setdefault(
            job.id,
            _new_cancel_event(),
        )
        return run_asr_alignment_f0(
            job.spec.video_paths[0],
            ctx=_job_context(job),
            cache_job_id=_resume_cache_job_id(job),
            cancel_event=cancel_event,
        )
    finally:
        _hf_progress.set_current_job_id("")


def _run_translation_and_write(job: JobState, asr_artifacts) -> list[str]:
    events.set_current_job_id(job.id)
    cancel_event = _cancel_events.setdefault(job.id, _new_cancel_event())
    return run_translation_and_write(
        job.spec.video_paths[0],
        asr_artifacts,
        ctx=_job_context(job),
        job_id=job.id,
        cancel_event=cancel_event,
    )


def _relative_artifacts(paths: list[str], output_dir: str | None) -> list[str]:
    base = Path(output_dir).expanduser() if output_dir else PROJECT_ROOT
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    base = base.resolve()
    result: list[str] = []
    for raw_path in paths:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            result.append(path.resolve().relative_to(base).as_posix())
        except ValueError:
            try:
                result.append(path.resolve().relative_to(PROJECT_ROOT).as_posix())
            except ValueError:
                result.append(str(path))
    return result


async def gpu_worker() -> None:
    while True:
        job = await gpu_queue.get()
        cancel_event: threading.Event | None = None
        try:
            async with _state_lock:
                if _jobs.get(job.id) is not job:
                    continue
                cancel_event = _cancel_events.setdefault(job.id, _new_cancel_event())
            if cancel_event.is_set() or job.status == "cancelled":
                await _set_job(
                    job,
                    status="cancelled",
                    expected_cancel_event=cancel_event,
                )
                continue
            await _set_job(
                job,
                status="asr",
                current_stage="asr",
                expected_cancel_event=cancel_event,
            )
            loop = asyncio.get_running_loop()
            asr_artifacts = await loop.run_in_executor(
                _executor,
                _run_asr_alignment_f0,
                job,
            )
            if cancel_event.is_set():
                await _set_job(
                    job,
                    status="cancelled",
                    current_stage="cancelled",
                    expected_cancel_event=cancel_event,
                )
                continue
            await _set_job(
                job,
                status="translating",
                current_stage="translation",
                expected_cancel_event=cancel_event,
            )
            await trans_queue.put((job, asr_artifacts))
        except Exception as exc:
            if (
                _is_pipeline_cancelled(exc)
                or job.status == "cancelled"
                or (cancel_event is not None and cancel_event.is_set())
            ):
                await _set_job(
                    job,
                    status="cancelled",
                    current_stage="cancelled",
                    expected_cancel_event=cancel_event,
                )
            else:
                await _set_job(
                    job,
                    status="failed",
                    error=str(exc),
                    expected_cancel_event=cancel_event,
                )
        finally:
            gpu_queue.task_done()


async def translation_worker() -> None:
    while True:
        job, asr_artifacts = await trans_queue.get()
        cancel_event: threading.Event | None = None
        try:
            async with _state_lock:
                if _jobs.get(job.id) is not job:
                    continue
                cancel_event = _cancel_events.setdefault(job.id, _new_cancel_event())
            if cancel_event.is_set() or job.status == "cancelled":
                await _set_job(
                    job,
                    status="cancelled",
                    expected_cancel_event=cancel_event,
                )
                continue
            await _set_job(
                job,
                status="translating",
                current_stage="translation_context",
                expected_cancel_event=cancel_event,
            )
            loop = asyncio.get_running_loop()
            output_paths = await loop.run_in_executor(
                _executor,
                _run_translation_and_write,
                job,
                asr_artifacts,
            )
            if cancel_event.is_set():
                await _set_job(
                    job,
                    status="cancelled",
                    current_stage="cancelled",
                    expected_cancel_event=cancel_event,
                )
                continue
            artifacts = _relative_artifacts(output_paths, job.spec.output_dir)
            await _set_job(
                job,
                status="done",
                current_stage="done",
                artifacts=artifacts,
                expected_cancel_event=cancel_event,
            )
        except Exception as exc:
            if (
                _is_pipeline_cancelled(exc)
                or job.status == "cancelled"
                or (cancel_event is not None and cancel_event.is_set())
            ):
                await _set_job(
                    job,
                    status="cancelled",
                    current_stage="cancelled",
                    expected_cancel_event=cancel_event,
                )
            else:
                await _set_job(
                    job,
                    status="failed",
                    error=str(exc),
                    expected_cancel_event=cancel_event,
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
        if job.status not in ("done", "failed", "cancelled"):
            job.status = "cancelled"
            job.current_stage = "cancelled"
        _write_jobs_unlocked()
        return True


async def retry_job(job_id: str) -> JobState | None:
    async with _state_lock:
        job = _jobs.get(job_id)
        if job is None or job.status not in _RETRYABLE_STATUSES:
            return None
        retried = job.model_copy(
            update={
                "status": "queued",
                "current_stage": None,
                "progress": {},
                "artifacts": [],
                "error": None,
            },
        )
        _cancel_events[job_id] = _new_cancel_event()
        _jobs[job_id] = retried
        _write_jobs_unlocked()
    await gpu_queue.put(retried)
    return retried


async def remove_job(job_id: str) -> bool:
    async with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if job.status in _FINISHED_STATUSES:
            _jobs.pop(job_id, None)
            _cancel_events.pop(job_id, None)
            _write_jobs_unlocked()
            _remove_job_temp_dir(job_id)
            return True

    return await cancel_job(job_id)


async def remove_finished_jobs() -> int:
    async with _state_lock:
        job_ids = [
            job_id
            for job_id, job in _jobs.items()
            if job.status in _FINISHED_STATUSES
        ]
        for job_id in job_ids:
            _jobs.pop(job_id, None)
            _cancel_events.pop(job_id, None)
            _remove_job_temp_dir(job_id)
        if job_ids:
            _write_jobs_unlocked()
        return len(job_ids)


async def evict_old_jobs(max_age_hours: int = 48) -> int:
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    cutoff_str = cutoff.isoformat(timespec="milliseconds")
    async with _state_lock:
        job_ids = [
            job_id
            for job_id, job in _jobs.items()
            if job.status in _FINISHED_STATUSES and job.created_at < cutoff_str
        ]
        for job_id in job_ids:
            _jobs.pop(job_id, None)
            _cancel_events.pop(job_id, None)
            _remove_job_temp_dir(job_id)
        if job_ids:
            _write_jobs_unlocked()
        return len(job_ids)


async def _eviction_loop() -> None:
    while True:
        await asyncio.sleep(3600)
        await evict_old_jobs()


async def update_job_progress(job_id: str, progress: dict[str, Any]) -> None:
    global _last_progress_write_ts
    async with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.progress = dict(progress)
        stage = progress.get("stage")
        if isinstance(stage, str) and stage:
            job.current_stage = stage
            status = _EVENT_STAGE_STATUS.get(stage)
            if status and job.status not in _FINISHED_STATUSES:
                job.status = status  # type: ignore[assignment]
        now = _time.monotonic()
        if now - _last_progress_write_ts >= 2.0:
            _write_jobs_unlocked()
            _last_progress_write_ts = now


async def shutdown_executor() -> None:
    _executor.shutdown(wait=False, cancel_futures=True)


async def start_workers(translation_workers: int | None = None) -> list[asyncio.Task]:
    count = translation_workers
    if count is None:
        count = max(1, int(os.getenv("TRANSLATION_PARALLEL_VIDEOS", "2")))
    tasks = [asyncio.create_task(gpu_worker())]
    tasks.extend(asyncio.create_task(translation_worker()) for _ in range(count))
    tasks.append(asyncio.create_task(_eviction_loop()))
    return tasks
