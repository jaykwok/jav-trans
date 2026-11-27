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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from weakref import WeakKeyDictionary

import main as pipeline_main
from core import events, gpu_admission, resources, updater  # noqa: F401
from core.config import DEFAULT_REASONING_EFFORT, DEFAULT_SETTINGS
from core.diagnostics import current_process_log_hint
from core.job_context import JobContext
from core.stage_errors import describe_stage_failure
from core.typed_config import env_int
from llm import backends as llm_backends  # noqa: F401
from main import run_asr_alignment, run_translation_and_write
from pipeline import artifact_store, gpu_worker as asr_gpu  # noqa: F401
from pipeline.aligned_cache import try_load_aligned_segments
from pipeline.artifacts import (
    AsrArtifacts,
    load_translation_artifacts_snapshot,
    write_translation_artifacts_snapshot,
)
from pipeline.audio import get_audio_cache_key, reclaim_stale_partials
from pipeline.ids import sanitize_job_id
from utils import subprocess_tools  # noqa: F401
from utils.model_paths import PROJECT_ROOT
from web import legacy_migration, output_leases, state_clock, workers
from web.job_store import JobPersistenceError, JobStoreWriter
from web.legacy_migration import (
    LEGACY_CLEANUP_FILE_NAME as _LEGACY_CLEANUP_FILE_NAME,
    LEGACY_INTEGRITY_REVIEW_FILE_NAME as _LEGACY_INTEGRITY_REVIEW_FILE_NAME,
    read_pending_legacy_cleanup_ids as _read_pending_legacy_cleanup_ids,  # noqa: F401
    rewrite_legacy_job_record as _rewrite_legacy_job_record,  # noqa: F401
    strip_retired_spec_fields as _strip_retired_spec_fields,
    write_pending_legacy_cleanup_ids as _write_pending_legacy_cleanup_ids,  # noqa: F401
)
from web.models import (
    JobSpec,
    JobState,
    normalize_llm_reasoning_effort as _normalize_llm_reasoning_effort,
)
from web.output_leases import JobOutputConflict
from web.state_clock import (
    EPOCH_FILE_NAME as _EPOCH_FILE_NAME,
    TOMBSTONE_LIMIT as _TOMBSTONE_LIMIT,
    StateClock,
)

log = logging.getLogger(__name__)
gpu_queue: asyncio.Queue[JobState] = asyncio.Queue()
trans_queue: asyncio.Queue[tuple[JobState, Any]] = asyncio.Queue()


def _executor_max_workers() -> int:
    # TRANSLATION_PARALLEL_VIDEOS drives how many translation_worker coroutines
    # start; translation jobs plus the lone ASR worker all run_in_executor into
    # this same pool, so size it to follow that env instead of silently capping
    # it at 4. The +1 gives the ASR path a slot alongside translation workers.
    parallel = env_int("TRANSLATION_PARALLEL_VIDEOS", 2, minimum=1)
    return max(4, 1 + parallel)


_EXECUTOR_MAX_WORKERS = _executor_max_workers()
_executor_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=_EXECUTOR_MAX_WORKERS)
        return _executor


_jobs: dict[str, JobState] = {}
_cancel_events: dict[str, threading.Event] = {}
_state_lock = asyncio.Lock()
_jobs_path = PROJECT_ROOT / "cache" / "web" / "jobs.json"
_FINISHED_STATUSES = {"done", "failed", "cancelled", "publish_failed", "export_failed"}
_RETRYABLE_STATUSES = {"failed", "cancelled"}
# "cancelling" is 已请求取消: the stop signal is out but the run has not reached
# a checkpoint yet. It is deliberately not retryable - a retry while the old run
# is still working is the situation run ids exist to survive, not to invite.
_CANCELLING_STATUSES = {"cancelled", "cancelling"}
_ACTIVE_STATUSES = {"queued", "asr", "translating", "writing", "cancelling", "saving"}
_DELIVERY_FAILURES = {"save_failed", "publish_failed", "export_failed"}
_GPU_BLOCKED_POLL_S = 2.0
_PROGRESS_PERSIST_INTERVAL_S = 2.0
_jobs_write_lock = threading.Lock()

_EVENT_STAGE_STATUS = {
    "translation": "translating",
    "write_output": "writing",
}
_TRANSLATION_RETRY_STAGES = {"translation_context", "translation", "write_output"}

_progress_persist_tasks: dict[str, asyncio.Task[None]] = {}
_progress_last_persist_at: dict[str, float] = {}


class JobStateConflictError(RuntimeError):
    """Raised when an operation conflicts with a job's current persistence state."""


class UpdateInProgressError(RuntimeError):
    """Raised when creating or retrying a job while an online self-update is active."""

    def __init__(self) -> None:
        super().__init__(
            "正在执行在线升级，暂不接收新任务或重试任务；如需继续处理任务请先取消升级或等待重启完成。"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_jobs_snapshot(
    snapshot: list[JobState],
    *,
    path: Path | None = None,
    commit: Callable[[Callable[[], None]], None] | None = None,
) -> None:
    gate = getattr(commit, "__self__", None)
    if callable(getattr(gate, "is_cancelled", None)) and gate.is_cancelled():
        raise JobPersistenceError("保存请求已超时或取消，未提交这次任务变更。")
    target = path if path is not None else _jobs_path
    payload = [job.model_dump() for job in snapshot]
    with _jobs_write_lock:
        if callable(getattr(gate, "is_cancelled", None)) and gate.is_cancelled():
            raise JobPersistenceError("保存请求已超时或取消，未提交这次任务变更。")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())

            def replace() -> None:
                for attempt in range(5):
                    try:
                        tmp_path.replace(target)
                        return
                    except PermissionError:
                        if attempt >= 4:
                            raise
                        _time.sleep(0.02 * (attempt + 1))

            if commit is not None:
                commit(replace)
            else:
                replace()
        finally:
            tmp_path.unlink(missing_ok=True)


class _JobStoreWriter(JobStoreWriter):
    def __init__(self) -> None:
        super().__init__(lambda snapshot: _write_jobs_snapshot(snapshot))


_job_store_writer = _JobStoreWriter()
_mutation_locks: WeakKeyDictionary = WeakKeyDictionary()


def _mutation_gate() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    return _mutation_locks.setdefault(loop, asyncio.Lock())


def _persist(snapshot: list[JobState], *, confirm: bool = False):
    target = _jobs_path
    return _job_store_writer.enqueue(
        snapshot,
        confirm=confirm,
        guarded_write=lambda frozen, commit: _write_jobs_snapshot(
            frozen, path=target, commit=commit
        ),
    )


def _write_jobs_unlocked(*, confirm: bool = False):
    return _persist(list(_jobs.values()), confirm=confirm)


def _apply_jobs(jobs: dict[str, JobState]) -> None:
    # Preserve worker references within one run. Retries replace the object.
    updated = {}
    for job_id, candidate in jobs.items():
        current = _jobs.get(job_id)
        if current is not None and current.run_id == candidate.run_id:
            for name in type(candidate).model_fields:
                setattr(current, name, getattr(candidate, name))
            updated[job_id] = current
        else:
            updated[job_id] = candidate
    _jobs.clear()
    _jobs.update(updated)


def _cancel_progress_persist_for(job_id: str) -> None:
    task = _progress_persist_tasks.pop(job_id, None)
    if task is not None and not task.done():
        task.cancel()
    _progress_last_persist_at.pop(job_id, None)


async def _commit_jobs(
    jobs: dict[str, JobState],
    *,
    deletions: tuple[str, ...] = (),
    revision: int = 0,
    new_cancel_events: dict[str, threading.Event] | None = None,
    discard_cancel_ids: tuple[str, ...] = (),
) -> None:
    # Mutation callers hold the gate, never the reader lock while awaiting IO.
    await _job_store_writer.confirm(_persist(list(jobs.values()), confirm=True))
    async with _state_lock:
        _apply_jobs(jobs)
        for job_id in discard_cancel_ids:
            _discard_cancel_event(job_id)
            _cancel_progress_persist_for(job_id)
        if new_cancel_events:
            for job_id, event in new_cancel_events.items():
                _discard_cancel_event(job_id)
                _cancel_events[job_id] = event
        for job_id in deletions:
            _discard_cancel_event(job_id)
            _cancel_progress_persist_for(job_id)
            _entomb(job_id, revision)


# ---------------------------------------------------------------------------
# StateClock facade (preserves test monkeypatching of _epoch/_revision/_tombstones)
# ---------------------------------------------------------------------------

_state_clock = StateClock(tombstone_limit=_TOMBSTONE_LIMIT)
_revision_counter = 0
_epoch = 0
_tombstones: dict[str, int] = _state_clock.tombstones


def _epoch_path() -> Path:
    return _jobs_path.with_name(_EPOCH_FILE_NAME)


def state_epoch() -> int:
    global _epoch
    resolved = _state_clock.resolve_epoch(
        _epoch_path(),
        current_epoch=_epoch,
        time_fn=_time.time,
    )
    _epoch = resolved
    return resolved


def _next_revision() -> int:
    """Tick the state clock. Callers hold ``_state_lock``."""
    global _revision_counter
    _revision_counter += 1
    _state_clock.revision = _revision_counter
    return _revision_counter


def _publish(job: JobState) -> JobState:
    """Stamp a job with the revision that dates this change."""
    global _revision_counter
    if job.revision > _revision_counter:
        _revision_counter = job.revision
        _state_clock.revision = _revision_counter
    job.revision = _next_revision()
    return job


def _entomb(job_id: str, revision: int) -> None:
    """Record a deletion. Callers hold ``_state_lock``."""
    state_clock.entomb_in(_tombstones, job_id, revision, limit=_TOMBSTONE_LIMIT)


def deleted_revision(job_id: str) -> int | None:
    """The reading at which this job was deleted, if it is still remembered."""
    return _tombstones.get(job_id)


# ---------------------------------------------------------------------------
# Legacy migration facade
# ---------------------------------------------------------------------------


def _legacy_cleanup_path() -> Path:
    return _jobs_path.with_name(_LEGACY_CLEANUP_FILE_NAME)


def _legacy_integrity_review_path() -> Path:
    return _jobs_path.with_name(_LEGACY_INTEGRITY_REVIEW_FILE_NAME)


def _drain_pending_legacy_temp_dirs(extra_job_ids: list[str] | None = None) -> None:
    legacy_migration.drain_pending_legacy_temp_dirs(
        extra_job_ids,
        project_root=PROJECT_ROOT,
        jobs_path=_jobs_path,
        rmtree_fn=shutil.rmtree,
    )


def _inspect_legacy_job_outputs(job: JobState) -> tuple[bool, dict[str, Any] | None]:
    return legacy_migration.inspect_legacy_job_outputs(
        job,
        project_root=PROJECT_ROOT,
        committed_outputs_fn=_committed_job_outputs,
        utc_now_fn=_utc_now,
    )


async def _migrate_legacy_web_state() -> None:
    await legacy_migration.migrate_legacy_web_state(
        project_root=PROJECT_ROOT,
        jobs_path=_jobs_path,
        epoch_path=_epoch_path(),
        epoch_file_name=_EPOCH_FILE_NAME,
        committed_outputs_fn=_committed_job_outputs,
        write_jobs_snapshot_fn=_write_jobs_snapshot,
        drain_fn=_drain_pending_legacy_temp_dirs,
        utc_now_fn=_utc_now,
    )


async def load_jobs() -> None:
    await _migrate_legacy_web_state()
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    if not _jobs_path.exists():
        return
    try:
        payload = json.loads(await asyncio.to_thread(_jobs_path.read_text, encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise JobPersistenceError("无法读取已有任务记录；请恢复 jobs.json 后重启，原文件未覆盖。") from exc
    if not isinstance(payload, list):
        raise JobPersistenceError("任务记录格式无效；请恢复 jobs.json 后重启，原文件未覆盖。")
    loaded: dict[str, JobState] = {}
    rejected = 0
    for item in payload:
        if not isinstance(item, dict):
            rejected += 1
            continue
        try:
            item = _strip_retired_spec_fields(item)
            job = JobState.model_validate(item)
        except Exception as exc:
            rejected += 1
            log.warning("Skipping malformed job record in %s: %s", _jobs_path, exc)
            continue
        loaded[job.id] = job
    if rejected:
        backup = _jobs_path.with_name(
            f"{_jobs_path.name}.rejected-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            if not backup.exists():
                backup.write_bytes(_jobs_path.read_bytes())
            log.warning("Kept %d unreadable job record(s) in %s", rejected, backup)
        except OSError as exc:
            raise JobPersistenceError("无法备份不兼容的任务记录；已停止加载，原文件未覆盖。") from exc
    changed = False
    completed: list[str] = []
    for job in loaded.values():
        if job.status in _ACTIVE_STATUSES | {"save_failed", "publish_failed", "export_failed", "failed", "done"}:
            try:
                lease, committed = await asyncio.to_thread(_committed_job_outputs, job)
                prepared = (
                    await asyncio.to_thread(artifact_store.read_prepared_manifest, lease)
                    if lease and committed is None
                    else None
                )
            except artifact_store.OutputIntegrityError as exc:
                log.exception("Output integrity check failed for job %s during startup", job.id)
                job.status = "publish_failed"
                job.current_stage = "publish_failed"
                job.error = str(exc)
                job.pending_status = None
                job.storage_error = None
                changed = True
                continue
            if lease is not None and committed is not None:
                job.status = "done" if artifact_store.output_exported(lease) else "export_failed"
                job.current_stage = job.status
                job.artifacts = _relative_artifacts(
                    artifact_store.manifest_paths(committed, project_root=PROJECT_ROOT),
                    job.spec.output_dir,
                )
                job.subtitle_cue_count = committed.get("subtitle_cue_count")
                if job.status == "done":
                    completed.append(job.id)
                job.error = None if job.status == "done" else "产物已保存，请重试导出字幕。"
            elif prepared is not None:
                job.status = "publish_failed"
                job.current_stage = "publish_failed"
                job.error = "完整产物已保存，请重试提交；无需重新计算。"
            elif lease is not None and job.status in {"done", "publish_failed", "export_failed"}:
                job.status = "publish_failed"
                job.current_stage = "publish_failed"
                job.error = str(artifact_store.OutputIntegrityError())
            elif job.pending_status in {"cancelled", "failed"}:
                job.status = job.pending_status
            elif job.status in _ACTIVE_STATUSES | {"save_failed"}:
                job.status = "failed"
                job.error = "进程异常退出（上次运行被中断）"
            else:
                continue
            job.pending_status = None
            job.storage_error = None
            changed = True
    global _revision_counter
    async with _mutation_gate():
        async with _state_lock:
            _jobs.clear()
            _jobs.update(loaded)
            _revision_counter = max(
                [_revision_counter, 0] + [job.revision for job in loaded.values()]
            )
            _state_clock.revision = _revision_counter
            if changed:
                for job in loaded.values():
                    _publish(job)
        if changed:
            await _commit_jobs(dict(loaded))
    for job_id in completed:
        await asyncio.to_thread(_remove_job_temp_dir, job_id)


async def _set_job(
    job: JobState,
    *,
    status: str | None = None,
    current_stage: str | None = None,
    progress: dict[str, Any] | None = None,
    artifacts: list[str] | None = None,
    subtitle_cue_count: int | None = None,
    error: str | None = None,
    expected_cancel_event: threading.Event | None = None,
    expected_run_id: str | None = None,
) -> JobState:
    async with _mutation_gate():
        async with _state_lock:
            current = _jobs.get(job.id)
            if current is None:
                return job
            if (
                expected_cancel_event is not None
                and _cancel_events.get(job.id) is not expected_cancel_event
            ):
                return current
            if expected_run_id is not None and current.run_id != expected_run_id:
                return current
            candidate = current.model_copy(deep=True)
            for name, value in (
                ("status", status),
                ("current_stage", current_stage),
                ("progress", progress),
                ("artifacts", artifacts),
                ("error", error),
                ("subtitle_cue_count", subtitle_cue_count),
            ):
                if value is not None:
                    setattr(candidate, name, value)
            confirm = candidate.status in _FINISHED_STATUSES
            if candidate.status == "done":
                candidate.error = None
            if confirm:
                waiting = candidate.model_copy(
                    update={
                        "status": "saving",
                        "pending_status": candidate.status,
                        "storage_error": None,
                    }
                )
                _apply_jobs({**_jobs, job.id: _publish(waiting)})
                candidate.pending_status = None
                candidate.storage_error = None
            candidate = _publish(candidate)
            snapshot = {**_jobs, job.id: candidate}
        if confirm:
            try:
                await _commit_jobs(snapshot, discard_cancel_ids=(job.id,))
            except JobPersistenceError as exc:
                async with _state_lock:
                    failed = candidate.model_copy(
                        update={
                            "status": "save_failed",
                            "pending_status": candidate.status,
                            "storage_error": str(exc),
                        }
                    )
                    _apply_jobs({**_jobs, job.id: _publish(failed)})
                raise
        else:
            async with _state_lock:
                _apply_jobs(snapshot)
                _persist(list(_jobs.values()))
        return _jobs[job.id]


def _new_cancel_event() -> threading.Event:
    return threading.Event()


def _new_run_id() -> str:
    return uuid.uuid4().hex


def active_run_id(job_id: str) -> str:
    """The run a job is executing right now, for filtering events by hand."""
    job = _jobs.get(job_id)
    return str(getattr(job, "run_id", "") or "") if job is not None else ""


def _discard_cancel_event(job_id: str) -> None:
    """Drop a job's cancel event, setting it on the way out."""
    event = _cancel_events.pop(job_id, None)
    if event is not None:
        event.set()


def signal_all_cancel_events() -> None:
    """Signal every registered job cancel event."""
    for event in list(_cancel_events.values()):
        event.set()


def _is_pipeline_cancelled(exc: BaseException) -> bool:
    cancelled_type = getattr(pipeline_main, "PipelineCancelledError", None)
    return isinstance(exc, artifact_store.OutputsSupersededError) or bool(
        cancelled_type is not None and isinstance(exc, cancelled_type)
    )


def _public_error_message(exc: BaseException) -> str:
    return describe_stage_failure(exc)


# ---------------------------------------------------------------------------
# Output leases & scratch path facade
# ---------------------------------------------------------------------------


def _jobs_temp_root() -> Path:
    return output_leases.jobs_temp_root(PROJECT_ROOT)


def _job_temp_dir(job_id: str) -> str:
    return output_leases.job_temp_dir(job_id, jobs_temp_root_fn=_jobs_temp_root)


async def reclaim_stale_audio_partials() -> int:
    try:
        return await asyncio.to_thread(reclaim_stale_partials, _jobs_temp_root())
    except Exception:
        log.exception("Failed to reclaim stale audio partials")
        return 0


def _remove_job_temp_dir(job_id: str) -> None:
    output_leases.remove_job_temp_dir(job_id, jobs_temp_root_fn=_jobs_temp_root)


def _translation_cache_path(job_id: str) -> str:
    return output_leases.translation_cache_path(job_id, job_temp_dir_fn=_job_temp_dir)


def _job_context(job: JobState) -> JobContext:
    return output_leases.job_context(
        job,
        job_temp_dir_fn=_job_temp_dir,
        translation_cache_path_fn=_translation_cache_path,
    )


def output_lease_for_job(job: JobState) -> artifact_store.OutputLease | None:
    return output_leases.output_lease_for_job(
        job,
        project_root=PROJECT_ROOT,
        job_context_fn=_job_context,
        output_target_fn=pipeline_main.output_target_for_ctx,
    )


def _job_output_target(job: JobState) -> Path:
    return output_leases.job_output_target(
        job,
        job_context_fn=_job_context,
        output_target_fn=pipeline_main.output_target_for_ctx,
    )


def _claim_job_output(job: JobState, target: Path) -> artifact_store.OutputLease:
    return output_leases.claim_job_output(job, target, project_root=PROJECT_ROOT)


def _reserving_job(target: Path, jobs: list[JobState]) -> JobState | None:
    return output_leases.reserving_job(
        target,
        jobs,
        active_statuses=_ACTIVE_STATUSES,
        delivery_failures=_DELIVERY_FAILURES,
        job_output_target_fn=_job_output_target,
        output_lease_fn=output_lease_for_job,
    )


def _same_submission(first: JobSpec, second: JobSpec) -> bool:
    return output_leases.same_submission(first, second, project_root=PROJECT_ROOT)


async def _rollback_claims(claims: list[artifact_store.OutputLease]) -> None:
    await output_leases.rollback_claims(claims)


def _committed_job_outputs(
    job: JobState,
    *,
    verify: bool = True,
) -> tuple[artifact_store.OutputLease | None, dict[str, Any] | None]:
    return output_leases.committed_job_outputs(
        job,
        verify=verify,
        output_lease_fn=output_lease_for_job,
    )


def retry_requires_computation(job: JobState) -> bool:
    return output_leases.retry_requires_computation(
        job,
        delivery_failures=_DELIVERY_FAILURES,
        committed_outputs_fn=_committed_job_outputs,
    )


def _relative_artifacts(paths: list[str], output_dir: str | None) -> list[str]:
    return output_leases.relative_artifacts(paths, output_dir, project_root=PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Runtime translation setting snapshot & pipeline execution entrypoints
# ---------------------------------------------------------------------------


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
        log.warning(
            "Failed to inspect resume aligned cache from job %s for job %s; falling back to full ASR",
            resume_from,
            job.id,
            exc_info=True,
        )
        return job.id
    return job.id


def _runtime_setting(key: str, fallback: str = "") -> str:
    if key in os.environ:
        return os.environ.get(key, "").strip()
    return str(DEFAULT_SETTINGS.get(key, fallback)).strip()


def _current_translation_settings() -> dict[str, str]:
    return {
        "target_lang": _runtime_setting("TARGET_LANG", "简体中文") or "简体中文",
        "translation_glossary": _runtime_setting("TRANSLATION_GLOSSARY", ""),
        "llm_reasoning_effort": _normalize_llm_reasoning_effort(
            _runtime_setting("LLM_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
        ),
    }


def _snapshot_translation_settings(spec: JobSpec) -> JobSpec:
    current = _current_translation_settings()
    updates: dict[str, str] = {}
    for name, value in current.items():
        given = getattr(spec, name, None)
        blank = given is None or (name != "translation_glossary" and not str(given).strip())
        if blank:
            updates[name] = value
    return spec.model_copy(update=updates) if updates else spec


def _refresh_translation_settings(spec: JobSpec) -> JobSpec:
    return spec.model_copy(update=_current_translation_settings())


def _run_asr_alignment(job: JobState, cancel_event: threading.Event, run_id: str):
    from utils import hf_progress as _hf_progress

    video_name = os.path.basename(job.spec.video_paths[0]) if job.spec.video_paths else ""
    with events.scoped_run_identity(job.id, run_id, video=video_name):
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
                try:
                    write_translation_artifacts_snapshot(artifacts)
                except BaseException:
                    snapshot_logger = (
                        artifacts.logger
                        if isinstance(artifacts.logger, logging.Logger)
                        else log
                    )
                    snapshot_logger.exception(
                        "translation_handoff_snapshot failed for job %s", job.id
                    )
                    pipeline_main._close_artifacts_logger(artifacts)
                    raise
                snapshot_elapsed = _time.perf_counter() - snapshot_started
                artifacts.pipeline_timings["translation_handoff_snapshot_s"] = snapshot_elapsed
                pipeline_main._log_stage(
                    artifacts.logger,
                    "stage_done translation_handoff_snapshot elapsed=%.2fs" % snapshot_elapsed,
                )
            return artifacts
        except SystemExit as exc:
            log.exception(
                "ASR worker thread hit unexpected SystemExit(%r) for job %s",
                exc.code,
                job.id,
            )
            raise RuntimeError(
                f"ASR 阶段触发了意外的 SystemExit({exc.code!r})，已中止当前任务（详细日志见 {current_process_log_hint()}）"
            ) from exc
        finally:
            _hf_progress.set_current_job_id("", "")


def _run_translation_and_write(
    job: JobState,
    asr_artifacts,
    cancel_event: threading.Event,
    run_id: str,
) -> list[str]:
    video_name = os.path.basename(job.spec.video_paths[0]) if job.spec.video_paths else ""
    with events.scoped_run_identity(job.id, run_id, video=video_name):
        try:
            return run_translation_and_write(
                job.spec.video_paths[0],
                asr_artifacts,
                ctx=_job_context(job),
                job_id=job.id,
                run_id=run_id,
                cancel_event=cancel_event,
            )
        except SystemExit as exc:
            log.exception(
                "Translation worker thread hit unexpected SystemExit(%r) for job %s",
                exc.code,
                job.id,
            )
            raise RuntimeError(
                f"翻译/写入阶段触发了意外的 SystemExit({exc.code!r})，已中止当前任务（详细日志见 {current_process_log_hint()}）"
            ) from exc


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
        log.warning(
            "Failed to load translation retry artifacts snapshot for job %s; falling back to full ASR",
            job.id,
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Workers & GPU status facade
# ---------------------------------------------------------------------------


def gpu_cleanup_status() -> dict[str, Any]:
    return workers.gpu_cleanup_status(_self_module())


def gpu_admission_blocked() -> bool:
    return workers.gpu_admission_blocked(_self_module())


async def retry_gpu_cleanup() -> dict[str, Any]:
    return await workers.retry_gpu_cleanup(_self_module())


async def _worker_set_job(*args, **kwargs) -> JobState:
    try:
        return await _set_job(*args, **kwargs)
    except JobPersistenceError:
        log.exception("Job result is waiting for a persistence retry")
        return _jobs.get(args[0].id, args[0])


async def _wait_for_gpu_release(
    job: JobState,
    cancel_event: threading.Event,
    run_id: str,
) -> bool:
    return await workers.wait_for_gpu_release(_self_module(), job, cancel_event, run_id)


async def gpu_worker() -> None:
    await workers.gpu_worker(_self_module())


async def translation_worker() -> None:
    await workers.translation_worker(_self_module())


async def _eviction_loop() -> None:
    await workers.eviction_loop(_self_module())


async def start_workers(translation_workers: int | None = None) -> list[asyncio.Task]:
    return await workers.start_workers(_self_module(), translation_workers=translation_workers)


def _self_module():
    import sys

    return sys.modules[__name__]


# ---------------------------------------------------------------------------
# Job CRUD & lifecycle API
# ---------------------------------------------------------------------------

_active_retry_recoveries = 0


async def run_under_update_gate(action: Callable[..., Any]) -> Any:
    """Serialize entering self-update with ``create_job`` and ``retry_job``."""
    async with _mutation_gate():
        async with _state_lock:
            active_count = (
                sum(1 for job in _jobs.values() if job.status in _ACTIVE_STATUSES)
                + _active_retry_recoveries
            )
        return await asyncio.to_thread(action, active_jobs_count=active_count)


async def create_job(spec: JobSpec) -> list[JobState]:
    parent_spec = _snapshot_translation_settings(spec)
    async with _mutation_gate():
        if updater.is_update_in_progress():
            raise UpdateInProgressError()
        async with _state_lock:
            existing_jobs = sorted(
                (job.model_copy(deep=True) for job in _jobs.values()),
                key=lambda job: job.output_generation,
                reverse=True,
            )
        selected: dict[Path, JobState] = {}
        created: dict[Path, JobState] = {}
        results: list[JobState] = []
        for video_path in spec.video_paths:
            child_spec = parent_spec.model_copy(update={"video_paths": [video_path]})
            candidate = JobState(
                id=uuid.uuid4().hex,
                spec=child_spec,
                created_at=_utc_now(),
                run_id=_new_run_id(),
                status="queued",
            )
            target = await asyncio.to_thread(_job_output_target, candidate)
            existing = selected.get(target)
            if existing is None:
                existing = await asyncio.to_thread(_reserving_job, target, existing_jobs)
            if existing is not None:
                if not await asyncio.to_thread(_same_submission, existing.spec, child_spec):
                    raise JobOutputConflict(existing.id if target not in created else None)
            else:
                existing = created[target] = candidate
            selected[target] = existing
            results.append(existing)
        if not created:
            return results

        claims = []
        try:
            for target, job in created.items():
                claims.append(await asyncio.to_thread(_claim_job_output, job, target))
            async with _state_lock:
                snapshot = dict(_jobs)
                for job in created.values():
                    snapshot[job.id] = _publish(job)
            await _commit_jobs(
                snapshot,
                new_cancel_events={job.id: _new_cancel_event() for job in created.values()},
            )
        except BaseException:
            await _rollback_claims(claims)
            raise
        for job in created.values():
            await gpu_queue.put(job)
    return results


async def get_job(job_id: str) -> JobState | None:
    async with _state_lock:
        job = _jobs.get(job_id)
        return job.model_copy(deep=True) if job is not None else None


async def list_jobs(status: str | None = None) -> list[JobState]:
    jobs, _revision = await jobs_snapshot(status=status)
    return jobs


async def jobs_snapshot(status: str | None = None) -> tuple[list[JobState], int]:
    async with _state_lock:
        jobs = [job.model_copy(deep=True) for job in _jobs.values()]
        # Reserved revisions may still be awaiting disk confirmation. Dating a
        # visible list with one would make the client reject the later commit
        # at that same revision, leaving saving/deleted cards stuck indefinitely.
        revision = state_clock.visible_revision(jobs, _tombstones)
    if status:
        jobs = [job for job in jobs if job.status == status]
    return sorted(jobs, key=lambda item: item.created_at), revision


async def cancel_job(job_id: str) -> bool:
    # Check terminal/saving status at the very start of _state_lock so cancelling
    # never revokes an already-committed or uncommitted-delivery output lease.
    async with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if job.status in _FINISHED_STATUSES | {"saving", "save_failed", "export_failed"}:
            if job.status == "saving":
                event = _cancel_events.setdefault(job_id, _new_cancel_event())
                event.set()
            return True
        event = _cancel_events.setdefault(job_id, _new_cancel_event())
        event.set()
        job_snapshot = job.model_copy(deep=True)

    try:
        lease = await asyncio.to_thread(output_lease_for_job, job_snapshot)
    except artifact_store.OutputIntegrityError:
        log.warning(
            "Ignoring corrupt output identity while cancelling job %s",
            job_id,
            exc_info=True,
        )
        lease = None

    if lease is not None:
        try:
            await asyncio.to_thread(artifact_store.revoke_output, lease)
        except (OSError, TimeoutError) as exc:
            log.exception("Failed to revoke output ownership while cancelling job %s", job_id)
            raise JobPersistenceError("取消任务时更新输出状态失败；请重试。") from exc

    async with _mutation_gate():
        async with _state_lock:
            job = _jobs.get(job_id)
            if job is None:
                return False
            if job.status in _FINISHED_STATUSES | {"saving", "save_failed", "export_failed"}:
                return True
            status = "cancelled" if job.status == "queued" else "cancelling"
            candidate = _publish(job.model_copy(update={"status": status, "current_stage": status}))
            snapshot = {**_jobs, job_id: candidate}
        await _commit_jobs(
            snapshot,
            discard_cancel_ids=(job_id,) if status in _FINISHED_STATUSES else (),
        )
    return True


async def retry_job(job_id: str) -> JobState | None:
    global _active_retry_recoveries
    async with _mutation_gate():
        if updater.is_update_in_progress():
            raise UpdateInProgressError()
        async with _state_lock:
            current = _jobs.get(job_id)
            if current is None:
                return None
            existing = current.model_copy(deep=True)
            _active_retry_recoveries += 1
    try:
        recovered = None
        if existing.status == "save_failed" and existing.pending_status:
            recovered = await _set_job(
                existing,
                status=existing.pending_status,
                expected_run_id=existing.run_id,
            )
        elif existing.status in _RETRYABLE_STATUSES | _DELIVERY_FAILURES:
            lease = await asyncio.to_thread(output_lease_for_job, existing)
            committed = await asyncio.to_thread(artifact_store.resume_output, lease) if lease else None
            if committed is not None:
                recovered = await _set_job(
                    existing,
                    status="done",
                    current_stage="done",
                    artifacts=_relative_artifacts(
                        artifact_store.manifest_paths(committed, project_root=PROJECT_ROOT),
                        existing.spec.output_dir,
                    ),
                    subtitle_cue_count=committed.get("subtitle_cue_count"),
                    expected_run_id=existing.run_id,
                )
            elif existing.status in _DELIVERY_FAILURES:
                raise artifact_store.OutputIntegrityError()
        if recovered is not None:
            if recovered.status == "done" and recovered.run_id == existing.run_id:
                await asyncio.to_thread(_remove_job_temp_dir, job_id)
            return recovered

        async with _mutation_gate():
            if updater.is_update_in_progress():
                raise UpdateInProgressError()
            async with _state_lock:
                job = _jobs.get(job_id)
                if job is None or job.status not in _RETRYABLE_STATUSES:
                    return None
                job_copy = job.model_copy(deep=True)
                retried = _publish(
                    job.model_copy(
                        deep=True,
                        update={
                            "spec": _refresh_translation_settings(job.spec),
                            "run_id": _new_run_id(),
                            "status": "queued",
                            "current_stage": None,
                            "progress": {},
                            "artifacts": [],
                            "error": None,
                            "subtitle_cue_count": None,
                            "pending_status": None,
                            "storage_error": None,
                        },
                    )
                )
                existing_jobs = sorted(
                    (candidate.model_copy(deep=True) for candidate in _jobs.values()),
                    key=lambda candidate: candidate.output_generation,
                    reverse=True,
                )
                snapshot = {**_jobs, job_id: retried}
            target = await asyncio.to_thread(_job_output_target, retried)
            conflict = await asyncio.to_thread(_reserving_job, target, existing_jobs)
            if conflict is not None:
                raise JobOutputConflict(conflict.id)
            retry_artifacts = await asyncio.to_thread(_load_translation_retry_artifacts, job_copy)
            claim = await asyncio.to_thread(_claim_job_output, retried, target)
            try:
                await _commit_jobs(
                    snapshot,
                    new_cancel_events={job_id: _new_cancel_event()},
                )
            except BaseException:
                await _rollback_claims([claim])
                raise
            if retry_artifacts is not None:
                await trans_queue.put((retried, retry_artifacts))
            else:
                await gpu_queue.put(retried)
        return retried
    finally:
        async with _state_lock:
            _active_retry_recoveries = max(0, _active_retry_recoveries - 1)


async def _remove_matching(predicate) -> list[JobState]:
    async with _mutation_gate():
        async with _state_lock:
            removed = [job for job in _jobs.values() if predicate(job)]
            if not removed:
                return []
            ids = tuple(job.id for job in removed)
            snapshot = {key: value for key, value in _jobs.items() if key not in ids}
            revision = _next_revision()
        await _commit_jobs(snapshot, deletions=ids, revision=revision)
    for job in removed:
        await asyncio.to_thread(_remove_job_temp_dir, job.id)
    return removed


async def remove_job(job_id: str) -> bool:
    removed = await _remove_matching(
        lambda job: job.id == job_id and job.status in _FINISHED_STATUSES
    )
    if removed:
        return True
    async with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if job.status in {"saving", "save_failed"}:
            raise JobStateConflictError(
                "任务正在确认保存或存在未保存的完成状态，请先重试保存后再删除。"
            )
    return await cancel_job(job_id)


async def remove_finished_jobs() -> int:
    return len(await _remove_matching(lambda job: job.status in _FINISHED_STATUSES))


async def evict_old_jobs(max_age_hours: int = 48) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat(
        timespec="milliseconds"
    )
    return len(
        await _remove_matching(
            lambda job: job.status in _FINISHED_STATUSES and job.created_at < cutoff
        )
    )


# ---------------------------------------------------------------------------
# Fast in-memory progress updates + per-job coalesced throttled persistence
# ---------------------------------------------------------------------------


def apply_job_progress_memory(
    job_id: str,
    progress: dict[str, Any],
    *,
    run_id: str = "",
) -> tuple[bool, bool]:
    """Synchronously apply progress & stamp revision in memory before SSE fan-out.

    Returns ``(updated, stage_changed)``.
    """
    job = _jobs.get(job_id)
    if job is None or job.status in _FINISHED_STATUSES | {"saving", "save_failed", "export_failed"}:
        return False, False
    if run_id and job.run_id and run_id != job.run_id:
        return False, False
    prev_status = job.status
    prev_stage = job.current_stage
    job.progress = dict(progress)
    stage = progress.get("stage")
    cancel_pending = job.status == "cancelling"
    if isinstance(stage, str) and stage and not cancel_pending:
        job.current_stage = stage
        mapped = _EVENT_STAGE_STATUS.get(stage)
        if mapped and job.status not in _FINISHED_STATUSES:
            job.status = mapped  # type: ignore[assignment]
    _publish(job)
    stage_changed = job.status != prev_status or job.current_stage != prev_stage
    return True, stage_changed


async def _flush_progress_persist(job_id: str, *, delay_s: float) -> None:
    try:
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        async with _mutation_gate():
            async with _state_lock:
                if job_id not in _jobs:
                    return
                persist_snapshot = list(_jobs.values())
            _progress_last_persist_at[job_id] = _time.monotonic()
            _persist(persist_snapshot, confirm=False)
    except asyncio.CancelledError:
        raise
    finally:
        current = asyncio.current_task()
        if _progress_persist_tasks.get(job_id) is current:
            _progress_persist_tasks.pop(job_id, None)


def schedule_progress_persist(
    job_id: str,
    *,
    immediate: bool = False,
) -> asyncio.Task[None] | None:
    """Coalesce and throttle progress persistence per ``job_id``."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    existing = _progress_persist_tasks.get(job_id)
    if immediate:
        if existing is not None and not existing.done():
            existing.cancel()
        task = loop.create_task(
            _flush_progress_persist(job_id, delay_s=0.0),
            name=f"progress-persist-{job_id}",
        )
        _progress_persist_tasks[job_id] = task
        return task
    if existing is not None and not existing.done():
        return existing
    last = _progress_last_persist_at.get(job_id)
    delay_s = (
        0.0
        if last is None
        else max(0.0, _PROGRESS_PERSIST_INTERVAL_S - (_time.monotonic() - last))
    )
    task = loop.create_task(
        _flush_progress_persist(job_id, delay_s=delay_s),
        name=f"progress-persist-{job_id}",
    )
    _progress_persist_tasks[job_id] = task
    return task


async def cancel_pending_progress_tasks() -> None:
    tasks = list(_progress_persist_tasks.values())
    _progress_persist_tasks.clear()
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def update_job_progress(
    job_id: str,
    progress: dict[str, Any],
    *,
    run_id: str = "",
) -> None:
    async with _mutation_gate():
        persist_snapshot: list[JobState] | None = None
        async with _state_lock:
            updated, stage_changed = apply_job_progress_memory(
                job_id,
                progress,
                run_id=run_id,
            )
            if not updated:
                return
            if stage_changed:
                _cancel_progress_persist_for(job_id)
                persist_snapshot = list(_jobs.values())
        if persist_snapshot is not None:
            _progress_last_persist_at[job_id] = _time.monotonic()
            _persist(persist_snapshot, confirm=False)
        else:
            schedule_progress_persist(job_id, immediate=False)


def _drain_trans_queue_loggers() -> None:
    """Close any open ``.run.log`` handlers sitting in ``trans_queue`` on shutdown."""
    while True:
        try:
            _job, asr_artifacts = trans_queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            workers.close_asr_artifacts_logger(asr_artifacts, pipeline_main)
        finally:
            trans_queue.task_done()


async def shutdown_executor() -> None:
    global _executor
    signal_all_cancel_events()
    await cancel_pending_progress_tasks()
    _drain_trans_queue_loggers()
    try:
        await asyncio.to_thread(_job_store_writer.flush)
    except JobPersistenceError:
        log.exception("Unconfirmed job state remains at shutdown")
    with _executor_lock:
        executor = _executor
        _executor = None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
