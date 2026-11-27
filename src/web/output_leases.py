"""Output lease claims, collision detection, and job temp/artifact path resolution."""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

from core.job_context import JobContext
from pipeline import artifact_store
from pipeline.ids import sanitize_job_id
from pipeline.output import resolve_jobs_temp_root
from web.job_store import JobPersistenceError
from web.models import JobSpec, JobState

log = logging.getLogger(__name__)


class JobOutputConflict(RuntimeError):
    def __init__(self, job_id: str | None = None) -> None:
        super().__init__(
            "此字幕输出路径已有任务正在处理或等待保存。请选择其他输出目录，或处理现有任务后再提交。"
            if job_id
            else "同一批次的不同视频会写入同一个字幕文件。请分开提交并选择不同的输出目录。"
        )
        self.job_id = job_id


def jobs_temp_root(project_root: Path) -> Path:
    """Resolve the job scratch root, honouring ``JOB_TEMP_DIR`` when set."""
    return resolve_jobs_temp_root(project_root)


def job_temp_dir(job_id: str, *, jobs_temp_root_fn: Callable[[], Path]) -> str:
    return str((jobs_temp_root_fn() / sanitize_job_id(job_id)).resolve())


def remove_job_temp_dir(
    job_id: str,
    *,
    jobs_temp_root_fn: Callable[[], Path],
) -> None:
    safe_id = sanitize_job_id(job_id)
    try:
        root = jobs_temp_root_fn().resolve()
        path = root / safe_id
        if path.resolve() == path and path.is_dir():
            shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except (OSError, RuntimeError) as exc:
        log.warning("Could not remove temporary files for job %s: %s", safe_id, type(exc).__name__)


def translation_cache_path(
    job_id: str,
    *,
    job_temp_dir_fn: Callable[[str], str],
) -> str:
    return str(Path(job_temp_dir_fn(job_id)) / "translation_cache.jsonl")


def job_context(
    job: JobState,
    *,
    job_temp_dir_fn: Callable[[str], str],
    translation_cache_path_fn: Callable[[str], str],
) -> JobContext:
    ctx = JobContext.from_spec(
        job.spec,
        job.id,
        job_temp_dir_fn(job.id),
        translation_cache_path_fn(job.id),
    )
    ctx.output_target = job.output_target
    ctx.output_generation = job.output_generation
    ctx.output_run_id = job.run_id
    return ctx


def output_lease_for_job(
    job: JobState,
    *,
    project_root: Path,
    job_context_fn: Callable[[JobState], JobContext],
    output_target_fn: Callable[[str, JobContext], Path],
) -> artifact_store.OutputLease | None:
    if job.output_generation == 0 and not job.output_target:
        return None
    if job.output_generation <= 0 or not job.output_target or not job.run_id:
        raise artifact_store.OutputIntegrityError()
    try:
        target = output_target_fn(job.spec.video_paths[0], job_context_fn(job))
        if target.resolve() != Path(job.output_target).resolve():
            raise artifact_store.OutputIntegrityError()
    except (OSError, ValueError, IndexError) as exc:
        raise artifact_store.OutputIntegrityError() from exc
    return artifact_store.OutputLease(
        target,
        project_root,
        job.id,
        job.run_id,
        job.output_generation,
    )


def job_output_target(
    job: JobState,
    *,
    job_context_fn: Callable[[JobState], JobContext],
    output_target_fn: Callable[[str, JobContext], Path],
) -> Path:
    return output_target_fn(job.spec.video_paths[0], job_context_fn(job)).resolve()


def claim_job_output(
    job: JobState,
    target: Path,
    *,
    project_root: Path,
) -> artifact_store.OutputLease:
    try:
        lease = artifact_store.claim_output(
            target,
            project_root=project_root,
            job_id=job.id,
            run_id=job.run_id,
        )
    except OSError as exc:
        raise JobPersistenceError("登记输出目标失败；任务尚未启动。") from exc
    job.output_target = str(lease.target)
    job.output_generation = lease.sequence
    return lease


def reserving_job(
    target: Path,
    jobs: list[JobState],
    *,
    active_statuses: Collection[str],
    delivery_failures: Collection[str],
    job_output_target_fn: Callable[[JobState], Path],
    output_lease_fn: Callable[[JobState], artifact_store.OutputLease | None],
) -> JobState | None:
    resolved_target = target.resolve()
    for job in jobs:
        status = job.status
        if status not in active_statuses and status not in delivery_failures:
            continue
        try:
            job_target = (
                Path(job.output_target).resolve()
                if job.output_target
                else job_output_target_fn(job)
            )
        except (OSError, ValueError, IndexError):
            continue
        if job_target != resolved_target:
            continue
        if status in active_statuses:
            return job
        try:
            lease = output_lease_fn(job)
        except artifact_store.OutputIntegrityError:
            return job
        if lease is not None and artifact_store.owns_output(lease):
            return job
    return None


def same_submission(
    first: JobSpec,
    second: JobSpec,
    *,
    project_root: Path,
) -> bool:
    fields = {"video_paths", "output_dir"}
    return (
        artifact_store.artifact_path(str(Path(first.video_paths[0]).expanduser()), project_root)
        == artifact_store.artifact_path(str(Path(second.video_paths[0]).expanduser()), project_root)
        and first.model_dump(exclude=fields) == second.model_dump(exclude=fields)
    )


async def rollback_claims(claims: list[artifact_store.OutputLease]) -> None:
    for lease in reversed(claims):
        try:
            await asyncio.to_thread(artifact_store.rollback_claim, lease)
        except Exception:
            log.exception("Failed to restore output ownership after an unconfirmed job transaction")


def committed_job_outputs(
    job: JobState,
    *,
    verify: bool = True,
    output_lease_fn: Callable[[JobState], artifact_store.OutputLease | None],
) -> tuple[artifact_store.OutputLease | None, dict[str, Any] | None]:
    lease = output_lease_fn(job)
    return lease, artifact_store.read_run_manifest(lease, verify=verify) if lease else None


def retry_requires_computation(
    job: JobState,
    *,
    delivery_failures: Collection[str],
    committed_outputs_fn: Callable[[JobState], tuple[artifact_store.OutputLease | None, dict[str, Any] | None]],
) -> bool:
    if job.status in delivery_failures:
        return False
    lease, committed = committed_outputs_fn(job)
    return committed is None and (
        lease is None or artifact_store.read_prepared_manifest(lease) is None
    )


def relative_artifacts(
    paths: list[str],
    output_dir: str | None,
    *,
    project_root: Path,
) -> list[str]:
    base = Path(output_dir).expanduser() if output_dir else project_root
    if not base.is_absolute():
        base = project_root / base
    base = base.resolve()
    result: list[str] = []
    for raw_path in paths:
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = project_root / path
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if not resolved.is_file():
            continue
        try:
            result.append(resolved.relative_to(base).as_posix())
        except ValueError:
            try:
                result.append(resolved.relative_to(project_root).as_posix())
            except ValueError:
                result.append(str(resolved))
    return result
