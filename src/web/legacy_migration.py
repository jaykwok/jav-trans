"""One-shot upgrade migration from 1.8.x ``tmp/outputs`` & ``tmp/web`` layouts."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from core.atomic_io import write_json_atomic
from pipeline import artifact_store
from pipeline.ids import sanitize_job_id
from web.job_store import JobPersistenceError
from web.models import JobState

log = logging.getLogger(__name__)

LEGACY_CLEANUP_FILE_NAME = "legacy-cleanup.json"
LEGACY_INTEGRITY_REVIEW_FILE_NAME = "legacy-integrity-review.json"
_RETIRED_SPEC_FIELDS = ("keep_quality_report", "keep_temp_files", "advanced")


def strip_retired_spec_fields(item: dict[str, Any]) -> dict[str, Any]:
    spec_data = item.get("spec")
    if isinstance(spec_data, dict):
        if any(k in spec_data for k in _RETIRED_SPEC_FIELDS):
            return {
                **item,
                "spec": {k: v for k, v in spec_data.items() if k not in _RETIRED_SPEC_FIELDS},
            }
    return item


def rewrite_legacy_job_record(item: dict[str, Any]) -> dict[str, Any]:
    updated = strip_retired_spec_fields(item)
    artifacts = updated.get("artifacts")
    if isinstance(artifacts, list):
        rewritten: list[Any] = []
        changed = False
        for entry in artifacts:
            if isinstance(entry, str):
                normalized = entry.replace("\\", "/")
                if normalized.lower().startswith("tmp/outputs/"):
                    rewritten.append("outputs/" + normalized[len("tmp/outputs/") :])
                    changed = True
                    continue
            rewritten.append(entry)
        if changed:
            updated = {**updated, "artifacts": rewritten}
    return updated


def read_pending_legacy_cleanup_ids(path: Path) -> list[str] | None:
    """Read pending legacy cleanup IDs, returning ``None`` on read/parse error.

    Returning ``None`` (instead of ``[]``) guarantees that a transient lock or
    read error never tricks cleanup into deleting the pending list.
    """
    if not path.exists():
        return []
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        log.warning(
            "Could not read pending legacy cleanup list at %s; preserving file for next startup",
            path,
            exc_info=True,
        )
        return None
    except UnicodeDecodeError:
        log.warning(
            "Pending legacy cleanup list at %s is not valid UTF-8; preserving file untouched",
            path,
            exc_info=True,
        )
        return None
    try:
        payload = json.loads(raw_text)
    except ValueError:
        log.warning(
            "Pending legacy cleanup list at %s is malformed JSON; preserving file untouched",
            path,
            exc_info=True,
        )
        return None
    if not isinstance(payload, list):
        log.warning(
            "Pending legacy cleanup list at %s is not a JSON list; preserving file untouched",
            path,
        )
        return None
    result: list[str] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, str) or not item.strip():
            continue
        safe = sanitize_job_id(item)
        if safe not in seen:
            seen.add(safe)
            result.append(safe)
    return result


def write_pending_legacy_cleanup_ids(path: Path, job_ids: list[str]) -> None:
    write_json_atomic(path, job_ids, indent=2, ensure_ascii=False)


def drain_pending_legacy_temp_dirs(
    extra_job_ids: list[str] | None = None,
    *,
    project_root: Path,
    jobs_path: Path,
    rmtree_fn: Callable[[Any], None] = shutil.rmtree,
) -> None:
    """Retry removing dropped legacy ``tmp/web/jobs/<id>`` directories and
    post-commit leftover ``tmp/web/jobs.json`` until cleanly gone."""
    pending_path = jobs_path.with_name(LEGACY_CLEANUP_FILE_NAME)
    existing = read_pending_legacy_cleanup_ids(pending_path)
    legacy_jobs_temp = project_root / "tmp" / "web" / "jobs"
    if existing is None:
        # Preserve pending_path on disk so the next startup retries cleanly.
        for raw_id in extra_job_ids or []:
            candidate = legacy_jobs_temp / sanitize_job_id(raw_id)
            if candidate.exists():
                try:
                    rmtree_fn(candidate)
                except OSError:
                    pass
        return

    merged: list[str] = []
    seen: set[str] = set()
    for raw_id in [*existing, *(extra_job_ids or [])]:
        safe = sanitize_job_id(raw_id)
        if safe not in seen:
            seen.add(safe)
            merged.append(safe)

    # If the new index was already committed (e.g. crash between writing
    # cache/web/jobs.json and unlinking tmp/web/jobs.json), finish removing
    # the migrated legacy index here.
    legacy_jobs_path = project_root / "tmp" / "web" / "jobs.json"
    legacy_index_still_present = False
    if jobs_path.is_file() and legacy_jobs_path.is_file():
        try:
            legacy_jobs_path.unlink(missing_ok=True)
        except OSError:
            log.warning(
                "Could not remove migrated legacy jobs.json at %s; will retry on next startup",
                legacy_jobs_path,
            )
        legacy_index_still_present = legacy_jobs_path.exists()

    remaining: list[str] = []
    for safe_id in merged:
        candidate = legacy_jobs_temp / safe_id
        if not candidate.exists():
            continue
        try:
            rmtree_fn(candidate)
        except OSError:
            log.warning("Could not remove legacy job temp dir %s; will retry on next startup", candidate)
        if candidate.exists():
            remaining.append(safe_id)

    if remaining or legacy_index_still_present:
        try:
            write_pending_legacy_cleanup_ids(pending_path, remaining)
        except OSError:
            log.warning("Could not persist pending legacy cleanup list at %s", pending_path)
    elif pending_path.exists():
        try:
            pending_path.unlink(missing_ok=True)
        except OSError:
            pass

    if legacy_jobs_temp.is_dir():
        try:
            if not any(legacy_jobs_temp.iterdir()):
                legacy_jobs_temp.rmdir()
        except OSError:
            pass
    legacy_web_dir = project_root / "tmp" / "web"
    try:
        if legacy_web_dir.is_dir() and not any(legacy_web_dir.iterdir()):
            legacy_web_dir.rmdir()
    except OSError:
        pass


def inspect_legacy_job_outputs(
    job: JobState,
    *,
    project_root: Path,
    committed_outputs_fn: Callable[[JobState], tuple[artifact_store.OutputLease | None, dict[str, Any] | None]],
    utc_now_fn: Callable[[], str],
) -> tuple[bool, dict[str, Any] | None]:
    """Check whether a legacy job has an actual verified durable generation.

    Returns ``(has_recoverable_output, integrity_issue_record)``.
    """
    try:
        lease, committed = committed_outputs_fn(job)
        prepared = (
            artifact_store.read_prepared_manifest(lease)
            if lease is not None and committed is None
            else None
        )
        return (committed is not None or prepared is not None), None
    except artifact_store.OutputIntegrityError as exc:
        # Distinguish transient I/O locks from confirmed missing/corrupt data:
        # pause migration and keep the legacy index so the next launch retries.
        cause = exc.__cause__
        if isinstance(cause, OSError) and not isinstance(cause, FileNotFoundError):
            raise JobPersistenceError(
                "检查旧版任务产物时文件暂时无法读取；请关闭占用文件后重启，旧版任务记录已保留。"
            ) from exc
        has_generations = False
        store_path = ""
        if job.output_target:
            try:
                store_root = artifact_store.output_store_root(
                    job.output_target,
                    project_root=project_root,
                )
                store_path = str(store_root)
                generations_dir = store_root / "generations"
                has_generations = generations_dir.is_dir() and any(generations_dir.iterdir())
            except OSError:
                has_generations = False
        return False, {
            "status": "integrity_review_required",
            "job_id": job.id,
            "legacy_status": job.status,
            "run_id": job.run_id,
            "output_target": job.output_target,
            "output_generation": job.output_generation,
            "store_path": store_path,
            "has_generations": has_generations,
            "error": str(exc),
            "recorded_at": utc_now_fn(),
            "record": job.model_dump(),
        }


async def migrate_legacy_web_state(
    *,
    project_root: Path,
    jobs_path: Path,
    epoch_path: Path,
    epoch_file_name: str,
    committed_outputs_fn: Callable[[JobState], tuple[artifact_store.OutputLease | None, dict[str, Any] | None]],
    write_jobs_snapshot_fn: Callable[..., None],
    drain_fn: Callable[..., None],
    utc_now_fn: Callable[[], str],
) -> None:
    """One-shot upgrade migration from 1.8.x ``tmp/outputs`` & ``tmp/web``.

    Moves committed output generations into top-level ``outputs/``, migrates
    completed or export/publish-recoverable jobs into ``cache/web/jobs.json``,
    and removes unfinished/uncommitted legacy jobs so upgrades start clean.
    """
    default_jobs_path = (project_root / "cache" / "web" / "jobs.json").resolve()
    try:
        using_default_layout = jobs_path.resolve() == default_jobs_path
    except OSError:
        using_default_layout = False
    if not using_default_layout:
        return

    legacy_outputs = project_root / "tmp" / "outputs"
    legacy_jobs_path = project_root / "tmp" / "web" / "jobs.json"
    new_outputs = project_root / "outputs"
    has_legacy_upgrade = legacy_outputs.is_dir() or (
        legacy_jobs_path.is_file() and not jobs_path.exists()
    )
    if legacy_outputs.is_dir() or new_outputs.is_dir():
        try:
            migrated_outputs = await asyncio.to_thread(
                artifact_store.migrate_legacy_outputs,
                project_root,
            )
            if migrated_outputs:
                log.info(
                    "Migrated %d legacy output store(s) from tmp/outputs/ to outputs/",
                    migrated_outputs,
                )
        except OSError as exc:
            log.exception("Failed to migrate legacy outputs from tmp/outputs/")
            if has_legacy_upgrade:
                raise JobPersistenceError(
                    "无法迁移旧版产物目录 tmp/outputs/；请关闭占用文件后重启，旧版任务记录已保留。"
                ) from exc

    legacy_epoch = project_root / "tmp" / "web" / epoch_file_name
    if legacy_epoch.is_file() and not epoch_path.exists():
        try:
            epoch_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(legacy_epoch), str(epoch_path))
        except OSError:
            log.warning("Could not migrate legacy %s", epoch_file_name, exc_info=True)

    # Always drain any leftover dropped-job temp dirs (or leftover tmp/web/jobs.json
    # after a post-commit crash) from an earlier migration.
    pending_path = jobs_path.with_name(LEGACY_CLEANUP_FILE_NAME)
    if pending_path.exists():
        await asyncio.to_thread(drain_fn)

    if jobs_path.exists() or not legacy_jobs_path.is_file():
        return

    try:
        payload = json.loads(await asyncio.to_thread(legacy_jobs_path.read_text, encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise JobPersistenceError(
            "无法读取旧版任务记录 tmp/web/jobs.json；请检查或移除该文件后重启。"
        ) from exc
    if not isinstance(payload, list):
        raise JobPersistenceError(
            "旧版任务记录 tmp/web/jobs.json 格式无效；请检查或移除该文件后重启。"
        )

    kept: list[JobState] = []
    dropped_ids: list[str] = []
    integrity_review: list[dict[str, Any]] = []
    rejected = 0
    for raw_item in payload:
        if not isinstance(raw_item, dict):
            rejected += 1
            continue
        try:
            job = JobState.model_validate(rewrite_legacy_job_record(raw_item))
        except Exception as exc:
            rejected += 1
            log.warning("Skipping unreadable legacy job record during upgrade: %s", exc)
            continue

        has_durable_output, integrity_issue = await asyncio.to_thread(
            inspect_legacy_job_outputs,
            job,
            project_root=project_root,
            committed_outputs_fn=committed_outputs_fn,
            utc_now_fn=utc_now_fn,
        )
        if integrity_issue is not None:
            integrity_review.append(integrity_issue)
            log.warning(
                "Legacy job %s (status=%s, has_generations=%s) failed output integrity check "
                "during upgrade; recorded for integrity review instead of treating as retryable",
                job.id,
                job.status,
                integrity_issue["has_generations"],
            )

        if has_durable_output or job.status == "done":
            kept.append(job)
        else:
            dropped_ids.append(job.id)

    if rejected:
        backup = jobs_path.with_name(
            f"{jobs_path.name}.legacy-rejected-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_bytes(legacy_jobs_path.read_bytes())
        except OSError as exc:
            raise JobPersistenceError("无法备份旧版不兼容的任务记录；已停止迁移，原文件未覆盖。") from exc

    if integrity_review:
        review_path = jobs_path.with_name(LEGACY_INTEGRITY_REVIEW_FILE_NAME)
        try:
            await asyncio.to_thread(
                write_json_atomic,
                review_path,
                integrity_review,
                indent=2,
                ensure_ascii=False,
            )
        except OSError as exc:
            raise JobPersistenceError("无法保存旧版产物完整性待核查记录；已停止迁移，原文件未覆盖。") from exc

    legacy_jobs_temp = project_root / "tmp" / "web" / "jobs"
    pending_cleanup_ids = [
        sanitize_job_id(job_id)
        for job_id in dropped_ids
        if (legacy_jobs_temp / sanitize_job_id(job_id)).exists()
    ]
    # Persist the cleanup marker BEFORE publishing cache/web/jobs.json so a crash
    # right after writing the new index still removes tmp/web/jobs.json and any
    # dropped WAV directories on the next startup.
    try:
        existing_pending = await asyncio.to_thread(read_pending_legacy_cleanup_ids, pending_path)
        if existing_pending is None:
            raise JobPersistenceError(
                "无法读取待清理记录 cache/web/legacy-cleanup.json；已暂停迁移以防丢失清理任务。"
            )
        merged_pending = list(dict.fromkeys([*existing_pending, *pending_cleanup_ids]))
        await asyncio.to_thread(write_pending_legacy_cleanup_ids, pending_path, merged_pending)
    except OSError as exc:
        raise JobPersistenceError(
            "无法写入待清理记录 cache/web/legacy-cleanup.json；已暂停迁移，旧版任务记录已保留。"
        ) from exc

    await asyncio.to_thread(write_jobs_snapshot_fn, kept, path=jobs_path)
    await asyncio.to_thread(drain_fn, pending_cleanup_ids)
    log.info(
        "Migrated %d completed/recoverable legacy job(s) to cache/web/jobs.json "
        "and cleared %d unfinished legacy job(s)",
        len(kept),
        len(dropped_ids),
    )
