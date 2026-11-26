from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _log_cleanup_interrupted(path: Path, exc: BaseException) -> None:
    if isinstance(exc, SystemExit):
        logger.warning(
            "Temporary cleanup for %s was interrupted by SystemExit(%r) "
            "(possibly an external sandbox or safe-delete hook); skipping cleanup",
            path,
            exc.code,
        )
        return
    logger.warning("Cleanup failed for %s: %s", path, exc)


def unlink_for_cleanup(path: Path) -> None:
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except (Exception, SystemExit) as exc:
        _log_cleanup_interrupted(path, exc)


def cleanup_translation_cache(cache_path: str = "") -> None:
    raw_path = cache_path or os.getenv("TRANSLATION_CACHE_PATH", "").strip()
    if not raw_path:
        return

    raw_cache_path = Path(raw_path)
    cache_paths = [raw_cache_path]
    if raw_cache_path.suffix.lower() == ".json":
        cache_paths.append(raw_cache_path.with_suffix(".jsonl"))
    elif raw_cache_path.suffix.lower() == ".jsonl":
        cache_paths.append(raw_cache_path.with_suffix(".json"))
    jsonl_cache_path = (
        raw_cache_path.with_suffix(".jsonl")
        if raw_cache_path.suffix.lower() == ".json"
        else raw_cache_path
    )
    cache_paths.append(jsonl_cache_path.with_name(f"{jsonl_cache_path.stem}.memory.jsonl"))

    for cache_path_item in cache_paths:
        unlink_for_cleanup(cache_path_item)
        for tmp_path in cache_path_item.parent.glob(f"{cache_path_item.name}.*.tmp"):
            unlink_for_cleanup(tmp_path)
        for glossary_path in cache_path_item.parent.glob("translation_global_glossary*.json"):
            unlink_for_cleanup(glossary_path)


def cleanup_job_temp(
    job_temp_dir: str,
    translation_cache_path: str = "",
) -> None:
    root = Path(job_temp_dir)
    cleanup_translation_cache(translation_cache_path)

    if not root.exists() or not root.is_dir():
        return

    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_file():
                path.unlink()
        except (Exception, SystemExit) as exc:
            _log_cleanup_interrupted(path, exc)

    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except (Exception, SystemExit) as exc:
            _log_cleanup_interrupted(path, exc)

    try:
        if root.exists() and not any(root.iterdir()):
            root.rmdir()
    except (Exception, SystemExit) as exc:
        _log_cleanup_interrupted(root, exc)


def cleanup_empty_runtime_dir_tree(root: Path, project_root: Path) -> None:
    try:
        resolved = root.resolve()
        resolved.relative_to(project_root)
    except Exception as exc:
        logger.warning("Cleanup skipped for %s: %s", root, exc)
        return

    if not resolved.exists() or not resolved.is_dir():
        return

    for path in sorted(resolved.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except (Exception, SystemExit) as exc:
            _log_cleanup_interrupted(path, exc)

    try:
        if resolved.exists() and not any(resolved.iterdir()):
            resolved.rmdir()
    except (Exception, SystemExit) as exc:
        _log_cleanup_interrupted(resolved, exc)


def cleanup_runtime_ephemeral_temp(
    *,
    job_temp_root: str | Path,
    asr_chunk_root: str | Path,
    project_root: Path,
) -> None:
    roots = [
        Path(job_temp_root).expanduser(),
        Path(asr_chunk_root).expanduser(),
    ]
    resolved_project_root = project_root.resolve()
    for root in roots:
        if not root.is_absolute():
            root = resolved_project_root / root
        cleanup_empty_runtime_dir_tree(root, resolved_project_root)
