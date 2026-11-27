from __future__ import annotations

import logging
import os
import time
from collections.abc import Collection
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_EMPTY_DIR_MIN_AGE_S = 60.0


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


def cleanup_translation_cache(
    cache_path: str | None = None,
    *,
    job_temp_dir: str | Path | None = None,
) -> None:
    """Remove translation cache files for a finished job.

    Distinguishes ``cache_path=None`` (falls back to ``TRANSLATION_CACHE_PATH``)
    from an explicit empty string ``""`` (no-op, never falls back to env).
    Global glossary files are cleaned once outside the candidate loop and only
    when they live inside ``job_temp_dir`` (the job's own ephemeral directory).
    """
    if cache_path is None:
        raw_path = os.getenv("TRANSLATION_CACHE_PATH", "").strip()
    else:
        raw_path = str(cache_path).strip()
    if not raw_path:
        return

    raw_cache_path = Path(raw_path)
    candidates: list[Path] = [raw_cache_path]
    if raw_cache_path.suffix.lower() == ".json":
        candidates.append(raw_cache_path.with_suffix(".jsonl"))
    elif raw_cache_path.suffix.lower() == ".jsonl":
        candidates.append(raw_cache_path.with_suffix(".json"))
    jsonl_cache_path = (
        raw_cache_path.with_suffix(".jsonl")
        if raw_cache_path.suffix.lower() == ".json"
        else raw_cache_path
    )
    candidates.append(jsonl_cache_path.with_name(f"{jsonl_cache_path.stem}.memory.jsonl"))

    seen: set[Path] = set()
    for cache_path_item in candidates:
        if cache_path_item in seen:
            continue
        seen.add(cache_path_item)
        unlink_for_cleanup(cache_path_item)
        try:
            tmp_matches = list(cache_path_item.parent.glob(f"{cache_path_item.name}.*.tmp"))
        except OSError:
            tmp_matches = []
        for tmp_path in tmp_matches:
            unlink_for_cleanup(tmp_path)

    if job_temp_dir is not None:
        try:
            job_root = Path(job_temp_dir).resolve()
            cache_parent = raw_cache_path.parent.resolve()
            inside_job_dir = cache_parent == job_root or cache_parent.is_relative_to(job_root)
        except OSError:
            inside_job_dir = False
        if inside_job_dir and cache_parent.is_dir():
            try:
                glossary_matches = list(cache_parent.glob("translation_global_glossary*.json"))
            except OSError:
                glossary_matches = []
            for glossary_path in glossary_matches:
                unlink_for_cleanup(glossary_path)


def cleanup_job_temp(
    job_temp_dir: str,
    translation_cache_path: str | None = None,
) -> None:
    root = Path(job_temp_dir)
    cleanup_translation_cache(translation_cache_path, job_temp_dir=root)

    if not root.exists() or not root.is_dir():
        return

    for current_root, _dirnames, filenames in os.walk(root, topdown=False):
        current_dir = Path(current_root)
        for filename in filenames:
            file_path = current_dir / filename
            try:
                file_path.unlink()
            except FileNotFoundError:
                pass
            except (Exception, SystemExit) as exc:
                _log_cleanup_interrupted(file_path, exc)
        try:
            current_dir.rmdir()
        except FileNotFoundError:
            pass
        except OSError as exc:
            try:
                if current_dir.exists() and any(current_dir.iterdir()):
                    continue
            except OSError:
                continue
            _log_cleanup_interrupted(current_dir, exc)
        except SystemExit as exc:
            _log_cleanup_interrupted(current_dir, exc)


def _is_dir_excluded(path: Path, root: Path, excluded: set[str]) -> bool:
    if not excluded:
        return False
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part in excluded for part in rel_parts)


def cleanup_empty_runtime_dir_tree(
    root: Path,
    project_root: Path,
    *,
    min_age_s: float = _DEFAULT_EMPTY_DIR_MIN_AGE_S,
    exclude_names: Collection[str] | None = None,
) -> None:
    """Prune stale empty directories under ``root`` without racing active jobs.

    Empty directories modified more recently than ``min_age_s`` (default 60s)
    or matching ``exclude_names`` are skipped so a concurrent job that just
    created ``tmp/jobs/<other_id>/audio/`` or ``tmp/chunks/<other>/`` before
    writing its first file never loses its directory out from under it.
    """
    try:
        resolved_project = project_root.resolve()
        resolved = root.resolve()
        resolved.relative_to(resolved_project)
    except Exception as exc:
        logger.warning("Cleanup skipped for %s: %s", root, exc)
        return

    if not resolved.exists() or not resolved.is_dir():
        return

    min_age = max(0.0, float(min_age_s))
    excluded = {str(name) for name in (exclude_names or ()) if str(name)}
    now = time.time()

    # Snapshot directory mtimes before any rmdir runs so removing a stale leaf
    # does not artificially refresh its parent's mtime and block pruning it in
    # the same pass.
    dir_mtimes: dict[Path, float] = {}
    walk_dirs: list[Path] = []
    for current_root, _dirnames, _filenames in os.walk(resolved, topdown=False):
        current_dir = Path(current_root)
        walk_dirs.append(current_dir)
        if min_age > 0.0:
            try:
                dir_mtimes[current_dir] = current_dir.stat().st_mtime
            except OSError:
                dir_mtimes[current_dir] = now

    for current_dir in walk_dirs:
        if current_dir != resolved and _is_dir_excluded(current_dir, resolved, excluded):
            continue
        if min_age > 0.0 and (now - dir_mtimes.get(current_dir, now)) < min_age:
            continue
        try:
            if current_dir.exists() and not any(current_dir.iterdir()):
                current_dir.rmdir()
        except FileNotFoundError:
            pass
        except OSError as exc:
            # A concurrent writer may have populated this directory after our
            # emptiness check; rmdir failing on non-empty is expected and safe.
            try:
                if current_dir.exists() and any(current_dir.iterdir()):
                    continue
            except OSError:
                continue
            _log_cleanup_interrupted(current_dir, exc)
        except SystemExit as exc:
            _log_cleanup_interrupted(current_dir, exc)


def cleanup_runtime_ephemeral_temp(
    *,
    job_temp_root: str | Path,
    asr_chunk_root: str | Path,
    project_root: Path,
    min_age_s: float = _DEFAULT_EMPTY_DIR_MIN_AGE_S,
    exclude_names: Collection[str] | None = None,
) -> None:
    roots = [
        Path(job_temp_root).expanduser(),
        Path(asr_chunk_root).expanduser(),
    ]
    resolved_project_root = project_root.resolve()
    for root in roots:
        if not root.is_absolute():
            root = resolved_project_root / root
        cleanup_empty_runtime_dir_tree(
            root,
            resolved_project_root,
            min_age_s=min_age_s,
            exclude_names=exclude_names,
        )
