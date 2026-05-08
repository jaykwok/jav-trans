from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable


def unlink_for_cleanup(path: Path) -> None:
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception as exc:
        print(f"[WARN] cleanup failed for {path}: {exc}", file=sys.stderr)


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

    for cache_path_item in cache_paths:
        unlink_for_cleanup(cache_path_item)
        for tmp_path in cache_path_item.parent.glob(f"{cache_path_item.name}.*.tmp"):
            unlink_for_cleanup(tmp_path)


def cleanup_asr_checkpoints(job_temp_dir: Path, checkpoint_root: Path) -> None:
    job_marker = str(job_temp_dir.resolve()).replace("\\", "/").lower()
    for checkpoint_path in checkpoint_root.glob("asr_checkpoint_*.json"):
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint_source = str(payload.get("audio_path", ""))
        except Exception:
            checkpoint_source = ""
        normalized_source = checkpoint_source.replace("\\", "/").lower()
        if job_marker and job_marker in normalized_source:
            unlink_for_cleanup(checkpoint_path)


def cleanup_asr_checkpoint_for_audio(
    audio_path: str,
    get_checkpoint_path: Callable[[str], str | Path],
) -> None:
    try:
        checkpoint_path = Path(get_checkpoint_path(audio_path))
    except Exception as exc:
        print(f"[WARN] cleanup failed to resolve ASR checkpoint: {exc}", file=sys.stderr)
        return
    unlink_for_cleanup(checkpoint_path)


def cleanup_job_temp(
    job_temp_dir: str,
    translation_cache_path: str = "",
    *,
    checkpoint_root: Path,
) -> None:
    root = Path(job_temp_dir)
    cleanup_translation_cache(translation_cache_path)
    cleanup_asr_checkpoints(root, checkpoint_root)

    if not root.exists() or not root.is_dir():
        return

    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_file():
                path.unlink()
        except Exception as exc:
            print(f"[WARN] cleanup failed for {path}: {exc}", file=sys.stderr)

    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except Exception as exc:
            print(f"[WARN] cleanup failed for {path}: {exc}", file=sys.stderr)

    try:
        if root.exists() and not any(root.iterdir()):
            root.rmdir()
    except Exception as exc:
        print(f"[WARN] cleanup failed for {root}: {exc}", file=sys.stderr)


def cleanup_empty_runtime_dir_tree(root: Path, project_root: Path) -> None:
    try:
        resolved = root.resolve()
        resolved.relative_to(project_root)
    except Exception as exc:
        print(f"[WARN] cleanup skipped for {root}: {exc}", file=sys.stderr)
        return

    if not resolved.exists() or not resolved.is_dir():
        return

    for path in sorted(resolved.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except Exception as exc:
            print(f"[WARN] cleanup failed for {path}: {exc}", file=sys.stderr)

    try:
        if resolved.exists() and not any(resolved.iterdir()):
            resolved.rmdir()
    except Exception as exc:
        print(f"[WARN] cleanup failed for {resolved}: {exc}", file=sys.stderr)


def cleanup_runtime_ephemeral_temp(
    *,
    job_temp_root: str | Path,
    asr_chunk_root: str | Path,
    recovery_output_root: str | Path,
    project_root: Path,
) -> None:
    roots = [
        Path(job_temp_root).expanduser(),
        Path(asr_chunk_root).expanduser(),
        Path(recovery_output_root).expanduser(),
    ]
    resolved_project_root = project_root.resolve()
    for root in roots:
        if not root.is_absolute():
            root = resolved_project_root / root
        cleanup_empty_runtime_dir_tree(root, resolved_project_root)
