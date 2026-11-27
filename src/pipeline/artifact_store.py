"""Immutable output generations, one atomic commit, and independent exports."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypeVar

from core import atomic_io

log = logging.getLogger(__name__)

MANIFEST_SCHEMA = "run_outputs_v2"
OWNER_SCHEMA = "output_owner_v1"

_STALE_STAGING_MAX_AGE_S = 3600.0
_WINDOWS_SAFE_PATH_CHARS = 240
_COMPOUND_ARTIFACT_SUFFIXES = (
    ".aligned_segments.json",
    ".asr_manifest.json",
    ".transcript.json",
    ".bilingual.json",
    ".timings.json",
    ".ja.srt",
    ".run.log",
)

# Payload fields whose values are subtitle prose, word lists, hashes, or binary
# encodings (e.g. 2 MB Base64 frame_class_track). Skipping them during path
# relativization prevents both multi-second Path() overhead and accidental
# backslash rewriting inside dialogue/kaomoji text.
_NON_PATH_CONTENT_KEYS = frozenset(
    {
        "frame_class_track",
        "words",
        "text",
        "ja_text",
        "zh_text",
        "raw_text",
        "raw_texts",
        "sha256",
        "sizes",
        "fingerprint",
        "classes",
        "data_b64",
        "schema",
        "translation_review",
        "translation_request_timings",
        "translation_api_retry_events",
    }
)

T = TypeVar("T")


class OutputsSupersededError(RuntimeError):
    """A newer run already published these targets, so this one must not.

    Not a failure of the run that raises it - its work was simply overtaken, and
    the right response is to leave the newer output alone.
    """


# --------------------------------------------------------------------------
# Paths as they are written into payloads
# --------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _root_pattern(project_root_text: str) -> re.Pattern[str]:
    # Anchored (matched, not searched) and only at a path-component boundary.
    # `escape(root) + "/?"` substituted anywhere in the string was substring
    # surgery, not a path operation: with the root `.../jav-sub`, a *sibling*
    # `.../jav-sub-backup/sample-a.srt` serialized as `-backup/sample-a.srt`,
    # which reads back as a different file than the one that was written. A
    # separator (or end of string) after the root makes this a prefix test.
    return re.compile(re.escape(project_root_text) + r"(?:/+|$)", re.IGNORECASE)


@lru_cache(maxsize=8)
def _resolved_root_text(project_root: Path) -> str:
    return project_root.resolve().as_posix()


def project_relative(path: str | Path | None, *, project_root: Path) -> str | None:
    if path is None:
        return None
    raw = str(path)
    if not raw or len(raw) > 4096:
        return raw
    if "/" not in raw and "\\" not in raw:
        # Not a path, so neither branch below can change it: the root pattern
        # cannot match without a separator, and a separator-less string is never
        # absolute. Returning early keeps subtitle text and word tokens - which
        # are the overwhelming majority of strings in these payloads - off the
        # filesystem. `Path.resolve()` is a syscall (~150us here), and paying it
        # per string made write_output 37.7s on a 2h09m film.
        return raw
    project_root_text = _resolved_root_text(project_root)
    normalized = raw.replace("\\", "/")
    # Only a string that *starts* with the root is under it. One that merely
    # contains it - a log line, a message - is left alone: the paths inside
    # those are already relativized where they are formatted, and rewriting a
    # match in the middle of prose is how a sibling directory lost its name.
    prefix = _root_pattern(project_root_text).match(normalized)
    if prefix is not None:
        return normalized[prefix.end():] or "."
    candidate = Path(raw)
    try:
        if candidate.is_absolute():
            return candidate.resolve().relative_to(project_root).as_posix()
    except (OSError, ValueError):
        return normalized
    return normalized


def relativize_payload_paths(value, *, project_root: Path):
    if isinstance(value, dict):
        return {
            key: (
                item
                if key in _NON_PATH_CONTENT_KEYS
                else relativize_payload_paths(item, project_root=project_root)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            relativize_payload_paths(item, project_root=project_root)
            for item in value
        ]
    if isinstance(value, str):
        return project_relative(value, project_root=project_root)
    return value


# --------------------------------------------------------------------------
# Writing one file at a time
# --------------------------------------------------------------------------


def temp_publish_path(target: Path) -> Path:
    """A staging name no other writer of the same target can pick.

    Keyed by process *and* by call: two threads writing one target - the same
    quality report from two jobs sharing a stem, a retry racing the run it
    replaced - used to agree on `<name>.<pid>.tmp`, so the second `replace()`
    moved the file out from under the first, which then failed with
    FileNotFoundError while publishing a file that had in fact been written.
    """
    return target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


def replace_atomically(source: Path, target: Path, *, attempts: int = 5) -> None:
    """`os.replace`, with the retry Windows makes necessary.

    Unique staging names stop two writers from moving the *same* file, but they
    still land on one target, and on Windows a replace fails with a sharing
    violation while another handle to the destination is open - which is exactly
    what the other writer's own replace is holding for a moment. Measured as an
    intermittent `PermissionError` in the two-writer test, not theorised.
    """
    atomic_io.replace_atomically(source, target, retries=attempts, base_delay_s=0.02)


def _read_text_with_retry(
    path: Path,
    *,
    attempts: int = 6,
    base_delay_s: float = 0.02,
) -> str:
    """Read UTF-8 text, retrying transient Windows sharing ``PermissionError``."""
    tries = max(1, int(attempts))
    for attempt in range(tries):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            if attempt + 1 >= tries:
                raise
            time.sleep(base_delay_s * (attempt + 1))
    return path.read_text(encoding="utf-8")


def write_json(path: str | Path, payload: dict, *, project_root: Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as writer:
        json.dump(
            relativize_payload_paths(payload, project_root=project_root),
            writer,
            ensure_ascii=False,
            indent=2,
        )


def write_json_atomic(
    path: str | Path,
    payload: dict,
    *,
    project_root: Path,
    indent: int | None = 2,
) -> None:
    relativized = relativize_payload_paths(payload, project_root=project_root)
    atomic_io.write_json_atomic(
        path,
        relativized,
        indent=indent,
        ensure_ascii=False,
        trailing_newline=False,
    )


# --------------------------------------------------------------------------
# Publishing a whole generation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputLease:
    target: Path
    project_root: Path
    job_id: str
    run_id: str
    sequence: int
    previous_owner: tuple | None = field(default=None, repr=False, compare=False)


class OutputExportError(RuntimeError):
    """A complete generation exists, but its external subtitle export failed."""

    def __init__(self, lease: OutputLease, *, superseded: bool = False) -> None:
        super().__init__(
            "产物已保存，但输出路径已由其他运行接管，不能覆盖；可从任务卡下载本次结果。"
            if superseded
            else "产物已保存，但字幕导出失败；请检查输出目录后重试导出。"
        )
        self.lease = lease
        self.superseded = superseded


class OutputCommitError(RuntimeError):
    """The generation is prepared and can be committed without recomputation."""

    def __init__(self, lease: OutputLease) -> None:
        super().__init__("完整产物已保存，但提交失败；请检查磁盘后重试提交，无需重新计算。")
        self.lease = lease


class OutputIntegrityError(RuntimeError):
    """An existing output record is unreadable or no longer matches its files."""

    def __init__(self) -> None:
        super().__init__(
            "产物记录或文件缺失、损坏或不可读；请恢复文件后重试。为避免重复计算，未启动新任务。"
        )


_locks_guard = threading.Lock()
_target_locks: dict[str, threading.RLock] = {}
_lock_depth = threading.local()


def _target_text(target: str | Path) -> str:
    return os.path.normcase(str(Path(target).expanduser().resolve()))


def output_store_root(target: str | Path, *, project_root: Path) -> Path:
    # 128 bits of the digest keep Windows paths short without using media names.
    key = hashlib.sha256(_target_text(target).encode("utf-8")).hexdigest()[:32]
    return project_root.resolve() / "outputs" / key


def _rewrite_legacy_output_relpath(raw: str) -> str:
    normalized = raw.replace("\\", "/")
    if normalized.lower().startswith("tmp/outputs/"):
        return "outputs/" + normalized[len("tmp/outputs/") :]
    return raw


def _rewrite_store_manifests(
    store_dir: Path,
    *,
    project_root: Path,
    is_legacy_source: bool,
) -> bool:
    """Rewrite legacy ``tmp/outputs/`` paths in a store's generation manifests.

    Returns ``True`` if at least one manifest was rewritten. Corrupt /
    non-UTF-8 manifests are isolated to their own generation (logged and
    skipped) so a single damaged artifact never blocks unrelated jobs or Web
    startup. When migrating a legacy source under ``tmp/outputs/`` or writing
    back a confirmed legacy manifest rewrite, ``OSError`` propagates so
    transient locks pause migration instead of publishing stale manifests.
    """
    generations_dir = store_dir / "generations"
    if not generations_dir.is_dir():
        return False
    rewrote_any = False
    for manifest_path in sorted(generations_dir.glob("*/manifest.json")):
        try:
            raw_text = manifest_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning(
                "Skipping non-UTF-8 manifest %s during output migration; integrity check will isolate this generation",
                manifest_path,
            )
            continue
        except OSError:
            if is_legacy_source:
                raise
            log.warning(
                "Could not read manifest %s during output scan; isolating to this generation",
                manifest_path,
                exc_info=True,
            )
            continue
        try:
            payload = json.loads(raw_text)
        except ValueError:
            log.warning(
                "Skipping malformed manifest %s during output migration; integrity check will isolate this generation",
                manifest_path,
            )
            continue
        if not isinstance(payload, dict):
            continue
        files = payload.get("files")
        if not isinstance(files, dict):
            continue
        updated_files = {}
        changed = False
        for name, raw in files.items():
            if isinstance(raw, str):
                rewritten = _rewrite_legacy_output_relpath(raw)
                if rewritten != raw:
                    changed = True
                updated_files[name] = rewritten
            else:
                updated_files[name] = raw
        if changed:
            payload = {**payload, "files": updated_files}
            write_json_atomic(manifest_path, payload, project_root=project_root)
            rewrote_any = True
    return rewrote_any


def _tree_file_sizes(root: Path) -> dict[str, int]:
    inventory: dict[str, int] = {}
    for current, _dirs, files in os.walk(root):
        base = Path(current)
        for name in files:
            path = base / name
            rel = path.relative_to(root).as_posix()
            inventory[rel] = path.stat().st_size
    return inventory


def _verify_staged_store_manifests(staging: Path, *, store_name: str) -> None:
    """Verify every generation file in a cross-volume staged copy against its manifest SHA-256."""
    generations_dir = staging / "generations"
    if not generations_dir.is_dir():
        return
    expected_prefixes = (
        f"outputs/{store_name}/".lower(),
        f"tmp/outputs/{store_name}/".lower(),
    )
    for manifest_path in sorted(generations_dir.glob("*/manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        files = payload.get("files")
        hashes = payload.get("sha256")
        if not isinstance(files, dict) or not isinstance(hashes, dict):
            continue
        for name, expected_hash in hashes.items():
            raw = files.get(name)
            if not isinstance(raw, str) or not isinstance(expected_hash, str):
                continue
            normalized = raw.replace("\\", "/")
            rel_under_store: str | None = None
            lowered = normalized.lower()
            for prefix in expected_prefixes:
                if lowered.startswith(prefix):
                    rel_under_store = normalized[len(prefix) :]
                    break
            if rel_under_store is None:
                continue
            staged_file = (staging / rel_under_store).resolve()
            try:
                staged_file.relative_to(manifest_path.parent.resolve())
            except ValueError as exc:
                raise OSError(f"旧版产物目录复制后路径校验失败: {store_name}/{name}") from exc
            if not staged_file.is_file() or _sha256(staged_file) != expected_hash:
                raise OSError(f"旧版产物目录复制后 SHA-256 校验失败: {store_name}/{name}")


def _rename_dir_with_retry(source: Path, destination: Path, *, attempts: int = 8) -> None:
    """`os.rename` for directories with short Windows backoff on transient sharing/Defender locks.

    Rewriting ``manifest.json`` immediately before moving its parent store dir
    triggers Windows Defender / NTFS filter scans that can briefly hold a file
    handle inside ``source`` for tens of milliseconds, causing ``MoveFileExW``
    to fail with ``PermissionError`` (WinError 5 / 32) on the first try.
    """
    for attempt in range(attempts):
        try:
            os.rename(source, destination)
            return
        except PermissionError:
            if attempt >= attempts - 1:
                raise
            time.sleep(0.02 * (attempt + 1))


def _move_store_atomically(entry: Path, destination: Path, *, new_root: Path) -> None:
    """Move a legacy output store into ``outputs/`` without ever leaving a partial destination.

    Uses atomic ``os.rename`` on the same volume (propagating same-volume
    ``OSError`` such as file locks without falling back to partial copy). Only
    on cross-device ``EXDEV`` does it copy into a hidden staging directory under
    ``outputs/``, verify every file size and manifest SHA-256, atomically
    publish via ``os.rename``, and finally remove the source directory.
    """
    new_root.mkdir(parents=True, exist_ok=True)
    try:
        _rename_dir_with_retry(entry, destination)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV and getattr(exc, "winerror", None) != 17:
            raise

    staging = new_root / f".migrate-{entry.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        try:
            shutil.copytree(entry, staging, copy_function=shutil.copy2)
        except shutil.Error as copy_exc:
            raise OSError(f"复制旧版产物目录失败: {copy_exc}") from copy_exc
        if _tree_file_sizes(entry) != _tree_file_sizes(staging):
            raise OSError(f"旧版产物目录复制后完整性校验不一致: {entry.name}")
        _verify_staged_store_manifests(staging, store_name=entry.name)
        _rename_dir_with_retry(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        shutil.rmtree(entry)
    except OSError:
        log.warning(
            "Published migrated output store %s but could not remove legacy copy %s",
            destination,
            entry,
            exc_info=True,
        )


def _cleanup_stale_staging_dirs(
    store_dir: Path,
    *,
    max_age_s: float = _STALE_STAGING_MAX_AGE_S,
) -> int:
    """Remove abandoned ``.s-*`` staging dirs older than ``max_age_s``."""
    removed = 0
    now = time.time()
    age_limit = max(0.0, float(max_age_s))
    try:
        candidates = sorted(store_dir.glob(".s-*"))
    except OSError:
        return 0
    for leftover in candidates:
        if not leftover.is_dir():
            continue
        try:
            if age_limit > 0.0 and (now - leftover.stat().st_mtime) < age_limit:
                continue
        except OSError:
            continue
        shutil.rmtree(leftover, ignore_errors=True)
        if not leftover.exists():
            removed += 1
    return removed


def migrate_legacy_outputs(
    project_root: Path,
    *,
    stale_staging_max_age_s: float = _STALE_STAGING_MAX_AGE_S,
) -> int:
    """Relocate committed 1.8.x output stores from ``tmp/outputs/`` into ``outputs/``.

    Rewrites ``tmp/outputs/`` path prefixes inside each generation's
    ``manifest.json`` atomically *before* publishing the store into ``outputs/``,
    also repairs any store already under ``outputs/`` whose manifests still
    reference ``tmp/outputs/`` (e.g. after an interrupted migration), and sweeps
    stale abandoned ``.s-*`` staging directories. Existing targets under
    ``outputs/`` are never overwritten. Raises the first ``OSError`` encountered
    after processing remaining stores so callers pause dependent cleanup until
    migration completes cleanly.
    """
    resolved_root = project_root.resolve()
    legacy_root = resolved_root / "tmp" / "outputs"
    new_root = resolved_root / "outputs"
    migrated = 0
    first_error: OSError | None = None

    # Clean up any abandoned hidden staging dirs left by a hard crash mid cross-volume copy.
    if new_root.is_dir():
        for leftover in sorted(new_root.glob(".migrate-*")):
            if leftover.is_dir():
                shutil.rmtree(leftover, ignore_errors=True)

    if legacy_root.is_dir():
        for entry in sorted(legacy_root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            destination = new_root / entry.name
            if destination.exists():
                continue
            try:
                _rewrite_store_manifests(
                    entry, project_root=resolved_root, is_legacy_source=True
                )
                _move_store_atomically(entry, destination, new_root=new_root)
                migrated += 1
            except OSError as exc:
                if first_error is None:
                    first_error = exc

    if new_root.is_dir():
        for store_dir in sorted(new_root.iterdir()):
            if not store_dir.is_dir() or store_dir.name.startswith("."):
                continue
            _cleanup_stale_staging_dirs(store_dir, max_age_s=stale_staging_max_age_s)
            try:
                if _rewrite_store_manifests(
                    store_dir, project_root=resolved_root, is_legacy_source=False
                ):
                    migrated += 1
            except OSError as exc:
                if first_error is None:
                    first_error = exc

    if first_error is None:
        with contextlib.suppress(OSError):
            if legacy_root.is_dir() and not any(legacy_root.iterdir()):
                legacy_root.rmdir()
    else:
        raise first_error
    return migrated


@contextlib.contextmanager
def _target_lock(target: str | Path, *, project_root: Path):
    root = output_store_root(target, project_root=project_root)
    key = str(root)
    with _locks_guard:
        lock = _target_locks.setdefault(key, threading.RLock())
    depths: dict[str, int] | None = getattr(_lock_depth, "depths", None)
    if depths is None:
        depths = {}
        _lock_depth.depths = depths

    with lock:
        depth = depths.get(key, 0)
        if depth > 0:
            depths[key] = depth + 1
            try:
                yield root
            finally:
                depths[key] = depth
            return

        root.mkdir(parents=True, exist_ok=True)
        # The file lock also covers a CLI and a Web process sharing a target.
        # Do NOT write to the lock file before acquiring the OS lock: on
        # Windows, writing byte 0 while another process holds an msvcrt lock on
        # byte 0 raises PermissionError immediately instead of retrying.
        with (root / "owner.lock").open("a+b") as handle:
            deadline = time.monotonic() + 15.0
            last_exc: OSError | None = None
            while True:
                try:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    last_exc = exc
                    if time.monotonic() >= deadline:
                        raise TimeoutError("等待产物提交锁超时") from last_exc
                    time.sleep(0.02)
            depths[key] = 1
            try:
                yield root
            finally:
                depths.pop(key, None)
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_owner(root: Path) -> dict:
    path = root / "current.json"
    try:
        payload = json.loads(_read_text_with_retry(path))
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != OWNER_SCHEMA
            or not isinstance(payload.get("owner"), dict)
            or not isinstance(payload.get("commits"), dict)
            or not isinstance(payload.get("exports", {}), dict)
        ):
            raise ValueError("invalid output owner")
        return payload
    except FileNotFoundError:
        # Losing the commit record is not the same as never having published.
        if (root / "generations").exists():
            raise OutputIntegrityError() from None
        return {"schema": OWNER_SCHEMA, "owner": {}, "commits": {}, "current": ""}
    except (OSError, ValueError, TypeError) as exc:
        raise OutputIntegrityError() from exc


def claim_output(
    target: str | Path,
    *,
    project_root: Path,
    job_id: str,
    run_id: str,
) -> OutputLease:
    if not run_id:
        raise ValueError("产物发布必须有 run_id")
    destination = Path(target).expanduser().resolve()
    with _target_lock(destination, project_root=project_root) as root:
        state = _read_owner(root)
        owner = state["owner"]
        previous_owner = None
        if owner.get("run_id") == run_id and owner.get("job_id") == job_id:
            if owner.get("cancelled"):
                raise OutputsSupersededError("本次运行已取消，不能重新取得产物所有权。")
            sequence = int(owner["sequence"])
        else:
            previous_owner = tuple(owner.items())
            sequence = max(int(state.get("sequence", 0)), int(owner.get("sequence", 0))) + 1
            state["sequence"] = sequence
            state["owner"] = {
                "run_id": run_id,
                "job_id": job_id,
                "sequence": sequence,
                "cancelled": False,
            }
            state["target"] = project_relative(destination, project_root=project_root)
            write_json_atomic(root / "current.json", state, project_root=project_root)
        return OutputLease(destination, project_root, job_id, run_id, sequence, previous_owner)


def rollback_claim(lease: OutputLease) -> None:
    """Undo a claim whose job transaction failed, unless a later owner won."""
    if lease.previous_owner is None:
        return
    with _target_lock(lease.target, project_root=lease.project_root) as root:
        state = _read_owner(root)
        if _owns(state, lease) and lease.run_id not in state["commits"]:
            state["owner"] = dict(lease.previous_owner)
            # The allocation clock is retained even when restoring an owner.
            write_json_atomic(root / "current.json", state, project_root=lease.project_root)


def _owns(state: dict, lease: OutputLease) -> bool:
    owner = state.get("owner", {})
    return (
        owner.get("run_id") == lease.run_id
        and owner.get("job_id") == lease.job_id
        and owner.get("sequence") == lease.sequence
        and not owner.get("cancelled", False)
    )


def owns_output(lease: OutputLease) -> bool:
    return _owns(
        _read_owner(output_store_root(lease.target, project_root=lease.project_root)),
        lease,
    )


def revoke_output(lease: OutputLease) -> bool:
    with _target_lock(lease.target, project_root=lease.project_root) as root:
        state = _read_owner(root)
        if _owns(state, lease):
            # Commit and cancel are ordered by this lock. A successful commit
            # is complete work; a late stop cannot invalidate its export.
            if lease.run_id in state["commits"]:
                return False
            state["owner"]["cancelled"] = True
            write_json_atomic(root / "current.json", state, project_root=lease.project_root)
            return True
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(raw: str, project_root: Path) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else project_root / path).resolve()


def _generation_dir(lease: OutputLease) -> Path:
    identity = hashlib.sha256(f"{lease.job_id}\0{lease.run_id}".encode()).hexdigest()[:16]
    return (
        output_store_root(lease.target, project_root=lease.project_root)
        / "generations"
        / f"{lease.sequence}-{identity}"
    )


def _validated_manifest(
    path: Path,
    lease: OutputLease,
    *,
    verify: bool,
    verify_files: Collection[str] | None = None,
) -> dict:
    root = output_store_root(lease.target, project_root=lease.project_root)
    checked_names = set(verify_files) if verify_files is not None else None
    try:
        path = path.resolve()
        path.relative_to((root / "generations").resolve())
        manifest = json.loads(_read_text_with_retry(path))
        if not isinstance(manifest, dict) or (
            manifest.get("schema") != MANIFEST_SCHEMA
            or manifest.get("run_id") != lease.run_id
            or manifest.get("job_id") != lease.job_id
            or manifest.get("sequence") != lease.sequence
        ):
            raise ValueError("invalid generation identity")
        files = manifest.get("files")
        if not isinstance(files, dict) or "srt" not in files or not files:
            raise ValueError("missing generation files")
        cue_count = manifest.get("subtitle_cue_count")
        if cue_count is not None and (type(cue_count) is not int or cue_count < 0):
            raise ValueError("invalid subtitle cue count")
        if set(files) != set(manifest["sizes"]) or set(files) != set(manifest["sha256"]):
            raise ValueError("incomplete file inventory")
        for name, raw in files.items():
            file = artifact_path(raw, lease.project_root)
            file.relative_to(path.parent)
            if not file.is_file() or file.stat().st_size != manifest["sizes"][name]:
                raise ValueError("missing or truncated output")
            should_hash = verify or (checked_names is not None and name in checked_names)
            if should_hash and _sha256(file) != manifest["sha256"][name]:
                raise ValueError("output checksum mismatch")
        return manifest
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise OutputIntegrityError() from exc


def read_run_manifest(
    lease: OutputLease,
    *,
    verify: bool = False,
    verify_files: Collection[str] | None = None,
) -> dict | None:
    root = output_store_root(lease.target, project_root=lease.project_root)
    state = _read_owner(root)
    if lease.run_id not in state["commits"]:
        return None
    relative = state["commits"][lease.run_id]
    if not isinstance(relative, str):
        raise OutputIntegrityError()
    return _validated_manifest(
        root / relative,
        lease,
        verify=verify,
        verify_files=verify_files,
    )


def read_prepared_manifest(lease: OutputLease, *, verify: bool = True) -> dict | None:
    directory = _generation_dir(lease)
    try:
        directory.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OutputIntegrityError() from exc
    return _validated_manifest(directory / "manifest.json", lease, verify=verify)


def commit_output(
    lease: OutputLease,
    *,
    still_current: Callable[[], bool] | None = None,
    _prepared_manifest: dict | None = None,
) -> dict:
    """One run has one prepared generation and one atomic visibility change."""
    try:
        with _target_lock(lease.target, project_root=lease.project_root) as root:
            state = _read_owner(root)
            if not _owns(state, lease) or (still_current is not None and not still_current()):
                raise OutputsSupersededError("本次运行已取消或输出目标已有更新运行。")
            committed = read_run_manifest(lease, verify=True)
            if committed is not None:
                return committed
            prepared = (
                _prepared_manifest
                if _prepared_manifest is not None
                else read_prepared_manifest(lease, verify=True)
            )
            if prepared is None:
                raise OutputIntegrityError()
            state["commits"][lease.run_id] = (
                (_generation_dir(lease) / "manifest.json").relative_to(root).as_posix()
            )
            state["current"] = lease.run_id
            write_json_atomic(root / "current.json", state, project_root=lease.project_root)
            return prepared
    except (OSError, TimeoutError) as exc:
        raise OutputCommitError(lease) from exc


def resume_output(lease: OutputLease) -> dict | None:
    """Recover saved work before entering any compute or backend queue."""
    with _target_lock(lease.target, project_root=lease.project_root):
        committed = read_run_manifest(lease, verify=True)
        if committed is None:
            prepared = read_prepared_manifest(lease, verify=True)
            if prepared is None:
                return None
            committed = commit_output(lease, _prepared_manifest=prepared)
        export_output(lease)
        return committed


def manifest_paths(manifest: dict, *, project_root: Path) -> list[str]:
    return [str(artifact_path(raw, project_root)) for raw in manifest["files"].values()]


def export_output(lease: OutputLease) -> dict:
    """Export only the player-facing subtitle, using a temp on its own volume.

    The committed generation remains readable if exporting or recording the
    export fails. A retry repeats this one copy, never computation.
    """
    try:
        with _target_lock(lease.target, project_root=lease.project_root) as root:
            state = _read_owner(root)
            # Verify sizes of all files in the generation and SHA-256 of the
            # exported subtitle itself, avoiding a redundant full-tree hash over
            # multi-megabyte JSON sidecars that were already verified on commit.
            manifest = read_run_manifest(lease, verify=False, verify_files=("srt",))
            if manifest is None:
                raise OutputIntegrityError()
            if not _owns(state, lease) or state.get("current") != lease.run_id:
                # Ownership of the export may change after commit. That must
                # never reclassify already saved work as a cancelled run.
                raise OutputExportError(lease, superseded=True)
            source = artifact_path(manifest["files"]["srt"], lease.project_root)
            target = lease.target
            staged = temp_publish_path(target)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, staged)
                with staged.open("r+b") as handle:
                    os.fsync(handle.fileno())
                replace_atomically(staged, target)
                state.setdefault("exports", {})[lease.run_id] = True
                write_json_atomic(root / "current.json", state, project_root=lease.project_root)
            finally:
                with contextlib.suppress(OSError):
                    staged.unlink(missing_ok=True)
            return manifest
    except (OSError, TimeoutError) as exc:
        raise OutputExportError(lease) from exc


def output_exported(lease: OutputLease) -> bool:
    root = output_store_root(lease.target, project_root=lease.project_root)
    return bool(_read_owner(root).get("exports", {}).get(lease.run_id))


@dataclass(frozen=True)
class StagedFile:
    name: str
    staged: Path
    target: Path


class RunOutputs:
    def __init__(self, *, lease: OutputLease) -> None:
        self.lease = lease
        self.run_id = lease.run_id
        self.job_id = lease.job_id
        self.project_root = lease.project_root
        self.sequence = lease.sequence
        self._root = output_store_root(lease.target, project_root=lease.project_root)
        self._staging_dir = self._root / f".s-{uuid.uuid4().hex[:16]}"
        # Recoverable even if updating current.json fails or the process exits.
        self._generation_dir = _generation_dir(lease)
        self._staged: list[StagedFile] = []

    def __enter__(self) -> RunOutputs:
        return self

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        if self._staged or self._staging_dir.exists():
            self.discard()
        return False

    @property
    def staging_dir(self) -> Path:
        return self._staging_dir

    def _safe_staged_filename(self, index: int, destination_name: str) -> str:
        """Keep internal generation paths safely under Windows MAX_PATH.

        Normal-length filenames are preserved verbatim; unusually long source
        stems are deterministically truncated with a short hash while keeping
        the artifact suffix intact. Full target paths remain recorded in
        ``manifest["targets"]``.
        """
        slot = f"{index:02d}"
        gen_probe = len(str(self._generation_dir / slot / destination_name))
        stage_probe = len(str(self._staging_dir / slot / destination_name))
        if max(gen_probe, stage_probe) <= _WINDOWS_SAFE_PATH_CHARS:
            return destination_name

        suffix = ""
        lowered = destination_name.lower()
        for compound in _COMPOUND_ARTIFACT_SUFFIXES:
            if lowered.endswith(compound):
                suffix = destination_name[-len(compound) :]
                break
        if not suffix:
            suffix = Path(destination_name).suffix
        stem = destination_name[: -len(suffix)] if suffix else destination_name
        digest = hashlib.sha1(destination_name.encode("utf-8", errors="ignore")).hexdigest()[:8]
        base_len = max(len(str(self._generation_dir / slot)), len(str(self._staging_dir / slot))) + 1
        budget = max(16, _WINDOWS_SAFE_PATH_CHARS - base_len)
        keep = max(1, budget - len(suffix) - len(digest) - 1)
        return f"{stem[:keep]}-{digest}{suffix}"

    def stage(self, target: str | Path, *, name: str, write: Callable[[Path], T]) -> T:
        if any(item.name == name for item in self._staged):
            raise ValueError(f"重复产物名称：{name}")
        destination = Path(target)
        index = len(self._staged)
        safe_name = self._safe_staged_filename(index, destination.name)
        staged = self._staging_dir / f"{index:02d}" / safe_name
        staged.parent.mkdir(parents=True, exist_ok=True)
        result = write(staged)
        self._staged.append(StagedFile(name, staged, destination))
        return result

    def stage_json(self, target: str | Path, payload: dict, *, name: str) -> None:
        self.stage(
            target,
            name=name,
            write=lambda path: write_json(path, payload, project_root=self.project_root),
        )

    def stage_file(self, source: str | Path, *, name: str) -> None:
        self.stage(source, name=name, write=lambda path: shutil.copyfile(source, path))

    def targets(self) -> list[Path]:
        return [item.target for item in self._staged]

    def publish(
        self,
        *,
        still_current: Callable[[], bool] | None = None,
        subtitle_cue_count: int | None = None,
    ) -> dict:
        if not self._staged or not any(item.name == "srt" for item in self._staged):
            raise ValueError("产物代次缺少字幕")
        files, sizes, hashes = {}, {}, {}
        for item in self._staged:
            with item.staged.open("r+b") as handle:
                os.fsync(handle.fileno())
            files[item.name] = project_relative(
                self._generation_dir / item.staged.relative_to(self._staging_dir),
                project_root=self.project_root,
            )
            sizes[item.name] = item.staged.stat().st_size
            hashes[item.name] = _sha256(item.staged)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "sequence": self.sequence,
            "files": files,
            "sizes": sizes,
            "sha256": hashes,
            "targets": {
                item.name: project_relative(item.target, project_root=self.project_root)
                for item in self._staged
            },
        }
        if subtitle_cue_count is not None:
            manifest["subtitle_cue_count"] = subtitle_cue_count
        write_json_atomic(
            self._staging_dir / "manifest.json",
            manifest,
            project_root=self.project_root,
        )
        self._generation_dir.parent.mkdir(parents=True, exist_ok=True)
        # Ownership check, staging->generation rename, and current.json commit
        # execute inside one reentrant `_target_lock` critical section:
        #   - cancelled/superseded runs discard before renaming (no orphan
        #     `generations/` directories);
        #   - rename retries transient Windows Defender/sharing locks and maps
        #     failures to `OutputCommitError`;
        #   - `commit_output` reuses the freshly hashed manifest without a
        #     redundant second full-tree SHA-256 pass.
        try:
            with _target_lock(self.lease.target, project_root=self.project_root) as root:
                state = _read_owner(root)
                if not _owns(state, self.lease) or (
                    still_current is not None and not still_current()
                ):
                    self.discard()
                    raise OutputsSupersededError("本次运行已取消或输出目标已有更新运行。")
                if self._generation_dir.exists():
                    prepared_manifest = _validated_manifest(
                        self._generation_dir / "manifest.json",
                        self.lease,
                        verify=True,
                    )
                    self.discard()
                else:
                    # Both directories belong to this store: always same-volume.
                    _rename_dir_with_retry(self._staging_dir, self._generation_dir)
                    prepared_manifest = manifest
                self._staged = []
                return commit_output(
                    self.lease,
                    still_current=still_current,
                    _prepared_manifest=prepared_manifest,
                )
        except (OSError, TimeoutError) as exc:
            raise OutputCommitError(self.lease) from exc

    def discard(self) -> None:
        self._staged = []
        shutil.rmtree(self._staging_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Reading an artifact back out
# --------------------------------------------------------------------------


def authorized_path(
    candidates: Iterable[str | Path],
    allowed_roots: Iterable[Path],
) -> Path | None:
    """The first candidate that exists inside one of the allowed roots.

    Opening and downloading ask the same question, and used to answer it with
    two different rules - so a tampered `jobs.json` entry could be opened from
    the page but 404 on download, or the other way round. One rule: resolve the
    candidate, then require it to sit under a root derived from the job's own
    specification. Path traversal is not a string problem, so this compares
    resolved paths rather than text.
    """
    roots = list(allowed_roots)
    for candidate in candidates:
        try:
            resolved = Path(candidate).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if not resolved.exists():
            continue
        for root in roots:
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            return resolved
    return None
