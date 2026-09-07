"""Immutable output generations, one atomic commit, and independent exports."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, TypeVar

MANIFEST_SCHEMA = "run_outputs_v2"
OWNER_SCHEMA = "output_owner_v1"

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
    # surgery, not a path operation: with the root `.../jav-trans`, a *sibling*
    # `.../jav-trans-backup/sample-a.srt` serialized as `-backup/sample-a.srt`,
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
    if not raw:
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
            key: relativize_payload_paths(item, project_root=project_root)
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
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt >= attempts - 1:
                raise
            time.sleep(0.02 * (attempt + 1))


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


def write_json_atomic(path: str | Path, payload: dict, *, project_root: Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = temp_publish_path(target)
    try:
        tmp_path.write_text(
            json.dumps(
                relativize_payload_paths(payload, project_root=project_root),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        with tmp_path.open("r+b") as handle:
            os.fsync(handle.fileno())
        replace_atomically(tmp_path, target)
    finally:
        # After a successful replace there is nothing left to remove; this is
        # for the failed write, which would otherwise leave staging files
        # beside the artifact it never became.
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)


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
            if superseded else
            "产物已保存，但字幕导出失败；请检查输出目录后重试导出。"
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
        super().__init__("产物记录或文件缺失、损坏或不可读；请恢复文件后重试。为避免重复计算，未启动新任务。")


_locks_guard = threading.Lock()
_target_locks: dict[str, threading.RLock] = {}


def _target_text(target: str | Path) -> str:
    return os.path.normcase(str(Path(target).expanduser().resolve()))


def output_store_root(target: str | Path, *, project_root: Path) -> Path:
    # 128 bits of the digest keep Windows paths short without using media names.
    key = hashlib.sha256(_target_text(target).encode("utf-8")).hexdigest()[:32]
    return project_root.resolve() / "tmp" / "outputs" / key


@contextlib.contextmanager
def _target_lock(target: str | Path, *, project_root: Path):
    root = output_store_root(target, project_root=project_root)
    with _locks_guard:
        lock = _target_locks.setdefault(str(root), threading.RLock())
    with lock:
        root.mkdir(parents=True, exist_ok=True)
        # The file lock also covers a CLI and a Web process sharing a target.
        with (root / "owner.lock").open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            deadline = time.monotonic() + 15.0
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
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("等待产物提交锁超时")
                    time.sleep(0.02)
            try:
                yield root
            finally:
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
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict) or payload.get("schema") != OWNER_SCHEMA
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
    target: str | Path, *, project_root: Path, job_id: str, run_id: str,
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
                "run_id": run_id, "job_id": job_id, "sequence": sequence,
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
    return output_store_root(lease.target, project_root=lease.project_root) / "generations" / f"{lease.sequence}-{identity}"


def _validated_manifest(path: Path, lease: OutputLease, *, verify: bool) -> dict:
    root = output_store_root(lease.target, project_root=lease.project_root)
    try:
        path = path.resolve()
        path.relative_to((root / "generations").resolve())
        manifest = json.loads(path.read_text(encoding="utf-8"))
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
        if set(files) != set(manifest["sizes"]) or set(files) != set(manifest["sha256"]):
            raise ValueError("incomplete file inventory")
        for name, raw in files.items():
            file = artifact_path(raw, lease.project_root)
            file.relative_to(path.parent)
            if not file.is_file() or file.stat().st_size != manifest["sizes"][name]:
                raise ValueError("missing or truncated output")
            if verify and _sha256(file) != manifest["sha256"][name]:
                raise ValueError("output checksum mismatch")
        return manifest
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise OutputIntegrityError() from exc


def read_run_manifest(lease: OutputLease, *, verify: bool = False) -> dict | None:
    root = output_store_root(lease.target, project_root=lease.project_root)
    state = _read_owner(root)
    if lease.run_id not in state["commits"]:
        return None
    relative = state["commits"][lease.run_id]
    if not isinstance(relative, str):
        raise OutputIntegrityError()
    return _validated_manifest(root / relative, lease, verify=verify)


def read_prepared_manifest(lease: OutputLease, *, verify: bool = True) -> dict | None:
    directory = _generation_dir(lease)
    try:
        directory.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OutputIntegrityError() from exc
    return _validated_manifest(directory / "manifest.json", lease, verify=verify)


def commit_output(lease: OutputLease, *, still_current: Callable[[], bool] | None = None) -> dict:
    """One run has one prepared generation and one atomic visibility change."""
    try:
        with _target_lock(lease.target, project_root=lease.project_root) as root:
            state = _read_owner(root)
            if not _owns(state, lease) or (still_current is not None and not still_current()):
                raise OutputsSupersededError("本次运行已取消或输出目标已有更新运行。")
            committed = read_run_manifest(lease, verify=True)
            if committed is not None:
                return committed
            prepared = read_prepared_manifest(lease)
            if prepared is None:
                raise OutputIntegrityError()
            state["commits"][lease.run_id] = (_generation_dir(lease) / "manifest.json").relative_to(root).as_posix()
            state["current"] = lease.run_id
            write_json_atomic(root / "current.json", state, project_root=lease.project_root)
            return prepared
    except (OSError, TimeoutError) as exc:
        raise OutputCommitError(lease) from exc


def resume_output(lease: OutputLease) -> dict | None:
    """Recover saved work before entering any compute or backend queue."""
    committed = read_run_manifest(lease, verify=True)
    if committed is None:
        if read_prepared_manifest(lease) is None:
            return None
        committed = commit_output(lease)
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
            manifest = read_run_manifest(lease, verify=True)
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

    @property
    def staging_dir(self) -> Path:
        return self._staging_dir

    def stage(self, target: str | Path, *, name: str, write: Callable[[Path], T]) -> T:
        if any(item.name == name for item in self._staged):
            raise ValueError(f"重复产物名称：{name}")
        destination = Path(target)
        staged = self._staging_dir / f"{len(self._staged):02d}" / destination.name
        staged.parent.mkdir(parents=True, exist_ok=True)
        result = write(staged)
        self._staged.append(StagedFile(name, staged, destination))
        return result

    def stage_json(self, target: str | Path, payload: dict, *, name: str) -> None:
        self.stage(
            target, name=name,
            write=lambda path: write_json(path, payload, project_root=self.project_root),
        )

    def stage_file(self, source: str | Path, *, name: str) -> None:
        self.stage(source, name=name, write=lambda path: shutil.copyfile(source, path))

    def targets(self) -> list[Path]:
        return [item.target for item in self._staged]

    def publish(self, *, still_current: Callable[[], bool] | None = None) -> dict:
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
            "schema": MANIFEST_SCHEMA, "run_id": self.run_id,
            "job_id": self.job_id, "sequence": self.sequence,
            "files": files, "sizes": sizes, "sha256": hashes,
            "targets": {
                item.name: project_relative(item.target, project_root=self.project_root)
                for item in self._staged
            },
        }
        write_json_atomic(
            self._staging_dir / "manifest.json", manifest, project_root=self.project_root,
        )
        self._generation_dir.parent.mkdir(parents=True, exist_ok=True)
        # Preparing is durable but invisible. The fixed per-run name lets a
        # restart find it without needing another pointer write to succeed.
        with _target_lock(self.lease.target, project_root=self.project_root) as root:
            if self._generation_dir.exists():
                _validated_manifest(self._generation_dir / "manifest.json", self.lease, verify=True)
                self.discard()
            else:
                # Both directories belong to this store: always same-volume.
                replace_atomically(self._staging_dir, self._generation_dir)
        self._staged = []
        return commit_output(self.lease, still_current=still_current)

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
