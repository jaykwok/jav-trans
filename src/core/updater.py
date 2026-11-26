"""Online self-update checker, verifier, and staging manager.

Checks GitHub ``releases/latest`` (strict ``x.y.z`` stable versions only,
ignoring draft/prerelease tags), downloads ``jav-sub-windows-x64.zip`` with
mandatory SHA-256 and ``release-manifest.json`` verification and safe
extraction (anti Zip-Slip / symlink), verifies ``release-files.json``, writes
``tmp/updates/pending-update.json``, and asks ``launcher.py`` to exit with
``UPDATE_EXIT_CODE`` so ``bootstrap.py`` (``jav-sub.exe``) can apply the
overlay cleanly after all DLLs and files are unlocked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import tomllib
import urllib.parse
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import httpx

from core.config import network_proxy_url_from_env
from utils.runtime_paths import runtime_root

log = logging.getLogger(__name__)

UPDATE_EXIT_CODE = 42
DEFAULT_UPDATE_REPO = "jaykwok/jav-sub"
UPDATE_ARCHIVE_NAME = "jav-sub-windows-x64.zip"
UPDATE_SHA256_NAME = "jav-sub-windows-x64.zip.sha256"
UPDATE_MANIFEST_NAME = "release-manifest.json"
RELEASE_INDEX_NAME = "release-files.json"
RELEASE_INDEX_SCHEMA = "jav_sub_release_files_v1"
RELEASE_MANIFEST_SCHEMA = "release_manifest_v2"
PENDING_UPDATE_SCHEMA = "jav_sub_pending_update_v1"
PENDING_UPDATE_FILE_NAME = "pending-update.json"

_SEMVER_RE = re.compile(r"^[vV]?(\d+)\.(\d+)\.(\d+)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_LINE_RE = re.compile(r"^\s*([0-9a-fA-F]{64})(?:\s+\*?(.+?))?\s*$")
_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_PROTECTED_TOP_DIRS = frozenset(
    {
        ".venv",
        ".python",
        "models",
        "config",
        "outputs",
        "cache",
        "logs",
        "tmp",
        ".git",
    }
)
_REQUIRED_PAYLOAD_ENTRIES = (
    "jav-sub.exe",
    "launcher.py",
    "pyproject.toml",
    "uv.lock",
    RELEASE_INDEX_NAME,
)
_CHECK_TIMEOUT_S = 8.0
_DOWNLOAD_TIMEOUT_S = 60.0
_DOWNLOAD_CHUNK_BYTES = 256 * 1024
_UPDATE_IN_PROGRESS_STATES = frozenset(
    {"downloading", "verifying", "ready_to_restart", "restarting"}
)


class UpdateError(RuntimeError):
    """Expected, user-facing update validation or transport failure."""


class UpdateCancelledError(UpdateError):
    """Raised when the user cancels an active update download."""


@dataclass(frozen=True)
class ReleaseUpdateInfo:
    version: str
    version_tuple: tuple[int, int, int]
    tag: str
    release_url: str
    release_notes: str
    archive_url: str
    archive_size: int
    asset_sha256: str
    sha256_url: str
    manifest_url: str
    published_at: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _project_root(root: Path | None = None) -> Path:
    return (root if root is not None else runtime_root()).resolve()


def parse_semver(raw: str | None) -> tuple[int, int, int] | None:
    """Parse strict ``[v]x.y.z`` stable release versions; reject prereleases."""
    if not isinstance(raw, str):
        return None
    match = _SEMVER_RE.fullmatch(raw.strip())
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def format_semver(version: tuple[int, int, int]) -> str:
    return f"{version[0]}.{version[1]}.{version[2]}"


def current_app_version(root: Path | None = None) -> str:
    """Read the installed project version from ``pyproject.toml``."""
    path = _project_root(root) / "pyproject.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        version = data.get("project", {}).get("version", "")
        if isinstance(version, str) and version.strip():
            return version.strip()
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return "0.0.0"


def auto_update_check_enabled() -> bool:
    """True unless ``JAV_SUB_DISABLE_UPDATE_CHECK`` is explicitly truthy."""
    raw = str(os.getenv("JAV_SUB_DISABLE_UPDATE_CHECK", "0") or "").strip().lower()
    return raw not in _TRUTHY_VALUES


def update_api_url() -> str:
    explicit = str(os.getenv("JAV_SUB_UPDATE_API_URL", "") or "").strip()
    if explicit:
        return explicit
    repo = str(os.getenv("JAV_SUB_UPDATE_REPO", DEFAULT_UPDATE_REPO) or DEFAULT_UPDATE_REPO).strip()
    repo = repo.strip("/") or DEFAULT_UPDATE_REPO
    return f"https://api.github.com/repos/{repo}/releases/latest"


def should_auto_check_on_startup(root: Path | None = None) -> bool:
    """Whether Web startup should fire a non-blocking background update check."""
    if not auto_update_check_enabled():
        return False
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    if os.getenv("JAV_SUB_UPDATE_API_URL"):
        return True
    base = _project_root(root)
    # In source checkouts skip automatic GitHub polls on every dev reload;
    # users can still invoke manual checks from the UI.
    return not (base / ".git").exists()


def can_self_update(root: Path | None = None) -> tuple[bool, str]:
    """Whether in-place self-update is allowed in this installation directory."""
    base = _project_root(root)
    if (base / ".git").exists():
        return False, "当前为源码开发目录（git 工作副本），不支持一键覆盖升级，请使用 git pull 更新。"
    if os.name != "nt":
        return False, "在线覆盖升级仅支持 Windows 发布版。"
    if not (base / "jav-sub.exe").is_file() or not (base / RELEASE_INDEX_NAME).is_file():
        return (
            False,
            "当前目录缺少 jav-sub.exe 或 release-files.json，不是标准发布版目录，请手动下载 ZIP 覆盖升级。",
        )
    return True, ""


def _is_allowed_download_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return False
    if parsed.scheme == "https" and bool(parsed.netloc):
        return True
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}:
        return True
    return False


def _extract_asset_sha256(asset: dict[str, Any]) -> str:
    digest = str(asset.get("digest") or "").strip().lower()
    if digest.startswith("sha256:"):
        candidate = digest[len("sha256:") :].strip()
        if _SHA256_RE.fullmatch(candidate):
            return candidate
    return ""


def parse_sha256_sidecar(text: str, expected_filename: str = UPDATE_ARCHIVE_NAME) -> str | None:
    """Parse a ``.sha256`` sidecar file strictly for ``expected_filename``."""
    expected_cf = expected_filename.casefold()
    bare_candidate: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _SHA256_LINE_RE.fullmatch(stripped)
        if not match:
            continue
        digest = match.group(1).lower()
        name = (match.group(2) or "").strip()
        if name:
            if Path(name).name.casefold() == expected_cf:
                return digest
        elif bare_candidate is None:
            bare_candidate = digest
    return bare_candidate


def select_release_update(
    payload: dict[str, Any],
    *,
    current_version: str,
) -> tuple[bool, ReleaseUpdateInfo | None]:
    """Compare a GitHub ``releases/latest`` payload against ``current_version``.

    Returns ``(is_newer, release_info)``. Drafts, prereleases, and non-``x.y.z``
    tags are ignored cleanly (returning ``(False, None)``).
    """
    if not isinstance(payload, dict):
        raise UpdateError("更新接口返回格式无效。")
    if payload.get("draft") or payload.get("prerelease"):
        return False, None

    tag = str(payload.get("tag_name") or "").strip()
    remote_tuple = parse_semver(tag)
    if remote_tuple is None:
        return False, None

    local_tuple = parse_semver(current_version)
    if local_tuple is None:
        local_tuple = (0, 0, 0)

    assets = payload.get("assets")
    archive_asset: dict[str, Any] | None = None
    sha_asset: dict[str, Any] | None = None
    manifest_asset: dict[str, Any] | None = None
    if isinstance(assets, list):
        for item in assets:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip().casefold()
            if name == UPDATE_ARCHIVE_NAME.casefold():
                archive_asset = item
            elif name == UPDATE_SHA256_NAME.casefold():
                sha_asset = item
            elif name == UPDATE_MANIFEST_NAME.casefold():
                manifest_asset = item

    archive_url = str((archive_asset or {}).get("browser_download_url") or "").strip()
    sha256_url = str((sha_asset or {}).get("browser_download_url") or "").strip()
    manifest_url = str((manifest_asset or {}).get("browser_download_url") or "").strip()
    asset_sha256 = _extract_asset_sha256(archive_asset or {})
    release_url = str(payload.get("html_url") or "").strip()
    release_notes = str(payload.get("body") or "").strip()
    published_at = str(payload.get("published_at") or "").strip()
    try:
        archive_size = int((archive_asset or {}).get("size") or 0)
    except (TypeError, ValueError):
        archive_size = 0

    info = ReleaseUpdateInfo(
        version=format_semver(remote_tuple),
        version_tuple=remote_tuple,
        tag=tag,
        release_url=release_url,
        release_notes=release_notes,
        archive_url=archive_url,
        archive_size=max(0, archive_size),
        asset_sha256=asset_sha256,
        sha256_url=sha256_url,
        manifest_url=manifest_url,
        published_at=published_at,
    )
    if remote_tuple <= local_tuple:
        return False, info
    if not archive_url or not _is_allowed_download_url(archive_url):
        raise UpdateError(f"最新版本 {tag} 缺少有效的 Windows 安装包资产（{UPDATE_ARCHIVE_NAME}）。")
    if sha256_url and not _is_allowed_download_url(sha256_url):
        sha256_url = ""
    if manifest_url and not _is_allowed_download_url(manifest_url):
        manifest_url = ""
    if not asset_sha256 and not sha256_url:
        raise UpdateError(f"最新版本 {tag} 缺少 SHA-256 校验摘要，已拒绝自动下载。")
    if not manifest_url:
        raise UpdateError(f"最新版本 {tag} 缺少发行清单（{UPDATE_MANIFEST_NAME}），已拒绝自动下载。")
    info = ReleaseUpdateInfo(
        version=info.version,
        version_tuple=info.version_tuple,
        tag=info.tag,
        release_url=info.release_url,
        release_notes=info.release_notes,
        archive_url=archive_url,
        archive_size=info.archive_size,
        asset_sha256=asset_sha256,
        sha256_url=sha256_url,
        manifest_url=manifest_url,
        published_at=info.published_at,
    )
    return True, info


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive_member_path(name: str) -> PurePosixPath:
    if "\x00" in name:
        raise UpdateError("安装包包含非法空字符路径，已拒绝解压。")
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise UpdateError(f"安装包包含绝对路径（{name}），已拒绝解压。")
    pure = PurePosixPath(normalized)
    if any(part in {"..", ""} for part in pure.parts):
        raise UpdateError(f"安装包包含越界路径（{name}），已拒绝解压。")
    return pure


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0o120000


def safe_extract_zip(
    archive_path: Path,
    dest_dir: Path,
    *,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Safely extract ``archive_path`` into ``dest_dir`` and return the payload root.

    Rejects absolute paths, drive prefixes, ``..`` traversal, and symlinks.
    Automatically unwraps a single top-level wrapper folder (e.g. ``jav-sub/``)
    so the returned directory directly contains ``jav-sub.exe``, ``launcher.py``,
    ``src/``, and ``release-files.json``.
    """
    extract_root = dest_dir.resolve()
    extract_root.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            infos = zf.infolist()
            if not infos:
                raise UpdateError("安装包为空，已拒绝升级。")
            for info in infos:
                if cancel_event is not None and cancel_event.is_set():
                    raise UpdateCancelledError("已取消更新。")
                if _is_zip_symlink(info):
                    raise UpdateError(f"安装包包含符号链接（{info.filename}），已拒绝解压。")
                pure = _validate_archive_member_path(info.filename)
                target = (extract_root / Path(*pure.parts)).resolve()
                try:
                    target.relative_to(extract_root)
                except ValueError as exc:
                    raise UpdateError(f"安装包包含越界路径（{info.filename}），已拒绝解压。") from exc
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, target.open("wb") as dst:
                    while True:
                        if cancel_event is not None and cancel_event.is_set():
                            raise UpdateCancelledError("已取消更新。")
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
    except zipfile.BadZipFile as exc:
        raise UpdateError("下载的更新包不是有效的 ZIP 文件。") from exc

    # Unwrap a single top-level folder (standard release archive ships jav-sub/...).
    if (extract_root / RELEASE_INDEX_NAME).is_file():
        return extract_root
    top_entries = [p for p in extract_root.iterdir() if p.name != "__MACOSX"]
    if len(top_entries) == 1 and top_entries[0].is_dir():
        return top_entries[0]
    return extract_root


def verify_staged_payload(
    staged_dir: Path,
    *,
    expected_version: str,
    manifest_payload: dict[str, Any] | None = None,
    cancel_event: threading.Event | None = None,
) -> bool:
    """Validate structure, ``release-files.json``, version, and mandatory manifest hashes."""
    if cancel_event is not None and cancel_event.is_set():
        raise UpdateCancelledError("已取消更新。")
    staged = staged_dir.resolve()
    for required in _REQUIRED_PAYLOAD_ENTRIES:
        if not (staged / required).is_file():
            raise UpdateError(f"更新包缺少必需文件：{required}")
    if not (staged / "src").is_dir():
        raise UpdateError("更新包缺少 src/ 目录。")

    try:
        index_data = json.loads((staged / RELEASE_INDEX_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UpdateError("更新包中的 release-files.json 无法解析。") from exc
    if not isinstance(index_data, dict) or index_data.get("schema") != RELEASE_INDEX_SCHEMA:
        raise UpdateError("更新包中的 release-files.json schema 无效。")
    indexed_files = index_data.get("files")
    if not isinstance(indexed_files, list) or not all(isinstance(p, str) for p in indexed_files):
        raise UpdateError("更新包中的 release-files.json 文件列表无效。")
    shipped = set(indexed_files)
    if "launcher.py" not in shipped or not any(p.startswith("src/") for p in shipped):
        raise UpdateError("更新包中的 release-files.json 缺少核心程序条目。")
    for rel in indexed_files:
        if cancel_event is not None and cancel_event.is_set():
            raise UpdateCancelledError("已取消更新。")
        pure = _validate_archive_member_path(rel)
        if pure.parts[0].casefold() in _PROTECTED_TOP_DIRS:
            raise UpdateError(f"更新包试图写入受保护目录（{rel}），已拒绝安装。")
        candidate = (staged / Path(*pure.parts)).resolve()
        try:
            candidate.relative_to(staged)
        except ValueError as exc:
            raise UpdateError(f"更新包清单包含越界路径（{rel}）。") from exc
        if not candidate.is_file():
            raise UpdateError(f"更新包缺少清单声明的文件：{rel}")

    staged_version = current_app_version(staged)
    staged_tuple = parse_semver(staged_version)
    expected_tuple = parse_semver(expected_version)
    if staged_tuple is None or expected_tuple is None or staged_tuple != expected_tuple:
        raise UpdateError(
            f"更新包版本号（{staged_version}）与目标版本（{expected_version}）不一致，已拒绝安装。"
        )

    if manifest_payload is None:
        raise UpdateError("缺少发行清单 release-manifest.json，已拒绝安装。")

    if not isinstance(manifest_payload, dict) or manifest_payload.get("schema") != RELEASE_MANIFEST_SCHEMA:
        raise UpdateError("发行清单 release-manifest.json schema 无效。")
    manifest_files = manifest_payload.get("files")
    if not isinstance(manifest_files, list) or not manifest_files:
        raise UpdateError("发行清单 release-manifest.json 缺少文件校验列表。")

    verified_dist_files = 0
    recorded_dist: set[str] = set()
    for entry in manifest_files:
        if cancel_event is not None and cancel_event.is_set():
            raise UpdateCancelledError("已取消更新。")
        if not isinstance(entry, dict):
            raise UpdateError("发行清单包含无效条目。")
        if entry.get("root") != "dist":
            continue
        rel = str(entry.get("path") or "")
        expected_hash = str(entry.get("sha256") or "").lower()
        expected_bytes = entry.get("bytes")
        if not rel or not _SHA256_RE.fullmatch(expected_hash) or type(expected_bytes) is not int:
            raise UpdateError(f"发行清单条目损坏：{rel!r}")
        pure = _validate_archive_member_path(rel)
        target = (staged / Path(*pure.parts)).resolve()
        try:
            target.relative_to(staged)
        except ValueError as exc:
            raise UpdateError(f"发行清单包含越界路径：{rel}") from exc
        if not target.is_file():
            raise UpdateError(f"更新包缺少发行清单中的文件：{rel}")
        if target.stat().st_size != expected_bytes:
            raise UpdateError(f"更新包文件大小校验失败：{rel}")
        if _sha256_file(target) != expected_hash:
            raise UpdateError(f"更新包文件 SHA-256 校验失败：{rel}")
        recorded_dist.add(pure.as_posix())
        verified_dist_files += 1
    if verified_dist_files == 0:
        raise UpdateError("发行清单中没有任何可核验的发布载荷文件。")
    if cancel_event is not None and cancel_event.is_set():
        raise UpdateCancelledError("已取消更新。")
    extra_files = [
        path.relative_to(staged).as_posix()
        for path in sorted(staged.rglob("*"))
        if path.is_file() and path.relative_to(staged).as_posix() not in recorded_dist
    ]
    if extra_files:
        raise UpdateError(f"更新包包含未登记的额外文件：{extra_files[0]}")
    return True


def other_launcher_instances_running(root: Path | None = None) -> bool:
    """Detect whether another ``launcher.py`` window is holding a slot in ``tmp/launcher``."""
    instance_dir = _project_root(root) / "tmp" / "launcher"
    if not instance_dir.is_dir():
        return False
    own_name = f"instance-{os.getpid()}.lock"
    try:
        slots = sorted(instance_dir.glob("instance-*.lock"))
    except OSError:
        return True
    for slot in slots:
        if slot.name == own_name:
            continue
        if os.name == "nt":
            try:
                slot.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                return True
        else:
            try:
                import fcntl

                with slot.open("r+", encoding="utf-8") as probe:
                    try:
                        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except OSError:
                        return True
                    fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
                slot.unlink(missing_ok=True)
            except FileNotFoundError:
                continue
            except OSError:
                return True
    return False


def _http_client(timeout_s: float) -> httpx.Client:
    proxy_url = network_proxy_url_from_env()
    kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(connect=10.0, read=timeout_s, write=15.0, pool=10.0),
        "follow_redirects": True,
        "headers": {
            "User-Agent": f"jav-sub/{current_app_version()}",
            "Accept": "application/vnd.github+json",
        },
        "trust_env": not bool(proxy_url),
    }
    if proxy_url:
        kwargs["proxy"] = proxy_url
    return httpx.Client(**kwargs)


def write_pending_update_descriptor(
    root: Path,
    *,
    from_version: str,
    info: ReleaseUpdateInfo,
    staged_dir: Path,
    zip_sha256: str,
    manifest_verified: bool,
) -> Path:
    base = _project_root(root)
    updates_dir = base / "tmp" / "updates"
    updates_dir.mkdir(parents=True, exist_ok=True)
    rel_staged = staged_dir.resolve().relative_to(base).as_posix()
    payload = {
        "schema": PENDING_UPDATE_SCHEMA,
        "from_version": from_version,
        "target_version": info.version,
        "tag": info.tag,
        "staged_dir": rel_staged,
        "zip_sha256": zip_sha256,
        "manifest_verified": bool(manifest_verified),
        "prepared_at": _utc_now(),
    }
    target = updates_dir / PENDING_UPDATE_FILE_NAME
    tmp = updates_dir / f"{PENDING_UPDATE_FILE_NAME}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


_exit_callback_lock = threading.Lock()
_update_exit_callback: Callable[[], None] | None = None


def register_update_exit_callback(callback: Callable[[], None] | None) -> None:
    """Registered by ``launcher.py`` so Web UI can trigger a clean exit(42)."""
    global _update_exit_callback
    with _exit_callback_lock:
        _update_exit_callback = callback


def request_update_restart() -> bool:
    with _exit_callback_lock:
        callback = _update_exit_callback
    if callback is None:
        return False
    callback()
    return True


class UpdateManager:
    """In-process singleton tracking update detection, download, and staging."""

    def __init__(self, *, root: Path | None = None) -> None:
        self._root_override = root
        self._lock = threading.Lock()
        self._state = "idle"
        self._latest: ReleaseUpdateInfo | None = None
        self._downloaded_bytes = 0
        self._total_bytes = 0
        self._error = ""
        self._checked_at = ""
        self._cancel_event: threading.Event | None = None
        self._worker: threading.Thread | None = None
        self._check_thread: threading.Thread | None = None
        self._check_generation = 0

    def _root(self) -> Path:
        return _project_root(self._root_override)

    def snapshot(self) -> dict[str, Any]:
        base = self._root()
        current_ver = current_app_version(base)
        current_tuple = parse_semver(current_ver) or (0, 0, 0)
        allowed, reason = can_self_update(base)
        with self._lock:
            latest = self._latest
            update_available = bool(latest is not None and latest.version_tuple > current_tuple)
            total = self._total_bytes or (latest.archive_size if latest else 0)
            downloaded = self._downloaded_bytes
            percent = int(min(100, (downloaded * 100) // total)) if total > 0 else 0
            return {
                "current_version": current_ver,
                "auto_check_enabled": auto_update_check_enabled(),
                "can_self_update": allowed,
                "cannot_update_reason": reason,
                "update_available": update_available,
                "state": self._state,
                "latest_version": latest.version if latest else "",
                "tag": latest.tag if latest else "",
                "release_url": latest.release_url if latest else "",
                "release_notes": latest.release_notes if latest else "",
                "published_at": latest.published_at if latest else "",
                "asset_size": latest.archive_size if latest else 0,
                "downloaded_bytes": downloaded,
                "total_bytes": total,
                "percent": percent,
                "error": self._error,
                "checked_at": self._checked_at,
            }

    def is_update_in_progress(self) -> bool:
        with self._lock:
            return self._state in _UPDATE_IN_PROGRESS_STATES

    def start_background_check(self) -> bool:
        with self._lock:
            if self._state == "checking" or self._state in _UPDATE_IN_PROGRESS_STATES:
                return False
            self._check_generation += 1
            generation = self._check_generation
            self._state = "checking"
            self._error = ""
            thread = threading.Thread(
                target=self._run_check,
                kwargs={"silent": True, "generation": generation},
                name="update-check",
                daemon=True,
            )
            self._check_thread = thread
        thread.start()
        return True

    def check_now(self) -> dict[str, Any]:
        with self._lock:
            if self._state in _UPDATE_IN_PROGRESS_STATES:
                in_progress = True
                generation = 0
            else:
                in_progress = False
                self._check_generation += 1
                generation = self._check_generation
                self._state = "checking"
                self._error = ""
        if in_progress:
            return self.snapshot()
        self._run_check(silent=False, generation=generation)
        return self.snapshot()

    def _run_check(self, *, silent: bool, generation: int) -> None:
        base = self._root()
        current_ver = current_app_version(base)
        url = update_api_url()
        try:
            with _http_client(_CHECK_TIMEOUT_S) as client:
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()
            is_newer, info = select_release_update(payload, current_version=current_ver)
            with self._lock:
                if generation != self._check_generation:
                    return
                self._checked_at = _utc_now()
                self._latest = info
                self._error = ""
                self._state = "available" if is_newer and info is not None else "up_to_date"
        except Exception as exc:  # noqa: BLE001 - network failures must never crash startup
            log.info("Update check failed (%s): %s", url, exc)
            with self._lock:
                if generation != self._check_generation:
                    return
                self._checked_at = _utc_now()
                if silent:
                    # Silent startup check failure stays unobtrusive (idle).
                    self._state = "idle"
                    self._error = ""
                else:
                    self._state = "error"
                    self._error = f"检查更新失败：{exc}"

    def start_update(self, *, active_jobs_count: int = 0) -> dict[str, Any]:
        base = self._root()
        allowed, reason = can_self_update(base)
        if not allowed:
            raise UpdateError(reason)
        if active_jobs_count > 0:
            raise UpdateError("当前有正在运行、排队或保存中的任务，请等待任务结束或取消后再升级。")
        if other_launcher_instances_running(base):
            raise UpdateError("检测到同目录还有其他 jav-sub 窗口正在运行，请先关闭其他窗口后再升级。")

        worker: threading.Thread | None = None
        with self._lock:
            if self._state in _UPDATE_IN_PROGRESS_STATES:
                already_running = True
            else:
                already_running = False
                info = self._latest
                if info is None or self._state not in {"available", "error"}:
                    raise UpdateError("尚未检测到可安装的新版本，请先点击「检查更新」。")
                local_tuple = parse_semver(current_app_version(base)) or (0, 0, 0)
                if info.version_tuple <= local_tuple:
                    raise UpdateError("当前已是最新版本。")
                cancel_event = threading.Event()
                self._cancel_event = cancel_event
                self._state = "downloading"
                self._downloaded_bytes = 0
                self._total_bytes = info.archive_size
                self._error = ""
                worker = threading.Thread(
                    target=self._run_download_and_stage,
                    args=(base, info, cancel_event),
                    name="update-download",
                    daemon=True,
                )
                self._worker = worker
        if already_running:
            return self.snapshot()
        assert worker is not None
        worker.start()
        return self.snapshot()

    def cancel_update(self) -> dict[str, Any]:
        with self._lock:
            if self._cancel_event is not None and self._state in {"downloading", "verifying"}:
                self._cancel_event.set()
        return self.snapshot()

    def _run_download_and_stage(
        self,
        base: Path,
        info: ReleaseUpdateInfo,
        cancel_event: threading.Event,
    ) -> None:
        from_version = current_app_version(base)
        version_dir = base / "tmp" / "updates" / f"v{info.version}-{uuid.uuid4().hex[:8]}"
        archive_part = version_dir / f"{UPDATE_ARCHIVE_NAME}.part"
        archive_path = version_dir / UPDATE_ARCHIVE_NAME
        extract_dir = version_dir / "extract"
        staged_dir = version_dir / "staged"
        try:
            version_dir.mkdir(parents=True, exist_ok=True)
            with _http_client(_DOWNLOAD_TIMEOUT_S) as client:
                expected_sha = info.asset_sha256
                if info.sha256_url:
                    sha_resp = client.get(info.sha256_url)
                    sha_resp.raise_for_status()
                    sidecar_sha = parse_sha256_sidecar(sha_resp.text, UPDATE_ARCHIVE_NAME)
                    if not sidecar_sha:
                        raise UpdateError("无法解析安装包 SHA-256 校验文件，已中止升级。")
                    if expected_sha and expected_sha != sidecar_sha:
                        raise UpdateError("安装包摘要与 .sha256 文件不一致，已中止升级。")
                    expected_sha = sidecar_sha
                if not expected_sha or not _SHA256_RE.fullmatch(expected_sha):
                    raise UpdateError("缺少有效的安装包 SHA-256 校验摘要，已拒绝下载。")

                if not info.manifest_url or not _is_allowed_download_url(info.manifest_url):
                    raise UpdateError(f"缺少有效的发行清单（{UPDATE_MANIFEST_NAME}），已拒绝下载。")
                m_resp = client.get(info.manifest_url)
                m_resp.raise_for_status()
                manifest_payload = m_resp.json()

                if cancel_event.is_set():
                    raise UpdateCancelledError("已取消更新下载。")

                digest = hashlib.sha256()
                downloaded = 0
                with client.stream("GET", info.archive_url) as stream:
                    stream.raise_for_status()
                    try:
                        header_total = int(stream.headers.get("Content-Length") or 0)
                    except ValueError:
                        header_total = 0
                    if header_total > 0:
                        with self._lock:
                            self._total_bytes = header_total
                    with archive_part.open("wb") as out:
                        for chunk in stream.iter_bytes(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                            if cancel_event.is_set():
                                raise UpdateCancelledError("已取消更新下载。")
                            if not chunk:
                                continue
                            out.write(chunk)
                            digest.update(chunk)
                            downloaded += len(chunk)
                            with self._lock:
                                self._downloaded_bytes = downloaded

                actual_sha = digest.hexdigest()
                if actual_sha != expected_sha:
                    raise UpdateError("安装包 SHA-256 校验失败，文件可能已损坏，请重试。")
                archive_part.replace(archive_path)

            with self._lock:
                if cancel_event.is_set():
                    raise UpdateCancelledError("已取消更新。")
                self._state = "verifying"

            extracted_payload = safe_extract_zip(
                archive_path,
                extract_dir,
                cancel_event=cancel_event,
            )
            if cancel_event.is_set():
                raise UpdateCancelledError("已取消更新。")
            if extracted_payload != staged_dir:
                shutil.move(str(extracted_payload), str(staged_dir))
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)

            manifest_verified = verify_staged_payload(
                staged_dir,
                expected_version=info.version,
                manifest_payload=manifest_payload,
                cancel_event=cancel_event,
            )
            # Atomically check cancellation and cross the point of no return
            # before writing pending-update.json so a concurrent cancel_update()
            # either aborts cleanly (before this lock) or sees "restarting".
            with self._lock:
                if cancel_event.is_set():
                    raise UpdateCancelledError("已取消更新。")
                self._state = "restarting"
                self._error = ""

            archive_path.unlink(missing_ok=True)
            write_pending_update_descriptor(
                base,
                from_version=from_version,
                info=info,
                staged_dir=staged_dir,
                zip_sha256=actual_sha,
                manifest_verified=manifest_verified,
            )
            if not request_update_restart():
                with self._lock:
                    self._state = "ready_to_restart"
        except UpdateCancelledError:
            shutil.rmtree(version_dir, ignore_errors=True)
            with self._lock:
                self._state = "available"
                self._downloaded_bytes = 0
                self._error = ""
        except Exception as exc:  # noqa: BLE001 - report clean error to UI
            log.exception("Self-update download/verification failed")
            shutil.rmtree(version_dir, ignore_errors=True)
            with self._lock:
                self._state = "error"
                self._downloaded_bytes = 0
                self._error = str(exc)


_manager = UpdateManager()


def get_update_manager() -> UpdateManager:
    return _manager


def is_update_in_progress() -> bool:
    return get_update_manager().is_update_in_progress()
