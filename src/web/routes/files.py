from __future__ import annotations

import asyncio
import ctypes
import ctypes.wintypes
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from fastapi import APIRouter, HTTPException
from starlette.responses import FileResponse

from core.diagnostics import format_exit_code
from pipeline import artifact_store
from utils.model_paths import PROJECT_ROOT
from utils.subprocess_tools import no_window_subprocess_kwargs
from web.models import JobState
from web.pipeline_manager import get_job, output_lease_for_job


log = logging.getLogger(__name__)
router = APIRouter()


class _PickerError(RuntimeError):
    """The OS file/folder picker failed (distinct from user cancellation)."""


_VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".ts", ".wmv", ".m2ts"}
MAX_FOLDER_SCAN_FILES = 500
MAX_FOLDER_SCAN_DEPTH = 12
_picker_locks: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = WeakKeyDictionary()


def _picker_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    return _picker_locks.setdefault(loop, asyncio.Lock())


def _safe_name(name: str) -> str:
    raw = str(name or "")
    if "\x00" in raw:
        return ""
    candidate = Path(raw or "upload").name
    return candidate or "upload"


def _safe_resolve_path(raw_path: str) -> Path | None:
    text = str(raw_path or "")
    if not text or "\x00" in text:
        return None
    try:
        return Path(text).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _resolve_output_base(output_dir: str | None) -> Path | None:
    if not output_dir:
        return PROJECT_ROOT.resolve()
    text = str(output_dir)
    if "\x00" in text:
        return None
    try:
        base = Path(text).expanduser()
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return base.resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _authorized_artifact_roots(job: JobState) -> list[Path]:
    """Return every directory where this job may legitimately write artifacts.

    Explicit ``output_dir`` and declared source-video parent directories come
    *before* ``PROJECT_ROOT`` so ``open-folder`` opens the user's output folder
    even after the original video file has been moved or removed.
    """
    roots: list[Path] = []
    if job.spec.output_dir:
        explicit = _resolve_output_base(job.spec.output_dir)
        if explicit is not None:
            roots.append(explicit)
    else:
        for raw_video_path in job.spec.video_paths:
            video_path = _safe_resolve_path(raw_video_path)
            if video_path is None:
                continue
            try:
                parent = video_path.parent
                if parent.is_dir():
                    roots.append(parent.resolve())
            except (OSError, RuntimeError, ValueError):
                continue

    try:
        roots.append(PROJECT_ROOT.resolve())
    except (OSError, RuntimeError, ValueError):
        pass

    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _resolve_existing_video_path(raw_path: str) -> Path | None:
    path = _safe_resolve_path(raw_path)
    if path is None:
        return None
    try:
        if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES:
            return path
    except (OSError, ValueError):
        return None
    return None


def _normalize_video_paths(paths: list[str]) -> list[str]:
    result: list[str] = []
    for raw_path in paths:
        path = _resolve_existing_video_path(raw_path)
        if path is not None:
            result.append(str(path))
    return result


def _scan_video_folder_with_limit(
    raw_folder: str,
    *,
    max_files: int = MAX_FOLDER_SCAN_FILES,
    max_depth: int = MAX_FOLDER_SCAN_DEPTH,
) -> tuple[list[str], bool]:
    folder = _safe_resolve_path(raw_folder)
    if folder is None:
        log.warning("Could not resolve video folder %r", raw_folder)
        return [], False
    try:
        if not folder.is_dir():
            return [], False
    except (OSError, ValueError) as exc:
        log.warning("Could not inspect video folder %r: %s", raw_folder, exc)
        return [], False

    paths: list[str] = []
    truncated = False

    def _on_walk_error(exc: OSError) -> None:
        log.warning("Skipping unreadable directory during folder scan: %s", exc)

    for dirpath, dirnames, filenames in os.walk(folder, onerror=_on_walk_error, followlinks=False):
        try:
            rel_parts = Path(dirpath).relative_to(folder).parts
            depth = len(rel_parts)
        except ValueError:
            depth = max_depth
        if depth >= max_depth:
            dirnames[:] = []
        else:
            dirnames.sort()

        for filename in sorted(filenames):
            candidate = Path(dirpath) / filename
            if candidate.suffix.lower() not in _VIDEO_SUFFIXES:
                continue
            try:
                if candidate.is_file():
                    if len(paths) >= max_files:
                        truncated = True
                        break
                    paths.append(str(candidate.resolve()))
            except (OSError, RuntimeError, ValueError) as exc:
                log.warning("Skipping unreadable file %s: %s", candidate, exc)
        if truncated:
            break
    return sorted(paths), truncated


def _scan_video_folder(raw_folder: str) -> list[str]:
    paths, _truncated = _scan_video_folder_with_limit(raw_folder)
    return paths


def _artifact_path_candidates(job: JobState, raw_path: str) -> list[Path]:
    raw_text = str(raw_path or "").strip()
    if not raw_text or "\x00" in raw_text:
        return []
    requested_name = Path(raw_text).name
    allowed_roots = _authorized_artifact_roots(job)
    candidates: list[Path] = []
    for artifact in job.artifacts:
        artifact_text = str(artifact)
        if "\x00" in artifact_text:
            continue
        artifact_path = Path(artifact_text)
        if raw_text not in {artifact_text, artifact_path.name}:
            continue
        if artifact_path.is_absolute():
            candidates.append(artifact_path)
        else:
            candidates.extend(root / artifact_path for root in allowed_roots)
    if requested_name:
        for artifact in job.artifacts:
            artifact_text = str(artifact)
            if "\x00" in artifact_text:
                continue
            artifact_path = Path(artifact_text)
            if artifact_path.name != requested_name:
                continue
            if artifact_path.is_absolute():
                candidates.append(artifact_path)
            else:
                candidates.extend(root / artifact_path for root in allowed_roots)
    return candidates


def _resolve_authorized_artifact_path(job: JobState, raw_path: str) -> Path | None:
    if not raw_path or "\x00" in str(raw_path):
        return None
    lease = output_lease_for_job(job)
    if job.output_generation:
        # New jobs read only committed immutable files. There is no fallback to
        # a mutable export if the manifest is missing or corrupt.
        manifest = artifact_store.read_run_manifest(lease, verify=True) if lease else None
        if manifest is None:
            return None
        requested = Path(raw_path).name
        candidates = [
            path
            for path in artifact_store.manifest_paths(manifest, project_root=PROJECT_ROOT)
            if Path(path).name == requested
        ]
        return artifact_store.authorized_path(
            candidates,
            [artifact_store.output_store_root(lease.target, project_root=PROJECT_ROOT)],
        )
    return artifact_store.authorized_path(
        _artifact_path_candidates(job, raw_path),
        _authorized_artifact_roots(job),
    )


def _is_job_video_path(job: JobState, path: Path) -> bool:
    for raw_video_path in job.spec.video_paths:
        video_path = _resolve_existing_video_path(raw_video_path)
        if video_path is not None and video_path == path:
            return True
    return False


def _sanitize_dialog_prompt(description: str) -> str:
    return (
        str(description or "选择视频文件夹")
        .replace("\x00", "")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
        or "选择视频文件夹"
    )


def _escape_powershell_single_quoted(text: str) -> str:
    return _sanitize_dialog_prompt(text).replace("'", "''")


def _pick_files_windows() -> list[str]:
    script = r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Filter = 'Video files (*.mp4;*.mkv;*.avi;*.ts;*.wmv;*.m2ts)|*.mp4;*.mkv;*.avi;*.ts;*.wmv;*.m2ts|All files (*.*)|*.*'
$dialog.Multiselect = $true
$dialog.CheckFileExists = $true
$dialog.Title = 'Select video files'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    $dialog.FileNames | ForEach-Object { [Console]::WriteLine($_) }
}
"""
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-STA",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **no_window_subprocess_kwargs(),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[:300]
        raise _PickerError(
            f"PowerShell 文件选择器退出码 {format_exit_code(completed.returncode)}"
            + (f"：{detail}" if detail else "")
        )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _pick_folder_windows(description: str = "选择视频文件夹") -> str:
    escaped_description = _escape_powershell_single_quoted(description)
    script = (
        "\n[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n"
        "Add-Type -AssemblyName System.Windows.Forms\n"
        "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog\n"
        f"$dialog.Description = '{escaped_description}'\n"
        "$dialog.ShowNewFolderButton = $false\n"
        "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {\n"
        "    [Console]::WriteLine($dialog.SelectedPath)\n"
        "}\n"
    )
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-STA",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **no_window_subprocess_kwargs(),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[:300]
        raise _PickerError(
            f"PowerShell 文件夹选择器退出码 {format_exit_code(completed.returncode)}"
            + (f"：{detail}" if detail else "")
        )
    return completed.stdout.strip()


def _pick_files_macos() -> list[str]:
    script = (
        'set chosenFiles to choose file with multiple selections allowed '
        'of type {"mp4", "mkv", "avi", "ts", "wmv", "m2ts"}\n'
        'set output to ""\n'
        'repeat with chosenFile in chosenFiles\n'
        'set output to output & POSIX path of chosenFile & linefeed\n'
        'end repeat\n'
        'return output'
    )
    completed = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        if "User canceled" in stderr or "(-128)" in stderr:
            return []
        raise _PickerError(
            f"osascript 文件选择器退出码 {completed.returncode}"
            + (f"：{stderr[:300]}" if stderr else "")
        )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _pick_folder_macos(description: str = "选择视频文件夹") -> str:
    prompt_text = _sanitize_dialog_prompt(description)
    script = (
        "on run argv\n"
        "  set promptText to item 1 of argv\n"
        "  return POSIX path of (choose folder with prompt promptText)\n"
        "end run"
    )
    completed = subprocess.run(
        ["osascript", "-e", script, "--", prompt_text],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        if "User canceled" in stderr or "(-128)" in stderr:
            return ""
        raise _PickerError(
            f"osascript 文件夹选择器退出码 {completed.returncode}"
            + (f"：{stderr[:300]}" if stderr else "")
        )
    return completed.stdout.strip()


def _pick_files_linux() -> list[str]:
    if shutil.which("zenity") is None:
        raise _PickerError("未安装 zenity，无法打开系统文件选择框")
    completed = subprocess.run(
        [
            "zenity",
            "--file-selection",
            "--multiple",
            "--separator=\n",
            "--file-filter=Video files | *.mp4 *.mkv *.avi *.ts *.wmv *.m2ts",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        if completed.returncode == 1 and not stderr:
            return []
        raise _PickerError(
            f"zenity 文件选择器退出码 {completed.returncode}"
            + (f"：{stderr[:300]}" if stderr else "")
        )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _pick_folder_linux(description: str = "选择视频文件夹") -> str:
    if shutil.which("zenity") is None:
        raise _PickerError("未安装 zenity，无法打开系统文件夹选择框")
    prompt_text = _sanitize_dialog_prompt(description)
    completed = subprocess.run(
        ["zenity", "--file-selection", "--directory", f"--title={prompt_text}"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        if completed.returncode == 1 and not stderr:
            return ""
        raise _PickerError(
            f"zenity 文件夹选择器退出码 {completed.returncode}"
            + (f"：{stderr[:300]}" if stderr else "")
        )
    return completed.stdout.strip()


def _pick_video_files_blocking() -> list[str]:
    try:
        if os.name == "nt":
            paths = _pick_files_windows()
        elif sys.platform == "darwin":
            paths = _pick_files_macos()
        else:
            paths = _pick_files_linux()
    except _PickerError:
        log.exception("Video file picker failed")
        raise
    except Exception as exc:
        log.exception("Unexpected error in video file picker")
        raise _PickerError(f"{type(exc).__name__}: {exc}") from exc
    return _normalize_video_paths(paths)


def _pick_folder_blocking(description: str = "选择视频文件夹") -> str:
    try:
        if os.name == "nt":
            return _pick_folder_windows(description)
        elif sys.platform == "darwin":
            return _pick_folder_macos(description)
        else:
            return _pick_folder_linux(description)
    except _PickerError:
        log.exception("Folder picker failed")
        raise
    except Exception as exc:
        log.exception("Unexpected error in folder picker")
        raise _PickerError(f"{type(exc).__name__}: {exc}") from exc


def _pick_video_folder_blocking() -> tuple[list[str], bool]:
    folder = _pick_folder_blocking()
    if not folder:
        return [], False
    return _scan_video_folder_with_limit(folder)


_POTPLAYER_EXES = frozenset(
    {"potplayermini64.exe", "potplayermini.exe", "potplayer64.exe", "potplayer.exe"}
)
_MPV_EXES = frozenset({"mpv.exe", "mpv"})
_VLC_EXES = frozenset({"vlc.exe", "vlc"})
_MPC_EXES = frozenset({"mpc-hc64.exe", "mpc-hc.exe", "mpc-be64.exe", "mpc-be.exe"})
_SMPLAYER_EXES = frozenset({"smplayer.exe", "smplayer"})


def _exe_basename_lower(player_exe: str | Path) -> str:
    text = str(player_exe or "").strip().strip('"')
    if not text:
        return ""
    return text.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _build_player_launch_command(
    player_exe: str | Path | None,
    video_path: str | Path,
    subtitle_path: str | Path | None,
) -> list[str] | None:
    """Build a player command that auto-loads ``subtitle_path`` alongside ``video_path``.

    Returns ``None`` when the player is unknown or no subtitle is available so
    callers smoothly fall back to OS default file association launch.
    """
    if not player_exe or not subtitle_path:
        return None
    exe = str(player_exe).strip().strip('"')
    video = str(video_path or "").strip()
    srt = str(subtitle_path or "").strip()
    if not exe or not video or not srt:
        return None
    name = _exe_basename_lower(exe)
    if name in _POTPLAYER_EXES:
        return [exe, video, f"/sub={srt}"]
    if name in _MPV_EXES or name in _VLC_EXES:
        return [exe, video, f"--sub-file={srt}"]
    if name in _MPC_EXES:
        return [exe, video, "/sub", srt]
    if name in _SMPLAYER_EXES:
        return [exe, "-sub", srt, video]
    return None


def _extract_command_executable(command_template: str) -> str | None:
    raw = str(command_template or "").strip()
    if not raw:
        return None
    if raw.startswith('"'):
        end = raw.find('"', 1)
        if end > 1:
            return raw[1:end].strip() or None
    try:
        tokens = shlex.split(raw, posix=False)
    except ValueError:
        tokens = raw.split()
    if not tokens:
        return None
    return tokens[0].strip().strip('"') or None


def _query_windows_associated_executable(extension: str) -> str | None:
    """Query Windows default executable associated with a video file extension."""
    if os.name != "nt":
        return None
    ext = str(extension or "").strip().lower()
    if not ext:
        return None
    if not ext.startswith("."):
        ext = f".{ext}"

    # 1. Preferred: Win32 shlwapi AssocQueryStringW(ASSOCSTR_EXECUTABLE = 2).
    try:
        shlwapi = ctypes.windll.shlwapi  # type: ignore[attr-defined]
        assoc_query = shlwapi.AssocQueryStringW
        assoc_query.argtypes = [
            ctypes.wintypes.DWORD,
            ctypes.wintypes.DWORD,
            ctypes.wintypes.LPCWSTR,
            ctypes.wintypes.LPCWSTR,
            ctypes.wintypes.LPWSTR,
            ctypes.POINTER(ctypes.wintypes.DWORD),
        ]
        assoc_query.restype = ctypes.c_long
        assocf_none = 0x00000000
        assocstr_executable = 2
        for verb in ("open", None):
            length = ctypes.wintypes.DWORD(0)
            hr = assoc_query(
                assocf_none,
                assocstr_executable,
                ext,
                verb,
                None,
                ctypes.byref(length),
            )
            if length.value > 1:
                buf = ctypes.create_unicode_buffer(length.value)
                hr2 = assoc_query(
                    assocf_none,
                    assocstr_executable,
                    ext,
                    verb,
                    buf,
                    ctypes.byref(length),
                )
                if hr2 == 0 and buf.value.strip():
                    return buf.value.strip().strip('"')
            elif hr == 0:
                continue
    except Exception:
        log.debug("AssocQueryStringW lookup failed for %s", ext, exc_info=True)

    # 2. Fallback: UserChoice ProgId in registry.
    try:
        import winreg

        prog_id = ""
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            rf"Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\{ext}\UserChoice",
        ) as key:
            prog_id, _ = winreg.QueryValueEx(key, "ProgId")
        if prog_id:
            with winreg.OpenKey(
                winreg.HKEY_CLASSES_ROOT,
                rf"{prog_id}\shell\open\command",
            ) as cmd_key:
                cmd_val, _ = winreg.QueryValueEx(cmd_key, "")
                return _extract_command_executable(str(cmd_val))
    except OSError:
        log.debug("Registry player association lookup unavailable for %s", ext)
    except Exception:
        log.debug("Registry player association lookup failed for %s", ext, exc_info=True)
    return None


def _resolve_job_subtitle_path(job: JobState) -> Path | None:
    """Resolve the job's primary exported ``.srt`` file, or authorized ``.srt`` artifact."""
    if job.output_target:
        try:
            exported = artifact_store.artifact_path(job.output_target, PROJECT_ROOT)
            if exported.suffix.lower() == ".srt" and exported.is_file():
                authorized = artifact_store.authorized_path(
                    [exported], _authorized_artifact_roots(job)
                )
                if authorized is not None and authorized.is_file():
                    return authorized
        except (OSError, RuntimeError, ValueError):
            pass

    for artifact in job.artifacts:
        if not str(artifact).lower().endswith(".srt"):
            continue
        try:
            resolved = _resolve_authorized_artifact_path(job, str(artifact))
        except Exception:
            continue
        if resolved is not None and resolved.suffix.lower() == ".srt" and resolved.is_file():
            return resolved
    return None


def _open_path_with_os(target: Path, *, label: str) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except OSError as exc:
        log.exception("Failed to open %s via OS shell: %s", label, target)
        raise HTTPException(
            status_code=500,
            detail=f"无法打开{label}：{exc}",
        ) from exc


def _open_video_with_subtitle(video_path: Path, subtitle_path: Path | None) -> None:
    """Launch video with automatic external subtitle loading when player supports it."""
    if subtitle_path is not None and subtitle_path.is_file() and os.name == "nt":
        player_exe = _query_windows_associated_executable(video_path.suffix)
        cmd = _build_player_launch_command(player_exe, video_path, subtitle_path)
        if cmd is not None:
            try:
                popen_kwargs: dict[str, Any] = {
                    "stdin": subprocess.DEVNULL,
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL,
                    "close_fds": True,
                    "creationflags": (
                        getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                    ),
                }
                subprocess.Popen(cmd, **popen_kwargs)
                return
            except (OSError, ValueError):
                log.warning(
                    "Player launch with subtitle failed (%s); falling back to OS default",
                    cmd[0],
                    exc_info=True,
                )
    _open_path_with_os(video_path, label="视频文件")


def _reveal_path_in_folder_with_os(target: Path) -> None:
    """Open a directory directly, or highlight an existing file inside its folder."""
    if target.is_dir():
        _open_path_with_os(target, label="文件夹")
        return
    if target.is_file():
        if os.name == "nt":
            target_text = str(target)
            if not any(ch in target_text for ch in ('"', "\x00", "\r", "\n")):
                try:
                    # Windows explorer expects `/select,"<path>"` (unquoted switch,
                    # quoted path) so paths containing spaces select the target file
                    # instead of opening Documents. Passing a command string directly
                    # to CreateProcess (shell=False) avoids cmd.exe metacharacters.
                    subprocess.Popen(f'explorer.exe /select,"{target_text}"')
                    return
                except OSError:
                    log.warning(
                        "Failed to reveal %s in Explorer; falling back to parent folder",
                        target,
                        exc_info=True,
                    )
        elif sys.platform == "darwin":
            try:
                subprocess.Popen(["open", "-R", str(target)])
                return
            except OSError:
                log.warning(
                    "Failed to reveal %s in Finder; falling back to parent folder",
                    target,
                    exc_info=True,
                )
    folder = target.parent
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")
    _open_path_with_os(folder, label="文件夹")


def _resolve_open_folder_target_blocking(job: JobState, path: str) -> Path | None:
    if not path or "\x00" in str(path):
        return None
    target: Path | None = None
    if job.output_target:
        try:
            exported = artifact_store.artifact_path(job.output_target, PROJECT_ROOT)
            requested = artifact_store.artifact_path(path, PROJECT_ROOT)
            if requested == exported:
                if exported.is_file():
                    target = artifact_store.authorized_path(
                        [exported], _authorized_artifact_roots(job)
                    )
                if target is None and exported.parent.is_dir():
                    target = artifact_store.authorized_path(
                        [exported.parent], _authorized_artifact_roots(job)
                    )
        except (OSError, RuntimeError, ValueError):
            target = None
    if target is None:
        target = _resolve_authorized_artifact_path(job, path)
    if target is None:
        video_path = _resolve_existing_video_path(path)
        if video_path is not None and _is_job_video_path(job, video_path):
            target = video_path
    return target


@router.post("/pick-files")
async def pick_files() -> dict[str, list[str]]:
    lock = _picker_lock()
    if lock.locked():
        raise HTTPException(status_code=409, detail="文件选择窗口已打开，请先完成当前选择。")
    async with lock:
        loop = asyncio.get_running_loop()
        try:
            paths = await loop.run_in_executor(None, _pick_video_files_blocking)
        except _PickerError as exc:
            raise HTTPException(status_code=500, detail=f"打开文件选择器失败：{exc}") from exc
    return {"paths": paths}


@router.post("/pick-folder")
async def pick_folder() -> dict[str, object]:
    lock = _picker_lock()
    if lock.locked():
        raise HTTPException(status_code=409, detail="文件夹选择窗口已打开，请先完成当前选择。")
    async with lock:
        loop = asyncio.get_running_loop()
        try:
            paths, truncated = await loop.run_in_executor(None, _pick_video_folder_blocking)
        except _PickerError as exc:
            raise HTTPException(status_code=500, detail=f"打开文件夹选择器失败：{exc}") from exc
    return {"paths": paths, "truncated": truncated}


@router.post("/open-video")
async def open_video(job_id: str, path: str) -> dict[str, bool]:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    resolved = await asyncio.to_thread(_resolve_existing_video_path, path)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Video file not found")
    if not await asyncio.to_thread(_is_job_video_path, job, resolved):
        raise HTTPException(status_code=403, detail="Video path is not part of this job")
    subtitle_path = await asyncio.to_thread(_resolve_job_subtitle_path, job)
    await asyncio.to_thread(_open_video_with_subtitle, resolved, subtitle_path)
    return {"ok": True}


@router.post("/open-folder")
async def open_folder(job_id: str, path: str) -> dict[str, bool]:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    target = await asyncio.to_thread(_resolve_open_folder_target_blocking, job, path)
    if target is None:
        raise HTTPException(status_code=403, detail="Path is not part of this job")
    await asyncio.to_thread(_reveal_path_in_folder_with_os, target)
    return {"ok": True}


@router.post("/open-artifact")
async def open_artifact(job_id: str, path: str) -> dict[str, bool]:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    target = await asyncio.to_thread(_resolve_authorized_artifact_path, job, path)
    if target is None:
        raise HTTPException(status_code=403, detail="Artifact is not part of this job")
    if not await asyncio.to_thread(target.is_file):
        raise HTTPException(status_code=404, detail="Artifact file not found")
    await asyncio.to_thread(_open_path_with_os, target, label="产物文件")
    return {"ok": True}


@router.get("/output/{job_id}/{filename}")
async def get_output_file(job_id: str, filename: str) -> FileResponse:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    requested_name = _safe_name(filename)
    if not requested_name:
        raise HTTPException(status_code=404, detail="Output file not found")
    target = await asyncio.to_thread(_resolve_authorized_artifact_path, job, requested_name)
    if target is None or not await asyncio.to_thread(target.is_file):
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(target, filename=requested_name)
