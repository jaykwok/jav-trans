from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from starlette.responses import FileResponse

from core.diagnostics import format_exit_code
from pipeline import artifact_store
from utils.model_paths import PROJECT_ROOT
from utils.subprocess_tools import no_window_subprocess_kwargs
from web.pipeline_manager import get_job, output_lease_for_job
from web.models import JobState


log = logging.getLogger(__name__)
router = APIRouter()


class _PickerError(RuntimeError):
    """The OS file/folder picker failed (distinct from user cancellation)."""

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".ts", ".wmv", ".m2ts"}


def _safe_name(name: str) -> str:
    candidate = Path(name or "upload").name
    return candidate or "upload"


def _resolve_output_base(output_dir: str | None) -> Path:
    if not output_dir:
        return PROJECT_ROOT
    base = Path(output_dir).expanduser()
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    return base.resolve()


def _authorized_artifact_roots(job: JobState) -> list[Path]:
    """Return every directory where this job may legitimately write artifacts.

    An explicit output directory is authoritative.  When it is omitted, the
    pipeline writes beside each source video, which may be outside the project
    tree (the normal desktop-app case).
    """
    roots = [PROJECT_ROOT.resolve()]
    if job.spec.output_dir:
        roots.append(_resolve_output_base(job.spec.output_dir))
    else:
        for raw_video_path in job.spec.video_paths:
            video_path = _resolve_existing_video_path(raw_video_path)
            if video_path is not None:
                roots.append(video_path.parent.resolve())

    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _resolve_existing_video_path(raw_path: str) -> Path | None:
    try:
        path = Path(raw_path).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    if path.exists() and path.suffix.lower() in _VIDEO_SUFFIXES:
        return path
    return None


def _normalize_video_paths(paths: list[str]) -> list[str]:
    result: list[str] = []
    for raw_path in paths:
        path = _resolve_existing_video_path(raw_path)
        if path is not None:
            result.append(str(path))
    return result


def _scan_video_folder(raw_folder: str) -> list[str]:
    try:
        folder = Path(raw_folder).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        log.warning("Could not resolve video folder %r: %s", raw_folder, exc)
        return []
    if not folder.is_dir():
        return []
    paths: list[str] = []

    def _on_walk_error(exc: OSError) -> None:
        log.warning("Skipping unreadable directory during folder scan: %s", exc)

    for dirpath, _dirnames, filenames in os.walk(folder, onerror=_on_walk_error):
        for filename in filenames:
            candidate = Path(dirpath) / filename
            if candidate.suffix.lower() not in _VIDEO_SUFFIXES:
                continue
            try:
                if candidate.is_file():
                    paths.append(str(candidate.resolve()))
            except OSError as exc:
                log.warning("Skipping unreadable file %s: %s", candidate, exc)
    return sorted(paths)


def _artifact_path_candidates(job: JobState, raw_path: str) -> list[Path]:
    raw_text = str(raw_path or "").strip()
    if not raw_text:
        return []
    requested_name = Path(raw_text).name
    allowed_roots = _authorized_artifact_roots(job)
    candidates: list[Path] = []
    for artifact in job.artifacts:
        artifact_text = str(artifact)
        artifact_path = Path(artifact_text)
        if raw_text not in {artifact_text, artifact_path.name}:
            continue
        if artifact_path.is_absolute():
            candidates.append(artifact_path)
        else:
            candidates.extend(root / artifact_path for root in allowed_roots)
    if requested_name:
        for artifact in job.artifacts:
            artifact_path = Path(str(artifact))
            if artifact_path.name != requested_name:
                continue
            if artifact_path.is_absolute():
                candidates.append(artifact_path)
            else:
                candidates.extend(root / artifact_path for root in allowed_roots)
    return candidates


def _resolve_authorized_artifact_path(job: JobState, raw_path: str) -> Path | None:
    lease = output_lease_for_job(job)
    if job.output_generation:
        # New jobs read only committed immutable files. There is no fallback to
        # a mutable export if the manifest is missing or corrupt.
        manifest = artifact_store.read_run_manifest(lease, verify=True) if lease else None
        if manifest is None:
            return None
        requested = Path(raw_path).name
        candidates = [
            path for path in artifact_store.manifest_paths(manifest, project_root=PROJECT_ROOT)
            if Path(path).name == requested
        ]
        return artifact_store.authorized_path(
            candidates, [artifact_store.output_store_root(lease.target, project_root=PROJECT_ROOT)],
        )
    # Which paths this job could plausibly mean is a question about the job;
    # whether the answer is allowed out of the process is a question about
    # artifacts, and `pipeline.artifact_store` is the one place that answers it -
    # so opening and downloading cannot drift into two different rules again.
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


def _pick_files_windows() -> list[str]:
    script = r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Filter = 'Video files (*.mp4;*.mkv;*.avi;*.ts;*.wmv)|*.mp4;*.mkv;*.avi;*.ts;*.wmv|All files (*.*)|*.*'
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
    script = r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '%s'
$dialog.ShowNewFolderButton = $false
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::WriteLine($dialog.SelectedPath)
}
""" % description
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
        'of type {"mp4", "mkv", "avi", "ts", "wmv"}\n'
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
    script = f'POSIX path of (choose folder with prompt "{description}")'
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
            "--file-filter=Video files | *.mp4 *.mkv *.avi *.ts *.wmv",
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
    completed = subprocess.run(
        ["zenity", "--file-selection", "--directory", f"--title={description}"],
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


def _pick_video_folder_blocking() -> list[str]:
    folder = _pick_folder_blocking()
    if not folder:
        return []
    return _scan_video_folder(folder)


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


@router.post("/pick-files")
async def pick_files() -> dict[str, list[str]]:
    loop = asyncio.get_running_loop()
    try:
        paths = await loop.run_in_executor(None, _pick_video_files_blocking)
    except _PickerError as exc:
        raise HTTPException(status_code=500, detail=f"打开文件选择器失败：{exc}") from exc
    return {"paths": paths}


@router.post("/pick-folder")
async def pick_folder() -> dict[str, list[str]]:
    loop = asyncio.get_running_loop()
    try:
        paths = await loop.run_in_executor(None, _pick_video_folder_blocking)
    except _PickerError as exc:
        raise HTTPException(status_code=500, detail=f"打开文件夹选择器失败：{exc}") from exc
    return {"paths": paths}


@router.post("/open-video")
async def open_video(job_id: str, path: str) -> dict[str, bool]:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    resolved = _resolve_existing_video_path(path)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Video file not found")
    if not _is_job_video_path(job, resolved):
        raise HTTPException(status_code=403, detail="Video path is not part of this job")
    _open_path_with_os(resolved, label="视频文件")
    return {"ok": True}


@router.post("/open-folder")
async def open_folder(job_id: str, path: str) -> dict[str, bool]:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    target = None
    if job.output_target:
        exported = artifact_store.artifact_path(job.output_target, PROJECT_ROOT)
        requested = artifact_store.artifact_path(path, PROJECT_ROOT)
        if requested == exported and exported.parent.is_dir():
            target = artifact_store.authorized_path([exported.parent], _authorized_artifact_roots(job))
    if target is None:
        target = await asyncio.to_thread(_resolve_authorized_artifact_path, job, path)
    if target is None:
        video_path = _resolve_existing_video_path(path)
        if video_path is not None and _is_job_video_path(job, video_path):
            target = video_path
    if target is None:
        raise HTTPException(status_code=403, detail="Path is not part of this job")
    folder = target if target.is_dir() else target.parent
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")
    _open_path_with_os(folder, label="文件夹")
    return {"ok": True}


@router.post("/open-artifact")
async def open_artifact(job_id: str, path: str) -> dict[str, bool]:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    target = await asyncio.to_thread(_resolve_authorized_artifact_path, job, path)
    if target is None:
        raise HTTPException(status_code=403, detail="Artifact is not part of this job")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    _open_path_with_os(target, label="产物文件")
    return {"ok": True}


@router.get("/output/{job_id}/{filename}")
async def get_output_file(job_id: str, filename: str) -> FileResponse:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # The same resolver the open routes use, deliberately. Downloading had its
    # own pair of roots - the job's output_dir and the project - which omitted
    # the one the pipeline actually writes to when no output_dir is set: the
    # directory beside each source video. A film outside the project therefore
    # produced subtitles the page would open and refuse to download. Two rules
    # for one authorisation question is one rule too many.
    requested_name = _safe_name(filename)
    target = await asyncio.to_thread(_resolve_authorized_artifact_path, job, requested_name)
    if target is None or not target.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(target, filename=requested_name)
