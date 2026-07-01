from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from starlette.responses import FileResponse

from utils.model_paths import PROJECT_ROOT
from web.pipeline_manager import get_job
from web.models import JobState


router = APIRouter()

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
    except (OSError, RuntimeError):
        return []
    if not folder.is_dir():
        return []
    paths: list[str] = []
    try:
        for candidate in folder.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in _VIDEO_SUFFIXES:
                paths.append(str(candidate.resolve()))
    except OSError:
        return paths
    return sorted(paths)


def _artifact_path_candidates(job: JobState, raw_path: str) -> list[Path]:
    raw_text = str(raw_path or "").strip()
    if not raw_text:
        return []
    requested_name = Path(raw_text).name
    output_base = _resolve_output_base(job.spec.output_dir)
    candidates: list[Path] = []
    for artifact in job.artifacts:
        artifact_text = str(artifact)
        artifact_path = Path(artifact_text)
        if raw_text not in {artifact_text, artifact_path.name}:
            continue
        if artifact_path.is_absolute():
            candidates.append(artifact_path)
        else:
            candidates.append(output_base / artifact_path)
            candidates.append(PROJECT_ROOT / artifact_path)
    if requested_name:
        for artifact in job.artifacts:
            artifact_path = Path(str(artifact))
            if artifact_path.name != requested_name:
                continue
            if artifact_path.is_absolute():
                candidates.append(artifact_path)
            else:
                candidates.append(output_base / artifact_path)
                candidates.append(PROJECT_ROOT / artifact_path)
    return candidates


def _resolve_authorized_artifact_path(job: JobState, raw_path: str) -> Path | None:
    for candidate in _artifact_path_candidates(job, raw_path):
        try:
            resolved = candidate.expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if resolved.exists():
            return resolved
    return None


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
    )
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _pick_folder_windows() -> str:
    script = r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '选择视频文件夹'
$dialog.ShowNewFolderButton = $false
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::WriteLine($dialog.SelectedPath)
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
    )
    if completed.returncode != 0:
        return ""
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
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _pick_folder_macos() -> str:
    script = "POSIX path of (choose folder with prompt \"选择视频文件夹\")"
    completed = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _pick_files_linux() -> list[str]:
    if shutil.which("zenity") is None:
        return []
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
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _pick_folder_linux() -> str:
    if shutil.which("zenity") is None:
        return ""
    completed = subprocess.run(
        ["zenity", "--file-selection", "--directory"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _pick_video_files_blocking() -> list[str]:
    try:
        if os.name == "nt":
            paths = _pick_files_windows()
        elif sys.platform == "darwin":
            paths = _pick_files_macos()
        else:
            paths = _pick_files_linux()
    except Exception:
        return []
    return _normalize_video_paths(paths)


def _pick_video_folder_blocking() -> list[str]:
    try:
        if os.name == "nt":
            folder = _pick_folder_windows()
        elif sys.platform == "darwin":
            folder = _pick_folder_macos()
        else:
            folder = _pick_folder_linux()
    except Exception:
        return []
    if not folder:
        return []
    return _scan_video_folder(folder)


@router.post("/pick-files")
async def pick_files() -> dict[str, list[str]]:
    loop = asyncio.get_running_loop()
    paths = await loop.run_in_executor(None, _pick_video_files_blocking)
    return {"paths": paths}


@router.post("/pick-folder")
async def pick_folder() -> dict[str, list[str]]:
    loop = asyncio.get_running_loop()
    paths = await loop.run_in_executor(None, _pick_video_folder_blocking)
    return {"paths": paths}


@router.get("/open-video")
async def open_video(job_id: str, path: str) -> dict[str, bool]:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    resolved = _resolve_existing_video_path(path)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Video file not found")
    if not _is_job_video_path(job, resolved):
        raise HTTPException(status_code=403, detail="Video path is not part of this job")
    if os.name == "nt":
        os.startfile(str(resolved))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(resolved)])
    return {"ok": True}


@router.get("/open-folder")
async def open_folder(job_id: str, path: str) -> dict[str, bool]:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    target = _resolve_authorized_artifact_path(job, path)
    if target is None:
        video_path = _resolve_existing_video_path(path)
        if video_path is not None and _is_job_video_path(job, video_path):
            target = video_path
    if target is None:
        raise HTTPException(status_code=403, detail="Path is not part of this job")
    folder = target if target.is_dir() else target.parent
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")
    if os.name == "nt":
        os.startfile(str(folder))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])
    return {"ok": True}


@router.get("/output/{job_id}/{filename}")
async def get_output_file(job_id: str, filename: str) -> FileResponse:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    requested_name = _safe_name(filename)
    output_base = _resolve_output_base(job.spec.output_dir)
    candidates: list[Path] = []
    for artifact in job.artifacts:
        artifact_path = Path(str(artifact))
        if artifact_path.name == requested_name:
            if artifact_path.is_absolute():
                candidates.append(artifact_path)
            else:
                candidates.append(output_base / artifact_path)
                candidates.append(PROJECT_ROOT / artifact_path)

    output_root = output_base.resolve()
    project_root = PROJECT_ROOT.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        # Guard against path traversal: only serve files inside the job's output
        # base or the project root, even if a persisted jobs.json artifact entry
        # was tampered to point elsewhere.
        try:
            resolved.relative_to(output_root)
        except ValueError:
            try:
                resolved.relative_to(project_root)
            except ValueError:
                continue
        return FileResponse(resolved, filename=requested_name)

    raise HTTPException(status_code=404, detail="Output file not found")
