from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from starlette.responses import FileResponse

from pipeline import artifact_store
from utils.model_paths import PROJECT_ROOT
from utils.subprocess_tools import no_window_subprocess_kwargs
from web.pipeline_manager import get_job, output_lease_for_job
from web.models import JobState


router = APIRouter()

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".ts", ".wmv", ".m2ts"}
_QUALITY_MARKDOWN_SUFFIX = ".quality_report.md"


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
        return []
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


def _pick_folder_linux(description: str = "选择视频文件夹") -> str:
    if shutil.which("zenity") is None:
        return ""
    completed = subprocess.run(
        ["zenity", "--file-selection", "--directory", f"--title={description}"],
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


def _pick_folder_blocking(description: str = "选择视频文件夹") -> str:
    try:
        if os.name == "nt":
            return _pick_folder_windows(description)
        elif sys.platform == "darwin":
            return _pick_folder_macos(description)
        else:
            return _pick_folder_linux(description)
    except Exception:
        return ""


def _pick_video_folder_blocking() -> list[str]:
    folder = _pick_folder_blocking()
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


@router.post("/pick-directory")
async def pick_directory(description: str = "选择文件夹") -> dict[str, str]:
    """Generic folder picker (unlike /pick-folder, does not scan for videos)."""
    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, _pick_folder_blocking, description)
    return {"path": path}


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
    if os.name == "nt":
        os.startfile(str(resolved))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(resolved)])
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
    if os.name == "nt":
        os.startfile(str(folder))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])
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
    if os.name == "nt":
        os.startfile(str(target))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])
    return {"ok": True}


def _quality_report_paths(job: JobState) -> tuple[Path, Path | None] | None:
    """Resolve both representations from the same committed generation."""
    for artifact in job.artifacts:
        name = Path(str(artifact)).name
        if not name.endswith(_QUALITY_MARKDOWN_SUFFIX):
            continue
        markdown_path = _resolve_authorized_artifact_path(job, name)
        if markdown_path is None:
            continue
        if job.output_generation:
            json_path = _resolve_authorized_artifact_path(job, Path(name).with_suffix(".json").name)
            return markdown_path, json_path
        json_path = markdown_path.with_suffix(".json")
        return markdown_path, json_path if json_path.is_file() else None
    return None


@router.get("/quality/{job_id}")
async def get_quality_report(job_id: str) -> dict:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    found = await asyncio.to_thread(_quality_report_paths, job)
    if found is None:
        # Not an error: the report is opt-in (QUALITY_REPORT_ENABLED / the
        # 「生成质量报告」 toggle), so its absence is the normal case and the page
        # says so rather than showing a failed request.
        return {"available": False, "reason": "not_generated"}

    markdown_path, json_path = found
    stem = markdown_path.name[: -len(_QUALITY_MARKDOWN_SUFFIX)]
    if json_path is None:
        # The Markdown is still openable; only the structured view is missing.
        return {
            "available": False,
            "reason": "markdown_only",
            "stem": stem,
            "markdown_name": markdown_path.name,
        }

    try:
        report = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"读取质量报告失败：{exc}",
        ) from exc

    return {
        "available": True,
        "stem": stem,
        "path": str(json_path),
        "markdown_name": markdown_path.name,
        "report": report,
    }


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
