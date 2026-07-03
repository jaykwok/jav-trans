from __future__ import annotations

import os
from pathlib import Path

from utils.runtime_paths import prepend_to_path, resource_root, runtime_root


_DLL_DIRECTORY_HANDLES: list[object] = []
_CONFIGURED_DIRECTORIES: tuple[str, ...] | None = None


def _is_shared_ffmpeg_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    required_patterns = ("avcodec-*.dll", "avformat-*.dll", "avutil-*.dll")
    return all(any(path.glob(pattern)) for pattern in required_patterns)


def _candidate_directories() -> list[Path]:
    candidates: list[Path] = []

    for name in (
        "JAV_TRANS_FFMPEG_SHARED_DIR",
        "JAV_TRANS_FFMPEG_EXE",
        "JAV_TRANS_FFPROBE_EXE",
    ):
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        candidates.append(
            path.parent
            if path.name.lower() in {"ffmpeg.exe", "ffprobe.exe", "ffmpeg", "ffprobe"}
            else path
        )

    candidates.extend((resource_root() / "bin", runtime_root() / "bin"))

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry.strip():
            candidates.append(Path(entry.strip().strip('"')))

    if os.name == "nt":
        local_app_data = os.getenv("LOCALAPPDATA", "").strip()
        if local_app_data:
            packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            if packages.is_dir():
                for package in packages.glob("Gyan.FFmpeg.Shared_*"):
                    candidates.extend(package.glob("ffmpeg-*-shared/bin"))

    return candidates


def configure_ffmpeg_shared_runtime() -> tuple[str, ...]:
    """Register FFmpeg shared DLL directories for TorchCodec on Windows."""
    global _CONFIGURED_DIRECTORIES
    if _CONFIGURED_DIRECTORIES is not None:
        return _CONFIGURED_DIRECTORIES

    configured: list[str] = []
    seen: set[str] = set()
    for candidate in _candidate_directories():
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key in seen or not _is_shared_ffmpeg_directory(resolved):
            continue
        seen.add(key)
        prepend_to_path(resolved)
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            try:
                # The returned handle must stay alive for the registration to
                # remain active, including while TorchCodec loads dependencies.
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(resolved)))
            except OSError:
                continue
        configured.append(str(resolved))

    _CONFIGURED_DIRECTORIES = tuple(configured)
    return _CONFIGURED_DIRECTORIES
