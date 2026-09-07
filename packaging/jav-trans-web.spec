# -*- mode: python ; coding: utf-8 -*-

import os
import shutil
import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)


block_cipher = None
ROOT = Path(SPECPATH).resolve().parent


def _package_dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"Python package not found: {name}")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def _require_path(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise SystemExit(f"{label} not found: {candidate}")
    return candidate


def _pyinstaller_icon_path() -> Path:
    return _require_path("src/assets/images/icon.ico", "application ico icon")


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _alignment_head_source() -> str:
    """Fetch the CTC alignment head the app will run, at the pinned revision.

    Resolved from ASR_ALIGNMENT_HEAD_PATH's default rather than hardcoded here,
    so the sha the build ships and the sha the source checkout downloads cannot
    drift apart. Bundled because a packaged first run has no network guarantee,
    and without the head the app silently falls back to proportional timing.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from asr.alignment import _parse_hf_reference
    from core.config import DEFAULT_SETTINGS
    from utils.model_paths import require_commit_sha

    reference = DEFAULT_SETTINGS["ASR_ALIGNMENT_HEAD_PATH"].strip()
    if not reference.lower().startswith("hf:"):
        return str(_require_path(reference, "CTC alignment head"))

    from huggingface_hub import hf_hub_download

    repo, revision, filename = _parse_hf_reference(reference)
    return hf_hub_download(repo_id=repo, filename=filename, revision=require_commit_sha(revision or ""))


def _which_tool(name: str, env_name: str) -> Path:
    override = os.getenv(env_name, "").strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        if candidate.exists():
            return candidate
    found = shutil.which(name)
    if not found:
        raise SystemExit(
            f"{name} not found. Put it on PATH or set {env_name} to the full executable path."
        )
    return Path(found).resolve()


def _ffmpeg_binaries() -> list[tuple[str, str]]:
    tools = [
        _which_tool("ffmpeg.exe", "JAV_TRANS_FFMPEG_EXE"),
        _which_tool("ffprobe.exe", "JAV_TRANS_FFPROBE_EXE"),
    ]
    shared_dll_patterns = ("avcodec-*.dll", "avformat-*.dll", "avutil-*.dll")
    if not any(
        all(any(tool.parent.glob(pattern)) for pattern in shared_dll_patterns)
        for tool in tools
    ):
        raise SystemExit(
            "FFmpeg Shared build required: the selected ffmpeg/ffprobe directory "
            "must contain avcodec, avformat, and avutil DLLs. On Windows install "
            "'winget install --id Gyan.FFmpeg.Shared --exact' and remove the "
            "static-only Gyan.FFmpeg package."
        )
    seen = set()
    bundled = []
    for tool in tools:
        key = str(tool).lower()
        if key not in seen:
            seen.add(key)
            bundled.append((str(tool), "bin"))
        for dll in tool.parent.glob("*.dll"):
            dll_key = str(dll.resolve()).lower()
            if dll_key not in seen:
                seen.add(dll_key)
                bundled.append((str(dll.resolve()), "bin"))
    return bundled


def _torchcodec_binaries() -> list[tuple[str, str]]:
    package_root = _package_dir("torchcodec")
    patterns = [
        "libtorchcodec_core*.dll",
        "libtorchcodec_custom_ops*.dll",
        "libtorchcodec_pybind_ops*.pyd",
    ]
    collected = []
    seen = set()
    for pattern in patterns:
        for file_path in package_root.glob(pattern):
            key = str(file_path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            collected.append((str(file_path.resolve()), "torchcodec"))
    if not any("libtorchcodec_custom_ops" in Path(source).name for source, _dest in collected):
        raise SystemExit("torchcodec custom ops DLLs not found; reinstall torchcodec in .venv")
    return collected


datas = collect_data_files("webview", include_py_files=False)
# OpenCC keeps its conversion tables as package data, not as code, so a frozen
# build without these raises at the first 简繁 conversion instead of at import.
datas += collect_data_files("opencc", include_py_files=False)
datas += copy_metadata("torchcodec")
datas += [
    (str(_require_path("src/web/static", "web static assets")), "src/web/static"),
    (str(_require_path("src/assets", "application assets")), "src/assets"),
]

if not _env_bool("JAV_TRANS_SKIP_MODELS"):
    sys.path.insert(0, str(ROOT / "packaging"))
    from prepare_default_model import verified_release_models
    from utils.model_paths import inference_model_files, model_dir_name

    release_models = verified_release_models()
    datas.append((_alignment_head_source(), "models"))
    for source, repo_id, _label in release_models:
        destination = Path("models") / model_dir_name(repo_id)
        for file_path in inference_model_files(source, include_receipt=True):
            relative_parent = file_path.relative_to(source).parent
            datas.append((str(file_path), (destination / relative_parent).as_posix()))

binaries = _ffmpeg_binaries()
binaries += _torchcodec_binaries()

hiddenimports = [
    "uvicorn.lifespan.on",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "fastapi",
    "anyio._backends._asyncio",
    "dotenv",
    "httpx",
    "huggingface_hub",
    "openai",
    "librosa",
    "soundfile",
    "scipy",
    "torch",
    "torchaudio",
    "torchcodec",
    "transformers",
]
hiddenimports += collect_submodules(
    "webview",
    filter=lambda name: not name.startswith("webview.platforms.android"),
)
hiddenimports += collect_submodules("transformers.generation")
hiddenimports += collect_submodules("torchcodec")
hiddenimports += collect_submodules("transformers.models.mamba2")
hiddenimports += collect_submodules("transformers.models.qwen2")
hiddenimports += collect_submodules("transformers.models.qwen3")
hiddenimports += collect_submodules("transformers.models.qwen3_asr")

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="jav-trans",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(_pyinstaller_icon_path()),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="jav-trans",
)
