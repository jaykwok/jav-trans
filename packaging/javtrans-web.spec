# -*- mode: python ; coding: utf-8 -*-

import os
import shutil
import importlib.util
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


NAGISA_PACKAGE_DIR = _package_dir("nagisa")
QWEN_ASR_PACKAGE_DIR = _package_dir("qwen_asr")


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
        _which_tool("ffmpeg.exe", "JAVTRANS_FFMPEG_EXE"),
        _which_tool("ffprobe.exe", "JAVTRANS_FFPROBE_EXE"),
    ]
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


datas = collect_data_files("webview", include_py_files=False)
datas += copy_metadata("torchcodec")
datas += copy_metadata("nagisa")
datas += [
    (str(_require_path("src/web/static", "web static assets")), "src/web/static"),
    (str(_require_path("src/assets", "application assets")), "src/assets"),
    (
        str(_require_path("src/boundary/checkpoints", "Boundary Refiner checkpoints")),
        "src/boundary/checkpoints",
    ),
    (
        str(_require_path("models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame", "default ASR model")),
        "models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame",
    ),
    (
        str(_require_path("models/Qwen-Qwen3-ForcedAligner-0.6B", "default forced aligner model")),
        "models/Qwen-Qwen3-ForcedAligner-0.6B",
    ),
    (
        str(NAGISA_PACKAGE_DIR / "data"),
        "nagisa/data",
    ),
    (
        str(QWEN_ASR_PACKAGE_DIR / "inference" / "assets"),
        "qwen_asr/inference/assets",
    ),
]

binaries = _ffmpeg_binaries()

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
    "qwen_asr",
    "mecab_system_eval",
    "model",
    "nagisa",
    "nagisa_utils",
    "prepro",
    "tagger",
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
hiddenimports += collect_submodules("nagisa")
hiddenimports += collect_submodules("qwen_asr")
hiddenimports += collect_submodules("torchcodec")
hiddenimports += collect_submodules("transformers.models.mamba2")
hiddenimports += collect_submodules("transformers.models.qwen2")
hiddenimports += collect_submodules("transformers.models.qwen3")

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT / "src"), str(NAGISA_PACKAGE_DIR)],
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
    name="JAVTrans",
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
    name="JAVTrans",
)
