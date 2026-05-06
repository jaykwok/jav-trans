# -*- mode: python ; coding: utf-8 -*-

import os
import shutil
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


block_cipher = None
ROOT = Path(SPECPATH).resolve().parent


def _require_path(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise SystemExit(f"{label} not found: {candidate}")
    return candidate


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
datas += [
    (str(_require_path("src/web/static", "web static assets")), "src/web/static"),
    (str(_require_path("icon.ico", "application icon")), "."),
    (str(_require_path("icon.png", "application png icon")), "."),
    (
        str(_require_path("models/efwkjn-whisper-ja-anime-v0.3", "default ASR model")),
        "models/efwkjn-whisper-ja-anime-v0.3",
    ),
]

binaries = _ffmpeg_binaries()
binaries += collect_dynamic_libs("onnxruntime")

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
    "onnxruntime",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "openai",
    "qwen_asr",
    "soundfile",
    "torch",
    "torchaudio",
    "transformers",
]
hiddenimports += collect_submodules(
    "webview",
    filter=lambda name: not name.startswith("webview.platforms.android"),
)

a = Analysis(
    [str(ROOT / "run_web.py")],
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
    name="JAVTrans",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ROOT / "icon.ico"),
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
