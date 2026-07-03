# -*- mode: python ; coding: utf-8 -*-

import os
import shutil
import importlib.util
import fnmatch
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


INFERENCE_IGNORE_PATTERNS = [
    "optimizer.pt",
    "**/optimizer.pt",
    "optimizer.bin",
    "**/optimizer.bin",
    "scheduler.pt",
    "**/scheduler.pt",
    "scaler.pt",
    "**/scaler.pt",
    "rng_state*.pth",
    "**/rng_state*.pth",
    "trainer_state.json",
    "**/trainer_state.json",
    "training_args.bin",
    "**/training_args.bin",
]


def _ignored_inference_file(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    name = Path(normalized).name
    return any(
        fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(name, pattern)
        for pattern in INFERENCE_IGNORE_PATTERNS
    )


def _collect_inference_model_dir(path: str, dest: str, label: str) -> list[tuple[str, str]]:
    source = _require_path(path, label)
    if not source.is_dir():
        raise SystemExit(f"{label} must be a directory: {source}")

    collected = []
    for file_path in source.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(source).as_posix()
        if _ignored_inference_file(relative):
            continue
        relative_parent = Path(relative).parent.as_posix()
        target_dir = dest if relative_parent == "." else (Path(dest) / relative_parent).as_posix()
        collected.append((str(file_path), target_dir))

    if not collected:
        raise SystemExit(f"{label} has no inference files to bundle: {source}")
    return collected


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
datas += copy_metadata("torchcodec")
datas += [
    (str(_require_path("src/web/static", "web static assets")), "src/web/static"),
    (str(_require_path("src/assets", "application assets")), "src/assets"),
    (
        str(_require_path("src/boundary/checkpoints", "Boundary Refiner checkpoints")),
        "src/boundary/checkpoints",
    ),
    (
        str(_require_path("src/boundary/ja/checkpoints", "SpeechBoundary-JA scorer checkpoints")),
        "src/boundary/ja/checkpoints",
    ),
    (
        str(_require_path("src/asr/checkpoints", "Pre-ASR CueQC checkpoints")),
        "src/asr/checkpoints",
    ),
]

if not _env_bool("JAV_TRANS_SKIP_MODELS"):
    datas += _collect_inference_model_dir(
        "models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
        "models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
        "bundled default 1.7B ASR / SpeechBoundary model",
    )
    datas += _collect_inference_model_dir(
        "models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf",
        "models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf",
        "bundled low-config 0.6B ASR / SpeechBoundary model",
    )

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
