# -*- mode: python ; coding: utf-8 -*-
"""Build the release entry point as a single console executable.

Deliberately the opposite of jav-trans-web.spec: that one freezes the whole
environment, this one freezes nothing but `bootstrap.py` and lets uv build a
real virtual environment at the user's machine. A PyInstaller bundle is not an
installable environment - you cannot `uv sync` into it - so the installer has to
stay outside the thing it installs.

Named `jav-trans`, not `jav-trans-setup`: installing is only what the first run
does, and every run after that launches the app. Both specs therefore produce a
`jav-trans` name - build_setup.ps1 passes its own --workpath so the two builds
cannot share PyInstaller's analysis cache, and onefile writes `dist/jav-trans.exe`
next to (not into) the full bundle's `dist/jav-trans/`.

`console=True` is the feature, not an oversight: the reason for shipping this
rather than a frozen bundle is that the user can watch uv download torch and
decide whether they need a proxy.
"""

from pathlib import Path


block_cipher = None
ROOT = Path(SPECPATH).resolve().parent


def _icon_path() -> str | None:
    candidate = ROOT / "src" / "assets" / "images" / "icon.ico"
    return str(candidate) if candidate.is_file() else None


# bootstrap.py imports nothing outside the standard library, so anything heavy
# appearing here would mean that invariant has been broken. Excluding them keeps
# the installer small and turns a violation into a build-time ImportError.
excludes = [
    "torch",
    "torchcodec",
    "transformers",
    "numpy",
    "scipy",
    "librosa",
    "sklearn",
    "PIL",
    "fastapi",
    "uvicorn",
    "webview",
    "huggingface_hub",
    "openai",
    "httpx",
    "tkinter",
    "unittest",
    "pydoc_data",
]

a = Analysis(
    [str(ROOT / "bootstrap.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="jav-trans",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    icon=_icon_path(),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
