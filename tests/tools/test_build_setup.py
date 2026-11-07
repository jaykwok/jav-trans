"""Exercise the release assembly script without building or downloading binaries."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


@pytest.mark.skipif(not POWERSHELL, reason="PowerShell is required for the Windows build script")
def test_setup_rebuild_drops_stale_payload_and_bytecode(tmp_path):
    project = tmp_path / "project"
    packaging = project / "packaging"
    packaging.mkdir(parents=True)
    shutil.copy2(ROOT / "packaging/build_setup.ps1", packaging)
    (packaging / "setup-readme.txt").write_text("Release help", encoding="utf-8")
    for name in ("launcher.py", "uv.lock", "pyproject.toml"):
        (project / name).write_text("synthetic input", encoding="utf-8")
    source = project / "src"
    (source / "__pycache__").mkdir(parents=True)
    (source / "__pycache__/module.pyc").write_bytes(b"bytecode")
    (source / "module.py").write_text("value = 1", encoding="utf-8")
    payload = project / "dist/setup-payload/jav-trans"
    (payload / "bin").mkdir(parents=True)
    (payload / "bin/old.dll").write_bytes(b"old dll")
    (payload / "uv.exe").write_bytes(b"old uv")
    tools = project / "test-tools"
    tools.mkdir()
    for name in ("ffmpeg.exe", "ffprobe.exe", "avcodec-1.dll", "avformat-1.dll", "avutil-1.dll"):
        (tools / name).write_bytes(b"synthetic binary")

    wrapper = tmp_path / "build.ps1"
    wrapper.write_text(r'''
$ErrorActionPreference = 'Stop'
function uv {
    if ($args -contains 'PyInstaller') {
        [IO.File]::WriteAllText((Join-Path $PWD 'dist/jav-trans.exe'), 'entry point')
    }
    $global:LASTEXITCODE = 0
}
$Project = Join-Path $PSScriptRoot 'project'
& (Join-Path $Project 'packaging/build_setup.ps1') -SkipArchive `
    -FfmpegExe (Join-Path $Project 'test-tools/ffmpeg.exe') `
    -FfprobeExe (Join-Path $Project 'test-tools/ffprobe.exe')
''', encoding="ascii")
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(wrapper)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (payload / "src/module.py").is_file()
    assert (payload / "bin/avcodec-1.dll").is_file()
    assert not (payload / "src/__pycache__").exists()
    assert not (payload / "uv.exe").exists()
    assert not (payload / "bin/old.dll").exists()
    archived = list((project / "agents/rm").glob("*/dist/setup-payload/jav-trans"))
    assert len(archived) == 1
    assert (archived[0] / "bin/old.dll").read_bytes() == b"old dll"
    assert (archived[0] / "uv.exe").read_bytes() == b"old uv"
