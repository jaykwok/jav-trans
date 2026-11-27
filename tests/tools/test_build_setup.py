"""Exercise the release assembly script without building or downloading binaries."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


def _prepare_synthetic_project(tmp_path: Path) -> tuple[Path, Path, Path]:
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
    payload = project / "dist/setup-payload/jav-sub"
    (payload / "bin").mkdir(parents=True)
    (payload / "bin/old.dll").write_bytes(b"old dll")
    (payload / "uv.exe").write_bytes(b"old uv")
    tools = project / "test-tools"
    tools.mkdir()
    for name in (
        "ffmpeg.exe",
        "ffprobe.exe",
        "avcodec-1.dll",
        "avformat-1.dll",
        "avutil-1.dll",
        "swresample-1.dll",
        "unrelated.dll",
    ):
        (tools / name).write_bytes(b"synthetic binary")
    return project, payload, tools


@pytest.mark.skipif(not POWERSHELL, reason="PowerShell is required for the Windows build script")
def test_setup_rebuild_drops_stale_payload_bytecode_and_non_ffmpeg_dlls(tmp_path):
    project, payload, _tools = _prepare_synthetic_project(tmp_path)

    wrapper = tmp_path / "build.ps1"
    wrapper.write_text(r'''
$ErrorActionPreference = 'Stop'
function uv {
    if ($args -contains 'PyInstaller') {
        [IO.File]::WriteAllText((Join-Path $PWD 'dist/jav-sub.exe'), 'entry point')
    }
    $global:LASTEXITCODE = 0
}
$Project = Join-Path $PSScriptRoot 'project'
& (Join-Path $Project 'packaging/build_setup.ps1') -SkipArchive -SkipSmoke `
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
    assert (payload / "bin/swresample-1.dll").is_file()
    assert not (payload / "bin/unrelated.dll").exists()
    assert not (payload / "src/__pycache__").exists()
    assert not (payload / "uv.exe").exists()
    assert not (payload / "bin/old.dll").exists()
    archived = list((project / "dev/rm").glob("*/dist/setup-payload/jav-sub"))
    assert len(archived) == 1
    assert (archived[0] / "bin/old.dll").read_bytes() == b"old dll"
    assert (archived[0] / "uv.exe").read_bytes() == b"old uv"


@pytest.mark.skipif(not POWERSHELL, reason="PowerShell is required for the Windows build script")
def test_setup_require_clean_and_temp_manifest_cleanup_when_archiving(tmp_path):
    project, _payload, _tools = _prepare_synthetic_project(tmp_path)
    uv_log = tmp_path / "uv-calls.txt"

    wrapper = tmp_path / "build_archive.ps1"
    wrapper.write_text(
        rf'''
$ErrorActionPreference = 'Stop'
$LogPath = '{uv_log}'
function uv {{
    Add-Content -Path $LogPath -Value ($args -join ' ')
    if ($args -contains 'PyInstaller') {{
        [IO.File]::WriteAllText((Join-Path $PWD 'dist/jav-sub.exe'), 'entry point')
    }}
    for ($i = 0; $i -lt $args.Count; $i++) {{
        if ($args[$i] -eq '--output' -and $i + 1 -lt $args.Count -and ($args -contains 'packaging/record_release_manifest.py')) {{
            [IO.File]::WriteAllText($args[$i + 1], '{{}}')
        }}
    }}
    $global:LASTEXITCODE = 0
}}
$Project = Join-Path $PSScriptRoot 'project'
& (Join-Path $Project 'packaging/build_setup.ps1') -SkipSmoke -RequireClean `
    -FfmpegExe (Join-Path $Project 'test-tools/ffmpeg.exe') `
    -FfprobeExe (Join-Path $Project 'test-tools/ffprobe.exe')
''',
        encoding="ascii",
    )
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(wrapper)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = uv_log.read_text(encoding="utf-8")
    assert "packaging/record_release_manifest.py" in calls
    assert "--require-clean" in calls
    # Intermediate setup-manifest.json must NOT remain at dist/release-assets/ root.
    assert not (project / "dist/release-assets/setup-manifest.json").exists()


@pytest.mark.skipif(not POWERSHELL, reason="PowerShell is required for the Windows build script")
def test_setup_smoke_check_fails_when_built_exe_cannot_run(tmp_path):
    _project, _payload, _tools = _prepare_synthetic_project(tmp_path)

    wrapper = tmp_path / "build_smoke_fail.ps1"
    wrapper.write_text(r'''
$ErrorActionPreference = 'Stop'
function uv {
    if ($args -contains 'PyInstaller') {
        [IO.File]::WriteAllText((Join-Path $PWD 'dist/jav-sub.exe'), 'not a valid executable')
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
    assert result.returncode != 0
