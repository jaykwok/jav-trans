param(
    [switch]$Clean,
    [string]$FfmpegExe = "",
    [string]$FfprobeExe = "",
    [string]$SpecPath = "packaging/javtrans-web.spec"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $Uv) {
    throw "Missing uv. Install uv, run 'uv venv', then install dependencies with 'uv pip install -r requirements.txt'."
}

if ($FfmpegExe) {
    $env:JAVTRANS_FFMPEG_EXE = (Resolve-Path $FfmpegExe).Path
}
if ($FfprobeExe) {
    $env:JAVTRANS_FFPROBE_EXE = (Resolve-Path $FfprobeExe).Path
}

& uv run --no-sync python "packaging/prepare_default_model.py"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Clean) {
    & uv run --no-sync python -m PyInstaller --noconfirm --clean $SpecPath
} else {
    & uv run --no-sync python -m PyInstaller --noconfirm $SpecPath
}
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$Exe = Join-Path $Root "dist/JAVTrans/JAVTrans.exe"
if (-not (Test-Path $Exe)) {
    throw "Build finished but executable was not found: $Exe"
}

Write-Host "DONE: $Exe"
