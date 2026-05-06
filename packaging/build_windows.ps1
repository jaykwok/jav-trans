param(
    [switch]$Clean,
    [string]$FfmpegExe = "",
    [string]$FfprobeExe = "",
    [string]$SpecPath = "packaging/javtrans-web.spec"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Python = Join-Path $Root ".venv/Scripts/python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing project venv: .venv/Scripts/python.exe"
}

if ($FfmpegExe) {
    $env:JAVTRANS_FFMPEG_EXE = (Resolve-Path $FfmpegExe).Path
}
if ($FfprobeExe) {
    $env:JAVTRANS_FFPROBE_EXE = (Resolve-Path $FfprobeExe).Path
}

& $Python "packaging/prepare_default_model.py"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Clean) {
    & $Python -m PyInstaller --noconfirm --clean $SpecPath
} else {
    & $Python -m PyInstaller --noconfirm $SpecPath
}
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$Exe = Join-Path $Root "dist/JAVTrans/JAVTrans.exe"
if (-not (Test-Path $Exe)) {
    throw "Build finished but executable was not found: $Exe"
}

Write-Host "DONE: $Exe"
