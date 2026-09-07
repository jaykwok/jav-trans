param(
    [switch]$Clean,
    [switch]$SkipModels,
    [string]$FfmpegExe = "",
    [string]$FfprobeExe = "",
    [string]$SpecPath = "packaging/jav-trans-web.spec"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

$Uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $Uv) {
    throw "Missing uv. Install uv, run 'uv venv', then install dependencies with 'uv pip install -r requirements.txt'."
}

if ($FfmpegExe) {
    $env:JAV_TRANS_FFMPEG_EXE = (Resolve-Path $FfmpegExe).Path
}
if ($FfprobeExe) {
    $env:JAV_TRANS_FFPROBE_EXE = (Resolve-Path $FfprobeExe).Path
}

function Move-BuildArtifactToRm {
    param(
        [string]$Path,
        [string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $RmRoot = Join-Path $Root "agents/rm"
    New-Item -ItemType Directory -Force -Path $RmRoot | Out-Null
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $Destination = Join-Path $RmRoot "$($Stamp)_$Label"
    if (Test-Path -LiteralPath $Destination) {
        $Destination = "$Destination-$([DateTimeOffset]::Now.ToUnixTimeMilliseconds())"
    }
    Move-Item -LiteralPath $Path -Destination $Destination
    Write-Host "Moved old build artifact to $Destination"
}

if ($SkipModels) {
    $env:JAV_TRANS_SKIP_MODELS = "1"
} else {
    Remove-Item Env:JAV_TRANS_SKIP_MODELS -ErrorAction SilentlyContinue
    & uv run --no-sync python "packaging/prepare_default_model.py"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if ($Clean) {
    Move-BuildArtifactToRm (Join-Path $Root "dist/JAVTrans") "dist-JAVTrans"
    Move-BuildArtifactToRm (Join-Path $Root "dist/jav-trans") "dist-jav-trans"
    & uv run --no-sync python -m PyInstaller --noconfirm --clean $SpecPath
} else {
    & uv run --no-sync python -m PyInstaller --noconfirm $SpecPath
}
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$Exe = Join-Path $Root "dist/jav-trans/jav-trans.exe"
if (-not (Test-Path $Exe)) {
    throw "Build finished but executable was not found: $Exe"
}

# Record what this build actually contains. The version says which commit; only
# this says which weights, which binaries, and which lock file.
$ManifestArgs = @("--dist", "dist/jav-trans")
if ($SkipModels) { $ManifestArgs += "--skip-models" }
& uv run --no-sync python "packaging/record_release_manifest.py" @ManifestArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "DONE: $Exe"
