param(
    [switch]$Clean,
    [switch]$SkipArchive,
    [switch]$BundleUv,
    [string]$UvExe = "",
    [string]$FfmpegExe = "",
    [string]$FfprobeExe = "",
    [string]$SpecPath = "packaging/jav-trans-setup.spec",
    [string]$OutputDir = "dist/release-assets",
    [string]$ArchiveName = "jav-trans-windows-x64.zip"
)

# Assemble the lightweight release: the entry-point executable, uv, FFmpeg, and
# the source it syncs against. Everything else - PyTorch, the ASR weights, the
# CTC head - is fetched on the user's machine at first run, which is what takes
# the release from ~6 GB to ~150 MB.
#
# The exe is jav-trans.exe, not jav-trans-setup.exe: installing is only what the
# first run does, and it is the launcher every time after that.

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

# Staged one level down so the in-zip folder can be named jav-trans without
# colliding with the full bundle's onedir output at dist/jav-trans.
$PayloadRoot = Join-Path $Root "dist/setup-payload"
$Payload = Join-Path $PayloadRoot "jav-trans"
# Not shared with the full bundle: both specs build a target named jav-trans, and
# one PyInstaller cache for two different Analysis inputs is a stale-build trap.
$WorkPath = Join-Path $Root "build/jav-trans-setup"

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

function Resolve-Tool {
    param(
        [string]$Override,
        [string]$Name,
        [string]$Hint
    )
    if ($Override) {
        return (Resolve-Path $Override).Path
    }
    $Found = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $Found) {
        throw "$Name not found. $Hint"
    }
    return $Found.Source
}

# uv is not bundled by default. It would add 50-80 MB to buy nothing: the
# installer fetches it from PyPI, and the very next thing it does is install
# dependencies from that same host, so a machine that cannot reach PyPI cannot
# be installed on either way. -BundleUv is there for building a release for a
# network where PyPI itself is unreachable and only the archive travels.
$UvSource = ""
if ($BundleUv) {
    $UvSource = Resolve-Tool -Override $UvExe -Name "uv.exe" `
        -Hint "Install uv or pass -UvExe with the full path."
}

# torchcodec loads the FFmpeg shared DLLs at import time and uv cannot install
# them, so they have to travel with the release exactly as the full bundle
# ships them.
$FfmpegSource = Resolve-Tool -Override $FfmpegExe -Name "ffmpeg.exe" `
    -Hint "Install Gyan.FFmpeg.Shared or pass -FfmpegExe."
$FfprobeSource = Resolve-Tool -Override $FfprobeExe -Name "ffprobe.exe" `
    -Hint "Install Gyan.FFmpeg.Shared or pass -FfprobeExe."
$FfmpegDir = Split-Path -Parent $FfmpegSource
foreach ($Pattern in @("avcodec-*.dll", "avformat-*.dll", "avutil-*.dll")) {
    if (-not (Get-ChildItem -Path $FfmpegDir -Filter $Pattern -ErrorAction SilentlyContinue)) {
        throw "FFmpeg Shared build required: $FfmpegDir has no $Pattern. Install Gyan.FFmpeg.Shared and remove the static-only Gyan.FFmpeg package."
    }
}

if ($Clean) {
    Move-BuildArtifactToRm $Payload "dist-jav-trans-setup-payload"
    & uv run --no-sync python -m PyInstaller --noconfirm --clean --workpath $WorkPath $SpecPath
} else {
    & uv run --no-sync python -m PyInstaller --noconfirm --workpath $WorkPath $SpecPath
}
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$EntryExe = Join-Path $Root "dist/jav-trans.exe"
if (-not (Test-Path $EntryExe)) {
    throw "Build finished but the executable was not found: $EntryExe"
}

New-Item -ItemType Directory -Force -Path $Payload | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Payload "bin") | Out-Null

Move-Item -LiteralPath $EntryExe -Destination (Join-Path $Payload "jav-trans.exe") -Force
if ($UvSource) {
    Copy-Item -LiteralPath $UvSource -Destination (Join-Path $Payload "uv.exe") -Force
}

Copy-Item -LiteralPath $FfmpegSource -Destination (Join-Path $Payload "bin") -Force
Copy-Item -LiteralPath $FfprobeSource -Destination (Join-Path $Payload "bin") -Force
Get-ChildItem -Path $FfmpegDir -Filter "*.dll" | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Payload "bin") -Force
}

# What uv syncs against, plus what launcher.py runs. Kept as plain source: the
# installed .venv imports it directly, so there is nothing to compile.
# uv.lock is required, not optional: without it every user resolves their own
# dependency versions and a release stops being one artifact.
foreach ($Item in @("launcher.py", "pyproject.toml", "uv.lock")) {
    $Source = Join-Path $Root $Item
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Missing $Item. Run 'uv sync' in the repository root before building."
    }
    Copy-Item -LiteralPath $Source -Destination $Payload -Force
}
# Named in ASCII on purpose: Windows PowerShell 5.1 reads a BOM-less .ps1 as the
# system code page, so a Chinese literal here would ship as a mojibake filename.
Copy-Item -LiteralPath (Join-Path $Root "packaging/setup-readme.txt") `
    -Destination (Join-Path $Payload "README.txt") -Force

$SrcDestination = Join-Path $Payload "src"
if (Test-Path -LiteralPath $SrcDestination) {
    Remove-Item -LiteralPath $SrcDestination -Recurse -Force
}
Copy-Item -LiteralPath (Join-Path $Root "src") -Destination $SrcDestination -Recurse -Force
Get-ChildItem -Path $SrcDestination -Directory -Recurse -Filter "__pycache__" |
    Remove-Item -Recurse -Force

$SizeMb = [math]::Round(
    ((Get-ChildItem -Path $Payload -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB), 1)
Write-Host "Payload: $Payload ($SizeMb MB)"

if ($SkipArchive) {
    Write-Host "DONE (archive skipped)"
    exit 0
}

# zip rather than the .7z the full bundle uses: this archive is small enough
# that the compression ratio does not matter, and Windows opens zip without
# installing anything.
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$Archive = Join-Path (Resolve-Path $OutputDir) $ArchiveName
Remove-Item -LiteralPath $Archive -Force -ErrorAction SilentlyContinue
Compress-Archive -Path $Payload -DestinationPath $Archive -CompressionLevel Optimal

$ArchiveMb = [math]::Round(((Get-Item -LiteralPath $Archive).Length / 1MB), 1)
Write-Host "DONE: $Archive ($ArchiveMb MB)"
