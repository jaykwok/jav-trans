param(
    [string]$SourceDir = "dist/JAVTrans",
    [string]$OutputDir = "dist/release-assets",
    [string]$ArchiveName = "JAVTrans-windows-x64.7z",
    [string]$VolumeSize = "1900m",
    [int]$Threads = [Environment]::ProcessorCount
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Source = Resolve-Path $SourceDir
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$Output = Join-Path (Resolve-Path $OutputDir) $ArchiveName

$SevenZipCandidates = @(
    "C:\Program Files\7-Zip\7z.exe",
    "C:\Program Files (x86)\7-Zip\7z.exe",
    "C:\Program Files\7-Zip-Zstandard\7z.exe",
    "C:\Program Files (x86)\7-Zip-Zstandard\7z.exe"
)
$SevenZipPath = $SevenZipCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $SevenZipPath) {
    $SevenZip = Get-Command 7z, 7za, 7zr -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($SevenZip) {
        $SevenZipPath = $SevenZip.Source
    }
}
if (-not $SevenZipPath) {
    throw "7-Zip CLI not found. Install 7-Zip or put 7z/7za/7zr on PATH."
}
if ($Threads -lt 1) {
    $Threads = [Environment]::ProcessorCount
}

Remove-Item -Path "$Output.*" -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $Output -Force -ErrorAction SilentlyContinue

Write-Host "Using $Threads compression threads."
Write-Host "Using 7-Zip: $SevenZipPath"
& $SevenZipPath a -t7z -m0=LZMA2 -mx=5 "-mmt=$Threads" "-v$VolumeSize" $Output $Source
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Get-ChildItem -Path (Resolve-Path $OutputDir) -Filter "$ArchiveName.*" |
    Select-Object Name,Length,LastWriteTime
