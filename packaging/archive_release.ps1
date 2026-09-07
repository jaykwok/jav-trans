param(
    [string]$SourceDir = "dist/jav-trans",
    [string]$OutputDir = "dist/release-assets",
    [string]$ArchiveName = "jav-trans-windows-x64.7z",
    [string]$ManifestPath = "dist/release-assets/release-manifest.json",
    [int]$Threads = [Environment]::ProcessorCount
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

$Source = (Resolve-Path -LiteralPath $SourceDir).Path
$Manifest = (Resolve-Path -LiteralPath $ManifestPath).Path

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

Write-Host "Using $Threads compression threads."
Write-Host "Using 7-Zip: $SevenZipPath"
& uv run --no-sync python "packaging/archive_release.py" `
    --source $Source --output $OutputDir --manifest $Manifest `
    --name $ArchiveName --threads $Threads --seven-zip $SevenZipPath
exit $LASTEXITCODE
