[CmdletBinding()]
param(
    [string]$BindAddress = $(if ($env:HOST) { $env:HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:PORT) { [int]$env:PORT } else { 8080 }),
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $(if ($env:PYTHONPATH) { $env:PYTHONPATH } else { "src" })
$env:UV_CACHE_DIR = $(if ($env:UV_CACHE_DIR) { $env:UV_CACHE_DIR } else { "agents/temp/uv-cache" })
$arguments = @(
    "run"
    "python"
    "-m"
    "tools.audits.serve_static"
    "--host"
    $BindAddress
    "--port"
    "$Port"
)
if ($Open) {
    $arguments += "--open"
}

Push-Location $rootDir
try {
    & uv @arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
