[CmdletBinding()]
param(
    [string]$BindAddress = $(if ($env:HOST) { $env:HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:PORT) { [int]$env:PORT } else { 8080 }),
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$middleware = Join-Path $rootDir "tools\audits\live_server_audit_middleware.js"
$arguments = @(
    "--host=$BindAddress"
    "--port=$Port"
    "--no-browser"
    "--middleware=$middleware"
    "--watch=agents/audits"
    "--wait=500"
    "."
)
if ($Open) {
    $arguments = @("--open=agents/audits/index.html") + ($arguments | Where-Object { $_ -ne "--no-browser" })
}

Push-Location $rootDir
try {
    Write-Host "Audit navigation: http://$BindAddress`:$Port/agents/audits/index.html"
    Write-Host "Latest audit entry: http://$BindAddress`:$Port/agents/audits/latest-audit.html"
    $liveServer = Get-Command "live-server.cmd" -ErrorAction SilentlyContinue
    if (-not $liveServer) {
        $liveServer = Get-Command "live-server" -ErrorAction SilentlyContinue
    }
    if ($liveServer) {
        & $liveServer.Source @arguments
    }
    else {
        $npx = Get-Command "npx.cmd" -ErrorAction SilentlyContinue
        if (-not $npx) {
            $npx = Get-Command "npx" -ErrorAction SilentlyContinue
        }
        if (-not $npx) {
            throw "live-server and npx were not found in PATH. Install live-server with npm first."
        }
        & $npx.Source --no-install live-server @arguments
    }

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
