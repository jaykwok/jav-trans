#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
OPEN="${OPEN:-0}"

ARGS=(
  "run"
  "python"
  "-m"
  "tools.audits.serve_static"
  "--host"
  "$HOST"
  "--port"
  "$PORT"
)

if [[ "$OPEN" == "1" ]]; then
  ARGS+=("--open")
fi

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTHONPATH="${PYTHONPATH:-src}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-agents/temp/uv-cache}"

exec uv "${ARGS[@]}"
