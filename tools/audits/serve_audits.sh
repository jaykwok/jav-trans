#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
MIDDLEWARE="$ROOT_DIR/tools/audits/live_server_audit_middleware.js"
OPEN="${OPEN:-0}"

ARGS=(
  "--host=$HOST"
  "--port=$PORT"
  "--no-browser"
  "--middleware=$MIDDLEWARE"
  "--watch=agents/audits"
  "--wait=500"
  "."
)

if [[ "$OPEN" == "1" ]]; then
  NEXT_ARGS=()
  for arg in "${ARGS[@]}"; do
    [[ "$arg" == "--no-browser" ]] && continue
    NEXT_ARGS+=("$arg")
  done
  ARGS=("--open=agents/audits/index.html" "${NEXT_ARGS[@]}")
fi

echo "Audit navigation: http://$HOST:$PORT/agents/audits/index.html"
echo "Latest audit entry: http://$HOST:$PORT/agents/audits/latest-audit.html"
exec live-server "${ARGS[@]}"
