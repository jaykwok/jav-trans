#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
MIDDLEWARE="$ROOT_DIR/tools/audits/live_server_audit_middleware.js"

exec live-server \
  --host="$HOST" \
  --port="$PORT" \
  --open=agents/audits/index.html \
  --middleware="$MIDDLEWARE" \
  --watch=agents/audits \
  --wait=500 \
  .
