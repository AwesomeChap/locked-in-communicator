#!/usr/bin/env bash
# Start the BCI WebSocket server and React dev server together.
# Usage: ./run.sh   (from the LockedInCommunicator repo root)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
BACKEND_BIN="$BACKEND/.venv/bin/lockedin-verification-server"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  local pid
  for pid in "$FRONTEND_PID" "$BACKEND_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT INT TERM

if [[ ! -x "$BACKEND_BIN" ]]; then
  echo "Backend not ready. Set it up first:"
  echo "  cd backend"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -e ."
  exit 1
fi

if [[ ! -d "$FRONTEND/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  (cd "$FRONTEND" && npm install)
fi

echo "Starting backend  (ws://localhost:8765)..."
"$BACKEND_BIN" &
BACKEND_PID=$!

sleep 1
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
  echo "Backend failed to start."
  exit 1
fi

echo "Starting frontend (http://localhost:5173)..."
(cd "$FRONTEND" && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "Both services running. Open http://localhost:5173"
echo "Press Ctrl+C to stop."
echo ""

# Bash 3.2 (macOS default) has no `wait -n`; wait for either process to exit.
while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$FRONTEND_PID" 2>/dev/null; do
  sleep 1
done
