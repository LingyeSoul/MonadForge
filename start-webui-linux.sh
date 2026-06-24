#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Starting MonadForge WebUI..."
echo "Access at http://127.0.0.1:8000"

	uv/bin/python -m webui "$@" &
WEBUI_PID=$!

sleep 3
if command -v xdg-open &>/dev/null; then
    xdg-open "http://127.0.0.1:8000" 2>/dev/null || true
elif command -v open &>/dev/null; then
    open "http://127.0.0.1:8000" 2>/dev/null || true
fi

echo "WebUI PID: $WEBUI_PID (Ctrl+C to stop)"
wait "$WEBUI_PID"
