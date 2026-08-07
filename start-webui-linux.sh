#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Detect Python interpreter: prefer .venv/bin/python (Linux venv), then system python
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

WEBUI_URL="http://127.0.0.1:8000"
ANALYZER_URL="http://127.0.0.1:8320"

echo "Starting MonadForge WebUI..."
echo "  WebUI    -> $WEBUI_URL"
echo "  Analyzer -> $ANALYZER_URL"

"$PYTHON" -m webui "$@" &
WEBUI_PID=$!

"$PYTHON" -m scripts.run_analyzer.server --port 8320 &
ANALYZER_PID=$!

cleanup() {
    kill "$ANALYZER_PID" 2>/dev/null || true
    kill "$WEBUI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 3
if command -v xdg-open &>/dev/null; then
    xdg-open "$WEBUI_URL" 2>/dev/null || true
    xdg-open "$ANALYZER_URL" 2>/dev/null || true
elif command -v open &>/dev/null; then
    open "$WEBUI_URL" 2>/dev/null || true
    open "$ANALYZER_URL" 2>/dev/null || true
fi

echo "WebUI PID: $WEBUI_PID / Analyzer PID: $ANALYZER_PID (Ctrl+C to stop)"
wait "$WEBUI_PID" "$ANALYZER_PID"
