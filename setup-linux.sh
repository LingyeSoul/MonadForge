#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  MonadForge - Linux One-Click Setup"
echo "  Forked from https://github.com/sorryhyun/anima_lora"
echo "============================================================"
echo

# ---------------------------------------------------------------
# 0. Check Python
# ---------------------------------------------------------------
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "[ERROR] Python not found in PATH."
    echo "        Install Python 3.13+ from your package manager or https://www.python.org/downloads/"
    exit 1
fi

PYTHON="$(command -v python3 || command -v python)"
PY_VER="$($PYTHON --version 2>&1)"
echo "[INFO] $PY_VER"

# ---------------------------------------------------------------
# 1. Install / update uv
# ---------------------------------------------------------------
if command -v uv &>/dev/null; then
    echo "[STEP 1/4] uv found, upgrading ..."
else
    echo "[STEP 1/4] Installing uv ..."
fi
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv &>/dev/null; then
    echo "[ERROR] uv install failed. Open a new shell and re-run."
    exit 1
fi

# ---------------------------------------------------------------
# 2. Sync project dependencies (torch, triton, flash-attn, etc.)
# ---------------------------------------------------------------
echo
echo "[STEP 2/4] Syncing project dependencies via uv ..."

SYNC_OK=0
for attempt in 1 2 3; do
    if uv sync; then
        SYNC_OK=1
        break
    fi
    if [ "$attempt" -lt 3 ]; then
        echo "[WARN] uv sync failed. Retrying (attempt $attempt/2) in 3s ..."
        sleep 3
    fi
done

if [ "$SYNC_OK" -eq 0 ]; then
    echo
    echo "[ERROR] uv sync did not complete after 3 attempts."
    echo "        Check your network connection or disk space, then re-run."
    exit 1
fi

# ---------------------------------------------------------------
# 3. Ensure Node.js (for WebUI frontend build)
# ---------------------------------------------------------------
echo

NODE_READY=0
if command -v node &>/dev/null; then
    NODE_VER="$(node --version 2>&1)"
    echo "[STEP 3/4] Node.js found: $NODE_VER"
    NODE_READY=1
fi

PORTABLE_NODE="$SCRIPT_DIR/tools/node"
if [ "$NODE_READY" -eq 0 ] && [ -x "$PORTABLE_NODE/bin/node" ]; then
    echo "[STEP 3/4] Using portable Node.js from tools/node/"
    export PATH="$PORTABLE_NODE/bin:$PATH"
    NODE_READY=1
fi

if [ "$NODE_READY" -eq 0 ]; then
    echo "[STEP 3/4] Node.js not found. Setting up portable Node.js ..."
    NODE_VERSION="v24.16.0"
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64)  NODE_ARCH="linux-x64" ;;
        aarch64) NODE_ARCH="linux-arm64" ;;
        *)       echo "[ERROR] Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    NODE_TAR_URL="https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-${NODE_ARCH}.tar.xz"
    NODE_TAR="$SCRIPT_DIR/node.tar.xz"
    NODE_EXTRACT_DIR="$SCRIPT_DIR/tools/node-${NODE_VERSION}-${NODE_ARCH}"

    mkdir -p "$SCRIPT_DIR/tools"

    echo "        Downloading $NODE_TAR_URL ..."
    curl -LsSf "$NODE_TAR_URL" -o "$NODE_TAR"

    echo "        Extracting ..."
    tar -xf "$NODE_TAR" -C "$SCRIPT_DIR/tools"
    rm -f "$NODE_TAR"

    mv "$NODE_EXTRACT_DIR" "$PORTABLE_NODE" 2>/dev/null || true

    if [ ! -x "$PORTABLE_NODE/bin/node" ]; then
        echo "[ERROR] Portable Node.js setup failed."
        echo "        Please install Node.js manually from https://nodejs.org/"
        exit 1
    fi

    export PATH="$PORTABLE_NODE/bin:$PATH"
    echo "        Portable Node.js $(node --version) installed to tools/node/"
fi

# ---------------------------------------------------------------
# 4. Build WebUI frontend
# ---------------------------------------------------------------
echo
echo "[STEP 4/4] Building WebUI frontend ..."

cd "$SCRIPT_DIR/webui/frontend"

if [ ! -d node_modules ]; then
    echo "        Installing npm dependencies ..."
    npm install
fi

npm run build
echo "[STEP 4/4] WebUI frontend built successfully."

# ---------------------------------------------------------------
# Done
# ---------------------------------------------------------------
echo
echo "============================================================"
echo "  Setup complete!"
echo
echo "  Quick start:"
echo "    ./start-webui-linux.sh       - launch WebUI (browser)"
echo "    python tasks.py lora         - train LoRA"
echo "    python tasks.py test         - inference test"
echo "    python tasks.py --help       - all commands"
echo
echo "  Model downloads:"
echo "    python tasks.py download-models"
echo "============================================================"
