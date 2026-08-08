#!/usr/bin/env bash
# Verify the ResShift SR sidecar can run in the root Anima venv.
#
# `make sr-setup` chooses either the standard locked dependency group or the V100
# additive install path. This verifier must never invoke `uv run`: uv run can perform
# an implicit project sync and replace a protected V100 Torch/FlashAttention stack.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SR_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
ROOT_DIR=$(cd "$SR_DIR/.." && pwd)
PYTHON=${SR_PYTHON:-$ROOT_DIR/.venv/bin/python}
if [ ! -x "$PYTHON" ]; then
    PYTHON=${PYTHON_FALLBACK:-python}
fi
cd "$SR_DIR"

echo "==> verifying (torch + vendored ResShift import in the root venv)"
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PY'
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path("resshift").resolve()))
from models.unet import UNetModelSwin  # noqa: F401 (vendored)
from ldm.models.autoencoder import VQModelTorch  # noqa: F401 (vendored, patched attn)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "cap", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print("vendored ResShift import OK")
PY
echo "==> SR sidecar ready (running in the root Anima venv)."
