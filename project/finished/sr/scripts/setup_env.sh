#!/usr/bin/env bash
# Verify the ResShift SR sidecar can run in the ROOT Anima venv.
#
# The sidecar no longer has its own venv. Its deps are an opt-in dependency group
# installed with `uv sync --group sr --inexact` (run before this script).
# torch/torchvision are already core Anima deps (the same torch 2.12 + cu132 Blackwell
# build), so there is nothing SR-specific to install beyond the group. xformers stays
# absent (the vendored VQGAN ships query-chunked SDPA — a Blackwell xformers build is a
# non-win, see README.md); ResShift's source is VENDORED basicsr-free under
# sr/resshift/, added to sys.path at import time, so there is no external clone to fetch.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> sr/

echo "==> verifying (torch + vendored ResShift import in the root venv)"
# `uv run` from sr/ discovers the root pyproject upward and uses the synced venv.
uv run python - <<'PY'
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
