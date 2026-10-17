"""Convert a register ``adapter.pt`` to the canonical ``adapter.safetensors``.

The safetensors artifact is what the ComfyUI node
(``custom_nodes/comfyui-anima-register``) loads: tensors flat, config JSON-encoded
into the header metadata. New ``train_registers.py`` runs write both; use this for
runs saved before that (or to re-derive the safetensors next to a .pt).

    uv run python bench/headroom/convert_adapter.py output/bench/headroom/<run>/adapter.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pt_path", help="path to adapter.pt")
    ap.add_argument("--out", default=None, help="output .safetensors (default: sibling)")
    args = ap.parse_args()

    p = Path(args.pt_path)
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    config = ckpt["meta"]["config"]
    sd = {
        k: v.detach().cpu().to(torch.float32).contiguous()
        for k, v in ckpt["state_dict"].items()
    }
    out = Path(args.out) if args.out else p.with_suffix(".safetensors")
    save_file(
        sd,
        str(out),
        metadata={
            "format": "anima-register-v1",
            "anima_register_config": json.dumps(config),
        },
    )
    print(f"wrote {out}  (config={config})")


if __name__ == "__main__":
    main()
