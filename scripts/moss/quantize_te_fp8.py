#!/usr/bin/env python3
"""MOSS-Image's Qwen3.5-9B text encoder → one fp8 safetensors (ComfyUI mixed-precision
format, ``library/moss/fp8.py``), rounded to nearest. CPU only. The published file
(catalog row ``moss_te_fp8``) is GPTQ-rounded instead: ``project/moss_lora/src/te_gptq.py``.
``--format int8_convrot`` writes ``library/moss/int8.py``'s INT8 ConvRot file (rotated
on the GPU when there is one). The TE's ``config.json`` is copied beside the file.

    python scripts/moss/quantize_te_fp8.py --out models/moss_image/fp8/qwen35_9b_fp8_e4m3fn.safetensors
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from library.env import resolve_under_home  # noqa: E402
from library.moss.fp8 import quantize_te  # noqa: E402
from library.moss.requests import resolve_model_dir, variant_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model_dir", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--format", default="fp8", choices=("fp8", "int8_convrot"))
    args = p.parse_args()
    src = variant_dir(resolve_model_dir(args.model_dir), "base") / "text_encoder"
    out = resolve_under_home(args.out)
    quantize_te(src, out, fmt=args.format)
    if not (out.parent / "config.json").exists():
        shutil.copy(src / "config.json", out.parent / "config.json")


if __name__ == "__main__":
    main()
