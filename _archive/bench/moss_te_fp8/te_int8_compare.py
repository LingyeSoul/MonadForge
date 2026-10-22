#!/usr/bin/env python3
"""INT8 ConvRot vs the shipped fp8 GPTQ v2 TE, as one daemon job: quantize → evaluate.

1. round-to-nearest INT8 ConvRot (``scripts/moss/quantize_te_fp8.py --format int8_convrot``)
2. GPTQ on the INT8 ConvRot grid, v2's calibration (``te_gptq.py --grid int8_convrot
   --manifest``, the ``te_calib.py`` calib split)
3. ``bench/moss_te_fp8`` text arms: bf16, v2 (the catalog file), each INT8 file W8A8
   (``comfy_kitchen.int8_linear``) and weight-only (``--w8_only``)
4. ``te_calib_eval.py`` per category on the eval split: v2 weight-only, the INT8
   arms W8A8 only

Steps 1–2 skip a file already on disk.

    make daemon-run ARGS="--queue --stall-timeout 1800 project/moss_lora/src/te_int8_compare.py"
"""

import os
import subprocess
import sys

from library.env import resolve_under_home

SRC = "project/moss_lora/src"
MANIFEST = "post_image_dataset/moss/_te_calib_v2/manifest.json"
V2 = "models/moss_image/fp8/qwen35_9b_fp8_e4m3fn.safetensors"
RTN = "/media/sorryhyun/new/moss_te_int8/qwen35_9b_int8_convrot.safetensors"  # NVMe is short
GPTQ = "models/moss_image/int8_convrot_gptq/qwen35_9b_int8_convrot_gptq.safetensors"


def run(argv):
    print(f"[int8_compare] {' '.join(argv)}", flush=True)
    subprocess.run(
        [sys.executable, *argv], check=True, cwd=resolve_under_home("."), env=os.environ
    )


def main() -> None:
    if not resolve_under_home(RTN).exists():
        run(
            [
                "scripts/moss/quantize_te_fp8.py",
                "--format",
                "int8_convrot",
                "--out",
                RTN,
            ]
        )
    if not resolve_under_home(GPTQ).exists():
        run(
            [
                f"{SRC}/te_gptq.py",
                *("--grid", "int8_convrot", "--manifest", MANIFEST, "--out", GPTQ),
            ]
        )
    files = {
        "v2": V2,
        "int8_rtn": RTN,
        "int8_rtn_w8": RTN,
        "int8_gptq": GPTQ,
        "int8_gptq_w8": GPTQ,
    }
    run(
        [
            "bench/moss_te_fp8/run_bench.py",
            "--no_images",
            *("--arms", ",".join(["bf16", *files])),
            *(
                a
                for name, path in files.items()
                for a in ("--fp8_file", f"{name}={path}")
            ),
            *("--w8_only", "int8_rtn_w8,int8_gptq_w8"),
            *("--label", "int8_convrot_text"),
        ]
    )
    run(
        [
            f"{SRC}/te_calib_eval.py",
            *(
                a
                for name in ("v2", "int8_rtn", "int8_gptq")
                for a in ("--fp8", f"{name}={files[name]}")
            ),
            *("--out", "output/tests/te_calib_eval/int8_convrot"),
        ]
    )


if __name__ == "__main__":
    main()
