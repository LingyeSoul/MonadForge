#!/usr/bin/env python3
"""INT8 ConvRot W8A8 with SmoothQuant and activation-aware GPTQ against the bf16 TE
and fp8 GPTQ v2, as one daemon job.

1. ``te_gptq.py --grid int8_convrot --manifest`` (v2's calibration) per variant:
   ``--smooth 0.5``, ``--act_aware``, both
2. ``te_calib_eval.py`` per category on the eval split, v2 and the INT8 arms (W8A8),
   into ``te_int8_compare.py``'s output dir (its bf16 / v2 embeddings are reused)

Step 1 skips a file already on disk.

    make daemon-run ARGS="--stall-timeout 1800 project/moss_lora/src/te_int8_act_compare.py"
"""

import os
import subprocess
import sys

from library.env import resolve_under_home
from te_int8_compare import MANIFEST, SRC, V2

ROOT = "/media/sorryhyun/new/moss_te_int8"  # NVMe is short
VARIANTS = {
    "int8_smooth": ["--smooth", "0.5"],
    "int8_act": ["--act_aware"],
    "int8_smooth_act": ["--smooth", "0.5", "--act_aware"],
}


def run(argv):
    print(f"[int8_act_compare] {' '.join(argv)}", flush=True)
    subprocess.run(
        [sys.executable, *argv], check=True, cwd=resolve_under_home("."), env=os.environ
    )


def main() -> None:
    files = {"v2": V2}
    for name, flags in VARIANTS.items():
        path = f"{ROOT}/{name}/qwen35_9b_{name}.safetensors"
        files[name] = path
        if not resolve_under_home(path).exists():
            run(
                [
                    f"{SRC}/te_gptq.py",
                    *("--grid", "int8_convrot", "--manifest", MANIFEST),
                    *("--out", path, *flags),
                ]
            )
    run(
        [
            f"{SRC}/te_calib_eval.py",
            *(a for name, path in files.items() for a in ("--fp8", f"{name}={path}")),
            *("--out", "output/tests/te_calib_eval/int8_convrot"),
        ]
    )


if __name__ == "__main__":
    main()
