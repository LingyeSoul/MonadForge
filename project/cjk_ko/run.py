#!/usr/bin/env python
"""cjk_ko front door: ``run.py <run> data | train`` on ``configs/<run>.toml``,
through the reseed line's front door (``../cjk_anima_reseed/run.py``: its
builder and trainer, its flags); outputs in ``output/cjk_ko/<run>/``.

    .venv/bin/python project/cjk_ko/run.py ko_chat_warm data --frac 0.03   # sizes
    .venv/bin/python project/cjk_ko/run.py ko_chat_warm data
    make daemon-run ARGS="project/cjk_ko/run.py ko_chat_warm train"
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent
RESEED = HOME.parent / "cjk_anima_reseed" / "run.py"

if __name__ == "__main__":
    a = sys.argv
    if len(a) > 1 and not a[1].startswith("-") and "/" not in a[1]:
        a[1] = str(HOME / "configs" / f"{a[1]}.toml")
    a[0] = str(RESEED)
    runpy.run_path(str(RESEED), run_name="__main__")
