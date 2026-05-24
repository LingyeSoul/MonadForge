#!/usr/bin/env python3
"""Cache pooled text embeddings (max over the sequence dim) from existing TE caches.

Reads each ``{stem}_anima_te.safetensors`` in a cache directory and writes a
matching ``{stem}_anima_pooled.safetensors`` sidecar holding
``pooled_v{i} = crossattn_emb_v{i}.amax(dim=0)`` for every variant present.

Consumed by ``scripts/distill_mod/distill.py`` (modulation guidance distillation):
``pooled_text_proj`` ingests this tensor at every training microstep and val
sigma; pre-caching it eliminates a redundant ``.max(dim=1)`` per step.

No GPU / text encoder needed -- pure tensor reduction on the cached crossattn.
The reduction loop lives in ``library/preprocess/text.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.io.cache import TE_CACHE_SUFFIX  # noqa: E402
from library.preprocess import cache_pooled_text, tqdm_progress  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Cache directory containing *_anima_te.safetensors files. Pooled "
        "sidecars are written into the same directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-emit pooled sidecars even when they already exist.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.dir)
    stats = cache_pooled_text(
        cache_dir,
        overwrite=args.overwrite,
        progress=tqdm_progress("Caching pooled"),
    )
    if stats.seen == 0:
        print(f"No {TE_CACHE_SUFFIX} files found in {cache_dir}")
        return

    print(
        f"Pooled cache: {stats.written} written, {stats.skipped} skipped (already "
        f"existed), {stats.failed} failed (no crossattn key)"
    )


if __name__ == "__main__":
    main()
