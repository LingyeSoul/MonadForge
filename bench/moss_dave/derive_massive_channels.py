#!/usr/bin/env python3
"""MOSS massive-activation channels that DAVE must leave alone, from a probe run.

Reads ``channels.npz`` from ``probe_dc_convergence.py`` (per-block channel ranking
by early DC energy) and writes the union of every block's top-k as one channel
list. CPU only, no model load.

    uv run python bench/moss_dave/derive_massive_channels.py \
        --probe bench/moss_dave/results/<run>/channels.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import numpy as np  # noqa: E402

from library.env import resolve_under_home  # noqa: E402
from library.moss.dave import DEFAULT_CHANNELS_PATH  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", required=True, help="channels.npz of a probe run")
    ap.add_argument("--k", type=int, default=4, help="top-k channels per block")
    ap.add_argument("--model", default="base", help="recorded as provenance")
    ap.add_argument("--out", default=DEFAULT_CHANNELS_PATH)
    args = ap.parse_args()

    d = np.load(args.probe)
    order, energy = d["order"], d["dc_energy_early"]
    top = order[:, : args.k]
    channels = np.unique(top)
    share = np.take_along_axis(energy, top, 1).sum(1) / energy.sum(1)
    print(f"top-{args.k} per block → {channels.size} channels: {channels.tolist()}")
    print(f"share of early DC energy per block: {share.min():.3f}–{share.max():.3f}")

    out = resolve_under_home(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        channels=channels.astype(np.int64),
        per_block_top=top.astype(np.int64),
        k=np.int64(args.k),
        model=args.model,
        source_probe=str(args.probe),
    )
    print(f"→ {out}")


if __name__ == "__main__":
    main()
