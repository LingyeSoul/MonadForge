#!/usr/bin/env python3
"""Spatial diff maps for the sigma-cutoff knockout sweep.

Global RMSE/SSIM are bulk-dominated and blind to small prompt-specified features
(logos, expressions, accessories) that get committed in the mid-sigma band. This
shows *where* each knockout condition differs from the full-prompt baseline, so
localized semantic steering is visible even when the global drift is small.

    python bench/cross_attn_drive/knockout_diffmap.py output/tests/knockout

Each row = one seed/prompt. Col 0 = baseline image; remaining cols = |baseline -
cutoff| heat (mean over channels), shared color scale across the whole figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _seed_of(p: Path) -> str:
    digits = [tok for tok in p.stem.split("_") if tok.isdigit()]
    return digits[-1] if digits else p.stem


def _index_dir(d: Path) -> dict[str, Path]:
    return {_seed_of(p): p for p in sorted(d.glob("*.png"))}


def _load(p: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    im = Image.open(p).convert("RGB")
    if size is not None:
        im = im.resize(size, Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_idx = _index_dir(args.root / "baseline")
    seeds = sorted(base_idx)
    cut_dirs = sorted(
        (d for d in args.root.glob("cut_*") if d.is_dir()),
        key=lambda d: float(d.name.split("_")[1]),
        reverse=True,
    )
    cut_idx = [(d.name.split("_")[1], _index_dir(d)) for d in cut_dirs]

    # precompute diffs (shared color scale)
    diffs: dict[tuple[str, str], np.ndarray] = {}
    vmax = 0.0
    for s in seeds:
        base = _load(base_idx[s])
        h, w = base.shape[:2]
        for label, idx in cut_idx:
            if s not in idx:
                continue
            other = _load(idx[s], size=(w, h))
            dm = np.abs(base - other).mean(axis=2)
            diffs[(s, label)] = dm
            vmax = max(vmax, float(np.percentile(dm, 99.5)))

    n_rows, n_cols = len(seeds), 1 + len(cut_idx)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows), squeeze=False
    )
    for r, s in enumerate(seeds):
        ax = axes[r][0]
        ax.imshow(_load(base_idx[s]))
        ax.set_ylabel(f"seed {s}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title("baseline", fontsize=9)
        for c, (label, _) in enumerate(cut_idx, start=1):
            ax = axes[r][c]
            dm = diffs.get((s, label))
            if dm is not None:
                ax.imshow(dm, cmap="inferno", vmin=0.0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"|Δ| σ<{label}", fontsize=9)

    fig.suptitle(
        "Where text-knockout changes the image (bright = differs from baseline)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = args.root / "diffmap.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}  (shared vmax={vmax:.3f})")


if __name__ == "__main__":
    main()
