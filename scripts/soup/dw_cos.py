#!/usr/bin/env python3
"""Pairwise ΔW cosine + energy share between soup ingredients.

The pre-build check: ingredients of one soup should carry roughly even energy
shares; a low cos alone is not a reason to drop one (``project/moss_lora/soup.md``).
<ΔW_i, ΔW_j> = tr((U_i^T U_j)(D_j D_i^T)), no ΔW materialized. Prints each
ingredient's ‖ΔW‖² and energy share, then per pair the per-module mean / median /
min / max cos and the full-sum (all modules concatenated) cos.

    python -m scripts.soup.dw_cos a.safetensors b.safetensors [c ...]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
from safetensors.torch import load_file

from scripts.soup.build import effective_factors, group_modules


def _inner(a, b) -> float:
    (da, ua), (db, ub) = a, b
    return float(
        ((ua.T.double() @ ub.double()) * (da.double() @ db.T.double()).T).sum()
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpts", nargs="+", help="plain-LoRA ingredient checkpoints")
    paths = ap.parse_args().ckpts
    if len(paths) < 2:
        ap.error("need at least two checkpoints")

    mods = []
    for p in paths:
        g = group_modules(load_file(p), p)
        mods.append({k: effective_factors(v, k, p) for k, v in g.items()})
    names = sorted(set.intersection(*(set(m) for m in mods)))

    norms = [{n: _inner(m[n], m[n]) for n in names} for m in mods]
    tot = [sum(n.values()) for n in norms]
    print(f"modules: {len(names)}")
    for i, p in enumerate(paths):
        print(
            f"[{i}] {Path(p).stem}  ||ΔW||²={tot[i]:.4g}  energy share={tot[i] / sum(tot):.1%}"
        )
    for i, j in itertools.combinations(range(len(paths)), 2):
        cs, num = [], 0.0
        for n in names:
            x = _inner(mods[i][n], mods[j][n])
            num += x
            cs.append(x / (norms[i][n] * norms[j][n]) ** 0.5)
        c = torch.tensor(cs)
        print(
            f"cos[{i},{j}]  per-module mean={c.mean():.4f}  median={c.median():.4f}  "
            f"min={c.min():.4f}  max={c.max():.4f}  full-sum={num / (tot[i] * tot[j]) ** 0.5:.4f}"
        )


if __name__ == "__main__":
    main()
