"""Pairwise ΔW cosine + energy share between LoRA ingredients.

<ΔW_i, ΔW_j> = tr((U_i^T U_j)(D_j D_i^T)) — no ΔW materialized.
Usage: python project/moss_lora/src/dw_cos.py a.safetensors b.safetensors [c ...]
Prints per-module mean / median cos and the full-sum (all modules concatenated) cos.
"""

import itertools
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.soup.build import _effective_factors, _group_modules  # noqa: E402

paths = sys.argv[1:]
mods = []
for p in paths:
    g = _group_modules(load_file(p), p)
    mods.append({k: _effective_factors(v, k, p) for k, v in g.items()})
names = sorted(set.intersection(*(set(m) for m in mods)))


def inner(a, b):
    (da, ua), (db, ub) = a, b
    return float(
        ((ua.T.double() @ ub.double()) * (da.double() @ db.T.double()).T).sum()
    )


norms = [{n: inner(m[n], m[n]) for n in names} for m in mods]
tot = [sum(n.values()) for n in norms]
print(f"modules: {len(names)}")
for i, p in enumerate(paths):
    print(
        f"[{i}] {Path(p).stem}  ||ΔW||²={tot[i]:.4g}  energy share={tot[i] / sum(tot):.1%}"
    )
for i, j in itertools.combinations(range(len(paths)), 2):
    cs, num = [], 0.0
    for n in names:
        x = inner(mods[i][n], mods[j][n])
        num += x
        cs.append(x / (norms[i][n] * norms[j][n]) ** 0.5)
    c = torch.tensor(cs)
    print(
        f"cos[{i},{j}]  per-module mean={c.mean():.4f}  median={c.median():.4f}  "
        f"min={c.min():.4f}  max={c.max():.4f}  full-sum={num / (tot[i] * tot[j]) ** 0.5:.4f}"
    )
