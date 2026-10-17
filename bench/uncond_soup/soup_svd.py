#!/usr/bin/env python3
"""Rank-r SVD truncation of a dW-average LoRA soup.

Takes the same ingredients as soup.py but instead of writing the exact
rank-sum(r_i) concatenation, truncates each module's averaged dW to its top-r
singular directions (Eckart-Young: the best rank-r Frobenius approximation).
Never materializes the full dW: QR on the concatenated factors reduces the
SVD to a (R x R) core where R = sum of ingredient ranks.

Prints per-module retained-energy stats (sum sigma_kept^2 / sum sigma^2) so the
truncation loss is measured, not assumed.

Usage:
    python soup_svd.py --ckpts a.safetensors b.safetensors c.safetensors \
        --rank 16 --out soup16.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# soup.py lives in the same directory (extracted from PR #70).
import sys

sys.path.insert(0, str(Path(__file__).parent))
from soup import _effective_factors, _group_modules  # noqa: E402


def truncated_soup(
    sds: list[dict],
    rank: int,
    weights: list[float] | None = None,
    *,
    paths: list[str] | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    n = len(sds)
    if weights is None:
        weights = [1.0 / n] * n
    paths = paths or [f"<sd {i}>" for i in range(n)]
    grouped = [_group_modules(sd, p) for sd, p in zip(sds, paths)]
    module_names = set(grouped[0])
    for g, p in zip(grouped[1:], paths[1:]):
        if set(g) != module_names:
            raise ValueError(f"Ingredient module sets differ ({p}).")
    if out_dtype is None:
        out_dtype = sds[0][next(iter(sds[0]))].dtype

    soup: dict[str, torch.Tensor] = {}
    energy: dict[str, float] = {}
    for module in sorted(module_names):
        downs, ups = [], []
        for g, w, p in zip(grouped, weights, paths):
            down, up = _effective_factors(g[module], module, p)
            downs.append(down)
            ups.append(up * w)
        # dW = U_cat @ V_cat, U_cat: (out, R), V_cat: (R, in)
        u_cat = torch.cat(ups, dim=1)
        v_cat = torch.cat(downs, dim=0)
        # Exact SVD of dW via QR reduction to the (R x R) core.
        qu, ru = torch.linalg.qr(u_cat)  # (out,R),(R,R)
        qv, rv = torch.linalg.qr(v_cat.T)  # (in,R),(R,R)
        core = ru @ rv.T  # (R,R)
        cu, s, cvh = torch.linalg.svd(core)
        r = min(rank, s.shape[0])
        total = float((s**2).sum())
        kept = float((s[:r] ** 2).sum())
        energy[module] = kept / total if total > 0 else 1.0
        sqrt_s = s[:r].sqrt()
        up_t = (qu @ cu[:, :r]) * sqrt_s  # (out, r)
        down_t = (cvh[:r] @ qv.T) * sqrt_s.unsqueeze(1)  # (r, in)
        soup[f"{module}.lora_up.weight"] = up_t.to(out_dtype)
        soup[f"{module}.lora_down.weight"] = down_t.to(out_dtype)
        # alpha = rank -> scale 1, matching soup.py's convention.
        soup[f"{module}.alpha"] = torch.tensor(float(r), dtype=torch.float32)
    return soup, energy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    sds = [load_file(c) for c in args.ckpts]
    with safe_open(args.ckpts[0], framework="pt") as f:
        metadata = dict(f.metadata() or {})

    soup, energy = truncated_soup(sds, args.rank, args.weights, paths=args.ckpts)

    vals = sorted(energy.items(), key=lambda kv: kv[1])
    es = torch.tensor(list(energy.values()))
    print(
        f"retained energy @rank {args.rank}: mean {es.mean():.4f}  "
        f"min {es.min():.4f}  p10 {es.quantile(0.1):.4f}  max {es.max():.4f}"
    )
    print("worst modules:")
    for name, e in vals[:8]:
        print(f"  {e:.4f}  {name}")

    metadata["ss_network_dim"] = str(args.rank)
    metadata["ss_network_alpha"] = str(float(args.rank))
    metadata["ss_soup_ingredients"] = json.dumps(
        {
            "ckpts": [Path(c).name for c in args.ckpts],
            "weights": args.weights or [1.0 / len(args.ckpts)] * len(args.ckpts),
            "svd_rank": args.rank,
            "retained_energy_mean": round(float(es.mean()), 6),
            "retained_energy_min": round(float(es.min()), 6),
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(soup, str(out), metadata=metadata)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
