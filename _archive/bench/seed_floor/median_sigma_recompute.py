"""Offline: recompute CMMD from already-rendered PNGs under median-heuristic sigma.

No re-render. Encodes saved gens + live refs through PE-Core once, then sweeps
sigma. Median sigma is taken from the REAL holdout pool only (model-independent)
so the kernel bandwidth is fixed across every model comparison.
"""
import sys
from pathlib import Path
from random import Random
from types import SimpleNamespace

import torch
from PIL import Image

sys.path.insert(0, str(Path.cwd()))
from bench.memorization.eyeball import _pick  # noqa: E402
from bench.memorization.generalize import (  # noqa: E402
    split_items, to_prompts, _pil_to_minus1to1, encode_pool,
)
from library.vision.encoder import load_pe_encoder  # noqa: E402
from library.training.cmmd import mmd_gaussian  # noqa: E402

args = SimpleNamespace(
    membership_adapter="output/ckpt/bench_sincos_half_plain.safetensors",
    method="lora", preset="default",
    num_prompts=24, num_refs=96, pick_seed=0,
)
rng = Random(args.pick_seed)
members, holdouts, target = split_items(args)
prompts = to_prompts(holdouts, args.num_prompts, rng)
prompt_paths = {p.ref_path for p in prompts}
ref_holdout_items = [it for it in holdouts if Path(it[0].absolute_path) not in prompt_paths]
ref_hold = _pick(ref_holdout_items, args.num_refs, rng)
ref_memb = _pick(members, args.num_refs, rng)
print(f"target={target} ref_hold={len(ref_hold)} ref_memb={len(ref_memb)}", flush=True)

device = torch.device("cuda")
bundle = load_pe_encoder(device)

def enc_paths(paths):
    return encode_pool(bundle, [_pil_to_minus1to1(Image.open(p)) for p in paths])

ref_hold_pool = enc_paths([it[0].absolute_path for it in ref_hold])
ref_memb_pool = enc_paths([it[0].absolute_path for it in ref_memb])

RUN = "bench/reft/results/20260706-1210-seed_floor_cfg4/renders"
REFT = "bench/reft/results/20260705-2242-reft_phase1p_cfg4/renders"
gen_dirs = {
    "base":     f"{RUN}/base",
    "plain":    f"{REFT}/bench_sincos_half_plain",
    "s1001":    f"{RUN}/bench_sincos_half_uncondpoolft",
    "s1002":    f"{RUN}/bench_sincos_half_uncondpoolft_s1002",
    "s1003":    f"{RUN}/bench_sincos_half_uncondpoolft_s1003",
    "soup_r48": f"{RUN}/bench_sincos_half_uncondpoolsoup3",
}
gen_pools = {}
for k, d in gen_dirs.items():
    pngs = sorted(Path(d).glob("*.png"))
    gen_pools[k] = enc_paths(pngs)
    print(f"  gen {k}: {len(pngs)} pngs", flush=True)

def median_pairwise_dist(pool):
    x = pool.to(torch.float32)
    xsq = (x * x).sum(1)
    d2 = (xsq[:, None] + xsq[None, :] - 2 * x @ x.t()).clamp_min(0)
    n = len(x)
    iu = torch.triu_indices(n, n, 1)
    return d2[iu[0], iu[1]].sqrt().median().item()

sig_med = median_pairwise_dist(ref_hold_pool)
print(f"\nmedian-heuristic sigma (from real holdout pool) = {sig_med:.4f}")
print(f"(compare: paper constant sigma = 10.0)\n")

half = len(ref_hold_pool) // 2
for sig in [10.0, sig_med]:
    floor = mmd_gaussian(ref_hold_pool[:half], ref_hold_pool[half:], sigma=sig).item()
    print(f"=== sigma={sig:.4f}   noise_floor(real 48-vs-48)={floor:.4f} ===")
    rows = []
    for k, gp in gen_pools.items():
        h = mmd_gaussian(ref_hold_pool, gp, sigma=sig).item()
        m = mmd_gaussian(ref_memb_pool, gp, sigma=sig).item()
        rows.append((k, h, m, h / floor if floor else float("nan")))
    for k, h, m, r in sorted(rows, key=lambda z: z[1]):
        flag = "  <-- BELOW FLOOR" if h < floor else ""
        print(f"   {k:9s} holdout={h:.4f}  member={m:.4f}  h/floor={r:.2f}{flag}")
    print()
