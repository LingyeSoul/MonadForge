#!/usr/bin/env python3
"""BYG bootstrap-target (ỹ_0) fidelity vs rollout NFE / sigma schedule.

The BYG cycle loss learns from ``ỹ_0`` — the *clean edited estimate* produced by
the bootstrap rollout (networks/methods/byg.py). Cut the rollout NFE too far and
``ỹ_0`` goes blurry, which silently degrades the teacher. Before committing a
long run to a reduced / warped rollout grid, measure how close its ``ỹ_0`` lands
to a high-NFE reference solve of the *same* ODE.

This rolls a single real source latent + instruction from a fixed noise seed
through several schedules and reports the relative L2 of each ``ỹ_0`` against a
high-NFE uniform reference (default 50 steps):

    uniform-10  (paper App. B.1)
    uniform-5   (naive halving — coarse t-grid + coarse tail)
    warped-5    (Anima-aware: dense in the forming band, one coarse tail step)

Anima resolves x0 by sigma~=0.45 (project_sigma_signal_resolves_by_045), so the
warped grid should track the reference far better than uniform-5 at the same NFE.
If warped-5 is within a small margin of uniform-10, the n=5 warp is safe.

Frozen base DiT only (no adapter) — this measures *integration* fidelity of the
schedule, which is what reducing NFE puts at risk; the LoRA edit delta rides on
top of the same field. Run after freeing the GPU from any live training.

Usage:
    python bench/byg/y0_fidelity.py --dit models/diffusion_models/anima-base-v1.0.safetensors
    # optional: --byg_sidecar <stem>_byg.safetensors --latent_npz <stem>_WxH_anima.npz
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench._anima import add_common_args, build_anima, resolve_dtype  # noqa: E402
from library.io.cache import load_cached_latents  # noqa: E402
from library.training.forward import (  # noqa: E402
    from_dit_5d,
    make_padding_mask,
    to_dit_5d,
)
from networks.methods.byg import (  # noqa: E402
    BYGConditioning,
    _crossattn_seqlens,
)

ROLE = "instruction"


def uniform(n: int) -> list[float]:
    return [1.0 - j / n for j in range(n + 1)]


# warped n=5: 4 fine steps over the forming band [0.40, 0.85], one coarse tail jump
WARPED5 = [1.0, 0.85, 0.70, 0.55, 0.40, 0.0]


def dit_velocity(dit, x_4d, t_B, c_emb, c_mask, padding_mask):
    v = dit(
        to_dit_5d(x_4d),
        t_B,
        c_emb,
        padding_mask=padding_mask,
        crossattn_seqlens=_crossattn_seqlens(c_mask),
    )
    return from_dit_5d(v)


@torch.no_grad()
def rollout_y0(dit, cond, x, eps, c_emb, c_mask, sigmas, device, dtype):
    """Euler-integrate eps -> ỹ_0 through ``sigmas`` with source=x (variable dσ)."""
    B = x.shape[0]
    padding_mask = make_padding_mask(x, dtype)
    src_x, src_rope = cond.encode_source(x)
    cond.set_source_precomputed(src_x, src_rope)
    y = eps.clone()
    n = len(sigmas) - 1
    for j in range(n):
        s = sigmas[j]
        dsig = sigmas[j] - sigmas[j + 1]
        s_B = torch.full((B,), s, device=device, dtype=dtype)
        v = dit_velocity(dit, y, s_B, c_emb, c_mask, padding_mask)
        y = y - dsig * v
    cond.clear_source()
    return y


def rel_l2(a, b) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def discover(args):
    sidecar = args.byg_sidecar
    if sidecar is None:
        hits = sorted(glob.glob(os.path.join(args.byg_text_dir, "*_byg.safetensors")))
        if not hits:
            sys.exit(
                f"No *_byg.safetensors under {args.byg_text_dir}; pass --byg_sidecar."
            )
        sidecar = hits[0]
    stem = os.path.basename(sidecar).replace("_byg.safetensors", "")
    latent = args.latent_npz
    if latent is None:
        hits = sorted(Path("post_image_dataset").rglob(f"{stem}_*_anima.npz"))
        if not hits:
            sys.exit(
                f"No latent npz for stem '{stem}' under post_image_dataset/; "
                "pass --latent_npz."
            )
        latent = str(hits[0])
    return sidecar, latent, stem


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--byg_sidecar", default=None, help="<stem>_byg.safetensors")
    p.add_argument("--latent_npz", default=None, help="<stem>_WxH_anima.npz")
    p.add_argument("--byg_text_dir", default="post_image_dataset/byg")
    p.add_argument("--ref_steps", type=int, default=50, help="uniform reference NFE")
    args = p.parse_args()

    sidecar, latent_path, stem = discover(args)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    bundle = build_anima(args, adapter=None, train_mode=False)
    dit = bundle.anima
    cond = BYGConditioning(dit)
    cond.apply()

    latents, _res, _h, _w = load_cached_latents(latent_path)
    x = latents.unsqueeze(0).to(device=device, dtype=dtype)  # [1,16,H,W]

    sc = load_file(sidecar)
    c_emb = sc[f"{ROLE}_emb"].unsqueeze(0).to(device=device, dtype=dtype)
    c_mask = sc.get(f"{ROLE}_mask")
    if c_mask is not None:
        c_mask = c_mask.unsqueeze(0).to(device)

    g = torch.Generator(device=device).manual_seed(args.seed)
    eps = torch.randn(x.shape, generator=g, device=device, dtype=dtype)

    schedules = {
        f"uniform-{args.ref_steps} (REF)": uniform(args.ref_steps),
        "uniform-10 (paper)": uniform(10),
        "uniform-5  (naive)": uniform(5),
        "warped-5   (Anima)": WARPED5,
    }

    print(
        f"\nstem={stem}  latent={os.path.basename(latent_path)}  "
        f"instr_tokens={c_emb.shape[1]}  dtype={dtype}"
    )
    print(f"{'schedule':<24} {'NFE':>4}  {'relL2 vs REF':>14}")
    print("-" * 48)

    ref_key = next(iter(schedules))
    y0 = {
        name: rollout_y0(dit, cond, x, eps, c_emb, c_mask, sig, device, dtype)
        for name, sig in schedules.items()
    }
    ref = y0[ref_key]
    for name, sig in schedules.items():
        nfe = len(sig) - 1
        d = rel_l2(y0[name], ref)
        tag = "  <- reference" if name == ref_key else ""
        print(f"{name:<24} {nfe:>4}  {d:>14.4e}{tag}")

    print(
        "\nRead: warped-5 relL2 close to uniform-10's  =>  the n=5 warp keeps a\n"
        "faithful ỹ_0 (safe to train). warped-5 >> uniform-10  =>  teacher degraded;\n"
        "add nodes in [0.4, 0.9] or keep uniform-10.\n"
    )


if __name__ == "__main__":
    main()
