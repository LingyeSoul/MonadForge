#!/usr/bin/env python3
"""SCFM Phase 0 — base-field consistency-residual scan (Eq. 11 headroom).

The SCFM proposal (`_archive/proposals/turbo_scfm.md` §2 — line CLOSED, verdict in
`docs/findings/scfm_velocity_fidelity_closed.md`) gated Phase 1 on two reads.
The image arms (28-step teacher / naive 4-step teacher / DP-DMD student) answer
"is the few-step teacher-fidelity ceiling high enough to beat the student". This
probe answers the *complementary* question: **is there headroom for SCFM's
straightening term (Term B) at the student step sizes** — i.e. on the base
velocity field, how far is one coarse Euler step from the composition of two
finer sub-steps across the student σ-grid.

For each coarse interval ``[σ_a, σ_c]`` of the student grid we split it at the
midpoint ``σ_b = (σ_a + σ_c)/2`` and, on a renoised real latent ``x_a``:

    v1        = V(x_a, σ_a)                       # base field, cond, no-grad
    x_b       = x_a - (σ_a - σ_b)·v1              # one fine Euler sub-step
    v2        = V(x_b, σ_b)
    v_target  = ((σ_a-σ_b)·v1 + (σ_b-σ_c)·v2) / (σ_a - σ_c)   # SCFM Eq. 12
    curvature = ‖v1 - v2‖ / ‖v1‖                  # field bend over the interval
    residual  = ‖v1 - v_target‖ / ‖v1‖           # = (σ_b-σ_c)/(σ_a-σ_c)·curvature

``residual`` is exactly the relative Eq.-11 consistency error a coarse Euler step
incurs vs the two-sub-step roll — the quantity Term B drives to zero. Read:
**large residual (esp. in the low-σ band where Anima text/detail resolves) ⇒
real straightening headroom ⇒ SCFM adds over naive Euler; near-zero everywhere ⇒
the field is already straight and SCFM ≈ naive 4-step teacher** (then the image
arms alone decide GO/NO-GO).

Cond is the image's *real* cached caption (per-image), matching how the field is
used at sampling time. Base DiT only — no adapter.

    python bench/turbo/probe_consistency_residual.py --n_images 24 --n_noise 2
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench._anima import add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.runtime.harness import build_anima  # noqa: E402
from library.io.cache import load_cached_latents, load_cached_text_features  # noqa: E402
from library.inference.sampling import get_timesteps_sigmas  # noqa: E402
from library.training.forward.renoise import renoise  # noqa: E402


def find_pairs(root: str, limit: int) -> list[tuple[str, str]]:
    """Deterministic (sorted) latent+TE cache pairs under ``root``."""
    npzs = sorted(glob.glob(os.path.join(root, "**", "*_anima.npz"), recursive=True))
    pairs = []
    for n in npzs:
        m = re.match(r"(.*)_(\d+x\d+)_anima\.npz$", n)
        if not m:
            continue
        te = m.group(1) + "_anima_te.safetensors"
        if os.path.exists(te):
            pairs.append((n, te))
        if len(pairs) >= limit:
            break
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    add_model_args(ap, vae=False, text_encoder=False)
    ap.add_argument("--cache_root", default="post_image_dataset/lora")
    ap.add_argument("--n_images", type=int, default=24)
    ap.add_argument(
        "--n_noise", type=int, default=2, help="noise renoises per (image,interval)"
    )
    ap.add_argument("--student_steps", type=int, default=4)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label", default="scfm-consistency-residual")
    args = ap.parse_args()

    device = torch.device(args.device)
    bundle = build_anima(args, adapter=None, train_mode=False)
    model, dtype = bundle.anima, bundle.dtype

    sigmas = get_timesteps_sigmas(args.student_steps, args.flow_shift, "cpu")[
        1
    ].tolist()
    # coarse intervals of the student grid, each split at its midpoint
    intervals = [(sigmas[i], sigmas[i + 1]) for i in range(len(sigmas) - 1)]

    @torch.no_grad()
    def vfield(x_4d: torch.Tensor, sig: float, c: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x_4d.shape
        t_b = torch.full((B,), sig, device=device, dtype=dtype)
        pad = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        with torch.autocast("cuda", dtype=dtype):
            v = model.forward_mini_train_dit(
                x_4d.unsqueeze(2).to(dtype),
                t_b,
                c,
                padding_mask=pad,
                skip_pooled_text_proj=True,
            )
        return v.squeeze(2).float()

    pairs = find_pairs(args.cache_root, args.n_images)
    if not pairs:
        raise SystemExit(f"no latent+TE pairs under {args.cache_root}")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    # accumulators per interval
    acc = {i: {"curv": [], "resid": []} for i in range(len(intervals))}

    for npz, te in pairs:
        lat, _res, _h, _w = load_cached_latents(npz)  # (16,H,W)
        c1, _pooled = load_cached_text_features(te)  # (S,1024)
        if c1 is None:
            continue
        x0 = lat.unsqueeze(0).to(device).float()  # (1,16,H,W)
        c = c1.unsqueeze(0).to(device, dtype=dtype)  # (1,S,1024)
        for ii, (a, cval) in enumerate(intervals):
            b = 0.5 * (a + cval)
            for _ in range(args.n_noise):
                eps = torch.randn(x0.shape, generator=gen, device=device).float()
                t = torch.full((x0.shape[0],), a, device=device)
                x_a = renoise(x0, t, eps)  # (1,16,H,W)
                v1 = vfield(x_a, a, c)
                x_b = x_a - (a - b) * v1
                v2 = vfield(x_b, b, c)
                w1, w2 = (a - b), (b - cval)
                v_target = (w1 * v1 + w2 * v2) / (a - cval)
                n1 = v1.norm().clamp_min(1e-8)
                curv = (v1 - v2).norm() / n1
                resid = (v1 - v_target).norm() / n1
                acc[ii]["curv"].append(curv.item())
                acc[ii]["resid"].append(resid.item())

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    bands = []
    print(f"\n{'σ_a':>6} {'σ_b':>6} {'σ_c':>6}  {'curvature':>10} {'residual':>10}  n")
    for ii, (a, cval) in enumerate(intervals):
        b = 0.5 * (a + cval)
        cu, re_ = mean(acc[ii]["curv"]), mean(acc[ii]["resid"])
        n = len(acc[ii]["curv"])
        print(f"{a:6.3f} {b:6.3f} {cval:6.3f}  {cu:10.4f} {re_:10.4f}  {n}")
        bands.append(
            {
                "sigma_a": a,
                "sigma_b": b,
                "sigma_c": cval,
                "curvature_rel": cu,
                "residual_rel": re_,
                "n": n,
            }
        )

    overall_resid = mean([v for ii in acc for v in acc[ii]["resid"]])
    overall_curv = mean([v for ii in acc for v in acc[ii]["curv"]])
    low_sigma = bands[-1]["residual_rel"]  # [σ=0.5 → 0] band (text/detail)
    print(
        f"\noverall residual_rel = {overall_resid:.4f}  | low-σ band [0.5→0] residual_rel = {low_sigma:.4f}"
    )

    run_dir = make_run_dir("turbo", label=args.label)
    write_result(
        run_dir,
        script=__file__,
        args=vars(args),
        metrics={
            "student_sigmas": sigmas,
            "bands": bands,
            "overall_residual_rel": overall_resid,
            "overall_curvature_rel": overall_curv,
            "low_sigma_band_residual_rel": low_sigma,
            "n_images_used": len(pairs),
        },
        device=device,
    )
    print(f"\nresult.json → {run_dir}")


if __name__ == "__main__":
    main()
