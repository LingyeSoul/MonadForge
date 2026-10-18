#!/usr/bin/env python3
"""CNS γ(f, t) for MOSS-Image, from deterministic Euler trajectories (Alg. 2).

Per prompt × seed: run the shipped sampler (Euler, CFG, uncond skip 22) and record
each step's guided x̂₀; γ = 1 − |X̂₀ − X₀|² / |X₀|² on the un-patchified grid,
radially binned (``library.moss.cns.gamma_for_trajectory``), averaged over
trajectories. One γ row per ``--sizes`` entry; ``MossCNS`` picks the row nearest
in aspect ratio, so the sizes must have distinct aspect ratios. Also prints the
precondition per size: σ at γ = 0.5 per band (the "staircase" — low bands must
resolve at higher σ than high bands, or CNS has nothing to reallocate).

    make daemon-run ARGS="bench/moss_spectrum/cns_calibrate.py --compile"

Writes the npz ``library.moss.cns.MossCNS.from_npz`` reads (default
``networks/calibration/moss_cns_gamma.npz``) plus a copy and ``result.json`` in the run dir.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling run_bench
import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from run_bench import PROMPTS  # noqa: E402


def sigma_at_half(col: np.ndarray, sig: np.ndarray) -> float:
    """σ where γ first crosses 0.5 (``sig`` descending), linear interp."""
    above = col >= 0.5
    if not above.any():
        return 0.0
    i = int(np.argmax(above))
    if i == 0:
        return float(sig[0])
    g0, g1 = col[i - 1], col[i]
    return float(sig[i - 1] + (0.5 - g0) / (g1 - g0 + 1e-12) * (sig[i] - sig[i - 1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="base")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument(
        "--sizes",
        default="1024x1024",
        help="comma-separated HxW pixel sizes (multiples of 16, distinct aspect ratios)",
    )
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--n_bins", type=int, default=32)
    ap.add_argument("--out", default="networks/calibration/moss_cns_gamma.npz")
    ap.add_argument("--model_dir", default="")
    ap.add_argument(
        "--te_cache", default="/media/sorryhyun/data/moss_image/bench/_te_cache"
    )
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    start_heartbeat(label="moss_cns_calibrate")

    from library.env import resolve_under_home
    from library.moss.cns import gamma_for_trajectory
    from library.moss.generate import _cached_embeds, denoise, initial_noise
    from library.moss.loader import compile_blocks, load_transformer, pad_embeds
    from library.moss.model import DEFAULT_UNCOND_SKIP, snapped_sigmas
    from library.moss.requests import resolve_model_dir

    keys = [k for k in args.prompts.split(",") if k]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",") if s]
    if any(h % 16 or w % 16 for h, w in sizes):
        raise SystemExit(f"--sizes must be multiples of 16: {args.sizes}")
    ars = [round(w / h, 4) for h, w in sizes]
    if len(set(ars)) != len(ars):
        raise SystemExit(f"--sizes need distinct aspect ratios (W/H {ars})")
    model_dir = resolve_model_dir(args.model_dir)
    run_dir = make_run_dir("moss_cns", label=args.label)
    device = torch.device("cuda")

    embeds = _cached_embeds(
        model_dir, [PROMPTS[k] for k in keys] + [""], Path(args.te_cache)
    )
    uncond, uncond_mask = pad_embeds([embeds[""]], device=device)
    transformer = load_transformer(model_dir, args.model, device=device)
    if args.compile:
        compile_blocks(transformer.layers)

    sigmas = snapped_sigmas(args.steps)
    sig_mid = sigmas.float().numpy()
    gammas, per_size = [], {}
    for h, w in sizes:
        per_traj = []
        for key in keys:
            cond, cond_mask = pad_embeds([embeds[PROMPTS[key]]], device=device)
            for seed in seeds:
                preds = []
                z = denoise(
                    transformer,
                    initial_noise(seed, h, w).to(device),
                    cond,
                    cond_mask,
                    uncond,
                    uncond_mask,
                    sigmas,
                    cfg=args.cfg,
                    uncond_skip=DEFAULT_UNCOND_SKIP,
                    on_step=lambda i, x0: preds.append(x0.clone()),
                )
                per_traj.append(gamma_for_trajectory(preds, z, args.n_bins))
                print(f"[cns] {h}x{w} {key} s{seed}: {len(preds)} steps", flush=True)

        gamma = np.mean(np.stack(per_traj), 0)  # (T, F)
        gammas.append(gamma)
        s50 = [sigma_at_half(gamma[:, f], sig_mid) for f in range(args.n_bins)]
        per_size[f"{h}x{w}"] = dict(
            sigma50_per_band=s50,
            spread_low_high=s50[0] - s50[-1],
            aggregate_sigma50=sigma_at_half(gamma.mean(1), sig_mid),
            n_trajectories=len(per_traj),
        )
        print(
            f"[{h}x{w}] σ@γ=0.5 per band (low→high): "
            + " ".join(f"{s:.2f}" for s in s50),
            flush=True,
        )
        print(
            f"[{h}x{w}] spread low−high {s50[0] - s50[-1]:+.3f}; "
            f"aggregate σ50 {per_size[f'{h}x{w}']['aggregate_sigma50']:.3f}",
            flush=True,
        )

    s50_all = np.array([m["sigma50_per_band"] for m in per_size.values()])
    cross_mad = float(np.abs(s50_all - s50_all.mean(0)).mean())
    print(f"cross-size σ50 MAD {cross_mad:.4f}", flush=True)

    payload = dict(
        gamma=np.stack(gammas).astype(np.float32),  # (A, T, F)
        aspects=np.array(sizes, dtype=np.int64),  # (A, 2) pixel (H, W)
        sigmas=np.append(sig_mid, 0.0).astype(np.float32),
        radial_centers=(np.arange(args.n_bins) + 0.5) / args.n_bins,
        cfg=np.float32(args.cfg),
        steps=np.int64(args.steps),
        grid=np.array([(h // 8, w // 8) for h, w in sizes], dtype=np.int64),
    )
    out = resolve_under_home(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    shutil.copy(out, run_dir / "cns_gamma.npz")
    print(f"gamma -> {out}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        artifacts=["cns_gamma.npz"],
        metrics=dict(
            per_size=per_size,
            cross_size_sigma50_mad=cross_mad,
            gamma_out=str(out),
        ),
    )


if __name__ == "__main__":
    main()
