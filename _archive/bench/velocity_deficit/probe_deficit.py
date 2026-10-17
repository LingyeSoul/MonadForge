"""Velocity-Deficit Phase-0 probe — does Anima's flow field underestimate the
transport speed, and is the deficit shaped like "The Velocity Deficit" (1798) on
ImageNet/SiT?

Background (paper 1798 — Initial Energy Injection for Flow Matching): the FM MSE
objective learns ``v* = E[v_target | x_t]``. Jensen on the strictly-convex
squared norm gives ``‖E[v]‖² < E[‖v‖²]`` whenever the conditional target has
variance (independent x0/x1 coupling), so the learned velocity *magnitude* is
systematically too small ("Velocity Deficit"). The paper claims this contraction
is harmful at the **noise end** (signal starvation → integration lag) and
benign at the **data end** (implicit denoising), and patches it two ways:

  - SSC (inference): scale velocity by γ(t)=s_start·(1−t)+s_end·t, 1.1→1.0.
  - MAFM (training): aux loss (‖v_θ‖−‖x1−x0‖)² with λ(t)=0.2·(1−t).

This probe is the decision gate for adopting *either* on Anima. It reproduces the
paper's Figure-2 measurement on real cached latents + real captions, with exact
σ control (no sampler), and reports whether the deficit exists, its σ-shape, and
how much of it is a *magnitude* problem (correctable by a γ scale — SSC/MAFM's
whole premise) vs a *direction* problem (not).

Anima convention (``library/runtime/noise.py``): the DiT time arg is σ directly;
``x_t = (1−σ)·x1 + σ·ε`` with x1=data latent, ε=noise; FM target ``v = ε − x1``.
So **σ=1 is the noise end** (paper's t=0, where the deficit is harmful) and σ=0
is the data end. The target ``ε − x1`` is σ-independent, so its mean norm is the
flat "training target magnitude" dashed line in Figure 2; the learned ‖v_θ‖
varying below it across σ is the deficit.

Per (real latent x1, its real caption embed c) and per σ we log, in float32:

    vnorm      = ‖v_θ(x_t, σ, c)‖                      (per-sample L2)
    tnorm      = ‖ε − x1‖
    cos        = cos(v_θ, ε − x1)                       (directional accuracy)
    proj       = ⟨v_θ, (ε−x1)/tnorm⟩                    (transport speed along truth)
    ratio      = vnorm / tnorm                          (<1 ⇒ magnitude deficit)
    proj_ratio = proj / tnorm                           (delivered / required speed)

A pure magnitude deficit (the paper's claim) shows ratio<1 with cos≈unchanged; a
direction problem shows cos falling. ``ideal_gamma = 1/proj_ratio`` is the per-σ
scale that would restore transport — compared against the paper's fixed linear
γ(σ)=s_start·σ+s_end·(1−σ) to see if SSC's shape even matches Anima's deficit.

Pass ``--adapter`` to probe a LoRA/T-LoRA/Hydra checkpoint (the router setters
fire so FEI/content-routed adapters condition correctly); omit it for the base
DiT. Run both and diff the curves to see whether a LoRA changes the deficit — the
actual question behind "integrate MAFM like REPA into the LoRA stack".

Usage:
    uv run python bench/velocity_deficit/probe_deficit.py \
        --num_samples 64 --seeds 2 --label base
    uv run python bench/velocity_deficit/probe_deficit.py \
        --adapter output/ckpt/anima_artist.safetensors --label artist
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.inference.adapters import (  # noqa: E402
    compute_and_set_hydra_fei,
    set_hydra_crossattn,
    set_hydra_sigma,
)
from library.io.cache import (  # noqa: E402
    discover_cached_pairs,
    load_cached_crossattn_emb,
    load_cached_latents,
)

# σ grid — denser at the noise end (σ→1), where the paper says the deficit bites
# (integration lag / signal starvation). 0 and 1 are excluded (degenerate: at
# σ=0 x_t is the clean latent, at σ=1 it is pure noise carrying no signal).
DEFAULT_SIGMAS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]


def _per_sample(x: torch.Tensor) -> torch.Tensor:
    """Flatten all but batch → (B,) float32."""
    return x.float().flatten(1)


def probe_one(
    anima,
    latent: torch.Tensor,  # (C, H, W) float32, data x1
    crossattn: torch.Tensor,  # (S, D) float32, the caption embed c
    sigmas: list[float],
    *,
    device: torch.device,
    seeds: int,
    base_seed: int,
) -> list[dict]:
    """Run the DiT at each σ on x_t built from a real latent + fresh noise."""
    rows: list[dict] = []
    x1 = latent.to(device)[None]  # (1, C, H, W)
    h_lat, w_lat = x1.shape[-2], x1.shape[-1]
    emb = crossattn.to(device, dtype=torch.bfloat16)[None]  # (1, S, D)
    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    for s in range(seeds):
        g = torch.Generator(device=device).manual_seed(base_seed + s)
        eps = torch.randn(x1.shape, generator=g, device=device, dtype=torch.float32)
        target = eps - x1  # FM target v = ε − x1 (σ-independent)
        t_flat = _per_sample(target)
        tnorm = t_flat.norm(dim=1)  # (1,)

        for sigma in sigmas:
            x_t = (1.0 - sigma) * x1 + sigma * eps  # (1, C, H, W)
            x_t5 = x_t.unsqueeze(2).to(torch.bfloat16)  # (1, C, 1, H, W) — DiT 5D
            t_arg = torch.full((1,), sigma, device=device, dtype=torch.bfloat16)

            # Router conditioning (no-op on base DiT / plain LoRA; load-bearing
            # for FEI/content/σ-routed Hydra adapters so v_θ is conditioned the
            # same way the sampler conditions it).
            set_hydra_sigma(anima, t_arg)
            set_hydra_crossattn(anima, emb)
            compute_and_set_hydra_fei(anima, x_t5)

            with torch.no_grad():
                v = anima(x_t5, t_arg, emb, padding_mask=padding_mask)
            v = v.squeeze(2)  # 5D → 4D (target dim 2 explicitly)

            v_flat = _per_sample(v)
            vnorm = v_flat.norm(dim=1)
            proj = (v_flat * t_flat).sum(dim=1) / tnorm.clamp_min(1e-8)
            cos = proj / vnorm.clamp_min(1e-8)
            rows.append(
                {
                    "sigma": sigma,
                    "seed": s,
                    "vnorm": float(vnorm),
                    "tnorm": float(tnorm),
                    "cos": float(cos),
                    "proj": float(proj),
                    "ratio": float(vnorm / tnorm.clamp_min(1e-8)),
                    "proj_ratio": float(proj / tnorm.clamp_min(1e-8)),
                }
            )
    return rows


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def aggregate(rows: list[dict], s_start: float, s_end: float) -> dict:
    """Per-σ curves + the SSC-vs-ideal magnitude comparison."""
    curve = []
    for sigma in sorted({r["sigma"] for r in rows}):
        sel = [r for r in rows if r["sigma"] == sigma]
        proj_ratio = _mean([r["proj_ratio"] for r in sel])
        # paper's linear γ in Anima's σ coordinate (σ=1 noise → s_start):
        gamma_ssc = s_start * sigma + s_end * (1.0 - sigma)
        ideal_gamma = 1.0 / proj_ratio if proj_ratio > 1e-6 else float("inf")
        curve.append(
            {
                "sigma": round(sigma, 4),
                "vnorm": round(_mean([r["vnorm"] for r in sel]), 5),
                "tnorm": round(_mean([r["tnorm"] for r in sel]), 5),
                "ratio": round(_mean([r["ratio"] for r in sel]), 5),
                "cos": round(_mean([r["cos"] for r in sel]), 5),
                "proj_ratio": round(proj_ratio, 5),
                "ideal_gamma": round(ideal_gamma, 5),
                "gamma_ssc": round(gamma_ssc, 5),
                # residual transport gap after applying the paper's fixed SSC γ:
                "proj_ratio_after_ssc": round(proj_ratio * gamma_ssc, 5),
                "n": len(sel),
            }
        )
    ratios = [c["ratio"] for c in curve]
    worst = min(curve, key=lambda c: c["ratio"])
    # Where does the deficit live? Noise end (σ≥0.7) vs data end (σ≤0.3).
    noise_end = _mean([c["ratio"] for c in curve if c["sigma"] >= 0.7])
    data_end = _mean([c["ratio"] for c in curve if c["sigma"] <= 0.3])
    return {
        "curve": curve,
        "deficit_present": all(r < 1.0 for r in ratios),
        "mean_ratio": round(_mean(ratios), 5),
        "noise_end_ratio": round(noise_end, 5),  # σ≥0.7 — paper's harmful regime
        "data_end_ratio": round(data_end, 5),  # σ≤0.3 — paper's benign regime
        "worst_sigma": worst["sigma"],
        "worst_ratio": worst["ratio"],
        "mean_cos": round(_mean([c["cos"] for c in curve]), 5),
        "note": "ratio=‖v‖/‖target‖ (<1 = magnitude deficit). proj_ratio = "
        "transport speed delivered/required. cos isolates direction. σ=1 is the "
        "noise end (paper t=0). ideal_gamma=1/proj_ratio is the per-σ magnitude "
        "fix; compare gamma_ssc (paper's fixed 1.1→1.0 linear).",
    }


def maybe_plot(agg: dict, out_dir: Path, label: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    c = agg["curve"]
    sig = [x["sigma"] for x in c]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].axhline(1.0, color="0.5", lw=0.8, ls="--", label="target (no deficit)")
    ax[0].plot(sig, [x["ratio"] for x in c], "o-", label="‖v‖/‖target‖")
    ax[0].plot(sig, [x["proj_ratio"] for x in c], "s-", label="proj/‖target‖")
    ax[0].plot(sig, [x["cos"] for x in c], "^-", color="0.6", label="cos(v,target)")
    ax[0].set_xlabel("σ  (0 = data end · 1 = noise end)")
    ax[0].set_ylabel("ratio")
    ax[0].set_title(f"velocity deficit — {label}")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=9)
    ax[1].plot(sig, [x["ideal_gamma"] for x in c], "o-", label="ideal γ = 1/proj_ratio")
    ax[1].plot(sig, [x["gamma_ssc"] for x in c], "s-", label="paper SSC γ (1.1→1.0)")
    ax[1].axhline(1.0, color="0.5", lw=0.8, ls="--")
    ax[1].set_xlabel("σ  (0 = data end · 1 = noise end)")
    ax[1].set_ylabel("velocity scale γ")
    ax[1].set_title("magnitude correction: ideal vs paper's SSC")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / "velocity_deficit.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--adapter", default=None, help="Optional LoRA/Hydra checkpoint.")
    p.add_argument("--adapter_multiplier", type=float, default=1.0)
    add_common_args(p)
    p.add_argument(
        "--cache_dir",
        default="post_image_dataset/lora",
        help="Dir of cached latent+TE pairs (real data + real captions).",
    )
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--seeds", type=int, default=2, help="Fresh-noise draws per sample.")
    p.add_argument(
        "--sigmas",
        default=None,
        help="Comma-separated σ grid (default: noise-end-dense 0.05..0.99).",
    )
    p.add_argument("--s_start", type=float, default=1.1, help="SSC γ at σ=1 (noise).")
    p.add_argument("--s_end", type=float, default=1.0, help="SSC γ at σ=0 (data).")
    return p.parse_args()


def main() -> None:
    args = build_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    sigmas = (
        [float(s) for s in args.sigmas.split(",")] if args.sigmas else DEFAULT_SIGMAS
    )

    # Real (latent, caption) pairs — the repo's "real captions mandatory" rule for
    # velocity-norm probes; mismatched captions measure off-manifold velocity.
    pairs = [p for p in discover_cached_pairs(args.cache_dir) if p.te_path]
    if not pairs:
        raise SystemExit(f"no latent+TE pairs under {args.cache_dir}")
    pairs = pairs[: args.num_samples]

    bundle = build_anima(
        args,
        adapter=args.adapter,
        train_mode=False,
        multiplier=args.adapter_multiplier,
    )
    anima = bundle.anima

    rows: list[dict] = []
    for i, pair in enumerate(pairs):
        latent, _res, _h, _w = load_cached_latents(pair.npz_path)
        emb = load_cached_crossattn_emb(pair.te_path)
        if emb is None:
            print(f"  skip {pair.stem}: no crossattn_emb", flush=True)
            continue
        rows.extend(
            probe_one(
                anima,
                latent,
                emb,
                sigmas,
                device=device,
                seeds=args.seeds,
                base_seed=1000 + i * 17,
            )
        )
        if (i + 1) % 8 == 0:
            print(f"[{i + 1}/{len(pairs)}] {pair.stem}", flush=True)

    agg = aggregate(rows, args.s_start, args.s_end)
    out_dir = make_run_dir("velocity_deficit", args.label)
    plot = maybe_plot(
        agg, out_dir, args.label or ("adapter" if args.adapter else "base")
    )

    with (out_dir / "per_sample.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sigma",
                "seed",
                "vnorm",
                "tnorm",
                "cos",
                "proj",
                "ratio",
                "proj_ratio",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    write_result(
        out_dir,
        script="bench/velocity_deficit/probe_deficit.py",
        label=args.label,
        args=args,
        device=device,
        metrics={
            "adapter": args.adapter,
            "n_samples": len(pairs),
            "n_seeds": args.seeds,
            "n_rows": len(rows),
            "sigmas": sigmas,
            **agg,
        },
        artifacts=["per_sample.csv"] + ([plot] if plot else []),
    )

    print(f"\ndeficit_present={agg['deficit_present']}  mean_ratio={agg['mean_ratio']}")
    print(
        f"noise-end (σ≥0.7) ratio={agg['noise_end_ratio']}  "
        f"data-end (σ≤0.3) ratio={agg['data_end_ratio']}  "
        f"mean_cos={agg['mean_cos']}"
    )
    print(f"worst: σ={agg['worst_sigma']} ratio={agg['worst_ratio']}")
    print(
        f"\n{'σ':>6} {'ratio':>8} {'proj_r':>8} {'cos':>8} {'ideal_γ':>8} {'ssc_γ':>8} {'post_ssc':>9}"
    )
    for c in agg["curve"]:
        print(
            f"{c['sigma']:>6} {c['ratio']:>8} {c['proj_ratio']:>8} {c['cos']:>8} "
            f"{c['ideal_gamma']:>8} {c['gamma_ssc']:>8} {c['proj_ratio_after_ssc']:>9}"
        )
    print(f"\nresults -> {out_dir}")


if __name__ == "__main__":
    main()
