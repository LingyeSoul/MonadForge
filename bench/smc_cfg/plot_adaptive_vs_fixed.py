"""Plot SMC-CFG controller budget at CFG=4 (production).

Two-panel figure rendered from a `measure_error_magnitude.py` per_step.csv:

  Top — instantaneous per-step magnitudes
    * intra-prompt voxel-spread envelope (`e_p50` → `e_p95`, averaged
      across prompts)
    * per-prompt `e_mean` thin lines (inter-prompt variability)
    * cross-prompt `|e_t|.mean` thick line
    * fixed k = {0.02, 0.1} reference levels

  Bottom — cumulative L1 drift budget over the trajectory
    Σ_{t' ≤ t} |Δσ_{t'}| · w · |Δe_{t'}|     (per voxel, w = CFG_focus)

    For SMC the per-element switching term has |Δe| = k_t uniformly across
    voxels (Δe = −k_t·sign(s)), so this is the controller's L1 intervention
    integrated through the Euler schedule. Plotted against the natural
    signal accumulation `Σ |Δσ|·w·|e_t|.mean` — the same integral with
    |e_t| in place of k_t — which is how big the actual CFG correction
    above v_uncond is over the whole trajectory.

    * α-adaptive (`k_t = α · |e_t|.mean`): accumulation = α · signal by
      construction — stays at a fixed ratio across the whole trajectory.
    * fixed k = 0.02 / 0.1: grows linearly with Σ|Δσ| regardless of |e|;
      ends up several × the natural signal even when the instantaneous
      gap looks modest.

The top panel says "fixed-k overshoots |e| in the σ ≈ 0.2–0.4 plateau."
The bottom panel says "and that overshoot integrates over 28 steps into
controller drift larger than the actual semantic correction."

Run:
    uv run python bench/smc_cfg/plot_adaptive_vs_fixed.py \\
        --results bench/smc_cfg/results/20260518-1014-cfg-sweep-4p \\
        --alpha 0.2
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_FIELDS = ("e_mean", "e_p50", "e_p95")


def load_per_step(csv_path: Path, cfg_focus: float):
    """Returns {field: (n_prompts, n_steps) array} at the given CFG, plus sigma per step."""
    grid: dict = defaultdict(lambda: defaultdict(dict))
    sigmas: dict[int, float] = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if float(r["cfg"]) != cfg_focus:
                continue
            step = int(r["step"])
            pi = int(r["prompt_idx"])
            sigmas[step] = float(r["sigma"])
            for fld in _FIELDS:
                grid[fld][step][pi] = float(r[fld])

    if not sigmas:
        raise SystemExit(f"no rows at cfg={cfg_focus} in {csv_path}")

    n_steps = max(sigmas) + 1
    n_prompts = max(max(d) for d in grid["e_mean"].values()) + 1
    out: dict = {}
    for fld in _FIELDS:
        arr = np.full((n_prompts, n_steps), np.nan)
        for step, by_prompt in grid[fld].items():
            for pi, v in by_prompt.items():
                arr[pi, step] = v
        out[fld] = arr
    sigma_arr = np.array([sigmas[i] for i in range(n_steps)])
    return out, sigma_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        type=Path,
        default=Path("bench/smc_cfg/results/20260518-1014-cfg-sweep-4p"),
    )
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--ks", type=float, nargs="+", default=[0.02, 0.1])
    ap.add_argument("--cfg_focus", type=float, default=4.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    csv_path = args.results / "per_step.csv"
    out_png = args.out or (args.results / "adaptive_vs_fixed.png")

    data, sigmas = load_per_step(csv_path, args.cfg_focus)
    steps = np.arange(len(sigmas))

    e_mean = data["e_mean"]
    e_p50  = data["e_p50"]
    e_p95  = data["e_p95"]

    e_mean_avg = np.nanmean(e_mean, axis=0)
    p50_avg = np.nanmean(e_p50, axis=0)
    p95_avg = np.nanmean(e_p95, axis=0)

    # Δσ for cumulative budget: σ_i − σ_{i+1}. CSV stores σ_0..σ_{n-1};
    # the last Euler step lands on σ_n = 0 (clean latent).
    sigmas_full = np.append(sigmas, 0.0)
    dsigma = sigmas_full[:-1] - sigmas_full[1:]
    w = args.cfg_focus

    # Per-step L1 contribution per voxel = |Δσ| · w · |Δe|.
    #   SMC switching term: |Δe| = k_t (uniform per voxel, Δe = −k·sign(s)).
    #   natural CFG correction above v_uncond: |Δe| = |e_t|.mean per voxel.
    cum_signal = np.cumsum(dsigma * w * e_mean_avg)
    cum_alpha  = np.cumsum(dsigma * w * (args.alpha * e_mean_avg))
    cum_fixed  = {k: np.cumsum(dsigma * w * k) for k in args.ks}

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10.5, 9.8), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.22},
    )

    # ---- top panel: instantaneous magnitudes --------------------------
    ax_top.fill_between(
        steps, p50_avg, p95_avg,
        color="#444444", alpha=0.18, zorder=1,
        label="intra-prompt voxel spread  ( p50 → p95 )",
    )

    prompt_palette = ["#1f78b4", "#33a02c", "#ff7f00", "#6a3d9a"]
    for pi in range(e_mean.shape[0]):
        ax_top.plot(
            steps, e_mean[pi],
            color=prompt_palette[pi % len(prompt_palette)],
            lw=1.0, alpha=0.55, zorder=2,
        )
    ax_top.plot(
        [], [], color="#444444", lw=1.0, alpha=0.7,
        label=rf"individual prompts (n={e_mean.shape[0]}, $|e_t|_{{\mathrm{{mean}}}}$)",
    )

    ax_top.plot(
        steps, e_mean_avg,
        color="black", lw=2.2, zorder=4,
        label=r"$|e_t|_{\mathrm{mean}}$  (avg across prompts)",
    )

    for k in args.ks:
        ax_top.axhline(
            k, ls="--", lw=1.4,
            color={0.02: "#2c7fb8", 0.1: "#d7191c"}.get(k, "black"),
            label=f"fixed k={k}" + ("  (paper)" if k == 0.1 else ""),
            zorder=3,
        )

    ax_top.set_ylabel(r"per-step magnitude  ($|e_t|$, controller gain $k$)")
    ax_top.set_yscale("log")
    ax_top.set_ylim(1e-3, 1e0)
    ax_top.set_title("instantaneous — controller gain vs signal envelope")
    ax_top.grid(True, which="both", ls=":", alpha=0.35)

    ax_top.secondary_xaxis(
        "top",
        functions=(
            lambda s: np.interp(s, steps, sigmas),
            lambda sig: np.interp(sig, sigmas[::-1], steps[::-1]),
        ),
    ).set_xlabel(r"$\sigma$", labelpad=4)

    ax_top.legend(loc="lower left", fontsize=8.5, framealpha=0.95, ncol=2)

    # ---- bottom panel: cumulative drift budget ------------------------
    final_signal = cum_signal[-1]

    ax_bot.plot(
        steps, cum_signal,
        color="black", lw=2.2, zorder=5,
        label=r"natural signal  $\sum |\Delta\sigma|\,w\,|e_t|_{\mathrm{mean}}$  (1.00×)",
    )
    ax_bot.plot(
        steps, cum_alpha,
        color="#1a9641", lw=2.8, zorder=4,
        label=(
            rf"α-adaptive  ($k_t={args.alpha}\cdot|e_t|$)"
            rf"     {cum_alpha[-1]/final_signal:.2f}× signal"
        ),
    )
    for k in args.ks:
        tag = "  (paper)" if k == 0.1 else ""
        ax_bot.plot(
            steps, cum_fixed[k],
            ls="--", lw=1.8,
            color={0.02: "#2c7fb8", 0.1: "#d7191c"}.get(k, "black"),
            label=(
                f"fixed k={k}{tag}"
                f"     {cum_fixed[k][-1]/final_signal:.2f}× signal"
            ),
            zorder=3,
        )

    ax_bot.set_xlabel("denoising step")
    ax_bot.set_ylabel(r"cumulative L1 drift  $\sum_{t' \leq t} |\Delta\sigma_{t'}|\,w\,|\Delta e_{t'}|$")
    ax_bot.set_title("cumulative — what the controller actually injects into $x$")
    ax_bot.grid(True, which="major", ls=":", alpha=0.35)
    ax_bot.set_xlim(-0.6, len(steps) - 0.4)

    ax_bot.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle(
        f"SMC-CFG controller budget at CFG={args.cfg_focus:g}    "
        r"per-step magnitudes (top) vs cumulative drift (bottom)",
        y=0.995, fontsize=12,
    )

    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
