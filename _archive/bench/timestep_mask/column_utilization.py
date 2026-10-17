"""T-LoRA per-column utilization probe.

Question this answers: under the *live* training config (σ sampling +
schedule), how often does each rank column actually receive gradient, and is
the top slice of the scheduled band earning its slot — or are we effectively
training at a lower rank than ``network_dim``?

First-order signal is analytical and needs no GPU/data. A rank column ``j``
receives gradient on a step **iff it is unmasked**, i.e. iff ``j < r(σ)`` for
that step's σ (mask build = ``arange < r``, networks/lora_anima/network.py:1050).
So the per-column *activation frequency* ``f(j) = P_σ[j < r(σ)]`` is a pure
function of the σ-sampling distribution and the schedule — exact, model-free.

Schedule (faithful to network.py:1044-1049, train_batch_size=1 → per-step σ):

    frac = clamp((1 - σ), 0, 1)
    r(σ) = clamp(frac**alpha_rank_scale * (R_max - min_rank) + min_rank, max=R_max)

σ sampling (faithful to library/runtime/noise.py:104-122). Defaults read from
the merged lora/default config; override on the CLI.

This is the screening bench. ``--empirical`` (separate run) confirms that
activation frequency translates to trained magnitude on the real model.

Usage::

    python bench/timestep_mask/column_utilization.py
    python bench/timestep_mask/column_utilization.py --alpha 0.5 1.0 1.5 2.0 --min_rank 1 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402


def sample_sigma(
    n: int,
    *,
    timestep_sampling: str,
    sigmoid_scale: float,
    sigmoid_bias: float,
    discrete_flow_shift: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw σ∈[0,1], mirroring noise.get_noisy_model_input_and_timesteps."""
    if timestep_sampling == "uniform":
        return rng.random(n)
    if timestep_sampling == "sigmoid":
        z = sigmoid_scale * rng.standard_normal(n) + sigmoid_bias
        return 1.0 / (1.0 + np.exp(-z))
    if timestep_sampling == "shift":
        z = sigmoid_scale * rng.standard_normal(n) + sigmoid_bias
        s = 1.0 / (1.0 + np.exp(-z))
        sh = discrete_flow_shift
        return (s * sh) / (1.0 + (sh - 1.0) * s)
    raise ValueError(f"unsupported timestep_sampling for probe: {timestep_sampling}")


def schedule_rank(sigma: np.ndarray, *, r_max: int, min_rank: int, alpha: float) -> np.ndarray:
    """r(σ) — faithful to network.py:1044-1049."""
    frac = np.clip(1.0 - sigma, 0.0, 1.0)
    r = np.power(frac, alpha) * (r_max - min_rank) + min_rank
    return np.clip(r, a_min=None, a_max=float(r_max))


def column_frequency(r: np.ndarray, r_max: int) -> np.ndarray:
    """f(j) = P[j < r(σ)] for j in 0..r_max-1 (mask = arange < r)."""
    cols = np.arange(r_max)[:, None]  # (R, 1)
    active = cols < r[None, :]  # (R, N)
    return active.mean(axis=1)


def analyze_one(
    sigma: np.ndarray, *, r_max: int, min_rank: int, alpha: float, train_threshold: float
) -> dict:
    r = schedule_rank(sigma, r_max=r_max, min_rank=min_rank, alpha=alpha)
    f = column_frequency(r, r_max)
    scheduled = f[min_rank:]  # the timestep-gated band (floor cols are always 1.0)
    starved = int(np.sum(f < train_threshold))
    return {
        "alpha": alpha,
        "min_rank": min_rank,
        "r_max": r_max,
        "eff_rank_mean": float(r.mean()),
        "eff_rank_p10": float(np.percentile(r, 10)),
        "eff_rank_p50": float(np.percentile(r, 50)),
        "eff_rank_p90": float(np.percentile(r, 90)),
        # columns active < train_threshold of the time = effectively untrained
        "starved_cols": starved,
        "starved_frac_of_total": starved / r_max,
        # how trained is the very top column, and the top scheduled quartile
        "freq_top_col": float(f[-1]),
        "freq_top_quartile_mean": float(scheduled[max(0, len(scheduled) - len(scheduled) // 4):].mean())
        if len(scheduled) else float("nan"),
        # effective "usable" rank = #cols trained >= threshold
        "usable_rank_at_threshold": int(np.sum(f >= train_threshold)),
        "_freq_curve": f.tolist(),
    }


def main() -> None:
    from library.config.io import load_method_preset

    cfg = load_method_preset("lora", "default", "configs", "methods")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=400_000, help="σ Monte-Carlo draws")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--alpha", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0],
        help="alpha_rank_scale values to sweep",
    )
    p.add_argument(
        "--min_rank", type=int, nargs="+", default=[int(cfg.get("min_rank", 16))],
        help="min_rank values to sweep",
    )
    p.add_argument("--r_max", type=int, default=int(cfg.get("network_dim", 48)))
    p.add_argument(
        "--train_threshold", type=float, default=0.02,
        help="activation-frequency floor below which a column is 'starved'",
    )
    p.add_argument("--timestep_sampling", default=str(cfg.get("timestep_sampling", "sigmoid")))
    p.add_argument("--sigmoid_scale", type=float, default=float(cfg.get("sigmoid_scale") or 1.0))
    p.add_argument("--sigmoid_bias", type=float, default=float(cfg.get("sigmoid_bias") or 0.0))
    p.add_argument(
        "--discrete_flow_shift", type=float, default=float(cfg.get("discrete_flow_shift") or 1.0)
    )
    p.add_argument("--label", default=None)
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    sigma = sample_sigma(
        args.n,
        timestep_sampling=args.timestep_sampling,
        sigmoid_scale=args.sigmoid_scale,
        sigmoid_bias=args.sigmoid_bias,
        discrete_flow_shift=args.discrete_flow_shift,
        rng=rng,
    )

    # Always include a uniform-σ reference for α=1.0 — the theoretical "matched"
    # case where each scheduled column gets equal training mass.
    sigma_uniform = rng.random(args.n)

    rows = []
    for mr in args.min_rank:
        for a in args.alpha:
            rows.append(analyze_one(
                sigma, r_max=args.r_max, min_rank=mr, alpha=a,
                train_threshold=args.train_threshold,
            ))
    ref = analyze_one(
        sigma_uniform, r_max=args.r_max, min_rank=int(args.min_rank[0]), alpha=1.0,
        train_threshold=args.train_threshold,
    )
    ref["_ref"] = "uniform_sigma_alpha1"

    run_dir = make_run_dir("timestep_mask", label=args.label or "column-util")

    # --- console summary ---
    print(f"\nσ sampling: {args.timestep_sampling} "
          f"(scale={args.sigmoid_scale}, bias={args.sigmoid_bias})  "
          f"R_max={args.r_max}  N={args.n:,}")
    print(f"σ mean={sigma.mean():.3f}  σ p10/50/90="
          f"{np.percentile(sigma, 10):.3f}/{np.percentile(sigma, 50):.3f}/{np.percentile(sigma, 90):.3f}")
    hdr = (f"\n{'min_rank':>8} {'alpha':>6} {'E[r]':>7} {'r_p10':>6} {'r_p90':>6} "
           f"{'usable':>7} {'starved':>8} {'f(top)':>8} {'f(topQ)':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['min_rank']:>8} {r['alpha']:>6.2f} {r['eff_rank_mean']:>7.1f} "
              f"{r['eff_rank_p10']:>6.1f} {r['eff_rank_p90']:>6.1f} "
              f"{r['usable_rank_at_threshold']:>7} {r['starved_cols']:>8} "
              f"{r['freq_top_col']:>8.4f} {r['freq_top_quartile_mean']:>8.4f}")
    print(f"{'[unif]':>8} {ref['alpha']:>6.2f} {ref['eff_rank_mean']:>7.1f} "
          f"{ref['eff_rank_p10']:>6.1f} {ref['eff_rank_p90']:>6.1f} "
          f"{ref['usable_rank_at_threshold']:>7} {ref['starved_cols']:>8} "
          f"{ref['freq_top_col']:>8.4f} {ref['freq_top_quartile_mean']:>8.4f}")
    print(f"\n(usable = #cols active >= {args.train_threshold:.0%} of steps; "
          f"starved = #cols below that)")

    # --- artifacts ---
    artifacts = []
    csv_path = run_dir / "per_column_freq.csv"
    with csv_path.open("w") as fh:
        keys = sorted({f"mr{r['min_rank']}_a{r['alpha']}" for r in rows})
        fh.write("column," + ",".join(keys) + ",uniform_a1\n")
        curves = {f"mr{r['min_rank']}_a{r['alpha']}": r["_freq_curve"] for r in rows}
        for j in range(args.r_max):
            vals = [f"{curves[k][j]:.6f}" for k in keys]
            fh.write(f"{j}," + ",".join(vals) + f",{ref['_freq_curve'][j]:.6f}\n")
    artifacts.append("per_column_freq.csv")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            for r in rows:
                ax1.plot(range(args.r_max), r["_freq_curve"],
                         label=f"min_rank={r['min_rank']}, α={r['alpha']}", marker=".", ms=4)
            ax1.plot(range(args.r_max), ref["_freq_curve"], "k--", alpha=0.6,
                     label="uniform σ, α=1 (matched)")
            ax1.axhline(args.train_threshold, color="r", ls=":", alpha=0.5,
                        label=f"starved < {args.train_threshold:.0%}")
            ax1.set_xlabel("rank column index j")
            ax1.set_ylabel("activation frequency  f(j) = P[j < r(σ)]")
            ax1.set_title(f"Per-column training exposure  ({args.timestep_sampling} σ)")
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)

            ax2.hist(sigma, bins=60, density=True, alpha=0.6, label="σ samples")
            ax2.set_xlabel("σ (noise level: 0=clean, 1=noise)")
            ax2.set_ylabel("density")
            ax2.set_title("σ sampling distribution")
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)
            fig.tight_layout()
            png = run_dir / "column_utilization.png"
            fig.savefig(png, dpi=110)
            plt.close(fig)
            artifacts.append("column_utilization.png")
        except Exception as e:  # noqa: BLE001
            print(f"[plot skipped: {e}]")

    metrics = {
        "sigma_mean": float(sigma.mean()),
        "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "uniform_ref": {k: v for k, v in ref.items() if not k.startswith("_")},
    }
    write_result(
        run_dir, script=__file__, args=args, metrics=metrics,
        label=args.label or "column-util", artifacts=artifacts,
    )
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
