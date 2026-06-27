#!/usr/bin/env python3
"""Plot + summarize the Tier-0 cross-attn-drive curve from a CFG-delta log.

The log is produced by a normal inference run with the probe enabled. For real
captions (recommended — synthetic prompts under-represent the effect), run a
batch from a prompt file, one caption per line:

    ANIMA_CFG_DELTA_LOG=output/tests/cfg_delta.json \
        make test ARGS="--from_file bench/cross_attn_drive/real_prompts.txt"

Then:

    python bench/cross_attn_drive/plot_cfg_delta.py output/tests/cfg_delta.json

`ratio = ||v_cond - v_uncond|| / ||v_cond||` is the text-drive proxy per sigma:
high => the cross-attention pathway is steering the update hard at that noise
level; low => the step is mostly base-prior / self-attn / latent driven.
`cos_cond_uncond` is the directional read (1 = text only rescales, <1 = rotates).

The log holds one entry per generation (per prompt). This aggregates across
generations: bold = mean over prompts, band = +/-1 std, faint lines = per prompt.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load_generations(log: Path) -> list[list[dict]]:
    """Return a list of per-generation step-row lists (handles both formats)."""
    blob = json.loads(log.read_text())
    if "generations" in blob:
        return [g["steps"] for g in blob["generations"] if g["steps"]]
    if "steps" in blob:  # legacy single-generation format
        return [blob["steps"]] if blob["steps"] else []
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="cfg_delta.json from the probe")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output PNG (default: alongside the log)",
    )
    args = ap.parse_args()

    gens = _load_generations(args.log)
    if not gens:
        raise SystemExit(f"no steps logged in {args.log} (was CFG enabled / cfg>1?)")

    # All generations share the sigma schedule (same infer_steps/flow_shift), so
    # aggregate by step index. Truncate to the shortest in case one was cut off.
    n_steps = min(len(g) for g in gens)
    sigma = [gens[0][i]["sigma"] for i in range(n_steps)]

    def col(key: str, i: int) -> list[float]:
        return [g[i][key] for g in gens]

    ratio_mean = [statistics.fmean(col("ratio", i)) for i in range(n_steps)]
    ratio_std = [
        statistics.pstdev(col("ratio", i)) if len(gens) > 1 else 0.0
        for i in range(n_steps)
    ]
    cos_mean = [statistics.fmean(col("cos_cond_uncond", i)) for i in range(n_steps)]

    # ---- console summary ----
    print(f"generations (prompts): {len(gens)}   steps: {n_steps}")
    print(f"{'step':>4} {'sigma':>8} {'ratio_mean':>11} {'ratio_std':>10} {'cos':>8}")
    for i in range(n_steps):
        print(
            f"{i:>4} {sigma[i]:>8.4f} {ratio_mean[i]:>11.4f} "
            f"{ratio_std[i]:>10.4f} {cos_mean[i]:>8.4f}"
        )
    peak = max(range(n_steps), key=lambda i: ratio_mean[i])
    print(
        f"\npeak text-drive: step {peak} "
        f"(sigma={sigma[peak]:.3f}, ratio={ratio_mean[peak]:.3f})"
    )
    print(
        f"min cos: {min(cos_mean):.4f} at sigma={sigma[cos_mean.index(min(cos_mean))]:.3f}"
        "  (1.0 => text only rescales base velocity; CFG already spans that)"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping plot, summary above)")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    for g in gens:  # faint per-prompt traces
        ax1.plot(
            [g[i]["sigma"] for i in range(n_steps)],
            [g[i]["ratio"] for i in range(n_steps)],
            "-",
            color="C0",
            alpha=0.12,
        )
    lo = [m - s for m, s in zip(ratio_mean, ratio_std)]
    hi = [m + s for m, s in zip(ratio_mean, ratio_std)]
    ax1.fill_between(sigma, lo, hi, color="C0", alpha=0.2)
    ax1.plot(sigma, ratio_mean, "o-", color="C0", label="text-drive ratio (mean)")
    ax1.set_xlabel("sigma (noise level, high -> low)")
    ax1.set_ylabel("||v_cond - v_uncond|| / ||v_cond||", color="C0")
    ax1.invert_xaxis()
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(sigma, cos_mean, "s--", color="C1", alpha=0.7, label="cos(cond, uncond)")
    ax2.set_ylabel("cosine(v_cond, v_uncond)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    ax1.set_title(
        f"Cross-attn drive vs sigma ({len(gens)} real prompts, Tier-0 CFG-delta)"
    )
    fig.tight_layout()

    out = args.out or args.log.with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
