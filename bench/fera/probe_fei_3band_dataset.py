#!/usr/bin/env python
"""3-band FEI on training latents — re-check the mid-band-collapse claim.

The decision to drop to 2 bands (``project_fera_probe_2band_decision``)
was made on *inference trajectories* from Anima base on 4 generic prompts
(``probe_fei.py``). The finding was ``e_mid ≤ 8%`` at (low_div=8,
mid_div=32) and ``≤ 1.5%`` at (8, 16). Since the dataset sweep
(``probe_fei_dataset.py``) showed that the training distribution can
shift FEI statistics noticeably vs. base-model trajectories, we re-test
the mid-band claim on real cached ``z_0`` to make sure the 2-band call
generalizes.

For each (low_div, mid_div) pair, compute 3-band FEI on
``z_t = (1-t)·z_0 + t·ε`` and report:

  - mean(e_mid), std(e_mid)             — magnitude + spread of mid band
  - frac(e_mid < 0.05)                  — "hard dead" sample fraction
  - mean(e_low), mean(e_high)           — for context (should still sum to 1)

If ``mean(e_mid) < 0.08`` AND ``std(e_mid) < 0.02`` across all interesting
t, the 2-band call holds. If mean grows above ~0.1 OR std rises (mid
carries content variation), 3-band might be worth revisiting.

Usage::

    uv run python bench/fera/probe_fei_3band_dataset.py --n_samples 256 \\
        --pairs 8:16,8:32,4:8,4:16 --ts 0.05,0.2,0.4,0.6,0.95 \\
        --label mid-band-recheck
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from bench.fera.probe_fei import fei_3band  # noqa: E402
from bench.fera.probe_fei_dataset import (  # noqa: E402
    _bucket_of,
    _load_latent,
    _scan_cache,
    _stratified_sample,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fera-3band-dataset")


def _parse_pairs(s: str) -> list[tuple[float, float]]:
    """``"8:16,8:32"`` → ``[(8, 16), (8, 32)]``.

    Each entry is ``low_div:mid_div``. Enforces ``low_div < mid_div`` so
    ``σ_low > σ_mid`` (the wider blur belongs to the low band).
    """
    out: list[tuple[float, float]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split(":")
        lo, mi = float(a), float(b)
        if lo >= mi:
            raise SystemExit(
                f"pair {tok}: low_div ({lo}) must be < mid_div ({mi}) so σ_low > σ_mid"
            )
        out.append((lo, mi))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--cache_dir",
        type=Path,
        default=ROOT / "post_image_dataset" / "lora",
    )
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument(
        "--pairs",
        type=str,
        default="8:16,8:32,4:8,4:16",
        help="Comma-sep ``low_div:mid_div`` pairs. Defaults cover the "
        "original bench (8:16, 8:32) plus the same ratios anchored to the "
        "dataset-sweep winner (4:8, 4:16).",
    )
    p.add_argument(
        "--ts",
        type=str,
        default="0.05,0.2,0.4,0.6,0.95",
        help="Flow-matching t-points (training-input convention).",
    )
    p.add_argument(
        "--dead_threshold",
        type=float,
        default=0.05,
        help="e_mid < threshold counts as 'dead' for the hard-dead fraction.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--label", default=None)
    args = p.parse_args()

    pairs = _parse_pairs(args.pairs)
    ts = [float(x) for x in args.ts.split(",") if x.strip()]
    if not pairs or not ts:
        raise SystemExit("need at least one --pairs and one --ts value")
    if any(not (0.0 < t < 1.0) for t in ts):
        raise SystemExit("--ts entries must lie strictly in (0, 1)")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    files = _scan_cache(args.cache_dir)
    if not files:
        raise SystemExit(f"no cached latents under {args.cache_dir}")
    sampled = _stratified_sample(files, args.n_samples, seed=args.seed)
    log.info(
        f"sampled {len(sampled)}/{len(files)} latents across "
        f"{len(set(_bucket_of(f) for f in sampled))} buckets"
    )
    log.info(f"pairs={pairs}, ts={ts}, dead_threshold={args.dead_threshold}")

    out_dir = make_run_dir("fera", label=args.label)
    log.info(f"output → {out_dir}")

    rng = np.random.default_rng(args.seed)

    # Aggregate per (pair, t): full population of (e_low, e_mid, e_high).
    pop: dict[tuple[tuple[float, float], float], list[tuple[float, float, float]]] = (
        defaultdict(list)
    )
    rows: list[dict] = []

    for idx, path in enumerate(sampled):
        try:
            z0 = _load_latent(path).to(device)  # (1, C, H, W)
        except Exception as exc:
            log.warning(f"skip {path.name}: {exc}")
            continue
        eps = torch.from_numpy(
            rng.standard_normal(size=z0.shape, dtype=np.float32)
        ).to(device)

        for t in ts:
            z_t = (1.0 - t) * z0 + t * eps
            min_d = float(min(z_t.shape[-2], z_t.shape[-1]))
            for low_div, mid_div in pairs:
                sigma_low = min_d / low_div
                sigma_mid = min_d / mid_div
                fei = fei_3band(z_t, sigma_low, sigma_mid)  # (1, 3) = [low, mid, high]
                e_low = float(fei[0, 0].item())
                e_mid = float(fei[0, 1].item())
                e_high = float(fei[0, 2].item())
                pop[((low_div, mid_div), t)].append((e_low, e_mid, e_high))
                rows.append(
                    {
                        "stem": path.stem,
                        "t": t,
                        "low_div": low_div,
                        "mid_div": mid_div,
                        "sigma_low": sigma_low,
                        "sigma_mid": sigma_mid,
                        "e_low": e_low,
                        "e_mid": e_mid,
                        "e_high": e_high,
                    }
                )

        if (idx + 1) % 32 == 0:
            log.info(f"  processed {idx + 1}/{len(sampled)}")

    csv_path = out_dir / "fei_3band_per_sample.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"wrote {csv_path} ({len(rows)} rows)")

    # ---- summary ---------------------------------------------------------
    summary: dict[str, list[dict]] = {}
    for (low_div, mid_div) in pairs:
        key = f"{low_div:g}:{mid_div:g}"
        summary[key] = []
        for t in ts:
            samples = pop[((low_div, mid_div), t)]
            if not samples:
                continue
            e_lows = [s[0] for s in samples]
            e_mids = [s[1] for s in samples]
            e_highs = [s[2] for s in samples]
            dead_frac = sum(1 for v in e_mids if v < args.dead_threshold) / len(e_mids)
            summary[key].append(
                {
                    "t": t,
                    "n": len(samples),
                    "mean_e_low": float(mean(e_lows)),
                    "mean_e_mid": float(mean(e_mids)),
                    "mean_e_high": float(mean(e_highs)),
                    "std_e_mid": float(pstdev(e_mids)),
                    "max_e_mid": float(max(e_mids)),
                    "frac_e_mid_dead": dead_frac,
                }
            )

    # ---- console table ---------------------------------------------------
    print("\n== mean(e_mid) across population — is the mid band carrying energy? ==")
    print("pair       | " + " | ".join(f"t={t:<5g}" for t in ts))
    print("-" * 78)
    for key, rows_ in summary.items():
        by_t = {r["t"]: r["mean_e_mid"] for r in rows_}
        cells = [f"{by_t.get(t, 0):.3f}" for t in ts]
        print(f"{key:<10} | " + " | ".join(c.ljust(5) for c in cells))

    print(
        f"\n== frac(e_mid < {args.dead_threshold}) — fraction of samples where "
        "mid band is 'hard dead' =="
    )
    print("pair       | " + " | ".join(f"t={t:<5g}" for t in ts))
    print("-" * 78)
    for key, rows_ in summary.items():
        by_t = {r["t"]: r["frac_e_mid_dead"] for r in rows_}
        cells = [f"{by_t.get(t, 0):.2f}" for t in ts]
        print(f"{key:<10} | " + " | ".join(c.ljust(5) for c in cells))

    print("\n== std(e_mid) across population — does mid band vary with content? ==")
    print("pair       | " + " | ".join(f"t={t:<5g}" for t in ts))
    print("-" * 78)
    for key, rows_ in summary.items():
        by_t = {r["t"]: r["std_e_mid"] for r in rows_}
        cells = [f"{by_t.get(t, 0):.3f}" for t in ts]
        print(f"{key:<10} | " + " | ".join(c.ljust(5) for c in cells))

    # ---- plot ------------------------------------------------------------
    artifacts = [csv_path.name]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(pairs), figsize=(4.5 * len(pairs), 4.5), sharey=True, squeeze=False)
        for ax, (low_div, mid_div) in zip(axes[0], pairs):
            key = f"{low_div:g}:{mid_div:g}"
            rows_ = summary[key]
            xs = [r["t"] for r in rows_]
            ax.stackplot(
                xs,
                [[r["mean_e_low"] for r in rows_],
                 [r["mean_e_mid"] for r in rows_],
                 [r["mean_e_high"] for r in rows_]],
                labels=["low", "mid", "high"],
                colors=["#3b82f6", "#10b981", "#ef4444"],
                alpha=0.85,
            )
            ax.set_title(f"low_div={low_div:g}, mid_div={mid_div:g}\n"
                         f"σ_mid/σ_low = {low_div/mid_div:.2f}")
            ax.set_xlabel("t  (flow-matching forward)")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
        axes[0, 0].set_ylabel("mean FEI (simplex)")
        axes[0, -1].legend(loc="upper right", fontsize=9)
        fig.suptitle(
            f"3-band FEI on training latents ({len(sampled)} samples)  "
            f"— is the mid band dead?"
        )
        fig.tight_layout()
        png = out_dir / "fei_3band_stack.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        artifacts.append(png.name)
        log.info(f"wrote {png}")
    except Exception as exc:
        log.warning(f"plot failed (continuing): {exc}")

    metrics = {
        "n_samples": len(sampled),
        "n_total_cache": len(files),
        "pairs": [list(p) for p in pairs],
        "ts": ts,
        "dead_threshold": args.dead_threshold,
        "summary": summary,
    }
    write_result(
        out_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        label=args.label,
        device=device,
    )
    log.info("done")


if __name__ == "__main__":
    main()
