#!/usr/bin/env python
"""Probe FEI on the real training dataset (not base-model trajectories).

The existing ``probe_fei.py`` samples ``z_T ~ N(0,I)`` and runs Anima base
through a full denoising trajectory. That measures the **inference-time**
router input. Training-time is different: under flow-matching the router
sees ``z_t = (1-t)·z_0_real + t·ε`` for ``z_0`` drawn from the cached
training latents under ``post_image_dataset/lora/``. Anime data has very
different power spectra (flat regions + sharp edges) from generic
photographic content the base model was eval'd against — so the
σ_low choice that maximises router signal in training can differ from
the inference-trajectory probe.

We sweep ``fei_sigma_low_div`` × ``t`` × cached ``z_0`` and report:

  - mean / std of ``e_low`` across the population at each (divisor, t)
  - cross-bucket spread of mean ``e_low`` at fixed t (aspect-invariance)
  - per-bucket sample counts (so a divisor that wins only on the
    dominant bucket is visible)

The metric that matters most for routing is **std(e_low) at mid-t**:
that's the router's discriminative signal. A larger std means different
content routes differently at the same noise level. Mean(e_low) should
also migrate monotonically with t (sanity).

No DiT, no text encoder — FEI is a pure function of ``z_t`` so this runs
end-to-end on CPU in well under a minute on 2k+ latents.

Usage::

    uv run python bench/fera/probe_fei_dataset.py \\
        --n_samples 256 --divisors 4,8,16,32,64,128 --label dataset-sweep
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.runtime.fei import compute_fei_2band  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fera-dataset-probe")


# Cached latent files look like ``<stem>_<W>x<H>_anima.npz`` with a single
# key ``latents_<H>x<W>`` of shape ``(C=16, H, W)``. The bucket suffix in the
# filename uses W×H (pixel-aligned latent extents).
_FNAME_RE = re.compile(r"_(\d{3,5})x(\d{3,5})_anima\.npz$")
_LATKEY_RE = re.compile(r"^latents_(\d+)x(\d+)$")


def _scan_cache(cache_dir: Path) -> list[Path]:
    files = sorted(cache_dir.glob("*_anima.npz"))
    return [f for f in files if _FNAME_RE.search(f.name)]


def _bucket_of(p: Path) -> tuple[int, int]:
    m = _FNAME_RE.search(p.name)
    assert m is not None  # filtered upstream
    return int(m.group(1)), int(m.group(2))  # (W_lat, H_lat) — pixel-aligned


def _stratified_sample(
    files: list[Path], n: int, seed: int
) -> list[Path]:
    """Sample ``n`` files proportionally to per-bucket counts.

    Pure-random sampling would let the dominant bucket (here 832×1248 ≈ 45%)
    swallow the population. Proportional stratification keeps every bucket
    represented while preserving the natural distribution.
    """
    rng = random.Random(seed)
    by_bucket: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for f in files:
        by_bucket[_bucket_of(f)].append(f)
    total = len(files)
    picked: list[Path] = []
    for bucket, group in by_bucket.items():
        rng.shuffle(group)
        # ceil so small buckets keep ≥1 sample even at small ``n``.
        take = max(1, round(n * len(group) / total))
        picked.extend(group[:take])
    rng.shuffle(picked)
    return picked[:n]


def _load_latent(npz_path: Path) -> torch.Tensor:
    """Return ``z_0`` as ``(1, C, H, W)`` float32 torch tensor."""
    with np.load(npz_path) as d:
        keys = [k for k in d.keys() if _LATKEY_RE.match(k)]
        if not keys:
            raise KeyError(f"{npz_path.name}: no ``latents_HxW`` key (keys={list(d.keys())})")
        arr = d[keys[0]]  # (C, H, W)
    t = torch.from_numpy(arr).float().unsqueeze(0)  # (1, C, H, W)
    return t


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--cache_dir",
        type=Path,
        default=ROOT / "post_image_dataset" / "lora",
        help="Directory holding ``*_anima.npz`` cached training latents.",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=256,
        help="Latents to draw (stratified across buckets).",
    )
    p.add_argument(
        "--divisors",
        type=str,
        default="4,8,16,32,64,128",
        help="Comma-sep list of ``fei_sigma_low_div`` values to compare.",
    )
    p.add_argument(
        "--ts",
        type=str,
        default="0.05,0.1,0.2,0.4,0.6,0.8,0.95",
        help="Flow-matching t-points: z_t = (1-t)·z_0 + t·ε.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Stratified-sample seed AND per-sample ε seed (one ε per sample, "
        "shared across (divisor, t) so variance across divisors reflects σ "
        "choice rather than noise reshuffle).",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--label", default=None)
    args = p.parse_args()

    divisors = [float(x) for x in args.divisors.split(",") if x.strip()]
    ts = [float(x) for x in args.ts.split(",") if x.strip()]
    if not divisors or not ts:
        raise SystemExit("need at least one --divisors and one --ts value")
    if any(not (0.0 < t < 1.0) for t in ts):
        raise SystemExit("--ts entries must lie strictly in (0, 1)")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    files = _scan_cache(args.cache_dir)
    if not files:
        raise SystemExit(f"no cached latents under {args.cache_dir}")
    sampled = _stratified_sample(files, args.n_samples, seed=args.seed)
    log.info(
        f"sampled {len(sampled)}/{len(files)} latents from {args.cache_dir} "
        f"across {len(set(_bucket_of(f) for f in sampled))} buckets"
    )
    log.info(f"divisors={divisors}, ts={ts}, device={device}")

    out_dir = make_run_dir("fera", label=args.label)
    log.info(f"output → {out_dir}")

    # ε is drawn once per sample (shared across all (divisor, t)) so the
    # only thing varying between divisors is the σ_low. That isolates the
    # divisor effect from noise-resampling variance.
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    # Per-(divisor, t, bucket) aggregate so the plot is one pass over rows.
    agg_e_low: dict[tuple[float, float, tuple[int, int]], list[float]] = defaultdict(list)

    for idx, path in enumerate(sampled):
        bucket = _bucket_of(path)
        try:
            z0 = _load_latent(path).to(device)  # (1, C, H, W)
        except Exception as exc:
            log.warning(f"skip {path.name}: {exc}")
            continue
        # One ε per sample, shape-matched.
        eps = torch.from_numpy(
            rng.standard_normal(size=z0.shape, dtype=np.float32)
        ).to(device)

        for t in ts:
            z_t = (1.0 - t) * z0 + t * eps  # flow-matching forward
            min_d = float(min(z_t.shape[-2], z_t.shape[-1]))
            for div in divisors:
                sigma_low = min_d / div
                fei = compute_fei_2band(z_t, sigma_low)  # (1, 2) -> [e_low, e_high]
                e_low = float(fei[0, 0].item())
                e_high = float(fei[0, 1].item())
                rows.append(
                    {
                        "stem": path.stem,
                        "bucket_w": bucket[0],
                        "bucket_h": bucket[1],
                        "h_lat": int(z_t.shape[-2]),
                        "w_lat": int(z_t.shape[-1]),
                        "t": t,
                        "divisor": div,
                        "sigma_low": sigma_low,
                        "e_low": e_low,
                        "e_high": e_high,
                    }
                )
                agg_e_low[(div, t, bucket)].append(e_low)

        if (idx + 1) % 32 == 0:
            log.info(f"  processed {idx + 1}/{len(sampled)}")

    # ---- write CSV --------------------------------------------------------
    csv_path = out_dir / "fei_per_sample.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"wrote {csv_path} ({len(rows)} rows)")

    # ---- per-(divisor, t) population stats -------------------------------
    by_dt: dict[tuple[float, float], list[float]] = defaultdict(list)
    for (div, t, _b), vals in agg_e_low.items():
        by_dt[(div, t)].extend(vals)

    pop_stats: dict[str, list[dict]] = {}
    for div in divisors:
        for t in ts:
            vals = by_dt.get((div, t), [])
            if not vals:
                continue
            pop_stats.setdefault(f"div_{div:g}", []).append(
                {
                    "t": t,
                    "n": len(vals),
                    "mean_e_low": float(mean(vals)),
                    "std_e_low": float(pstdev(vals)),
                    "min_e_low": float(min(vals)),
                    "max_e_low": float(max(vals)),
                }
            )

    # Cross-bucket spread: max(mean) − min(mean) of e_low across buckets at fixed (div, t).
    bucket_spread: dict[str, list[dict]] = {}
    buckets = sorted({b for (_d, _t, b) in agg_e_low.keys()})
    for div in divisors:
        for t in ts:
            means = []
            for b in buckets:
                vals = agg_e_low.get((div, t, b), [])
                if vals:
                    means.append(mean(vals))
            if len(means) >= 2:
                spread = max(means) - min(means)
                bucket_spread.setdefault(f"div_{div:g}", []).append(
                    {"t": t, "spread_e_low": float(spread), "n_buckets": len(means)}
                )

    # ---- plot -------------------------------------------------------------
    artifacts: list[str] = [csv_path.name]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax_mean, ax_std) = plt.subplots(1, 2, figsize=(12, 5))
        cmap = plt.get_cmap("viridis")
        for ci, div in enumerate(divisors):
            xs = ts
            mu = [mean(by_dt[(div, t)]) for t in ts]
            sd = [pstdev(by_dt[(div, t)]) for t in ts]
            color = cmap(ci / max(1, len(divisors) - 1))
            ax_mean.plot(xs, mu, marker="o", color=color, label=f"div={div:g}")
            ax_std.plot(xs, sd, marker="o", color=color, label=f"div={div:g}")
        ax_mean.set_xlabel("t  (flow-matching, 0 = clean, 1 = noise)")
        ax_mean.set_ylabel("mean e_low across population")
        ax_mean.set_title("Migration: monotonic ↓ in t is required")
        ax_mean.set_ylim(0, 1)
        ax_mean.grid(alpha=0.3)
        ax_mean.legend(fontsize=8)
        ax_std.set_xlabel("t")
        ax_std.set_ylabel("std(e_low) across population")
        ax_std.set_title("Router discriminative signal — higher is better")
        ax_std.grid(alpha=0.3)
        ax_std.legend(fontsize=8)
        fig.suptitle(
            f"FEI on training latents ({len(sampled)} samples × {len(buckets)} buckets, "
            f"z_t = (1−t)·z_0 + t·ε)"
        )
        fig.tight_layout()
        png = out_dir / "fei_sigma_sweep.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        artifacts.append(png.name)
        log.info(f"wrote {png}")
    except Exception as exc:
        log.warning(f"plot failed (continuing): {exc}")

    # ---- summary console output ------------------------------------------
    print("\n== std(e_low) across population — higher = more discriminative ==")
    header = "div".ljust(8) + " | " + " | ".join(f"t={t:<5g}" for t in ts)
    print(header)
    print("-" * len(header))
    for div in divisors:
        cells = []
        for t in ts:
            vals = by_dt[(div, t)]
            cells.append(f"{pstdev(vals):.3f}" if vals else "  -  ")
        print(f"{div:<8g} | " + " | ".join(c.ljust(5) for c in cells))

    print("\n== mean(e_low) across population — should be monotonic in t ==")
    print(header)
    print("-" * len(header))
    for div in divisors:
        cells = []
        for t in ts:
            vals = by_dt[(div, t)]
            cells.append(f"{mean(vals):.3f}" if vals else "  -  ")
        print(f"{div:<8g} | " + " | ".join(c.ljust(5) for c in cells))

    # ---- result envelope --------------------------------------------------
    bucket_counts = defaultdict(int)
    for f in sampled:
        bucket_counts[f"{_bucket_of(f)[0]}x{_bucket_of(f)[1]}"] += 1
    metrics = {
        "n_samples": len(sampled),
        "n_total_cache": len(files),
        "n_buckets": len(buckets),
        "bucket_counts": dict(bucket_counts),
        "divisors": divisors,
        "ts": ts,
        "population_stats": pop_stats,
        "bucket_spread": bucket_spread,
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
