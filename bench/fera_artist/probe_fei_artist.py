#!/usr/bin/env python
"""FEI divisor sweep with **artist-balanced** sampling of training latents.

The archived ``bench/fera/probe_fei_dataset.py`` ranked
``fei_sigma_low_div`` by ``std(e_low)`` over a 256-sample
**bucket-stratified** draw from ``post_image_dataset/lora/``. That
preserves the natural bucket distribution but lets prolific artists swing
the std: in the current dataset the top-3 artists own ~22 % of all
images, so the population the divisor is tuned to is implicitly
weighted toward a handful of styles.

This probe re-runs the same DoG decomposition + ``std(e_low)`` maxim­isation
but draws **K stems per artist** from ``caption_index.json::groups.artist``
(K=1 by default). One image per artist => the std being optimised is
the *between-artist* spread, not the spread inflated by a few prolific
contributors. Untagged images (no ``@artist`` tag) are dropped by default.

When ``--control proportional`` is set we *also* draw a bucket-stratified
sample of matching N — the old archived sampler — so the two argmaxes
can be compared in one run. If artist-balancing changes the divisor
recommendation, the deltas show up here.

Pure CPU/GPU FEI math — no DiT, no text encoder. Runs in well under a
minute on 75 artists × the cache.

Usage::

    uv run python bench/fera_artist/probe_fei_artist.py \\
        --k_per_artist 1 --divisors 4,8,16,32,64,128 --label artist-balanced
"""

from __future__ import annotations

import argparse
import csv
import json
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
log = logging.getLogger("fera-artist-probe")


_FNAME_RE = re.compile(r"^(?P<stem>.+)_(?P<w>\d{3,5})x(?P<h>\d{3,5})_anima\.npz$")
_LATKEY_RE = re.compile(r"^latents_(\d+)x(\d+)$")
_UNTAGGED = "__untagged__"


def _scan_cache(cache_dir: Path) -> dict[str, list[Path]]:
    """Map ``stem`` -> all cache files for that stem.

    A stem can have multiple bucket files if it was re-cached at
    different aspect ratios, though in practice we see one per stem.
    """
    out: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(cache_dir.rglob("*_anima.npz")):
        m = _FNAME_RE.match(f.name)
        if m is None:
            continue
        out[m.group("stem")].append(f)
    return out


def _bucket_of(p: Path) -> tuple[int, int]:
    m = _FNAME_RE.match(p.name)
    assert m is not None
    return int(m.group("w")), int(m.group("h"))  # (W_lat, H_lat) — pixel-aligned


def _build_artist_groups(
    caption_index: Path,
    cache_by_stem: dict[str, list[Path]],
    include_untagged: bool,
) -> dict[str, list[str]]:
    """Return ``artist -> [stems]`` restricted to stems with cached latents.

    Reads ``image_meta`` directly (not ``groups.artist``) so we can decide
    what to do with stems that have no ``@artist`` tag.
    """
    data = json.loads(caption_index.read_text())
    image_meta = data.get("image_meta", {})
    groups: dict[str, list[str]] = defaultdict(list)
    n_missing_cache = 0
    n_untagged = 0
    for stem, meta in image_meta.items():
        if stem not in cache_by_stem:
            n_missing_cache += 1
            continue
        artists = [a for a in meta.get("artist", []) if a]
        if not artists:
            n_untagged += 1
            if include_untagged:
                groups[_UNTAGGED].append(stem)
            continue
        # If a stem is co-credited to multiple artists, put it under the
        # *first* one — anything else double-counts.
        groups[artists[0]].append(stem)
    log.info(
        f"caption_index: {len(image_meta)} stems, "
        f"{n_missing_cache} missing cache, "
        f"{n_untagged} untagged "
        f"(include_untagged={include_untagged})"
    )
    return dict(groups)


def _sample_per_artist(
    groups: dict[str, list[str]],
    cache_by_stem: dict[str, list[Path]],
    k: int,
    seed: int,
) -> list[Path]:
    """Pick up to ``k`` cache paths per artist, seed-deterministic.

    For a stem with multiple cache files we pick the first deterministically
    (stable file sort upstream).
    """
    rng = random.Random(seed)
    picked: list[Path] = []
    for artist in sorted(groups.keys()):  # deterministic artist order
        stems = list(groups[artist])
        rng.shuffle(stems)
        for stem in stems[:k]:
            paths = cache_by_stem.get(stem, [])
            if not paths:
                continue
            picked.append(paths[0])
    rng.shuffle(picked)
    return picked


def _sample_proportional(
    cache_by_stem: dict[str, list[Path]],
    n: int,
    seed: int,
) -> list[Path]:
    """Bucket-stratified control sample of size ``n`` — mirrors the archived
    probe so the two argmaxes are directly comparable.
    """
    rng = random.Random(seed)
    files: list[Path] = []
    for paths in cache_by_stem.values():
        if paths:
            files.append(paths[0])
    by_bucket: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for f in files:
        by_bucket[_bucket_of(f)].append(f)
    total = len(files)
    picked: list[Path] = []
    for group in by_bucket.values():
        rng.shuffle(group)
        take = max(1, round(n * len(group) / total))
        picked.extend(group[:take])
    rng.shuffle(picked)
    return picked[:n]


def _load_latent(npz_path: Path) -> torch.Tensor:
    with np.load(npz_path) as d:
        keys = [k for k in d.keys() if _LATKEY_RE.match(k)]
        if not keys:
            raise KeyError(f"{npz_path.name}: no ``latents_HxW`` key (keys={list(d.keys())})")
        arr = d[keys[0]]  # (C, H, W)
    return torch.from_numpy(arr).float().unsqueeze(0)  # (1, C, H, W)


def _stem_of(p: Path) -> str:
    m = _FNAME_RE.match(p.name)
    assert m is not None
    return m.group("stem")


def _stem_to_artist(groups: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for artist, stems in groups.items():
        for s in stems:
            out.setdefault(s, artist)
    return out


def _run_sweep(
    sampled: list[Path],
    divisors: list[float],
    ts: list[float],
    seed: int,
    device: torch.device,
    stem_to_artist: dict[str, str] | None,
    sample_tag: str,
) -> tuple[list[dict], dict[tuple[float, float], list[float]]]:
    """FEI sweep over ``(divisor, t, sample)``. Returns (rows, by_dt)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    by_dt: dict[tuple[float, float], list[float]] = defaultdict(list)
    for idx, path in enumerate(sampled):
        bucket = _bucket_of(path)
        stem = _stem_of(path)
        artist = stem_to_artist.get(stem, "") if stem_to_artist else ""
        try:
            z0 = _load_latent(path).to(device)
        except Exception as exc:
            log.warning(f"skip {path.name}: {exc}")
            continue
        eps = torch.from_numpy(
            rng.standard_normal(size=z0.shape, dtype=np.float32)
        ).to(device)
        for t in ts:
            z_t = (1.0 - t) * z0 + t * eps
            min_d = float(min(z_t.shape[-2], z_t.shape[-1]))
            for div in divisors:
                sigma_low = min_d / div
                fei = compute_fei_2band(z_t, sigma_low)
                e_low = float(fei[0, 0].item())
                e_high = float(fei[0, 1].item())
                rows.append(
                    {
                        "sample": sample_tag,
                        "stem": stem,
                        "artist": artist,
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
                by_dt[(div, t)].append(e_low)
        if (idx + 1) % 32 == 0:
            log.info(f"  [{sample_tag}] processed {idx + 1}/{len(sampled)}")
    return rows, by_dt


def _pop_stats(
    by_dt: dict[tuple[float, float], list[float]],
    divisors: list[float],
    ts: list[float],
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for div in divisors:
        for t in ts:
            vals = by_dt.get((div, t), [])
            if not vals:
                continue
            out.setdefault(f"div_{div:g}", []).append(
                {
                    "t": t,
                    "n": len(vals),
                    "mean_e_low": float(mean(vals)),
                    "std_e_low": float(pstdev(vals)),
                    "min_e_low": float(min(vals)),
                    "max_e_low": float(max(vals)),
                }
            )
    return out


def _argmax_divisor(
    by_dt: dict[tuple[float, float], list[float]],
    divisors: list[float],
    ts: list[float],
    t_window: tuple[float, float],
) -> tuple[float, float]:
    """Return ``(divisor, mean_std_in_window)`` ranked by std(e_low) averaged
    over t ∈ [t_window[0], t_window[1]]. This is the call the calibrator makes.
    """
    lo, hi = t_window
    best_div, best_score = divisors[0], -1.0
    for div in divisors:
        scores = [
            pstdev(by_dt[(div, t)])
            for t in ts
            if lo <= t <= hi and by_dt.get((div, t))
        ]
        score = mean(scores) if scores else 0.0
        if score > best_score:
            best_div, best_score = div, score
    return best_div, best_score


def _print_table(
    title: str,
    by_dt: dict[tuple[float, float], list[float]],
    divisors: list[float],
    ts: list[float],
    reducer,
) -> None:
    print(f"\n== {title} ==")
    header = "div".ljust(8) + " | " + " | ".join(f"t={t:<5g}" for t in ts)
    print(header)
    print("-" * len(header))
    for div in divisors:
        cells = []
        for t in ts:
            vals = by_dt.get((div, t), [])
            cells.append(f"{reducer(vals):.3f}" if vals else "  -  ")
        print(f"{div:<8g} | " + " | ".join(c.ljust(5) for c in cells))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--cache_dir",
        type=Path,
        default=ROOT / "post_image_dataset" / "lora",
    )
    p.add_argument(
        "--caption_index",
        type=Path,
        default=ROOT / "post_image_dataset" / "captions" / "caption_index.json",
    )
    p.add_argument(
        "--k_per_artist",
        type=int,
        default=1,
        help="Latents drawn per artist. K=1 strict-balanced (default); K>1 "
        "trades some prolificacy back for sample size on small N.",
    )
    p.add_argument(
        "--include_untagged",
        action="store_true",
        help="Pool stems without an @artist tag into a synthetic "
        "__untagged__ group (still K samples). Default: drop.",
    )
    p.add_argument(
        "--control",
        choices=["none", "proportional"],
        default="proportional",
        help="Also draw a bucket-stratified sample of matching N (mirrors "
        "the archived probe). Default: proportional, so the two argmaxes "
        "are visible side-by-side.",
    )
    p.add_argument(
        "--divisors",
        type=str,
        default="4,8,16,32,64,128",
    )
    p.add_argument(
        "--ts",
        type=str,
        default="0.05,0.1,0.2,0.4,0.6,0.8,0.95",
    )
    p.add_argument(
        "--t_window",
        type=str,
        default="0.05,0.4",
        help="t range over which std(e_low) is averaged to rank divisors. "
        "Default matches where training spends most density on flow-matching.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--label", default=None)
    args = p.parse_args()

    divisors = [float(x) for x in args.divisors.split(",") if x.strip()]
    ts = [float(x) for x in args.ts.split(",") if x.strip()]
    if not divisors or not ts:
        raise SystemExit("need at least one --divisors and one --ts value")
    if any(not (0.0 < t < 1.0) for t in ts):
        raise SystemExit("--ts entries must lie strictly in (0, 1)")
    tw = [float(x) for x in args.t_window.split(",") if x.strip()]
    if len(tw) != 2 or not (0.0 <= tw[0] < tw[1] <= 1.0):
        raise SystemExit("--t_window must be 'lo,hi' with 0 <= lo < hi <= 1")
    t_window = (tw[0], tw[1])

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    cache_by_stem = _scan_cache(args.cache_dir)
    if not cache_by_stem:
        raise SystemExit(f"no cached latents under {args.cache_dir}")
    if not args.caption_index.exists():
        raise SystemExit(
            f"caption_index not found at {args.caption_index} — run `make caption-index` first"
        )

    groups = _build_artist_groups(
        args.caption_index, cache_by_stem, include_untagged=args.include_untagged
    )
    artist_sizes = sorted([len(v) for v in groups.values()], reverse=True)
    log.info(
        f"artists: n={len(groups)} (incl. untagged={_UNTAGGED in groups}), "
        f"top={artist_sizes[:5]}, median={artist_sizes[len(artist_sizes)//2] if artist_sizes else 0}, "
        f"tail={artist_sizes[-3:] if len(artist_sizes) >= 3 else artist_sizes}"
    )

    sampled_artist = _sample_per_artist(
        groups, cache_by_stem, k=args.k_per_artist, seed=args.seed
    )
    log.info(
        f"artist-balanced sample: {len(sampled_artist)} latents "
        f"(k={args.k_per_artist} per artist) across "
        f"{len(set(_bucket_of(f) for f in sampled_artist))} buckets"
    )

    sampled_control: list[Path] = []
    if args.control == "proportional":
        sampled_control = _sample_proportional(
            cache_by_stem, n=len(sampled_artist), seed=args.seed
        )
        log.info(
            f"control (proportional): {len(sampled_control)} latents across "
            f"{len(set(_bucket_of(f) for f in sampled_control))} buckets"
        )

    out_dir = make_run_dir("fera_artist", label=args.label)
    log.info(f"output → {out_dir}")

    stem_to_artist = _stem_to_artist(groups)

    rows_a, by_dt_a = _run_sweep(
        sampled_artist, divisors, ts, args.seed, device, stem_to_artist, "artist"
    )
    rows_c, by_dt_c = (
        _run_sweep(
            sampled_control, divisors, ts, args.seed, device, stem_to_artist, "control"
        )
        if sampled_control
        else ([], defaultdict(list))
    )
    rows = rows_a + rows_c

    csv_path = out_dir / "fei_per_sample.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"wrote {csv_path} ({len(rows)} rows)")

    pop_a = _pop_stats(by_dt_a, divisors, ts)
    pop_c = _pop_stats(by_dt_c, divisors, ts) if by_dt_c else {}

    best_a = _argmax_divisor(by_dt_a, divisors, ts, t_window)
    best_c = (
        _argmax_divisor(by_dt_c, divisors, ts, t_window) if by_dt_c else (None, None)
    )

    # Per-artist std(e_low) at div=4 / mid-t — surfaces which artists drive
    # the bulk of the std and whether the win is broad or concentrated.
    per_artist_rows = [r for r in rows_a if r["sample"] == "artist"]
    per_artist_e: dict[tuple[str, float, float], float] = {}
    for r in per_artist_rows:
        per_artist_e[(r["artist"], r["divisor"], r["t"])] = r["e_low"]

    artifacts: list[str] = [csv_path.name]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_panels = 2 if by_dt_c else 1
        fig, axes = plt.subplots(1, 2 * n_panels, figsize=(6 * n_panels, 5), squeeze=False)
        cmap = plt.get_cmap("viridis")
        plots: list[tuple[str, dict]] = [("artist-balanced", by_dt_a)]
        if by_dt_c:
            plots.append(("control (proportional)", by_dt_c))
        for col, (tag, bd) in enumerate(plots):
            ax_mean = axes[0, 2 * col]
            ax_std = axes[0, 2 * col + 1]
            for ci, div in enumerate(divisors):
                xs = ts
                mu = [mean(bd[(div, t)]) if bd.get((div, t)) else None for t in ts]
                sd = [pstdev(bd[(div, t)]) if bd.get((div, t)) else None for t in ts]
                color = cmap(ci / max(1, len(divisors) - 1))
                ax_mean.plot(xs, mu, marker="o", color=color, label=f"div={div:g}")
                ax_std.plot(xs, sd, marker="o", color=color, label=f"div={div:g}")
            ax_mean.set_xlabel("t")
            ax_mean.set_ylabel("mean e_low")
            ax_mean.set_title(f"{tag} — mean")
            ax_mean.set_ylim(0, 1)
            ax_mean.grid(alpha=0.3)
            ax_mean.legend(fontsize=7)
            ax_std.set_xlabel("t")
            ax_std.set_ylabel("std(e_low)")
            ax_std.set_title(f"{tag} — std (higher = better)")
            ax_std.grid(alpha=0.3)
            ax_std.legend(fontsize=7)
        fig.suptitle(
            f"FEI divisor sweep — artist={len(sampled_artist)} | "
            f"control={len(sampled_control) if sampled_control else 0} | "
            f"K/artist={args.k_per_artist}"
        )
        fig.tight_layout()
        png = out_dir / "fei_sigma_sweep.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        artifacts.append(png.name)
        log.info(f"wrote {png}")
    except Exception as exc:
        log.warning(f"plot failed (continuing): {exc}")

    _print_table(
        "std(e_low) — artist-balanced",
        by_dt_a, divisors, ts, pstdev,
    )
    _print_table(
        "mean(e_low) — artist-balanced (should ↓ in t)",
        by_dt_a, divisors, ts, mean,
    )
    if by_dt_c:
        _print_table(
            "std(e_low) — control (proportional)",
            by_dt_c, divisors, ts, pstdev,
        )

    print(
        f"\n== argmax divisor over t∈[{t_window[0]}, {t_window[1]}] "
        "(by mean std(e_low)) =="
    )
    print(f"  artist-balanced : div={best_a[0]:g}   score={best_a[1]:.4f}")
    if best_c[0] is not None:
        print(f"  control         : div={best_c[0]:g}   score={best_c[1]:.4f}")
        if best_a[0] != best_c[0]:
            print(
                "  >> argmax DIFFERS — artist prolificacy was biasing the divisor "
                "ranking on the proportional sample."
            )
        else:
            print(
                "  >> argmax MATCHES — div=4 ranking survives the artist-balancing "
                "audit (or both samples are noise-limited)."
            )

    bucket_counts_a: dict[str, int] = defaultdict(int)
    for f in sampled_artist:
        bw, bh = _bucket_of(f)
        bucket_counts_a[f"{bw}x{bh}"] += 1
    bucket_counts_c: dict[str, int] = defaultdict(int)
    for f in sampled_control:
        bw, bh = _bucket_of(f)
        bucket_counts_c[f"{bw}x{bh}"] += 1

    metrics = {
        "n_artists": len(groups),
        "k_per_artist": args.k_per_artist,
        "include_untagged": args.include_untagged,
        "n_artist_sample": len(sampled_artist),
        "n_control_sample": len(sampled_control),
        "n_total_cache": sum(1 for v in cache_by_stem.values() if v),
        "bucket_counts_artist": dict(bucket_counts_a),
        "bucket_counts_control": dict(bucket_counts_c),
        "divisors": divisors,
        "ts": ts,
        "t_window": list(t_window),
        "population_stats_artist": pop_a,
        "population_stats_control": pop_c,
        "best_divisor_artist": {"div": best_a[0], "mean_std_in_window": best_a[1]},
        "best_divisor_control": (
            {"div": best_c[0], "mean_std_in_window": best_c[1]}
            if best_c[0] is not None
            else None
        ),
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
