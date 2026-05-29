"""Turbo dataset-curation prep (item 5; see ``item5_plan.md``).

Treats curation as a *prep stage* in front of ``scripts/distill_turbo`` — score
every cached training sample once, so a later cut narrows the turbo input pool
to the high-signal subset. **No model change, no new loss** — data selection
only.

The pipeline: score every stem (two cheap no-GPU scores), then stratify by
``(aspect_bucket, artist)``, keep the highest-scoring stems down to ``--target``
(default 500), repair coverage so no aspect bucket / character is dropped and
≥ 80% of artists survive, and emit ``keep_list.json`` — the single artifact that
gates the turbo train reader. Phase 0's diagnostic ``scores.csv`` + plot are
gone; every score is embedded in ``keep_list.json`` instead.

Two scores, both no-GPU and ~ms/stem:

* ``hf_ratio``    — latent HF/total band energy (``compute_fei_2band`` e_high on
  the cached VAE latent). Universal; the same FEI used by training/item 2.
* ``noise_sigma`` — Immerkær (1996) fast noise-std estimate on the *resized*
  RGB image (``post_image_dataset/resized/``), the causally-correct surface:
  exactly what the VAE encodes into the latent the student trains on. This is
  the "noise-alike" axis — it isolates baked-in grain/film-noise the 4-step
  student cannot reproduce, which a whole-artist tier label conflates with
  legitimate texture deficit. ``flat_floor`` (low-percentile local std) rides
  along as a high-precision detector of the heavy-grain tail.

Why ``noise_sigma`` and not the plan's original ``jpeg_q``: the source pool is
~87% webp, so a JPEG quantization-table quality estimate is NaN for almost
every stem. The noise detector is format-agnostic and, per the Phase-0 probe,
separates the grain-heavy Tier-A subset (@asou, @egami, @esearu, @hayate) from
the legitimate-texture Tier-A artists (@audirenze, @eufoniuz) at AUC ≈ 0.62 —
which the tier label alone cannot do. Must be measured at the pipeline's resize
target (not native, not an arbitrary smaller size): anti-aliased downscaling
low-passes grain away, so a too-aggressive shrink kills the signal.

The composite keep-score is ``-α·z(hf_ratio) - β·z(noise_sigma)`` (z-normed
globally, higher = keep). The sign on noise is negative because high noise =
drop (the inverse of the plan's quality-positive ``jpeg_q``). Phase 0 found
``hf_ratio`` orthogonal to noise and Tier-A spanning all hf values, so
``noise_sigma`` is the load-bearing axis; ``α`` (the hf weight) is the uncertain
knob and is left tunable for the Phase-2 A/B.

Usage::

    # produce the keep list (default target 500)
    python -m scripts.distill_turbo.prep

    # tune the cut size / axis weights
    python -m scripts.distill_turbo.prep --target 1000 --alpha 0.5

    # print the cut summary without writing keep_list.json
    python -m scripts.distill_turbo.prep --dry_run
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import math
import os
import re
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# {stem}_{Wpix}x{Hpix}_anima.npz  (pixel dims — the aspect_bucket label)
_RES_RE = re.compile(r"_(\d{3,5}x\d{3,5})_anima\.npz$")


# ──────────────────────────────────────────────────────────────────────────
# Per-stem scores (worker-side; kept import-light for ProcessPool fork).
# ──────────────────────────────────────────────────────────────────────────
def _immerkaer_sigma(g: np.ndarray) -> float:
    """Immerkær '96 fast noise-std estimate via one 3×3 Laplacian-of-Laplacian.

    ``g`` is grayscale in [0, 1]. The kernel's design cancels most structured
    2nd-derivative content, leaving a robust estimate of additive noise std.
    """
    from scipy.signal import convolve2d

    K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    lap = convolve2d(g, K, mode="valid")
    h, w = g.shape
    return float(np.sum(np.abs(lap)) * math.sqrt(math.pi / 2) / (6.0 * (w - 2) * (h - 2)))


def _flat_floor(g: np.ndarray, win: int = 7, pct: float = 5.0) -> float:
    """Noise floor = low percentile of local std — residual in the flattest
    patches, where a clean illustration is ≈ 0 but grain is not."""
    from scipy.ndimage import uniform_filter

    mu = uniform_filter(g, win)
    var = np.clip(uniform_filter(g * g, win) - mu * mu, 0.0, None)
    return float(np.percentile(np.sqrt(var), pct))


def _hf_ratio(npz_path: str, fei_sigma_low_div: float) -> float:
    """Latent HF/total band energy = ``compute_fei_2band`` e_high.

    Reads the single ``latents_{HxW}`` array, runs the 2-band FEI on CPU. torch
    threads are pinned to 1 so a ProcessPool of N workers doesn't oversubscribe.
    """
    import torch

    from library.runtime.fei import compute_fei_2band, fei_sigma_low

    torch.set_num_threads(1)
    with np.load(npz_path) as d:
        key = next(k for k in d.files if k.startswith("latents_"))
        lat = d[key]  # (C, H, W)
    z = torch.from_numpy(np.ascontiguousarray(lat)).float().unsqueeze(0)  # (1,C,H,W)
    h_lat, w_lat = z.shape[-2], z.shape[-1]
    e = compute_fei_2band(z, fei_sigma_low(h_lat, w_lat, fei_sigma_low_div))
    return float(e[0, 1])  # e_high


def _load_gray(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def score_one(job: tuple) -> dict:
    """Worker: compute all Phase-0 scores for one stem.

    ``job`` = (stem, artist, aspect_bucket, npz_path, png_path, fei_div).
    Returns a row dict; ``error`` is set (and scores NaN) on failure so the
    pool never aborts the whole pass on one bad file.
    """
    stem, artist, bucket, npz_path, png_path, fei_div = job
    row = {
        "stem": stem,
        "artist": artist,
        "aspect_bucket": bucket,
        "hf_ratio": float("nan"),
        "noise_sigma": float("nan"),
        "flat_floor": float("nan"),
        "error": "",
    }
    try:
        row["hf_ratio"] = _hf_ratio(npz_path, fei_div)
        if png_path:
            g = _load_gray(png_path)
            row["noise_sigma"] = _immerkaer_sigma(g)
            row["flat_floor"] = _flat_floor(g)
        else:
            row["error"] = "no_resized_png"
    except Exception as ex:  # noqa: BLE001 — isolate per-stem failures
        row["error"] = f"{type(ex).__name__}: {ex}"
    return row


# ──────────────────────────────────────────────────────────────────────────
# Driver-side helpers.
# ──────────────────────────────────────────────────────────────────────────
def _tier_a_artists(fei_probe: Path) -> set[str]:
    """Reproduce the Tier-A artist set from the 30-artist FEI-gap probe:
    bottom tertile of per-artist mean ``delta_low`` at stage 3, div 16."""
    if not fei_probe.exists():
        logger.warning("FEI probe %s missing — Tier-A overlay disabled.", fei_probe)
        return set()
    acc: dict[str, list[float]] = collections.defaultdict(list)
    with open(fei_probe) as f:
        for r in csv.DictReader(f):
            if r["stage"] == "3" and float(r["div"]) == 16.0:
                acc[r["artist"]].append(float(r["delta_low"]))
    means = {a: sum(v) / len(v) for a, v in acc.items()}
    if not means:
        return set()
    cut = statistics.quantiles(list(means.values()), n=3)[0]  # lower tertile
    return {a for a, v in means.items() if v <= cut}


def _zscores(vals: list[float]) -> list[float]:
    """Population z-scores; degenerate (zero-std) input → all-zero."""
    a = np.asarray(vals, dtype=np.float64)
    sd = a.std()
    if sd == 0:
        return [0.0] * len(vals)
    return ((a - a.mean()) / sd).tolist()


def _score_pool(args: argparse.Namespace) -> tuple[list[dict], dict[str, list[str]]]:
    """Walk the cache, score every stem (multiprocessed), attach metadata.

    Returns ``(rows, stem_chars)`` where each row carries
    ``stem / artist / aspect_bucket / hf_ratio / noise_sigma / flat_floor`` and
    ``stem_chars`` maps stem → its character tags (for coverage repair).
    """
    cache_dir = Path(args.cache_dir)
    resized_dir = Path(args.resized_dir)

    cidx = json.loads(Path(args.caption_index).read_text())
    stem_artist: dict[str, str] = {}
    stem_relpath: dict[str, str] = {}
    stem_chars: dict[str, list[str]] = {}
    for stem, e in cidx["image_meta"].items():
        arts = e.get("artist") or []
        stem_artist[stem] = arts[0] if arts else "<none>"
        stem_relpath[stem] = str(Path(e["path"]).with_suffix(".png"))
        stem_chars[stem] = list(e.get("character") or [])

    npz_paths = sorted(str(p) for p in cache_dir.rglob("*_anima.npz"))
    if not npz_paths:
        raise SystemExit(f"no `*_anima.npz` under {cache_dir}")

    jobs = []
    missing_png = 0
    for p in npz_paths:
        name = Path(p).name
        m = _RES_RE.search(name)
        bucket = m.group(1) if m else "?"
        stem = name[: m.start()] if m else name.removesuffix("_anima.npz")
        rel = stem_relpath.get(stem)
        png = str(resized_dir / rel) if rel else ""
        if png and not Path(png).exists():
            png = ""
            missing_png += 1
        jobs.append((stem, stem_artist.get(stem, "<none>"), bucket, p, png,
                     args.fei_sigma_low_div))

    if args.max_samples:
        jobs = jobs[: args.max_samples]
    logger.info(
        "scoring %d stems (%d missing resized png) with %d workers",
        len(jobs), missing_png, args.workers,
    )

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(score_one, jobs, chunksize=8), 1):
            rows.append(row)
            if i % 500 == 0:
                logger.info("  %d/%d", i, len(jobs))

    n_err = sum(1 for r in rows if r["error"])
    if n_err:
        logger.warning("%d stems errored; excluded from the cut", n_err)
    return rows, stem_chars


def _select(
    rows: list[dict],
    stem_chars: dict[str, list[str]],
    *,
    target: int,
    alpha: float,
    beta: float,
    min_artist_frac: float,
    min_char: int,
    char_threshold: int,
) -> tuple[set[str], dict]:
    """Stratified top-K cut with coverage repair (item5_plan.md Phase 1).

    1. composite keep-score ``-α·z(hf) - β·z(noise)`` (higher = keep);
    2. global top-K = ``target``;
    3. repair so every aspect bucket + every well-populated character survives
       and ≥ ``min_artist_frac`` of artists are retained — promoting the
       best-scored stem in each underfilled cell regardless of threshold.
    """
    # Only rows with both scores present are eligible.
    pool = [r for r in rows if not (math.isnan(r["hf_ratio"]) or math.isnan(r["noise_sigma"]))]
    z_hf = _zscores([r["hf_ratio"] for r in pool])
    z_ns = _zscores([r["noise_sigma"] for r in pool])
    for r, zh, zn in zip(pool, z_hf, z_ns):
        r["keep_score"] = -alpha * zh - beta * zn

    ranked = sorted(pool, key=lambda r: r["keep_score"], reverse=True)
    kept: set[str] = {r["stem"] for r in ranked[: min(target, len(ranked))]}

    # ── coverage repair ─────────────────────────────────────────────────
    # (a) every aspect bucket present in the pool keeps ≥ 1 stem.
    buckets_pool = {r["aspect_bucket"] for r in pool}
    kept_buckets = {r["aspect_bucket"] for r in ranked if r["stem"] in kept}
    promoted_bucket = 0
    for r in ranked:  # ranked → first hit per bucket is its best-scored stem
        if len(kept_buckets) >= len(buckets_pool):
            break
        if r["aspect_bucket"] not in kept_buckets:
            kept.add(r["stem"])
            kept_buckets.add(r["aspect_bucket"])
            promoted_bucket += 1

    # (b) ≥ min_artist_frac of distinct artists retained.
    artists_pool = {r["artist"] for r in pool}
    need_artists = math.ceil(min_artist_frac * len(artists_pool))
    represented = {r["artist"] for r in ranked if r["stem"] in kept}
    promoted_artist = 0
    for r in ranked:  # best-scored stem of each unrepresented artist, in rank order
        if len(represented) >= need_artists:
            break
        if r["artist"] not in represented:
            kept.add(r["stem"])
            represented.add(r["artist"])
            promoted_artist += 1

    # (c) characters with ≥ char_threshold in the pool keep ≥ min_char stems.
    char_pool: dict[str, int] = collections.Counter()
    for r in pool:
        for ch in stem_chars.get(r["stem"], []):
            char_pool[ch] += 1
    promoted_char = 0
    for ch, cnt in char_pool.items():
        if cnt < char_threshold:
            continue
        have = sum(1 for r in pool if r["stem"] in kept and ch in stem_chars.get(r["stem"], []))
        for r in ranked:
            if have >= min_char:
                break
            if r["stem"] in kept or ch not in stem_chars.get(r["stem"], []):
                continue
            kept.add(r["stem"])
            have += 1
            promoted_char += 1

    summary = {
        "n_pool": len(pool),
        "n_kept": len(kept),
        "promoted_for_coverage": {
            "bucket": promoted_bucket,
            "artist": promoted_artist,
            "character": promoted_char,
        },
        "buckets_pool": len(buckets_pool),
        "buckets_kept": len(kept_buckets),
        "artists_pool": len(artists_pool),
        "artists_kept": len(represented),
        "artists_kept_frac": round(len(represented) / max(1, len(artists_pool)), 3),
    }
    return kept, summary


def run_cut(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    rows, stem_chars = _score_pool(args)
    kept, summary = _select(
        rows, stem_chars,
        target=args.target, alpha=args.alpha, beta=args.beta,
        min_artist_frac=args.min_artist_frac, min_char=args.min_char,
        char_threshold=args.char_threshold,
    )

    # Sanity log: how much of the FEI-gap Tier-A did the noise cut actually drop?
    tier_a = _tier_a_artists(Path(args.fei_probe))
    if tier_a:
        a_rows = [r for r in rows if r["artist"] in tier_a]
        a_dropped = sum(1 for r in a_rows if r["stem"] not in kept)
        logger.info(
            "Tier-A: dropped %d/%d stems (%.0f%%) vs %.0f%% pool-wide",
            a_dropped, len(a_rows),
            100 * a_dropped / max(1, len(a_rows)),
            100 * (1 - len(kept) / max(1, summary["n_pool"])),
        )

    logger.info(
        "cut: kept %d/%d (target %d); coverage repair +%s; artists %.0f%% buckets %d/%d",
        summary["n_kept"], summary["n_pool"], args.target,
        summary["promoted_for_coverage"], 100 * summary["artists_kept_frac"],
        summary["buckets_kept"], summary["buckets_pool"],
    )

    if args.dry_run:
        logger.info("--dry_run: not writing keep_list.json")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "keep_list.json"
    scores = {
        r["stem"]: {
            "hf_ratio": r["hf_ratio"],
            "noise_sigma": r["noise_sigma"],
            "flat_floor": r["flat_floor"],
            "keep_score": r.get("keep_score"),
            "kept": r["stem"] in kept,
        }
        for r in rows
    }
    payload = {
        "version": 1,
        "target": args.target,
        "params": {
            "alpha": args.alpha,
            "beta": args.beta,
            "min_artist_frac": args.min_artist_frac,
            "min_char": args.min_char,
            "char_threshold": args.char_threshold,
            "fei_sigma_low_div": args.fei_sigma_low_div,
            "composite": "-alpha*z(hf_ratio) - beta*z(noise_sigma)",
        },
        "summary": summary,
        "kept": sorted(kept),
        "scores": scores,
    }
    out_path.write_text(json.dumps(payload, indent=1))
    logger.info("wrote %s (%d kept)", out_path, len(kept))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", default="post_image_dataset/lora")
    parser.add_argument("--resized_dir", default="post_image_dataset/resized")
    parser.add_argument(
        "--caption_index",
        default="post_image_dataset/captions/caption_index.json",
    )
    parser.add_argument("--out_dir", default="post_image_dataset/turbo_prep")
    parser.add_argument(
        "--fei_probe",
        default="bench/fera_artist/results/20260528-1902-turbo_C_phase0/paired_gap.csv",
        help="30-artist FEI-gap probe CSV; source of the Tier-A drop-rate log.",
    )
    parser.add_argument("--target", type=int, default=500, help="Target kept count.")
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Weight on -z(hf_ratio) in the keep-score (the uncertain axis).",
    )
    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="Weight on -z(noise_sigma) (the load-bearing noise-drop axis).",
    )
    parser.add_argument(
        "--min_artist_frac", type=float, default=0.8,
        help="Repair floor: fraction of distinct artists that must survive.",
    )
    parser.add_argument(
        "--min_char", type=int, default=3,
        help="Repair floor: kept stems per well-populated character.",
    )
    parser.add_argument(
        "--char_threshold", type=int, default=5,
        help="A character counts as well-populated at ≥ this many pool stems.",
    )
    parser.add_argument(
        "--fei_sigma_low_div", type=float, default=4.0,
        help="FEI σ_low divisor (matches library/runtime/fei.py default).",
    )
    parser.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4)))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Score + select + log the cut summary, but don't write keep_list.json.",
    )
    args = parser.parse_args()
    run_cut(args)


if __name__ == "__main__":
    main()
