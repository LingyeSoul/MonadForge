"""Per-artist breakdown of the Anima latent power-law exponent beta.

The aggregate fit in `measure_latent_spectrum.py` (beta≈2.26) pools all
artists. But anime styles vary a lot — cel-shaded/flat vs painterly/detailed —
so the SPD premise could hold on average yet fail for whole sub-styles. This
script fits beta *within each artist folder* and reports the distribution, so
we know whether SPD's resolution schedule would be robust across the dataset or
need per-style adaptation.

Usage:
  uv run python -m bench.spd.per_artist_spectrum --n_artists 30 --per_artist 12
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bench._common import make_run_dir, write_result
from bench.spd.measure_latent_spectrum import (
    fit_slope,
    radial_profile,
    vae_dims,
)
from library.datasets.image_utils import IMAGE_EXTENSIONS, IMAGE_TRANSFORMS
from library.models import qwen_vae

log = logging.getLogger("bench.spd.per_artist")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image_dir", default="image_dataset")
    ap.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    ap.add_argument("--n_artists", type=int, default=30)
    ap.add_argument("--per_artist", type=int, default=12)
    ap.add_argument("--min_images", type=int, default=6,
                    help="Skip folders with fewer than this many images.")
    ap.add_argument("--max_side", type=int, default=1280)
    ap.add_argument("--n_bins", type=int, default=96)
    ap.add_argument("--band", type=float, nargs=2, default=(0.06, 0.5))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--label", default="per-artist")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    exts = {e.lower() for e in IMAGE_EXTENSIONS}
    root = Path(args.image_dir)
    rng = random.Random(args.seed)

    folders = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = [p for p in d.iterdir() if p.suffix.lower() in exts]
        if len(imgs) >= args.min_images:
            folders.append((d.name, imgs))
    rng.shuffle(folders)
    folders = folders[: args.n_artists]
    if not folders:
        raise SystemExit(f"No artist folders with >= {args.min_images} images under {root}/")
    log.info(f"Fitting beta within {len(folders)} artist folders "
             f"({args.per_artist} imgs each, max_side={args.max_side}) ...")

    vae = qwen_vae.load_vae(args.vae, device=args.device)
    vae.eval()
    n_ch = 16
    lo, hi = args.band

    rows = []  # (artist, n, beta, r2)
    curves = {}  # artist -> (k, logp_mean)
    for name, imgs in folders:
        rng.shuffle(imgs)
        sel = imgs[: args.per_artist]
        sum_logp = np.zeros((n_ch, args.n_bins))
        cnt = np.zeros((n_ch, args.n_bins))
        used = 0
        for p in sel:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            w, h = img.size
            nw, nh = vae_dims(max(w, h), w, h, args.max_side)
            img = img.resize((nw, nh), Image.LANCZOS)
            x = IMAGE_TRANSFORMS(np.array(img)).unsqueeze(0).to(args.device, dtype)
            with torch.no_grad():
                lat = vae.encode_pixels_to_latents(x).float().cpu().numpy()
            lat = lat.reshape(n_ch, lat.shape[-2], lat.shape[-1])
            for c in range(n_ch):
                f = np.fft.fftshift(np.fft.fft2(lat[c]))
                prof = radial_profile(f.real ** 2 + f.imag ** 2, args.n_bins)
                good = np.isfinite(prof) & (prof > 0)
                sum_logp[c, good] += np.log10(prof[good])
                cnt[c, good] += 1
            used += 1
        if used == 0:
            continue
        k = (np.arange(args.n_bins) + 0.5) / args.n_bins
        with np.errstate(invalid="ignore"):
            logp = np.nanmean(sum_logp / np.where(cnt > 0, cnt, np.nan), axis=0)
        beta, r2, _ = fit_slope(k, logp, lo, hi)
        if beta is None:
            continue
        rows.append((name, used, beta, r2))
        curves[name] = (k, logp)
        log.info(f"  {name:28s} n={used:2d}  beta={beta:.3f}  R²={r2:.4f}")

    betas = np.array([b for (_, _, b, _) in rows])
    in_range = int(((betas >= 2.0) & (betas <= 3.0)).sum())
    summary = {
        "n_artists_fit": len(rows),
        "beta_mean": float(betas.mean()),
        "beta_std": float(betas.std()),
        "beta_min": float(betas.min()),
        "beta_max": float(betas.max()),
        "n_in_paper_range": in_range,
        "frac_in_paper_range": in_range / len(rows),
        "decays_all": bool((betas >= 1.0).all()),
    }

    run_dir = make_run_dir("spd", label=args.label)
    csv = run_dir / "per_artist_beta.csv"
    csv.write_text("artist,n_images,beta,r2\n" +
                   "\n".join(f"{a},{n},{b:.4f},{r:.4f}" for (a, n, b, r) in
                            sorted(rows, key=lambda x: x[2])) + "\n")
    artifacts = ["per_artist_beta.csv"]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
        for name, (k, logp) in curves.items():
            a1.plot(k, logp, "-", lw=0.7, alpha=0.5)
        a1.set_xscale("log"); a1.set_xlabel("k / k_Nyquist"); a1.set_ylabel("log10 P(k)")
        a1.axvspan(lo, hi, color="0.88", zorder=0)
        a1.set_title(f"per-artist spectra (n={len(rows)})")
        a2.hist(betas, bins=12, color="#1f4e8c", alpha=0.85)
        a2.axvspan(2.0, 3.0, color="#2ecc71", alpha=0.18, label="paper [2,3]")
        a2.axvline(betas.mean(), color="#c0392b", ls="--", label=f"mean {betas.mean():.2f}")
        a2.set_xlabel("beta"); a2.set_ylabel("# artists"); a2.legend(fontsize=8)
        a2.set_title("beta distribution across artists")
        fig.tight_layout(); fig.savefig(run_dir / "per_artist.png", dpi=130)
        artifacts.append("per_artist.png")
    except Exception as e:
        log.warning(f"plot skipped: {e}")

    write_result(run_dir, script=__file__, args=args, metrics=summary, artifacts=artifacts)
    log.info("\n" + "=" * 64)
    log.info(f"  beta across {len(rows)} artists: {betas.mean():.3f} ± {betas.std():.3f}  "
             f"[{betas.min():.2f}, {betas.max():.2f}]")
    log.info(f"  in paper's [2,3]: {in_range}/{len(rows)}  "
             f"({100 * in_range / len(rows):.0f}%);  all decay (beta≥1): {summary['decays_all']}")
    log.info(f"  → {run_dir}")
    log.info("=" * 64)


if __name__ == "__main__":
    main()
