"""Measure the radially-averaged power spectrum of Anima VAE latents.

This is the precondition test for Spectral Progressive Diffusion (SPD,
Xiao et al. arXiv:2605.18736). SPD's entire optimal-resolution schedule is
derived from Eq. (4):  P_omega  ∝  |omega|^{-beta},  beta ∈ [2, 3]
(they report beta≈1.92 for FLUX image latents). The method only pays off if
high frequencies genuinely carry far less signal than low frequencies — i.e.
the latent power spectrum decays steeply as a power law.

Anima trains on anime / illustration data, whose spectral statistics are NOT
obviously natural-image-like: large flat color fills (little HF energy) plus
hard line art (lots of HF energy at edges). So whether beta lands in the
paper's range is an open empirical question for *our* VAE + *our* data — that
is what this script answers.

Faithfulness choices:
  * Source images are read from `image_dataset/` ORIGINALS, not the resized
    cache under `post_image_dataset/resized/` — resizing is a low-pass that
    would artificially steepen the HF tail we are trying to measure.
  * Pixels go through the exact training transform (`IMAGE_TRANSFORMS`,
    i.e. [-1,1]) and `vae.encode_pixels_to_latents`, so we measure the
    per-channel-standardized latent the DiT actually denoises.

Output: per-bin radial profile CSV, a log-log plot with the fitted slope,
and a result.json envelope with beta / R^2 / per-channel spread + a verdict.

Usage:
  uv run python -m bench.spd.measure_latent_spectrum --num_images 200
  uv run python -m bench.spd.measure_latent_spectrum --max_side 2048 --num_images 120 --label hires
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
from library.datasets.image_utils import IMAGE_EXTENSIONS, IMAGE_TRANSFORMS
from library.models import qwen_vae

log = logging.getLogger("bench.spd.spectrum")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def discover_images(root: Path, n: int, seed: int) -> list[Path]:
    """Round-robin across subfolders so no single artist dominates the pool."""
    exts = {e.lower() for e in IMAGE_EXTENSIONS}
    by_folder: dict[Path, list[Path]] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            by_folder.setdefault(p.parent, []).append(p)
    rng = random.Random(seed)
    for paths in by_folder.values():
        rng.shuffle(paths)
    folders = sorted(by_folder.keys())
    rng.shuffle(folders)
    picks: list[Path] = []
    i = 0
    while len(picks) < n and any(by_folder.values()):
        f = folders[i % len(folders)]
        if by_folder[f]:
            picks.append(by_folder[f].pop())
        i += 1
        if i > n * 50:  # safety against pathological emptiness
            break
    return picks[:n]


def vae_dims(side_long: int, w: int, h: int, max_side: int) -> tuple[int, int]:
    """Resize longest side to <= max_side, round both to a multiple of 32px."""
    scale = min(1.0, max_side / float(side_long))
    nw = max(32, int(round(w * scale / 32)) * 32)
    nh = max(32, int(round(h * scale / 32)) * 32)
    return nw, nh


def radial_profile(power: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean power binned by normalized radial frequency k in [0, 1] (Nyquist=1).

    power: (H, W) fftshift-ed |F|^2 for one channel. Returns (n_bins,) with NaN
    for empty bins; caller aggregates in log space.
    """
    h, w = power.shape
    cy, cx = h // 2, w // 2
    ky = (np.arange(h) - cy) / (h / 2.0)
    kx = (np.arange(w) - cx) / (w / 2.0)
    kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)  # 0..sqrt(2)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(kr.ravel(), edges) - 1
    flat = power.ravel()
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out[b] = flat[m].mean()
    return out


def fit_slope(k: np.ndarray, logp: np.ndarray, lo: float, hi: float):
    """Linear fit log10(P) = a - beta*log10(k) over band k in [lo, hi]."""
    band = (k >= lo) & (k <= hi) & np.isfinite(logp)
    if band.sum() < 4:
        return None, None, int(band.sum())
    x = np.log10(k[band])
    y = logp[band]
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, _), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.linalg.lstsq(A, y, rcond=None)[0]
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(-slope), float(r2), int(band.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image_dir", default="image_dataset")
    ap.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    ap.add_argument("--num_images", type=int, default=200)
    ap.add_argument("--max_side", type=int, default=1536,
                    help="Resize longest side to <= this (px) before VAE encode.")
    ap.add_argument("--n_bins", type=int, default=96)
    ap.add_argument("--band", type=float, nargs=2, default=(0.06, 0.5),
                    help="Mid-frequency fit band as fraction of Nyquist.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--vae_chunk", type=int, default=0,
                    help="VAE spatial_chunk_size (0=off); raise if OOM at hires.")
    ap.add_argument("--label", default="latent-spectrum")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    root = Path(args.image_dir)
    picks = discover_images(root, args.num_images, args.seed)
    if not picks:
        raise SystemExit(f"No images found under {root}/")
    log.info(f"Sampled {len(picks)} images across {len({p.parent for p in picks})} folders.")

    log.info(f"Loading VAE from {args.vae} on {args.device} ({args.dtype}) ...")
    vae = qwen_vae.load_vae(
        args.vae, device=args.device,
        spatial_chunk_size=(args.vae_chunk or None),
    )
    vae.eval()

    n_ch = 16
    # Accumulate per-image, per-channel radial profiles in log10 space.
    sum_logp = np.zeros((n_ch, args.n_bins))
    cnt_logp = np.zeros((n_ch, args.n_bins))
    used = 0
    for j, p in enumerate(picks):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            log.warning(f"skip {p.name}: {e}")
            continue
        w, h = img.size
        nw, nh = vae_dims(max(w, h), w, h, args.max_side)
        img = img.resize((nw, nh), Image.LANCZOS)
        x = IMAGE_TRANSFORMS(np.array(img)).unsqueeze(0).to(args.device, dtype)
        with torch.no_grad():
            lat = vae.encode_pixels_to_latents(x)  # (1,16,h/8,w/8) or (1,16,1,h/8,w/8)
        lat = lat.float().cpu().numpy()
        lat = lat.reshape(n_ch, lat.shape[-2], lat.shape[-1])
        for c in range(n_ch):
            f = np.fft.fftshift(np.fft.fft2(lat[c]))
            power = (f.real ** 2 + f.imag ** 2)
            prof = radial_profile(power, args.n_bins)
            good = np.isfinite(prof) & (prof > 0)
            sum_logp[c, good] += np.log10(prof[good])
            cnt_logp[c, good] += 1
        used += 1
        if (j + 1) % 25 == 0:
            log.info(f"  {j + 1}/{len(picks)} encoded ...")

    k = (np.arange(args.n_bins) + 0.5) / args.n_bins  # bin-center normalized freq
    with np.errstate(invalid="ignore"):
        logp_per_ch = sum_logp / np.where(cnt_logp > 0, cnt_logp, np.nan)
    logp_mean = np.nanmean(logp_per_ch, axis=0)  # mean over channels

    lo, hi = args.band
    beta, r2, n_pts = fit_slope(k, logp_mean, lo, hi)
    per_ch = [fit_slope(k, logp_per_ch[c], lo, hi) for c in range(n_ch)]
    ch_betas = [b for (b, _, _) in per_ch if b is not None]

    in_paper_range = beta is not None and 2.0 <= beta <= 3.0
    decays = beta is not None and beta >= 1.0
    if in_paper_range:
        verdict = "FITS: beta within paper's [2,3] — SPD premise holds for Anima latents."
    elif decays:
        verdict = (f"PARTIAL: power-law decays (beta={beta:.2f}) but outside [2,3]; "
                   "SPD schedule would need re-derivation, premise weakened.")
    else:
        verdict = ("FAILS: no clean power-law decay — SPD's signal/noise-by-frequency "
                   "assumption does not hold for Anima latents.")

    run_dir = make_run_dir("spd", label=args.label)
    # radial profile CSV
    csv = run_dir / "radial_profile.csv"
    hdr = "k_norm,logP_mean," + ",".join(f"logP_ch{c}" for c in range(n_ch))
    rows = [hdr]
    for i in range(args.n_bins):
        vals = ",".join(f"{logp_per_ch[c, i]:.5f}" for c in range(n_ch))
        rows.append(f"{k[i]:.5f},{logp_mean[i]:.5f},{vals}")
    csv.write_text("\n".join(rows) + "\n")
    artifacts = ["radial_profile.csv"]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(k, logp_mean, "o-", ms=3, label="Anima latents (mean over 16 ch)", color="#1f4e8c")
        if beta is not None:
            band = (k >= lo) & (k <= hi)
            xb = np.log10(k[band])
            a = float(np.nanmean(logp_mean[band] + beta * xb))
            ax.plot(k[band], a - beta * np.log10(k[band]), "--", color="#c0392b",
                    label=f"fit  P∝k^-{beta:.2f}  (R²={r2:.3f})")
        ax.set_xscale("log"); ax.set_xlabel("normalized radial frequency  k / k_Nyquist")
        ax.set_ylabel("log10 P(k)"); ax.set_title("Anima VAE latent power spectrum")
        ax.axvspan(lo, hi, color="0.85", zorder=0, label="fit band")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(run_dir / "spectrum.png", dpi=130)
        artifacts.append("spectrum.png")
    except Exception as e:
        log.warning(f"plot skipped: {e}")

    metrics = {
        "beta": beta,
        "r2": r2,
        "fit_n_bins": n_pts,
        "fit_band": [lo, hi],
        "per_channel_beta_mean": float(np.mean(ch_betas)) if ch_betas else None,
        "per_channel_beta_std": float(np.std(ch_betas)) if ch_betas else None,
        "per_channel_beta_min": float(np.min(ch_betas)) if ch_betas else None,
        "per_channel_beta_max": float(np.max(ch_betas)) if ch_betas else None,
        "images_used": used,
        "max_side": args.max_side,
        "n_bins": args.n_bins,
        "paper_reference_beta_flux": 1.92,
        "paper_range": [2.0, 3.0],
        "in_paper_range": bool(in_paper_range),
        "verdict": verdict,
    }
    write_result(run_dir, script=__file__, args=args, metrics=metrics, artifacts=artifacts)

    log.info("\n" + "=" * 64)
    log.info(f"  beta = {beta:.3f}   R² = {r2:.3f}   (fit over k∈[{lo},{hi}], {n_pts} bins)")
    if ch_betas:
        log.info(f"  per-channel beta: {np.mean(ch_betas):.2f} ± {np.std(ch_betas):.2f}  "
                 f"[{np.min(ch_betas):.2f}, {np.max(ch_betas):.2f}]")
    log.info(f"  paper: FLUX beta≈1.92, claimed range [2,3]")
    log.info(f"  {verdict}")
    log.info(f"  → {run_dir}")
    log.info("=" * 64)


if __name__ == "__main__":
    main()
