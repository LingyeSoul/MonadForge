"""Cheap σ-signal probe for the training-time timestep-sampling schedule.

THE QUESTION (see the conversation that spawned this): is Anima's default
``timestep_sampling = "sigmoid"`` (a logit-normal(0,1) over σ, bell at σ=0.5)
pointed at the σ region where a LoRA adapter actually has something to learn?
``discrete_flow_shift`` is a *no-op* under sigmoid sampling — it's only read by
the ``"shift"`` branch (``library/runtime/noise.py:110``) — so the only live
schedule knob here is where the sampling density sits on the σ axis.

This is a NO-TRAINING probe. For a handful of *real* cached dataset latents
(``post_image_dataset/lora/**``, paired with their cached ``crossattn_emb``)
it does, for each σ on a grid, a single bare-DiT forward and reconstructs the
model's x0 estimate:

    x_σ      = (1-σ)·x0 + σ·ε            (Anima FM noising, noise.py:164)
    v        = DiT(x_σ, σ, crossattn)    (σ∈[0,1] is the time arg — generation.py:296)
    x0_pred  = x_σ − σ·v                 (target velocity = ε − x0, train.py:922)

At low σ the base already reconstructs x0 almost exactly (nothing for an
adapter to add); at high σ x0_pred collapses toward the conditional mean
(global structure being decided — where capacity matters). The deliverable is
**visual**: ``x0_vs_sigma.png`` is a per-sample strip of decoded x0_pred across
σ, next to the true x0. Eyeball where the prediction stops resembling the
target — that's where training signal lives.

Alongside it, ``density_overlay.png`` plots each candidate schedule's *sampling
density* over the same σ axis (sigmoid default, uniform, a high-σ-skewed
logit-normal, and sigmoid∘t_max=0.95) overlaid on the measured per-σ
reconstruction error. If the sigmoid bell spends a lot of mass in the low-σ
region where error≈0, that mass is wasted on samples the base already nails.

HONEST CAVEAT (your own finding — ``project_fm_val_loss_uninformative``):
FM-MSE does NOT track final quality on Anima. The numeric curves here are a
*diagnostic of where the base is uncertain*, not proof that resampling there
trains a better adapter. Only a CMMD-scored training sweep settles "optimal".
This probe exists to pick informed arms for that sweep and to make the
schedule-vs-signal mismatch visible.

Usage:
  uv run python -m bench.timestep_sampling.probe_sigma_signal
  uv run python -m bench.timestep_sampling.probe_sigma_signal --num_samples 4 --seed 0
  uv run python -m bench.timestep_sampling.probe_sigma_signal --adapter output/ckpt/foo.safetensors
  uv run python -m bench.timestep_sampling.probe_sigma_signal \
      --sigmas 0.05 0.15 0.3 0.45 0.6 0.75 0.9 0.97
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from bench._anima import add_common_args, build_anima
from bench._common import make_run_dir, write_result
from library.io.cache import load_cached_crossattn_emb, load_cached_latents

log = logging.getLogger("bench.timestep_sampling.probe")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
DEFAULT_VAE = "models/vae/qwen_image_vae.safetensors"
DEFAULT_DATA = "post_image_dataset/lora"
DEFAULT_SIGMAS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

_RES_RE = re.compile(r"_(\d+)x(\d+)_anima\.npz$")


# ── cached-pair discovery (recursive — caches are nested per-artist) ──────────


def discover_pairs(data_dir: str, n: int, seed: int) -> list[tuple[str, str]]:
    """Find ``(latent_npz, te_safetensors)`` pairs under ``data_dir`` (recursive).

    ``discover_bucketed_samples`` in library/io/cache.py is non-recursive and
    the lora cache is nested by artist, so we rglob here. Each sample is its
    own forward (no same-shape batching needed), so we don't group by bucket.
    """
    pairs: list[tuple[str, str]] = []
    for p in sorted(Path(data_dir).rglob("*_anima.npz")):
        m = _RES_RE.search(p.name)
        if not m:
            continue
        stem = p.name[: m.start()]
        te = p.parent / f"{stem}_anima_te.safetensors"
        if te.exists():
            pairs.append((str(p), str(te)))
    if not pairs:
        raise SystemExit(f"no paired (latent, TE) caches under {data_dir}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    return [pairs[int(i)] for i in idx]


# ── candidate sampling schedules (Monte-Carlo density over σ) ─────────────────


def schedule_density(name: str, sigma_grid: np.ndarray, n: int = 200_000) -> np.ndarray:
    """Return a normalized density of σ under one candidate schedule, evaluated
    at ``sigma_grid`` bin centers. Drawn empirically so shift/clamp transforms
    are trivial to express (matches library/runtime/noise.py exactly)."""
    rng = np.random.default_rng(12345)
    z = rng.standard_normal(n)
    if name == "sigmoid (default, scale=1)":
        s = 1.0 / (1.0 + np.exp(-z))  # sigmoid(N(0,1))
    elif name == "uniform":
        s = rng.random(n)
    elif name == "logit_normal μ=+0.5 (high-σ skew)":
        s = 1.0 / (1.0 + np.exp(-(z + 0.5)))  # sigmoid(N(0.5,1))
    elif name == "sigmoid ∘ t_max=0.95":
        s = (1.0 / (1.0 + np.exp(-z))) * 0.95  # noise.py t_max rescale, lo=0
    else:
        raise ValueError(name)
    edges = np.concatenate([[0.0], (sigma_grid[:-1] + sigma_grid[1:]) / 2.0, [1.0]])
    hist, _ = np.histogram(s, bins=edges, density=True)
    return hist


SCHEDULES = [
    "sigmoid (default, scale=1)",
    "uniform",
    "logit_normal μ=+0.5 (high-σ skew)",
    "sigmoid ∘ t_max=0.95",
]


# ── decode helpers ────────────────────────────────────────────────────────────


def _to_pil(pixels: torch.Tensor) -> Image.Image:
    """(C,H,W) in [-1,1] → PIL RGB."""
    arr = pixels.clamp(-1, 1).add(1).mul(127.5).round().byte()
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())


def _thumb(img: Image.Image, max_px: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    scale = max_px / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def build_sample_strip(
    images: list[Image.Image],
    col_labels: list[str],
    title: str,
    max_px: int,
) -> Image.Image:
    """One strip for ONE sample: a single row of [true x0] + x0_pred(σ).

    Kept per-sample (rather than a stacked grid) so each is legible at a
    glance — open them side by side in an image viewer to compare samples.
    """
    thumbs = [_thumb(im, max_px) for im in images]
    cell_w = max(im.width for im in thumbs)
    cell_h = max(im.height for im in thumbs)
    pad, header, title_h = 6, 26, 20
    ncol = len(images)
    W = ncol * (cell_w + pad) + pad
    H = title_h + header + cell_h + 2 * pad
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 4), title, fill="black")
    y = title_h + header + pad
    for c, (lab, im) in enumerate(zip(col_labels, thumbs)):
        x = c * (cell_w + pad) + pad
        draw.multiline_text((x + 2, title_h + 2), lab, fill="black", spacing=2)
        canvas.paste(im, (x, y))
    return canvas


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dit", default=DEFAULT_DIT)
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA checkpoint. Default: bare base DiT (the model a "
        "fresh adapter would be adapting — the right reference for 'where is "
        "there signal to learn').",
    )
    ap.add_argument("--num_samples", type=int, default=3)
    ap.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=DEFAULT_SIGMAS,
        help="σ grid for both the strip columns and the numeric curves.",
    )
    ap.add_argument(
        "--strip_max_px",
        type=int,
        default=200,
        help="Max edge (px) of each decoded tile in x0_vs_sigma.png.",
    )
    add_common_args(p := ap)  # --label/--seed/--device/--dtype/--attn_mode/--compile/…
    args = p.parse_args()

    sigma_grid = sorted(float(s) for s in args.sigmas)
    log.info(f"σ grid: {sigma_grid}")

    pairs = discover_pairs(args.data_dir, args.num_samples, args.seed)
    log.info(f"{len(pairs)} sample(s) drawn from {args.data_dir}")

    bundle = build_anima(args, adapter=args.adapter, train_mode=False)
    anima, device, dtype = bundle.anima, bundle.device, bundle.dtype

    from library.models import qwen_vae

    log.info("loading VAE …")
    vae = qwen_vae.load_vae(args.vae, device="cpu", spatial_chunk_size=64)
    vae.to(torch.bfloat16).eval()

    def decode(lat5d: torch.Tensor) -> Image.Image:
        vae.to(device)
        with torch.no_grad():
            px = vae.decode_to_pixels(lat5d.to(device, dtype=vae.dtype))
        vae.to("cpu")
        if px.ndim == 5:
            px = px.squeeze(2)
        return _to_pil(px[0].float().cpu())

    # per-σ accumulators (averaged across samples)
    fm_mse = {s: [] for s in sigma_grid}
    px_mse = {s: [] for s in sigma_grid}
    lat_mse = {s: [] for s in sigma_grid}
    # per-sample render data: (title, [ref + pred imgs], [px_mse per σ])
    samples: list[tuple[str, list[Image.Image], list[float]]] = []

    for si, (npz_path, te_path) in enumerate(pairs):
        lat, _res, _oh, _ow = load_cached_latents(npz_path)  # (C,H,W) float32
        emb = load_cached_crossattn_emb(te_path)  # (S,D) float32
        if emb is None:
            log.warning(f"  sample {si}: no crossattn_emb, skipping")
            continue
        x0 = lat.to(device, dtype).unsqueeze(0).unsqueeze(2)  # (1,C,1,H,W)
        emb = emb.to(device, dtype).unsqueeze(0)  # (1,S,D)
        H, W = x0.shape[-2], x0.shape[-1]
        pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)

        # one fixed noise draw per sample → strip columns are comparable
        g = torch.Generator(device=device).manual_seed(args.seed + si)
        eps = torch.randn(x0.shape, generator=g, device=device, dtype=dtype)
        x0_f, eps_f = x0.float(), eps.float()
        v_target = eps_f - x0_f  # rectified-flow target (train.py:922)

        ref_img = decode(x0)
        imgs = [ref_img]
        px_row: list[float] = []
        log.info(f"=== sample {si}: {Path(npz_path).name}  ({H * 8}×{W * 8}px) ===")
        for s in sigma_grid:
            noisy = ((1.0 - s) * x0_f + s * eps_f).to(dtype)
            t = torch.full((1,), float(s), device=device, dtype=dtype)
            with torch.no_grad():
                v = anima(noisy, t, emb, padding_mask=pad).float()
            x0_pred = noisy.float() - s * v
            fm_mse[s].append(float(((v - v_target) ** 2).mean()))
            lat_mse[s].append(float(((x0_pred - x0_f) ** 2).mean()))
            pred_img = decode(x0_pred.to(dtype))
            ra = np.asarray(ref_img.resize((96, 96)), np.float32) / 255.0
            pa = np.asarray(pred_img.resize((96, 96)), np.float32) / 255.0
            this_px = float(((ra - pa) ** 2).mean())
            px_mse[s].append(this_px)
            px_row.append(this_px)
            imgs.append(pred_img)
            log.info(
                f"  σ={s:.2f}  fm_mse={fm_mse[s][-1]:.4f}  "
                f"x0_lat_mse={lat_mse[s][-1]:.4f}  x0_px_mse={this_px:.4f}"
            )
        title = (
            f"s{si}: {Path(npz_path).name}  ({H * 8}×{W * 8}px)  —  x0_pred = x_σ − σ·v"
        )
        samples.append((title, imgs, px_row))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not samples:
        raise SystemExit("no samples produced (all missing crossattn_emb?)")

    run_dir = make_run_dir(
        "timestep_sampling", label=args.label or ("adapter" if args.adapter else "base")
    )

    # ── x0-vs-σ strips — one file per sample (legible at a glance) ──
    strip_names: list[str] = []
    for si, (title, imgs, px_row) in enumerate(samples):
        col_labels = ["x0 (true)"] + [
            f"σ={s:.2f}\nmse={e:.4f}" for s, e in zip(sigma_grid, px_row)
        ]
        strip = build_sample_strip(imgs, col_labels, title, args.strip_max_px)
        name = f"x0_vs_sigma_s{si}.png"
        strip.save(run_dir / name)
        strip_names.append(name)

    # ── mean curves ──
    mean_fm = np.array([float(np.mean(fm_mse[s])) for s in sigma_grid])
    mean_px = np.array([float(np.mean(px_mse[s])) for s in sigma_grid])
    mean_lat = np.array([float(np.mean(lat_mse[s])) for s in sigma_grid])
    sg = np.array(sigma_grid)

    # ── CSV ──
    csv = run_dir / "sigma_signal.csv"
    with open(csv, "w") as f:
        f.write("sigma,fm_mse,x0_lat_mse,x0_px_mse\n")
        for i, s in enumerate(sigma_grid):
            f.write(f"{s},{mean_fm[i]},{mean_lat[i]},{mean_px[i]}\n")

    # ── density overlay ──
    densities = {name: schedule_density(name, sg) for name in SCHEDULES}
    overlay_ok = True
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(9, 5))
        for name, dens in densities.items():
            ax1.plot(sg, dens, marker="o", ms=3, label=name)
        ax1.set_xlabel("σ  (0 = clean latent, 1 = pure noise)")
        ax1.set_ylabel("sampling density  p(σ)")
        ax1.set_xlim(0, 1)
        ax2 = ax1.twinx()
        sig = mean_px / (mean_px.max() + 1e-9)
        ax2.plot(sg, sig, "k--", lw=2, label="recon error (norm.) — where signal is")
        ax2.set_ylabel("normalized base recon error  (x0_px_mse)")
        ax2.set_ylim(0, 1.05)
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(
            lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper center"
        )
        ttl = "adapter" if args.adapter else "bare base DiT"
        ax1.set_title(
            f"schedule sampling density vs where the base is uncertain ({ttl})"
        )
        fig.tight_layout()
        fig.savefig(run_dir / "density_overlay.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001 — plotting is best-effort
        overlay_ok = False
        log.warning(f"density_overlay.png skipped: {e}")

    # ── quantify the schedule-vs-signal mismatch ──
    # "low-signal" σ = where the base already reconstructs well (norm err < 0.2).
    norm_err = mean_px / (mean_px.max() + 1e-9)
    low_signal_max_sigma = float(
        max([s for s, e in zip(sigma_grid, norm_err) if e < 0.2], default=0.0)
    )
    # fraction of each schedule's sampling mass spent at/below that σ (≈ wasted)
    rng = np.random.default_rng(777)
    wasted = {}
    for name in SCHEDULES:
        z = rng.standard_normal(200_000)
        if name == "sigmoid (default, scale=1)":
            s = 1 / (1 + np.exp(-z))
        elif name == "uniform":
            s = rng.random(200_000)
        elif name == "logit_normal μ=+0.5 (high-σ skew)":
            s = 1 / (1 + np.exp(-(z + 0.5)))
        else:
            s = (1 / (1 + np.exp(-z))) * 0.95
        wasted[name] = float((s <= low_signal_max_sigma).mean())

    metrics = {
        "model": "adapter" if args.adapter else "bare_base_dit",
        "adapter": args.adapter,
        "n_samples": len(samples),
        "sigma_grid": sigma_grid,
        "mean_fm_mse": mean_fm.tolist(),
        "mean_x0_lat_mse": mean_lat.tolist(),
        "mean_x0_px_mse": mean_px.tolist(),
        "low_signal_max_sigma": low_signal_max_sigma,
        "low_signal_mass_fraction_per_schedule": wasted,
        "note": (
            "FM-MSE/recon-error is a 'where is the base uncertain' diagnostic, "
            "NOT a quality metric (project_fm_val_loss_uninformative). "
            "Coherence/where-signal-dies is a VISUAL call — open x0_vs_sigma_s*.png. "
            "low_signal_mass_fraction = sampling mass spent at σ where the base "
            "already reconstructs x0 (norm recon err < 0.2) ≈ wasted training draws."
        ),
    }
    artifacts = [*strip_names, "sigma_signal.csv"]
    if overlay_ok:
        artifacts.append("density_overlay.png")
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        device=device,
    )

    log.info("\n" + "=" * 70)
    log.info(
        f"  base reconstructs well (low signal) up to σ≈{low_signal_max_sigma:.2f}"
    )
    for name in SCHEDULES:
        log.info(f"  mass below that σ — {name:38s}: {wasted[name]:.1%}")
    log.info("  → higher % = more sampling draws spent where the base needs no help")
    log.info(f"  open: {run_dir}/x0_vs_sigma_s*.png  and  density_overlay.png")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
