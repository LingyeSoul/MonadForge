#!/usr/bin/env python3
"""σ staircase on MOSS-Image: how close is one bare forward's ``x0_pred`` at each σ.

Tier 3 §3 of ``docs/proposal/moss_anima_negatives_revisit.md``. The Anima reading
(``docs/findings/sigma_signal_where_anima_resolves.md``): ``x0_pred`` is a
recognisable picture at σ ≈ 0.75 and essentially final below σ ≈ 0.45, with
normalised latent-MSE 0.331 / 0.214 / 0.142 / 0.092 at σ 0.75 / 0.65 / 0.55 / 0.45
(normalised by the σ = 0.95 value).

Each cached latent is noised as ``(1 − σ)·x0 + σ·ε`` (one ε per sample, shared over
σ) at every σ of the sampler's own grid (``snapped_sigmas(steps, shift)``), one
reference per-sample forward at true caption length (CFG 1, no LoRA), then
``x0_pred = x_σ − σ·v``. Per σ, pooled over samples:

  * ``mse`` — ``mean((x0_pred − x0)²)``
  * ``rel`` — ``mse / mean(x0²)``: against predicting zero (latents are BN-normalised)
  * ``norm95`` — ``mse / mse(σ ≈ 0.95)``: Anima's normalisation
  * ``rel_low`` / ``norm95_low`` — the same on a 4×4 average-pooled latent (layout,
    1/64 of the image side), so the composition knee separates from the detail knee

``knees`` interpolates the σ where ``norm95`` crosses each Anima level, and how many
sampler steps sit above it. ``training_mass_above`` is the share of training σ draws
(``train.sample_sigma`` at the ``TrainRequest`` defaults) above each knee.

Then the DiT leaves the card and the VAE decodes ``x0_pred`` for ``--grid_n`` samples
at the grid σ nearest each of ``--grid_sigmas`` (``grid_<model>_<stem>.png``).

    make daemon-run ARGS="bench/moss_sigma_staircase/probe_x0.py --label base-sft"

Writes ``staircase.npz``, ``staircase.png``, the grids and ``result.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.moss import cache as moss_cache  # noqa: E402
from library.moss.loader import empty_cache, load_transformer, load_vae  # noqa: E402
from library.moss.model import decode, snapped_sigmas  # noqa: E402
from library.moss.requests import (  # noqa: E402
    DEFAULT_SHIFT,
    DEFAULT_STEPS,
    TrainRequest,
)
from library.moss.train import sample_sigma  # noqa: E402
from library.runtime.blockswap import place  # noqa: E402

# docs/findings/sigma_signal_where_anima_resolves.md, Evidence 1
ANIMA_NORM95 = {
    0.05: 0.005,
    0.15: 0.019,
    0.25: 0.035,
    0.35: 0.058,
    0.45: 0.092,
    0.55: 0.142,
    0.65: 0.214,
    0.75: 0.331,
    0.85: 0.537,
    0.95: 1.000,
}
# the Anima σ each level was read at: "recognisable", 20 %, "essentially final"
KNEE_LEVELS = {"0.75_recognisable": 0.331, "0.20": 0.2, "0.45_final": 0.092}
ANIMA_STEPS, ANIMA_SHIFT = 28, 3.0
POOL = 4


def crossing(sigmas: np.ndarray, values: np.ndarray, level: float) -> float | None:
    """Largest σ below which ``values`` stays under ``level`` (σ descending in)."""
    order = np.argsort(sigmas)
    s, v = sigmas[order], values[order]
    above = np.nonzero(v >= level)[0]
    if len(above) == 0:
        return float(s[-1])
    i = above[0]
    if i == 0:
        return None
    # linear between the last point under and the first point over
    s0, s1, v0, v1 = s[i - 1], s[i], v[i - 1], v[i]
    return float(s0 + (level - v0) * (s1 - s0) / (v1 - v0))


def anima_grid() -> np.ndarray:
    u = np.linspace(1.0, 0.0, ANIMA_STEPS + 1)[:-1]
    return ANIMA_SHIFT * u / (1 + (ANIMA_SHIFT - 1) * u)


def training_draws(n: int, seed: int) -> np.ndarray:
    req = TrainRequest()
    gen = torch.Generator().manual_seed(seed)
    return np.array(
        [
            sample_sigma(
                gen, req.sigma_mu, req.sigma_s, req.sigma_cutoff, req.sigma_uniform
            )
            for _ in range(n)
        ]
    )


def plot(path, sigmas, per_model):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    a_s = np.array(sorted(ANIMA_NORM95))
    a_v = np.array([ANIMA_NORM95[k] for k in a_s])
    for name, m in per_model.items():
        axes[0].plot(sigmas, m["norm95"], marker=".", label=f"MOSS {name}")
        axes[0].plot(sigmas, m["norm95_low"], ls="--", label=f"MOSS {name} (4×4 pool)")
        axes[1].plot(sigmas, m["rel"], marker=".", label=f"MOSS {name}")
        axes[1].plot(sigmas, m["rel_low"], ls="--", label=f"MOSS {name} (4×4 pool)")
    axes[0].plot(a_s, a_v, "k-o", ms=3, label="Anima base")
    for level in KNEE_LEVELS.values():
        axes[0].axhline(level, color="grey", ls=":", lw=1)
    for ax in axes:
        for s in sigmas:
            ax.axvline(s, color="tab:red", alpha=0.08, lw=1)
        ax.set(xlabel="σ", xlim=(0, 1), ylim=(0, None))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set(ylabel="latent-MSE / MSE(σ≈0.95)", title="x0_pred error (Anima norm)")
    axes[1].set(ylabel="latent-MSE / E[x0²]", title="x0_pred error vs predicting 0")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_grid(path, truth, preds, labels, tile=256):
    from PIL import Image, ImageDraw

    tiles = [truth, *preds]
    names = ["x0", *labels]
    w0, h0 = tiles[0].size
    th = round(tile * h0 / w0)
    cols = min(len(tiles), 8)
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * tile, rows * (th + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (im, name) in enumerate(zip(tiles, names)):
        x, y = (i % cols) * tile, (i // cols) * (th + 18)
        sheet.paste(im.resize((tile, th)), (x, y + 18))
        draw.text((x + 4, y + 3), name, fill="black")
    sheet.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default="post_image_dataset/moss/_calib_mixed")
    ap.add_argument("--n", type=int, default=16, help="samples, evenly spaced by stem")
    ap.add_argument("--models", default="base,sft")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--shift", type=float, default=DEFAULT_SHIFT)
    ap.add_argument("--grid_n", type=int, default=4, help="samples decoded as grids")
    ap.add_argument(
        "--grid_sigmas", default="1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.6,0.5,0.4,0.3,0.1"
    )
    ap.add_argument(
        "--blocks_to_swap", type=int, default=-1, help="-1 = size from free VRAM"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sigmas = snapped_sigmas(args.steps, args.shift).numpy().astype(np.float64)
    items = moss_cache.list_items(resolve_under_home(args.cache))
    if not items:
        raise SystemExit(f"no cached pairs under {args.cache}")
    pick = np.linspace(0, len(items) - 1, min(args.n, len(items))).round().astype(int)
    items = [items[i] for i in sorted(set(pick))]
    grid_items = items[:: max(1, len(items) // args.grid_n)][: args.grid_n]
    want = [float(s) for s in args.grid_sigmas.split(",")]
    grid_idx = sorted({int(np.abs(sigmas - s).argmin()) for s in want})
    print(
        f"{len(items)} samples × {len(sigmas)} σ (steps {args.steps}, shift "
        f"{args.shift}) from {args.cache}, models {models}",
        flush=True,
    )

    i95 = int(np.abs(sigmas - 0.95).argmin())
    start_heartbeat()
    per_model: dict[str, dict[str, np.ndarray]] = {}
    per_sample: dict[str, list[list[float]]] = {}
    grids: dict[tuple[str, str], list[torch.Tensor]] = {}
    for model in models:
        transformer = load_transformer(args.model_dir, model)
        transformer.requires_grad_(False).eval()
        attached = place(
            transformer,
            "layers",
            device,
            blocks_to_swap=None if args.blocks_to_swap < 0 else args.blocks_to_swap,
            activation_reserve_gb=4.0,
            label="transformer",
        )
        empty_cache()
        # sums over samples: err², err² pooled, x0², x0² pooled (per σ)
        acc = np.zeros((4, len(sigmas)))
        rel_rows = []
        gen = torch.Generator(device=device)
        t0 = time.time()
        done, total = 0, len(items) * len(sigmas)
        with torch.no_grad():
            for k, item in enumerate(items):
                latent, embeds = moss_cache.load_item(item)
                x0 = latent.unsqueeze(0).to(device, torch.float32)
                embeds = embeds.unsqueeze(0).to(device)
                gen.manual_seed(args.seed * 100003 + k)
                noise = torch.randn(x0.shape, generator=gen, device=device)
                x0_low = F.avg_pool2d(x0, POOL)
                ref2 = x0.square().sum().item()
                ref2_low = x0_low.square().sum().item()
                row = []
                for s, sigma in enumerate(sigmas):
                    noisy = (1.0 - sigma) * x0 + sigma * noise
                    timestep = torch.tensor([1000.0 * sigma], device=device)
                    v = transformer(noisy, timestep, embeds, return_dict=False)[0]
                    x0_pred = noisy - sigma * v.float()
                    err = (x0_pred - x0).square().sum().item()
                    err_low = (
                        (F.avg_pool2d(x0_pred, POOL) - x0_low).square().sum().item()
                    )
                    acc[:, s] += (err, err_low, ref2, ref2_low)
                    row.append(err / ref2)
                    if item in grid_items and s in grid_idx:
                        grids.setdefault((model, item.stem), []).append(x0_pred.cpu())
                    done += 1
                    print(
                        f"  [{model}] progress {done}/{total}  {item.stem} "
                        f"σ={sigma:.3f} rel {err / ref2:.3f}  "
                        f"{(time.time() - t0) / done:.2f}s/fwd",
                        flush=True,
                    )
                rel_rows.append(row)
                if item in grid_items:
                    grids.setdefault((model, item.stem), []).insert(0, x0.cpu())
        if attached is not None:
            attached.detach()
        transformer.to("cpu")
        del transformer
        empty_cache()
        mse_rel = acc[0] / acc[2]
        mse_rel_low = acc[1] / acc[3]
        per_model[model] = {
            "rel": mse_rel,
            "rel_low": mse_rel_low,
            "norm95": mse_rel / mse_rel[i95],
            "norm95_low": mse_rel_low / mse_rel_low[i95],
        }
        per_sample[model] = rel_rows

    run_dir = make_run_dir("moss_sigma_staircase", args.label)
    np.savez(
        run_dir / "staircase.npz",
        sigmas=sigmas,
        models=np.array(models),
        **{f"{m}_{k}": v for m, d in per_model.items() for k, v in d.items()},
        **{f"{m}_per_sample_rel": np.array(v) for m, v in per_sample.items()},
    )
    plot(run_dir / "staircase.png", sigmas, per_model)

    # decode with the DiT off the card
    from diffusers.image_processor import VaeImageProcessor

    vae = load_vae(args.model_dir, device=device)
    processor = VaeImageProcessor(vae_scale_factor=16)
    labels = [f"s={sigmas[i]:.3f}" for i in grid_idx]  # PIL default font has no σ
    grid_files = []
    for (model, stem), lats in grids.items():
        ims = [
            processor.postprocess(decode(z.to(device), vae), output_type="pil")[0]
            for z in lats
        ]
        name = f"grid_{model}_{stem}.png"
        save_grid(run_dir / name, ims[0], ims[1:], labels)
        grid_files.append(name)
        print(f"[vae] {name}", flush=True)
    del vae
    empty_cache()

    def r4(v):
        return round(float(v), 4)

    draws = training_draws(20000, args.seed)
    a_grid = anima_grid()
    metrics = {}
    for model, d in per_model.items():
        knees = {}
        for name, level in KNEE_LEVELS.items():
            for kind in ("norm95", "norm95_low"):
                sk = crossing(sigmas, d[kind], level)
                knees[f"{kind}@{name}"] = {
                    "sigma": None if sk is None else r4(sk),
                    "steps_above": None if sk is None else int((sigmas > sk).sum()),
                    "training_mass_above": None
                    if sk is None
                    else r4((draws > sk).mean()),
                }
        metrics[model] = {
            "knees": knees,
            "by_sigma": {
                f"{s:.3f}": {k: r4(v[i]) for k, v in d.items()}
                for i, s in enumerate(sigmas)
            },
            "per_sample_rel_spread_at_0.5": r4(
                np.std(np.array(per_sample[model])[:, np.abs(sigmas - 0.5).argmin()])
            ),
        }
        print(model, knees, flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["staircase.npz", "staircase.png", *grid_files],
        device=device,
        extra={
            "sigmas": [r4(s) for s in sigmas],
            "norm95_reference_sigma": r4(sigmas[i95]),
            "samples": [it.stem for it in items],
            "sizes": [f"{it.width}x{it.height}" for it in items],
            "text_tokens": [it.n_text for it in items],
            "anima_reference_norm95": {str(k): v for k, v in ANIMA_NORM95.items()},
            "anima_sampler_steps_above": {
                str(s): int((a_grid > s).sum()) for s in (0.75, 0.55, 0.45)
            },
            "anima_sampler": {"steps": ANIMA_STEPS, "shift": ANIMA_SHIFT},
            "moss_sampler_steps_above": {
                str(s): int((sigmas > s).sum()) for s in (0.75, 0.55, 0.45)
            },
            "training_mass_above": {
                str(s): r4((draws > s).mean()) for s in (0.85, 0.75, 0.55, 0.45)
            },
        },
    )
    print(f"wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
