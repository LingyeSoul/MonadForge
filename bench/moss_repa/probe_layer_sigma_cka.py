#!/usr/bin/env python3
"""REPA layer × σ probe on MOSS-Image — which block holds a PE-Spatial-alignable
representation, before any training.

Port of Anima's ``_archive/bench/repa/probe_layer_sigma_cka.py`` (findings in
``_archive/bench/repa/results.md``). Same ruler and the same three confound
controls:

  * **centered linear CKA** of a block's image tokens, pooled to the PE grid, vs
    the image's own PE-Spatial patch tokens;
  * **mismatched control** — the same tokens vs a *different* image's PE on the
    same grid; ``gap = matched − mismatched`` is the content-specific part;
  * **σ = 1 floor** — ``useful_gap = gap − gap(σ=1)`` drops what a block rebuilds
    from the caption alone (pure noise in);
  * **DoG target** (``σ₁ = min(gh, gw)/16``) beside the raw target — strips the
    broad low band.

Plus ``dog_gram``: the relational DoG-Gram loss the training term would
optimise, on the base model (``library/training/repa.py``). It is the headroom —
near 0 means the block is already aligned and REPA has little to add there.

MOSS specifics: the block output is ``[image (H·W) | text (S)]``; ``layer_hook``
gets ``n_img`` and only the image rows are scored. Latents are ``(128, H/16,
W/16)``, so the token grid is the latent grid. PE targets are the Anima sidecars
(``post_image_dataset/lora/<set>/{stem}_anima_pe_spatial.safetensors``) of the
resized PNG the MOSS latent was encoded from, used only when the latent was not
resampled (``source_size`` == cache size).

    make daemon-run ARGS="bench/moss_repa/probe_layer_sigma_cka.py --label base"

Writes ``cka.npz`` (per-image values), ``table.csv`` (layer × σ means),
``heatmap.png`` and ``result.json`` under ``bench/moss_repa/results/<stamp>/``.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.moss import cache as moss_cache  # noqa: E402
from library.moss.dave import massive_channels  # noqa: E402
from library.moss.loader import empty_cache, load_transformer  # noqa: E402
from library.moss.repa import ENCODER, pe_sidecar  # noqa: E402
from library.runtime.blockswap import place  # noqa: E402
from library.training.repa import (  # noqa: E402
    dog_standardize,
    pool_dit_tokens_to_grid,
    relational_gram_loss,
    resolve_pe_grid,
)
from library.vision.buckets import get_bucket_spec  # noqa: E402

DEFAULT_CACHES = (
    "post_image_dataset/moss/channel_(caststation)_layout",
    "post_image_dataset/moss/_calib_mixed",
    "post_image_dataset/moss/sincos_layout",
    "post_image_dataset/moss/hews_layout",
    "post_image_dataset/moss/ama_mitsuki_layout",
    "post_image_dataset/moss/group_wsw_layout",
)
DEFAULT_SIGMAS = "0.1,0.2,0.3,0.45,0.6,0.7,0.8,0.85,0.9,0.95,1.0"
METRICS = (
    "raw_m",
    "raw_x",
    "dog_m",
    "dog_x",
    "dog_gram",
    "dog_gram_c",
    "dog_gram_m",
    "dog_gram_cm",
    "dit_cos",
)


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """Centered linear CKA between ``x (N, p)`` and ``y (N, q)``, Gram form."""
    x = x.float() - x.float().mean(0, keepdim=True)
    y = y.float() - y.float().mean(0, keepdim=True)
    k = x @ x.T
    ell = y @ y.T
    return float((k * ell).sum() / (k.norm() * ell.norm() + eps))


def collect(caches, pe_root, n, keep_first, seed):
    """``n`` (item, sidecar) pairs: every one of the first cache's ``keep_first``,
    the rest drawn at random from the others, deduplicated by source PNG."""
    rng = random.Random(seed)
    first, rest, seen = [], [], set()
    for ci, cache in enumerate(caches):
        for item in moss_cache.list_items(resolve_under_home(cache)):
            pe = pe_sidecar(
                item.latent_path, item.stem, item.width, item.height, pe_root
            )
            if pe is None or pe in seen:
                continue
            seen.add(pe)
            (first if ci == 0 else rest).append((item, pe))
    first = first[:keep_first] if keep_first >= 0 else first
    rng.shuffle(rest)
    return first + rest[: max(0, n - len(first))]


def plot(path, table, layers, sigmas, band_cols):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, key, title in zip(
        axes,
        ("useful_raw", "useful_dog", "dog_gram"),
        (
            "useful_gap, raw target",
            "useful_gap, DoG target",
            "DoG-Gram loss (headroom)",
        ),
    ):
        im = ax.imshow(table[key].T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("block")
        ax.set_ylabel("σ")
        ax.set_yticks(range(len(sigmas)), [f"{s:g}" for s in sigmas])
        ax.set_xticks(
            range(0, len(layers), 3), [str(layers[i]) for i in range(0, len(layers), 3)]
        )
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for key in ("useful_raw", "useful_dog"):
        ax.plot(layers, table[key][:, band_cols].mean(1), label=f"{key} (band mean)")
    ax2 = ax.twinx()
    ax2.plot(layers, table["dog_gram"][:, band_cols].mean(1), "k--", label="dog_gram")
    ax.set_xlabel("block")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path.with_name("band.png"), dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--caches", default=",".join(DEFAULT_CACHES))
    ap.add_argument("--pe_root", default="post_image_dataset/lora")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument(
        "--keep_first", type=int, default=-1, help="cap on the first cache (-1 = all)"
    )
    ap.add_argument("--sigmas", default=DEFAULT_SIGMAS)
    ap.add_argument("--band_lo", type=float, default=0.45)
    ap.add_argument("--band_hi", type=float, default=0.9)
    ap.add_argument("--dog_sigma1_div", type=float, default=16.0)
    ap.add_argument("--model", default="base", help="base | sft")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument(
        "--blocks_to_swap", type=int, default=-1, help="-1 = size from free VRAM"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    spec = get_bucket_spec(ENCODER)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    if sigmas[-1] != 1.0:
        raise SystemExit("--sigmas must end at 1.0 (the floor)")
    band_cols = [j for j, s in enumerate(sigmas) if args.band_lo <= s <= args.band_hi]
    caches = [c for c in args.caches.split(",") if c]
    pairs = collect(
        caches,
        resolve_under_home(args.pe_root),
        args.n,
        args.keep_first,
        args.seed,
    )
    if len(pairs) < 2:
        raise SystemExit("fewer than 2 images with a PE sidecar")

    # PE targets on CPU, grouped by grid for the mismatched partner.
    store = []
    for item, pe_path in pairs:
        pe = load_file(str(pe_path))["image_features"].float()[None]
        n_pe = pe.shape[1] - (1 if spec.use_cls else 0)
        h, w = item.height // 16, item.width // 16
        gh, gw = resolve_pe_grid(spec, n_pe, h, w)
        raw = pe[:, 1:] if spec.use_cls else pe
        dog = dog_standardize(raw, gh, gw, args.dog_sigma1_div)
        store.append(
            {"item": item, "hw": (h, w), "grid": (gh, gw), "raw": raw, "dog": dog}
        )
    rng = np.random.default_rng(args.seed)
    by_grid: dict[tuple, list[int]] = {}
    for i, rec in enumerate(store):
        by_grid.setdefault(rec["grid"], []).append(i)
    for group in by_grid.values():
        perm = rng.permutation(len(group))
        for k, i in enumerate(group):
            j = group[perm[k]]
            if j == i:
                j = group[(perm[k] + 1) % len(group)]
            store[i]["partner"] = j if j != i else None
    print(
        f"{len(store)} images over {len(by_grid)} PE grids "
        f"({sum(r['partner'] is None for r in store)} without a same-grid partner)",
        flush=True,
    )

    transformer = load_transformer(args.model_dir, args.model)
    transformer.requires_grad_(False).eval()
    num_blocks = len(transformer.layers)
    attached = place(
        transformer,
        "layers",
        device,
        blocks_to_swap=None if args.blocks_to_swap < 0 else args.blocks_to_swap,
        activation_reserve_gb=3.0,
        label="transformer",
    )
    empty_cache()
    start_heartbeat()

    # values[metric][image, layer, σ]; NaN where a mismatched partner is missing.
    values = {
        m: np.full((len(store), num_blocks, len(sigmas)), np.nan) for m in METRICS
    }
    cur: dict = {}
    massive = set(massive_channels().tolist())
    keep = torch.tensor(
        [c for c in range(transformer.config.dim) if c not in massive], device=device
    )

    def hook(index, x, n_img):
        rec = cur["rec"]
        gh, gw = rec["grid"]
        tok = pool_dit_tokens_to_grid(x[:, :n_img], rec["hw"], 1, gh, gw)
        i, j = cur["i"], cur["j"]
        values["raw_m"][i, index, j] = linear_cka(tok[0], cur["raw"][0])
        values["dog_m"][i, index, j] = linear_cka(tok[0], cur["dog"][0])
        values["dog_gram"][i, index, j] = float(relational_gram_loss(tok, cur["dog"]))
        # The massive channels make every token point one way; center over tokens
        # and/or drop them to see the Gram underneath.
        tok_c = tok - tok.mean(1, keepdim=True)
        tok_m = tok[..., keep]
        tok_cm = tok_m - tok_m.mean(1, keepdim=True)
        values["dog_gram_c"][i, index, j] = float(
            relational_gram_loss(tok_c, cur["dog"])
        )
        values["dog_gram_m"][i, index, j] = float(
            relational_gram_loss(tok_m, cur["dog"])
        )
        values["dog_gram_cm"][i, index, j] = float(
            relational_gram_loss(tok_cm, cur["dog"])
        )
        hat = torch.nn.functional.normalize(tok[0], dim=-1)
        values["dit_cos"][i, index, j] = float((hat @ hat.T).mean())
        if cur["raw_x"] is not None:
            values["raw_x"][i, index, j] = linear_cka(tok[0], cur["raw_x"][0])
            values["dog_x"][i, index, j] = linear_cka(tok[0], cur["dog_x"][0])
        return x

    transformer.layer_hook = hook
    gen = torch.Generator(device=device)
    t0 = time.time()
    done, total = 0, len(store) * len(sigmas)
    with torch.no_grad():
        for i, rec in enumerate(store):
            latent, embeds = moss_cache.load_item(rec["item"])
            latent = latent[None].to(device, torch.float32)
            embeds = embeds[None].to(device)
            partner = store[rec["partner"]] if rec["partner"] is not None else None
            cur.update(
                rec=rec,
                i=i,
                raw=rec["raw"].to(device),
                dog=rec["dog"].to(device),
                raw_x=None if partner is None else partner["raw"].to(device),
                dog_x=None if partner is None else partner["dog"].to(device),
            )
            for j, sigma in enumerate(sigmas):
                cur["j"] = j
                gen.manual_seed(args.seed * 100003 + i)  # same ε across σ per image
                noise = torch.randn(latent.shape, generator=gen, device=device)
                noisy = (1.0 - sigma) * latent + sigma * noise
                timestep = torch.tensor([1000.0 * sigma], device=device)
                transformer(noisy, timestep, embeds, return_dict=False)
                done += 1
                if done % 4 == 0 or done == total:
                    print(
                        f"  progress {done}/{total}  {(time.time() - t0) / done:.2f}s/fwd",
                        flush=True,
                    )
    transformer.layer_hook = None
    if attached is not None:
        attached.detach()

    # Aggregate: means over images (mismatched ones over images with a partner).
    mean = {m: np.nanmean(values[m], axis=0) for m in METRICS}  # (layers, σ)
    table = {
        k: mean[k]
        for k in ("dog_gram", "dog_gram_c", "dog_gram_m", "dog_gram_cm", "dit_cos")
    }
    for t in ("raw", "dog"):
        gap = values[f"{t}_m"] - values[f"{t}_x"]  # per image
        useful = gap - gap[:, :, -1:]  # minus its own σ=1 floor
        table[f"cka_{t}"] = mean[f"{t}_m"]
        table[f"gap_{t}"] = np.nanmean(gap, axis=0)
        table[f"useful_{t}"] = np.nanmean(useful, axis=0)
        table[f"useful_{t}_se"] = np.nanstd(useful, axis=0) / np.sqrt(
            np.sum(~np.isnan(useful), axis=0).clip(min=1)
        )

    run_dir = make_run_dir("moss_repa", args.label)
    np.savez(
        run_dir / "cka.npz",
        sigmas=np.array(sigmas),
        stems=np.array([r["item"].stem for r in store]),
        **values,
        **{f"table_{k}": v for k, v in table.items()},
    )
    keys = (
        "cka_raw",
        "gap_raw",
        "useful_raw",
        "cka_dog",
        "gap_dog",
        "useful_dog",
        "dog_gram",
        "dog_gram_c",
        "dog_gram_m",
        "dog_gram_cm",
        "dit_cos",
    )
    with open(run_dir / "table.csv", "w") as f:
        f.write("layer,sigma," + ",".join(keys) + "\n")
        for ell in range(num_blocks):
            for j, s in enumerate(sigmas):
                f.write(
                    f"{ell},{s:g},"
                    + ",".join(f"{table[k][ell, j]:.5f}" for k in keys)
                    + "\n"
                )
    layers = list(range(num_blocks))
    plot(run_dir / "heatmap.png", table, layers, sigmas, band_cols)

    band = {
        k: table[k][:, band_cols].mean(1)
        for k in (
            "useful_raw",
            "useful_dog",
            "dog_gram",
            "dog_gram_c",
            "dog_gram_cm",
            "dit_cos",
        )
    }
    band_se = {
        k: table[f"{k}_se"][:, band_cols].mean(1) for k in ("useful_raw", "useful_dog")
    }
    print(f"\nband σ∈[{args.band_lo}, {args.band_hi}] means over {len(store)} images")
    print(
        "block  useful_raw        useful_dog        dog_gram  gram_c  gram_cm  dit_cos"
    )
    for ell in layers:
        print(
            f"{ell:5d}  {band['useful_raw'][ell]:.4f}±{band_se['useful_raw'][ell]:.4f}"
            f"   {band['useful_dog'][ell]:.4f}±{band_se['useful_dog'][ell]:.4f}"
            f"   {band['dog_gram'][ell]:.4f}  {band['dog_gram_c'][ell]:.4f}"
            f"  {band['dog_gram_cm'][ell]:.4f}  {band['dit_cos'][ell]:.3f}",
            flush=True,
        )
    ridge = {
        t: [int(np.argmax(table[f"useful_{t}"][:, j])) for j in range(len(sigmas) - 1)]
        for t in ("raw", "dog")
    }
    print(f"argmax block per σ (raw): {dict(zip(sigmas[:-1], ridge['raw']))}")
    print(f"argmax block per σ (dog): {dict(zip(sigmas[:-1], ridge['dog']))}")

    metrics = {
        "n_images": len(store),
        "n_grids": len(by_grid),
        "sigmas": sigmas,
        "band": [args.band_lo, args.band_hi],
        "band_useful_raw": band["useful_raw"].round(5).tolist(),
        "band_useful_dog": band["useful_dog"].round(5).tolist(),
        "band_dog_gram": band["dog_gram"].round(5).tolist(),
        "band_dog_gram_c": band["dog_gram_c"].round(5).tolist(),
        "band_dog_gram_cm": band["dog_gram_cm"].round(5).tolist(),
        "band_dit_cos": band["dit_cos"].round(5).tolist(),
        "argmax_band_useful_raw": int(np.argmax(band["useful_raw"])),
        "argmax_band_useful_dog": int(np.argmax(band["useful_dog"])),
        "ridge_raw": ridge["raw"],
        "ridge_dog": ridge["dog"],
        "sigma_profile_dog": table["useful_dog"].max(0).round(5).tolist(),
        "seconds": time.time() - t0,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["cka.npz", "table.csv", "heatmap.png", "band.png"],
        device=device,
        extra={"stems": [r["item"].stem for r in store]},
    )
    print(f"wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
