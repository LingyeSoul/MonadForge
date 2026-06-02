#!/usr/bin/env python3
"""Mangafy + cache condition latents for the colorization EasyControl task.

Two idempotent stages:

1. **Mangafy** — walk every color image under ``--src`` (the existing resized
   training images), screen each to a synthetic B&W manga page, and write the
   result to ``--staging`` mirroring the source subpath. Two engines via
   ``--engine``: ``sd`` (default, Phase B) runs the learned sketch2manga screening
   (:func:`screentone_sd.screentone_array`, on-manifold tones, needs
   ``make exp-easycontrol-download EASYADAPTER=colorize``); ``cv2`` runs the fast
   XDoG + algorithmic-screentone fallback (:func:`mangafy.mangafy_array`, no
   downloads). Skips already-staged PNGs unless ``--overwrite``.

2. **Encode** — VAE-encode the staged manga images into ``--cond_cache_dir`` via
   the existing ``library.preprocess.cache_latents`` (same ``{stem}_{WxH}_anima.npz``
   format as the target latent cache), at the image's **native size** so each
   cond latent shape matches its target latent exactly. Skips cached resolutions.

The cond cache is paired with the color target at train time by the loader's
``cond_cache_dir`` subset knob (stem-matched). Run from the repo root::

    python easycontrol_adapters/colorization/prep.py            # full dataset
    python easycontrol_adapters/colorization/prep.py --limit 8  # quick QA batch
    make exp-easycontrol-preprocess EASYADAPTER=colorize        # via task runner
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

from typing import Callable

import numpy as np
import torch
from PIL import Image

from library.preprocess import cache_latents, tqdm_progress
from library.preprocess._dataset import walk_images

# Screener: color RGB uint8 (H,W,3) + per-stem seed → B&W RGB uint8 (H,W,3) same size.
Screener = Callable[[np.ndarray, int], np.ndarray]


def _stable_seed(name: str) -> int:
    """Process-stable per-stem seed (Python's hash() is salted; crc32 isn't)."""
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def stage_mangafy(
    src: Path,
    staging: Path,
    *,
    screener: Screener,
    recursive: bool,
    overwrite: bool,
    limit: int | None,
) -> tuple[int, int]:
    images = walk_images(src, recursive=recursive)
    if limit:
        images = images[:limit]
    written = 0
    progress = tqdm_progress("Mangafy")
    progress(0, total=len(images))
    for p in images:
        rel = p.relative_to(src)
        out = (staging / rel).with_suffix(".png")
        if out.exists() and not overwrite:
            progress(1, detail=f"skip {p.name}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        img = np.array(Image.open(p).convert("RGB"))
        manga = screener(img, _stable_seed(rel.stem))
        Image.fromarray(manga).save(out)
        written += 1
        progress(1, detail=p.name)
    return len(images), written


def build_screener(args) -> Screener:
    """Resolve ``--engine`` to a ``(img, seed) → manga`` callable.

    Imports are deferred so the heavy ``sd`` stack (diffusers / controlnet_aux)
    is only loaded when actually selected, and ``cv2`` stays download-free."""
    if args.engine == "cv2":
        from mangafy import mangafy_array

        return lambda img, seed: mangafy_array(img, seed=seed)

    from screentone_sd import screentone_array

    return lambda img, seed: screentone_array(
        img,
        seed=seed,
        steps=args.steps,
        cfg=args.cfg,
        long_side=args.long_side,
        strength=args.strength,
        tone_period=args.tone_period,
    )


def stage_encode(
    staging: Path,
    cond_cache_dir: Path,
    *,
    vae_path: str,
    batch_size: int,
    chunk_size: int,
    recursive: bool,
):
    from library.models import qwen_vae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading VAE from {vae_path} ...")
    vae = qwen_vae.load_vae(
        vae_path,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=chunk_size,
        disable_cache=True,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()

    stats = cache_latents(
        staging,
        vae,
        cache_dir=cond_cache_dir,
        recursive=recursive,
        batch_size=batch_size,
        progress=tqdm_progress("Caching cond latents"),
    )

    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="post_image_dataset/resized")
    parser.add_argument("--staging", default="post_image_dataset/colorize_staging")
    parser.add_argument("--cond_cache_dir", default="post_image_dataset/colorize_cond")
    parser.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    parser.add_argument(
        "--engine",
        choices=["sd", "cv2"],
        default="sd",
        help="condition synthesizer: sd = learned sketch2manga screening (Phase B, "
        "needs `make exp-easycontrol-download EASYADAPTER=colorize`); "
        "cv2 = XDoG+halftone fallback (no downloads)",
    )
    parser.add_argument("--steps", type=int, default=40, help="sd engine: diffusion steps")
    parser.add_argument("--cfg", type=float, default=9.0, help="sd engine: guidance scale")
    parser.add_argument(
        "--strength",
        type=float,
        default=0.6,
        help="sd engine: img2img denoise (lower = more faithful to source structure)",
    )
    parser.add_argument(
        "--tone_period",
        type=float,
        default=4.5,
        help="sd engine: halftone dot period (px at long_side); larger = coarser dots",
    )
    parser.add_argument(
        "--long_side", type=int, default=1024, help="sd engine: screening resolution"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="re-mangafy staged PNGs that exist"
    )
    parser.add_argument("--limit", type=int, default=None, help="cap #images (QA)")
    parser.add_argument("--skip_mangafy", action="store_true")
    parser.add_argument("--skip_encode", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    staging = Path(args.staging)
    cond_cache_dir = Path(args.cond_cache_dir)

    if not args.skip_mangafy:
        seen, written = stage_mangafy(
            src,
            staging,
            screener=build_screener(args),
            recursive=args.recursive,
            overwrite=args.overwrite,
            limit=args.limit,
        )
        print(
            f"\nMangafy ({args.engine}): {written} written, "
            f"{seen - written} skipped/seen={seen}"
        )
        if args.engine == "sd":
            # Free the SD pipeline's VRAM before the encode stage loads the Qwen VAE.
            from screentone_sd import unload

            unload()

    if not args.skip_encode:
        stats = stage_encode(
            staging,
            cond_cache_dir,
            vae_path=args.vae,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            recursive=args.recursive,
        )
        print(
            f"\nCond latent caching complete: {stats.written} cached, "
            f"{stats.skipped} skipped (already existed)"
        )


if __name__ == "__main__":
    main()
