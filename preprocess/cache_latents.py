#!/usr/bin/env python3
"""Cache VAE latents for all images in a dataset directory.

Encodes images through the Qwen Image VAE and saves latent caches (.npz)
alongside the images (or under ``--cache_dir``).  Skips already-cached
entries (idempotent).

The walk → group-by-resolution → encode → save loop lives in
``library/preprocess/latents.py``; this file is argparse + VAE load + reporting.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.preprocess import cache_latents, tqdm_progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=str, required=True, help="Dataset directory")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Optional directory to write latent caches into (created if needed). "
            "Defaults to writing alongside each source image."
        ),
    )
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE weights")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="VAE encoding batch size (default: 4)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="VAE spatial chunk size (default: 64)",
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        default=True,
        help="Disable VAE internal cache (default: True)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Walk subfolders under --dir. Caches mirror the source subdir "
            "structure under --cache_dir; stems must be unique within each "
            "subfolder but the same stem can repeat across folders."
        ),
    )
    args = parser.parse_args()

    from library.models import qwen_vae as qwen_image_autoencoder_kl

    data_dir = Path(args.dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading VAE from {args.vae} ...")
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.chunk_size,
        disable_cache=args.disable_cache,
    )
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()

    stats = cache_latents(
        data_dir,
        vae,
        cache_dir=cache_dir,
        recursive=args.recursive,
        batch_size=args.batch_size,
        progress=tqdm_progress("Caching latents"),
    )
    print(
        f"\nLatent caching complete: {stats.written} cached, "
        f"{stats.skipped} skipped (already existed)"
    )

    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
