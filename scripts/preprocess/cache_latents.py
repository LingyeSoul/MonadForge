#!/usr/bin/env python3
"""Cache VAE latents for all images in a dataset directory.

Encodes images through the Qwen Image VAE and saves latent caches (.npz)
alongside the images (or under ``--cache_dir``).  Skips already-cached
entries (idempotent).

The walk → group-by-resolution → encode → save loop lives in
``library/preprocess/latents.py``; this file is argparse + VAE load + reporting.
"""

import argparse
from pathlib import Path

import torch


from library.preprocess import cache_latents, count_pending_latents, tqdm_progress
from library.runtime.argparse_groups import add_io_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_args(
        parser,
        cache_noun="latent caches",
        include_batch_size=True,
        batch_size_default=2,
    )
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE weights")
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
    # 2D VAE fold is ON by default: image-only pipeline, ~2x faster encode at
    # ~0.65-0.7x peak VRAM, latents equivalent within bf16 noise. See
    # _archive/bench/qwen_vae_2d/. Opt out with --no_vae_2d for the stock 3D causal VAE.
    parser.add_argument(
        "--qwen_image_vae_2d",
        "--vae_2d",
        dest="vae_2d",
        action="store_true",
        default=True,
        help="Fold the causal Conv3d VAE into 2D convs (image-only). Default ON.",
    )
    parser.add_argument(
        "--no_vae_2d",
        "--qwen_image_vae_3d",
        dest="vae_2d",
        action="store_false",
        help="Use the stock 3D causal-Conv3d VAE instead of the 2D fold.",
    )
    # Encode in fp32 instead of bf16. The fp32 save already happens regardless;
    # this also encodes in fp32, removing the (structured but ~17 dB-below-recon)
    # bf16 accumulation error and making the 2D fold bit-exact. See
    # _archive/bench/qwen_vae_2d/encode_dtype_probe.py — quality-neutral, hygiene only.
    parser.add_argument(
        "--no_half_vae",
        "--fp32_vae",
        dest="no_half_vae",
        action="store_true",
        help="Encode latents in fp32 (default: bf16). Bit-exact 2D fold; slower.",
    )
    parser.add_argument(
        "--path_pattern",
        "--path-pattern",
        dest="path_pattern",
        default="*",
        help=(
            "Only cache images whose path relative to --dir matches this "
            "fnmatch glob. Use | to separate alternatives. Default: *"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-encode every latent even if its (W,H) cache already exists. Use "
            "after a VAE / encode-dtype change, which the per-resolution skip "
            "can't detect (existing caches are otherwise skipped)."
        ),
    )
    args = parser.parse_args()

    from library.models import qwen_vae as qwen_image_autoencoder_kl

    data_dir = Path(args.dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # Pre-flight: a fully-cached dataset needs no VAE — skip the (slow) load.
    pending, total = count_pending_latents(
        data_dir,
        cache_dir=cache_dir,
        recursive=args.recursive,
        path_pattern=args.path_pattern,
        overwrite=args.overwrite,
    )
    if pending == 0:
        print(f"Latent caching: all {total} images already cached — skipping VAE load.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.no_half_vae else torch.bfloat16

    print(f"{pending}/{total} images need latents.")
    print(f"Loading VAE from {args.vae} (encode dtype: {dtype}) ...")
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.chunk_size,
        disable_cache=args.disable_cache,
    )
    vae.to(device, dtype=dtype)
    if args.vae_2d:
        n = vae.convert_to_2d()
        print(f"Folded VAE to 2D (image-only): {n} Conv3d -> Conv2d")
    vae.requires_grad_(False)
    vae.eval()

    stats = cache_latents(
        data_dir,
        vae,
        cache_dir=cache_dir,
        recursive=args.recursive,
        path_pattern=args.path_pattern,
        batch_size=args.batch_size,
        progress=tqdm_progress("Caching latents"),
        overwrite=args.overwrite,
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
