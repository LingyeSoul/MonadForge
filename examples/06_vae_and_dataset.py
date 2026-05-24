#!/usr/bin/env python3
"""Read the dataset cache and round-trip pixels through the VAE.

Two things a scripts/ author repeatedly needs:

  A. VAE round-trip — encode an image to a latent and decode it back. The clean
     helpers are encode_pixels_to_latents() / decode_to_pixels(), both expecting
     pixels in [-1, 1].

  B. Iterate the preprocessed cache that training actually consumes.
     CachedDataset (library/datasets/distill.py) yields
     (idx, latent, crossattn_emb, pooled_text) straight from the
     `{stem}_{WxH}_anima.npz` + `{stem}_anima_te.safetensors` files under
     post_image_dataset/lora/ — no DiT/encoder needed, it's all on disk.
     Samples are bucket-grouped so each batch is one resolution.

    python examples/06_vae_and_dataset.py --image some/photo.png      # part A
    python examples/06_vae_and_dataset.py --data_dir post_image_dataset/lora  # part B
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from library.datasets.distill import CachedDataset
from library.models import qwen_vae

VAE = os.environ.get("ANIMA_VAE", "models/vae/qwen_image_vae.safetensors")


def _load_vae(device):
    vae = qwen_vae.load_vae(VAE, device=device, disable_mmap=True)
    vae.to(torch.bfloat16).eval()
    return vae


def vae_roundtrip(image_path: str, out_path: str, device) -> None:
    """Part A — pixels → latent → pixels."""
    from PIL import Image
    from torchvision import transforms

    vae = _load_vae(device)

    img = Image.open(image_path).convert("RGB")
    to_tensor = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]  # → [-1, 1]
    )
    pixels = to_tensor(img).unsqueeze(0).to(device, dtype=torch.bfloat16)  # [1,3,H,W]

    with torch.no_grad():
        latent = vae.encode_pixels_to_latents(pixels)  # [1, 16, 1, H/8, W/8]
        recon = vae.decode_to_pixels(latent)  # [-1, 1]

    print(
        f"image {tuple(pixels.shape)} → latent {tuple(latent.shape)} → recon {tuple(recon.shape)}"
    )

    recon01 = (recon.squeeze(0).float().clamp(-1, 1) + 1) / 2  # → [0, 1]
    transforms.ToPILImage()(recon01.cpu()).save(out_path)
    print(f"saved reconstruction → {out_path}")


def iterate_cache(data_dir: str, device) -> None:
    """Part B — read the on-disk training cache and decode one cached latent."""
    dataset = CachedDataset(data_dir, batch_size=1)
    print(f"CachedDataset({data_dir}): {len(dataset.samples)} samples")
    if not dataset.samples:
        print("  (empty — run `make preprocess` first)")
        return

    idx, latent, crossattn_emb, pooled = dataset[0]
    print(
        f"  sample 0: latent={tuple(latent.shape)}  "
        f"crossattn_emb={tuple(crossattn_emb.shape)}  pooled={tuple(pooled.shape)}"
    )

    # Decode the cached latent back to an image to confirm it's the real thing.
    vae = _load_vae(device)
    with torch.no_grad():
        recon = vae.decode_to_pixels(
            latent.unsqueeze(0).to(device, dtype=torch.bfloat16)
        )
    recon01 = (recon.squeeze(0).float().clamp(-1, 1) + 1) / 2
    from torchvision import transforms

    out = "output/tests/example_06_cached_sample0.png"
    transforms.ToPILImage()(recon01.cpu()).save(out)
    print(f"  decoded cached latent → {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", help="run part A (VAE round-trip) on this image")
    p.add_argument(
        "--data_dir",
        default="post_image_dataset/lora",
        help="run part B: iterate this cache dir",
    )
    p.add_argument("--out", default="output/tests/example_06_roundtrip.png")
    opts = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opts.image:
        vae_roundtrip(opts.image, opts.out, device)
    else:
        iterate_cache(opts.data_dir, device)


if __name__ == "__main__":
    main()
