#!/usr/bin/env python3
"""Minimal text-to-image generation with the Anima base model (no adapter).

The programmatic equivalent of:

    python inference.py --dit <…> --vae <…> --text_encoder <…> \
        --prompt "…" --save_path output/tests/example_01.png

Run from the repo root (anima_lora/) after `make download-models`:

    python examples/01_generate.py --prompt "a red fox in a snowy forest"

The three steps any embedder needs — settings → generate → decode — are
spelled out in main().
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python examples/<script>.py`: put the repo root on sys.path so
# `import library` / `inference` resolve. Model/config paths resolve against the
# repo home regardless of CWD (set ANIMA_HOME for a relocated checkout); only the
# output paths below are written relative to your CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

# The curated entry points live on the top-level `anima_lora` package — the
# programmatic front door (a thin lazy re-export of the `library.*` homes).
# `GenerationRequest` is the typed constructor for a single generation; its
# `.to_args()` feeds the request through `inference.parse_args` under the hood,
# so every optional knob the generation code reads via getattr() still gets a
# default (building a bare Namespace by hand silently drops dozens of them).
from anima_lora import (
    GenerationRequest,
    generate,
    get_generation_settings,
    load_vae,
    save_output,
)
from library.runtime.device import clean_memory_on_device

# Default checkpoint locations (from configs/base.toml). Override via env if
# your weights live elsewhere.
DIT = os.environ.get("ANIMA_DIT", "models/diffusion_models/anima-base-v1.0.safetensors")
VAE = os.environ.get("ANIMA_VAE", "models/vae/qwen_image_vae.safetensors")
TEXT_ENCODER = os.environ.get(
    "ANIMA_TEXT_ENCODER", "models/text_encoders/qwen_3_06b_base.safetensors"
)


def build_request(
    prompt: str,
    save_path: str,
    *,
    steps: int,
    cfg: float,
    size: tuple[int, int],
    seed: int,
) -> GenerationRequest:
    """Describe the generation as a typed request (the CLI is one consumer)."""
    return GenerationRequest(
        dit=DIT,
        vae=VAE,
        text_encoder=TEXT_ENCODER,
        prompt=prompt,
        save_path=save_path,
        infer_steps=steps,
        guidance_scale=cfg,
        image_size=size,  # (H, W)
        seed=seed,
    )


def generate_image(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # 1. Settings carry the device + DiT weight dtype (bf16).
    gen_settings = get_generation_settings(args)

    # 2. generate() lazily loads the DiT, encodes the prompt (max-padded — the
    #    pretrained model treats padding as attention sinks; trimming gives
    #    black images), and runs the sampler. Returns the clean latent.
    latent = generate(args, gen_settings)

    # 3. Free the DiT before bringing up the VAE (the lazy-load discipline that
    #    keeps peak VRAM down).
    clean_memory_on_device(device)

    # 4. Decode latent → PNG (+ generation metadata) via save_output.
    vae = load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
        dtype=torch.bfloat16,  # load_vae handles the bf16 cast + eval() for you
        eval=True,
    )
    save_output(args, vae, latent, device)
    print(f"saved → {args.save_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompt", default="a red fox sitting in a snowy forest, golden hour"
    )
    p.add_argument("--save_path", default="output/tests/example_01.png")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--seed", type=int, default=42)
    opts = p.parse_args()

    req = build_request(
        opts.prompt,
        opts.save_path,
        steps=opts.steps,
        cfg=opts.cfg,
        size=tuple(opts.size),
        seed=opts.seed,
    )
    # .to_args() runs the request through the CLI parser, so the returned
    # Namespace has every optional knob populated for generate().
    generate_image(req.to_args())


if __name__ == "__main__":
    main()
