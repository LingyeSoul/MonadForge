#!/usr/bin/env python3
"""Generate with one or more trained LoRA adapters attached.

Same flow as 01_generate.py, plus `--lora_weight`. The adapter is applied
*inside* generate() → load_dit_model(): when args.lora_weight is set the DiT
loader instantiates the network from the checkpoint and either merges it
(plain LoRA / OrthoLoRA / T-LoRA) or keeps it live (HydraLoRA / FeRA), driven
entirely by the checkpoint's own metadata — the embedder doesn't pick the
adapter family, the .safetensors does.

Run from the repo root (anima_lora/):

    python examples/02_generate_with_lora.py \
        --lora_weight output/ckpt/my_lora.safetensors \
        --prompt "a portrait of <subject>"

Pass --lora_weight multiple times (with matching --multiplier) to stack
adapters.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python examples/<script>.py`: put the repo root on sys.path so
# `import library` / `inference` resolve. Still run from the repo root —
# configs/ and models/ paths are resolved relative to the CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import inference
from library.inference import generate, get_generation_settings, save_output
from library.models import qwen_vae
from library.runtime.device import clean_memory_on_device

DIT = os.environ.get("ANIMA_DIT", "models/diffusion_models/anima-base-v1.0.safetensors")
VAE = os.environ.get("ANIMA_VAE", "models/vae/qwen_image_vae.safetensors")
TEXT_ENCODER = os.environ.get(
    "ANIMA_TEXT_ENCODER", "models/text_encoders/qwen_3_06b_base.safetensors"
)


def build_inference_args(opts: argparse.Namespace) -> argparse.Namespace:
    argv = [
        "--dit",
        DIT,
        "--vae",
        VAE,
        "--text_encoder",
        TEXT_ENCODER,
        "--prompt",
        opts.prompt,
        "--save_path",
        opts.save_path,
        "--infer_steps",
        str(opts.steps),
        "--guidance_scale",
        str(opts.cfg),
        "--image_size",
        str(opts.size[0]),
        str(opts.size[1]),
        "--seed",
        str(opts.seed),
        # --lora_weight / --lora_multiplier take nargs="*"; forward each path.
        "--lora_weight",
        *opts.lora_weight,
        "--lora_multiplier",
        *[str(m) for m in opts.multiplier],
    ]
    return inference.parse_args(argv)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lora_weight",
        nargs="+",
        required=True,
        help="one or more adapter .safetensors",
    )
    p.add_argument("--multiplier", type=float, nargs="+", default=[1.0])
    p.add_argument(
        "--prompt", default="a red fox sitting in a snowy forest, golden hour"
    )
    p.add_argument("--save_path", default="output/tests/example_02.png")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--seed", type=int, default=42)
    opts = p.parse_args()

    # Broadcast a single multiplier across all adapters.
    if len(opts.multiplier) == 1 and len(opts.lora_weight) > 1:
        opts.multiplier = opts.multiplier * len(opts.lora_weight)

    args = build_inference_args(opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    gen_settings = get_generation_settings(args)
    latent = generate(args, gen_settings)  # adapter attached during DiT load
    clean_memory_on_device(device)

    vae = qwen_vae.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
    )
    vae.to(torch.bfloat16)
    vae.eval()
    save_output(args, vae, latent, device)
    print(f"saved → {args.save_path}  (adapters: {', '.join(opts.lora_weight)})")


if __name__ == "__main__":
    main()
