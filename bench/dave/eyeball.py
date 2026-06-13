#!/usr/bin/env python
"""DAVE eyeball sweep — render baseline vs DAVE settings across seeds.

Not a metrics bench: just generates the `make test` prompt under a few DAVE
configs (the model is loaded once and shared) and dumps labeled PNGs into
``output/tests/dave/`` so the diversity/quality tradeoff can be eyeballed.

    uv run python bench/dave/eyeball.py
    uv run python bench/dave/eyeball.py --seeds 1000 1001 1002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima_lora import (  # noqa: E402
    GenerationRequest,
    decode_to_pil,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_dit_model,
    load_vae,
    prepare_text_inputs,
)

from bench._anima import DEFAULT_NEG as DEFAULT_NEGATIVE  # noqa: E402
from bench._anima import DEFAULT_PROMPT  # noqa: E402

# (label, dave-on, strength, block_lo, block_hi, tau). block_hi=-1 → last block.
# tau>0 = DAVE paper's early-step cutoff (overrides the σ window); 0 = full schedule.
# The shipped mask is now the flat statistical pool (blocks 8–18, final-stage 19–27
# baked to 0), so block_lo/hi=-1 already excludes the dot blocks. These configs
# validate the production defaults and probe the headroom the cap opened up.
CONFIGS = [
    # Tighter early-window (first ~10% of steps only), flat pool 8–18. Gentle vs hot
    # dose: does τ0.10 hold diversity with less text/hand cost, and does s0.80 stay
    # coherent once the window is this tight? (baseline/default already characterized.)
    ("tau0.10_s0.30", True, 0.30, 0, -1, 0.10),
    ("tau0.10_s0.80", True, 0.80, 0, -1, 0.10),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001])
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE)
    opts = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).resolve().parents[2] / "output" / "tests" / "dave"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = default_checkpoints()
    req = GenerationRequest(
        dit=ckpts.dit,
        vae=ckpts.vae,
        text_encoder=ckpts.text_encoder,
        prompt=opts.prompt,
        negative_prompt=opts.negative_prompt,
        save_path="output/tests/dave/_unused.png",
        infer_steps=opts.steps,
        guidance_scale=opts.cfg,
        image_size=(1024, 1024),
        sampler="euler",
        flow_shift=3.0,
        seed=opts.seeds[0],
    )
    args = req.to_args()
    args.device = device
    args.compile = False
    args.compile_blocks = False

    gen_settings = get_generation_settings(args)
    print("[eyeball] loading DiT + VAE…")
    anima = load_dit_model(args, device, torch.bfloat16)
    # VAE stays on CPU; decode_latent moves it on/off device per decode, so the
    # shared DiT and the VAE never both sit on the GPU.
    vae = load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
        dtype=torch.bfloat16,
        eval=True,
    )
    context, context_null = prepare_text_inputs(args, device, anima)
    text_data = {"context": context, "context_null": context_null}
    shared = {"model": anima}

    for seed in opts.seeds:
        for label, on, strength, blo, bhi, tau in CONFIGS:
            args.seed = seed
            args.dave = "auto" if on else None
            args.dave_strength = strength
            args.dave_block_lo = blo
            args.dave_block_hi = bhi
            args.dave_sigma_lo = 0.0
            args.dave_sigma_hi = 1.0
            args.dave_tau = tau  # >0 → setup_dave converts to the early-σ window
            print(f"[eyeball] seed={seed} {label}")
            latent = generate(
                args,
                gen_settings,
                shared_models=shared,
                precomputed_text_data=text_data,
            )
            img = decode_to_pil(vae, latent, device)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"seed{seed}_{label}.png".replace(".", "p", label.count("."))
            img.save(out_dir / fname)

    print(f"\n→ {out_dir}")


if __name__ == "__main__":
    main()
