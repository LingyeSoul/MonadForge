#!/usr/bin/env python3
"""Generate with a training-free sampler correction (DCW or Spectrum).

Same flow as 01_generate.py, but this one shows the **escape hatch** for the
long tail of method knobs that `GenerationRequest` doesn't model as typed
fields: `extra_argv`. Anything you'd pass on the `inference.py` command line —
`--dcw`, `--spectrum`, their sub-knobs — goes here as verbatim CLI tokens, and
`.to_args()` feeds them through `inference.parse_args` so the generation code
sees them exactly as a CLI run would. `extra_argv` is appended last, so it can
also override a structured field.

Two corrections are wired as examples (both compose with the normal sampler,
both are training-free):

  * **dcw**      — SNR-t bias correction at the post-step boundary. `--dcw`
                   turns it on; `--dcw_lambda` is the scaler (negative on Anima —
                   see docs/inference/dcw.md). Negligible overhead.
  * **spectrum** — Chebyshev feature-forecasting acceleration. `--spectrum`
                   turns it on; cached steps skip the transformer blocks.
                   `--spectrum_warmup` is the full-forward warmup count.

The same pattern carries any other tail knob (ip-adapter, easycontrol, smc-cfg,
cns, …): build the token list and hand it to `extra_argv`.

Run from the repo root (anima_lora/):

    python examples/03_generate_with_correction.py --correction dcw
    python examples/03_generate_with_correction.py --correction dcw --dcw_lambda -0.012
    python examples/03_generate_with_correction.py --correction spectrum --spectrum_warmup 6
    python examples/03_generate_with_correction.py --correction none   # baseline

Compare the saved PNGs against `--correction none` to eyeball the effect.
"""

from __future__ import annotations

import argparse
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
from anima_lora import (
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_vae,
    save_output,
)
from library.runtime.device import clean_memory_on_device

# env (ANIMA_DIT / ANIMA_VAE / ANIMA_TEXT_ENCODER, incl. a project-root `.env`)
# → configs/base.toml → built-in fallbacks. See `.env.example`.
_ckpt = default_checkpoints()
DIT = _ckpt.dit
VAE = _ckpt.vae
TEXT_ENCODER = _ckpt.text_encoder


def correction_argv(opts: argparse.Namespace) -> list[str]:
    """Build the verbatim `inference.py` tokens for the chosen correction.

    These are exactly what you'd type after `python inference.py …` — the
    request doesn't model them as typed fields, so they ride `extra_argv`.
    """
    if opts.correction == "dcw":
        # store_true flag + one sub-knob. λ is negative on Anima (see docs).
        return ["--dcw", "--dcw_lambda", str(opts.dcw_lambda)]
    if opts.correction == "spectrum":
        return ["--spectrum", "--spectrum_warmup", str(opts.spectrum_warmup)]
    return []  # "none" → plain sampler, no correction


def build_request(opts: argparse.Namespace) -> GenerationRequest:
    return GenerationRequest(
        dit=DIT,
        vae=VAE,
        text_encoder=TEXT_ENCODER,
        prompt=opts.prompt,
        save_path=opts.save_path,
        infer_steps=opts.steps,
        guidance_scale=opts.cfg,
        image_size=tuple(opts.size),  # (H, W)
        seed=opts.seed,
        # The escape hatch: long-tail method flags as raw CLI tokens.
        extra_argv=correction_argv(opts),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--correction",
        choices=["dcw", "spectrum", "none"],
        default="dcw",
        help="Which training-free correction to enable via extra_argv.",
    )
    p.add_argument(
        "--dcw_lambda",
        type=float,
        default=-0.015,
        help="DCW scaler λ (negative on Anima). Used when --correction dcw.",
    )
    p.add_argument(
        "--spectrum_warmup",
        type=int,
        default=6,
        help="Spectrum full-forward warmup steps. Used when --correction spectrum.",
    )
    p.add_argument(
        "--prompt", default="a red fox sitting in a snowy forest, golden hour"
    )
    p.add_argument("--save_path", default="output/tests/example_03.png")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--seed", type=int, default=42)
    opts = p.parse_args()

    request = build_request(opts)
    extra = request.extra_argv
    print(f"correction={opts.correction!r}  extra_argv={list(extra)}")

    args = request.to_args()  # routes extra_argv through inference.parse_args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    gen_settings = get_generation_settings(args)
    latent = generate(args, gen_settings)
    clean_memory_on_device(device)

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
    print(f"saved → {args.save_path}  (correction: {opts.correction})")


if __name__ == "__main__":
    main()
