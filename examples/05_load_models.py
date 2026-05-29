#!/usr/bin/env python3
"""Access the three core models directly — the primitives a scripts/ tool builds on.

inference.py and train.py wrap these loaders; if you're writing your own script
(a probe, a metric, a one-off batch job) you want them raw:

  - DiT  : library.anima.weights.load_anima_model()
  - VAE  : library.models.qwen_vae.load_vae()
  - Text : library.inference.models.load_text_encoder()  (Qwen3)

This script loads all three, then encodes a prompt to the DiT-ready cross-attn
embedding via the supported prepare_text_inputs() helper — which is where the
text-encoder padding invariant lives (max-pad to 512; the DiT projects the
encoder hidden states through `_preprocess_text_embeds`, so encoding genuinely
needs the DiT, not just the text encoder).

    python examples/05_load_models.py --prompt "a lighthouse at dusk"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from library.anima import weights as anima_weights
from library.inference.models import load_text_encoder
from library.inference.text import (
    MAX_CROSSATTN_TOKENS,
    ensure_text_strategies,
    prepare_text_inputs,
)
from library.models import qwen_vae

DIT = os.environ.get("ANIMA_DIT", "models/diffusion_models/anima-base-v1.0.safetensors")
VAE = os.environ.get("ANIMA_VAE", "models/vae/qwen_image_vae.safetensors")
TEXT_ENCODER = os.environ.get(
    "ANIMA_TEXT_ENCODER", "models/text_encoders/qwen_3_06b_base.safetensors"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default="a lighthouse at dusk, dramatic clouds")
    opts = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- DiT ----------------------------------------------------------------
    # load_anima_model takes explicit paths/dtypes (no args namespace). attn_mode
    # "torch" is the portable default; "flash"/"flex" need the matching kernels.
    dit = anima_weights.load_anima_model(
        device=device,
        dit_path=DIT,
        attn_mode="torch",
        loading_device=device,
        dit_weight_dtype=torch.bfloat16,
    )
    dit.eval()
    n_dit = sum(p.numel() for p in dit.parameters())
    print(
        f"DiT   : {type(dit).__name__}  {n_dit / 1e9:.2f}B params  patch_spatial={dit.patch_spatial}"
    )

    # --- VAE ----------------------------------------------------------------
    vae = qwen_vae.load_vae(
        VAE, device=device, disable_mmap=True, dtype=torch.bfloat16, eval=True
    )
    print(f"VAE   : {type(vae).__name__}  z_dim={vae.z_dim}")

    # --- Text encoder + encode a prompt -------------------------------------
    # Prompt encoding goes through two process-global strategy singletons (the
    # strategy pattern in library/anima/strategy.py). ensure_text_strategies()
    # installs them from the text-encoder path — a no-op if already set, and the
    # same call prepare_text_inputs() makes internally — and hands them back so
    # you can see (and use) them rather than fishing them out of the globals.
    # (Skip it and the first tokenize() dies with `'NoneType' has no tokenize`.)
    tokenize_strategy, encoding_strategy = ensure_text_strategies(
        TEXT_ENCODER, max_length=MAX_CROSSATTN_TOKENS
    )
    print(
        f"Strat : {type(tokenize_strategy).__name__} + "
        f"{type(encoding_strategy).__name__}"
    )

    # Loading the encoder needs only its path — pass it as a keyword. (No prompt,
    # no save_path: those belong to generation, not to loading a model.)
    text_encoder = load_text_encoder(
        text_encoder=TEXT_ENCODER,
        dtype=torch.bfloat16,
        device=device,
    )
    text_encoder.eval()
    print(f"Text  : {type(text_encoder).__name__} (Qwen3)")

    # prepare_text_inputs returns (context, context_null); context['embed'][0] is
    # the cross-attn embedding the DiT consumes, max-padded to MAX_CROSSATTN_TOKENS.
    # Pass the request as keywords — the encoder is already loaded, so hand it in
    # via shared_models and prepare_text_inputs won't reload it.
    context, _context_null = prepare_text_inputs(
        device=device,
        anima=dit,
        prompt=opts.prompt,
        shared_models={"text_encoder": text_encoder},
    )
    crossattn = context["embed"][0]
    print(
        f"\nprompt → cross-attn embedding: shape={tuple(crossattn.shape)} "
        f"(padded to {MAX_CROSSATTN_TOKENS} tokens — do NOT trim, padding acts as "
        f"attention sinks)"
    )


if __name__ == "__main__":
    main()
