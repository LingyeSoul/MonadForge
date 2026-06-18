#!/usr/bin/env python3
"""EasyControl end-to-end from Python: config → train → image-conditioned infer.

EasyControl is *image-conditioned* generation — a frozen DiT with per-block
condition-LoRA on self-attn (q/k/v/o) + FFN, gated by a learned scalar `b_cond`
(extended self-attention over the reference image's VAE latent). Unlike the
text-only LoRA flows, it has two halves worth showing together:

  1. **Training** is a *normal* `--method easycontrol`. The self-contained config
     `configs/easycontrol/easycontrol.toml` carries both the method settings AND
     an inline dataset blueprint (paired source→target — here ref==target, like
     IP-Adapter v1), and `_resolve_method_path` auto-discovers the per-method dir
     over the flat `configs/methods/`. So it rides the *exact same* merge-chain +
     `AnimaTrainer().train()` path as `examples/02_config_and_train.py` — the only
     difference is the method name. Source images + `.txt` captions go in
     `easycontrol-dataset/`; caches are redirected to
     `post_image_dataset/easycontrol/` via the subset `cache_dir`.

  2. **Inference** is request-driven like `01`/`03`, but adds two typed fields the
     other flows leave unset: `easycontrol_weight` (the trained adapter) and
     `easycontrol_image` (the reference). `GenerationRequest` models both as
     first-class fields — no `extra_argv` needed for them. The two remaining
     knobs (`--easycontrol_image_match_size`, `--easycontrol_scale`) aren't typed,
     so they ride `extra_argv` exactly like the corrections in `03`.

Three parts, each opt-in so the cheap one runs by default:

    python examples/09_easycontrol_train_and_infer.py                  # part 1: print config
    python examples/09_easycontrol_train_and_infer.py --train --max_train_epochs 6
    python examples/09_easycontrol_train_and_infer.py --infer --ref path/to/ref.png

Part 2 prereq: `make download-models` + put paired data in `easycontrol-dataset/`
then `make preprocess` (training reads only the cached latents/embeddings). Part 3
prereq: a trained `anima_easycontrol*.safetensors` (defaults to the newest under
`output/ckpt/`) + a reference image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python examples/<script>.py`: put the repo root on sys.path so
# `import library` / `train` resolve. Model/config paths resolve against the
# repo home regardless of CWD (set ANIMA_HOME for a relocated checkout); only the
# output paths below are written relative to your CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

# Curated entry points on the `anima_lora` front door (thin lazy re-export).
from anima_lora import (
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_method_preset,
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

# The keys EasyControl's method file forces on top of base+preset — what makes
# this an image-conditioning run rather than a text-only LoRA. (See the merge
# chain: method settings win over preset on overlap.)
EASYCONTROL_KEYS = (
    "network_module",
    "network_dim",
    "use_easycontrol",
    "easycontrol_drop_p",
    "easycontrol_cond_noise_max",
    "output_name",
)


def show_config(preset: str) -> dict:
    """Part 1 — merge + print the EasyControl-defining keys with provenance.

    Identical helper to examples/02, just pinned to `--method easycontrol` and
    its method-specific keys. No GPU, no weights.
    """
    merged, provenance = load_method_preset(
        "easycontrol", preset, return_provenance=True
    )
    print(f"\nmethod='easycontrol'  preset={preset!r}")
    print("-" * 72)
    for k in EASYCONTROL_KEYS:
        src = provenance.get(k, "(unset → code default)")
        print(f"  {k:26} = {merged.get(k)!r:24}  ← {src}")
    print("-" * 72)
    print(f"  ({len(merged)} keys total in the merged config)\n")
    return merged


def run_training(preset: str, extra_argv: list[str]) -> None:
    """Part 2 — reproduce train.py's __main__ block for `--method easycontrol`.

    Byte-for-byte the same wiring as examples/02's run_training (setup_parser →
    populate_schema → parse → read_config_from_file → AnimaTrainer().train); the
    only change is the hard-coded method. EasyControl is not special here — the
    frozen-DiT + cond-LoRA build happens inside AnimaTrainer, driven entirely by
    the resolved config.
    """
    from train import (
        AnimaTrainer,
        build_network_extras,
        setup_parser,
        verify_command_line_training_args,
    )
    from library.config import schema as config_schema

    from anima_lora import read_config_from_file

    argv = ["--method", "easycontrol", "--preset", preset, *extra_argv]

    parser = setup_parser()
    config_schema.populate_schema(parser, extras=build_network_extras())
    args = parser.parse_args(argv)
    verify_command_line_training_args(args)
    # base→preset→method merge, then our CLI overrides on top (so e.g.
    # --max_train_epochs wins). argv passed explicitly so the override layer is
    # driven by our list, not the process sys.argv.
    args = read_config_from_file(args, parser, argv=argv)
    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility
    AnimaTrainer().train(args)


def _latest_easycontrol_weight() -> str:
    """Newest `anima_easycontrol*.safetensors` under output/ckpt/ (matches the
    `output_name` in the method config). Mirrors how `make test-easycontrol`
    resolves the weight via `latest_output('anima_easycontrol')`."""
    ckpt_dir = Path("output/ckpt")
    cands = sorted(
        ckpt_dir.glob("anima_easycontrol*.safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise SystemExit(
            "no anima_easycontrol*.safetensors under output/ckpt/ — train first "
            "(--train) or pass --weight <path>."
        )
    return str(cands[0])


def run_inference(opts: argparse.Namespace) -> None:
    """Part 3 — image-conditioned generation with the trained adapter.

    Same request → generate → decode → save flow as 01/03, but with the two
    EasyControl typed fields populated. `--easycontrol_image_match_size` (resize
    the cond latent to the generation size) and `--easycontrol_scale` (cond
    strength) aren't typed fields, so they ride `extra_argv` — the same escape
    hatch 03 uses for the sampler corrections.
    """
    weight = opts.weight or _latest_easycontrol_weight()

    extra = ["--easycontrol_image_match_size"]
    if opts.scale is not None:
        extra += ["--easycontrol_scale", str(opts.scale)]

    request = GenerationRequest(
        dit=DIT,
        vae=VAE,
        text_encoder=TEXT_ENCODER,
        prompt=opts.prompt,
        save_path=opts.save_path,
        infer_steps=opts.steps,
        guidance_scale=opts.cfg,
        image_size=tuple(opts.size),  # (H, W)
        seed=opts.seed,
        easycontrol_weight=weight,  # typed field — no extra_argv needed
        easycontrol_image=opts.ref,  # typed field — the reference image
        extra_argv=extra,  # the untyped knobs (match_size / scale)
    )
    print(f"easycontrol_weight = {weight}")
    print(f"easycontrol_image  = {opts.ref}")
    print(f"extra_argv         = {extra}")

    args = request.to_args()  # routes everything through inference.parse_args
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
        dtype=torch.bfloat16,
        eval=True,
    )
    save_output(args, vae, latent, device)
    print(f"saved → {args.save_path}")


def main() -> None:
    # parse_known_args so training overrides (e.g. --max_train_epochs 6) forward
    # verbatim to the trainer, exactly like the CLI override layer in 02.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="default")
    p.add_argument(
        "--train",
        action="store_true",
        help="part 2: run EasyControl training in-process (needs preprocessed cache)",
    )
    p.add_argument(
        "--infer",
        action="store_true",
        help="part 3: image-conditioned generation with the trained adapter",
    )
    # Inference knobs (ignored unless --infer).
    p.add_argument("--ref", help="reference image for --infer (the conditioning input)")
    p.add_argument(
        "--weight",
        help="EasyControl adapter .safetensors (default: newest anima_easycontrol* in output/ckpt/)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help="--easycontrol_scale (cond strength); omitted → parser default",
    )
    p.add_argument("--prompt", default="a girl in a coffee shop, detailed background")
    p.add_argument("--save_path", default="output/tests/example_09.png")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--seed", type=int, default=42)
    opts, extra_argv = p.parse_known_args()

    if opts.train:
        run_training(opts.preset, extra_argv)
        return
    if opts.infer:
        if not opts.ref:
            raise SystemExit("--infer needs --ref <reference_image>")
        run_inference(opts)
        return

    # Default (cheap): just show the merged config — what `--method easycontrol`
    # resolves to before any weight is touched.
    show_config(opts.preset)


if __name__ == "__main__":
    main()
