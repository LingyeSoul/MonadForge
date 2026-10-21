#!/usr/bin/env python3
"""Render a MOSS LoRA with its delta kept off the 10 chat-template rows.

``masked``: the LoRA at ×1, but every targeted linear returns the base output on
the template rows (``<|im_start|>user\\n`` before the caption, the 7 suffix tokens
after it), so the template's Q / K / V / O / FFN stay at Base. Compiled like
``scripts/moss/generate.py``, so it pairs with that script's base / LoRA renders of
the same prompts and seeds (``--ref``); ``--arms base,lora,masked`` renders all three
here instead.

``lora`` vs ``masked`` is what the LoRA does through the template rows
(``docs/moss/findings/attention_sink_chat_template.md``). The uncond text is the
template alone, so ``masked`` leaves uncond's text rows at Base entirely.

    make daemon-run ARGS="bench/moss_sinks/render_template_mask.py \\
        --lora output/ckpt_moss/<lora>.safetensors --out output/tests/<dir>"

Images land in ``<out>/<prompt>/<arm>/s<seed>.png``, plus ``<out>/grid_<prompt>.png``
(rows = ``--ref`` arms then the rendered arms, columns = seeds).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "moss_spectrum"))
import torch  # noqa: E402

from bench._common import start_heartbeat  # noqa: E402
from run_bench import PROMPTS  # noqa: E402

ARMS = ("base", "lora", "masked")


class TemplateMask:
    """Per-adapter ``forward`` wrapper: on the joint sequence ``[image | text]``,
    the template rows take the base output instead of base + delta."""

    def __init__(self, n_img, prefix, suffix):
        self.n_img, self.prefix, self.suffix = n_img, prefix, suffix
        self.on = False
        self._rows = {}

    def rows(self, n, device):
        """``(1, N, 1)`` bool, True on the template rows of an ``N``-long sequence."""
        if n not in self._rows:
            m = torch.zeros(n, dtype=torch.bool, device=device)
            m[self.n_img : self.n_img + self.prefix] = True
            m[n - self.suffix :] = True
            self._rows[n] = m[None, :, None]
        return self._rows[n]

    def wrap(self, adapter):
        forward = adapter.forward

        def masked(x, base_out):
            out = forward(x, base_out)
            if not self.on or x.shape[1] <= self.n_img:
                return out
            return torch.where(self.rows(x.shape[1], out.device), base_out, out)

        adapter.forward = masked


def grid(paths, labels, out, tile=384):
    from PIL import Image, ImageDraw

    cols = max(len(r) for r in paths)
    g = Image.new("RGB", (90 + cols * tile, len(paths) * tile), "white")
    draw = ImageDraw.Draw(g)
    for r, (row, label) in enumerate(zip(paths, labels)):
        draw.text((6, r * tile + tile // 2), label, fill="black")
        for c, p in enumerate(row):
            im = Image.open(p).convert("RGB").resize((tile, tile))
            g.paste(im, (90 + c * tile, r * tile))
    g.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lora", required=True)
    ap.add_argument("--model", default="base")
    ap.add_argument("--prompts", default="anime_sunset,ja_manga")
    ap.add_argument("--prompt_format", default="text", choices=("text", "layout_json"))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--arms", default="masked", help=f"subset of {','.join(ARMS)}")
    ap.add_argument(
        "--ref",
        default="base:base,lora:lr5e5dc",
        help="grid rows taken from existing renders in <out>/<prompt>/<folder>/*_s<seed>.png, "
        "as label:folder pairs ('' = none)",
    )
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_dir", default="")
    ap.add_argument(
        "--te_cache", default="/media/sorryhyun/data/moss_image/bench/_te_cache"
    )
    args = ap.parse_args()
    start_heartbeat(label="moss_template_mask")

    from diffusers.image_processor import VaeImageProcessor

    from library.env import resolve_under_home
    from library.moss.generate import cached_embeds, denoise, initial_noise
    from library.moss.loader import (
        empty_cache,
        load_tokenizer,
        load_transformer,
        load_vae,
        pad_embeds,
    )
    from library.moss.lora import load_for_inference
    from library.moss.model import decode, snapped_sigmas
    from library.moss.requests import DEFAULT_UNCOND_SKIP, resolve_model_dir

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from probe_attention import template_split

    keys = [k for k in args.prompts.split(",") if k]
    texts = {k: PROMPTS[k] for k in keys}
    if args.prompt_format != "text":
        from library.moss.cache import caption_converter

        convert = caption_converter(args.prompt_format)
        texts = {k: convert(t) for k, t in texts.items()}
    model_dir = resolve_model_dir(args.model_dir)
    device = torch.device("cuda")
    out_root = resolve_under_home(args.out)
    arms = [a for a in args.arms.split(",") if a]
    if set(arms) - set(ARMS):
        raise SystemExit(f"--arms must be a subset of {ARMS}")
    refs = [tuple(r.split(":", 1)) for r in args.ref.split(",") if r]

    tok = load_tokenizer(model_dir)
    prefix, _, suffix = template_split(tok, "")
    del tok
    embeds = cached_embeds(model_dir, [*texts.values(), ""], Path(args.te_cache))
    uncond, uncond_mask = pad_embeds([embeds[""]], device=device)
    transformer = load_transformer(model_dir, args.model, device=device)
    network = load_for_inference(transformer, resolve_under_home(args.lora), 1.0)
    mask = TemplateMask((args.size // 16) ** 2, prefix, suffix)
    for adapter in network.adapters.values():
        mask.wrap(adapter)
    print(f"[lora] {args.lora}; template rows: {prefix} prefix + {suffix} suffix")
    if not args.no_compile:
        from library.moss.loader import compile_blocks

        compile_blocks(transformer.layers)

    sigmas = snapped_sigmas(args.steps)
    latents = {}
    for arm in arms:
        network.set_multiplier(0.0 if arm == "base" else 1.0)
        mask.on = arm == "masked"
        for key in keys:
            cond, cond_mask = pad_embeds([embeds[texts[key]]], device=device)
            for s in range(args.seeds):
                seed = args.seed0 + s
                latents[arm, key, seed] = denoise(
                    transformer,
                    initial_noise(seed, args.size, args.size).to(device),
                    cond,
                    cond_mask,
                    uncond,
                    uncond_mask,
                    sigmas,
                    cfg=args.cfg,
                    uncond_skip=DEFAULT_UNCOND_SKIP,
                ).cpu()
                print(f"[render] {arm} {key} s{seed}", flush=True)

    # decode with the DiT off the card
    del network, transformer
    empty_cache()
    vae = load_vae(model_dir, device=device)
    processor = VaeImageProcessor(vae_scale_factor=16)
    paths = {}
    for (arm, key, seed), z in latents.items():
        path = out_root / key / arm / f"s{seed}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image = decode(z.to(device), vae)
        processor.postprocess(image, output_type="pil")[0].save(path)
        paths[arm, key, seed] = path
    seeds = [args.seed0 + s for s in range(args.seeds)]
    for key in keys:
        rows = [
            [next((out_root / key / folder).glob(f"*_s{seed}.png")) for seed in seeds]
            for _label, folder in refs
        ]
        rows += [[paths[arm, key, seed] for seed in seeds] for arm in arms]
        grid(rows, [label for label, _ in refs] + arms, out_root / f"grid_{key}.png")
        print(f"→ {out_root / f'grid_{key}.png'}", flush=True)


if __name__ == "__main__":
    main()
