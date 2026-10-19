#!/usr/bin/env python3
"""Held-out eval for the TE-only colorize LoRA (``colorize_cache.py``'s holdout).

Per held-out page, same seed: base with page + colour caption (the zero-shot
floor), the LoRA with page + colour caption, the LoRA with the page alone. Grid
columns: B&W page | colour target | base | lora+caption | lora page-only; ER-SDE
with Spectrum, blocks compiled after the LoRA is attached. A holdout record with a
``ref`` (``--ref_scale`` cache) feeds that reference latent to the LoRA arms.

    make daemon-run ARGS="project/moss_lora/src/colorize_eval.py --lora output/ckpt_moss/<name>.safetensors"
"""

import argparse
import json

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file

from library.env import resolve_under_home
from library.moss.generate import (
    _build_spectrum,
    _cached_embeds,
    denoise,
    initial_noise,
)
from library.moss.loader import (
    compile_blocks,
    empty_cache,
    load_transformer,
    load_vae,
    pad_embeds,
)
from library.moss.lora import load_for_inference
from library.moss.model import decode, snapped_sigmas
from library.moss.requests import GenerateRequest

# name: (TE key, LoRA multiplier)
ARMS = {
    "base": ("embeds", 0.0),
    "lora_cap": ("embeds", 1.0),
    "lora_page": ("embeds_plain", 1.0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--cache", default="post_image_dataset/moss/colorize_te")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    lora = resolve_under_home(args.lora)
    out = resolve_under_home(args.out or f"output/tests/colorize_{lora.stem}")
    (out / "latents").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    items = json.loads((resolve_under_home(args.cache) / "holdout.json").read_text())

    uncond = _cached_embeds(
        "", [""], resolve_under_home("post_image_dataset/moss_prompt_te")
    )[""]
    uncond, uncond_mask = pad_embeds([uncond], device)
    sigmas = snapped_sigmas(args.steps)

    def lat(item, arm):
        return out / "latents" / f"{item['stem']}_{arm}_s{args.seed}.safetensors"

    todo = [(it, a) for it in items for a in ARMS if not lat(it, a).exists()]
    if todo:
        tr = load_transformer("", "base", device=device)
        net = load_for_inference(tr, lora)
        compile_blocks(tr.layers)
        for it, arm in todo:
            key, mult = ARMS[arm]
            net.set_multiplier(mult)
            cond, cond_mask = pad_embeds([load_file(it["te"])[key]], device)
            # The reference rides with the LoRA arms only: base never saw one.
            ref = None
            if "ref" in it and mult:
                ref = load_file(it["ref"])["latent"][None].to(device, torch.float32)
            z = denoise(
                tr,
                initial_noise(args.seed, it["height"], it["width"]).to(device),
                cond,
                cond_mask,
                uncond,
                uncond_mask,
                sigmas,
                cfg=args.cfg,
                sampler="er_sde",
                seed=args.seed,
                spectrum=_build_spectrum(GenerateRequest(spectrum=True), {}),
                ref=ref,
            )
            save_file({"latent": z.cpu().contiguous()}, str(lat(it, arm)))
            print(f"[dit] {it['stem']} {arm}", flush=True)
        del net, tr
        empty_cache()

    vae = load_vae("", device=device)
    cell_h = 640
    rows = []
    for it in items:
        cells = [
            ("page", Image.open(it["condition"]).convert("RGB")),
            ("target", Image.open(it["target"]).convert("RGB")),
        ]
        for arm in ARMS:
            png = lat(it, arm).with_suffix(".png")
            if not png.exists():
                z = load_file(str(lat(it, arm)))["latent"].to(device)
                x = decode(z, vae)[0].float().add(1).mul(127.5).round().clamp(0, 255)
                Image.fromarray(x.permute(1, 2, 0).byte().cpu().numpy()).save(png)
            cells.append((arm, Image.open(png).convert("RGB")))
        rows.append((it, cells))
    del vae
    empty_cache()

    cw = max(round(it["width"] * cell_h / it["height"]) for it, _ in rows)
    grid = Image.new("RGB", (cw * 5, (cell_h + 30) * len(rows)), "white")
    d = ImageDraw.Draw(grid)
    for ri, (it, cells) in enumerate(rows):
        y = ri * (cell_h + 30)
        for ci, (label, im) in enumerate(cells):
            th = im.resize(
                (round(im.width * cell_h / im.height), cell_h), Image.LANCZOS
            )
            grid.paste(th, (ci * cw, y + 30))
            d.text((ci * cw + 6, y + 8), f"{it['stem']} {label}", fill="black")
    grid.save(out / "grid.png")
    (out / "index.json").write_text(
        json.dumps(
            {"args": vars(args), "items": [it["stem"] for it in items]}, indent=2
        )
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
