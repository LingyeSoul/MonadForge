#!/usr/bin/env python3
"""Held-out eval for the TE-only colorize LoRA (``colorize_cache.py``'s holdout).

Per held-out page, same seed: base with page + colour caption (the zero-shot
floor), the LoRA with page + colour caption, the LoRA with the page alone. Grid
columns: B&W page | colour target | base | lora+caption | lora page-only; ER-SDE
with Spectrum, blocks compiled after the LoRA is attached. A holdout record with a
``ref`` (``--ref_scale`` cache) feeds that reference latent to the LoRA arms. When
every holdout TE file carries ``embeds_v1`` (``colorize_cache.py --variants``), a
``lora_sparse`` arm adds the LoRA with page + that tag-dropped colour caption.
Latent names carry a digest of the LoRA (path + mtime), cache, steps, cfg and
sampler, so a rerun resumes only what matches.

    make daemon-run ARGS="project/moss_lora/src/colorize_eval.py --lora output/ckpt_moss/<name>.safetensors"
"""

import argparse
import json

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from _render import (
    decoded,
    grid,
    load_dit,
    lora_id,
    render,
    save_latent,
    settings_tag,
    spectrum_settings,
    uncond_embeds,
)
from library.env import resolve_under_home
from library.moss.cache import read_te
from library.moss.loader import empty_cache, load_vae
from library.moss.model import snapped_sigmas

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
    cache = resolve_under_home(args.cache)
    out = resolve_under_home(args.out or f"output/tests/colorize_{lora.stem}")
    (out / "latents").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    items = json.loads((cache / "holdout.json").read_text())
    arms = dict(ARMS)

    def te_keys(path):
        with safe_open(path, "pt") as f:
            return set(f.keys())

    if all("embeds_v1" in te_keys(it["te"]) for it in items):
        arms["lora_sparse"] = ("embeds_v1", 1.0)

    _, uncond, uncond_mask = uncond_embeds(device)
    sigmas = snapped_sigmas(args.steps)
    tag = settings_tag(
        lora=lora_id(lora),
        cache=str(cache),
        steps=args.steps,
        cfg=args.cfg,
        sampler="er_sde",
        spectrum=spectrum_settings(),
    )

    def lat(item, arm):
        return out / "latents" / f"{item['stem']}_{arm}_s{args.seed}_{tag}.safetensors"

    todo = [(it, a) for it in items for a in arms if not lat(it, a).exists()]
    if todo:
        tr, net = load_dit(lora, device)
        for it, arm in todo:
            key, mult = arms[arm]
            net.set_multiplier(mult)
            # The reference rides with the LoRA arms only: base never saw one.
            ref = None
            if "ref" in it and mult:
                ref = load_file(it["ref"])["latent"][None]
            z = render(
                tr,
                read_te(it["te"], key),
                uncond,
                uncond_mask,
                sigmas,
                seed=args.seed,
                height=it["height"],
                width=it["width"],
                cfg=args.cfg,
                ref=ref,
            )
            save_latent(z, lat(it, arm))
            print(f"[dit] {it['stem']} {arm}", flush=True)
        del net, tr
        empty_cache()

    vae = load_vae("", device=device)
    rows = []
    for it in items:
        cells = [
            ("page", Image.open(it["condition"]).convert("RGB")),
            ("target", Image.open(it["target"]).convert("RGB")),
        ]
        cells += [(arm, decoded(lat(it, arm), vae, device)) for arm in arms]
        rows.append((it["stem"], cells))
    del vae
    empty_cache()

    grid(rows).save(out / "grid.png")
    (out / "index.json").write_text(
        json.dumps(
            {"args": vars(args), "tag": tag, "items": [it["stem"] for it in items]},
            indent=2,
        )
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
