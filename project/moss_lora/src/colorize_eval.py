#!/usr/bin/env python3
"""Held-out eval for a colorize LoRA on a ``colorize_cache.py`` cache.

Per held-out page, same seed, one arm per caption the LoRA trained on:

    base         no LoRA, no reference, the OCR caption (the zero-shot floor)
    lora_ocr     ``embeds``: the layout JSON with the page's OCR lines
    lora_sparse  ``embeds_v1``: a tag-dropped colour caption, no OCR
    lora_page    ``embeds_plain``: the page alone

The LoRA arms get the page's reference latent. Grid columns: B&W page | colour
target | the arms; ER-SDE with Spectrum, blocks compiled after the LoRA is
attached, the cache's ``uncond.safetensors`` as the empty prompt. ``index.json``
scores each arm: luma r against the target (128 px), chroma (mean CbCr distance
from grey at 1/8 size) and chroma error (mean CbCr distance to the target at 1/8
size). Latent names carry a digest of the LoRA (path + mtime), cache, steps, cfg
and sampler, so a rerun resumes only what matches. The base arm does not depend on
the LoRA: its latents live under ``output/tests/colorize_base/<cache>/``, shared by
every LoRA evaluated on that cache.

    make daemon-run ARGS="project/moss_lora/src/colorize_eval.py --lora output/ckpt_moss/<name>.safetensors --cache post_image_dataset/moss/<name>"
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from colorize_wild import chroma, luma_r
from _render import (
    decoded,
    grid,
    load_dit,
    lora_id,
    render,
    save_latent,
    settings_tag,
    spectrum_settings,
)
from library.env import resolve_under_home
from library.moss.cache import read_te
from library.moss.loader import empty_cache, load_vae, pad_embeds
from library.moss.model import snapped_sigmas

# name: (TE key, LoRA multiplier)
ARMS = {
    "base": ("embeds", 0.0),
    "lora_ocr": ("embeds", 1.0),
    "lora_sparse": ("embeds_v1", 1.0),
    "lora_page": ("embeds_plain", 1.0),
}


def chroma_error(im: Image.Image, target: Image.Image) -> float:
    size = (im.width // 8, im.height // 8)

    def cbcr(x):
        return np.asarray(x.resize(size, Image.BOX).convert("YCbCr"), np.float64)[
            ..., 1:
        ]

    return float(np.linalg.norm(cbcr(im) - cbcr(target), axis=-1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--cache", required=True)
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
    uncond, uncond_mask = pad_embeds(
        [load_file(str(cache / "uncond.safetensors"))["embeds"]], device
    )
    sigmas = snapped_sigmas(args.steps)
    settings = dict(
        cache=str(cache),
        steps=args.steps,
        cfg=args.cfg,
        sampler="er_sde",
        spectrum=spectrum_settings(),
    )
    tag = settings_tag(lora=lora_id(lora), **settings)
    base_tag = settings_tag(**settings)
    base_dir = resolve_under_home(f"output/tests/colorize_base/{cache.name}")
    base_dir.mkdir(parents=True, exist_ok=True)

    def lat(item, arm):
        if arm == "base":
            return base_dir / f"{item['stem']}_base_s{args.seed}_{base_tag}.safetensors"
        return out / "latents" / f"{item['stem']}_{arm}_s{args.seed}_{tag}.safetensors"

    todo = [(it, a) for it in items for a in ARMS if not lat(it, a).exists()]
    if todo:
        tr, net = load_dit(lora, device)
        for it, arm in todo:
            key, mult = ARMS[arm]
            net.set_multiplier(mult)
            ref = load_file(it["ref"])["latent"][None] if mult else None
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
    rows, scores = [], {}
    for it in items:
        target = Image.open(it["target"]).convert("RGB")
        cells = [
            ("page", Image.open(it["condition"]).convert("RGB")),
            ("target", target),
        ]
        for arm in ARMS:
            im = decoded(lat(it, arm), vae, device)
            cells.append((arm, im))
            scores.setdefault(arm, {})[it["stem"]] = {
                "luma_r": round(luma_r(target, im), 4),
                "chroma": round(chroma(im), 2),
                "chroma_err": round(chroma_error(im, target), 2),
            }
        rows.append((it["stem"], cells))
    del vae
    empty_cache()

    grid(rows).save(out / "grid.png")
    for arm, per in scores.items():
        mean = {
            k: np.mean([v[k] for v in per.values()])
            for k in ("luma_r", "chroma", "chroma_err")
        }
        print(
            f"{arm}: luma r {mean['luma_r']:.3f}, chroma {mean['chroma']:.1f}, "
            f"chroma err {mean['chroma_err']:.1f}",
            flush=True,
        )
    (out / "index.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "tag": tag,
                "base_tag": base_tag,
                "items": [it["stem"] for it in items],
                "scores": scores,
            },
            indent=2,
        )
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
