#!/usr/bin/env python3
"""Held-out eval for the TE-only colorize LoRA (``colorize_cache.py``'s holdout).

Per held-out page, same seed: base with page + colour caption (the zero-shot
floor), the LoRA with page + colour caption, the LoRA with the page alone. Grid
columns: B&W page | colour target | base | lora+caption | lora page-only; ER-SDE
with Spectrum, blocks compiled after the LoRA is attached. A holdout record with a
``ref`` (``--ref_scale`` cache) feeds that reference latent to the LoRA arms. When
every holdout TE file carries ``embeds_v1`` (``colorize_cache.py --variants``), a
``lora_sparse`` arm adds the LoRA with page + that tag-dropped colour caption.
The uncond is the cache's ``uncond.safetensors`` when it has one
(``colorize_te_fp8.py`` writes it with the cache's own TE), else the empty prompt at
``--te_precision``; ``--uncond bf16`` forces the bf16 empty prompt. ``index.json`` scores each arm: luma r against
the target (128 px), chroma (mean CbCr distance from grey at 1/8 size) and chroma
error (mean CbCr distance to the target at 1/8 size).
Latent names carry a digest of the LoRA (path + mtime), cache, steps, cfg and
sampler, so a rerun resumes only what matches.

    make daemon-run ARGS="project/moss_lora/src/colorize_eval.py --lora output/ckpt_moss/<name>.safetensors"
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
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
    uncond_embeds,
)
from library.env import resolve_under_home
from library.moss.cache import read_te
from library.moss.loader import empty_cache, load_vae, pad_embeds
from library.moss.model import snapped_sigmas

# name: (TE key, LoRA multiplier)
ARMS = {
    "base": ("embeds", 0.0),
    "lora_cap": ("embeds", 1.0),
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
    ap.add_argument("--cache", default="post_image_dataset/moss/colorize_te")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--te_precision", default="bf16", choices=("bf16", "fp8"))
    ap.add_argument("--uncond", default="auto", choices=("auto", "bf16"))
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

    own = cache / "uncond.safetensors"
    if args.uncond == "auto" and own.is_file():
        uncond, uncond_mask = pad_embeds([load_file(str(own))["embeds"]], device)
        uncond_src = "cache"
    else:
        precision = "bf16" if args.uncond == "bf16" else args.te_precision
        _, uncond, uncond_mask = uncond_embeds(device, precision=precision)
        uncond_src = precision
    print(f"[uncond] {uncond_src}", flush=True)
    sigmas = snapped_sigmas(args.steps)
    # bf16 leaves the tag as it was, so earlier latents still resume.
    te = {} if args.te_precision == "bf16" else {"te_precision": args.te_precision}
    if uncond_src != args.te_precision:
        te["uncond"] = uncond_src
    tag = settings_tag(
        lora=lora_id(lora),
        cache=str(cache),
        steps=args.steps,
        cfg=args.cfg,
        sampler="er_sde",
        spectrum=spectrum_settings(),
        **te,
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
    rows, scores = [], {}
    for it in items:
        target = Image.open(it["target"]).convert("RGB")
        cells = [
            ("page", Image.open(it["condition"]).convert("RGB")),
            ("target", target),
        ]
        for arm in arms:
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
                "items": [it["stem"] for it in items],
                "scores": scores,
            },
            indent=2,
        )
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
