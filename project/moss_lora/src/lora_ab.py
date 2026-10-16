#!/usr/bin/env python3
"""LoRA A/B on MOSS-Image Base and SFT: each prompt × seed over ``--arms``.

An arm is ``<sampler>:<multiplier>``; sampler ``euler``, ``er_sde`` or ``cns``
(er_sde with ``--cns_gamma``'s recolored noise). The default ``euler:1.0,euler:0.0``
is the plain adapter A/B. Same noise per (prompt, seed), so each row differs by the
arm only. Writes one row per (variant, prompt, seed) into ``--out``.

    make daemon-run ARGS="project/moss_lora/src/lora_ab.py --lora output/moss/lora/x.safetensors"
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file

from library.env import resolve_under_home
from library.moss.cns import MossCNS
from library.moss.generate import _cached_embeds, denoise, initial_noise
from library.moss.loader import empty_cache, load_transformer, load_vae, pad_embeds
from library.moss.lora import load_for_inference
from library.moss.model import decode, snapped_sigmas

PROMPTS = [
    # Tag bag in the training captions' own format, characters it saw.
    "safe, 1girl, hatsune miku, vocaloid, @channel (caststation), aqua eyes, aqua hair, "
    "twintails, smile, peace sign, upper body, looking at viewer, simple background, white background",
    # A character the set never shows.
    "safe, 1girl, akita neru, vocaloid, @channel (caststation), yellow eyes, blonde hair, "
    "side ponytail, holding phone, looking at viewer, upper body, simple background",
    # Natural language, no tag grammar.
    "An illustration in the style of @channel (caststation): a girl with short black hair "
    "in a sailor school uniform sitting on a bench under cherry blossoms, smiling.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--variants", default="base,sft")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument(
        "--arms",
        default="euler:1.0,euler:0.0",
        help="<euler|er_sde|cns>:<multiplier>, comma-separated",
    )
    ap.add_argument("--cns_gamma", default="output/moss/cns_gamma.npz")
    ap.add_argument(
        "--out", default="", help="default: output/moss/lora/ab_<lora stem>_<steps>st"
    )
    args = ap.parse_args()
    w, h = map(int, args.size.split("x"))
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [(a.split(":")[0], float(a.split(":")[1])) for a in args.arms.split(",")]
    out = resolve_under_home(
        args.out or f"output/moss/lora/ab_{Path(args.lora).stem}_{args.steps}st"
    )
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    embeds = _cached_embeds(
        "", [*PROMPTS, ""], resolve_under_home("output/moss/te_cache")
    )
    uncond, uncond_mask = pad_embeds([embeds[""]], device)
    sigmas = snapped_sigmas(args.steps)

    # Each latent hits disk as soon as its denoise ends; reruns skip what is there.
    lat_dir = out / "latents"
    lat_dir.mkdir(exist_ok=True)
    variants = args.variants.split(",")
    jobs = [
        (v, pi, s, a)
        for v in variants
        for pi in range(len(PROMPTS))
        for s in seeds
        for a in arms
    ]

    def stem(job):
        # Euler keeps the pre-arms names, so earlier runs resume.
        v, pi, s, (sampler, m) = job
        return f"{v}_p{pi}_s{s}_x{m}" + ("" if sampler == "euler" else f"_{sampler}")

    def lat_path(job):
        return lat_dir / f"{stem(job)}.safetensors"

    for variant in variants:
        todo = [j for j in jobs if j[0] == variant and not lat_path(j).exists()]
        if not todo:
            continue
        tr = load_transformer("", variant, device=device)
        net = load_for_inference(tr, resolve_under_home(args.lora))
        for job in todo:
            _, pi, seed, (sampler, m) = job
            cond, cond_mask = pad_embeds([embeds[PROMPTS[pi]]], device)
            net.set_multiplier(m)
            cns = (
                MossCNS.from_npz(resolve_under_home(args.cns_gamma))
                if sampler == "cns"
                else None
            )
            z = denoise(
                tr,
                initial_noise(seed, h, w).to(device),
                cond,
                cond_mask,
                uncond,
                uncond_mask,
                sigmas,
                cfg=args.cfg,
                sampler="euler" if sampler == "euler" else "er_sde",
                seed=seed,
                cns=cns,
            )
            save_file({"latent": z.cpu().contiguous()}, str(lat_path(job)))
            print(
                f"[{variant}] p{pi} s{seed} {sampler} x{m} -> {lat_path(job).name}",
                flush=True,
            )
        del net, tr
        empty_cache()

    vae = load_vae("", device=device)
    rows = []
    for job in jobs:
        variant, pi, seed, (sampler, m) = job
        path = out / f"{stem(job)}.png"
        if not path.exists():
            z = load_file(str(lat_path(job)))["latent"]
            img = (
                decode(z.to(device), vae)[0]
                .float()
                .add(1)
                .mul(127.5)
                .round()
                .clamp(0, 255)
            )
            Image.fromarray(img.permute(1, 2, 0).byte().cpu().numpy()).save(path)
        rows.append(
            {
                "variant": variant,
                "prompt": pi,
                "seed": seed,
                "sampler": sampler,
                "multiplier": m,
                "path": str(path),
            }
        )
    for variant in args.variants.split(","):
        for pi in range(len(PROMPTS)):
            for seed in seeds:
                ims = [
                    Image.open(out / f"{stem((variant, pi, seed, a))}.png")
                    for a in arms
                ]
                grid = Image.new("RGB", (w * len(ims), h + 32), "white")
                d = ImageDraw.Draw(grid)
                for i, ((sampler, m), im) in enumerate(zip(arms, ims)):
                    grid.paste(im, (i * w, 32))
                    d.text(
                        (i * w + 8, 8),
                        f"{variant} p{pi} s{seed} {sampler} lora x{m}",
                        fill="black",
                    )
                suffix = (
                    ""
                    if args.arms == "euler:1.0,euler:0.0"
                    else "_" + args.arms.replace(":", "").replace(",", "_")
                )
                grid.save(out / f"grid_{variant}_p{pi}_s{seed}{suffix}.png")
    (out / "index.json").write_text(
        json.dumps({"args": vars(args), "prompts": PROMPTS, "rows": rows}, indent=2)
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
