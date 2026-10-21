#!/usr/bin/env python3
"""Corrector A/B: CFG against correction on the same denoiser LoRA.

An arm is ``<cfg>:<K>`` — CFG scale and corrections per step. ``K`` corrects the
cond branch before CFG (``correction.corrected_velocity``); the corrector never
trained on the empty prompt, so the uncond branch is left alone. Forwards per
step: ``1 + K`` (+1 when cfg ≠ 1), so ``1:1`` costs what ``3:0`` does. ER-SDE on
the snapped shift-5 grid, uncond skip 22, no Spectrum (its forecast has no
corrected counterpart), blocks compiled. Prompts and seeds are ``lora_ab.py``'s;
latents are cached under a digest of both LoRAs, prompts and sampler settings.

    make daemon-run ARGS="project/moss_lora/src/correction_ab.py \\
        --denoiser output/ckpt_moss/<floor>.safetensors \\
        --corrector output/ckpt_moss/<corrector>.safetensors"
"""

import argparse

import torch

from _render import decoded, grid, lora_id, save_latent, settings_tag, uncond_embeds
from correction import CorrectorLoRA, corrected_velocity
from lora_ab import PROMPTS
from library.env import resolve_under_home
from library.inference.sampling import ERSDESampler
from library.moss.cache import caption_converter
from library.moss.generate import initial_noise
from library.moss.loader import (
    compile_blocks,
    empty_cache,
    load_transformer,
    load_vae,
    pad_embeds,
)
from library.moss.lora import load_for_inference
from library.moss.model import snapped_sigmas
from library.moss.requests import DEFAULT_UNCOND_SKIP


@torch.no_grad()
def sample(tr, net, z, cond, cond_mask, uncond, uncond_mask, sigmas, *, cfg, k, seed):
    device = z.device
    sig = torch.cat([sigmas.to(device, torch.float32), torch.zeros(1, device=device)])
    er_sde = ERSDESampler(sig, seed=seed, device=device)
    skip = tuple(DEFAULT_UNCOND_SKIP)
    for i, t in enumerate(sig[:-1] * 1000):
        v = tr(z, t, cond, cond_mask, return_dict=False)[0].float()
        v = corrected_velocity(tr, net, z, sig[i], t, cond, cond_mask, v, k)
        if cfg != 1.0:
            vu = tr(z, t, uncond, uncond_mask, return_dict=False, skip_layers=skip)[
                0
            ].float()
            v = vu + cfg * (v - vu)
        z = er_sde.step(z, z - sig[i] * v, i)
    return z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoiser", required=True)
    ap.add_argument("--corrector", required=True)
    ap.add_argument("--artist", default="@channel (caststation)")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument(
        "--arms", default="3:0,1:0,1:1,1:2,1:4", help="<cfg>:<K>, comma-separated"
    )
    ap.add_argument(
        "--prompt_format", default="layout_json", choices=("text", "layout_json")
    )
    ap.add_argument("--out", default="", help="default: output/tests/corr_<corrector>")
    args = ap.parse_args()
    w, h = map(int, args.size.split("x"))
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [
        (float(a.split(":")[0]), int(a.split(":")[1])) for a in args.arms.split(",")
    ]
    denoiser = resolve_under_home(args.denoiser)
    corrector = resolve_under_home(args.corrector)
    out = resolve_under_home(args.out or f"output/tests/corr_{corrector.stem}")
    lat_dir = out / "latents"
    lat_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    prompts = [
        caption_converter(args.prompt_format)(p.format(artist=args.artist))
        for p in PROMPTS
    ]
    embeds, uncond, uncond_mask = uncond_embeds(device, prompts)
    sigmas = snapped_sigmas(args.steps)
    tag = settings_tag(
        denoiser=lora_id(denoiser),
        corrector=lora_id(corrector),
        prompts=prompts,
        steps=args.steps,
        size=args.size,
        variant=args.variant,
        sampler="er_sde",
    )
    jobs = [(pi, s, a) for pi in range(len(prompts)) for s in seeds for a in arms]

    def stem(job):
        pi, s, (cfg, k) = job
        return f"p{pi}_s{s}_cfg{cfg:g}_k{k}_{tag}"

    todo = [j for j in jobs if not (lat_dir / f"{stem(j)}.safetensors").exists()]
    if todo:
        tr = load_transformer("", args.variant, device=device)
        load_for_inference(tr, denoiser)
        net = CorrectorLoRA.load(tr, corrector)
        g = net.gamma()
        print(
            f"γ mean {g.mean().item():.3f} min {g.min().item():.3f} "
            f"max {g.max().item():.3f}",
            flush=True,
        )
        compile_blocks(tr.layers)
        for job in todo:
            pi, seed, (cfg, k) = job
            cond, cond_mask = pad_embeds([embeds[prompts[pi]]], device)
            z = sample(
                tr,
                net,
                initial_noise(seed, h, w).to(device),
                cond,
                cond_mask,
                uncond,
                uncond_mask,
                sigmas,
                cfg=cfg,
                k=k,
                seed=seed,
            )
            save_latent(z, lat_dir / f"{stem(job)}.safetensors")
            print(f"p{pi} s{seed} cfg {cfg:g} K {k} -> {stem(job)}", flush=True)
        del net, tr
        empty_cache()

    vae = load_vae("", device=device)
    rows = []
    for pi in range(len(prompts)):
        for seed in seeds:
            cells = []
            for arm in arms:
                job = (pi, seed, arm)
                png = out / f"{stem(job)}.png"
                im = decoded(lat_dir / f"{stem(job)}.safetensors", vae, device, png)
                cells.append((f"cfg {arm[0]:g} K {arm[1]}", im))
            rows.append((f"p{pi} s{seed}", cells))
    sheet = out / f"grid_{tag}.jpg"
    grid(rows).save(sheet, quality=90)
    print(f"wrote {sheet}", flush=True)


if __name__ == "__main__":
    main()
