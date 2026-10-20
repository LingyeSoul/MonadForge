#!/usr/bin/env python3
"""Colorize LoRAs on real B&W pages: no target, no caption, page only.

The training conditions are synthetic (EasyControl's Mangafy pass over colour
originals); this checks the ref LoRAs on pages that never went through it. Per page,
the vision-tower TE of the page alone (``embeds_plain``, as ``colorize_cache.py``
writes it) and a VAE reference at 1/``ref_scale`` of the target's sides. One row per
page: the page, then one column per LoRA; ER-SDE + Spectrum, same seed.

Alignment is the 128 px luma Pearson r against the page (``index.json``), the
held-out eval's luma r with the page in place of the target; chroma is the mean CbCr
distance from grey at 1/8 size.

    make daemon-run ARGS="project/moss_lora/src/colorize_wild.py \
        --pages a.png b.webp --loras output/ckpt_moss/x.safetensors output/ckpt_moss/y.safetensors"
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file

from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.generate import (
    _build_spectrum,
    _cached_embeds,
    denoise,
    initial_noise,
)
from library.moss.loader import (
    compile_blocks,
    empty_cache,
    load_tokenizer,
    load_transformer,
    load_vae,
    pad_embeds,
)
from library.moss.lora import load_for_inference, ref_timestep
from library.moss.model import decode, snapped_sigmas
from library.moss.model import encode as vae_encode
from library.moss.requests import GenerateRequest
from vision_te import encode, load_te


def luma_r(a: Image.Image, b: Image.Image) -> float:
    size = (128, round(128 * a.height / a.width))
    x = np.asarray(a.convert("L").resize(size, Image.LANCZOS), np.float64).ravel()
    y = np.asarray(b.convert("L").resize(size, Image.LANCZOS), np.float64).ravel()
    return float(np.corrcoef(x, y)[0, 1])


def chroma(im: Image.Image) -> float:
    small = im.resize((im.width // 8, im.height // 8), Image.BOX).convert("YCbCr")
    cbcr = np.asarray(small, np.float64)[..., 1:] - 128.0
    return float(np.linalg.norm(cbcr, axis=-1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", nargs="+", required=True)
    ap.add_argument("--loras", nargs="+", required=True)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--ref_scale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--out", default="output/tests/colorize_wild")
    args = ap.parse_args()
    out = resolve_under_home(args.out)
    for sub in ("te", "ref", "latents"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    pages = []
    for p in args.pages:
        path = resolve_under_home(p)
        image = moss_cache.load_rgb(path)
        w, h = moss_cache.target_size(*image.size, args.resolution)
        stem = path.stem.replace(" ", "_")
        pages.append((stem, moss_cache.fit(image, w, h), w, h))
    loras = [resolve_under_home(p) for p in args.loras]

    todo = [p for p in pages if not (out / "te" / f"{p[0]}.safetensors").exists()]
    if todo:
        tok = load_tokenizer("")
        te = load_te(tok, device)
        for stem, page, w, h in todo:
            plain, n_img = encode(tok, te, page, "", w * h, device)
            save_file(
                {"embeds_plain": plain.to("cpu", torch.bfloat16).contiguous()},
                str(out / "te" / f"{stem}.safetensors"),
            )
            print(
                f"[te] {stem} {w}x{h}: {plain.shape[0]} rows ({n_img} image)",
                flush=True,
            )
        del te, tok
        empty_cache()

    todo = [p for p in pages if not (out / "ref" / f"{p[0]}.safetensors").exists()]
    if todo:
        vae = load_vae("", device=device)
        for stem, page, w, h in todo:
            rw = max(16, round(w / args.ref_scale / 16) * 16)
            rh = max(16, round(h / args.ref_scale / 16) * 16)
            latent = vae_encode(
                moss_cache.to_pixels(moss_cache.fit(page, rw, rh)), vae
            )[0]
            save_file(
                {"latent": latent.cpu().contiguous()},
                str(out / "ref" / f"{stem}.safetensors"),
            )
            print(f"[ref] {stem} {w}x{h} -> {rw}x{rh}", flush=True)
        del vae
        empty_cache()

    uncond = _cached_embeds(
        "", [""], resolve_under_home("post_image_dataset/moss_prompt_te")
    )[""]
    uncond, uncond_mask = pad_embeds([uncond], device)
    sigmas = snapped_sigmas(args.steps)

    def lat(stem, lora):
        return out / "latents" / f"{stem}_{lora.stem}_s{args.seed}.safetensors"

    for lora in loras:
        todo = [p for p in pages if not lat(p[0], lora).exists()]
        if not todo:
            continue
        # A fresh DiT per LoRA: the compiled blocks trace the attached adapter.
        tr = load_transformer("", "base", device=device)
        load_for_inference(tr, lora)
        tr.ref_timestep = ref_timestep(lora)
        compile_blocks(tr.layers)
        for stem, _, w, h in todo:
            cond, cond_mask = pad_embeds(
                [load_file(str(out / "te" / f"{stem}.safetensors"))["embeds_plain"]],
                device,
            )
            ref = load_file(str(out / "ref" / f"{stem}.safetensors"))["latent"]
            z = denoise(
                tr,
                initial_noise(args.seed, h, w).to(device),
                cond,
                cond_mask,
                uncond,
                uncond_mask,
                sigmas,
                cfg=args.cfg,
                sampler="er_sde",
                seed=args.seed,
                spectrum=_build_spectrum(GenerateRequest(spectrum=True), {}),
                ref=ref[None].to(device, torch.float32),
            )
            save_file({"latent": z.cpu().contiguous()}, str(lat(stem, lora)))
            print(f"[dit] {stem} {lora.stem}", flush=True)
        del tr
        empty_cache()

    vae = load_vae("", device=device)
    rows, scores = [], {}
    for stem, page, w, h in pages:
        cells = [("page", page)]
        for lora in loras:
            png = lat(stem, lora).with_suffix(".png")
            if not png.exists():
                z = load_file(str(lat(stem, lora)))["latent"].to(device)
                x = decode(z, vae)[0].float().add(1).mul(127.5).round().clamp(0, 255)
                Image.fromarray(x.permute(1, 2, 0).byte().cpu().numpy()).save(png)
            im = Image.open(png).convert("RGB")
            cells.append((lora.stem, im))
            scores.setdefault(lora.stem, {})[stem] = {
                "luma_r": round(luma_r(page, im), 4),
                "chroma": round(chroma(im), 2),
            }
        rows.append((stem, h / w, cells))
    del vae
    empty_cache()

    cell_h = 640
    cw = max(round(cell_h / aspect) for _, aspect, _ in rows)
    ncol = 1 + len(loras)
    grid = Image.new("RGB", (cw * ncol, (cell_h + 30) * len(rows)), "white")
    d = ImageDraw.Draw(grid)
    for ri, (stem, _, cells) in enumerate(rows):
        y = ri * (cell_h + 30)
        for ci, (label, im) in enumerate(cells):
            th = im.resize(
                (round(im.width * cell_h / im.height), cell_h), Image.LANCZOS
            )
            grid.paste(th, (ci * cw, y + 30))
            d.text((ci * cw + 6, y + 8), f"{stem} {label}", fill="black")
    grid.save(out / "grid.png")
    for name, per in scores.items():
        r = np.mean([v["luma_r"] for v in per.values()])
        c = np.mean([v["chroma"] for v in per.values()])
        print(f"{name}: luma r {r:.3f}, chroma {c:.1f}", flush=True)
    (out / "index.json").write_text(
        json.dumps({"args": vars(args), "scores": scores}, indent=2)
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
