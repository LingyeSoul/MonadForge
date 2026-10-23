#!/usr/bin/env python3
"""Run colorize LoRAs on real (not mangafied) B&W pages: no target, no caption.

Per page, the fp8 vision-tower TE of the page alone (``embeds_plain``, as
``colorize_cache.py`` writes it) and a VAE reference at 1/``ref_scale`` of the
target's sides. Grid: one row per page, one column per LoRA, same seed.

``index.json`` scores alignment as the 128 px luma Pearson r against the page, and
chroma as the mean CbCr distance from grey at 1/8 size. Page TE and ref files are
named by source path digest and size, the latents by a digest of the LoRA
(path + mtime), those files, steps, cfg and sampler, so a rerun resumes only what
matches.

    make daemon-run ARGS="project/moss_lora/src/colorize_wild.py \
        --pages a.png b.webp --loras output/ckpt_moss/x.safetensors output/ckpt_moss/y.safetensors"
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

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
from library.moss import cache as moss_cache
from library.moss.loader import empty_cache, load_tokenizer, load_vae
from library.moss.model import encode as vae_encode
from library.moss.model import snapped_sigmas
from library.moss.vision_te import encode, load_te


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

    # (stem, page, w, h, ref w, ref h, source digest)
    pages = []
    for p in args.pages:
        path = resolve_under_home(p)
        image = moss_cache.load_rgb(path)
        w, h = moss_cache.target_size(*image.size, args.resolution)
        rw = max(16, round(w / args.ref_scale / 16) * 16)
        rh = max(16, round(h / args.ref_scale / 16) * 16)
        stem = path.stem.replace(" ", "_")
        src = settings_tag(src=str(path))
        pages.append((stem, moss_cache.fit(image, w, h), w, h, rw, rh, src))
    loras = [resolve_under_home(p) for p in args.loras]

    def te_path(pg):
        return out / "te" / f"{pg[0]}_{pg[6]}_{pg[2]}x{pg[3]}.safetensors"

    def ref_path(pg):
        return out / "ref" / f"{pg[0]}_{pg[6]}_{pg[4]}x{pg[5]}.safetensors"

    todo = [pg for pg in pages if not te_path(pg).exists()]
    if todo:
        tok = load_tokenizer("")
        te = load_te(tok, device, precision="fp8")
        for pg in todo:
            stem, page, w, h, _, _, _ = pg
            plain, n_img = encode(tok, te, page, "", w * h, device)
            save_file(
                {"embeds_plain": plain.to("cpu", torch.bfloat16).contiguous()},
                str(te_path(pg)),
            )
            print(
                f"[te] {stem} {w}x{h}: {plain.shape[0]} rows ({n_img} image)",
                flush=True,
            )
        del te, tok
        empty_cache()

    todo = [pg for pg in pages if not ref_path(pg).exists()]
    if todo:
        vae = load_vae("", device=device)
        for pg in todo:
            stem, page, w, h, rw, rh, _ = pg
            latent = vae_encode(
                moss_cache.to_pixels(moss_cache.fit(page, rw, rh)), vae
            )[0]
            save_file({"latent": latent.cpu().contiguous()}, str(ref_path(pg)))
            print(f"[ref] {stem} {w}x{h} -> {rw}x{rh}", flush=True)
        del vae
        empty_cache()

    _, uncond, uncond_mask = uncond_embeds(device, precision="fp8")
    sigmas = snapped_sigmas(args.steps)
    spectrum = spectrum_settings()

    def lat(pg, lora):
        tag = settings_tag(
            lora=lora_id(lora),
            te=te_path(pg).name,
            ref=ref_path(pg).name,
            steps=args.steps,
            cfg=args.cfg,
            sampler="er_sde",
            spectrum=spectrum,
        )
        return out / "latents" / f"{pg[0]}_{lora.stem}_s{args.seed}_{tag}.safetensors"

    for lora in loras:
        todo = [pg for pg in pages if not lat(pg, lora).exists()]
        if not todo:
            continue
        tr, _ = load_dit(lora, device)
        for pg in todo:
            stem, _, w, h, _, _, _ = pg
            z = render(
                tr,
                load_file(str(te_path(pg)))["embeds_plain"],
                uncond,
                uncond_mask,
                sigmas,
                seed=args.seed,
                height=h,
                width=w,
                cfg=args.cfg,
                ref=load_file(str(ref_path(pg)))["latent"][None],
            )
            save_latent(z, lat(pg, lora))
            print(f"[dit] {stem} {lora.stem}", flush=True)
        del tr
        empty_cache()

    vae = load_vae("", device=device)
    rows, scores = [], {}
    for pg in pages:
        stem, page = pg[0], pg[1]
        cells = [("page", page)]
        for lora in loras:
            im = decoded(lat(pg, lora), vae, device)
            cells.append((lora.stem, im))
            scores.setdefault(lora.stem, {})[stem] = {
                "luma_r": round(luma_r(page, im), 4),
                "chroma": round(chroma(im), 2),
            }
        rows.append((stem, cells))
    del vae
    empty_cache()

    grid(rows).save(out / "grid.png")
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
