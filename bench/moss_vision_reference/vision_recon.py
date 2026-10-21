#!/usr/bin/env python3
"""Zero-shot reconstruction through Qwen3.5's vision tower: does an image put into
the TE's user turn come back out of MOSS-Image with no adapter?

Each source image is rendered once per arm, same seed, Base, no LoRA:

    cap         caption only (the T2I baseline)
    img         image only, vision tower at the output's pixel count
    img_lo      image only, vision tower at 384² (~144 tokens)
    img_tmpl    image only, image-token rows dropped: only the chat-template rows,
                which saw the image through the LM's attention, reach the DiT
    img_cap     image + caption
    img_desc    image + "Describe this image in detail."

The TE call is the pipeline's (chat template, add_generation_prompt, enable_thinking,
last_hidden_state, no padding) with the image item in the user content and
Qwen3.5-9B's own processor settings (patch 16, merge 2, mean/std 0.5). Writes
``<out>/grid.png`` (source column first) and ``index.json`` with a 64 px luma PSNR
against the source per cell.

The vision-tower TE call is ``project/moss_lora/src/vision_te.py`` (the colorize
probes share it), put on ``sys.path`` below. Finding:
``docs/moss/findings/vision_tower_reference.md``.

    make daemon-run ARGS="--stall-timeout 900 bench/moss_vision_reference/vision_recon.py"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "project/moss_lora/src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from library.env import resolve_under_home  # noqa: E402
from library.moss.cache import caption_converter  # noqa: E402
from library.moss.generate import build_spectrum, denoise, initial_noise  # noqa: E402
from library.moss.loader import (  # noqa: E402
    compile_blocks,
    empty_cache,
    load_tokenizer,
    load_transformer,
    load_vae,
    pad_embeds,
)
from library.moss.model import decode, snapped_sigmas  # noqa: E402
from library.moss.requests import GenerateRequest  # noqa: E402
from vision_te import encode, load_te  # noqa: E402

IMAGES = [
    "post_image_dataset/resized/ningen_mame/dan_6768037.png",
    "post_image_dataset/resized/yamamoto_souichirou/13133868.png",
    "post_image_dataset/resized/sushispin/9859367.png",
    "post_image_dataset/resized/eight_tohyama/11115380.png",
]
DESCRIBE = "Describe this image in detail."
# name: (image?, vision max pixels (0 = output's), text kind, drop image rows?)
ARMS = {
    "cap": (False, 0, "caption", False),
    "img": (True, 0, "", False),
    "img_lo": (True, 384 * 384, "", False),
    "img_tmpl": (True, 0, "", True),
    "img_cap": (True, 0, "caption", False),
    "img_desc": (True, 0, DESCRIBE, False),
}


def _luma64(img):
    return np.asarray(img.convert("L").resize((64, 64), Image.BICUBIC), np.float32)


def _psnr(a, b):
    mse = float(((_luma64(a) - _luma64(b)) ** 2).mean())
    return round(10 * np.log10(255**2 / max(mse, 1e-6)), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=",".join(IMAGES))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--variant", default="base")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument(
        "--prompt_format", default="layout_json", choices=("text", "layout_json")
    )
    ap.add_argument("--out", default="output/tests/moss_vision_recon")
    args = ap.parse_args()
    images = [resolve_under_home(p) for p in args.images.split(",")]
    arms = args.arms.split(",")
    out = resolve_under_home(args.out)
    (out / "te").mkdir(parents=True, exist_ok=True)
    (out / "latents").mkdir(exist_ok=True)
    device = torch.device("cuda")
    convert = caption_converter(args.prompt_format)

    sources = {}
    for p in images:
        im = Image.open(p).convert("RGB")
        w, h = (im.width // 16) * 16, (im.height // 16) * 16
        sources[p.stem] = (
            im.crop((0, 0, w, h)),
            convert(p.with_suffix(".txt").read_text().strip()),
        )
    cells = [(s, a) for s in sources for a in arms]

    def te_path(key):
        return out / "te" / f"{key}.safetensors"

    todo = [c for c in cells if not te_path(f"{c[0]}_{c[1]}").exists()]
    if not te_path("_uncond").exists():
        todo.append(("_uncond", None))
    if todo:
        tok = load_tokenizer("")
        te = load_te(tok, device)
        for stem, arm in todo:
            if arm is None:
                hidden, n_img = encode(tok, te, None, "", 0, device)
                key = "_uncond"
            else:
                src, caption = sources[stem]
                use_img, max_px, text, drop = ARMS[arm]
                text = caption if text == "caption" else text
                hidden, n_img = encode(
                    tok,
                    te,
                    src if use_img else None,
                    text,
                    max_px or src.width * src.height,
                    device,
                    drop_rows=drop,
                )
                key = f"{stem}_{arm}"
            save_file(
                {"embeds": hidden.cpu().contiguous(), "n_img": torch.tensor(n_img)},
                str(te_path(key)),
            )
            print(
                f"[te] {key}: {hidden.shape[0]} rows, {n_img} image tokens", flush=True
            )
        del te, tok
        empty_cache()

    uncond, uncond_mask = pad_embeds(
        [load_file(str(te_path("_uncond")))["embeds"]], device
    )
    sigmas = snapped_sigmas(args.steps)
    lat = {
        c: out / "latents" / f"{c[0]}_{c[1]}_s{args.seed}.safetensors" for c in cells
    }
    pending = [c for c in cells if not lat[c].exists()]
    if pending:
        tr = load_transformer("", args.variant, device=device)
        compile_blocks(tr.layers)
        for stem, arm in pending:
            src, _ = sources[stem]
            cond, cond_mask = pad_embeds(
                [load_file(str(te_path(f"{stem}_{arm}")))["embeds"]], device
            )
            z = denoise(
                tr,
                initial_noise(args.seed, src.height, src.width).to(device),
                cond,
                cond_mask,
                uncond,
                uncond_mask,
                sigmas,
                cfg=args.cfg,
                sampler="er_sde",
                seed=args.seed,
                spectrum=build_spectrum(GenerateRequest(spectrum=True)),
            )
            save_file({"latent": z.cpu().contiguous()}, str(lat[(stem, arm)]))
            print(f"[dit] {stem} {arm} -> {lat[(stem, arm)].name}", flush=True)
        del tr
        empty_cache()

    vae = load_vae("", device=device)
    rows, results = [], []
    cell_h = 512
    for stem, (src, caption) in sources.items():
        row = [("source", src, None)]
        for arm in arms:
            png = lat[(stem, arm)].with_suffix(".png")
            if not png.exists():
                z = load_file(str(lat[(stem, arm)]))["latent"].to(device)
                x = decode(z, vae)[0].float().add(1).mul(127.5).round().clamp(0, 255)
                Image.fromarray(x.permute(1, 2, 0).byte().cpu().numpy()).save(png)
            im = Image.open(png).convert("RGB")
            te_rec = load_file(str(te_path(f"{stem}_{arm}")))
            rec = {
                "image": stem,
                "arm": arm,
                "te_rows": int(te_rec["embeds"].shape[0]),
                "image_tokens": int(te_rec["n_img"]),
                "psnr64": _psnr(im, src),
                "path": str(png),
            }
            results.append(rec)
            row.append((arm, im, rec))
        rows.append(row)
    del vae
    empty_cache()

    widths = [max(round(r[0][1].width * cell_h / r[0][1].height) for r in rows)]
    cw = widths[0]
    grid = Image.new("RGB", (cw * (len(arms) + 1), (cell_h + 36) * len(rows)), "white")
    d = ImageDraw.Draw(grid)
    for ri, row in enumerate(rows):
        y = ri * (cell_h + 36)
        for ci, (label, im, rec) in enumerate(row):
            th = im.resize(
                (round(im.width * cell_h / im.height), cell_h), Image.LANCZOS
            )
            if th.width > cw:
                th = th.resize((cw, round(th.height * cw / th.width)), Image.LANCZOS)
            grid.paste(th, (ci * cw, y + 36))
            text = (
                label
                if rec is None
                else (
                    f"{label}  rows {rec['te_rows']} img {rec['image_tokens']}  psnr64 {rec['psnr64']}"
                )
            )
            d.text((ci * cw + 6, y + 10), text, fill="black")
    grid.save(out / "grid.png")
    (out / "index.json").write_text(
        json.dumps({"args": vars(args), "results": results}, indent=2)
    )
    for r in results:
        print(
            f"{r['image']:>14} {r['arm']:>9} rows {r['te_rows']:5d} img {r['image_tokens']:5d} psnr64 {r['psnr64']}"
        )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
