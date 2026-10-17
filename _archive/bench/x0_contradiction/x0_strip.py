#!/usr/bin/env python3
"""x̂₀ filmstrip — decode the running clean-latent prediction at every step.

Makes the "in-step contradiction" visible: decode x̂₀ = x_t − σ_i·v at each
denoising step, crop to the face, and tile base (top) vs base+LoRA (bottom) so
the same column = same step. If a region wobbles across columns (e.g. eyes
open→weird→reform) that's backtracking (contradiction); if it snaps to its final
look early and holds, it's a committed mode. Answers "is this awkward feature a
contradiction or a stable-but-bad draw?" for one specific image.

Usage::

    python bench/x0_contradiction/x0_strip.py \
        --prompt "1girl, waving hello, smiling, looking at viewer" \
        --seed 0 --steps 24 --adapter output/ckpt/anima_sincos.safetensors
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir  # noqa: E402

import anima_lora  # noqa: E402
import library.inference.sampling as _sampling  # noqa: E402
from library.runtime.harness import build_inference_bundle  # noqa: E402

# face/eye region as fractions of (W,H) — tuned for these portrait comps.
FACE_BOX = (0.30, 0.07, 0.70, 0.46)
_CAP: list[tuple[int, float, torch.Tensor]] = []


@contextmanager
def capture_full():
    """Record the full 5D x̂₀ (1,C,1,H,W) at every euler step."""
    _CAP.clear()
    orig = _sampling.step

    def patched(latents, noise_pred, sigmas, i):
        x0 = latents.float() - sigmas[i] * noise_pred.float()
        _CAP.append((int(i), float(sigmas[i]), x0.detach().to("cpu")))
        return orig(latents, noise_pred, sigmas, i)

    _sampling.step = patched
    try:
        yield _CAP
    finally:
        _sampling.step = orig


def make_args(prompt, seed, a, with_lora):
    ckpt = anima_lora.default_checkpoints()
    req = anima_lora.GenerationRequest(
        dit=ckpt.dit,
        vae=ckpt.vae,
        text_encoder=ckpt.text_encoder,
        prompt=prompt,
        image_size=(a.size, a.size),
        infer_steps=a.steps,
        guidance_scale=a.cfg,
        flow_shift=a.flow_shift,
        sampler="euler",
        seed=seed,
        lora_weight=[a.adapter] if with_lora else None,
        lora_multiplier=[a.multiplier] if with_lora else None,
        extra_argv=("--compile_blocks",) if getattr(a, "compile", False) else (),
    )
    return req.to_args()


def face_crop(img: Image.Image, cell: int) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = FACE_BOX
    crop = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    return crop.resize((cell, cell), Image.LANCZOS)


def decode_strip(
    bundle, captured, a, cell: int
) -> tuple[list[Image.Image], list[float]]:
    steps = sorted(captured, key=lambda r: r[0])
    frames, sigmas = [], []
    for _, sig, x0 in steps:
        latent = x0.to(bundle.device)
        img = anima_lora.decode_to_pil(bundle.vae, latent, bundle.device)
        frames.append(face_crop(img, cell))
        sigmas.append(sig)
    return frames, sigmas


def tile(
    rows: list[tuple[str, list[Image.Image]]], sigmas: list[float], cell: int
) -> Image.Image:
    n = len(sigmas)
    pad, top, left = 4, 18, 64
    W = left + n * (cell + pad) + pad
    H = top + len(rows) * (cell + pad) + pad
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for c, sig in enumerate(sigmas):
        x = left + c * (cell + pad) + pad
        mark = "*" if sig < 0.45 else ""  # * = resolve tail
        draw.text((x, 2), f"{sig:.2f}{mark}", fill=(0, 0, 0))
    for r, (label, frames) in enumerate(rows):
        y = top + r * (cell + pad) + pad
        draw.text((4, y + cell // 2), label, fill=(0, 0, 0))
        for c, fr in enumerate(frames):
            x = left + c * (cell + pad) + pad
            canvas.paste(fr, (x, y))
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompt", default="1girl, waving hello, smiling, looking at viewer"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adapter", default="output/ckpt/anima_sincos.safetensors")
    p.add_argument("--multiplier", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow-shift", dest="flow_shift", type=float, default=3.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--cell", type=int, default=128)
    p.add_argument("--label", default=None)
    p.add_argument(
        "--compile",
        action="store_true",
        help="compile_blocks the DiT (fixed-res, static graph)",
    )
    a = p.parse_args()

    if not Path(a.adapter).is_absolute():
        a.adapter = str(REPO_ROOT / a.adapter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir("x0_contradiction", label=a.label or "x0strip")
    print(f"run dir: {run_dir}\nprompt: {a.prompt}\nseed: {a.seed}")

    rows = []
    sigmas_ref = None
    for with_lora, name in ((False, "base"), (True, "lora")):
        print(f"  generating {name}...")
        bundle = build_inference_bundle(
            make_args(a.prompt, a.seed, a, with_lora), device=device, with_vae=True
        )
        with capture_full() as cap:
            bundle.generate()
        frames, sigmas = decode_strip(bundle, cap, a, a.cell)
        sigmas_ref = sigmas
        rows.append((name, frames))
        del bundle
        torch.cuda.empty_cache()

    out = tile(rows, sigmas_ref, a.cell)
    path = run_dir / f"x0strip_seed{a.seed}.png"
    out.save(path)
    print(f"\nsaved: {path}")
    print("columns = denoising steps (σ shown; * = resolve tail σ<0.45).")
    print("top=base, bottom=lora. Watch the eyes across columns:")
    print("  wobble/reform -> contradiction;  snap-early-and-hold -> committed mode.")


if __name__ == "__main__":
    main()
