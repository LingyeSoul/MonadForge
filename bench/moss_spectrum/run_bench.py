#!/usr/bin/env python3
"""MOSS-Image Spectrum bench — wall-clock and fidelity vs the plain 50-step sampler.

Every arm runs through ``library.moss.generate.denoise`` (the shipped path) at a
fixed seed per prompt. Arm spec: ``p<N>`` = plain N steps, ``s<N>[:k=v,…]`` =
Spectrum at N steps with ``MossSpectrum`` overrides (``ws`` window_size, ``fw``
flex_window, ``warm`` warmup, ``stop`` stop_caching_step, ``w``, ``m``, ``lam``,
``K`` history, ``cal`` calibration); add ``b=1`` to either for the batched
cond+uncond forward.

    make daemon-run ARGS="bench/moss_spectrum/run_bench.py --label base3 \
        --models base --prompts anime_sunset,danbooru_tags,ja_manga"

Per arm: mean denoise seconds, speed-up vs the first arm (``p50`` by default),
full-forward count, denoise peak VRAM, and the final latent's relative L2 to the
reference arm's. Full PNGs land under ``--image_root``; per (model, prompt) a
side-by-side grid of every arm lands in the run dir.

``--decode_only <image dir>`` decodes saved latents and writes the envelope with
no DiT.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402

# The 2026-10-16 first-look prompts (project/moss_lora/plan.md).
PROMPTS = {
    "anime_sunset": "An anime illustration of a girl with long silver hair and red eyes, wearing a black "
    "school uniform, sitting on a classroom windowsill at sunset, cherry blossom petals drifting "
    "outside the window, detailed cel shading, soft rim light.",
    "danbooru_tags": "1girl, solo, akita neru, yellow eyes, blonde hair, side ponytail, gray shirt, "
    "holding phone, looking at viewer, upper body, simple background, white background",
    "zh_fridge": "A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky "
    'note on the refrigerator door is handwritten in Chinese:\n"记得买牛奶和鸡蛋！\n牛奶两盒，鸡蛋一打。\n顺便带一袋面包。"',
    "ja_manga": "A black-and-white manga panel of a boy with messy hair shouting toward the reader, speed "
    'lines in the background, with a large speech bubble that reads "待ってくれ！" in Japanese.',
    "ko_neon": "A cozy café storefront on a rainy night, with a pink neon sign above the door that reads "
    '"오늘도 수고했어" in Korean, reflections on the wet street.',
}
DEFAULT_ARMS = "p50,p30,p20,s50,s50:fw=1.0,s30"
_KNOBS = {
    "ws": ("window_size", float),
    "fw": ("flex_window", float),
    "warm": ("warmup", int),
    "stop": ("stop_caching_step", int),
    "w": ("w", float),
    "m": ("m", int),
    "lam": ("lam", float),
    "K": ("history", int),
    "cal": ("calibration", float),
}


def parse_arm(spec: str) -> dict:
    head, _, rest = spec.partition(":")
    if head[0] not in "ps" or not head[1:].isdigit():
        raise SystemExit(f"bad arm {spec!r}: want p<N> or s<N>[:k=v,…]")
    arm = dict(name=spec, steps=int(head[1:]), spectrum=head[0] == "s", batched=False, knobs={})
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        if k == "b":
            arm["batched"] = bool(int(v))
        elif k in _KNOBS and arm["spectrum"]:
            name, kind = _KNOBS[k]
            arm["knobs"][name] = kind(v)
        else:
            raise SystemExit(f"bad knob {k!r} in {spec!r}")
    return arm


def _grid(tiles, out_path, tile=512):
    """tiles: [(label, PIL image)] side by side, each scaled to ``tile`` px high."""
    scaled = [(lbl, im.resize((round(im.width * tile / im.height), tile))) for lbl, im in tiles]
    width = sum(im.width for _, im in scaled)
    grid = Image.new("RGB", (width, tile + 28), "white")
    draw = ImageDraw.Draw(grid)
    x = 0
    for lbl, im in scaled:
        grid.paste(im, (x, 28))
        draw.text((x + 6, 6), lbl, fill="black")
        x += im.width
    grid.save(out_path)


def _slug(arm_name: str) -> str:
    return arm_name.replace(":", "_").replace(",", "_")


def _latent_path(image_dir: Path, model: str, arm: str, key: str) -> Path:
    return image_dir / "latents" / f"{model}__{_slug(arm)}__{key}.safetensors"


def save_latent(path: Path, z: torch.Tensor, row: dict) -> None:
    """One finished image: the fp32 latent + its row (seconds, forwards, peak) as
    safetensors metadata. Written via a temp file, so a kill never leaves a stub."""
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file({"latent": z.contiguous()}, str(tmp), metadata={"row": json.dumps(row)})
    tmp.replace(path)


def load_latent(path: Path) -> tuple[torch.Tensor, dict]:
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        return f.get_tensor("latent"), json.loads(f.metadata()["row"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="base", help="comma list of base,sft")
    ap.add_argument("--prompts", default="anime_sunset,danbooru_tags,ja_manga", help="keys of PROMPTS")
    ap.add_argument("--arms", default=DEFAULT_ARMS, help="first arm is the reference")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--label", default=None, help="also names the image dir, so a rerun with the same label resumes")
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--image_root", default="/media/sorryhyun/data/moss_image/bench")
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile every block (dynamic shapes) before the warm-up; the warm-up time is reported apart",
    )
    ap.add_argument(
        "--decode_only",
        default="",
        help="an existing image dir: decode its saved latents and write the envelope, no DiT",
    )
    args = ap.parse_args()
    start_heartbeat(label="moss_spectrum")

    from diffusers.image_processor import VaeImageProcessor

    from library.moss.generate import _cached_embeds, denoise, initial_noise
    from library.moss.loader import compile_blocks, empty_cache, load_transformer, load_vae, pad_embeds
    from library.moss.model import DEFAULT_UNCOND_SKIP, decode, snapped_sigmas
    from library.moss.requests import resolve_model_dir
    from library.moss.spectrum import MossSpectrum

    arms = [parse_arm(a) for a in filter(None, args.arms.split(","))]
    keys = [k for k in args.prompts.split(",") if k]
    models = [m for m in args.models.split(",") if m]
    model_dir = resolve_model_dir(args.model_dir)
    run_dir = make_run_dir("moss_spectrum", label=args.label)
    if args.decode_only:
        image_dir = Path(args.decode_only)
    else:
        image_dir = Path(args.image_root) / (args.label or run_dir.name)
    image_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    h = w = args.size

    todo = {
        model: [(arm, key) for arm in arms for key in keys
                if not _latent_path(image_dir, model, arm["name"], key).is_file()]
        for model in models
    }
    if args.decode_only and any(todo.values()):
        raise SystemExit(f"--decode_only: {sum(map(len, todo.values()))} latents missing in {image_dir}")
    if any(todo.values()):
        # TE outputs are cached per text on disk too (_cached_embeds).
        embeds = _cached_embeds(model_dir, [PROMPTS[k] for k in keys] + [""], Path(args.image_root) / "_te_cache")
        uncond, uncond_mask = pad_embeds([embeds[""]], device=device)
    warmup_s = {}
    for model in models:
        if not todo[model]:
            print(f"[{model}] all {len(arms) * len(keys)} latents on disk, skipping the DiT", flush=True)
            continue
        print(f"[{model}] {len(todo[model])} to render, {len(arms) * len(keys) - len(todo[model])} resumed", flush=True)
        transformer = load_transformer(model_dir, model, device=device)
        if args.compile:
            compile_blocks(transformer.layers)
        # Warm the kernels (and the compile) so the first timed image is not the slow one.
        cond, cond_mask = pad_embeds([embeds[PROMPTS[keys[0]]]], device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        denoise(transformer, initial_noise(0, h, w).to(device), cond, cond_mask, uncond, uncond_mask,
                snapped_sigmas(50)[:2], cfg=args.cfg, uncond_skip=DEFAULT_UNCOND_SKIP)
        torch.cuda.synchronize()
        warmup_s[model] = round(time.perf_counter() - t0, 2)
        print(f"[{model}] warm-up {warmup_s[model]}s (compile={args.compile})", flush=True)
        for arm, key in todo[model]:
            sigmas = snapped_sigmas(arm["steps"])
            cond, cond_mask = pad_embeds([embeds[PROMPTS[key]]], device=device)
            spectrum = MossSpectrum(**arm["knobs"]) if arm["spectrum"] else None
            noise = initial_noise(args.seed, h, w).to(device)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            z = denoise(transformer, noise, cond, cond_mask, uncond, uncond_mask, sigmas,
                        cfg=args.cfg, uncond_skip=DEFAULT_UNCOND_SKIP, batched=arm["batched"],
                        spectrum=spectrum)
            torch.cuda.synchronize()
            secs = time.perf_counter() - t0
            full = spectrum.full_steps if spectrum else len(sigmas)
            row = dict(model=model, arm=arm["name"], prompt=key, steps=len(sigmas),
                       full_forwards=full, seconds=round(secs, 2),
                       peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            save_latent(_latent_path(image_dir, model, arm["name"], key), z.cpu(), row)
            print(f"[{model}] {arm['name']:>16} {key:<14} {secs:6.1f}s full {full:2d}/{len(sigmas)} "
                  f"peak {row['peak_gib']} GiB", flush=True)
        transformer.to("cpu")
        del transformer
        empty_cache()

    rows, latents = [], {}
    for model in models:
        for arm in arms:
            for key in keys:
                z, row = load_latent(_latent_path(image_dir, model, arm["name"], key))
                latents[(model, arm["name"], key)] = z
                rows.append(row)
    ref = arms[0]["name"]
    for r in rows:
        a = latents[(r["model"], r["arm"], r["prompt"])]
        b = latents[(r["model"], ref, r["prompt"])]
        r["rel_l2_vs_ref"] = round(float((a - b).norm() / b.norm()), 4)

    vae = load_vae(model_dir, device=device)
    proc = VaeImageProcessor(vae_scale_factor=16)
    artifacts = []
    for model in models:
        for key in keys:
            tiles = []
            for arm in arms:
                png = image_dir / f"{model}_{key}_{_slug(arm['name'])}.png"
                if png.is_file():
                    im = Image.open(png).convert("RGB")
                else:
                    z = latents[(model, arm["name"], key)].to(device)
                    im = proc.postprocess(decode(z, vae), output_type="pil")[0]
                    im.save(png)
                r = next(r for r in rows if (r["model"], r["arm"], r["prompt"]) == (model, arm["name"], key))
                tiles.append((f"{arm['name']}  {r['seconds']}s  fwd {r['full_forwards']}", im))
            grid = run_dir / f"grid_{model}_{key}.png"
            _grid(tiles, grid)
            artifacts.append(grid.name)

    summary = {}
    for model in models:
        base_s = None
        for arm in arms:
            sel = [r for r in rows if r["model"] == model and r["arm"] == arm["name"]]
            mean_s = sum(r["seconds"] for r in sel) / len(sel)
            base_s = base_s or mean_s
            summary[f"{model}/{arm['name']}"] = dict(
                mean_seconds=round(mean_s, 2),
                speedup=round(base_s / mean_s, 2),
                full_forwards=sel[0]["full_forwards"],
                steps=sel[0]["steps"],
                peak_gib=max(r["peak_gib"] for r in sel),
                mean_rel_l2_vs_ref=round(sum(r["rel_l2_vs_ref"] for r in sel) / len(sel), 4),
            )
    for k, v in summary.items():
        print(f"{k:>24}: {v}", flush=True)
    write_result(run_dir, script=__file__, args=args, label=args.label, artifacts=artifacts,
                 metrics=dict(summary=summary, rows=rows, image_dir=str(image_dir),
                              compile=args.compile, warmup_seconds=warmup_s))
    print(f"result -> {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
