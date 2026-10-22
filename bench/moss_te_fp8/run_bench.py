#!/usr/bin/env python3
"""FP8 Qwen3.5-9B text encoder for MOSS-Image: embedding error, encode speed, images.

Every ``nn.Linear`` in the 32 decoder layers (6.92 B of the 6.92 B layer params) is
stored as ``float8_e4m3fn``; norms, the GDN ``conv1d`` / ``A_log`` / ``dt_bias``, the
token embedding (CPU, as in ``loader``) and the vision tower are left alone. Arms:

  bf16         the shipped loader (``load_text_encoder``: layers that do not fit paged)
  w8_tensor    fp8 weight, one scale per tensor, dequantised to bf16 for the matmul
  w8_row       fp8 weight, one scale per output channel, dequantised
  w8a8_tensor  per-tensor weight + per-tensor dynamic activation, ``torch._scaled_mm``
  w8a8_row     per-channel weight + per-token dynamic activation, ``torch._scaled_mm``
  <name>       ``--fp8_file <name>=<path>``: an fp8 or INT8 ConvRot file in
               ``library/moss/fp8.py``'s format (e.g. ``project/moss_lora/src/te_gptq.py``'s),
               through ``load_fp8_text_encoder``; an INT8 arm runs W8A8 (``comfy_kitchen.int8_linear``),
               or weight-only (dequantised, bf16 activations) when named in ``--w8_only``

Text: cached ``*.moss_te.safetensors`` captions sampled across
``post_image_dataset/moss/*`` (their ``embeds`` is the on-disk bf16 reference; the
fresh bf16 arm against it is the floor), ``bench/moss_spectrum`` PROMPTS and the
empty prompt. Error is per text: relative L2 over all tokens, the same with the
massive-activation channels (per-channel mean |x| ≥ 20× the median, from the bf16
reference) left out, and per-token cosine.

Images: each arm's embeds for PROMPTS (uncond = that arm's empty prompt) through
the stock sampler (Euler 50, CFG 3, skip 22, seed 42, 1024²), pixel rel L2 / PSNR
against the bf16 arm's image, and a sheet (rows prompts, columns arms).

    make daemon-run ARGS="bench/moss_te_fp8/run_bench.py"
    make daemon-run ARGS="bench/moss_te_fp8/run_bench.py --no_images --per_dir 8"
"""

from __future__ import annotations

import argparse
import gc
import glob
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "moss_spectrum"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from run_bench import PROMPTS  # noqa: E402

ARMS = ("bf16", "w8_tensor", "w8_row", "w8a8_tensor", "w8a8_row")
FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max


class FP8Linear(torch.nn.Module):
    """A bias-free ``nn.Linear`` with its weight in fp8 (``scale`` per tensor or per
    output channel); ``act`` quantises the input too and runs ``_scaled_mm``."""

    def __init__(self, lin: torch.nn.Linear, device, rowwise: bool, act: bool):
        super().__init__()
        assert lin.bias is None
        w = lin.weight.data.to(device, torch.float32)
        amax = w.abs().amax(1, keepdim=True) if rowwise else w.abs().amax()
        scale = amax.clamp(min=1e-12) / FP8_MAX
        self.weight = torch.nn.Parameter((w / scale).to(FP8), requires_grad=False)
        self.register_buffer("scale", scale)
        self.rowwise, self.act = rowwise, act
        self.err = ((self.weight.float() * scale - w).norm() / w.norm()).item()

    def forward(self, x):
        if not self.act:
            w = self.weight.to(x.dtype) * self.scale.to(x.dtype)
            return F.linear(x, w)
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float()
        amax = x2.abs().amax(1, keepdim=True) if self.rowwise else x2.abs().amax()
        xs = amax.clamp(min=1e-12) / FP8_MAX
        out = torch._scaled_mm(
            (x2 / xs).to(FP8),
            self.weight.t(),
            scale_a=xs,
            scale_b=self.scale.t() if self.rowwise else self.scale,
            out_dtype=x.dtype,
        )
        return out.reshape(*shape[:-1], out.shape[-1])


def quantize(te, device, arm):
    """Swap every decoder-layer Linear for an ``FP8Linear`` on ``device`` and put the
    rest of the language model there, embedding on the CPU. Returns the originals
    (to swap back) and the per-module weight error."""
    from library.moss.loader import _embed_on_cpu

    rowwise, act = arm.endswith("_row"), arm.startswith("w8a8")
    lm = te.language_model
    for name, child in lm.named_children():
        if name == "embed_tokens":
            if not getattr(child, "_on_cpu", False):
                _embed_on_cpu(child, device)
                child._on_cpu = True
        elif name != "layers":
            child.to(device)
    swapped, errs = [], {}
    for layer in lm.layers:
        for parent in list(layer.modules()):
            for cname, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear):
                    q = FP8Linear(child, device, rowwise, act)
                    setattr(parent, cname, q)
                    swapped.append((parent, cname, child))
                    errs.setdefault(cname, []).append(q.err)
        for p in layer.parameters():
            if p.dtype != FP8:
                p.data = p.data.to(device)
        for m in layer.modules():
            for k, b in m._buffers.items():
                if b is not None:
                    m._buffers[k] = b.to(device)
    return swapped, {k: float(np.mean(v)) for k, v in errs.items()}


def restore(te, swapped):
    for parent, cname, child in swapped:
        setattr(parent, cname, child)
    for layer in te.language_model.layers:
        layer.to("cpu")
    torch.cuda.empty_cache()


def sample_cached(per_dir: int, seed: int):
    """``[(text, ref_embeds, source)]`` from TE caches whose stored caption hashes
    to its ``caption_sha256``."""
    from library.moss.scan import caption_sha

    rng = random.Random(seed)
    out = []
    for d in sorted(glob.glob("post_image_dataset/moss/*/")):
        files = sorted(glob.glob(f"{d}*.moss_te.safetensors"))
        rng.shuffle(files)
        got = 0
        for f in files:
            with safe_open(f, "pt") as s:
                meta = s.metadata() or {}
                if meta.get("max_text_tokens") != "4096":
                    continue
                cap = meta.get("caption")
                if cap is None or caption_sha(cap) != meta.get("caption_sha256"):
                    continue
                out.append((cap, s.get_tensor("embeds").to(torch.bfloat16), f))
            got += 1
            if got >= per_dir:
                break
    return out


def compare(x, ref, massive):
    x, ref = x.float(), ref.float()
    if x.shape != ref.shape:
        return {"shape_mismatch": [list(x.shape), list(ref.shape)]}
    keep = torch.ones(ref.shape[-1], dtype=torch.bool)
    keep[massive] = False
    d = x - ref
    cos = F.cosine_similarity(x, ref, dim=-1)
    cos_ex = F.cosine_similarity(x[:, keep], ref[:, keep], dim=-1)
    return {
        "rel_l2": (d.norm() / ref.norm()).item(),
        "rel_l2_ex": (d[:, keep].norm() / ref[:, keep].norm()).item(),
        "cos_mean": cos.mean().item(),
        "cos_min": cos.min().item(),
        "cos_ex_mean": cos_ex.mean().item(),
        "cos_ex_min": cos_ex.min().item(),
    }


def encode_all(tok, te, texts, device):
    from library.moss.loader import encode_prompt

    torch.cuda.reset_peak_memory_stats()
    out, secs = {}, []
    encode_prompt(tok, te, texts[0], device)  # warm-up
    for t in texts:
        torch.cuda.synchronize()
        t0 = time.time()
        out[t] = encode_prompt(tok, te, t, device).cpu()
        torch.cuda.synchronize()
        secs.append(time.time() - t0)
    tokens = sum(v.shape[0] for v in out.values())
    stats = {
        "encode_s_total": round(sum(secs), 2),
        "encode_s_mean": round(float(np.mean(secs)), 3),
        "tokens_per_s": round(tokens / sum(secs), 1),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    return out, stats


def run_te(args, run_dir, device):
    from transformers import AutoModel

    from library.moss.loader import load_text_encoder, load_tokenizer
    from library.moss.requests import resolve_model_dir, variant_dir

    model_dir = resolve_model_dir(args.model_dir)
    cached = sample_cached(args.per_dir, args.seed)
    texts = [c[0] for c in cached] + list(PROMPTS.values()) + [""]
    texts = list(dict.fromkeys(texts))
    print(f"[texts] {len(cached)} cached captions + {len(PROMPTS)} prompts + ''")
    tok = load_tokenizer(model_dir)
    arms = [a for a in args.arms.split(",") if a]
    embeds, speed, werr = {}, {}, {}

    if "bf16" in arms:
        t0 = time.time()
        _, te = load_text_encoder(model_dir, device, 4096, tok=tok)
        speed["bf16"] = {"load_s": round(time.time() - t0, 1)}
        e, s = encode_all(tok, te, texts, device)
        speed["bf16"].update(s, paged=sum(1 for _ in _paged(te)))
        embeds["bf16"] = e
        del te
        gc.collect()
        torch.cuda.empty_cache()

    files = dict(f.split("=", 1) for f in args.fp8_file)
    w8_only = set(filter(None, args.w8_only.split(",")))
    for arm in [a for a in arms if a in files]:
        from library.moss.fp8 import load_fp8_text_encoder
        from library.moss.loader import offload_by_layer, te_reserve_gb

        t0 = time.time()
        te = load_fp8_text_encoder(files[arm], int8_act=arm not in w8_only)
        offload_by_layer(te, device, te_reserve_gb(4096))
        speed[arm] = {"load_s": round(time.time() - t0, 1)}
        e, s = encode_all(tok, te, texts, device)
        speed[arm].update(s)
        embeds[arm] = e
        print(f"[{arm}] {speed[arm]}", flush=True)
        del te
        gc.collect()
        torch.cuda.empty_cache()

    fp8_arms = [a for a in arms if a != "bf16" and a not in files]
    if fp8_arms:
        root = variant_dir(model_dir, "base")
        te = AutoModel.from_pretrained(root / "text_encoder", dtype=torch.bfloat16)
        te.eval()
        for arm in fp8_arms:
            t0 = time.time()
            swapped, werr[arm] = quantize(te, device, arm)
            torch.cuda.synchronize()
            speed[arm] = {
                "quantize_s": round(time.time() - t0, 1),
                "resident_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            }
            e, s = encode_all(tok, te, texts, device)
            speed[arm].update(s)
            embeds[arm] = e
            print(f"[{arm}] {speed[arm]}", flush=True)
            restore(te, swapped)
        del te
        torch.cuda.empty_cache()

    for arm, e in embeds.items():
        save_file(
            {f"t{i}": e[t].contiguous() for i, t in enumerate(texts)},
            run_dir / f"embeds_{arm}.safetensors",
        )

    # References: the on-disk cache where there is one, else the fresh bf16 arm.
    refs = {c[0]: (c[1], "cache") for c in cached}
    if "bf16" in embeds:
        for t in texts:
            refs.setdefault(t, (embeds["bf16"][t], "bf16"))
    pooled = torch.cat([r.float().abs() for r, _ in refs.values()]).mean(0)
    massive = torch.nonzero(pooled >= 20 * pooled.median()).flatten().tolist()
    print(f"[massive] channels {massive}")

    rows = []
    for i, t in enumerate(texts):
        ref, kind = refs[t]
        kind_text = (
            "prompt" if t in PROMPTS.values() else ("empty" if not t else "caption")
        )
        for arm, e in embeds.items():
            if arm == "bf16" and kind == "bf16":
                continue
            rows.append(
                {
                    "text": i,
                    "kind": kind_text,
                    "ref": kind,
                    "arm": arm,
                    "tokens": int(ref.shape[0]),
                    **compare(e[t], ref, massive),
                }
            )
    summary = {}
    for arm in embeds:
        for kind in ("caption", "prompt", "empty", "all"):
            sel = [
                r
                for r in rows
                if r["arm"] == arm
                and "rel_l2" in r
                and (kind == "all" or r["kind"] == kind)
            ]
            if not sel:
                continue
            summary[f"{arm}/{kind}"] = {
                "n": len(sel),
                **{
                    k: round(float(np.mean([r[k] for r in sel])), 5)
                    for k in ("rel_l2", "rel_l2_ex", "cos_mean", "cos_ex_mean")
                },
                "rel_l2_max": round(max(r["rel_l2"] for r in sel), 5),
                "cos_ex_min": round(min(r["cos_ex_min"] for r in sel), 5),
            }
    for k, v in summary.items():
        print(f"[err] {k:24s} {v}")
    return (
        texts,
        embeds,
        {
            "summary": summary,
            "speed": speed,
            "weight_rel_err": werr,
            "massive_channels": massive,
            "rows": rows,
        },
    )


def _paged(te):
    for layer in te.language_model.layers:
        if layer._forward_pre_hooks:
            yield layer


def run_images(args, run_dir, texts, embeds, device):
    from diffusers.image_processor import VaeImageProcessor

    from library.moss.generate import denoise, initial_noise
    from library.moss.loader import (
        compile_blocks,
        load_transformer,
        load_vae,
        pad_embeds,
    )
    from library.moss.model import decode, snapped_sigmas
    from library.moss.requests import resolve_model_dir

    model_dir = resolve_model_dir(args.model_dir)
    arms = [a for a in args.image_arms.split(",") if a in embeds]
    tr = load_transformer(model_dir, "base", device=device)
    compile_blocks(tr.layers)
    sigmas = snapped_sigmas(50)
    lat_dir = run_dir / "latents"
    lat_dir.mkdir(exist_ok=True)
    for arm in arms:
        uncond, umask = pad_embeds([embeds[arm][""]], device=device)
        for key, prompt in PROMPTS.items():
            path = lat_dir / f"{key}_{arm}.safetensors"
            if path.is_file():
                continue
            cond, cmask = pad_embeds([embeds[arm][prompt]], device=device)
            noise = initial_noise(args.seed, 1024, 1024).to(device)
            t0 = time.time()
            lat = denoise(tr, noise, cond, cmask, uncond, umask, sigmas)
            save_file({"latent": lat.cpu().contiguous()}, path)
            print(f"[dit] {key} {arm} {time.time() - t0:.1f}s", flush=True)
    tr.to("cpu")
    del tr
    torch.cuda.empty_cache()

    vae = load_vae(model_dir, device=device)
    proc = VaeImageProcessor(vae_scale_factor=16)
    img_dir = run_dir / "images"
    img_dir.mkdir(exist_ok=True)
    pixels = {}
    for arm in arms:
        for key in PROMPTS:
            lat = load_file(lat_dir / f"{key}_{arm}.safetensors")["latent"].to(device)
            with torch.no_grad():
                img = proc.postprocess(decode(lat, vae), output_type="pil")[0]
            img.save(img_dir / f"{key}_{arm}.png")
            pixels[(key, arm)] = np.asarray(img, dtype=np.float32) / 255.0
    del vae
    torch.cuda.empty_cache()

    rows, summary = [], {}
    for arm in arms:
        if arm == "bf16":
            continue
        for key in PROMPTS:
            a, b = pixels[(key, arm)], pixels[(key, "bf16")]
            mse = float(np.mean((a - b) ** 2))
            rows.append(
                {
                    "prompt": key,
                    "arm": arm,
                    "rel_l2": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                    "psnr": float(10 * np.log10(1 / max(mse, 1e-12))),
                }
            )
        sel = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            k: round(float(np.mean([r[k] for r in sel])), 4) for k in ("rel_l2", "psnr")
        }
        print(f"[img] {arm} vs bf16 {summary[arm]}")

    from PIL import Image

    cell = 384
    sheet = Image.new("RGB", (cell * len(arms), cell * len(PROMPTS)), "white")
    for r, key in enumerate(PROMPTS):
        for c, arm in enumerate(arms):
            im = Image.open(img_dir / f"{key}_{arm}.png").resize((cell, cell))
            sheet.paste(im, (c * cell, r * cell))
    sheet.save(run_dir / "sheet.jpg", quality=90)
    print(f"[sheet] columns {arms} -> {run_dir / 'sheet.jpg'}")
    return {"summary": summary, "rows": rows, "arms": arms}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--image_arms", default=",".join(ARMS))
    p.add_argument(
        "--fp8_file", action="append", default=[], help="name=path, repeatable"
    )
    p.add_argument(
        "--w8_only", default="", help="INT8 file arms run weight-only (comma list)"
    )
    p.add_argument("--per_dir", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_images", action="store_true")
    p.add_argument("--model_dir", default="")
    p.add_argument("--label", default=None)
    args = p.parse_args()

    start_heartbeat()
    device = torch.device("cuda")
    run_dir = make_run_dir("moss_te_fp8", args.label)
    texts, embeds, te_metrics = run_te(args, run_dir, device)
    metrics = {"te": te_metrics}
    if not args.no_images:
        metrics["images"] = run_images(args, run_dir, texts, embeds, device)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        device=device,
    )
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()
