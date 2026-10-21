#!/usr/bin/env python3
"""Phase 0 gate: library.moss vs the reference MOSSImagePipeline.

1. TE: ``loader.encode_prompt`` vs ``MOSSImagePipeline.encode_prompt`` on the same
   loaded Qwen3.5-9B (expect bit-exact).
2. Reference pipeline (its own ``MOSSImageTransformer``, ``output_type='latent'``)
   from fixed embeds + fixed fp32 noise.
3. Vendored DiT + ``generate.denoise`` unbatched (expect bit-exact) and batched
   (max|Δ| / mean|Δ|), plus wall-clock per image for all three.

Writes ``result.json`` (the bench envelope, the report under ``metrics``) and the
final latents (``{i}_{reference,unbatched,batched}.pt``) into
``bench/moss_parity/results/<ts>[-label]/``.

    make daemon-run ARGS="bench/moss_parity/parity.py"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import torch  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from library.moss.generate import denoise, initial_noise  # noqa: E402
from library.moss.loader import (  # noqa: E402
    empty_cache,
    encode_prompt,
    load_text_encoder,
    load_transformer,
    pad_embeds,
)
from library.moss.model import snapped_sigmas  # noqa: E402
from library.moss.requests import variant_dir  # noqa: E402

PROMPTS = [
    "An anime illustration of a girl with long silver hair and red eyes, wearing a black "
    "school uniform, sitting on a classroom windowsill at sunset, cherry blossom petals drifting "
    "outside the window, detailed cel shading, soft rim light.",
    "A black-and-white manga panel of a boy with messy hair shouting toward the reader, speed "
    'lines in the background, with a large speech bubble that reads "待ってくれ！" in Japanese.',
]


def _reference_module(model_dir):
    path = variant_dir(model_dir, "base") / "pipeline_moss_image.py"
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _delta(a, b):
    d = (a.float() - b.float()).abs()
    return dict(equal=bool(torch.equal(a, b)), max=float(d.max()), mean=float(d.mean()))


def _timed(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base")
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    w, h = map(int, args.size.split("x"))
    out = make_run_dir("moss_parity", args.label)
    dev = torch.device("cuda")
    ref = _reference_module(args.model_dir)
    report = {"te": [], "images": []}

    # --- TE parity
    tok, te = load_text_encoder(args.model_dir, dev)
    shim = SimpleNamespace(text_encoder=te, tokenizer=tok, _execution_device=dev)
    embeds = {}
    for text in PROMPTS + [""]:
        ours = encode_prompt(tok, te, text, dev)
        theirs, _mask = ref.MOSSImagePipeline.encode_prompt(shim, text)
        report["te"].append(dict(tokens=int(ours.shape[0]), **_delta(ours, theirs[0])))
        print(f"[te] {ours.shape[0]} tokens {report['te'][-1]}", flush=True)
        embeds[text] = ours.cpu()
    del tok, te, shim
    empty_cache()

    noise = initial_noise(args.seed, h, w)
    sigmas = snapped_sigmas(args.steps)
    neg, neg_mask = pad_embeds([embeds[""]], device=dev)

    # --- reference pipeline, its own transformer class
    tr = (
        ref.MOSSImageTransformer.from_pretrained(
            variant_dir(args.model_dir, args.model) / "transformer",
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .to(dev)
    )
    pipe = ref.MOSSImagePipeline(tr, ref.MOSSImageScheduler())
    pipe.set_progress_bar_config(disable=True)
    ref_lat = []
    for p in PROMPTS:
        e, m = pad_embeds([embeds[p]], device=dev)
        lat, sec = _timed(
            lambda: (
                pipe(
                    prompt_embeds=e,
                    prompt_attention_mask=m,
                    negative_prompt_embeds=neg,
                    negative_prompt_attention_mask=neg_mask,
                    height=h,
                    width=w,
                    num_inference_steps=args.steps,
                    latents=noise.clone(),
                    output_type="latent",
                )
                .images.float()
                .cpu()
            )
        )
        ref_lat.append(lat)
        report["images"].append(dict(tokens=int(e.shape[1]), reference_s=round(sec, 2)))
        print(f"[ref] {sec:.1f}s", flush=True)
    pipe = tr = None
    empty_cache()

    # --- ours
    tr = load_transformer(args.model_dir, args.model, device=dev)
    for i, p in enumerate(PROMPTS):
        e, m = pad_embeds([embeds[p]], device=dev)
        rec = report["images"][i]
        for batched in (False, True):
            torch.cuda.reset_peak_memory_stats()
            lat, sec = _timed(
                lambda: denoise(
                    tr,
                    noise.to(dev),
                    e,
                    m,
                    neg,
                    neg_mask,
                    sigmas,
                    cfg=3.0,
                    uncond_skip=(22,),
                    batched=batched,
                ).cpu()
            )
            key = "batched" if batched else "unbatched"
            rec[f"{key}_s"] = round(sec, 2)
            rec[f"{key}_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            rec[key] = _delta(lat, ref_lat[i])
            rec[f"{key}_latent_absmean"] = float(lat.abs().mean())
            torch.save(lat, out / f"{i}_{key}.pt")
            print(f"[{key}] {i}: {sec:.1f}s {rec[key]}", flush=True)
        torch.save(ref_lat[i], out / f"{i}_reference.pt")
    write_result(
        out,
        script=__file__,
        args=args,
        metrics=report,
        label=args.label,
        artifacts=sorted(p.name for p in out.glob("*.pt")),
        device=dev,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
