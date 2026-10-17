#!/usr/bin/env python3
"""Base vs base+LoRA full-trajectory wander on sincos / others captions.

Closes the caveat in report.md: Finding 3's ladder is base-only, and Finding 1's
base-vs-LoRA was *tail* wander. This measures FULL and EARLY (σ>0.45) wander for
both models on both caption groups, so we can state whether the sincos LoRA moves
full-trajectory contradiction or only re-aims the destination.

Fixed 1024² square gen -> static graph; pass --compile for compile_blocks
(bit-exact to eager per the DiT docs, so numbers stay comparable).

Usage::

    python bench/x0_contradiction/base_vs_lora_wander.py --seeds 0,1 --compile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402

import anima_lora  # noqa: E402
from bench.x0_contradiction.run_bench import capture_x0  # noqa: E402
from library.runtime.harness import build_inference_bundle  # noqa: E402


def wander(bundle, args) -> tuple[float, float]:
    with capture_x0() as cap:
        bundle.generate(args)
    st = sorted(cap, key=lambda r: r[0])
    sig = [s for _, s, _ in st]
    x = [t for _, _, t in st]

    def seg(idx):
        path = sum((x[i] - x[i - 1]).norm(dim=0).sum().item() for i in idx)
        net = (x[idx[-1]] - x[idx[0] - 1]).norm(dim=0).sum().item()
        return path / (net + 1e-9)

    allidx = list(range(1, len(x)))
    early = [i for i in allidx if sig[i] >= 0.45]
    return seg(allidx), seg(early)


def load_caps(name, n):
    return [
        ln.strip()
        for ln in Path(f"bench/x0_contradiction/prompts/{name}.txt")
        .read_text()
        .splitlines()
        if ln.strip() and not ln.startswith("#")
    ][:n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="output/ckpt/anima_sincos.safetensors")
    p.add_argument("--multiplier", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow-shift", dest="flow_shift", type=float, default=3.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--n-captions", type=int, default=3)
    p.add_argument("--label", default="base-vs-lora")
    p.add_argument("--compile", action="store_true")
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if not Path(a.adapter).is_absolute():
        a.adapter = str(REPO_ROOT / a.adapter)
    groups = {
        "sincos": load_caps("sincos", a.n_captions),
        "others": load_caps("others", a.n_captions),
    }
    ck = anima_lora.default_checkpoints()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir("x0_contradiction", label=a.label)
    extra = ("--compile_blocks",) if a.compile else ()

    def mkargs(prompt, seed, with_lora):
        return anima_lora.GenerationRequest(
            dit=ck.dit,
            vae=ck.vae,
            text_encoder=ck.text_encoder,
            prompt=prompt,
            image_size=(a.size, a.size),
            infer_steps=a.steps,
            guidance_scale=a.cfg,
            flow_shift=a.flow_shift,
            sampler="euler",
            seed=seed,
            lora_weight=[a.adapter] if with_lora else None,
            lora_multiplier=[a.multiplier] if with_lora else None,
            extra_argv=extra,
        ).to_args()

    cells = {}  # (model, group) -> dict(full, early)
    for with_lora, model in ((False, "base"), (True, "lora")):
        print(f"\nloading {model} bundle (compile={a.compile})...")
        bundle = build_inference_bundle(
            mkargs(groups["sincos"][0], seeds[0], with_lora),
            device=device,
            with_vae=False,
        )
        for g, caps in groups.items():
            fulls, earlys = [], []
            for cap in caps:
                for seed in seeds:
                    fw, ew = wander(bundle, mkargs(cap, seed, with_lora))
                    fulls.append(fw)
                    earlys.append(ew)
            cells[(model, g)] = {
                "full": float(np.mean(fulls)),
                "full_std": float(np.std(fulls)),
                "early": float(np.mean(earlys)),
            }
            print(
                f"  {model:4} / {g:6}: full {cells[(model, g)]['full']:.2f} "
                f"±{cells[(model, g)]['full_std']:.2f}  early {cells[(model, g)]['early']:.2f}"
            )
        del bundle
        torch.cuda.empty_cache()

    print("\n=== FULL-trajectory wander (path/net) ===")
    print(f"{'':6}{'sincos':>10}{'others':>10}")
    for model in ("base", "lora"):
        print(
            f"{model:6}{cells[(model, 'sincos')]['full']:>10.2f}"
            f"{cells[(model, 'others')]['full']:>10.2f}"
        )
    for g in ("sincos", "others"):
        r = cells[("lora", g)]["full"] / (cells[("base", g)]["full"] + 1e-9)
        print(f"lora/base {g}: x{r:.3f}")

    write_result(
        run_dir,
        script=__file__,
        args=a,
        metrics={"cells": {f"{m}/{g}": v for (m, g), v in cells.items()}},
        label=a.label,
        device=device,
    )
    print(f"\nresult.json in: {run_dir}")


if __name__ == "__main__":
    main()
