#!/usr/bin/env python3
"""Caption-sparsity sweep — does specifying detail reduce x̂₀ wander?

Tests the actionable claim from the wander ladder (cube 1.7 → sincos 6.2):
wander scales with caption under-determination. Here we hold the *image intent*
fixed (one real @sincos caption) and progressively strip its descriptive tags:
full → ¾ → ½ → ¼ → scaffold-only → empty. If wander climbs as tags are removed,
caption specificity is a real lever on the in-step contradiction — no retraining.

Stripping is nested (shuffle descriptive tags once with a fixed seed, keep a
prefix) so ½ ⊂ ¾ ⊂ full. The scaffold (rating, subject count, character, series,
@trigger up to and including the @-tag) is preserved until the 'empty' level.

Runs on the real shipped model (base + sincos LoRA). Reuses one bundle.

Usage::

    python bench/x0_contradiction/sparsity_sweep.py --seeds 0,1 --steps 24
"""

from __future__ import annotations

import argparse
import random
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

LEVELS = [
    ("full", 1.0),
    ("3/4", 0.75),
    ("1/2", 0.5),
    ("1/4", 0.25),
    ("scaffold", 0.0),
    ("empty", -1.0),
]


def split_caption(cap: str):
    tags = [t.strip() for t in cap.split(",") if t.strip()]
    trig = next((i for i, t in enumerate(tags) if t.startswith("@")), None)
    cut = (trig + 1) if trig is not None else min(3, len(tags))
    return tags[:cut], tags[cut:]  # scaffold, descriptive


def strip_to(cap: str, frac: float) -> tuple[str, int]:
    if frac < 0:  # empty = unconditional
        return "", 0
    scaffold, desc = split_caption(cap)
    shuffled = desc[:]
    random.Random(42).shuffle(shuffled)
    keep = shuffled[: round(frac * len(desc))]
    # restore original order for the kept subset (reads naturally)
    kept = [t for t in desc if t in set(keep)]
    tags = scaffold + kept
    return ", ".join(tags), len(tags)


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
    p.add_argument("--label", default="sparsity")
    p.add_argument(
        "--compile",
        action="store_true",
        help="compile_blocks the DiT (fixed-res, static graph)",
    )
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if not Path(a.adapter).is_absolute():
        a.adapter = str(REPO_ROOT / a.adapter)
    caps = [
        ln.strip()
        for ln in Path("bench/x0_contradiction/prompts/sincos.txt")
        .read_text()
        .splitlines()
        if ln.strip() and not ln.startswith("#")
    ][: a.n_captions]

    ck = anima_lora.default_checkpoints()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir("x0_contradiction", label=a.label)
    print(
        f"run dir: {run_dir}\n{len(caps)} captions x {len(LEVELS)} levels x {len(seeds)} seeds"
    )

    def mkargs(prompt, seed):
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
            lora_weight=[a.adapter],
            lora_multiplier=[a.multiplier],
            extra_argv=("--compile_blocks",) if a.compile else (),
        ).to_args()

    bundle = build_inference_bundle(
        mkargs("init", seeds[0]), device=device, with_vae=False
    )

    per_level = []
    for name, frac in LEVELS:
        fulls, earlys, ntags = [], [], []
        for cap in caps:
            stripped, n = strip_to(cap, frac)
            for seed in seeds:
                fw, ew = wander(bundle, mkargs(stripped, seed))
                fulls.append(fw)
                earlys.append(ew)
            ntags.append(n)
        entry = {
            "level": name,
            "frac": frac,
            "mean_tags": float(np.mean(ntags)),
            "wander_full": float(np.mean(fulls)),
            "wander_full_std": float(np.std(fulls)),
            "wander_early": float(np.mean(earlys)),
        }
        per_level.append(entry)
        print(
            f"  {name:9} tags~{entry['mean_tags']:4.1f}  "
            f"wander {entry['wander_full']:.2f} ±{entry['wander_full_std']:.2f}  "
            f"(early {entry['wander_early']:.2f})"
        )

    # plot wander vs tag count
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [e["mean_tags"] for e in per_level]
    ys = [e["wander_full"] for e in per_level]
    es = [e["wander_full_std"] for e in per_level]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=2)
    for e in per_level:
        ax.annotate(
            e["level"],
            (e["mean_tags"], e["wander_full"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    ax.axhline(1.8, ls="--", c="gray", lw=1)
    ax.text(max(xs) * 0.6, 1.9, "unambiguous floor ≈1.8", color="gray", fontsize=9)
    ax.set_xlabel("caption tag count (more → more specified)")
    ax.set_ylabel("x̂₀ wander (path/net, full trajectory)")
    ax.set_title("Caption specificity vs in-step contradiction (sincos LoRA)")
    ax.invert_xaxis()  # left = sparse/empty, right = full
    fig.tight_layout()
    plot = run_dir / "wander_vs_sparsity.png"
    fig.savefig(plot, dpi=110)
    plt.close(fig)

    write_result(
        run_dir,
        script=__file__,
        args=a,
        metrics={"per_level": per_level, "captions": caps},
        label=a.label,
        artifacts=["wander_vs_sparsity.png"],
        device=device,
    )
    print(f"\nsaved plot + result.json in: {run_dir}")


if __name__ == "__main__":
    main()
