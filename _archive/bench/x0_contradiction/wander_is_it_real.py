#!/usr/bin/env python3
"""Does the LoRA's extra x̂₀ wander *degrade*, or does it just buy detail?

report.md closes with: the sincos LoRA adds +8-12% full-trajectory wander at a
*fixed caption*, and calls it "most likely benign" by borrowing Finding 5's law
(wander tracks scene complexity) -> "LoRA adds stylistic detail => busier scene
=> more wander, not destabilization." But Finding 5 was fit on the CAPTION axis
(stripping tags); the LoRA's +10% is at CONSTANT caption. So "benign" is an
extrapolation across a different axis. Two readings fit the same +10% and the
report can't separate them:

  benign  : LoRA paints a busier scene -> extra wander lands WHERE it adds detail
  harmful : LoRA destabilizes the path at constant content -> extra wander lands
            in under-determined / seed-lottery regions, decoupled from detail

There is no quality reward for Anima (memory: null_tta), so we can't score
"broken" directly. Instead we use the report's own two proxies as opposite poles
and ask which one the LoRA's *extra contradiction* spatially coincides with:

  * Δdetail  — per-patch high-freq (Sobel) energy of the decoded final, lora-base.
               High where the LoRA painted texture/detail.  (benign pole)
  * crossvar — per-patch cross-seed variance of x̂₀ at REF_SIGMA (run_bench's
               regime classifier). High = genuinely multimodal / seed-lottery =
               the breakage-prone regime.  (harmful pole)

Contradiction signal (where, per patch):
  * Δbacktrack — early-band (σ≥split) Σ max(0,-cos(d_i,d_{i+1})) energy, lora-base.
    Pure non-monotonicity (draw-then-erase), not mere motion. THE report metric,
    spatialized over the early band (where ~95% of wander lives).

Verdict logic (per prompt, seed-averaged maps, then meaned over prompts):
  corr(Δbacktrack, Δdetail)  >  corr(Δbacktrack, crossvar)   => benign-leaning
  corr(Δbacktrack, crossvar) >  corr(Δbacktrack, Δdetail)    => harmful-leaning
Plus the across-prompt scalar check: do prompts whose wander the LoRA inflates
most also gain the most detail?  (benign) — Pearson(Δbacktrack_sum, Δdetail_sum).

Honest limit: crossvar marks *breakage-prone* regions, not confirmed breakage;
a coincidence is suggestive, not a hand/anatomy detector. This narrows benign-vs-
harmful; it doesn't measure pixel failures. (That follow-up needs a detector.)

Usage::

    python bench/x0_contradiction/wander_is_it_real.py \
        --prompts-file bench/x0_contradiction/prompts/sincos.txt \
        --seeds 0,1,2 --compile --label sincos-real
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
from bench.x0_contradiction.run_bench import (  # noqa: E402
    REF_SIGMA,
    capture_x0,
    crossseed_var,
    pearson,
)
from library.runtime.harness import build_inference_bundle  # noqa: E402


# --------------------------------------------------------------------------- #
# per-patch early-band reduction (path + backtrack energy + ref snapshot)
# --------------------------------------------------------------------------- #
def early_maps(cap, split: float):
    """Reduce a captured x̂₀ trajectory to early-band (σ≥split) per-patch maps.

    Returns (H,W) path + backtrack maps and the (C,H,W) x̂₀ snapshot nearest
    REF_SIGMA (for the cross-seed-variance regime map).
    """
    steps = sorted(cap, key=lambda r: r[0])
    sigmas = [s for _, s, _ in steps]
    x0s = [x for _, _, x in steps]  # (C,H,W) each
    HW = x0s[0].shape[1:]

    path = torch.zeros(HW)
    backtrack = torch.zeros(HW)
    prev_d = None
    for i in range(1, len(x0s)):
        diff = x0s[i] - x0s[i - 1]  # (C,H,W)
        if sigmas[i] < split:  # early band only (σ≥split); tail is 4-5% of motion
            prev_d = None
            continue
        path += diff.norm(dim=0)
        if prev_d is not None:
            num = (prev_d * diff).sum(dim=0)
            den = prev_d.norm(dim=0) * diff.norm(dim=0) + 1e-8
            backtrack += torch.clamp(-(num / den), min=0.0)
        prev_d = diff

    ref_idx = min(range(len(sigmas)), key=lambda k: abs(sigmas[k] - REF_SIGMA))
    return {
        "path": path.numpy(),
        "backtrack": backtrack.numpy(),
        "ref_x0": x0s[ref_idx],
    }


def detail_map(img, hw) -> np.ndarray:
    """Per-patch high-freq (Sobel-gradient) energy of a decoded PIL image -> (H,W).

    Grayscale -> gradient magnitude -> bilinear-resize to the latent grid `hw`.
    High = textured/detailed region.
    """
    from PIL import Image

    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)  # (Hpix, Wpix)
    H, W = hw
    return np.asarray(
        Image.fromarray(mag).resize((W, H), Image.BILINEAR), dtype=np.float32
    )


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def save_panels(run_dir, tag, d_backtrack, d_detail, crossvar) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sym(m):
        lim = float(np.abs(m).max()) or 1.0
        return {"vmin": -lim, "vmax": lim, "cmap": "coolwarm"}

    panels = [
        ("Δ backtrack (lora-base)", d_backtrack, sym(d_backtrack)),
        ("Δ detail (lora-base)", d_detail, sym(d_detail)),
        ("lora cross-seed var", crossvar, {"cmap": "viridis"}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (title, m, kw) in zip(axes, panels):
        im = ax.imshow(m, **kw)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(tag, fontsize=11)
    fig.tight_layout()
    name = f"realcheck_{tag}.png"
    fig.savefig(run_dir / name, dpi=96, bbox_inches="tight")
    plt.close(fig)
    return name


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="output/ckpt/anima_sincos.safetensors")
    p.add_argument("--multiplier", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow-shift", dest="flow_shift", type=float, default=3.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--split", type=float, default=0.45)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--n-captions", type=int, default=4)
    p.add_argument("--label", default="wander-real")
    p.add_argument("--compile", action="store_true")
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if a.prompts_file:
        prompts = [
            ln.strip()
            for ln in Path(a.prompts_file).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ][: a.n_captions]
    else:
        prompts = [
            "1girl, waving hello, smiling, looking at viewer",
            "1girl, holding a small object in her hands, detailed",
            "1girl, sitting at a table with a cup, indoors",
        ]
    if not Path(a.adapter).is_absolute():
        a.adapter = str(REPO_ROOT / a.adapter)
    if not Path(a.adapter).exists():
        sys.exit(f"adapter not found: {a.adapter}")

    ck = anima_lora.default_checkpoints()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir("x0_contradiction", label=a.label)
    extra = ("--compile_blocks",) if a.compile else ()
    print(f"run dir: {run_dir}")
    print(
        f"{len(prompts)} prompts x {len(seeds)} seeds; adapter {Path(a.adapter).name}"
    )

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

    def run(bundle, prompt, seed, with_lora):
        with capture_x0() as cap:
            latent = bundle.generate(mkargs(prompt, seed, with_lora))
        maps = early_maps(cap, a.split)
        img = anima_lora.decode_to_pil(bundle.vae, latent, bundle.device)
        maps["detail"] = detail_map(img, maps["path"].shape)
        return maps

    artifacts: list[str] = []
    per_prompt: list[dict] = []
    # scalars for the across-prompt benign check
    dbt_sums, ddetail_sums = [], []

    for with_lora, name in ((False, "base"), (True, "lora")):
        print(f"\nloading {name} bundle (compile={a.compile})...")
        bundle = build_inference_bundle(
            mkargs(prompts[0], seeds[0], with_lora), device=device, with_vae=True
        )
        if name == "base":
            base_store: dict = {}
            for pi, prompt in enumerate(prompts):
                base_store[pi] = [run(bundle, prompt, s, False) for s in seeds]
                print(f"  base [{pi + 1}/{len(prompts)}] done")
        else:
            for pi, prompt in enumerate(prompts):
                lora_runs = [run(bundle, prompt, s, True) for s in seeds]
                base_runs = base_store[pi]

                def avg(runs, key):
                    return np.mean([r[key] for r in runs], axis=0)

                d_bt = avg(lora_runs, "backtrack") - avg(base_runs, "backtrack")
                d_detail = avg(lora_runs, "detail") - avg(base_runs, "detail")
                d_path = avg(lora_runs, "path") - avg(base_runs, "path")
                lora_cv = crossseed_var([r["ref_x0"] for r in lora_runs])

                c_bt_detail = pearson(d_bt, d_detail)
                c_bt_crossvar = pearson(d_bt, lora_cv)
                c_path_detail = pearson(d_path, d_detail)
                entry = {
                    "prompt": prompt,
                    "corr_dbacktrack_ddetail": c_bt_detail,
                    "corr_dbacktrack_crossvar": c_bt_crossvar,
                    "corr_dpath_ddetail": c_path_detail,
                    "dbacktrack_sum": float(d_bt.sum()),
                    "ddetail_sum": float(d_detail.sum()),
                }
                per_prompt.append(entry)
                dbt_sums.append(entry["dbacktrack_sum"])
                ddetail_sums.append(entry["ddetail_sum"])
                print(
                    f"  lora [{pi + 1}/{len(prompts)}] {prompt[:42]!r:44} "
                    f"corr(Δbt,Δdetail)={c_bt_detail:+.2f}  "
                    f"corr(Δbt,crossvar)={c_bt_crossvar:+.2f}"
                )
                artifacts.append(
                    save_panels(run_dir, f"p{pi}", d_bt, d_detail, lora_cv)
                )
        del bundle
        torch.cuda.empty_cache()

    mean_detail = float(np.nanmean([e["corr_dbacktrack_ddetail"] for e in per_prompt]))
    mean_crossvar = float(
        np.nanmean([e["corr_dbacktrack_crossvar"] for e in per_prompt])
    )
    across = pearson(np.array(dbt_sums), np.array(ddetail_sums))
    verdict = (
        "benign-leaning (extra contradiction coincides with added detail)"
        if mean_detail > mean_crossvar
        else "harmful-leaning (extra contradiction coincides with seed-lottery regions)"
    )
    overall = {
        "mean_corr_dbacktrack_ddetail": mean_detail,
        "mean_corr_dbacktrack_crossvar": mean_crossvar,
        "across_prompt_corr_dbt_ddetail": across,
        "verdict": verdict,
    }
    write_result(
        run_dir,
        script=__file__,
        args=a,
        metrics={"overall": overall, "per_prompt": per_prompt},
        label=a.label,
        artifacts=artifacts,
        device=device,
    )

    print("\n=== verdict: is the LoRA's extra wander real degradation? ===")
    print(f"  mean corr(Δbacktrack, Δdetail)   = {mean_detail:+.3f}  (benign pole)")
    print(f"  mean corr(Δbacktrack, crossvar)  = {mean_crossvar:+.3f}  (harmful pole)")
    print(f"  across-prompt corr(Δbt, Δdetail) = {across:+.3f}")
    print(f"  -> {verdict}")
    print(f"\nresult.json + panels in: {run_dir}")


if __name__ == "__main__":
    main()
