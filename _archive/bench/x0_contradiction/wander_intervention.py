#!/usr/bin/env python3
"""INTERVENTION: does *adding* wander reduce hand breakage? (er_sde s_noise sweep)

wander_vs_breakage.py found a robust-but-observational negative association:
where the LoRA adds hand-region wander it adds fewer broken fingers (Spearman
-0.38 p=0.04, strengthens to -0.47 dropping the outlier). Observational ≠ causal
— we measured wander, never manipulated it. This manipulates it.

er_sde's ``s_noise`` is a clean exploration knob: 0 = deterministic (low wander),
higher = more injected per-step noise = more trajectory exploration. We hold the
LoRA, prompts, seeds, steps, CFG fixed and sweep ONLY s_noise, then measure:

  * manipulation check — does early-σ x̂₀ wander actually rise with s_noise?
  * causal test        — does SAM3 excess-finger breakage fall as wander rises?

If breakage drops as injected wander rises, "wander is protective" becomes
causal. If flat/U-shaped (some churn helps, too much mushes), we learn the
operating point. If breakage is flat or rises, the -0.38 was a correlation only.

Capture seam: er_sde uses ERSDESampler.step(latents, denoised, i) where
``denoised`` IS x̂₀ — patch it to record the trajectory; patch __init__ to force
s_noise (it's hardcoded to 1.0 in generation.py, no CLI knob).

NB s_noise>0 makes er_sde stochastic, so the same seed gives different draws per
level — that's the intervention, not a confound; we average over seeds.

Usage::

    python bench/x0_contradiction/wander_intervention.py \
        --prompts-file bench/x0_contradiction/prompts/hands.txt \
        --seeds 0,1,2,3,4 --s-noise 0.0,1.0,2.0,3.0 --compile --label snoise
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if not hasattr(np, "bool"):
    np.bool = np.bool_

from bench._common import make_run_dir, write_result  # noqa: E402

import anima_lora  # noqa: E402
import library.inference.sampling as _sampling  # noqa: E402
from bench.x0_contradiction.wander_is_it_real import pearson  # noqa: E402
from bench.x0_contradiction.wander_vs_breakage import (  # noqa: E402
    HandFingerDetector,
    downsample_mask,
)
from library.runtime.harness import build_inference_bundle  # noqa: E402


@contextmanager
def capture_ersde(s_noise: float):
    """Force ERSDESampler.s_noise and record x̂₀ (the `denoised` arg) per step."""
    cap: list[tuple[int, float, torch.Tensor]] = []
    orig_init = _sampling.ERSDESampler.__init__
    orig_step = _sampling.ERSDESampler.step

    def pinit(self, *a, **k):
        orig_init(self, *a, **k)
        self.s_noise = s_noise

    def pstep(self, latents, denoised, step_i):
        cap.append(
            (
                int(step_i),
                float(self.sigmas[step_i]),
                denoised[0, :, 0].detach().to("cpu", torch.float32),  # (C,H,W)
            )
        )
        return orig_step(self, latents, denoised, step_i)

    _sampling.ERSDESampler.__init__ = pinit
    _sampling.ERSDESampler.step = pstep
    try:
        yield cap
    finally:
        _sampling.ERSDESampler.__init__ = orig_init
        _sampling.ERSDESampler.step = orig_step


def reduce_maps(cap, split: float):
    """-> (full_wander, early_wander scalars, early-σ per-patch path map (h,w))."""
    st = sorted(cap, key=lambda r: r[0])
    sig = [s for _, s, _ in st]
    x = [t for _, _, t in st]

    def seg(idx):
        path = sum((x[i] - x[i - 1]).norm(dim=0).sum().item() for i in idx)
        net = (x[idx[-1]] - x[idx[0] - 1]).norm(dim=0).sum().item()
        return path / (net + 1e-9)

    allidx = list(range(1, len(x)))
    early = [i for i in allidx if sig[i] >= split]
    emap = torch.zeros(x[0].shape[1:])
    for i in early:
        emap += (x[i] - x[i - 1]).norm(dim=0)
    return seg(allidx), (seg(early) if len(early) > 1 else float("nan")), emap.numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="output/ckpt/anima_sincos.safetensors")
    p.add_argument("--multiplier", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow-shift", dest="flow_shift", type=float, default=3.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--split", type=float, default=0.45)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--s-noise", dest="s_noise", default="0.0,1.0,2.0,3.0")
    p.add_argument("--prompts-file", default="bench/x0_contradiction/prompts/hands.txt")
    p.add_argument("--label", default="snoise")
    p.add_argument("--compile", action="store_true")
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    levels = [float(s) for s in a.s_noise.split(",") if s.strip()]
    prompts = [
        ln.strip()
        for ln in Path(a.prompts_file).read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if not Path(a.adapter).is_absolute():
        a.adapter = str(REPO_ROOT / a.adapter)

    ck = anima_lora.default_checkpoints()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir("x0_contradiction", label=a.label)
    (run_dir / "img").mkdir(exist_ok=True)
    extra = ("--compile_blocks",) if a.compile else ()
    print(f"run dir: {run_dir}")
    print(f"levels s_noise={levels}  x {len(prompts)} prompts x {len(seeds)} seeds")

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
            sampler="er_sde",
            seed=seed,
            lora_weight=[a.adapter],
            lora_multiplier=[a.multiplier],
            extra_argv=extra,
        ).to_args()

    # ---- Phase 1: generate across s_noise levels (one bundle, reused) --------
    # recs[(lvl, pi, seed)] = {img, full, early, emap}
    recs: dict = {}
    print(f"\n[phase1] loading LoRA er_sde bundle (compile={a.compile})...")
    bundle = build_inference_bundle(
        mkargs(prompts[0], seeds[0]), device=device, with_vae=True
    )
    for lvl in levels:
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                with capture_ersde(lvl) as cap:
                    latent = bundle.generate(mkargs(prompt, seed))
                full, early, emap = reduce_maps(cap, a.split)
                img = anima_lora.decode_to_pil(bundle.vae, latent, bundle.device)
                ipath = run_dir / "img" / f"sn{lvl}_p{pi}_s{seed}.png"
                img.save(ipath)
                recs[(lvl, pi, seed)] = {
                    "img": ipath,
                    "full": full,
                    "early": early,
                    "emap": emap,
                    "epath": float(emap.mean()),  # raw early-σ motion (the obs. signal)
                }
        print(f"  s_noise={lvl} done")
    del bundle
    torch.cuda.empty_cache()

    # ---- Phase 2: SAM3 finger-count breakage --------------------------------
    print("\n[phase2] loading SAM3 detector...")
    from PIL import Image

    det = HandFingerDetector(device=device)
    for key, rec in recs.items():
        try:
            d = det.detect(Image.open(rec["img"]).convert("RGB"))
        except Exception as exc:
            print(f"  [detect failed {key}: {exc}]")
            rec.update(excess=np.nan, total_fingers=np.nan, hand_wander=np.nan)
            continue
        hw = rec["emap"].shape
        hmask = downsample_mask(d["hand_union"], hw) > 0.25
        rec.update(
            excess=d["excess_fingers"],
            total_fingers=d["total_fingers"],
            hand_wander=float(rec["emap"][hmask].mean()) if hmask.any() else np.nan,
        )
    del det
    torch.cuda.empty_cache()

    # ---- Phase 3: ladder + causal test --------------------------------------
    def lvl_mean(lvl, key):
        v = [
            recs[(lvl, pi, s)][key]
            for pi in range(len(prompts))
            for s in seeds
            if not np.isnan(recs[(lvl, pi, s)][key])
        ]
        return float(np.mean(v)) if v else float("nan")

    def lvl_frac_broken(lvl):
        v = [recs[(lvl, pi, s)]["excess"] for pi in range(len(prompts)) for s in seeds]
        v = [x for x in v if not np.isnan(x)]
        return float(np.mean([x > 0 for x in v])) if v else float("nan")

    ladder = []
    for lvl in levels:
        ladder.append(
            {
                "s_noise": lvl,
                "early_path": lvl_mean(
                    lvl, "epath"
                ),  # raw motion (obs. breakage signal)
                "full_wander": lvl_mean(lvl, "full"),
                "early_wander": lvl_mean(lvl, "early"),
                "excess_per_img": lvl_mean(lvl, "excess"),
                "total_fingers": lvl_mean(lvl, "total_fingers"),
                "frac_broken": lvl_frac_broken(lvl),
            }
        )

    # pooled per-image causal correlation (wander manipulated via s_noise)
    allw = np.array([r["early"] for r in recs.values()], float)
    alle = np.array([r["excess"] for r in recs.values()], float)
    ok = ~(np.isnan(allw) | np.isnan(alle))
    pooled = pearson(allw[ok], alle[ok])
    # level-mean trend (manipulation strength vs breakage)
    lw = np.array([d["early_wander"] for d in ladder])
    le = np.array([d["excess_per_img"] for d in ladder])
    trend = pearson(lw, le)

    overall = {
        "ladder": ladder,
        "pooled_corr_earlywander_excess": pooled,
        "levelmean_corr_wander_excess": trend,
        "n_per_level": len(prompts) * len(seeds),
    }
    write_result(
        run_dir,
        script=__file__,
        args=a,
        metrics={
            "overall": overall,
            "recs": {
                f"{lvl}/{pi}/{s}": {
                    k: recs[(lvl, pi, s)][k]
                    for k in ("full", "early", "excess", "total_fingers", "hand_wander")
                }
                for lvl in levels
                for pi in range(len(prompts))
                for s in seeds
            },
        },
        label=a.label,
        device=device,
    )

    print("\n=== INTERVENTION ladder (s_noise = injected stochasticity) ===")
    print(
        f"{'s_noise':>8}{'early_PATH':>11}{'early_ratio':>12}{'excess/img':>12}{'%broken':>9}"
    )
    for d in ladder:
        print(
            f"{d['s_noise']:>8.1f}{d['early_path']:>11.3f}{d['early_wander']:>12.2f}"
            f"{d['excess_per_img']:>12.3f}{100 * d['frac_broken']:>8.0f}%"
        )
    print("\n  manipulation check: early wander should RISE with s_noise")
    print(f"  causal test: pooled corr(early wander, excess) = {pooled:+.3f}")
    print(f"               level-mean corr(wander, excess)    = {trend:+.3f}")
    print("  negative + breakage falling up the ladder ⇒ wander is CAUSALLY protective")
    print(f"\nresult.json + img/ in: {run_dir}")


if __name__ == "__main__":
    main()
