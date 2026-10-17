#!/usr/bin/env python3
"""x0-contradiction bench — per-step x̂₀ trajectory instability, base vs LoRA.

Operationalizes the "in-step contradiction" idea: under an under-specified
caption the posterior ``p(x0|c)`` is multimodal, so the velocity field hedges
(mode-averaging) and the model's running clean-latent guess ``x̂₀(t)`` keeps
re-pointing instead of converging — most visibly in the σ<0.45 *resolve tail*
(memory: Anima resolves x̂₀ by σ≈0.45). A region still moving in the tail is the
contradiction signal.

Two questions, one harness:

  1. Is it real?  — per-patch x̂₀ total-variation (TV) across steps, split at
     σ=0.45 into early vs tail. High *tail* TV = a region redrawn after it
     should have frozen.

  2. Is *your training* causing it? — same prompt, same seed, BASE vs BASE+LoRA.
     A LoRA over a frozen base can *raise* tail-TV by regressing the field
     toward its small-data mean. The headline metric is the lora/base tail-TV
     ratio.

  + regime classifier — cross-seed variance of x̂₀ at a fixed early σ separates
    "genuinely multimodal region" (healthy seed lottery, training can't fix)
    from "score error" (mode-averaging, training-reducible). Reported as the
    Pearson r between the cross-seed-variance map and the seed-averaged tail-TV
    map: high positive r ⇒ the instability coincides with ambiguity ⇒ the
    reducible regime.

Capture seam: monkeypatch ``library.inference.sampling.step`` (the euler ODE
step), recomputing the *guided* x̂₀ = x_t − σ_i·v exactly as generation.py:799.
Forces the default euler sampler (no er_sde / spectrum) so the seam fires.

Usage::

    python bench/x0_contradiction/run_bench.py \
        --adapter output/ckpt/anima_sincos.safetensors \
        --seeds 0,1,2 --steps 24 --cfg 4.0 --label sincos

    # supply the artist trigger / your own prompts:
    python bench/x0_contradiction/run_bench.py --prompts-file my_prompts.txt
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

# bench/ is not an installed package — bootstrap repo root onto sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402

import anima_lora  # noqa: E402
import library.inference.sampling as _sampling  # noqa: E402
from library.runtime.harness import build_inference_bundle  # noqa: E402

# Hand-implying but deliberately *under-specified* prompts — the regime where
# the posterior is multimodal and the field should hedge. Override with
# --prompts-file (one prompt per line) to inject the artist trigger / style tag.
DEFAULT_PROMPTS = [
    "1girl, waving hello, smiling, looking at viewer",
    "1girl, holding a small object in her hands, detailed",
    "1girl, hands near her face, thoughtful pose",
    "1girl, sitting at a table with a cup, indoors",
    "2girls, standing together, casual clothes",
]

TAIL_SPLIT = 0.45  # σ below this = resolve tail (memory: x̂₀ done by ~0.45)
REF_SIGMA = 0.70  # early σ for the cross-seed multimodality probe


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #
_CAP: list[tuple[int, float, torch.Tensor]] = []


@contextmanager
def capture_x0():
    """Patch the euler step to record the guided x̂₀ at every denoising step.

    x̂₀ = x_t − σ_i·v  (generation.py:799). Latent is 5D (B,C,1,H,W); B=1 here
    (one prompt per generate, CFG runs as two separate forwards). Take batch 0,
    drop the singleton temporal axis at dim 1 (explicit index, never squeeze()).
    """
    _CAP.clear()
    orig = _sampling.step

    def patched(latents, noise_pred, sigmas, step_i):
        x0 = latents.float() - sigmas[step_i] * noise_pred.float()
        x0_chw = x0[0, :, 0].detach().to("cpu", torch.float32)  # (C, H, W)
        _CAP.append((int(step_i), float(sigmas[step_i]), x0_chw))
        return orig(latents, noise_pred, sigmas, step_i)

    _sampling.step = patched
    try:
        yield _CAP
    finally:
        _sampling.step = orig


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def trajectory_maps(cap: list[tuple[int, float, torch.Tensor]], split: float):
    """Reduce a captured x̂₀ trajectory to per-patch maps + a ref-σ snapshot.

    Raw TV (path length) counts *all* x̂₀ motion, including a healthy monotone
    resolve. The contradiction signal is non-monotonicity, captured two ways:

      * wander = path / net_displacement  — =1 for a straight resolve, >1 when
        x̂₀ backtracks (draw-then-erase). Computed over the TAIL segment.
      * backtrack = Σ max(0, −cos(d_i, d_{i+1}))  — energy of direction reversals
        between consecutive tail increments (per patch, channel-vector cosine).

    Returns numpy (H,W) maps: tv_total, tv_early, tv_tail (path lengths),
    net_tail (net displacement across the tail), backtrack_tail; plus the x̂₀
    snapshot (C,H,W) at the step whose σ is closest to REF_SIGMA.
    """
    steps = sorted(cap, key=lambda r: r[0])
    sigmas = [s for _, s, _ in steps]
    x0s = [x for _, _, x in steps]  # each (C,H,W)
    HW = x0s[0].shape[1:]

    tv_total = torch.zeros(HW)
    tv_early = torch.zeros(HW)
    tv_tail = torch.zeros(HW)
    backtrack_tail = torch.zeros(HW)
    tail_incrs: list[torch.Tensor] = []  # (C,H,W) increments in the tail
    tail_x0s: list[torch.Tensor] = []  # x̂₀ snapshots bounding the tail

    prev_tail_d = None
    for i in range(1, len(x0s)):
        diff = x0s[i] - x0s[i - 1]  # (C,H,W)
        d = diff.norm(dim=0)  # (H,W) path increment
        sig = sigmas[i]  # σ at the end of the increment
        tv_total += d
        if sig < split:
            tv_tail += d
            if not tail_x0s:
                tail_x0s.append(x0s[i - 1])
            tail_x0s.append(x0s[i])
            tail_incrs.append(diff)
            if prev_tail_d is not None:
                # cosine between consecutive channel-vector increments per patch
                num = (prev_tail_d * diff).sum(dim=0)
                den = prev_tail_d.norm(dim=0) * diff.norm(dim=0) + 1e-8
                cos = num / den
                backtrack_tail += torch.clamp(-cos, min=0.0)
            prev_tail_d = diff
        else:
            tv_early += d

    if len(tail_x0s) >= 2:
        net_tail = (tail_x0s[-1] - tail_x0s[0]).norm(dim=0).numpy()
    else:
        net_tail = np.zeros(HW)

    ref_idx = min(range(len(sigmas)), key=lambda k: abs(sigmas[k] - REF_SIGMA))
    return {
        "tv_total": tv_total.numpy(),
        "tv_early": tv_early.numpy(),
        "tv_tail": tv_tail.numpy(),
        "net_tail": net_tail,
        "backtrack_tail": backtrack_tail.numpy(),
        "ref_x0": x0s[ref_idx],  # (C,H,W) torch
        "ref_sigma": sigmas[ref_idx],
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def crossseed_var(ref_x0s: list[torch.Tensor]) -> np.ndarray:
    """Per-patch variance of x̂₀ across seeds at the reference σ -> (H,W).

    High = the region is genuinely under-determined (many valid completions).
    """
    stack = torch.stack(ref_x0s, dim=0)  # (S, C, H, W)
    var = stack.var(dim=0, unbiased=False)  # (C, H, W)
    return var.mean(dim=0).numpy()  # (H, W)


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def save_heatmaps(run_dir: Path, tag: str, base_m, lora_m, crossvar) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta = lora_m["tv_tail"] - base_m["tv_tail"]
    panels = [
        ("base tail-TV", base_m["tv_tail"], "magma", None),
        ("lora tail-TV", lora_m["tv_tail"], "magma", None),
        ("Δ tail-TV (lora-base)", delta, "coolwarm", "sym"),
        ("lora cross-seed var", crossvar, "viridis", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, (title, m, cmap, mode) in zip(axes, panels):
        kw = {}
        if mode == "sym":
            lim = float(np.abs(m).max()) or 1.0
            kw = {"vmin": -lim, "vmax": lim}
        im = ax.imshow(m, cmap=cmap, **kw)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(tag, fontsize=11)
    fig.tight_layout()
    name = f"heatmap_{tag}.png"
    fig.savefig(run_dir / name, dpi=96, bbox_inches="tight")
    plt.close(fig)
    return name


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def make_args(ckpt, prompt, seed, a, save_path=None, with_lora=False):
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
        save_path=str(save_path) if save_path else None,
        lora_weight=[a.adapter] if with_lora else None,
        lora_multiplier=[a.multiplier] if with_lora else None,
        # fixed-res square gen -> one static token count, so per-block compile
        # (no dynamic_seq) is safe; first gen pays warmup, reused bundle amortizes.
        extra_argv=("--compile_blocks",) if getattr(a, "compile", False) else (),
    )
    return req.to_args()


def run_one(bundle, ckpt, prompt, seed, a, with_lora, run_dir, decode_tag):
    """Drive one generation, return reduced trajectory maps (+ decode finals)."""
    save_path = run_dir / f"final_{decode_tag}.png" if a.decode else None
    args = make_args(ckpt, prompt, seed, a, save_path=save_path, with_lora=with_lora)
    with capture_x0() as cap:
        latent = bundle.generate(args)
    maps = trajectory_maps(cap, a.split)
    if (
        a.decode
        and bundle.vae is not None
        and save_path is not None
        and not save_path.exists()
    ):
        try:
            img = anima_lora.decode_to_pil(bundle.vae, latent, bundle.device)
            img.save(save_path)
        except Exception as exc:  # best-effort; metric doesn't need pixels
            print(f"  [decode skipped: {exc}]")
    return maps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="output/ckpt/anima_sincos.safetensors")
    p.add_argument("--multiplier", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow-shift", dest="flow_shift", type=float, default=3.0)
    p.add_argument("--size", type=int, default=1024, help="pixel edge (square)")
    p.add_argument("--split", type=float, default=TAIL_SPLIT)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--decode", action="store_true", help="also save decoded finals")
    p.add_argument(
        "--compile",
        action="store_true",
        help="compile_blocks the DiT (fixed-res, static graph)",
    )
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if a.prompts_file:
        prompts = [
            ln.strip()
            for ln in Path(a.prompts_file).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        prompts = DEFAULT_PROMPTS

    if not Path(a.adapter).is_absolute():
        a.adapter = str(REPO_ROOT / a.adapter)
    if not Path(a.adapter).exists():
        sys.exit(f"adapter not found: {a.adapter}")

    ckpt = anima_lora.default_checkpoints()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir("x0_contradiction", label=a.label)
    print(f"run dir: {run_dir}")
    print(f"adapter: {a.adapter}")
    print(f"{len(prompts)} prompts x {len(seeds)} seeds, {a.steps} steps, cfg {a.cfg}")

    # Two model loads total: base (no adapter) and base+LoRA. Each reused across
    # all prompts/seeds via bundle.generate(per_call_args).
    print("loading BASE bundle...")
    base_args = make_args(ckpt, prompts[0], seeds[0], a, with_lora=False)
    base_bundle = build_inference_bundle(base_args, device=device, with_vae=a.decode)

    print("loading LoRA bundle...")
    lora_args = make_args(ckpt, prompts[0], seeds[0], a, with_lora=True)
    lora_bundle = build_inference_bundle(lora_args, device=device, with_vae=a.decode)

    artifacts: list[str] = []
    per_prompt: list[dict] = []
    ratios: list[float] = []

    for pi, prompt in enumerate(prompts):
        print(f"\n[{pi + 1}/{len(prompts)}] {prompt}")
        base_runs, lora_runs = [], []
        for seed in seeds:
            tag = f"p{pi}_s{seed}"
            print(f"  seed {seed}: base...", flush=True)
            base_runs.append(
                run_one(
                    base_bundle, ckpt, prompt, seed, a, False, run_dir, f"{tag}_base"
                )
            )
            print(f"  seed {seed}: lora...", flush=True)
            lora_runs.append(
                run_one(
                    lora_bundle, ckpt, prompt, seed, a, True, run_dir, f"{tag}_lora"
                )
            )

        # seed-averaged tail-TV maps
        base_tail = np.mean([m["tv_tail"] for m in base_runs], axis=0)
        lora_tail = np.mean([m["tv_tail"] for m in lora_runs], axis=0)
        base_bt = np.mean([m["backtrack_tail"] for m in base_runs], axis=0)
        lora_bt = np.mean([m["backtrack_tail"] for m in lora_runs], axis=0)
        # wander = total tail path / total tail net displacement (ratio of sums,
        # robust to per-patch division); =1 straight resolve, >1 backtracking.
        base_net = np.sum([m["net_tail"] for m in base_runs], axis=0)
        lora_net = np.sum([m["net_tail"] for m in lora_runs], axis=0)
        base_path = np.sum([m["tv_tail"] for m in base_runs], axis=0)
        lora_path = np.sum([m["tv_tail"] for m in lora_runs], axis=0)
        base_wander = float(base_path.sum() / (base_net.sum() + 1e-9))
        lora_wander = float(lora_path.sum() / (lora_net.sum() + 1e-9))

        # cross-seed multimodality maps (regime classifier)
        base_cv = crossseed_var([m["ref_x0"] for m in base_runs])
        lora_cv = crossseed_var([m["ref_x0"] for m in lora_runs])

        ratio = float(lora_tail.mean() / (base_tail.mean() + 1e-9))
        ratios.append(ratio)
        entry = {
            "prompt": prompt,
            "base_tail_tv_mean": float(base_tail.mean()),
            "lora_tail_tv_mean": float(lora_tail.mean()),
            "lora_over_base_ratio": ratio,
            # wander ratio = path/net in tail — the actual non-monotonicity signal
            "base_tail_wander": base_wander,
            "lora_tail_wander": lora_wander,
            "wander_ratio_lora_over_base": lora_wander / (base_wander + 1e-9),
            # backtrack energy (direction reversals) in tail
            "base_backtrack_mean": float(base_bt.mean()),
            "lora_backtrack_mean": float(lora_bt.mean()),
            "backtrack_ratio_lora_over_base": float(
                lora_bt.mean() / (base_bt.mean() + 1e-9)
            ),
            # regime: does instability coincide with cross-seed ambiguity?
            "pearson_crossvar_tailtv_base": pearson(base_cv, base_tail),
            "pearson_crossvar_tailtv_lora": pearson(lora_cv, lora_tail),
        }
        per_prompt.append(entry)
        print(
            f"  -> wander base {base_wander:.3f} lora {lora_wander:.3f} "
            f"(x{entry['wander_ratio_lora_over_base']:.2f}); "
            f"backtrack x{entry['backtrack_ratio_lora_over_base']:.2f}; "
            f"tail-TV x{ratio:.2f}; regime r={entry['pearson_crossvar_tailtv_lora']:.2f}"
        )

        artifacts.append(
            save_heatmaps(
                run_dir,
                f"p{pi}",
                {"tv_tail": base_tail},
                {"tv_tail": lora_tail},
                lora_cv,
            )
        )

    if a.decode:
        artifacts += sorted(pn.name for pn in run_dir.glob("final_*.png"))

    overall = {
        "n_prompts": len(prompts),
        "n_seeds": len(seeds),
        "mean_lora_over_base_ratio": float(np.mean(ratios)),
        "median_lora_over_base_ratio": float(np.median(ratios)),
        "mean_wander_ratio_lora_over_base": float(
            np.mean([e["wander_ratio_lora_over_base"] for e in per_prompt])
        ),
        "mean_backtrack_ratio_lora_over_base": float(
            np.mean([e["backtrack_ratio_lora_over_base"] for e in per_prompt])
        ),
        "mean_base_tail_wander": float(
            np.mean([e["base_tail_wander"] for e in per_prompt])
        ),
        "mean_lora_tail_wander": float(
            np.mean([e["lora_tail_wander"] for e in per_prompt])
        ),
        "mean_pearson_crossvar_tailtv_lora": float(
            np.nanmean([e["pearson_crossvar_tailtv_lora"] for e in per_prompt])
        ),
    }
    metrics = {"overall": overall, "per_prompt": per_prompt, "split_sigma": a.split}

    write_result(
        run_dir,
        script=__file__,
        args=a,
        metrics=metrics,
        label=a.label,
        artifacts=artifacts,
        device=device,
    )
    print("\n=== summary ===")
    print(
        f"tail wander (path/net): base {overall['mean_base_tail_wander']:.3f} "
        f"-> lora {overall['mean_lora_tail_wander']:.3f} "
        f"(x{overall['mean_wander_ratio_lora_over_base']:.3f})"
    )
    print(
        "  wander>1 ⇒ x̂₀ backtracks in the tail (the contradiction). "
        "ratio>1 ⇒ the LoRA worsens it."
    )
    print(
        f"mean backtrack-energy ratio (lora/base): "
        f"{overall['mean_backtrack_ratio_lora_over_base']:.3f}"
    )
    print(f"mean lora/base tail-TV ratio: {overall['mean_lora_over_base_ratio']:.3f}")
    print(
        "  >1 ⇒ the LoRA amplifies tail instability (your training is creating "
        "contradiction); ≈1 ⇒ inherited from base."
    )
    print(f"mean regime r (lora): {overall['mean_pearson_crossvar_tailtv_lora']:.3f}")
    print(
        "  high+ ⇒ instability tracks cross-seed ambiguity = mode-averaging "
        "(training-reducible); ≈0 ⇒ diffuse score error."
    )
    print(f"\nresult.json + heatmaps in: {run_dir}")


if __name__ == "__main__":
    main()
