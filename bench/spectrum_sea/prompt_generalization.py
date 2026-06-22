#!/usr/bin/env python3
"""Does the SEA-schedule δ generalize across prompts? (spectrum_sea_schedule.md)

The SEA trigger (``networks/spectrum.py``, ``schedule="sea"``) calibrates a single
δ once and reuses it — prompt is deliberately *not* in the δ cache key, on the
premise that a fixed δ with a per-prompt-varying refresh *pattern* is the whole
point of content-adaptivity. This bench tests that premise on **real captions
from post_image_dataset/** (the training distribution — generic prompts gave
misleading SEA signal before; see ``project_dynamic_spectrum_vnorm_gate``):

  Q1 (matched compute)  Calibrate δ on prompt #0 (exactly as the live disk-cache
                        does), apply to every prompt → how tight is the resulting
                        actual-forward-count distribution? Wide spread = the
                        single-δ "matched compute" claim leaks per prompt.
  Q2 (right-δ spread)   Calibrate δ *per prompt* to the target ratio → how much
                        does the ideal δ itself vary? (cv = std/mean.)
  Q3 (pattern variety)  Do prompts pick *different* steps to skip? Mean pairwise
                        disagreement over the decision region. The blind window
                        is identical for all prompts (=0) by construction — any
                        positive number is content-adaptivity the window can't do.
  Q4 (reallocation)     Per-step actual-rate (averaged over prompts) vs the blind
                        window — does SEA move compute out of the σ<0.45 tail
                        into mid-σ, as the findings doc predicts?

All SEA distances are measured offline on the **full-compute** euler trajectory
(one all-actual generate per prompt — the same capture seam run_bench.py uses),
then the accumulate-until-δ rule is replayed. No quality claim here — this is the
δ-portability check that gates whether the single-key cache design is sound
before the ComfyUI mirror. Quality is the P2 CMMD A/B.

Usage::

    python bench/spectrum_sea/prompt_generalization.py --n-prompts 16 --steps 28
    python bench/spectrum_sea/prompt_generalization.py --prompts-file my.txt
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
from bench.spectrum_sea.run_bench import Capture, make_args  # noqa: E402

import anima_lora  # noqa: E402
from library.runtime.harness import build_inference_bundle  # noqa: E402
from networks.spectrum_sea import (  # noqa: E402
    l1rel,
    sea_filter,
    solve_delta_for_refresh_ratio,
)

CAPTION_ROOT = REPO_ROOT / "post_image_dataset" / "resized"
TAIL_SPLIT = 0.45  # σ below this = resolve tail (Anima x̂₀ done by ~0.45)


# --------------------------------------------------------------------------- #
# real-caption sampling
# --------------------------------------------------------------------------- #
def sample_real_prompts(n: int, root: Path = CAPTION_ROOT) -> list[str]:
    """Deterministically sample ``n`` real captions from the training set.

    Plain ``.txt`` sidecars only (skip ``.variants.txt`` auto-gen files); first
    non-empty line of each; deduped; evenly strided over the sorted file list so
    the sample is reproducible and spans many artists rather than one folder.
    """
    paths = sorted(
        p for p in root.rglob("*.txt") if not p.name.endswith(".variants.txt")
    )
    seen: set[str] = set()
    caps: list[str] = []
    for p in paths:
        try:
            line = next(
                (ln.strip() for ln in p.read_text().splitlines() if ln.strip()), ""
            )
        except OSError:
            continue
        if not line or line.startswith("#") or line in seen:
            continue
        seen.add(line)
        caps.append(line)
    if not caps:
        raise SystemExit(f"no captions found under {root}")
    if len(caps) <= n:
        return caps
    stride = len(caps) / n
    return [caps[int(i * stride)] for i in range(n)]


# --------------------------------------------------------------------------- #
# offline SEA-schedule simulation (mirrors networks/spectrum.py exactly)
# --------------------------------------------------------------------------- #
def sea_step_distances(steps: list[dict], beta: float) -> list[float]:
    """Per-step SEA distance d_i = l1rel(SEA(x_t_i), SEA(x_t_{i-1})), d_0 = 0."""
    steps = sorted(steps, key=lambda r: r["i"])
    dists = [0.0]
    for i in range(1, len(steps)):
        xt, xt_p = steps[i]["x_t"], steps[i - 1]["x_t"]
        s, s_p = steps[i]["sigma"], steps[i - 1]["sigma"]
        dists.append(l1rel(sea_filter(xt, s, beta), sea_filter(xt_p, s_p, beta)))
    return dists


def simulate_modes(dists: list[float], warmup: int, stop_at: int, delta: float):
    """Replay the live accumulate-until-δ rule → per-step actual/cached mask.

    Matches the loop in ``spectrum_denoise``: distance accrues every step, the
    forced warmup/stop steps and any decision-step crossing δ are actual, and
    every actual forward resets the accumulator.
    """
    n = len(dists)
    accum = 0.0
    modes = []
    for i in range(n):
        accum += dists[i]  # dists[0] == 0
        if i < warmup or i >= stop_at:
            actual = True
        else:
            actual = accum >= delta
        modes.append(actual)
        if actual:
            accum = 0.0
    return modes


def decision_dists(dists: list[float], warmup: int, stop_at: int) -> list[float]:
    """The distance slice the live auto-δ calibration sees (decision region)."""
    return dists[warmup:stop_at]


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-prompts", type=int, default=16)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow-shift", dest="flow_shift", type=float, default=3.0)
    p.add_argument("--size", type=int, default=1024, help="pixel edge (square)")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--warmup", type=int, default=7)
    p.add_argument(
        "--stop-caching-step",
        dest="stop_caching_step",
        type=int,
        default=27,
        help="force actual from this step (mirrors --spectrum_stop_caching_step)",
    )
    p.add_argument(
        "--refresh-ratio",
        dest="refresh_ratio",
        type=float,
        default=-1.0,
        help="<=0 (default) matches the growing-window's own refresh fraction at "
        "this step count → compute-matched arms. A fixed 0.62 over-computes at "
        "28 steps; never hard-code it.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--no-compile", dest="no_compile", action="store_true")
    a = p.parse_args()

    if a.prompts_file:
        prompts = [
            ln.strip()
            for ln in Path(a.prompts_file).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        prompts = sample_real_prompts(a.n_prompts)

    from networks.spectrum import _window_decision_fraction
    from bench.spectrum_sea.run_bench import FLEX, WINDOW

    warmup = a.warmup
    stop_at = a.stop_caching_step if a.stop_caching_step >= 0 else a.steps - 3
    n_decision = max(0, stop_at - warmup)
    if a.refresh_ratio <= 0.0:  # match the window schedule (compute-matched arms)
        a.refresh_ratio = _window_decision_fraction(
            a.steps, warmup, stop_at, WINDOW, FLEX
        )
    target_refreshes = max(1, round(a.refresh_ratio * n_decision))

    ckpt = anima_lora.default_checkpoints()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir("spectrum_sea", label=a.label or "prompt_gen")
    mode = "eager" if a.no_compile else "compiled"
    print(f"run dir: {run_dir}")
    print(
        f"{len(prompts)} REAL prompts, {a.steps} steps, cfg {a.cfg}, β {a.beta}, "
        f"warmup {warmup}, stop {stop_at} → {n_decision} decision steps, "
        f"target {target_refreshes} refreshes ({a.refresh_ratio:.2f}), {mode}"
    )

    bundle = build_inference_bundle(
        make_args(ckpt, prompts[0], a.seed, a), device=device, with_vae=False
    )

    # ---- capture full-compute trajectory + SEA distances per prompt ---------
    per_prompt = []
    for pi, prompt in enumerate(prompts):
        print(f"\n[{pi + 1}/{len(prompts)}] {prompt[:70]}")
        cap = Capture()
        h = bundle.model.final_layer.register_forward_pre_hook(
            cap.final_layer_pre_hook
        )
        try:
            with cap.patched_step():
                bundle.generate(make_args(ckpt, prompt, a.seed, a))
        finally:
            h.remove()
        dists = sea_step_distances(cap.steps, a.beta)
        sigmas = [s["sigma"] for s in sorted(cap.steps, key=lambda r: r["i"])]
        dd = decision_dists(dists, warmup, stop_at)
        delta_opt = solve_delta_for_refresh_ratio(dd, a.refresh_ratio)
        per_prompt.append(
            {
                "prompt": prompt,
                "dists": dists,
                "sigmas": sigmas,
                "decision_dists": dd,
                "decision_sum": float(sum(dd)),
                "delta_opt": delta_opt,
            }
        )
        print(
            f"  Σd(decision)={sum(dd):.3f}  per-prompt-ideal δ={delta_opt:.4f}"
        )

    n = len(per_prompt[0]["dists"])

    # ---- Q1: calibrate δ on prompt #0, apply to all (the live behavior) ------
    delta_p0 = per_prompt[0]["delta_opt"]
    actual_counts = []
    decision_actual_counts = []
    masks = []
    for pp in per_prompt:
        modes = simulate_modes(pp["dists"], warmup, stop_at, delta_p0)
        masks.append(modes)
        actual_counts.append(int(sum(modes)))
        decision_actual_counts.append(int(sum(modes[warmup:stop_at])))
    actual_counts = np.array(actual_counts)
    dec_counts = np.array(decision_actual_counts)
    target_total = warmup + target_refreshes + (n - stop_at)
    within1 = int(np.sum(np.abs(actual_counts - target_total) <= 1))

    # ---- Q2: spread of the per-prompt-ideal δ --------------------------------
    deltas_opt = np.array([pp["delta_opt"] for pp in per_prompt])
    delta_cv = float(deltas_opt.std() / (deltas_opt.mean() + 1e-12))

    # ---- Q3: pattern disagreement over the decision region -------------------
    dec_masks = np.array([m[warmup:stop_at] for m in masks], dtype=float)
    pair_disagree = []
    for i in range(len(dec_masks)):
        for j in range(i + 1, len(dec_masks)):
            pair_disagree.append(float(np.mean(dec_masks[i] != dec_masks[j])))
    mean_disagree = float(np.mean(pair_disagree)) if pair_disagree else 0.0
    per_step_actual_rate = dec_masks.mean(axis=0)  # over prompts, per decision step

    # ---- Q4: reallocation vs blind window (per-step actual rate by σ) --------
    # Blind window mask (identical for all prompts).
    from bench.spectrum_sea.run_bench import blind_schedule

    blind = blind_schedule(a.steps, warmup, WINDOW, FLEX)
    blind_actual = np.array([m == "actual" for m in blind], dtype=float)
    sea_actual_full = np.array(masks, dtype=float).mean(axis=0)  # per step, over prompts
    sig0 = np.array(per_prompt[0]["sigmas"])
    tail = sig0 < TAIL_SPLIT
    midhi = ~tail
    sea_tail_rate = float(sea_actual_full[tail].mean()) if tail.any() else float("nan")
    blind_tail_rate = float(blind_actual[tail].mean()) if tail.any() else float("nan")
    sea_mid_rate = float(sea_actual_full[midhi].mean()) if midhi.any() else float("nan")
    blind_mid_rate = (
        float(blind_actual[midhi].mean()) if midhi.any() else float("nan")
    )

    metrics = {
        "config": {
            "n_prompts": len(prompts),
            "steps": a.steps,
            "cfg": a.cfg,
            "beta": a.beta,
            "warmup": warmup,
            "stop_at": stop_at,
            "n_decision": n_decision,
            "refresh_ratio": a.refresh_ratio,
            "target_refreshes": target_refreshes,
            "target_total_actual": target_total,
        },
        "q1_single_delta_compute": {
            "delta_from_prompt0": delta_p0,
            "total_actual_mean": float(actual_counts.mean()),
            "total_actual_std": float(actual_counts.std()),
            "total_actual_min": int(actual_counts.min()),
            "total_actual_max": int(actual_counts.max()),
            "decision_actual_mean": float(dec_counts.mean()),
            "decision_actual_std": float(dec_counts.std()),
            "within_1_of_target": f"{within1}/{len(prompts)}",
            "per_prompt_total_actual": actual_counts.tolist(),
        },
        "q2_ideal_delta_spread": {
            "delta_mean": float(deltas_opt.mean()),
            "delta_std": float(deltas_opt.std()),
            "delta_cv": delta_cv,
            "delta_min": float(deltas_opt.min()),
            "delta_max": float(deltas_opt.max()),
        },
        "q3_pattern_variety": {
            "mean_pairwise_disagreement": mean_disagree,
            "per_decision_step_actual_rate": per_step_actual_rate.tolist(),
            "blind_disagreement": 0.0,
        },
        "q4_reallocation": {
            "sea_tail_actual_rate": sea_tail_rate,
            "blind_tail_actual_rate": blind_tail_rate,
            "sea_midhi_actual_rate": sea_mid_rate,
            "blind_midhi_actual_rate": blind_mid_rate,
        },
        "per_prompt": [
            {
                "prompt": pp["prompt"][:120],
                "delta_opt": pp["delta_opt"],
                "decision_sum": pp["decision_sum"],
            }
            for pp in per_prompt
        ],
    }

    # per-step CSV (re-analysis without another GPU run)
    import csv

    with (run_dir / "per_step.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["prompt_idx", "step", "sigma", "sea_dist", "actual_p0delta"])
        for pi, pp in enumerate(per_prompt):
            for i in range(n):
                wr.writerow(
                    [pi, i, pp["sigmas"][i], pp["dists"][i], int(masks[pi][i])]
                )

    write_result(
        run_dir,
        script=__file__,
        args=a,
        metrics=metrics,
        label=a.label,
        artifacts=["per_step.csv"],
        device=device,
    )

    # ---- verdict --------------------------------------------------------------
    print("\n=== Q1  single-δ matched compute (δ from prompt #0 applied to all) ===")
    print(
        f"  δ={delta_p0:.4f}  total actual forwards: "
        f"{actual_counts.mean():.1f} ± {actual_counts.std():.1f} "
        f"(range {actual_counts.min()}–{actual_counts.max()}, target {target_total}); "
        f"within ±1 of target: {within1}/{len(prompts)}"
    )
    print("=== Q2  per-prompt IDEAL δ spread (how prompt-specific is the right δ) ===")
    print(
        f"  δ = {deltas_opt.mean():.4f} ± {deltas_opt.std():.4f}  "
        f"(cv={delta_cv:.2f}, range {deltas_opt.min():.4f}–{deltas_opt.max():.4f})"
    )
    print("=== Q3  pattern variety (0 = blind window; >0 = content-adaptive) ===")
    print(f"  mean pairwise step-disagreement = {mean_disagree:.2%}")
    print("=== Q4  reallocation vs blind window (actual-forward rate) ===")
    print(
        f"  σ<{TAIL_SPLIT} tail : SEA {sea_tail_rate:.0%}  vs blind {blind_tail_rate:.0%}"
    )
    print(
        f"  σ≥{TAIL_SPLIT} mid+ : SEA {sea_mid_rate:.0%}  vs blind {blind_mid_rate:.0%}"
    )
    print(f"\nresult.json + per_step.csv in: {run_dir}")


if __name__ == "__main__":
    main()
