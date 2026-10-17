#!/usr/bin/env python3
"""Turbo 4-step schedule allocation probe — where does the coarse schedule hurt?

The shipped 4-step student runs the flow_shift=3 schedule, whose σ-nodes are
[1.0, 0.9, 0.75, 0.5, 0] — i.e. the **final step σ0.5→0 covers the whole low-σ
detail band in one Euler jump** (Δσ=0.5, vs 0.1 for the first step). turbo's known
weak spots (blur / bg detail) live exactly there, and the `get_timesteps_sigmas`
docstring already notes local discretization error concentrates in σ<0.45.

This probe asks, for the **student's own PF-ODE field** (cfg=1, its deployment
guidance), where the 4 steps are mis-allocated — no training, no VAE decode.

Method (per prompt × seed, all latent-space):
  1. Fine reference rollout: N_fine Euler steps (flow_shift=3, cfg=1). x_fine(σ)
     ≈ the exact ODE solution of the student field; record x and ‖Δv‖ per node.
  2. Per coarse step [σ_a→σ_b] of the real 4-step schedule, measure the LOCAL
     truncation error: start from the accurate x_fine(σ_a), take ONE big Euler
     step with v(x_fine(σ_a),σ_a), compare to x_fine(σ_b). This isolates the
     schedule-coarseness error of that step from accumulated drift.
  3. Turning integrand g(σ)=‖dv/dσ‖ along the fine trajectory → cumulative C(σ);
     the "equal-turning" 4-step boundaries (C at k/4) are a training-free
     suggestion for a better allocation, compared against flow_shift=3.

    python bench/turbo/probe_schedule.py --turbo output/ckpt/anima_turbo_R/anima_turbo_R_4500.safetensors \
        --n_prompts 6 --n_seeds 2 --n_fine 128 --student_steps 4 --size 1024 1024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from anima_lora import GenerationRequest  # noqa: E402
from bench._anima import add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.runtime.harness import build_anima  # noqa: E402
from library.inference.models import load_shared_models  # noqa: E402
from library.inference.text import prepare_text_inputs  # noqa: E402
from library.inference.sampling import get_timesteps_sigmas  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

REAL_PROMPTS = REPO_ROOT / "bench" / "turbo" / "real_prompts.txt"


def _load_prompts(n: int) -> list[str]:
    lines = [ln.strip() for ln in REAL_PROMPTS.read_text().splitlines() if ln.strip()]
    return lines[:n] if n > 0 else lines


def _interp_x(
    sigmas_desc: list[float], xs: list[torch.Tensor], s: float
) -> torch.Tensor:
    """Linear-in-σ interpolation of the fine trajectory at σ=s (σ list descending)."""
    if s >= sigmas_desc[0]:
        return xs[0]
    if s <= sigmas_desc[-1]:
        return xs[-1]
    for i in range(len(sigmas_desc) - 1):
        hi, lo = sigmas_desc[i], sigmas_desc[i + 1]
        if lo <= s <= hi:
            w = (s - lo) / (hi - lo + 1e-12)  # w=1 at hi
            return w * xs[i] + (1.0 - w) * xs[i + 1]
    return xs[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--turbo", required=True, help="turbo student LoRA .safetensors")
    p.add_argument("--n_prompts", type=int, default=6)
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--n_fine", type=int, default=128, help="fine reference step count")
    p.add_argument("--student_steps", type=int, default=4)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="turbo-schedule")
    opts = p.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R, (H, W) = opts.n_seeds, opts.size
    prompts = _load_prompts(opts.n_prompts)

    req = GenerationRequest(
        dit=opts.dit,
        vae=opts.vae,
        text_encoder=opts.text_encoder,
        prompt=prompts[0],
        save_path="output/tests/_turbo_sched.png",
        infer_steps=opts.student_steps,
        guidance_scale=1.0,
        image_size=(H, W),
        seed=opts.seed,
    )
    args = req.to_args()
    args.device = device
    args.dtype = "bf16"
    args.attn_mode = getattr(args, "attn_mode", "flash") or "flash"
    args.gradient_checkpointing = False
    args.compile = False
    args.compile_mode = None

    # Schedules (σ on [0,1]); student = coarse, fine = reference. Same flow_shift.
    sig_fine = get_timesteps_sigmas(opts.n_fine, opts.flow_shift, "cpu")[1].tolist()
    sig_coarse = get_timesteps_sigmas(opts.student_steps, opts.flow_shift, "cpu")[
        1
    ].tolist()
    print(
        f"[probe] coarse {opts.student_steps}-step σ = "
        f"{[round(s, 3) for s in sig_coarse]}"
    )

    print("[probe] loading student (base + turbo LoRA) …")
    bundle = build_anima(args, dit_path=opts.dit, adapter=opts.turbo, train_mode=False)
    anima = bundle.anima
    shared = load_shared_models(args)

    n_coarse = opts.student_steps
    # accumulators over prompt×seed
    trunc_abs = [0.0] * n_coarse  # ‖x_coarse_b - x_fine_b‖
    trunc_rel = [0.0] * n_coarse  # … / ‖x_fine_b - x_fine_a‖ (segment displacement)
    seg_turn = [0.0] * n_coarse  # ∫‖Δv‖ over each coarse interval
    count = 0
    # fine turning integrand binned by σ for the equal-turning schedule
    turn_nodes: list[tuple[float, float]] = []  # (σ_mid, ‖Δv‖) accumulated mean

    for pi, prompt in enumerate(prompts):
        ctx, _ = prepare_text_inputs(
            args,
            device=device,
            anima=anima,
            shared_models=shared,
            prompt=prompt,
            negative_prompt="",
            text_encoder_path=args.text_encoder,
        )
        e_pos = ctx["embed"][0].to(device, torch.bfloat16).expand(R, -1, -1)
        pad = torch.zeros(R, 1, H // 8, W // 8, dtype=torch.bfloat16, device=device)
        gens = [
            torch.Generator(device="cpu").manual_seed(opts.seed + pi * 100 + j)
            for j in range(R)
        ]
        x = torch.stack(
            [
                torch.randn(16, 1, H // 8, W // 8, generator=g, dtype=torch.float32)
                for g in gens
            ]
        ).to(device, torch.bfloat16)

        # ---- fine reference rollout ----
        xs: list[torch.Tensor] = []  # cpu fp32, per fine node
        prev_v = None
        per_node_turn: list[float] = []  # ‖Δv‖ per-sample mean, aligned to node i
        for i in range(opts.n_fine):
            s_i = torch.tensor(sig_fine[i], device=device, dtype=torch.bfloat16)
            v = anima(x, s_i.expand(R), e_pos, padding_mask=pad).float()
            xs.append(x.float().cpu())
            if prev_v is not None:
                dv = (v - prev_v).reshape(R, -1).norm(dim=1).mean().item()
                per_node_turn.append(dv)
                # (σ_hi, σ_lo, ‖Δv‖) over the fine sub-step [sig_fine[i], sig_fine[i-1]]
                turn_nodes.append((sig_fine[i - 1], sig_fine[i], dv))
            prev_v = v
            dt = sig_fine[i] - sig_fine[i + 1]
            x = (x.float() - dt * v).to(x.dtype)
        xs.append(x.float().cpu())  # σ=0 endpoint

        # ---- per coarse-step local truncation error (accurate start state) ----
        for k in range(n_coarse):
            s_a, s_b = sig_coarse[k], sig_coarse[k + 1]
            x_a = _interp_x(sig_fine, xs, s_a).to(device, torch.bfloat16)
            x_fb = _interp_x(sig_fine, xs, s_b).to(device)  # fp32
            v_a = anima(
                x_a,
                torch.tensor(s_a, device=device, dtype=torch.bfloat16).expand(R),
                e_pos,
                padding_mask=pad,
            ).float()
            x_cb = x_a.float() - (s_a - s_b) * v_a  # one big Euler step
            seg = (x_fb - x_a.float()).reshape(R, -1).norm(dim=1)  # displacement
            err = (x_cb - x_fb).reshape(R, -1).norm(dim=1)  # local error
            trunc_abs[k] += err.mean().item()
            trunc_rel[k] += (err / (seg + 1e-6)).mean().item()
            # ∫‖Δv‖ over [s_a, s_b): sum of fine per_node_turn whose σ_mid in band
            band_sum = sum(
                dv
                for (s_hi, s_lo, dv) in turn_nodes[-len(per_node_turn) :]
                if s_b <= 0.5 * (s_hi + s_lo) < s_a
            )
            seg_turn[k] += band_sum
        count += 1
        del e_pos, pad, xs
        clean_memory_on_device(device)
        print(f"[probe] prompt {pi + 1}/{len(prompts)} done")

    del anima, bundle, shared
    clean_memory_on_device(device)

    trunc_abs = [v / count for v in trunc_abs]
    trunc_rel = [v / count for v in trunc_rel]
    seg_turn = [v / count for v in seg_turn]
    tot_abs = sum(trunc_abs) + 1e-12
    tot_turn = sum(seg_turn) + 1e-12

    # Two schedule criteria from the fine turning curve. For each fine sub-step
    # [σ_hi, σ_lo] with ‖Δv‖=dv and width w=σ_hi-σ_lo: g=dv/w ≈ ‖dv/dσ‖.
    #   arc-length (equal-turning): weight = dv  (∫g dσ)         → Δσ ∝ g⁻¹
    #   error-equidistribution:     weight = √(g)·w = √(dv·w)    → Δσ ∝ g^(−1/2)
    # The error criterion is the one that minimizes max Euler truncation error
    # (error ∝ g·Δσ²); arc-length does NOT and ≈ reproduces flow_shift=3.
    def _equipartition(weighted, n_steps):
        # weighted: list of (sigma_lo, weight), accumulated σ-descending.
        nodes = sorted(weighted, key=lambda t: -t[0])
        csum, cum = [], 0.0
        for slo, wt in nodes:
            cum += wt
            csum.append((slo, cum))
        total = cum + 1e-12
        bounds = [1.0]
        for k in range(1, n_steps):
            target = total * k / n_steps
            bounds.append(round(next((s for s, c in csum if c >= target), 0.0), 3))
        bounds.append(0.0)
        return bounds

    arc_w = [(s_lo, dv) for (s_hi, s_lo, dv) in turn_nodes]
    err_w = [(s_lo, (dv * (s_hi - s_lo)) ** 0.5) for (s_hi, s_lo, dv) in turn_nodes]
    eq_bounds = _equipartition(arc_w, n_coarse)  # equal-turning (g¹)
    err_bounds = _equipartition(err_w, n_coarse)  # error-equidist (g^½), 4-step
    err_bounds_p1 = _equipartition(err_w, n_coarse + 1)  # error-equidist, 5-step
    # binned turning curve (σ midpoint → mean g) for persistence / re-partitioning
    from collections import defaultdict

    gbin = defaultdict(lambda: [0.0, 0])
    for s_hi, s_lo, dv in turn_nodes:
        key = round(0.5 * (s_hi + s_lo), 3)
        g = dv / max(s_hi - s_lo, 1e-9)
        gbin[key][0] += g
        gbin[key][1] += 1
    turn_curve = sorted(
        ({"sigma": k, "g": v[0] / v[1]} for k, v in gbin.items()),
        key=lambda d: -d["sigma"],
    )

    # ---- report ----
    print(
        f"\n{'coarse step':>12} {'σ_a→σ_b':>14} {'Δσ':>6} "
        f"{'trunc_abs':>10} {'%err':>6} {'trunc_rel':>10} {'turn%':>6}"
    )
    for k in range(n_coarse):
        s_a, s_b = sig_coarse[k], sig_coarse[k + 1]
        print(
            f"{k:>12} {f'{s_a:.3f}→{s_b:.3f}':>14} {s_a - s_b:>6.3f} "
            f"{trunc_abs[k]:>10.2f} {100 * trunc_abs[k] / tot_abs:>5.1f}% "
            f"{trunc_rel[k]:>10.3f} {100 * seg_turn[k] / tot_turn:>5.1f}%"
        )
    worst = max(range(n_coarse), key=lambda k: trunc_abs[k])
    print(
        f"\nworst step (local truncation): step {worst} "
        f"σ{sig_coarse[worst]:.3f}→{sig_coarse[worst + 1]:.3f} "
        f"({100 * trunc_abs[worst] / tot_abs:.0f}% of total error)"
    )
    print(
        f"flow_shift={opts.flow_shift} boundaries        : "
        f"{[round(s, 3) for s in sig_coarse]}"
    )
    print(
        f"equal-turning (arc-length, g¹)    : {eq_bounds}   "
        f"← ≈flow_shift, does NOT relieve the tail"
    )
    print(
        f"error-equidist (g^½), {n_coarse}-step      : {err_bounds}   "
        f"← the schedule that minimizes max truncation error"
    )
    print(
        f"error-equidist (g^½), {n_coarse + 1}-step      : {err_bounds_p1}   "
        f"← if you can afford one more step"
    )

    run_dir = make_run_dir("turbo", label=opts.label)
    write_result(
        run_dir,
        script=__file__,
        args=vars(opts),
        metrics={
            "coarse_sigmas": [round(s, 4) for s in sig_coarse],
            "per_step": [
                {
                    "step": k,
                    "sigma_a": round(sig_coarse[k], 4),
                    "sigma_b": round(sig_coarse[k + 1], 4),
                    "d_sigma": round(sig_coarse[k] - sig_coarse[k + 1], 4),
                    "trunc_abs": trunc_abs[k],
                    "trunc_abs_frac": trunc_abs[k] / tot_abs,
                    "trunc_rel": trunc_rel[k],
                    "turn_frac": seg_turn[k] / tot_turn,
                }
                for k in range(n_coarse)
            ],
            "worst_step": worst,
            "equal_turning_boundaries": eq_bounds,
            "error_equidist_boundaries": err_bounds,
            "error_equidist_boundaries_plus1": err_bounds_p1,
            "turning_curve": turn_curve,
            "flow_shift": opts.flow_shift,
            "n_fine": opts.n_fine,
            "n_prompts": len(prompts),
            "n_seeds": R,
            "size": [H, W],
        },
        device=device,
    )
    print(f"\nresult.json → {run_dir}")


if __name__ == "__main__":
    main()
