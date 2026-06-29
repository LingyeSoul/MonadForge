#!/usr/bin/env python3
"""Foresight Guidance (FSG) — Phase-0 golden-path premise probe.

PAPER. "Towards a Golden Classifier-Free Guidance Path via Foresight Fixed
Point Iterations" (NeurIPS 2025, arXiv 23177). FSG reframes CFG as fixed-point
*calibration*: at scheduled **early** timesteps it runs K forward-backward
iterations over a **long interval** Δσ to pull the latent x_t onto a *golden
path* — a state x̂ that *unconditionally* generates the conditioned image, i.e.
where the conditional and unconditional predictions agree. Their Fig 2(b) is the
load-bearing evidence: the *prediction gap* ‖ε^c(x̂) − ε^u(x̂)‖ shrinks after
calibration. The whole method (and the paper's Theorem 1 contraction bound)
rests on that mechanism existing.

THE QUESTION (go/no-go for building FSG on Anima). FSG was validated only on
ε-prediction DDIM/DDPM (SDXL, SD2.1, Hunyuan-DiT, ImageNet-DiT). Anima is
**flow-matching velocity-prediction**, sampled at **CFG=4 / er_sde**. Does the
golden-path mechanism even exist on Anima's velocity field? If the fixed-point
operator doesn't contract, or the prediction gap doesn't shrink under iteration,
FSG is dead on arrival here — and we close the line for the cost of one
afternoon instead of a doomed plugin build. (This echoes, but is mechanistically
distinct from, the closed σ-reshape / TRDI-inversion lines — those reshaped the
*schedule*; FSG iterates a *consistency* operator. It is also NOT the null-TTA
failure mode: FSG needs no quality reward, only a fixed point.)

THE OPERATOR (flow-matching translation of FSG's DDIM forward-backward). At a
probed σ with latent x, interval Δσ, guidance γ:

    v^γ(x,σ) = v^u(x,σ) + γ·(v^c(x,σ) − v^u(x,σ))      # CFG-guided velocity
    forward  :  x' = x − Δσ · v^γ(x, σ)                  # denoise σ → σ−Δσ (cond)
    invert   :  x'' = x' + Δσ · v^u(x', σ−Δσ)            # re-noise back (uncond)
    F(x) = x''                                            # one calibration step

This is the exact reversible Euler ODE the production sampler already uses
(``library/inference/sampling.py::step``), so no DDIM machinery is needed — the
translation is clean.

CONSTRUCTION (no training, faithful to *generation*). For each prompt × seed we
run a deterministic Euler-CFG trajectory from pure noise (σ=1) — the production
operating point, CFG=4 — and at the steps nearest a target σ grid we *branch*:
clone x_t and iterate F on the clone (the trajectory itself is never disturbed).
Per iteration k we record on the clone x_k:

  * prediction gap   g_k = ‖v^c(x_k,σ) − v^u(x_k,σ)‖ / ‖v^u(x_k,σ)‖   (Fig 2b)
  * step movement    d_k = ‖x_k − x_{k−1}‖
  * contraction      ρ_k = d_k / d_{k−1}                 (well-posedness; <1 good)

PRE-REGISTERED READOUT (the gate). At early σ (σ ≥ --gate_sigma, default 0.75):
  * GAP-SHRINKS  : (g_0 − g_K)/g_0 ≥ --gate_gap (default 0.05), AND
  * CONTRACTS    : median ρ_k < --gate_rho (default 0.95).
Both → **GOLDEN-PATH PRESENT** — proceed to the staged build (cheapest first:
plain CFG×K on the linear operator, then full FSG plugin). Either fails at every
early σ → **CLOSE THE LINE**, write the negative finding.

Phase-0 deliberately stops at the *mechanism*. Whether a smaller gap yields
better samples is Phase-1 (eyeball A/B), because Anima has no trustworthy
quality reward (see memory: null-TTA negative; CMMD/PE are global-tone only).

Run from anima_lora/::

    uv run python bench/fsg/probe_golden_path.py
    uv run python bench/fsg/probe_golden_path.py --adapter output/ckpt/<artist>.safetensors \
        --guidance 4.0 --d_sigma 0.1 --k_iters 4 --num_seeds 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import models as anima_models  # noqa: E402
from library.inference.adapters import (  # noqa: E402
    clear_hydra_fei,
    clear_hydra_sigma,
    compute_and_set_hydra_fei,
    set_hydra_content,
    set_hydra_crossattn,
    set_hydra_sigma,
    set_step_expert_index,
)
from library.inference.sampling import get_timesteps_sigmas  # noqa: E402
from library.inference.text import prepare_text_inputs  # noqa: E402

log = logging.getLogger("bench.fsg.golden_path")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Real training captions are sampled from this dir (Anima tag format) so the
# conditional signal matches the distribution the model was trained on, rather
# than synthetic prose. Override with --prompts / --caption_dir.
DEFAULT_CAPTION_DIR = "post_image_dataset/resized"


def _sample_captions(caption_dir: str, n: int, seed: int) -> list[str]:
    """Random real captions from ``<caption_dir>/**/*.txt`` (skip .variants.txt)."""
    root = Path(caption_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    files = [p for p in root.rglob("*.txt") if not p.name.endswith(".variants.txt")]
    if not files:
        raise SystemExit(f"no caption .txt under {root}")
    rng = np.random.default_rng(seed)
    picks = [files[int(i)] for i in rng.choice(len(files), min(n, len(files)), replace=False)]
    out: list[str] = []
    for p in picks:
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if txt:
            out.append(txt)
    if not out:
        raise SystemExit(f"sampled {len(picks)} captions but all were empty")
    return out
# Empty string = the true CFG-unconditional (golden path = unconditionally
# generates the image). Production renders use a worded negative, but the
# golden-path premise is defined against the *null*. Override with --neg_prompt.
DEFAULT_NEG = ""


def _fmt(x: float, spec: str = ".4f") -> str:
    return format(x, spec) if np.isfinite(x) else "nan"


@torch.no_grad()
def _velocity(anima, x, sigma: float, embed, pad, step_i: int) -> torch.Tensor:
    """Full DiT velocity forward, mirroring generation.py's per-step call.

    Sets the hydra/step/FEI/content/crossattn context exactly as the sampler
    does (all no-ops on a base DiT), then returns the velocity prediction (same
    5D shape as ``x``). ``embed`` selects conditional vs unconditional.
    """
    bs = x.shape[0]
    t_b = torch.full((bs,), float(sigma), device=x.device, dtype=x.dtype)
    set_hydra_sigma(anima, t_b)
    set_step_expert_index(anima, step_i)
    compute_and_set_hydra_fei(anima, x)
    set_hydra_content(anima, embed)
    set_hydra_crossattn(anima, embed)
    return anima(x, t_b, embed, padding_mask=pad)


def _norm(t: torch.Tensor) -> float:
    return float(t.float().pow(2).sum().sqrt())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap, vae=False)  # need --text_encoder; VAE not used (no decode)
    ap.add_argument("--adapter", default=None, help="Optional LoRA checkpoint.")
    ap.add_argument(
        "--prompts", nargs="+", default=None, help="Explicit prompts (skip sampling)."
    )
    ap.add_argument("--caption_dir", default=DEFAULT_CAPTION_DIR)
    ap.add_argument(
        "--num_prompts", type=int, default=6, help="Real captions to sample."
    )
    ap.add_argument("--neg_prompt", default=DEFAULT_NEG)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument(
        "--infer_steps", type=int, default=20, help="Base trajectory step count."
    )
    ap.add_argument("--flow_shift", type=float, default=3.0, help="Canonical schedule.")
    ap.add_argument(
        "--guidance", type=float, default=4.0, help="CFG scale (production = 4)."
    )
    ap.add_argument(
        "--fsg_gamma",
        type=float,
        default=None,
        help="γ for the FSG conditional forward (default: --guidance).",
    )
    ap.add_argument(
        "--probe_sigmas",
        type=float,
        nargs="+",
        default=[0.95, 0.85, 0.75, 0.60, 0.40],
        help="Target σ levels; nearest trajectory step is probed at each.",
    )
    ap.add_argument(
        "--d_sigma",
        type=float,
        default=0.1,
        help="Foresight interval Δσ (absolute, in σ units). FSG: long & early.",
    )
    ap.add_argument(
        "--d_sigma_frac",
        type=float,
        default=None,
        help="If set, Δσ = d_sigma_frac · σ (interval proportional to the current "
        "σ, overriding --d_sigma). This is the paper's actual prescription — App C "
        "Table 9 measures contraction with intervals as a fraction of t (t/4, t/2, "
        "t), and the long-interval operator contracts at *early* t where a fixed "
        "short Δσ does not. Use e.g. 0.5 to test whether the σ≈0.94 divergence is a "
        "fixed-Δσ artifact rather than a real Anima property.",
    )
    ap.add_argument(
        "--k_iters", type=int, default=4, help="Fixed-point iterations per probe."
    )
    ap.add_argument(
        "--cfgpp",
        action="store_true",
        help="Advance the UNDISTURBED trajectory on the CFG++ substrate (App A.2) "
        "instead of plain CFG, so the golden path is measured at the x_t states FSG "
        "actually rides. CFG++ is manifold-constrained — CFG=4 over-shoots "
        "off-manifold at high σ, which may be the real cause of the σ≈0.94 "
        "divergence rather than the golden path itself. The foresight clone operator "
        "is unchanged (it always uses γ-combine forward + v^u backward).",
    )
    ap.add_argument(
        "--cfgpp_lambda",
        type=float,
        default=2.0,
        help="CFG++ strength λ (flow-space: guidance enters as λ·(1−σ')·σ·Δv). "
        "λ≈1.5-2 ≈ CFG=4 total guidance. Only used with --cfgpp.",
    )
    ap.add_argument("--num_seeds", type=int, default=2, help="Noise draws per prompt.")
    # Pre-registered gate.
    ap.add_argument("--gate_sigma", type=float, default=0.75)
    ap.add_argument("--gate_gap", type=float, default=0.05, help="min rel. gap drop")
    ap.add_argument("--gate_rho", type=float, default=0.95, help="max median ρ")
    add_common_args(ap)  # --seed/--device/--dtype/--attn_mode/--compile/...
    args = ap.parse_args()

    prompts = args.prompts or _sample_captions(
        args.caption_dir, args.num_prompts, args.seed
    )
    gamma = args.guidance if args.fsg_gamma is None else args.fsg_gamma
    targets = sorted({float(s) for s in args.probe_sigmas}, reverse=True)

    bundle = build_anima(args, adapter=args.adapter, train_mode=False)
    anima, device, dtype = bundle.anima, bundle.device, bundle.dtype

    timesteps, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sig = [float(s) for s in sigmas]  # length infer_steps+1, σ[0]=1 → σ[-1]=0
    # Map each target σ to the trajectory step whose σ[i] is closest.
    probe_steps: dict[int, float] = {}
    for tgt in targets:
        i = int(min(range(args.infer_steps), key=lambda j: abs(sig[j] - tgt)))
        probe_steps[i] = sig[i]  # dedup if two targets snap to same step
    log.info(
        f"schedule σ[0..]={[round(s, 3) for s in sig[: args.infer_steps]]}\n"
        f"probing steps {sorted(probe_steps)} at σ={[round(probe_steps[i], 3) for i in sorted(probe_steps)]}"
        f"  |  Δσ={f'{args.d_sigma_frac}·σ' if args.d_sigma_frac is not None else args.d_sigma}"
        f"  γ={gamma}  CFG={args.guidance}  K={args.k_iters}"
    )

    C = anima_models.Anima.LATENT_CHANNELS
    hl, wl = args.height // 8, args.width // 8
    pad = torch.zeros(1, 1, hl, wl, dtype=dtype, device=device)

    # records[σ]["gap"] = list of per-(prompt,seed) gap traces (len K+1)
    #          [σ]["rho"] = list of median ρ over the iteration
    #          [σ]["drift"] = list of ‖x_K − x_0‖ / ‖x_0‖
    rec: dict[float, dict[str, list]] = {
        probe_steps[i]: {"gap": [], "rho": [], "drift": []} for i in probe_steps
    }

    for pi, prompt in enumerate(prompts):
        ctx, ctx_null = prepare_text_inputs(
            device=device,
            anima=anima,
            prompt=prompt,
            negative_prompt=args.neg_prompt,
            text_encoder_path=args.text_encoder,
        )
        embed = ctx["embed"][0].to(device, dtype).expand(1, -1, -1)
        nembed = ctx_null["embed"][0].to(device, dtype).expand(1, -1, -1)

        for sj in range(max(1, args.num_seeds)):
            g = torch.Generator(device=device).manual_seed(args.seed + pi * 1000 + sj)
            x = torch.randn(
                (1, C, 1, hl, wl), generator=g, device=device, dtype=dtype
            )

            for i in range(args.infer_steps):
                s_i = sig[i]
                vc = _velocity(anima, x, s_i, embed, pad, i)
                vu = _velocity(anima, x, s_i, nembed, pad, i)
                v_cfg = vu + args.guidance * (vc - vu)

                # ── FSG probe at scheduled steps (branch off a clone) ──────────
                if i in probe_steps:
                    ds_target = (
                        args.d_sigma_frac * s_i
                        if args.d_sigma_frac is not None
                        else args.d_sigma
                    )
                    s_lo = max(s_i - ds_target, 1e-3)
                    dsig = s_i - s_lo
                    xk = x.clone()
                    gap_trace: list[float] = []
                    moves: list[float] = []
                    # gap at k=0 reuses the trajectory's vc/vu (same x, same σ).
                    gap_trace.append(_norm(vc - vu) / max(_norm(vu), 1e-8))
                    x_prev = xk
                    for _k in range(args.k_iters):
                        vc_k = _velocity(anima, xk, s_i, embed, pad, i)
                        vu_k = _velocity(anima, xk, s_i, nembed, pad, i)
                        vg_k = vu_k + gamma * (vc_k - vu_k)
                        x_fwd = xk - dsig * vg_k  # denoise σ → σ−Δσ (guided)
                        vu_lo = _velocity(anima, x_fwd, s_lo, nembed, pad, i)
                        xk = x_fwd + dsig * vu_lo  # invert back (uncond)
                        moves.append(_norm(xk - x_prev))
                        x_prev = xk
                        # gap of the calibrated state (cond/uncond agreement).
                        vc_n = _velocity(anima, xk, s_i, embed, pad, i)
                        vu_n = _velocity(anima, xk, s_i, nembed, pad, i)
                        gap_trace.append(_norm(vc_n - vu_n) / max(_norm(vu_n), 1e-8))
                    # contraction ratios ρ_k = d_k / d_{k−1}.
                    ratios = [
                        moves[m] / moves[m - 1]
                        for m in range(1, len(moves))
                        if moves[m - 1] > 1e-12
                    ]
                    s_key = probe_steps[i]
                    rec[s_key]["gap"].append(gap_trace)
                    rec[s_key]["rho"].append(
                        float(np.median(ratios)) if ratios else float("nan")
                    )
                    rec[s_key]["drift"].append(_norm(xk - x) / max(_norm(x), 1e-8))

                # advance the (undisturbed) trajectory: CFG++ substrate if asked
                # (integrate along v^u with σ-weighted guidance), else Euler-CFG.
                step = sig[i] - sig[i + 1]
                if args.cfgpp:
                    w = (1.0 - sig[i + 1]) * s_i
                    x = x - step * vu - args.cfgpp_lambda * w * (vc - vu)
                else:
                    x = x - step * v_cfg

            log.info(f"  [{pi + 1}/{len(prompts)} seed {sj}] trajectory done")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── aggregate ────────────────────────────────────────────────────────────
    summary: dict[float, dict] = {}
    for s_key, d in rec.items():
        if not d["gap"]:
            continue
        gaps = np.asarray(d["gap"])  # (N, K+1)
        g0 = gaps[:, 0]
        gK = gaps[:, -1]
        drop = (g0 - gK) / np.maximum(g0, 1e-8)
        rho = np.asarray(d["rho"])
        summary[s_key] = {
            "n": int(gaps.shape[0]),
            "gap_mean_trace": gaps.mean(axis=0).tolist(),
            "gap0_mean": float(g0.mean()),
            "gapK_mean": float(gK.mean()),
            "gap_drop_mean": float(drop.mean()),
            "gap_drop_std": float(drop.std()),
            "frac_shrunk": float((drop > 0).mean()),
            "rho_median": float(np.nanmedian(rho)),
            "drift_mean": float(np.mean(d["drift"])),
        }

    # ── pre-registered gate ──────────────────────────────────────────────────
    reasons: list[str] = []
    for s_key in sorted(summary, reverse=True):
        if s_key < args.gate_sigma:
            continue
        m = summary[s_key]
        shrinks = m["gap_drop_mean"] >= args.gate_gap
        contracts = m["rho_median"] < args.gate_rho
        if shrinks and contracts:
            reasons.append(
                f"σ={s_key:.2f}: gap −{m['gap_drop_mean'] * 100:.1f}% "
                f"(frac {m['frac_shrunk']:.2f}), ρ̃={m['rho_median']:.2f} < {args.gate_rho}"
            )
    present = bool(reasons)
    verdict = (
        "GOLDEN-PATH PRESENT (premise holds — proceed to staged build)"
        if present
        else "ABSENT (close the line; write the negative finding)"
    )

    # ── render markdown ──────────────────────────────────────────────────────
    L = ["# FSG golden-path premise probe (Phase 0)\n"]
    L.append(
        f"- adapter: `{args.adapter or 'base DiT'}`\n"
        f"- prompts: **{len(prompts)}** × {args.num_seeds} seeds · "
        f"{args.height}×{args.width} · infer_steps {args.infer_steps} "
        f"(flow_shift {args.flow_shift})\n"
        f"- operator: Δσ={f'{args.d_sigma_frac}·σ' if args.d_sigma_frac is not None else args.d_sigma}, "
        f"γ={gamma}, K={args.k_iters}; "
        f"trajectory CFG={args.guidance}, neg=`{args.neg_prompt or '∅ (null)'}`\n"
        f"- gate: gap drop ≥ {args.gate_gap:.0%} AND median ρ < {args.gate_rho} "
        f"at any σ ≥ {args.gate_sigma}\n"
        f"- **verdict: {verdict}**\n"
    )
    for r in reasons:
        L.append(f"  - GATE: {r}")

    L.append("\n## Per-σ summary\n")
    L.append(
        "| σ | gap₀ | gap_K | gap drop | frac shrunk | median ρ | drift | n |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for s_key in sorted(summary, reverse=True):
        m = summary[s_key]
        L.append(
            f"| {s_key:.2f} | {_fmt(m['gap0_mean'])} | {_fmt(m['gapK_mean'])} "
            f"| {m['gap_drop_mean'] * 100:+.1f}% | {m['frac_shrunk']:.2f} "
            f"| {_fmt(m['rho_median'], '.3f')} | {_fmt(m['drift_mean'], '.3f')} "
            f"| {m['n']} |"
        )

    L.append("\n## Gap trace g_k (mean, k=0..K)\n")
    L.append("| σ | " + " | ".join(f"k={k}" for k in range(args.k_iters + 1)) + " |")
    L.append("|---|" + "---|" * (args.k_iters + 1))
    for s_key in sorted(summary, reverse=True):
        tr = summary[s_key]["gap_mean_trace"]
        L.append(f"| {s_key:.2f} | " + " | ".join(_fmt(v) for v in tr) + " |")

    L.append("\n## Reading it\n")
    L.append(
        "- **gap** = ‖v^c − v^u‖/‖v^u‖, the flow-matching analogue of FSG Fig 2(b). "
        "A falling trace as k grows is the golden-path signature (cond/uncond "
        "converge). `frac shrunk` is per-sample sign consistency of the total drop.\n"
        "- **median ρ** = ‖Δx_k‖/‖Δx_{k−1}‖; < 1 means the fixed-point operator "
        "contracts (Theorem 1's premise). ≥ 1 = the iteration walks away.\n"
        "- **PRESENT** ⇒ staged build: first CFG×K on the linear operator "
        "(cheapest, paper shows it already helps), then the full FSG correction "
        "plugin in `library/inference/corrections/`.\n"
        "- **ABSENT** ⇒ Anima's velocity field has no golden path to calibrate "
        "toward; FSG's mechanism doesn't transfer from ε-pred DDIM. Close it.\n"
        "- Phase-0 is mechanism-only; the gap→quality link is a Phase-1 eyeball "
        "A/B (no trustworthy Anima quality reward — see null-TTA memory).\n"
    )

    run_dir = make_run_dir("fsg", label=args.label or "golden-path")
    (run_dir / "golden_path.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    csv = run_dir / "golden_path.csv"
    with csv.open("w") as f:
        f.write("sigma,gap0,gapK,gap_drop,frac_shrunk,rho_median,drift,n\n")
        for s_key in sorted(summary, reverse=True):
            m = summary[s_key]
            f.write(
                f"{s_key},{m['gap0_mean']},{m['gapK_mean']},{m['gap_drop_mean']},"
                f"{m['frac_shrunk']},{m['rho_median']},{m['drift_mean']},{m['n']}\n"
            )

    metrics = {
        "adapter": args.adapter,
        "n_prompts": len(prompts),
        "num_seeds": args.num_seeds,
        "guidance": args.guidance,
        "fsg_gamma": gamma,
        "d_sigma": args.d_sigma,
        "d_sigma_frac": args.d_sigma_frac,
        "trajectory_substrate": "cfg++" if args.cfgpp else "cfg",
        "cfgpp_lambda": args.cfgpp_lambda if args.cfgpp else None,
        "k_iters": args.k_iters,
        "flow_shift": args.flow_shift,
        "resolution": [args.height, args.width],
        "summary": {str(s): summary[s] for s in summary},
        "gate": {
            "sigma": args.gate_sigma,
            "gap": args.gate_gap,
            "rho": args.gate_rho,
            "reasons": reasons,
        },
        "present": present,
        "verdict": verdict,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["golden_path.md", "golden_path.csv"],
        device=device,
    )

    log.info("\n" + "=" * 72)
    log.info(f"  FSG golden-path probe → {run_dir}")
    for s_key in sorted(summary, reverse=True):
        m = summary[s_key]
        log.info(
            f"  σ={s_key:.2f}  gap {m['gap0_mean']:.4f}→{m['gapK_mean']:.4f} "
            f"({m['gap_drop_mean'] * 100:+.1f}%)  ρ̃={m['rho_median']:.3f}  "
            f"drift={m['drift_mean']:.3f}  n={m['n']}"
        )
    log.info(f"  VERDICT: {verdict}")
    for r in reasons:
        log.info(f"    - {r}")
    log.info("=" * 72)
    if device.type == "cuda":
        clear_hydra_sigma(anima)
        clear_hydra_fei(anima)


if __name__ == "__main__":
    main()
