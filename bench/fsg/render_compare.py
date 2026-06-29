#!/usr/bin/env python3
"""FSG render A/B — VAE-decode baseline vs golden-path-calibrated generations.

Phase-1 eyeball test for Foresight Guidance (see bench/fsg/probe_golden_path.py
for the Phase-0 premise probe). The probe established the fixed-point operator
*contracts* and the conditional/unconditional gap *shrinks* in σ∈[~0.45, 0.85],
but a falling ‖v^c−v^u‖/‖v^u‖ is mechanism-only — it cannot tell whether
calibration nudges the latent toward a better-aligned **same image** or simply
drifts it to a flatter, lower-velocity (mushy) region. The only way to know is
to look. This renders, for the same noise/prompt/seed:

  * **baseline**       — plain deterministic Euler-CFG, no calibration
  * **fsg/cfg**        — foresight loop at σ ∈ [--narrow_lo, --narrow_hi] grafted
                         onto the CFG substrate (what the shipped plugin does)
  * **cfg++**          — plain CFG++ substrate (paper App A.2), no foresight: step
                         along v^u with guidance via λ·σ·(v^c−v^u)
  * **fsg/cfg++**      — faithful Algorithm 1: foresight loop ON the CFG++ substrate

The last two are dropped with --no_cfgpp. The cfg++ vs baseline column isolates
the substrate switch; fsg/cfg++ vs cfg++ isolates the foresight win *on the
substrate FSG is actually defined on* — the paper grafts foresight onto CFG++,
not CFG, and the shipped plugin's CFG graft was the gap this A/B closes.

Calibration per scheduled step is FSG's forward-backward operator applied to the
trajectory itself (Algorithm 1's x_t → x̂_t), then the denoise step proceeds from
x̂_t. Everything but the calibration/substrate is held fixed across arms, so any
visible difference is the intervention. We also print the latent drift ‖x_fsg−x_base‖ at
σ=0 (relative) per image — large drift + degraded image = the drift-to-mush
confound; small drift + cleaner image = a real refinement.

Sampler: ``--sampler euler`` (default) is deterministic and isolates the operator;
``--sampler er_sde`` runs the production stochastic SDE (the er_sde seed is shared
across arms so the injected-noise sequence is identical — differences are the
intervention, not noise luck). The CFG++ reweight composes with either.

Plan A (λ sweep — pick λ* where cfg++ tracks CFG=4 saturation/contrast): pass
``--cfgpp_lambdas`` for one ``cfg++ λ`` arm per value vs the CFG=4 baseline. Per-arm
mean HSV saturation + RMS contrast (and Δ% vs baseline) land in result.json's
``sat_contrast`` and on each panel label — λ* = the arm closest to baseline tone.

Run from anima_lora/::

    uv run python bench/fsg/render_compare.py --num_prompts 4 --compile
    uv run python bench/fsg/render_compare.py --prompts "1girl, ..." --num_seeds 2
    # Plan A — λ sweep on the production sampler:
    uv run python bench/fsg/render_compare.py --sampler er_sde --infer_steps 28 \
        --cfgpp_lambdas 1.0 1.5 2.0 3.0 --num_prompts 6 --num_seeds 2 --compile
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
from PIL import Image, ImageDraw  # noqa: E402

from anima_lora import decode_to_pil, load_vae  # noqa: E402
from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.fsg.probe_golden_path import (  # noqa: E402
    _sample_captions,
    _velocity,
)
from library.anima import models as anima_models  # noqa: E402
from library.inference.sampling import (  # noqa: E402
    ERSDESampler,
    cfgpp_guidance_weight,
    get_timesteps_sigmas,
)
from library.inference.text import prepare_text_inputs  # noqa: E402

log = logging.getLogger("bench.fsg.render")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _norm(t: torch.Tensor) -> float:
    return float(t.float().pow(2).sum().sqrt())


def _sat_contrast(im: Image.Image) -> tuple[float, float]:
    """(mean HSV saturation, RMS contrast) of an RGB image, both in [0,1].

    The whole FSG "win" question is whether it's just a global tone bump — these
    two scalars quantify it so λ* selection (Plan A) keys off numbers, not eyeball
    alone. Saturation = mean HSV S = mean((max−min)/max); RMS contrast = std of
    Rec.601 luminance. Both computed on the decoded RGB, not the latent.
    """
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    lum = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return float(sat.mean()), float(lum.std())


@torch.no_grad()
def _fsg_calibrate(anima, x, s_i, dsig, gamma, k_iters, embed, nembed, pad, step_i):
    """FSG forward-backward calibration of x at σ=s_i (Algorithm 1 inner loop)."""
    for _ in range(k_iters):
        vc = _velocity(anima, x, s_i, embed, pad, step_i)
        vu = _velocity(anima, x, s_i, nembed, pad, step_i)
        vg = vu + gamma * (vc - vu)
        x_fwd = x - dsig * vg  # denoise σ → σ−Δσ (guided)
        vu_lo = _velocity(anima, x_fwd, max(s_i - dsig, 1e-3), nembed, pad, step_i)
        x = x_fwd + dsig * vu_lo  # invert back (uncond)
    return x


@torch.no_grad()
def _trajectory(
    anima,
    x0,
    sig,
    sigmas_t,
    embed,
    nembed,
    pad,
    guidance,
    *,
    band,
    dsig,
    gamma,
    k,
    cfgpp_lambda,
    sampler,
    er_seed,
    device,
    dtype,
):
    """Trajectory; ``band`` is (lo, hi) or None (no foresight).

    Outer denoise substrate (Algorithm 1 lines 9–12):

    * ``cfgpp_lambda is None`` → plain **CFG**: step along the guided velocity
      ``v^u + w(v^c − v^u)``. This is what the shipped plugin grafts the foresight
      loop onto — *not* the paper's substrate.
    * ``cfgpp_lambda`` set → faithful **CFG++** (paper App A.2): deliver guidance
      through the σ-scheduled reweight ``v^u + w_eff·(v^c − v^u)`` (a pure cond/
      uncond combine change, so it composes with ANY integrator). λ is a
      flow-space coeff, NOT the paper's DDIM λ=0.6.

    ``sampler`` selects the integrator: ``"euler"`` (deterministic ODE) or
    ``"er_sde"`` (production stochastic SDE). The CFG++ reweight is sampler-
    agnostic — er_sde consumes the reweighted prediction unchanged. The FSG
    foresight inner loop (``_fsg_calibrate``) stays Euler-deterministic in both:
    it's a probe of the trajectory's golden-path geometry, not a denoise step.

    A fresh ER-SDE solver is built per call seeded by ``er_seed`` so the injected
    noise sequence is identical across arms — differences are the operator, not
    noise luck (the same isolation Euler gives for free).
    """
    x = x0.clone()
    n = len(sig) - 1
    er = (
        ERSDESampler(sigmas_t, seed=er_seed, device=device)
        if sampler == "er_sde"
        else None
    )
    for i in range(n):
        s_i = sig[i]
        if band is not None and band[0] <= s_i <= band[1]:
            x = _fsg_calibrate(anima, x, s_i, dsig, gamma, k, embed, nembed, pad, i)
        vc = _velocity(anima, x, s_i, embed, pad, i)
        vu = _velocity(anima, x, s_i, nembed, pad, i)
        if cfgpp_lambda is None:
            noise_pred = vu + guidance * (vc - vu)
        else:
            w_eff = cfgpp_guidance_weight(s_i, sig[i + 1], cfgpp_lambda)
            noise_pred = vu + w_eff * (vc - vu)
        if er is not None:
            denoised = x.float() - s_i * noise_pred.float()
            x = er.step(x, denoised, i).to(dtype)  # match production's per-step recast
        else:
            x = x - (sig[i] - sig[i + 1]) * noise_pred
    return x


def _panel(images: list[Image.Image], labels: list[str], caption: str) -> Image.Image:
    """Horizontal strip of images with column labels + a wrapped caption header."""
    w, h = images[0].size
    head = 88
    panel = Image.new("RGB", (w * len(images), h + head), "white")
    d = ImageDraw.Draw(panel)
    cap = caption if len(caption) <= 180 else caption[:177] + "…"
    # crude wrap at ~95 chars
    lines = [cap[j : j + 95] for j in range(0, len(cap), 95)][:2]
    d.text((6, 4), "\n".join(lines), fill="black")
    for k, (im, lab) in enumerate(zip(images, labels)):
        panel.paste(im, (k * w, head))
        d.text((k * w + 6, head - 22), lab, fill="black")
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap)  # --dit/--vae/--text_encoder
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompts", nargs="+", default=None)
    ap.add_argument("--caption_dir", default="post_image_dataset/resized")
    ap.add_argument("--num_prompts", type=int, default=4)
    ap.add_argument("--neg_prompt", default="")
    ap.add_argument("--num_seeds", type=int, default=1)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--infer_steps", type=int, default=20)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--fsg_gamma", type=float, default=None)
    ap.add_argument("--d_sigma", type=float, default=0.1)
    ap.add_argument("--k_iters", type=int, default=4)
    ap.add_argument("--narrow_lo", type=float, default=0.75)
    ap.add_argument("--narrow_hi", type=float, default=0.85)
    ap.add_argument(
        "--cfgpp_lambda",
        type=float,
        default=1.5,
        help="CFG++ strength λ (FLOW-space coefficient: guidance enters as "
        "λ·(1−σ')·σ·Δv, NOT the paper's DDIM ξ̃-space λ=0.6). Default 1.5 = λ* from "
        "the er_sde/28-step Plan-A sweep. The substrate integrates along v^u.",
    )
    ap.add_argument(
        "--cfgpp_lambdas",
        type=float,
        nargs="+",
        default=None,
        help="Plan-A λ SWEEP: one `cfg++ λ` arm per value (vs the CFG=4 baseline), "
        "e.g. --cfgpp_lambdas 1.0 1.5 2.0 3.0. Replaces the default 4-arm set; the "
        "shipped fsg/cfg arm is dropped (use --with_foresight to add fsg/cfg++ per λ "
        "for the Plan-C confound read). λ* = closest sat/contrast to baseline.",
    )
    ap.add_argument(
        "--with_foresight",
        action="store_true",
        help="In --cfgpp_lambdas sweep mode, also render a fsg/cfg++ arm per λ.",
    )
    ap.add_argument(
        "--sampler",
        choices=["euler", "er_sde"],
        default="euler",
        help="Integrator: 'euler' (deterministic, isolates the operator) or "
        "'er_sde' (production stochastic SDE). Default euler.",
    )
    ap.add_argument(
        "--no_cfgpp",
        action="store_true",
        help="Drop the CFG++ arms; render only baseline + FSG-on-CFG (old behaviour).",
    )
    add_common_args(ap)
    args = ap.parse_args()

    prompts = args.prompts or _sample_captions(
        args.caption_dir, args.num_prompts, args.seed
    )
    gamma = args.guidance if args.fsg_gamma is None else args.fsg_gamma

    bundle = build_anima(args, adapter=args.adapter, train_mode=False)
    anima, device, dtype = bundle.anima, bundle.device, bundle.dtype
    vae = load_vae(args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True)

    _, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sig = [float(s) for s in sigmas]

    # (label, band, cfgpp_lambda). band=None → no foresight; λ=None → CFG substrate.
    nb = (args.narrow_lo, args.narrow_hi)
    if args.cfgpp_lambdas:
        # Plan-A λ sweep: baseline CFG=4 + one cfg++ arm per λ (+ optional fsg/cfg++).
        arms = [("baseline", None, None)]
        for lam in args.cfgpp_lambdas:
            arms.append((f"cfg++ λ{lam:g}", None, lam))
            if args.with_foresight:
                arms.append((f"fsg/cfg++ λ{lam:g} {nb[0]:g}-{nb[1]:g}", nb, lam))
    else:
        lam = args.cfgpp_lambda
        arms = [
            ("baseline", None, None),  # plain CFG (reference)
            (f"fsg/cfg {nb[0]:g}-{nb[1]:g}", nb, None),  # shipped: foresight on CFG
        ]
        if not args.no_cfgpp:
            arms += [
                (f"cfg++ λ{lam:g}", None, lam),  # CFG++ substrate alone (no foresight)
                (f"fsg/cfg++ {nb[0]:g}-{nb[1]:g}", nb, lam),  # faithful Algorithm 1
            ]
    band_steps = {
        lab: [round(s, 3) for s in sig[:-1] if b and b[0] <= s <= b[1]]
        for lab, b, _ in arms
    }
    log.info(f"arms: {[a[0] for a in arms]}")
    for lab, b, _ in arms:
        if b:
            log.info(
                f"  '{lab}' calibrates {len(band_steps[lab])} steps at σ={band_steps[lab]}"
            )

    C = anima_models.Anima.LATENT_CHANNELS
    hl, wl = args.height // 8, args.width // 8
    pad = torch.zeros(1, 1, hl, wl, dtype=dtype, device=device)

    log.info(f"sampler: {args.sampler}")
    run_dir = make_run_dir("fsg", label=args.label or "render-compare")
    panels: list[Image.Image] = []
    drift_log: list[dict] = []
    # arm → list of (saturation, rms_contrast) across all prompts/seeds.
    sat_con: dict[str, list[tuple[float, float]]] = {lab: [] for lab, _, _ in arms}

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
            x0 = torch.randn((1, C, 1, hl, wl), generator=g, device=device, dtype=dtype)

            er_seed = args.seed + pi * 1000 + sj  # shared across arms for fair A/B
            imgs, labels = [], []
            base_lat = None
            for lab, band, cfgpp_lambda in arms:
                lat = _trajectory(
                    anima,
                    x0,
                    sig,
                    sigmas,
                    embed,
                    nembed,
                    pad,
                    args.guidance,
                    band=band,
                    dsig=args.d_sigma,
                    gamma=gamma,
                    k=args.k_iters,
                    cfgpp_lambda=cfgpp_lambda,
                    sampler=args.sampler,
                    er_seed=er_seed,
                    device=device,
                    dtype=dtype,
                )
                if lab == "baseline":
                    base_lat = lat
                    drift = 0.0
                else:
                    drift = _norm(lat - base_lat) / max(_norm(base_lat), 1e-8)
                im = decode_to_pil(vae, lat, device)
                imgs.append(im)
                sat, con = _sat_contrast(im)
                sat_con[lab].append((sat, con))
                is_base = lab == "baseline"
                labels.append(
                    f"{lab}  sat{sat:.3f} c{con:.3f}"
                    + (f"  Δ={drift:.3f}" if not is_base else "")
                )
                _safe = lab.replace(" ", "_").replace(".", "p").replace("/", "-")
                fn = f"p{pi}_s{sj}_{_safe}.png"
                imgs[-1].save(run_dir / fn)
                drift_log.append(
                    {
                        "prompt_idx": pi,
                        "seed": sj,
                        "arm": lab,
                        "drift": drift,
                        "saturation": sat,
                        "rms_contrast": con,
                    }
                )
            panels.append(_panel(imgs, labels, prompt))
            log.info(
                f"  [{pi + 1}/{len(prompts)} seed {sj}] rendered  "
                + "  ".join(f"{labels[k]}" for k in range(1, len(labels)))
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # stack all panels vertically into one contact sheet
    if panels:
        W = max(p.width for p in panels)
        Htot = sum(p.height for p in panels) + 4 * (len(panels) - 1)
        sheet = Image.new("RGB", (W, Htot), "white")
        y = 0
        for p in panels:
            sheet.paste(p, (0, y))
            y += p.height + 4
        sheet.save(run_dir / "contact_sheet.png")

    # Per-arm sat/contrast aggregate + Δ vs baseline (the λ* selector for Plan A).
    base_sat = (
        float(np.mean([s for s, _ in sat_con["baseline"]]))
        if sat_con["baseline"]
        else 0.0
    )
    base_con = (
        float(np.mean([c for _, c in sat_con["baseline"]]))
        if sat_con["baseline"]
        else 0.0
    )
    sat_contrast: dict[str, dict] = {}
    log.info("\narm                          sat     Δsat%    contrast  Δcon%")
    for lab, _, _ in arms:
        vals = sat_con[lab]
        if not vals:
            continue
        msat = float(np.mean([s for s, _ in vals]))
        mcon = float(np.mean([c for _, c in vals]))
        dsat = 100.0 * (msat - base_sat) / base_sat if base_sat else 0.0
        dcon = 100.0 * (mcon - base_con) / base_con if base_con else 0.0
        sat_contrast[lab] = {
            "saturation": msat,
            "rms_contrast": mcon,
            "d_saturation_pct": dsat,
            "d_rms_contrast_pct": dcon,
            "n": len(vals),
        }
        log.info(f"{lab:28s} {msat:.4f}  {dsat:+6.1f}   {mcon:.4f}   {dcon:+6.1f}")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "n_prompts": len(prompts),
            "num_seeds": args.num_seeds,
            "sampler": args.sampler,
            "infer_steps": args.infer_steps,
            "arms": [a[0] for a in arms],
            "band_steps": band_steps,
            "guidance": args.guidance,
            "cfgpp_lambda": None if args.no_cfgpp else args.cfgpp_lambda,
            "cfgpp_lambdas": args.cfgpp_lambdas,
            "fsg_gamma": gamma,
            "d_sigma": args.d_sigma,
            "k_iters": args.k_iters,
            "sat_contrast": sat_contrast,
            "drift": drift_log,
        },
        label=args.label,
        artifacts=["contact_sheet.png"],
        device=device,
    )
    log.info(f"\n→ {run_dir}/contact_sheet.png")


if __name__ == "__main__":
    main()
