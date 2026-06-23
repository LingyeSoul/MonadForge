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

NOTE production sampling is er_sde CFG=4; this A/B uses deterministic Euler CFG=4
to isolate the operator (er_sde stochasticity would swamp the comparison). A
production-sampler render is a follow-up, not this script's job.

Run from anima_lora/::

    uv run python bench/fsg/render_compare.py --num_prompts 4 --compile
    uv run python bench/fsg/render_compare.py --prompts "1girl, ..." --num_seeds 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

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
    cfgpp_guidance_weight,
    get_timesteps_sigmas,
)
from library.inference.text import prepare_text_inputs  # noqa: E402

log = logging.getLogger("bench.fsg.render")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _norm(t: torch.Tensor) -> float:
    return float(t.float().pow(2).sum().sqrt())


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
    anima, x0, sig, embed, nembed, pad, guidance, *, band, dsig, gamma, k, cfgpp_lambda
):
    """Deterministic trajectory; ``band`` is (lo, hi) or None (no foresight).

    Outer denoise substrate (Algorithm 1 lines 9–12):

    * ``cfgpp_lambda is None`` → plain **CFG**: step along the guided velocity
      ``v^u + w(v^c − v^u)``. This is what the shipped plugin grafts the foresight
      loop onto — *not* the paper's substrate.
    * ``cfgpp_lambda`` set → faithful **CFG++** (paper App A.2): deliver guidance
      through a σ-scaled calibration ``x̂ = x − λ·σ·(v^c − v^u)`` and integrate
      along the **unconditional** field ``x − Δσ·v^u``. λ∈[0,1] (paper uses 0.6).
      This is the substrate FSG is actually defined on; the σ-schedule keeps
      guidance stable through early σ (paper Fig 6) where CFG's coefficient decays.
    """
    x = x0.clone()
    n = len(sig) - 1
    for i in range(n):
        s_i = sig[i]
        if band is not None and band[0] <= s_i <= band[1]:
            x = _fsg_calibrate(anima, x, s_i, dsig, gamma, k, embed, nembed, pad, i)
        vc = _velocity(anima, x, s_i, embed, pad, i)
        vu = _velocity(anima, x, s_i, nembed, pad, i)
        step = sig[i] - sig[i + 1]
        if cfgpp_lambda is None:
            v_cfg = vu + guidance * (vc - vu)
            x = x - step * v_cfg
        else:
            # CFG++ as a σ-scheduled guidance reweight (App A.2) — integrator-
            # agnostic, identical to the calibrate-then-step form, and the same
            # path the production sampler uses (composes with er_sde).
            w_eff = cfgpp_guidance_weight(s_i, sig[i + 1], cfgpp_lambda)
            x = x - step * (vu + w_eff * (vc - vu))
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
        default=2.0,
        help="CFG++ strength λ (FLOW-space coefficient: guidance enters as "
        "λ·(1−σ')·σ·Δv, NOT the paper's DDIM ξ̃-space λ=0.6). λ≈1.5-2 ≈ CFG=4 total "
        "guidance; sweep it. The substrate integrates along v^u.",
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
            log.info(f"  '{lab}' calibrates {len(band_steps[lab])} steps at σ={band_steps[lab]}")

    C = anima_models.Anima.LATENT_CHANNELS
    hl, wl = args.height // 8, args.width // 8
    pad = torch.zeros(1, 1, hl, wl, dtype=dtype, device=device)

    run_dir = make_run_dir("fsg", label=args.label or "render-compare")
    panels: list[Image.Image] = []
    drift_log: list[dict] = []

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

            imgs, labels = [], []
            base_lat = None
            for lab, band, cfgpp_lambda in arms:
                lat = _trajectory(
                    anima, x0, sig, embed, nembed, pad, args.guidance,
                    band=band, dsig=args.d_sigma, gamma=gamma, k=args.k_iters,
                    cfgpp_lambda=cfgpp_lambda,
                )
                if lab == "baseline":
                    base_lat = lat
                    drift = 0.0
                else:
                    drift = _norm(lat - base_lat) / max(_norm(base_lat), 1e-8)
                imgs.append(decode_to_pil(vae, lat, device))
                is_base = lab == "baseline"
                labels.append(f"{lab}" + (f"  Δ={drift:.3f}" if not is_base else ""))
                _safe = lab.replace(" ", "_").replace(".", "p").replace("/", "-")
                fn = f"p{pi}_s{sj}_{_safe}.png"
                imgs[-1].save(run_dir / fn)
                drift_log.append(
                    {"prompt_idx": pi, "seed": sj, "arm": lab, "drift": drift}
                )
            panels.append(_panel(imgs, labels, prompt))
            log.info(f"  [{pi + 1}/{len(prompts)} seed {sj}] rendered  "
                     + "  ".join(f"{labels[k]}" for k in range(1, len(labels))))
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

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "n_prompts": len(prompts),
            "num_seeds": args.num_seeds,
            "arms": [a[0] for a in arms],
            "band_steps": band_steps,
            "guidance": args.guidance,
            "cfgpp_lambda": None if args.no_cfgpp else args.cfgpp_lambda,
            "fsg_gamma": gamma,
            "d_sigma": args.d_sigma,
            "k_iters": args.k_iters,
            "drift": drift_log,
        },
        label=args.label,
        artifacts=["contact_sheet.png"],
        device=device,
    )
    log.info(f"\n→ {run_dir}/contact_sheet.png")


if __name__ == "__main__":
    main()
