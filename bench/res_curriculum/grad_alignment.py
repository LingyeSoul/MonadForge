#!/usr/bin/env python3
"""Resolution-curriculum Phase-0 probe — does low-res training move the adapter
the same way high-res training would? (detector-free, no training run).

THE PROPOSAL UNDER TEST. A *resolution curriculum*: spend most epochs at a cheap
low tier (e.g. 896 → ~3000 tok) and only the final epoch(s) at the expensive high
tier (1024 → ~4032/4200 tok), to buy more gradient steps per FLOP without losing
final-resolution quality. The open question is whether the bulk low-res phase is
*productive* toward the high-res objective or whether it pulls the adapter in a
direction the high-res finish has to undo — i.e. is low-res a faithful cheap proxy
for the high-res gradient, and if not, *where* (which σ) does the high-res polish
earn its keep.

WHY THIS IS ANSWERABLE WITHOUT TRAINING. Curriculum is a statement about the
*training gradient* the adapter receives. So measure exactly that, on the live
adapter, at two resolutions of the *same* image, and compare directions:

    g_res(σ)  = ∂ L_fm / ∂θ_adapter   for the SAME image at tier T,

    where L_fm = MSE(v_pred, ε − x0) is the rectified-flow loss
    (train.py::get_noise_pred_and_target) and θ_adapter are the LoRA's
    trainable params — the thing the optimizer actually steps.

The ruler is the **cosine** between the low-tier and high-tier gradient over the
flattened adapter-param vector. High cosine ⇒ a low-res step is (almost) a high-res
step ⇒ the cheap bulk phase is productive ⇒ curriculum is FLOP-efficient and safe.
Low cosine ⇒ low-res training misdirects the adapter ⇒ curriculum risks fighting
itself at matched FLOPs.

We measure the adapter gradient (small, ~1e7 params) rather than the
weight-space FM gradient (the whole DiT, ~1e9): the adapter grad IS the training
signal, it is low-dim enough to flatten and cosine faithfully, and ΔW lives in the
same Linears (the parameterization-invariant spirit of
bench/timestep_mask/learned_rank.py). Requires a trained adapter checkpoint — the
curriculum question is *about a specific LoRA*, and at the init point (down≈0) the
gradient regime is unrepresentative.

THREE COSINES PER σ (signal bracketed by two controls — the global flattened
cosine is meaningful here, unlike the REPA tap probe, because both gradients are
the *same* loss on the *same* content, not random high-dim vectors):

  * cos_res  = cos(g_low , g_high)            same image · same noise · DIFFERENT res
               → the SIGNAL: how much resolution alone rotates the training step.
  * cos_seed = cos(g_high^a , g_high^b)       same image · same res · different noise
               → the CEILING: cos can't beat the stochastic-noise-draw ceiling even
                 with everything else identical. Computed when --num_seeds ≥ 2.
  * cos_img  = cos(g_high^i , g_high^j)       DIFFERENT image · same res
               → the FLOOR: how aligned two adapter grads are just from sharing the
                 model + objective (content-agnostic baseline).

HEADLINE — the normalized curriculum score per σ:

    score(σ) = (cos_res − cos_img) / (cos_seed − cos_img)      ∈ ~[0, 1]

  ≈1  resolution change is negligible vs the noise draw → low-res is an excellent
      proxy → curriculum maximally safe (and high-res adds little — you could even
      train low-res only).
  ≈0  resolution change rotates the step as much as swapping the image entirely →
      low-res training misdirects → curriculum risky.

Plus ``mag(σ) = ‖g_high‖ / ‖g_low‖`` — does high-res carry a *larger* training
signal (extra detail the low tier can't express)? A score near 1 with mag>1 at low
σ is the textbook curriculum case: structure is scale-invariant (cheap to learn
low-res) but fine detail lives in the low-σ / high-res regime (the polish phase's
real job).

THE GATE (printed verdict):
  * BUILD-justified  — score high at high σ but DROPS toward low σ (and/or mag>1 at
    low σ): low-res learns structure for free, high-res finish does real detail work
    → the curriculum *shape* is exactly right; the stronger build biases the high-res
    phase to low σ (a σ×res schedule).
  * NEUTRAL/cheap-win — score high & mag≈1 across all σ: low-res ≈ high-res
    everywhere → curriculum is safe but the high-res phase buys ~nothing; just train
    low-res for the FLOP saving.
  * KILL — score low across σ: low-res pulls the adapter off the high-res direction
    → curriculum likely degrades quality at matched FLOPs; don't build.

FAITHFULNESS / LIMITATIONS (stated up front):
  * "Same image at two resolutions" is produced by decode(cached latent) → resize
    (bicubic+antialias, the EasyControl-validated resample) → VAE re-encode at each
    tier's free-fit target. Both tiers share the one initial decode round-trip, so
    its VAE error is a common-mode confound, not a between-tier bias. Tiers use
    ``freefit_band_for_edge`` / ``freefit_bucket`` — the exact training resize.
  * Local linearization at probe time: predicts the instantaneous gradient
    interaction, not the full training trajectory (low-res and high-res optima can
    diverge over many steps even if the instantaneous directions agree). It measures
    the *right* quantity (the training step the optimizer takes); the matched-FLOPs
    A/B (low-res×full-epochs vs native, scored on CMMD + the memorization probe) is
    ground truth — this only gates whether that A/B is worth running.

Run from anima_lora/ (eager + grad-checkpoint forced; needs a trained adapter)::

    uv run python bench/res_curriculum/grad_alignment.py \
        --adapter output/ckpt/artist_3_16.safetensors \
        --low_tier 896 --high_tier 1024 \
        --num_samples 32 --num_seeds 2
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from anima_lora import load_vae  # noqa: E402
from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.soft_tokens_contrastive.reward_premise_probe import (  # noqa: E402
    EmbCache,
    discover_pairs,
)
from library.datasets.buckets import freefit_band_for_edge, freefit_bucket  # noqa: E402
from library.io.cache import load_cached_latents  # noqa: E402

log = logging.getLogger("bench.res_curriculum.grad_alignment")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATA = "post_image_dataset/lora"
# Same σ grid the REPA / cross-attn probes use, so they read side by side. The
# semantic-commitment window is σ≈0.45; structure forms high-σ, detail low-σ.
DEFAULT_SIGMAS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]


def _freefit_target(orig_w: int, orig_h: int, tier: int) -> tuple[int, int]:
    """Pixel (W, H) for ``tier`` via the exact training free-fit solver."""
    band = freefit_band_for_edge(tier)
    return freefit_bucket(orig_w, orig_h, band)  # (W, H), multiples of patch


def _encode_at(vae, pixels_m11: torch.Tensor, wh: tuple[int, int]) -> torch.Tensor:
    """Resize a [-1,1] (B,3,H,W) render to ``wh`` and VAE-encode → 4D latent.

    Pixels are [-1,1] to match the training pipeline (IMAGE_TRANSFORMS =
    ToTensor→Normalize([0.5],[0.5])); ``encode_pixels_to_latents`` consumes that
    range and applies the latent mean/std exactly as preprocess does.
    """
    w, h = wh
    resized = F.interpolate(
        pixels_m11.float(), size=(h, w), mode="bicubic", align_corners=False,
        antialias=True,
    ).clamp(-1.0, 1.0)
    with torch.no_grad():
        lat = vae.encode_pixels_to_latents(resized.to(vae.device, vae.dtype))
    return lat.float().cpu()  # (1, C, h//ds, w//ds)


def _adapter_grad(
    anima, network, x0_4d: torch.Tensor, emb: torch.Tensor, sigma: float,
    seed: int, device, dtype,
) -> tuple[torch.Tensor, float]:
    """Flat adapter-param gradient of L_fm at one (latent, σ, noise seed).

    Returns ``(g_flat_float32_on_device, l_fm_value)``. Unused params (e.g. an
    inactive T-LoRA branch at this σ) contribute zeros so the vector layout is
    identical across calls and cosines are well defined.
    """
    x0 = x0_4d.to(device, dtype).unsqueeze(2)  # (1,C,1,H,W) — DiT singleton at dim 2
    x0_f = x0.float()
    emb_b = emb.to(device, dtype)
    H, W = x0.shape[-2], x0.shape[-1]
    pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)

    g = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(x0.shape, generator=g, device=device, dtype=dtype)
    noisy = ((1.0 - sigma) * x0_f + sigma * eps.float()).to(dtype)
    t_b = torch.full((1,), float(sigma), device=device, dtype=dtype)
    target = (eps - x0).float()  # rectified-flow target

    trainable = [p for p in network.parameters() if p.requires_grad]
    with torch.enable_grad():
        velocity = anima.forward_mini_train_dit(
            noisy, t_b, emb_b, padding_mask=pad, skip_pooled_text_proj=True,
        )
        l_fm = F.mse_loss(velocity.float(), target)
        grads = torch.autograd.grad(l_fm, trainable, retain_graph=False,
                                    allow_unused=True)
    flat = torch.cat([
        (gr.detach().float().reshape(-1) if gr is not None
         else torch.zeros(p.numel(), device=device))
        for gr, p in zip(grads, trainable)
    ])
    return flat, float(l_fm.detach())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if float(na) <= 1e-20 or float(nb) <= 1e-20:
        return float("nan")
    return float((a @ b) / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap, text_encoder=False)  # needs --dit and --vae
    ap.add_argument("--adapter", required=True, help="trained LoRA checkpoint")
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--low_tier", type=int, default=896)
    ap.add_argument("--high_tier", type=int, default=1024)
    ap.add_argument("--num_samples", type=int, default=32)
    ap.add_argument(
        "--num_seeds", type=int, default=2,
        help="noise draws per (image, σ, tier). ≥2 enables the cos_seed ceiling.",
    )
    ap.add_argument(
        "--control_samples", type=int, default=4,
        help="images whose high-tier grad is retained (CPU fp16) for the "
        "cross-image cos_img floor (pairwise per σ).",
    )
    ap.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    ap.add_argument("--multiplier", type=float, default=1.0)
    add_common_args(ap)  # provides --label/--seed/--device/--dtype/--compile/…
    args = ap.parse_args()
    if args.label is None:
        args.label = "grad-align"

    if getattr(args, "compile", False):
        log.warning("grad-alignment probe runs eager; ignoring --compile.")
        args.compile = False
    if not getattr(args, "gradient_checkpointing", False):
        log.info("forcing gradient checkpointing (full-stack adapter backprop)")
        args.gradient_checkpointing = True

    sigma_grid = sorted(float(s) for s in args.sigmas)
    n_seeds = max(1, args.num_seeds)

    pairs = discover_pairs(args.data_dir)
    stems_all = sorted(pairs)
    if not stems_all:
        raise SystemExit(f"no cached (npz, te) pairs under {args.data_dir}")
    rng = np.random.default_rng(args.seed)
    take = min(args.num_samples, len(stems_all))
    stems = [stems_all[int(i)] for i in rng.choice(len(stems_all), take, replace=False)]
    embs = EmbCache(pairs)

    # Trained adapter, train mode → live LoRA training-path forward (T-LoRA mask,
    # fp32 bottleneck) and grad-checkpointed full-stack backprop.
    bundle = build_anima(args, adapter=args.adapter, train_mode=True,
                         multiplier=args.multiplier)
    anima, network = bundle.anima, bundle.network
    device, dtype = bundle.device, bundle.dtype
    vae = load_vae(args.vae, device="cpu", vae_2d=True)
    vae.to(device).to(torch.bfloat16).eval()

    log.info(
        f"{take} images | tiers {args.low_tier}→{args.high_tier} | σ {sigma_grid} | "
        f"{n_seeds} seed(s) | adapter {Path(args.adapter).name}"
    )

    # per-σ accumulators
    cos_res: dict[float, list[float]] = {s: [] for s in sigma_grid}
    cos_seed: dict[float, list[float]] = {s: [] for s in sigma_grid}
    mag: dict[float, list[float]] = {s: [] for s in sigma_grid}
    lfm_low: dict[float, list[float]] = {s: [] for s in sigma_grid}
    lfm_high: dict[float, list[float]] = {s: [] for s in sigma_grid}
    # control: high-tier grad (seed 0) on CPU fp16 for the cross-image floor.
    ctrl_vecs: dict[float, list[torch.Tensor]] = {s: [] for s in sigma_grid}

    n_skipped = 0
    for ai, stem in enumerate(stems):
        npz_path, _te = pairs[stem]
        emb = embs.get(stem)
        if emb is None:
            n_skipped += 1
            continue
        lat, _res, orig_h, orig_w = load_cached_latents(npz_path)
        # decode the cached latent ONCE → a faithful [-1,1] render, then re-encode
        # at each tier (shared decode = common-mode VAE error, not a tier bias).
        with torch.no_grad():
            render = vae.decode_to_pixels(lat.unsqueeze(0).to(device, vae.dtype))
        render = render.float().cpu()  # (1,3,H,W) in [-1,1]
        wh_low = _freefit_target(orig_w, orig_h, args.low_tier)
        wh_high = _freefit_target(orig_w, orig_h, args.high_tier)
        x0_low = _encode_at(vae, render, wh_low)
        x0_high = _encode_at(vae, render, wh_high)
        emb_b = emb.unsqueeze(0)

        keep_ctrl = ai < args.control_samples
        for s in sigma_grid:
            g_high_seeds: list[torch.Tensor] = []
            for sj in range(n_seeds):
                seed = args.seed + ai * 10000 + sj
                g_lo, l_lo = _adapter_grad(anima, network, x0_low, emb_b, s, seed,
                                           device, dtype)
                g_hi, l_hi = _adapter_grad(anima, network, x0_high, emb_b, s, seed,
                                           device, dtype)
                c = _cos(g_lo, g_hi)
                if not np.isnan(c):
                    cos_res[s].append(c)
                    mag[s].append(float(g_hi.norm() / (g_lo.norm() + 1e-20)))
                    lfm_low[s].append(l_lo)
                    lfm_high[s].append(l_hi)
                g_high_seeds.append(g_hi)
            # cos_seed ceiling: same image/res, the two noise draws
            if n_seeds >= 2:
                cs = _cos(g_high_seeds[0], g_high_seeds[1])
                if not np.isnan(cs):
                    cos_seed[s].append(cs)
            if keep_ctrl:
                ctrl_vecs[s].append(g_high_seeds[0].half().cpu())
            del g_high_seeds
        if (ai + 1) % 4 == 0:
            log.info(f"  {ai + 1}/{take} images")

    # cross-image floor: pairwise cos over the retained high-tier grads per σ
    cos_img: dict[float, list[float]] = {s: [] for s in sigma_grid}
    for s in sigma_grid:
        v = [t.float() for t in ctrl_vecs[s]]
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                c = _cos(v[i], v[j])
                if not np.isnan(c):
                    cos_img[s].append(c)

    def _m(d: dict[float, list[float]], s: float) -> float:
        return float(np.mean(d[s])) if d[s] else float("nan")

    # ── per-σ table + normalized curriculum score ─────────────────────────────
    rows = []
    scores = []
    for s in sigma_grid:
        cr, cse, ci = _m(cos_res, s), _m(cos_seed, s), _m(cos_img, s)
        denom = cse - ci
        score = (cr - ci) / denom if (not np.isnan(denom) and abs(denom) > 1e-6) else float("nan")
        if not np.isnan(score):
            scores.append((s, score))
        rows.append({
            "sigma": s, "cos_res": cr, "cos_seed": cse, "cos_img": ci,
            "curriculum_score": score, "mag_high_over_low": _m(mag, s),
            "lfm_low": _m(lfm_low, s), "lfm_high": _m(lfm_high, s),
            "n": len(cos_res[s]),
        })

    log.info("\nσ      cos_res  cos_seed cos_img  score   mag(hi/lo)  Lfm_lo  Lfm_hi")
    for r in rows:
        log.info(
            f"{r['sigma']:.2f}   {r['cos_res']:+.3f}   {r['cos_seed']:+.3f}  "
            f"{r['cos_img']:+.3f}  {r['curriculum_score']:+.3f}  "
            f"{r['mag_high_over_low']:.3f}      "
            f"{r['lfm_low']:.4f}  {r['lfm_high']:.4f}"
        )

    # ── verdict ───────────────────────────────────────────────────────────────
    hi_sig = [s for s in sigma_grid if s >= 0.55]
    lo_sig = [s for s in sigma_grid if s <= 0.45]
    sc = dict(scores)
    mean_score = float(np.mean([v for _, v in scores])) if scores else float("nan")
    hi_score = float(np.nanmean([sc[s] for s in hi_sig if s in sc])) if hi_sig else float("nan")
    lo_score = float(np.nanmean([sc[s] for s in lo_sig if s in sc])) if lo_sig else float("nan")
    mag_lo = float(np.nanmean([_m(mag, s) for s in lo_sig]))

    if not np.isnan(mean_score) and mean_score < 0.3:
        verdict = ("KILL — low-res gradient is nearly as far from high-res as a "
                   "different image is. Curriculum likely degrades quality at "
                   "matched FLOPs; don't build.")
    elif (not np.isnan(hi_score) and not np.isnan(lo_score)
          and hi_score - lo_score > 0.15 and mag_lo > 1.05):
        verdict = ("BUILD-justified — structure (high σ) is scale-invariant but "
                   "detail (low σ) and gradient magnitude live at high-res. The "
                   "curriculum shape is right; bias the high-res phase to low σ "
                   "(σ×res schedule). Run the matched-FLOPs A/B.")
    elif not np.isnan(mean_score) and mean_score > 0.7:
        verdict = ("NEUTRAL/cheap-win — low-res ≈ high-res across σ (mag≈1). "
                   "Curriculum is safe but the high-res phase buys little; the "
                   "FLOP win is just from training low-res. Verify on CMMD.")
    else:
        verdict = ("INCONCLUSIVE — partial alignment, no clean σ structure. "
                   "Widen --num_samples / --num_seeds before deciding.")

    log.info(
        f"\nmean score {mean_score:+.3f} | hi-σ {hi_score:+.3f} | lo-σ {lo_score:+.3f}"
        f" | mag@lo-σ {mag_lo:.3f}\nVERDICT: {verdict}"
    )

    run_dir = make_run_dir("res_curriculum", label=args.label)
    csv_path = run_dir / "per_sigma.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_result(
        run_dir, script=__file__, args=args, label=args.label, device=device,
        metrics={
            "mean_score": mean_score, "hi_sigma_score": hi_score,
            "lo_sigma_score": lo_score, "mag_at_lo_sigma": mag_lo,
            "verdict": verdict, "per_sigma": rows,
            "low_tier": args.low_tier, "high_tier": args.high_tier,
            "n_images": take, "n_skipped": n_skipped,
        },
        artifacts=["per_sigma.csv"],
    )
    log.info(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
