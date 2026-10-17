#!/usr/bin/env python3
"""REPA encoder manifold-gap probe — does each encoder's alignment survive the
move from the TRAINING manifold to the GENERATION manifold?

THE QUESTION. ``bench/repa/probe_encoder_screen.py`` measures REPA-target
alignment on **noised real latents** — the *training* input distribution
(``x_σ = (1−σ)x0_real + σε``). But REPA's whole job is to anchor the DiT's
representation, and at inference the DiT runs on its **own reverse-sampling
trajectory**, which drifts off the real-data manifold (exposure bias). If an
encoder aligns well on real data but poorly on the generated manifold, the REPA
anchoring learned on real images won't bite where generation actually lives.

So this probe measures the same relational-CKA ``useful_gap`` in **two regimes**
on the same captions, at a **matched σ schedule**:

  * **real** — noise the cached real latent at each σ → DiT → CKA vs the *real*
    image's encoder features (identical to the screen).
  * **generated** — roll out the flow-matching ODE from noise with the caption
    (CFG=1, Anima's v=ε−x0 + Euler + flow_shift=3 schedule), then at each σ on
    the trajectory → DiT → CKA vs the *generated* image's own encoder features.

READOUTS, per encoder: ``useful_gap_real``, ``useful_gap_gen`` (repr-regime
band), and ``delta = real − gen``. The decision-relevant question is whether the
**encoder ranking is preserved** across regimes (does PE still lead on-manifold?)
and whether PE's gap holds; the signed per-encoder delta is secondary.

CONFOUND (flagged, not hidden). The generated target is the model's *own* output
for that caption, so the DiT representation at ``x_σ_gen`` is causally producing
``x0_gen`` — it can align *more* to the generated image than a real one would,
biasing ``delta`` negative regardless of true transfer. Read the **ranking
preservation** and the **gen gap magnitude**, not the raw signed delta. Still a
no-training proxy.

Run from anima_lora/::

    uv run python bench/repa/probe_encoder_manifold_gap.py --num_samples 48 --infer_steps 20
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

from bench._anima import add_common_args, add_model_args, build_anima, resolve_dtype  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.repa.probe_encoder_screen import (  # noqa: E402
    DEFAULT_ENCODERS,
    ENCODER_SPECS,
    GRID,
    build_encoder,
)
from bench.repa.probe_layer_sigma_cka import _all_block_features, linear_cka  # noqa: E402
from bench.soft_tokens_contrastive.reward_premise_probe import (  # noqa: E402
    EmbCache,
    discover_pairs,
)
from library.inference.sampling import get_timesteps_sigmas  # noqa: E402
from library.io.cache import load_cached_latents  # noqa: E402
from library.training.repa import pool_dit_tokens_to_grid  # noqa: E402

log = logging.getLogger("bench.repa.manifold_gap")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATA = "post_image_dataset/lora"
SHIPPED_LAYER = 8


@torch.no_grad()
def _rollout(anima, noise, emb, pad, timesteps, sigmas, device, dtype):
    """Anima flow-matching reverse rollout (CFG=1). Returns final x0_gen +
    the trajectory latent at each σ in ``timesteps`` (the state the DiT *saw*
    when it predicted the step). v = ε − x0; Euler: x_next = x − (σ_i−σ_{i+1})·v.
    """
    x = noise.to(device, dtype)
    traj = []
    for i in range(len(timesteps)):
        traj.append(x.clone())
        t_b = timesteps[i].expand(x.shape[0]).to(device, dtype)
        vel = anima.forward_mini_train_dit(x, t_b, emb, padding_mask=pad, skip_pooled_text_proj=True)
        x = (x.float() - (float(sigmas[i]) - float(sigmas[i + 1])) * vel.float()).to(dtype)
    return x, traj  # x = x0_gen; traj[i] is x_{σ_i}


def _useful_gap(cka_m, cka_x, layers, sigma_grid, band_cols, repr_rows):
    """matched − mismatch − (σ→1 floor), band-averaged per layer; + repr best."""
    m = np.array([[float(np.mean(cka_m[ell][s])) for s in sigma_grid] for ell in layers])
    x = np.array([[float(np.mean(cka_x[ell][s])) for s in sigma_grid] for ell in layers])
    gap = m - x
    useful = np.clip(gap - gap[:, -1:], 0.0, None)  # σ→1 floor (max-σ col)
    band = useful[:, band_cols].mean(axis=1)
    if repr_rows:
        rr = repr_rows[int(np.argmax(band[repr_rows]))]
        return useful, band, int(layers[rr]), float(band[rr])
    return useful, band, SHIPPED_LAYER, float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap, vae=True, text_encoder=False)
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--num_samples", type=int, default=48)
    ap.add_argument("--infer_steps", type=int, default=20)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="default: all")
    ap.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    ap.add_argument("--band_lo", type=float, default=0.45)
    ap.add_argument("--band_hi", type=float, default=0.90)
    ap.add_argument("--repr_regime_max", type=int, default=11)
    add_common_args(ap)
    args = ap.parse_args()

    device = torch.device(args.device if isinstance(args.device, str) else "cuda")
    dtype = resolve_dtype(args.dtype)

    timesteps, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sigma_grid = sorted(float(s) for s in timesteps.tolist())
    # Map each sorted σ back to its schedule index (for the trajectory lookup).
    sched = [float(s) for s in timesteps.tolist()]  # descending 1→0
    band = [s for s in sigma_grid if args.band_lo <= s <= args.band_hi] or sigma_grid
    band_cols = [sigma_grid.index(s) for s in band]

    pairs = discover_pairs(args.data_dir)
    embs = EmbCache(pairs)
    rng = np.random.default_rng(args.seed)
    pool = sorted(pairs)
    rng.shuffle(pool)

    # ── load VAE, gather real latents + captions; roll out generated latents ──
    from library.models.qwen_vae import load_vae

    vae = load_vae(args.vae, device=device, dtype=torch.bfloat16, eval=True, vae_2d=True)

    # Defer DiT (rollout needs it) — load it now since rollout + decode interleave.
    bundle = build_anima(args, adapter=None, train_mode=False)
    anima = bundle.anima
    patch = int(anima.patch_spatial)
    n_blocks = len(anima.blocks)
    layers = sorted(args.layers) if args.layers else list(range(n_blocks))
    layer_set = set(layers)
    repr_rows = [i for i, ell in enumerate(layers) if ell <= args.repr_regime_max]
    ship_row = int(np.argmin([abs(ell - SHIPPED_LAYER) for ell in layers]))

    samples = []
    for stem in pool:
        if len(samples) >= args.num_samples:
            break
        emb = embs.get(stem)
        if emb is None:
            continue
        npz_path, _te = pairs[stem]
        lat, _res, _oh, _ow = load_cached_latents(npz_path)
        H, W = int(lat.shape[-2]), int(lat.shape[-1])
        x0_real = lat.unsqueeze(0).unsqueeze(2)  # (1,C,1,H,W) CPU
        emb_b = emb.unsqueeze(0).to(device, dtype)
        pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)
        # roll out a generated trajectory from this image's seed
        g = torch.Generator(device=device).manual_seed(args.seed + len(samples))
        noise = torch.randn(1, lat.shape[0], 1, H, W, generator=g, device=device, dtype=dtype)
        x0_gen, traj = _rollout(anima, noise, emb_b, pad, timesteps, sigmas, device, dtype)
        with torch.no_grad():
            rgb_real = ((vae.decode_to_pixels(x0_real[:, :, 0].to(device, torch.bfloat16)).float() + 1) / 2).clamp(0, 1).cpu()
            rgb_gen = ((vae.decode_to_pixels(x0_gen[:, :, 0]).float() + 1) / 2).clamp(0, 1).cpu()
        samples.append({
            "stem": stem, "hw": (H, W), "emb": emb.unsqueeze(0),
            "x0_real": x0_real, "traj": [t.cpu() for t in traj],
            "rgb_real": rgb_real, "rgb_gen": rgb_gen,
        })
        if len(samples) % 8 == 0:
            log.info(f"  rolled out {len(samples)}/{args.num_samples}")
    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    n_img = len(samples)
    if n_img < 2:
        raise SystemExit("need ≥2 samples")
    partner = [(i + 1) % n_img for i in range(n_img)]
    log.info(f"{n_img} images · {args.infer_steps} steps · σ∈[{sched[-1]:.3f},{sched[0]:.3f}] · encoders={args.encoders}")

    # ── per-encoder dense tokens for BOTH real and generated images ────────────
    enc_tok_real: dict[str, list] = {}
    enc_tok_gen: dict[str, list] = {}
    enc_dim: dict[str, int] = {}
    for name in args.encoders:
        if name not in ENCODER_SPECS:
            continue
        try:
            enc = build_encoder(name, device, dtype)
            tr, tg = [], []
            for s in samples:
                tr.append(enc.encode(s["rgb_real"].to(device))[0].half().cpu())
                tg.append(enc.encode(s["rgb_gen"].to(device))[0].half().cpu())
        except Exception as e:  # noqa: BLE001
            log.warning(f"  ✗ {name}: {type(e).__name__}: {str(e)[:120]} — skipped")
            continue
        enc_tok_real[name], enc_tok_gen[name], enc_dim[name] = tr, tg, enc.d
        log.info(f"  ✓ {name}: D={enc.d}")
        del enc
        if device.type == "cuda":
            torch.cuda.empty_cache()
    encoders = [n for n in args.encoders if n in enc_tok_real]
    if not encoders:
        raise SystemExit("no encoder loaded")

    # ── score both regimes at the matched σ schedule ───────────────────────────
    def _blank():
        return {e: {ell: {s: [] for s in sigma_grid} for ell in layers} for e in encoders}

    rm, rx, gm, gx = _blank(), _blank(), _blank(), _blank()  # real/gen matched/mismatch

    for ai, rec in enumerate(samples):
        H, W = rec["hw"]
        emb_b = rec["emb"].to(device, dtype)
        pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)
        x0r = rec["x0_real"].to(device, dtype).float()
        per = {e: {"r": enc_tok_real[e][ai].to(device).float().unsqueeze(0),
                   "g": enc_tok_gen[e][ai].to(device).float().unsqueeze(0),
                   "rx": enc_tok_real[e][partner[ai]].to(device).float().unsqueeze(0),
                   "gx": enc_tok_gen[e][partner[ai]].to(device).float().unsqueeze(0)} for e in encoders}
        for si, s in enumerate(sigma_grid):
            sched_i = sched.index(s)
            # real: noise x0_real to σ; generated: the trajectory state at σ
            gg = torch.Generator(device=device).manual_seed(args.seed + ai * 997 + si)
            eps = torch.randn(x0r.shape, generator=gg, device=device, dtype=dtype)
            x_real = ((1.0 - s) * x0r + s * eps.float()).to(dtype)
            x_gen = rec["traj"][sched_i].to(device, dtype)
            t_b = torch.full((1,), float(s), device=device, dtype=dtype)
            fr = _all_block_features(anima, x_real, t_b, emb_b, pad, layer_set)
            fg = _all_block_features(anima, x_gen, t_b, emb_b, pad, layer_set)
            for ell in layers:
                dr = pool_dit_tokens_to_grid(fr[ell], (H, W), patch, GRID, GRID)[0]
                dg = pool_dit_tokens_to_grid(fg[ell], (H, W), patch, GRID, GRID)[0]
                for e in encoders:
                    rm[e][ell][s].append(linear_cka(dr, per[e]["r"][0]))
                    rx[e][ell][s].append(linear_cka(dr, per[e]["rx"][0]))
                    gm[e][ell][s].append(linear_cka(dg, per[e]["g"][0]))
                    gx[e][ell][s].append(linear_cka(dg, per[e]["gx"][0]))
        if (ai + 1) % 8 == 0 or ai + 1 == n_img:
            log.info(f"  [{ai + 1}/{n_img}] scored")
        if device.type == "cuda" and (ai + 1) % 12 == 0:
            torch.cuda.empty_cache()

    # ── aggregate ──────────────────────────────────────────────────────────────
    res = {}
    for e in encoders:
        ur, br, lr, sr = _useful_gap(rm[e], rx[e], layers, sigma_grid, band_cols, repr_rows)
        ug, bg, lg, sg = _useful_gap(gm[e], gx[e], layers, sigma_grid, band_cols, repr_rows)
        res[e] = {
            "real_repr_l": lr, "real_repr_gap": sr,
            "gen_repr_l": lg, "gen_repr_gap": sg,
            "delta_repr": sr - sg,
            "real_band_layer8": float(br[ship_row]),
            "gen_band_layer8": float(bg[ship_row]),
            "d_enc": enc_dim[e], "useful_real": ur, "useful_gen": ug,
        }
    rank_real = sorted(encoders, key=lambda e: res[e]["real_repr_gap"], reverse=True)
    rank_gen = sorted(encoders, key=lambda e: res[e]["gen_repr_gap"], reverse=True)
    preserved = rank_real == rank_gen

    # ── artifacts ──────────────────────────────────────────────────────────────
    run_dir = make_run_dir("repa", label=args.label or "encoder-manifold-gap")
    M = ["# REPA encoder manifold-gap — real (training) vs generated (inference)\n"]
    M.append(
        f"- images: **{n_img}** · {args.infer_steps}-step rollout (flow_shift "
        f"{args.flow_shift}, CFG=1) · matched σ schedule · base DiT\n"
        f"- headline: repr-regime (≤{args.repr_regime_max}) useful_gap in each "
        f"regime; **ranking preservation** is the decision signal\n"
        f"- ⚠️ generated target is the model's OWN output (self-consistent) → "
        f"signed delta is biased; read ranking + gen-gap magnitude\n"
    )
    M.append(
        f"\n**Ranking real:** {' > '.join(rank_real)}\n\n"
        f"**Ranking gen:**  {' > '.join(rank_gen)}\n\n"
        f"**Order preserved across manifolds: {'YES' if preserved else 'NO'}**\n"
    )
    M.append("\n## Per-encoder (representation-regime useful_gap)\n")
    M.append("| encoder | D | real gap @l | gen gap @l | Δ(real−gen) | real @l8 | gen @l8 |")
    M.append("|---|---|---|---|---|---|---|")
    for e in rank_real:
        r = res[e]
        M.append(
            f"| {e} | {r['d_enc']} | {r['real_repr_gap']:.4f} @{r['real_repr_l']} | "
            f"{r['gen_repr_gap']:.4f} @{r['gen_repr_l']} | {r['delta_repr']:+.4f} | "
            f"{r['real_band_layer8']:.4f} | {r['gen_band_layer8']:.4f} |"
        )
    M.append(
        "\n## Reading it\n"
        "- **Order preserved + PE leads gen** ⇒ the screen verdict is robust "
        "on-manifold; PE stays.\n"
        "- **gen gap ≫ real gap** for all ⇒ self-consistency inflation (expected), "
        "not a real transfer signal — compare encoders *within* the gen column.\n"
        "- **Order flips on gen** ⇒ exposure bias changes which encoder REPA "
        "should anchor to at inference; worth a closer look before trusting the "
        "real-data screen.\n"
    )
    (run_dir / "summary.md").write_text("\n".join(M) + "\n", encoding="utf-8")
    np.savez(
        run_dir / "heatmaps.npz",
        layers=np.array(layers), sigmas=np.array(sigma_grid), encoders=np.array(encoders),
        **{f"real_{e}": res[e]["useful_real"] for e in encoders},
        **{f"gen_{e}": res[e]["useful_gen"] for e in encoders},
    )
    write_result(
        run_dir, script=__file__, args=args,
        metrics={
            "n_images": n_img, "infer_steps": args.infer_steps, "flow_shift": args.flow_shift,
            "encoders": encoders, "layers": layers, "sigma_grid": sigma_grid,
            "rank_real": rank_real, "rank_gen": rank_gen, "order_preserved": bool(preserved),
            "per_encoder": {e: {k: v for k, v in res[e].items() if not k.startswith("useful_")} for e in encoders},
        },
        label=args.label, artifacts=["summary.md", "heatmaps.npz"], device=device,
    )

    log.info("\n" + "=" * 72)
    log.info(f"  manifold gap → {run_dir}")
    log.info(f"  real rank: {rank_real}")
    log.info(f"  gen  rank: {rank_gen}   (order preserved: {preserved})")
    for e in rank_real:
        r = res[e]
        log.info(f"  {e}: real {r['real_repr_gap']:.4f} | gen {r['gen_repr_gap']:.4f} | Δ {r['delta_repr']:+.4f}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
