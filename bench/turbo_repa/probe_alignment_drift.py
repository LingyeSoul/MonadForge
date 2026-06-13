#!/usr/bin/env python3
"""Turbo × REPA alignment-drift probe (Phase 0 of docs/proposal/turbo_repa.md).

THE QUESTION. Does DP-DMD distillation pull the student's mid-block *visual*
representation off the base manifold, the way the caption-ranking probe showed
it degrades the *text* axis (shuffled rank@1 0.750 vs base 0.958 at σ=1.0)?
REPA gives us a calibrated ruler: the unweighted relational (Gram) alignment
loss of block-8 features vs cached PE-Spatial patch tokens — the exact
quantity the validated LoRA-family REPA term optimizes
(``library/training/repa.py::relational_align_loss``, spatial_norm on).

CONSTRUCTION (no training). For each cached real latent with a PE-Spatial
sidecar, renoise ``x_τ = (1−τ)·x0 + τ·ε`` at each σ on the grid, run the
feature-tap partial forward (``forward_mini_train_dit(...,
return_block_features={layer}, return_features_early=True)`` — exits after
block 8, ~9/28 of a forward) with the image's own matched caption, and score
the alignment loss for both arms on the SAME states and noise draws:

* **base** — LoRA multiplier 0 (the teacher's backbone),
* **student** — multiplier 1 (the trained turbo checkpoint).

Drift reads as student-above-base at matched σ (higher loss = worse aligned).

PRE-REGISTERED READOUT (the gate for everything downstream):

* student align loss clearly worse than base at σ ≥ 0.75 (where the ranking
  gap lives) → the premise holds, proceed to Phase 1 (the ``[repa]`` section
  in turbo.toml + A/B);
* student ≈ base everywhere → CLOSE the line without a training run: the
  drift isn't PE-visible, the caption degradation is a text-conditioning
  problem, and soft-rank alone owns the fix.

"Clearly worse" is operationalized as relative excess
``(student − base)/base ≥ --gate_excess`` (default 0.10) at any σ ≥ 0.75;
per-image paired deltas + sign consistency are reported alongside so the
verdict can be sanity-checked by eye. The full σ profile (including 0.5/0.25)
is reported because it picks between the primary (real-data PE) and fallback
(teacher-Gram) Phase-1 arms — see the proposal's "Secondary arm".

Run from anima_lora/::

    uv run python bench/turbo_repa/probe_alignment_drift.py \
        --adapter output/ckpt/anima_turbo_N_1250.safetensors
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from bench._anima import (  # noqa: E402
    add_common_args,
    add_model_args,
    build_anima,
)
from bench._common import make_run_dir, write_result  # noqa: E402

# Pair discovery + caption-embedding cache are shared with the reward-premise /
# caption-ranking probes — same pool semantics keeps results comparable.
from bench.soft_tokens_contrastive.reward_premise_probe import (  # noqa: E402
    EmbCache,
    _fmt,
    discover_pairs,
)
from library.io.cache import load_cached_latents  # noqa: E402
from library.training.repa import relational_align_loss  # noqa: E402
from library.vision.buckets import get_bucket_spec  # noqa: E402

log = logging.getLogger("bench.turbo_repa.alignment_drift")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATA = "post_image_dataset/lora"
# The caption-probe σ grid (proposal §Phase 0): the 4-step student's operating
# band plus 0.25 to expose the clean-end profile for the Phase-1 arm pick.
DEFAULT_SIGMAS = [0.25, 0.50, 0.75, 0.97, 1.00]

ARMS = ("base", "student")  # multiplier 0.0 / 1.0 on the same DiT


def _set_arm(network, arm: str) -> None:
    """Teacher/student toggle via set_multiplier — zeroes the LoRA delta
    without changing control flow (the caption-ranking-probe pattern)."""
    network.set_multiplier(0.0 if arm == "base" else 1.0)


def _pe_sidecar(te_path: str, stem: str, encoder: str) -> str:
    """PE sidecar lives next to the TE cache: ``{stem}_anima_{encoder}.safetensors``
    (candidate 1 of the dataset's resolution chain — the common layout)."""
    return os.path.join(os.path.dirname(te_path), f"{stem}_anima_{encoder}.safetensors")


@torch.no_grad()
def _block_features(anima, x_t, t_b, emb, pad, layer: int) -> torch.Tensor:
    """Feature-tap partial forward → raw block-``layer`` output.

    ``skip_pooled_text_proj=True`` matches the distill loop's forward exactly
    (bit-equivalent for base-DiT + LoRA, where pooled-text modulation is off).
    """
    feats = anima.forward_mini_train_dit(
        x_t,
        t_b,
        emb,
        padding_mask=pad,
        skip_pooled_text_proj=True,
        return_block_features={layer},
        return_features_early=True,
    )
    return feats[layer]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap, vae=False, text_encoder=False)
    ap.add_argument(
        "--adapter",
        default="output/ckpt/anima_turbo_N_1250.safetensors",
        help="Turbo student checkpoint (plain LoRA).",
    )
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument(
        "--num_samples", type=int, default=200, help="images (proposal: N ≥ 200)"
    )
    ap.add_argument(
        "--num_seeds",
        type=int,
        default=2,
        help="noise draws averaged per (image, σ); arms share the draws (paired).",
    )
    ap.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    ap.add_argument(
        "--layer", type=int, default=8, help="block tap (matches LoRA REPA)"
    )
    ap.add_argument("--encoder", default="pe_spatial")
    ap.add_argument(
        "--spatial_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="iREPA target standardization (the validated LoRA operating point)",
    )
    # Pre-registered gate.
    ap.add_argument(
        "--gate_excess",
        type=float,
        default=0.10,
        help="relative excess (student−base)/base counting as 'clearly worse'",
    )
    ap.add_argument(
        "--gate_sigma", type=float, default=0.75, help="gate applies at σ ≥ this"
    )
    add_common_args(ap)
    args = ap.parse_args()

    sigma_grid = sorted(float(s) for s in args.sigmas)
    spec = get_bucket_spec(args.encoder)
    log.info(f"σ grid: {sigma_grid}  |  layer={args.layer}  encoder={args.encoder}")

    pairs = discover_pairs(args.data_dir)
    # Keep only stems whose PE sidecar exists next to the TE cache.
    pe_paths = {
        stem: p
        for stem, (_npz, te) in pairs.items()
        if os.path.exists(p := _pe_sidecar(te, stem, args.encoder))
    }
    pool_list = sorted(pe_paths)
    if not pool_list:
        raise SystemExit(
            f"no {args.encoder} sidecars next to the TE caches under {args.data_dir} "
            "— run `make preprocess-pe` for the spatial encoder first"
        )
    log.info(
        f"{len(pool_list)}/{len(pairs)} cached pairs carry a {args.encoder} sidecar"
    )

    rng = np.random.default_rng(args.seed)
    take = min(args.num_samples, len(pool_list))
    stems = [pool_list[int(i)] for i in rng.choice(len(pool_list), take, replace=False)]

    bundle = build_anima(args, adapter=args.adapter, train_mode=False)
    anima, network = bundle.anima, bundle.network
    device, dtype = bundle.device, bundle.dtype
    patch = int(anima.patch_spatial)

    embs = EmbCache(pairs)
    n_seeds = max(1, args.num_seeds)

    # losses[arm][σ] = per-image seed-averaged align loss, paired across arms.
    losses: dict[str, dict[float, list[float]]] = {
        a: {s: [] for s in sigma_grid} for a in ARMS
    }
    n_skipped = 0

    for ai, stem in enumerate(stems):
        npz_path, _te = pairs[stem]
        emb = embs.get(stem)
        if emb is None:
            n_skipped += 1
            continue
        pe_sd = load_file(pe_paths[stem])
        pe = pe_sd.get("image_features")
        if pe is None:
            n_skipped += 1
            continue
        pe = pe.float().unsqueeze(0).to(device)  # (1, T, d_enc), CLS at 0

        lat, _res, _oh, _ow = load_cached_latents(npz_path)
        x0 = lat.to(device, dtype).unsqueeze(0).unsqueeze(2)  # (1,C,1,H,W)
        H, W = x0.shape[-2], x0.shape[-1]
        pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)
        x0_f = x0.float()
        emb_b = emb.to(device, dtype).unsqueeze(0)

        for s in sigma_grid:
            acc = {a: [] for a in ARMS}
            for sj in range(n_seeds):
                g = torch.Generator(device=device).manual_seed(
                    args.seed + ai * 1000 + sj
                )
                eps = torch.randn(x0.shape, generator=g, device=device, dtype=dtype)
                noisy = ((1.0 - s) * x0_f + s * eps.float()).to(dtype)
                t_b = torch.full((1,), float(s), device=device, dtype=dtype)
                for arm in ARMS:
                    _set_arm(network, arm)
                    feat = _block_features(anima, noisy, t_b, emb_b, pad, args.layer)
                    loss = relational_align_loss(
                        feat,
                        pe,
                        (H, W),
                        patch,
                        spec,
                        spatial_norm=args.spatial_norm,
                    )
                    acc[arm].append(float(loss))
            for arm in ARMS:
                losses[arm][s].append(float(np.mean(acc[arm])))

        if (ai + 1) % 20 == 0 or ai + 1 == len(stems):
            log.info(f"  [{ai + 1}/{len(stems)}] scored")
        if device.type == "cuda" and (ai + 1) % 50 == 0:
            torch.cuda.empty_cache()

    n_scored = len(losses["base"][sigma_grid[0]])
    if n_scored == 0:
        raise SystemExit("no images scored — check caches")

    # ── aggregate (paired per image) ─────────────────────────────────────────
    summary: dict[float, dict] = {}
    for s in sigma_grid:
        b = np.asarray(losses["base"][s])
        st = np.asarray(losses["student"][s])
        delta = st - b
        summary[s] = {
            "base_mean": float(b.mean()),
            "base_std": float(b.std()),
            "student_mean": float(st.mean()),
            "student_std": float(st.std()),
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std()),
            "excess": float((st.mean() - b.mean()) / max(b.mean(), 1e-12)),
            "frac_worse": float((delta > 0).mean()),
            "n": int(len(b)),
        }

    # ── pre-registered gate ──────────────────────────────────────────────────
    reasons: list[str] = []
    for s in sigma_grid:
        if s < args.gate_sigma:
            continue
        e = summary[s]["excess"]
        if e >= args.gate_excess:
            reasons.append(
                f"student align loss +{e * 100:.1f}% over base at σ={s} "
                f"(frac_worse={summary[s]['frac_worse']:.2f})"
            )
    drifted = bool(reasons)
    verdict = (
        "DRIFT (premise holds — Phase 1 unlocked)"
        if drifted
        else "NO-DRIFT (close the line; soft-rank owns the fix)"
    )

    # ── render markdown ──────────────────────────────────────────────────────
    L = ["# Turbo × REPA alignment-drift probe (Phase 0)\n"]
    L.append(
        f"- adapter: `{args.adapter}`\n"
        f"- images: **{n_scored}** ({n_skipped} skipped) · {n_seeds} noise draws "
        f"averaged · σ grid {sigma_grid}\n"
        f"- ruler: unweighted relational (Gram) align loss, block {args.layer} vs "
        f"{args.encoder}, spatial_norm={'on' if args.spatial_norm else 'off'} — "
        f"identical math to the training term (`relational_align_loss`)\n"
        f"- gate: excess ≥ {args.gate_excess:.0%} at any σ ≥ {args.gate_sigma}\n"
        f"- **verdict: {verdict}**\n"
    )
    for r in reasons:
        L.append(f"  - GATE: {r}")

    L.append("\n## Align loss by σ (higher = worse aligned to PE-Spatial)\n")
    L.append("| σ | base | student | Δ (student−base) | excess | frac worse | n |")
    L.append("|---|---|---|---|---|---|---|")
    for s in sigma_grid:
        m = summary[s]
        L.append(
            f"| {s:.2f} | {_fmt(m['base_mean'], '.5f')} ± {_fmt(m['base_std'], '.5f')} "
            f"| {_fmt(m['student_mean'], '.5f')} ± {_fmt(m['student_std'], '.5f')} "
            f"| {_fmt(m['delta_mean'], '+.5f')} | {m['excess'] * 100:+.1f}% "
            f"| {m['frac_worse']:.2f} | {m['n']} |"
        )

    L.append("\n## Reading it\n")
    L.append(
        "- Arms share states AND noise draws, so Δ is a paired per-image "
        "statistic; `frac worse` is its sign consistency.\n"
        "- **DRIFT** ⇒ proceed to Phase 1 (the `[repa]` section in turbo.toml). "
        "Note *where*: drift at σ ≥ 0.75 matches the caption-ranking gap and "
        "supports the primary real-data arm; a flat-then-clean-end profile "
        "would instead point at the teacher-Gram fallback arm.\n"
        "- **NO-DRIFT** ⇒ the representation damage is not PE-visible — write "
        "the negative finding and close; the caption degradation is "
        "text-conditioning, owned by soft-rank "
        "(`docs/proposal/turbo_caption_ranking.md`).\n"
        "- Base-arm values double as the σ profile of REPA's own signal on this "
        "backbone (how much PE structure survives at each noise level).\n"
    )

    run_dir = make_run_dir("turbo_repa", label=args.label or "alignment-drift")
    (run_dir / "alignment_drift.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    csv = run_dir / "alignment_drift.csv"
    with csv.open("w") as f:
        f.write(
            "sigma,base_mean,base_std,student_mean,student_std,"
            "delta_mean,delta_std,excess,frac_worse,n\n"
        )
        for s in sigma_grid:
            m = summary[s]
            f.write(
                f"{s},{m['base_mean']},{m['base_std']},{m['student_mean']},"
                f"{m['student_std']},{m['delta_mean']},{m['delta_std']},"
                f"{m['excess']},{m['frac_worse']},{m['n']}\n"
            )

    metrics = {
        "adapter": args.adapter,
        "n_images": n_scored,
        "n_skipped": n_skipped,
        "num_seeds": n_seeds,
        "sigma_grid": sigma_grid,
        "layer": args.layer,
        "encoder": args.encoder,
        "spatial_norm": bool(args.spatial_norm),
        "summary": {str(s): summary[s] for s in sigma_grid},
        "gate": {
            "excess": args.gate_excess,
            "sigma": args.gate_sigma,
            "reasons": reasons,
        },
        "drifted": drifted,
        "verdict": verdict,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["alignment_drift.md", "alignment_drift.csv"],
        device=device,
    )

    log.info("\n" + "=" * 70)
    log.info(f"  Turbo × REPA alignment-drift probe → {run_dir}")
    for s in sigma_grid:
        m = summary[s]
        log.info(
            f"  σ={s:.2f}  base {m['base_mean']:.5f}  student {m['student_mean']:.5f}"
            f"  excess {m['excess'] * 100:+.1f}%  frac_worse {m['frac_worse']:.2f}"
        )
    log.info(f"  VERDICT: {verdict}")
    for r in reasons:
        log.info(f"    - {r}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
