#!/usr/bin/env python3
"""REPA layer × σ probe — where does the *base* DiT hold its PE-alignable
representation, and does that depth move with the noise level?

THE QUESTION. The shipped REPA term anchors block **8** of 28 (≈29% depth) —
a number transplanted from Yu et al.'s SiT-XL/2 ablation (arXiv:2410.06940),
never measured on Anima's backbone + PE-Spatial target + flow schedule. Two
things could be left on the table:

  1. a *better static layer* — the depth where the frozen DiT's mid-block
     representation is naturally most aligned to the encoder, where REPA
     supervision has the most signal and the least work to do;
  2. a *timestep-dependent layer* — at high σ the network needs more depth to
     extract semantics (peak shifts deeper, ceiling lower); at low σ semantics
     appear early (peak shallower, higher). If the alignment ridge moves with
     σ, a fixed layer is leaving alignment on the table at the σ ends.

THE RULER = **centered linear CKA** (Kornblith et al. 2019) between the pooled
DiT block tokens and the PE-Spatial patch tokens. CKA is a normalized
Gram-of-Grams overlap — the *no-training analog of the relational Gram loss the
training term optimizes* (``library/training/repa.py::relational_gram_loss``),
and its centering removes the per-token DC the same way ``spatial_norm`` does.
So argmax-over-layer CKA on the frozen model predicts where the trained Gram
match will have the most leverage, with **zero training compute**. We report the
actual ``relational_align_loss`` (spatial_norm on) alongside as a cross-check —
argmin-loss should track argmax-CKA; if it does, CKA is a faithful proxy.

THE CONFOUND + ITS CONTROL. Raw matched CKA is inflated by *shared spatial
layout*: any two feature maps of the same image registered to the same grid
correlate at the token-token level regardless of semantic content, and the
deep DiT blocks (which reconstruct the caption-driven output field) trip this
hard — their CKA to PE stays high even at σ→1 (pure noise), which cannot be
genuine input-content alignment. To strip it we add a **mismatched control**:
CKA of image *i*'s DiT tokens against a *different same-grid image*'s PE
tokens. The **gap** ``CKA_matched − CKA_mismatched`` is the content-*specific*
alignment — the part REPA can actually inject (semantic identity, not generic
layout). The headline readout is the gap; argmax-gap is the confound-free
predicted layer. (Same in-group/out-group philosophy as
``bench/repa/probe_dog_target.py``.)

CONSTRUCTION (no training, base DiT only). For each cached real latent with a
PE-Spatial sidecar, renoise ``x_σ = (1−σ)·x0 + σ·ε`` at each σ on the grid and
run ONE feature-tap forward capturing **every** block
(``forward_mini_train_dit(..., return_block_features={0..L-1},
return_features_early=True)`` — runs the full stack, skips final_layer +
unpatchify), then for each layer pool the captured tokens to the encoder grid
and score CKA / Gram-loss vs the image's own cached PE features. The image's
matched caption conditions the forward (the training operating point), mirroring
``bench/turbo_repa/probe_alignment_drift.py``.

READOUTS (all from the one layer×σ heatmap):

  * ``l*`` static — argmax-layer of the σ-band-averaged CKA (band = the
    semantic-commitment window, default σ∈[0.45, 0.9]; Anima resolves x0 by
    σ≈0.45). Reported against the shipped layer 8.
  * ``l*(σ)`` ridge — argmax-layer per σ column; its span + sign of the
    layer-vs-σ trend say whether a dynamic / soft-multilayer schedule is
    justified (deeper-at-high-σ is the predicted shape).
  * σ-weight profile — the per-σ *alignment ceiling* (max-over-layer CKA),
    normalized to its peak: where the ceiling collapses there is no PE signal
    to align to (locked σ<0.45 tail / noise σ→1 ceiling), so REPA weight is
    wasted there. Drops straight into an σ-dependent ``repa_weight`` schedule.

GATES (pre-registered):

  * STATIC win — ``|l* − 8| ≥ --layer_margin`` AND ``CKA(l*) − CKA(8) ≥
    --cka_margin`` on the band ⇒ move the layer; else layer 8 is within margin.
  * DYNAMIC win — ridge span over the band ``≥ --ridge_span`` ⇒ a fixed layer
    is suboptimal across σ; realize as soft multi-layer alignment weighted by
    ``w_l(σ)`` (hard per-step switching fights the mixed-σ batch / single hook).
  * σ-WEIGHT — report ceiling collapse at the σ ends as the data for an anneal /
    σ-weight, regardless of the layer verdicts.

Run from anima_lora/::

    uv run python bench/repa/probe_layer_sigma_cka.py --num_samples 96
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

from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402

# Pair discovery + caption-embedding cache shared with the drift / reward probes
# so the pool semantics (and thus the σ profile) stay comparable across REPA
# benches.
from bench.soft_tokens_contrastive.reward_premise_probe import (  # noqa: E402
    EmbCache,
    _fmt,
    discover_pairs,
)
from library.io.cache import load_cached_latents  # noqa: E402
from library.training.repa import (  # noqa: E402
    dog_standardize,
    pool_dit_tokens_to_grid,
    relational_gram_loss,
    resolve_pe_grid,
)
from library.vision.buckets import get_bucket_spec  # noqa: E402

log = logging.getLogger("bench.repa.layer_sigma_cka")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATA = "post_image_dataset/lora"
# Dense grid straddling the σ≈0.45 x0-resolution inflection so the ridge motion
# (if any) is visible at both ends; the shipped REPA term sees the full [0,1].
DEFAULT_SIGMAS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
SHIPPED_LAYER = 8


def _pe_sidecar(te_path: str, stem: str, encoder: str) -> str:
    """PE sidecar next to the TE cache: ``{stem}_anima_{encoder}.safetensors``."""
    return os.path.join(os.path.dirname(te_path), f"{stem}_anima_{encoder}.safetensors")


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """Centered linear CKA between token clouds ``x (N,p)`` and ``y (N,q)``.

    Dual (Gram) form: center each side over the ``N`` tokens, build
    ``K = XcXcᵀ`` / ``L = YcYcᵀ`` ``(N,N)``, return ``⟨K,L⟩_F /(‖K‖‖L‖)``.
    Scale-/dimension-invariant in [0,1]; the centering is the CKA analog of the
    relational arm's ``spatial_norm`` DC removal. ``N`` (the encoder grid token
    count) is small, so the ``N×N`` Grams are cheap regardless of ``p=2048``.
    """
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    k = x @ x.transpose(0, 1)
    ell = y @ y.transpose(0, 1)
    hsic = (k * ell).sum()
    denom = k.norm() * ell.norm()
    return float(hsic / (denom + eps))


@torch.no_grad()
def _all_block_features(anima, x_s, t_b, emb, pad, layers: set[int]) -> dict:
    """One forward → ``{layer: raw block output}`` for every requested layer.

    ``return_features_early`` runs the full stack up to the deepest tap then
    returns the capture dict (no final_layer / unpatchify). ``no_grad`` keeps
    all 28 captured states activation-free — pure measurement.
    """
    return anima.forward_mini_train_dit(
        x_s,
        t_b,
        emb,
        padding_mask=pad,
        skip_pooled_text_proj=True,
        return_block_features=set(layers),
        return_features_early=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap, vae=False, text_encoder=False)
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument(
        "--num_samples", type=int, default=96, help="images carrying a PE sidecar"
    )
    ap.add_argument(
        "--num_seeds",
        type=int,
        default=1,
        help="noise draws averaged per (image, σ); all layers share the draw.",
    )
    ap.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="block indices to tap (default: all DiT blocks).",
    )
    ap.add_argument("--encoder", default="pe_spatial")
    # REPA-DoG target band-pass (mirrors the shipped repa_target_dog lever): apply
    # difference-of-Gaussians to the PE target before the CKA/Gram, stripping the
    # low/global band. Diagnostic for whether the deep-layer alignment is
    # low-freq global-composition reconstruction (DoG should crush it) vs genuine
    # high-freq content (survives DoG). The DiT side stays centered-only, exactly
    # as training only preprocesses the target.
    ap.add_argument("--target_dog", action="store_true", help="DoG-filter the PE target")
    ap.add_argument("--dog_sigma1_div", type=float, default=16.0, help="Phase-0 winner")
    ap.add_argument("--dog_sigma2_div", type=float, default=0.0, help="0 = high-pass only")
    ap.add_argument("--dog_norm_std", type=float, default=0.0, help="0 = empirical std")
    # Semantic-commitment band for the static-layer argmax + ridge span.
    ap.add_argument("--band_lo", type=float, default=0.45)
    ap.add_argument("--band_hi", type=float, default=0.90)
    # The early "representation-forming" regime ends before the mid-stack
    # bottleneck. We separately report the best layer at/below this index — the
    # REPA-defensible target (least redundant with the flow-matching objective),
    # as opposed to the near-output deep peak.
    ap.add_argument("--repr_regime_max", type=int, default=11)
    # Pre-registered gate thresholds.
    ap.add_argument(
        "--layer_margin",
        type=int,
        default=2,
        help="min |l* − 8| (blocks) to call a static-layer move",
    )
    ap.add_argument(
        "--cka_margin",
        type=float,
        default=0.01,
        help="min band-CKA gain of l* over layer 8 to call a static-layer move",
    )
    ap.add_argument(
        "--ridge_span",
        type=int,
        default=3,
        help="min argmax-layer span across the band to justify a dynamic schedule",
    )
    add_common_args(ap)
    args = ap.parse_args()

    sigma_grid = sorted(float(s) for s in args.sigmas)
    spec = get_bucket_spec(args.encoder)
    band = [s for s in sigma_grid if args.band_lo <= s <= args.band_hi] or sigma_grid

    pairs = discover_pairs(args.data_dir)
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
    log.info(f"{len(pool_list)}/{len(pairs)} cached pairs carry a {args.encoder} sidecar")

    rng = np.random.default_rng(args.seed)
    take = min(args.num_samples, len(pool_list))
    stems = [pool_list[int(i)] for i in rng.choice(len(pool_list), take, replace=False)]

    bundle = build_anima(args, adapter=None, train_mode=False)  # base DiT only
    anima = bundle.anima
    device, dtype = bundle.device, bundle.dtype
    patch = int(anima.patch_spatial)
    n_blocks = len(anima.blocks)
    layers = sorted(args.layers) if args.layers else list(range(n_blocks))
    if any(not (0 <= ell < n_blocks) for ell in layers):
        raise SystemExit(f"--layers out of range (DiT has {n_blocks} blocks)")
    layer_set = set(layers)
    log.info(
        f"σ grid: {sigma_grid}  |  layers: {layers[0]}..{layers[-1]} "
        f"({len(layers)} taps)  |  encoder={args.encoder}  band={band}"
    )

    embs = EmbCache(pairs)
    n_seeds = max(1, args.num_seeds)

    # ── preload: PE patch tokens (CLS dropped) + grid + latent/emb, on CPU ────
    # Needed up front so we can assign each image a *same-grid* mismatched
    # partner for the confound control (CKA needs equal token count N = gh*gw).
    store: dict[str, dict] = {}
    n_skipped = 0
    for stem in stems:
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
        pe = pe.float().unsqueeze(0)  # (1, T, d_enc), CLS at 0
        n_pe = pe.shape[1] - (1 if spec.use_cls else 0)
        lat, _res, _oh, _ow = load_cached_latents(npz_path)
        x0 = lat.unsqueeze(0).unsqueeze(2)  # (1,C,1,H,W) on CPU
        H, W = x0.shape[-2], x0.shape[-1]
        gh, gw = resolve_pe_grid(spec, n_pe, H, W)
        pe_tok = (pe[:, 1:, :] if spec.use_cls else pe).contiguous()  # (1, gh*gw, d_enc)
        if args.target_dog:
            # Band-pass the target on its (gh, gw) grid — identical preprocessing
            # to the training term under repa_target_dog. CKA then re-centers.
            pe_tok = dog_standardize(
                pe_tok, gh, gw, args.dog_sigma1_div, args.dog_sigma2_div,
                args.dog_norm_std,
            ).contiguous()
        store[stem] = {
            "x0": x0,
            "emb": emb.unsqueeze(0),
            "pe_tok": pe_tok,
            "grid": (gh, gw),
            "hw": (H, W),
        }
    kept = [s for s in stems if s in store]
    if not kept:
        raise SystemExit("no images scored — check caches")

    # Same-grid mismatched partner per image (deterministic). Singletons in a
    # grid get no partner → their gap cell is dropped (recorded as NaN).
    by_grid: dict[tuple, list[str]] = {}
    for stem in kept:
        by_grid.setdefault(store[stem]["grid"], []).append(stem)
    partner: dict[str, str | None] = {}
    for grid, group in by_grid.items():
        if len(group) < 2:
            for stem in group:
                partner[stem] = None
            continue
        perm = rng.permutation(len(group))
        for k, stem in enumerate(group):
            j = perm[k]
            if group[j] == stem:  # avoid self-pairing
                j = (j + 1) % len(group)
            partner[stem] = group[j]
    n_no_partner = sum(1 for s in kept if partner[s] is None)
    log.info(
        f"preloaded {len(kept)} images over {len(by_grid)} grids "
        f"({n_no_partner} singletons → no mismatched control)"
    )

    # cka_m / cka_x / gram[layer][σ] = per-image (seed-averaged) value lists.
    cka_m: dict[int, dict[float, list[float]]] = {
        ell: {s: [] for s in sigma_grid} for ell in layers
    }
    cka_x: dict[int, dict[float, list[float]]] = {
        ell: {s: [] for s in sigma_grid} for ell in layers
    }
    gram: dict[int, dict[float, list[float]]] = {
        ell: {s: [] for s in sigma_grid} for ell in layers
    }

    for ai, stem in enumerate(kept):
        rec = store[stem]
        H, W = rec["hw"]
        gh, gw = rec["grid"]
        x0 = rec["x0"].to(device, dtype)
        x0_f = x0.float()
        emb_b = rec["emb"].to(device, dtype)
        pe_tok = rec["pe_tok"].to(device)
        pe_x = (
            store[partner[stem]]["pe_tok"].to(device)
            if partner[stem] is not None
            else None
        )
        pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)

        for s in sigma_grid:
            acc_m = {ell: [] for ell in layers}
            acc_x = {ell: [] for ell in layers}
            acc_g = {ell: [] for ell in layers}
            for sj in range(n_seeds):
                g = torch.Generator(device=device).manual_seed(
                    args.seed + ai * 1000 + sj
                )
                eps = torch.randn(x0.shape, generator=g, device=device, dtype=dtype)
                noisy = ((1.0 - s) * x0_f + s * eps.float()).to(dtype)
                t_b = torch.full((1,), float(s), device=device, dtype=dtype)
                feats = _all_block_features(anima, noisy, t_b, emb_b, pad, layer_set)
                for ell in layers:
                    dit_tok = pool_dit_tokens_to_grid(
                        feats[ell], (H, W), patch, gh, gw
                    )  # (1, gh*gw, D) fp32
                    acc_m[ell].append(linear_cka(dit_tok[0], pe_tok[0]))
                    if pe_x is not None:
                        acc_x[ell].append(linear_cka(dit_tok[0], pe_x[0]))
                    acc_g[ell].append(
                        float(relational_gram_loss(dit_tok, pe_tok, spatial_norm=True))
                    )
            for ell in layers:
                cka_m[ell][s].append(float(np.mean(acc_m[ell])))
                if pe_x is not None:
                    cka_x[ell][s].append(float(np.mean(acc_x[ell])))
                gram[ell][s].append(float(np.mean(acc_g[ell])))

        if (ai + 1) % 10 == 0 or ai + 1 == len(kept):
            log.info(f"  [{ai + 1}/{len(kept)}] scored")
        if device.type == "cuda" and (ai + 1) % 25 == 0:
            torch.cuda.empty_cache()

    n_scored = len(cka_m[layers[0]][sigma_grid[0]])
    n_control = len(cka_x[layers[0]][sigma_grid[0]])
    if n_scored == 0:
        raise SystemExit("no images scored — check caches")

    # ── aggregate to layer×σ means ───────────────────────────────────────────
    L_idx = layers
    S_idx = sigma_grid

    def _mean_grid(d):
        return np.array([[float(np.mean(d[ell][s])) for s in S_idx] for ell in L_idx])

    cka_mean = _mean_grid(cka_m)  # matched CKA
    cka_std = np.array(
        [[float(np.std(cka_m[ell][s])) for s in S_idx] for ell in L_idx]
    )
    gram_mean = _mean_grid(gram)
    # Mismatched (control) + gap; gap is the content-specific alignment.
    if n_control > 0:
        cka_mismatch = _mean_grid(cka_x)
        gap_mean = cka_mean - cka_mismatch
    else:
        cka_mismatch = np.full_like(cka_mean, np.nan)
        gap_mean = np.full_like(cka_mean, np.nan)

    # σ→1 floor subtraction. At the largest σ the input is ~pure noise, so any
    # remaining matched/gap alignment is caption-driven *output reconstruction*
    # — the part the diffusion loss already trains. Subtracting each layer's own
    # floor isolates the alignment gained from *processing the noisy input*,
    # which is exactly what REPA regularizes (a decoder layer reconstructing the
    # caption keeps a high floor → its useful_gap collapses; a representation
    # layer has ~0 floor → its low-σ gap survives). This is the metric the
    # static-layer / ridge verdicts key on.
    floor = gap_mean[:, -1:].copy()  # per-layer gap at the largest σ
    useful_gap = np.clip(gap_mean - floor, 0.0, None) if n_control > 0 else gap_mean

    band_cols = [S_idx.index(s) for s in band]

    def _layer_at(row_idx: int) -> int:
        return L_idx[int(row_idx)]

    # Headline metric = floor-subtracted content gap (useful_gap): strips both
    # the shared-layout confound (via the mismatch) and the caption-reconstruction
    # confound (via the σ→1 floor). Fall back to matched CKA only if the control
    # couldn't run (all singletons — shouldn't happen at N≈96).
    have_control = n_control > 0
    score = useful_gap if have_control else cka_mean

    band_score = score[:, band_cols].mean(axis=1)  # per-layer band useful_gap
    band_gap = gap_mean[:, band_cols].mean(axis=1)  # pre-floor gap, for report
    band_cka = cka_mean[:, band_cols].mean(axis=1)  # matched, for the report
    band_floor = floor[:, 0] if have_control else np.full(len(L_idx), np.nan)
    band_mismatch = (
        cka_mismatch[:, band_cols].mean(axis=1)
        if have_control
        else np.full(len(L_idx), np.nan)
    )

    l_star_band = _layer_at(int(np.argmax(band_score)))
    l_star_all = _layer_at(int(np.argmax(score.mean(axis=1))))
    l_star_matched = _layer_at(int(np.argmax(band_cka)))  # the confounded argmax

    # Early "representation regime" best layer (≤ repr_regime_max) — the
    # REPA-defensible target, least redundant with the flow-matching objective.
    early_rows = [i for i, ell in enumerate(L_idx) if ell <= args.repr_regime_max]
    if early_rows:
        early_row = early_rows[int(np.argmax(band_score[early_rows]))]
        l_star_early = L_idx[early_row]
        early_band_score = float(band_score[early_row])
    else:
        l_star_early, early_band_score = SHIPPED_LAYER, float("nan")
    ridge = [_layer_at(int(np.argmax(score[:, j]))) for j in range(len(S_idx))]
    ridge_band = [ridge[S_idx.index(s)] for s in band]
    ridge_span = (max(ridge_band) - min(ridge_band)) if ridge_band else 0

    # σ-weight profile = per-σ ceiling of the content-specific gap, peak-norm.
    ceiling = score.max(axis=0)
    sigma_weight = ceiling / max(float(ceiling.max()), 1e-12)

    # Layer-8 reference (nearest tapped layer to the shipped one).
    ship_row = int(np.argmin([abs(ell - SHIPPED_LAYER) for ell in L_idx]))
    ship_layer = L_idx[ship_row]
    ship_band_score = float(band_score[ship_row])
    best_band_score = float(band_score.max())
    # Does the confounded (matched-CKA) argmax disagree with the gap argmax? If
    # so, the deep-layer peak was layout, not content — the whole point.
    confound_flag = l_star_matched != l_star_band

    # Direction of the ridge vs σ (Spearman-ish sign via Pearson on ranks-free
    # raw values is fine here — we only want the sign / rough strength).
    ridge_arr = np.array([ridge[S_idx.index(s)] for s in S_idx], dtype=float)
    sig_arr = np.array(S_idx, dtype=float)
    if ridge_arr.std() > 0:
        ridge_corr = float(np.corrcoef(ridge_arr, sig_arr)[0, 1])
    else:
        ridge_corr = 0.0

    # ── gates ─────────────────────────────────────────────────────────────────
    metric_name = (
        "useful_gap (gap − σ→1 floor)" if have_control else "matched CKA"
    )
    static_move = (
        abs(l_star_band - ship_layer) >= args.layer_margin
        and (best_band_score - ship_band_score) >= args.cka_margin
    )
    dynamic_win = ridge_span >= args.ridge_span
    # ceiling collapse at the σ ends → REPA weight wasted there.
    lo_collapse = float(sigma_weight[0])
    hi_collapse = float(sigma_weight[-1])

    static_verdict = (
        f"STATIC MOVE: l*={l_star_band} beats shipped layer {ship_layer} on the "
        f"σ∈[{args.band_lo},{args.band_hi}] band "
        f"({metric_name} {best_band_score:.3f} vs {ship_band_score:.3f}, "
        f"+{best_band_score - ship_band_score:.3f})."
        if static_move
        else (
            f"NO STATIC MOVE: l*={l_star_band} within margin of shipped layer "
            f"{ship_layer} (band {metric_name} {best_band_score:.3f} vs "
            f"{ship_band_score:.3f}, Δ {best_band_score - ship_band_score:+.3f} "
            f"< {args.cka_margin})."
        )
    )
    confound_verdict = (
        f"CONFOUND CONFIRMED: raw matched-CKA argmax is layer {l_star_matched} but "
        f"the floor-subtracted content gap (useful_gap) argmax is layer "
        f"{l_star_band} — the deep-layer matched-CKA peak is caption-driven output "
        "reconstruction (high σ→1 floor), not input-representation alignment. "
        "Trust useful_gap."
        if (have_control and confound_flag)
        else (
            f"NO CONFOUND SPLIT: raw matched-CKA and useful_gap argmax agree "
            f"(layer {l_star_band})."
            if have_control
            else "CONTROL UNAVAILABLE: all images were grid-singletons; reporting "
            "raw matched CKA (confounded — interpret with care)."
        )
    )
    dynamic_verdict = (
        f"DYNAMIC WIN: argmax-layer ridge spans {min(ridge_band)}..{max(ridge_band)} "
        f"({ridge_span} blocks) across the band (corr l*vsσ {ridge_corr:+.2f}, "
        f"{'deeper@high-σ' if ridge_corr > 0 else 'shallower@high-σ' if ridge_corr < 0 else 'flat'}) "
        "⇒ realize as σ-weighted soft multi-layer alignment."
        if dynamic_win
        else (
            f"NO DYNAMIC WIN: ridge span {ridge_span} block(s) over the band "
            f"(< {args.ridge_span}) — a fixed layer is fine; spend the lever on "
            "the σ-weight instead."
        )
    )

    # ── artifacts ──────────────────────────────────────────────────────────────
    run_dir = make_run_dir("repa", label=args.label or "layer-sigma-cka")

    # full heatmaps + derived schedules for downstream wiring
    np.savez(
        run_dir / "heatmap.npz",
        layers=np.array(L_idx),
        sigmas=np.array(S_idx),
        cka_mean=cka_mean,
        cka_std=cka_std,
        cka_mismatch=cka_mismatch,
        gap_mean=gap_mean,
        useful_gap=useful_gap,
        floor=floor[:, 0],
        gram_mean=gram_mean,
        band_score=band_score,
        ridge=np.array(ridge),
        sigma_weight=sigma_weight,
        l_star_band=l_star_band,
        l_star_all=l_star_all,
        l_star_matched=l_star_matched,
        shipped_layer=ship_layer,
        have_control=have_control,
    )

    # CSV: layer × σ — matched, mismatched, gap side by side.
    with (run_dir / "cka_by_layer_sigma.csv").open("w") as f:
        cols = ",".join(
            f"{tag}_s{s:g}" for tag in ("match", "mismatch", "gap") for s in S_idx
        )
        f.write("layer," + cols + ",band_gap\n")
        for i, ell in enumerate(L_idx):
            m = ",".join(f"{cka_mean[i, j]:.4f}" for j in range(len(S_idx)))
            x = ",".join(f"{cka_mismatch[i, j]:.4f}" for j in range(len(S_idx)))
            gp = ",".join(f"{gap_mean[i, j]:.4f}" for j in range(len(S_idx)))
            f.write(f"{ell},{m},{x},{gp},{band_score[i]:.4f}\n")

    # Markdown summary.
    M = ["# REPA layer × σ probe — base-model PE alignment landscape\n"]
    M.append(
        f"- images: **{n_scored}** ({n_skipped} skipped, {n_control} with a "
        f"same-grid mismatched control) · {n_seeds} noise draw(s) · base DiT, no "
        f"adapter · encoder={args.encoder}\n"
        f"- ruler: centered linear CKA(pooled block tokens, PE patch tokens). "
        f"**Headline = the content-specific gap** (matched − mismatched), which "
        f"strips the shared-layout confound; matched CKA shown for contrast.\n"
        f"- shipped layer **{ship_layer}** · band σ∈[{args.band_lo}, {args.band_hi}]"
        f"{' · **target=DoG** (σ1=min/' + format(args.dog_sigma1_div, 'g') + ')' if args.target_dog else ''}\n"
    )
    M.append("\n## Verdicts\n")
    M.append(f"- **{static_verdict}**")
    M.append(f"- **{confound_verdict}**")
    M.append(
        f"- **REPR-REGIME l\\*:** best layer ≤{args.repr_regime_max} (the "
        f"non-redundant, representation-forming zone) is **{l_star_early}** "
        f"(useful_gap {early_band_score:.3f}). The global l\\*={l_star_band} sits "
        "in the near-output decoder regime — high alignment, but most redundant "
        "with the flow-matching loss; a base-model probe can't tell whether "
        "injecting REPA there helps or just duplicates the objective. Trust the "
        "repr-regime layer unless a training A/B says otherwise."
    )
    M.append(f"- **{dynamic_verdict}**")
    M.append(
        f"- **σ-WEIGHT:** gap ceiling (peak-normalized) = "
        f"{lo_collapse:.2f} at σ={S_idx[0]:g} → {hi_collapse:.2f} at σ={S_idx[-1]:g}; "
        f"peak at σ={S_idx[int(np.argmax(ceiling))]:g}. Where the content-specific "
        "ceiling is low there is no semantic PE structure to align to — "
        "down-weight REPA there."
    )

    M.append(
        "\n## useful_gap by layer × σ "
        "(floor-subtracted content gap; **bold** = column argmax)\n"
    )
    header = "| layer | " + " | ".join(f"σ={s:g}" for s in S_idx) + " | band |"
    M.append(header)
    M.append("|---|" + "---|" * (len(S_idx) + 1))
    col_arg = [int(np.argmax(score[:, j])) for j in range(len(S_idx))]
    for i, ell in enumerate(L_idx):
        cells = []
        for j in range(len(S_idx)):
            v = f"{score[i, j]:.3f}"
            cells.append(f"**{v}**" if col_arg[j] == i else v)
        star = (
            " ⟵ l*"
            if ell == l_star_band
            else (" (shipped)" if ell == ship_layer else "")
        )
        M.append(f"| {ell}{star} | " + " | ".join(cells) + f" | {band_score[i]:.3f} |")

    M.append(
        "\n## Decomposition by layer (band-averaged) — how the confound peels off\n"
    )
    M.append("| layer | matched | mismatched | gap | σ→1 floor | useful_gap |")
    M.append("|---|---|---|---|---|---|")
    for i, ell in enumerate(L_idx):
        tag = (
            " ⟵ raw argmax"
            if ell == l_star_matched
            else (" ⟵ l*" if ell == l_star_band else "")
        )
        M.append(
            f"| {ell}{tag} | {band_cka[i]:.3f} | {band_mismatch[i]:.3f} | "
            f"{band_gap[i]:.3f} | {band_floor[i]:.3f} | {band_score[i]:.3f} |"
        )

    M.append("\n## Derived schedules (from the gap)\n")
    M.append("| σ | argmax layer l*(σ) | gap ceiling | σ-weight (norm) |")
    M.append("|---|---|---|---|")
    for j, s in enumerate(S_idx):
        M.append(f"| {s:g} | {ridge[j]} | {ceiling[j]:.3f} | {sigma_weight[j]:.3f} |")

    M.append("\n## Reading it\n")
    M.append(
        "- The **gap** (matched − mismatched) is the alignment REPA can actually "
        "inject: a layer with high matched CKA but ~zero gap is matching generic "
        "spatial layout, not semantic identity.\n"
        "- A deep-layer *matched-CKA* peak that vanishes in the gap is the "
        "output-reconstruction confound (the velocity field is registered to the "
        "target image); aligning it just duplicates the diffusion loss.\n"
        "- **l\\*(σ) ridge** climbing with σ ⇒ alignable content lives deeper "
        "under more noise; one layer can't sit on the ridge at both ends → soft "
        "multi-layer alignment with weights peaked at l\\*(σ).\n"
        "- **σ-weight** is a drop-in σ-dependent `repa_weight` profile.\n"
    )
    (run_dir / "summary.md").write_text("\n".join(M) + "\n", encoding="utf-8")

    metrics = {
        "n_images": n_scored,
        "n_control": n_control,
        "n_skipped": n_skipped,
        "num_seeds": n_seeds,
        "encoder": args.encoder,
        "have_control": bool(have_control),
        "headline_metric": metric_name,
        "layers": L_idx,
        "sigma_grid": S_idx,
        "band": band,
        "shipped_layer": ship_layer,
        "l_star_band": l_star_band,
        "l_star_all": l_star_all,
        "l_star_matched": l_star_matched,
        "l_star_early": l_star_early,
        "early_band_score": early_band_score,
        "repr_regime_max": args.repr_regime_max,
        "target_dog": bool(args.target_dog),
        "dog_sigma1_div": args.dog_sigma1_div if args.target_dog else None,
        "confound_flag": bool(confound_flag),
        "band_score_l_star": best_band_score,
        "band_score_shipped": ship_band_score,
        "band_cka_shipped": float(band_cka[ship_row]),
        "band_floor": [float(x) for x in band_floor],
        "band_gap": [float(x) for x in band_gap],
        "ridge": ridge,
        "ridge_band_span": int(ridge_span),
        "ridge_corr_sigma": ridge_corr,
        "sigma_weight": [float(w) for w in sigma_weight],
        "ceiling": [float(c) for c in ceiling],
        "static_move": bool(static_move),
        "dynamic_win": bool(dynamic_win),
        "static_verdict": static_verdict,
        "confound_verdict": confound_verdict,
        "dynamic_verdict": dynamic_verdict,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["summary.md", "cka_by_layer_sigma.csv", "heatmap.npz"],
        device=device,
    )

    log.info("\n" + "=" * 72)
    log.info(f"  REPA layer × σ probe → {run_dir}")
    log.info(f"  headline metric: {metric_name}")
    log.info(
        f"  shipped layer {ship_layer}: band {ship_band_score:.3f} "
        f"(matched {band_cka[ship_row]:.3f})"
    )
    log.info(f"  l* (global)   {l_star_band}: band {best_band_score:.3f}")
    log.info(
        f"  l* (repr ≤{args.repr_regime_max}) {l_star_early}: band {early_band_score:.3f}"
    )
    log.info(f"  matched-CKA argmax: {l_star_matched} (confound={confound_flag})")
    log.info(f"  l*(σ) ridge:  {ridge}  (span {ridge_span}, corr {ridge_corr:+.2f})")
    log.info(f"  σ-weight:     {[round(float(w), 2) for w in sigma_weight]}")
    log.info(f"  {static_verdict}")
    log.info(f"  {confound_verdict}")
    log.info(f"  {dynamic_verdict}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
