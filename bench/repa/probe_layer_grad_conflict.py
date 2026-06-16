#!/usr/bin/env python3
"""REPA layer × σ probe — does adding the alignment loss at layer ℓ *help* the
flow-matching objective, or just duplicate / fight it?

WHY THIS EXISTS (the CKA probe's blind spot). ``probe_layer_sigma_cka.py``
measures *where PE-alignable structure already lives* in the frozen DiT — a
**correlational, single-objective** question. It (correctly, on its own terms)
ranks the near-output decoder blocks (24–27) highest, because that is where the
backbone's representation is already entangled with the caption-driven velocity
field the diffusion loss trains. But the training decision is a different,
**causal / two-objective** question: at which layer does *injecting* an
alignment gradient shape directions the flow-matching (FM) loss leaves
unconstrained (a free win) versus directions it already pins (redundant) or
pushes against (interfering)?

The shipped layer-8 vs layer-26 A/B (``anima_repa4_tenth`` / ``anima_repa5_tenth``)
settled it the expensive way: layer 26 reached *better* alignment but a *worse*
FM loss — high CKA, negative transfer. The CKA argmax pointed at the layer the
A/B rejected. This probe measures the quantity the A/B actually cares about, on
the base model, all 28 layers in one pass, for a fraction of one training run.

THE RULER = the **gradient interaction** between the two losses at each block's
output activation hₗ. One full forward (no early exit — we need the velocity head
for L_fm) captures every block output as a live graph node, then:

  * ``g_fm[ℓ] = ∂L_fm/∂hₗ``      — one backward of the rectified-flow loss
    ``MSE(v_pred, ε−x0)`` w.r.t. *all* taps (full-graph total derivative: how the
    FM objective wants to move layer ℓ's output).
  * ``g_repa[ℓ] = ∂L_repa(ℓ)/∂hₗ`` — per-layer *local* backward of the relational
    Gram loss at ℓ (the subgraph is just pool → normalize → Gram → MSE, the same
    near-free probe the training grad-heatmap uses). Local on purpose: we want
    REPA-at-ℓ's own push, not deep-REPA backprop leaking into shallow taps.

Measured **per spatial token** (cosine over the D-channel axis at each grid
location), then aggregated **weighted by the FM per-token grad norm**. The global
flattened cosine is useless here — in ~12M-dim activation space any two gradients
are near-orthogonal by construction (cos noise floor ~1/√dim ≈ 3e-4), so it
collapses to 0 for *every* layer. The interaction that matters is local: does FM
and REPA want to move each location's feature the same way, on the tokens FM
actually stresses (the Fisher-relevant weighting — high-FM-grad tokens are the
constrained subspace).

READOUTS (per layer, averaged over images and the σ band):

  * ``cos`` (signed, FM-stress-weighted per-token) = ⟨g_fm, g_repa⟩ per token.
    **+** the two objectives reinforce on FM's stressed tokens (REPA duplicates an
    FM direction → redundant), **0** orthogonal (REPA shapes what FM ignores → the
    free, non-redundant win REPA is *for*), **−** conflict (REPA fights FM on the
    tokens it cares about → negative transfer, raises FM loss).
  * ``redundancy`` = FM-weighted per-token cos². The fraction of the REPA
    gradient's per-token energy lying along the FM per-token direction (the local,
    rank-1 empirical-Fisher direction): 0 = REPA in FM's null space (ideal), 1 =
    fully inside the FM-constrained subspace (redundant *or* conflicting — the sign
    of ``cos`` disambiguates). Weighting by ‖g_fm_token‖ restricts the estimate to
    the subspace the flow-matching loss occupies, sidestepping the intractable
    activation-space covariance.
  * ``mag`` = ‖g_repa‖/‖g_fm‖ (global at the tap) — is REPA's push even material
    here? A *large* mag with cos≈0 is itself negative transfer: a big orthogonal
    push drags the representation sideways off the FM manifold even without a
    sign flip — the deep-layer signature when that representation feeds the head.

THE GATE (what the A/B confirmed, now a one-pass predictor). The REPA-favorable
layer minimizes redundancy among layers whose push is **non-conflicting**
(cos ≥ −``--conflict_eps``) and **material** (mag ≥ ``--min_mag``). We print the
shipped-8 vs deep-26 head-to-head explicitly: the prediction is that 8 sits at
lower redundancy / less conflict than 26, i.e. the gradient geometry — not the
static CKA — is what tracks the training outcome.

LIMITATION (stated up front). This is a *local linearization at probe time*: it
predicts the instantaneous interaction, not the full training trajectory (a layer
can start orthogonal and curve into conflict). It measures the **right** quantity
(interaction with FM) where CKA measured a confounded proxy (static
alignability), but it remains a proxy — the training A/B is ground truth. Run on
the base DiT (no adapter): the question is where REPA *would* help, before any
adaptation has happened.

Run from anima_lora/::

    uv run python bench/repa/probe_layer_grad_conflict.py --num_samples 48
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
from bench.soft_tokens_contrastive.reward_premise_probe import (  # noqa: E402
    EmbCache,
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

log = logging.getLogger("bench.repa.layer_grad_conflict")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATA = "post_image_dataset/lora"
# Same σ grid as the CKA probe so the two are read side by side; the band default
# matches (semantic-commitment window, Anima resolves x0 by σ≈0.45).
DEFAULT_SIGMAS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
SHIPPED_LAYER = 8
DEEP_LAYER = 26  # the CKA-argmax / failed-A/B layer, printed head-to-head


def _pe_sidecar(te_path: str, stem: str, encoder: str) -> str:
    return os.path.join(os.path.dirname(te_path), f"{stem}_anima_{encoder}.safetensors")


def _repa_loss_at(
    captured: torch.Tensor,
    pe_tok: torch.Tensor,
    latent_hw: tuple[int, int],
    patch: int,
    gh: int,
    gw: int,
    *,
    spatial_norm: bool,
    dog: bool,
    dog_sigma1_div: float,
    dog_sigma2_div: float,
    dog_norm_std: float,
) -> torch.Tensor:
    """Relational Gram alignment at one captured tap — the training term.

    ``pe_tok`` is CLS-dropped ``(1, gh*gw, d_enc)`` already (and DoG-filtered if
    ``dog``; we filter once at preload, exactly as the adapter filters the target
    each step). So this just pools the DiT side and matches the Gram structure —
    the identical math to ``REPAMethodAdapter.extra_forwards``.
    """
    dit_tok = pool_dit_tokens_to_grid(captured, latent_hw, patch, gh, gw)
    # DoG already applied to pe_tok at preload → never re-apply spatial_norm here
    # (DoG and spatial_norm are mutually exclusive in the adapter).
    return relational_gram_loss(dit_tok, pe_tok, spatial_norm=spatial_norm and not dog)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap, vae=False, text_encoder=False)
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument(
        "--num_samples",
        type=int,
        default=48,
        help="images carrying a PE sidecar (grad forward is heavier than the "
        "no-grad CKA probe — default lower).",
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
    # Mirror the run-under-test's target preprocessing so the REPA gradient is the
    # one the adapter actually backprops (repa4/repa5 trained with target_dog).
    ap.add_argument("--spatial_norm", action="store_true", help="iREPA target std")
    ap.add_argument(
        "--target_dog", action="store_true", help="DoG-filter the PE target"
    )
    ap.add_argument("--dog_sigma1_div", type=float, default=16.0)
    ap.add_argument("--dog_sigma2_div", type=float, default=0.0)
    ap.add_argument("--dog_norm_std", type=float, default=0.0)
    ap.add_argument("--band_lo", type=float, default=0.45)
    ap.add_argument("--band_hi", type=float, default=0.90)
    ap.add_argument(
        "--repr_regime_max",
        type=int,
        default=11,
        help="report the best gradient-favorable layer at/below this index too.",
    )
    # Gate thresholds for the "REPA-favorable layer" pick.
    ap.add_argument(
        "--conflict_eps",
        type=float,
        default=0.0,
        help="layers with mean cos < −conflict_eps are excluded as conflicting "
        "before picking the gentlest (lowest-mag) layer.",
    )
    add_common_args(ap)
    args = ap.parse_args()

    if getattr(args, "compile", False):
        # Grad flows back into captured intermediate activations; the native-flatten
        # compiled-block path + per-image dynamic shapes make autograd.grad on taps
        # fragile and buy nothing for a few-dozen-image probe. Force eager.
        log.warning("grad-conflict probe runs eager; ignoring --compile.")
        args.compile = False

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
            f"no {args.encoder} sidecars under {args.data_dir} — run "
            "`make preprocess-pe` for the spatial encoder first"
        )
    log.info(
        f"{len(pool_list)}/{len(pairs)} cached pairs carry a {args.encoder} sidecar"
    )

    rng = np.random.default_rng(args.seed)
    take = min(args.num_samples, len(pool_list))
    stems = [pool_list[int(i)] for i in rng.choice(len(pool_list), take, replace=False)]

    # Full-stack backprop through 28 blocks at native resolution OOMs a 16GB card
    # if every block's activations are retained. Gradient checkpointing keeps only
    # the (cheap) captured block-output tensors and recomputes within-block
    # activations during backward — exactly what the taps need. It's gated on
    # anima.training, so train_mode=True; the DiT has no train-only stochastic
    # layers (deterministic AdaLN, no dropout/BN) and its params stay frozen, so
    # numerics are unchanged from eval. --cpu_offload_checkpointing trims further
    # on very tight cards.
    if not getattr(args, "gradient_checkpointing", False):
        log.info("forcing gradient checkpointing (full-stack grad backprop)")
        args.gradient_checkpointing = True
    bundle = build_anima(args, adapter=None, train_mode=True)  # base DiT, frozen
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
        f"{'  | target=DoG' if args.target_dog else ''}"
    )

    embs = EmbCache(pairs)
    n_seeds = max(1, args.num_seeds)

    # ── preload: PE patch tokens (CLS dropped, DoG once) + grid + latent/emb ───
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
        pe_tok = (pe[:, 1:, :] if spec.use_cls else pe).contiguous()  # (1, gh*gw, d)
        if args.target_dog:
            pe_tok = dog_standardize(
                pe_tok,
                gh,
                gw,
                args.dog_sigma1_div,
                args.dog_sigma2_div,
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
    log.info(f"preloaded {len(kept)} images ({n_skipped} skipped)")

    # cos / cos² / mag / raw norms per (layer, σ) → per-image (seed-avg) lists.
    cos_d: dict[int, dict[float, list[float]]] = {
        ell: {s: [] for s in sigma_grid} for ell in layers
    }
    cos2_d: dict[int, dict[float, list[float]]] = {
        ell: {s: [] for s in sigma_grid} for ell in layers
    }
    mag_d: dict[int, dict[float, list[float]]] = {
        ell: {s: [] for s in sigma_grid} for ell in layers
    }
    # raw norms disambiguate the mag trend: is the orthogonal-push ratio growing
    # with depth because ‖g_repa‖ rises or because ‖g_fm‖ falls toward the head?
    nfm_d: dict[int, dict[float, list[float]]] = {
        ell: {s: [] for s in sigma_grid} for ell in layers
    }
    nrp_d: dict[int, dict[float, list[float]]] = {
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
        pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)

        for s in sigma_grid:
            acc_cos = {ell: [] for ell in layers}
            acc_cos2 = {ell: [] for ell in layers}
            acc_mag = {ell: [] for ell in layers}
            acc_nfm = {ell: [] for ell in layers}
            acc_nrp = {ell: [] for ell in layers}
            for sj in range(n_seeds):
                g = torch.Generator(device=device).manual_seed(
                    args.seed + ai * 1000 + sj
                )
                eps = torch.randn(x0.shape, generator=g, device=device, dtype=dtype)
                noisy = ((1.0 - s) * x0_f + s * eps.float()).to(dtype)
                noisy = noisy.detach().requires_grad_(True)  # leaf → graph tracks
                t_b = torch.full((1,), float(s), device=device, dtype=dtype)
                # rectified-flow target (train.py::get_noise_pred_and_target).
                target = (eps - x0).float()

                with torch.enable_grad():
                    velocity, feats = anima.forward_mini_train_dit(
                        noisy,
                        t_b,
                        emb_b,
                        padding_mask=pad,
                        skip_pooled_text_proj=True,
                        return_block_features=layer_set,
                    )
                    l_fm = torch.nn.functional.mse_loss(velocity.float(), target)
                    # FM grad into every tap in one full-graph backward.
                    feats_list = [feats[ell] for ell in layers]
                    g_fm = torch.autograd.grad(l_fm, feats_list, retain_graph=True)
                    # REPA grad per tap — local subgraph only (no cross-layer leak).
                    for idx, ell in enumerate(layers):
                        l_repa = _repa_loss_at(
                            feats[ell],
                            pe_tok,
                            (H, W),
                            patch,
                            gh,
                            gw,
                            spatial_norm=args.spatial_norm,
                            dog=args.target_dog,
                            dog_sigma1_div=args.dog_sigma1_div,
                            dog_sigma2_div=args.dog_sigma2_div,
                            dog_norm_std=args.dog_norm_std,
                        )
                        (g_rp,) = torch.autograd.grad(
                            l_repa, feats[ell], retain_graph=True
                        )
                        # Per-TOKEN cosine over the channel axis, not a single
                        # cosine over the whole flattened activation: in ~12M-dim
                        # activation space any two gradients are near-orthogonal by
                        # construction (cos noise floor ~1/√dim), washing the global
                        # cosine out to 0 for every layer. The interaction lives at
                        # the spatial-token granularity — does FM and REPA want to
                        # move each location's D-dim feature the same way — so we
                        # cosine per token (D≈3072) and aggregate. The aggregate is
                        # **weighted by the FM per-token grad norm**: the tokens the
                        # flow-matching loss actually stresses are the ones whose
                        # redundancy/conflict matters (the Fisher-relevant weighting).
                        d_dim = g_fm[idx].shape[-1]
                        a = g_fm[idx].detach().float().reshape(-1, d_dim)  # (N,D)
                        b = g_rp.detach().float().reshape(-1, d_dim)
                        na_t = a.norm(dim=-1)  # (N,) FM per-token grad norm
                        nb_t = b.norm(dim=-1)
                        valid = (na_t > 1e-12) & (nb_t > 1e-12)
                        if not bool(valid.any()):
                            continue
                        cos_t = (a * b).sum(-1) / (na_t * nb_t + 1e-20)  # (N,)
                        w = na_t[valid]  # FM-stress weight
                        wsum = float(w.sum())
                        cos_w = float((w * cos_t[valid]).sum() / (wsum + 1e-20))
                        cos2_w = float((w * cos_t[valid] ** 2).sum() / (wsum + 1e-20))
                        # global magnitude ratio ‖g_repa‖/‖g_fm‖ at this tap + the
                        # raw norms behind it.
                        n_fm = float(a.norm())
                        n_rp = float(b.norm())
                        mag = n_rp / (n_fm + 1e-20)
                        acc_cos[ell].append(cos_w)
                        acc_cos2[ell].append(cos2_w)
                        acc_mag[ell].append(mag)
                        acc_nfm[ell].append(n_fm)
                        acc_nrp[ell].append(n_rp)
                # free the graph before the next σ/seed
                del velocity, feats, feats_list, g_fm
            for ell in layers:
                if acc_cos[ell]:
                    cos_d[ell][s].append(float(np.mean(acc_cos[ell])))
                    cos2_d[ell][s].append(float(np.mean(acc_cos2[ell])))
                    mag_d[ell][s].append(float(np.mean(acc_mag[ell])))
                    nfm_d[ell][s].append(float(np.mean(acc_nfm[ell])))
                    nrp_d[ell][s].append(float(np.mean(acc_nrp[ell])))

        if (ai + 1) % 5 == 0 or ai + 1 == len(kept):
            log.info(f"  [{ai + 1}/{len(kept)}] scored")
        if device.type == "cuda" and (ai + 1) % 10 == 0:
            torch.cuda.empty_cache()

    n_scored = len(cos_d[layers[0]][sigma_grid[0]])
    if n_scored == 0:
        raise SystemExit("no images scored — check caches / grad path")

    # ── aggregate to layer×σ means ────────────────────────────────────────────
    L_idx = layers
    S_idx = sigma_grid

    def _grid(d):
        return np.array(
            [
                [float(np.mean(d[ell][s])) if d[ell][s] else np.nan for s in S_idx]
                for ell in L_idx
            ]
        )

    cos_g = _grid(cos_d)
    cos2_g = _grid(cos2_d)
    mag_g = _grid(mag_d)
    nfm_g = _grid(nfm_d)
    nrp_g = _grid(nrp_d)

    band_cols = [S_idx.index(s) for s in band]
    cos_band = np.nanmean(cos_g[:, band_cols], axis=1)
    cos2_band = np.nanmean(cos2_g[:, band_cols], axis=1)  # redundancy
    mag_band = np.nanmean(mag_g[:, band_cols], axis=1)
    nfm_band = np.nanmean(nfm_g[:, band_cols], axis=1)
    nrp_band = np.nanmean(nrp_g[:, band_cols], axis=1)

    def _row(ell):
        return L_idx.index(ell)

    # What the data actually shows (see smoke runs): directionally REPA is
    # near-orthogonal to FM at EVERY depth (|cos| ≲ noise floor), so the
    # "deep = directionally redundant" hypothesis is NOT the discriminator. The
    # depth-dependent signal is the **orthogonal-push ratio** mag = ‖g_repa‖/‖g_fm‖:
    # a large orthogonal push to a near-output representation (which feeds the
    # velocity head with few layers left to re-absorb it) is negative transfer
    # even without a sign flip. So the negative-transfer RISK index is mag itself,
    # and the favorable layer is the one with the SMALLEST orthogonal push that is
    # still non-conflicting (cos ≥ −conflict_eps).
    ortho = float(np.nanmax(np.abs(cos_band)))  # how orthogonal everything is
    nonconf = [i for i, _ in enumerate(L_idx) if cos_band[i] >= -args.conflict_eps]
    pool = nonconf if nonconf else list(range(len(L_idx)))
    best_row = pool[int(np.argmin(mag_band[pool]))]
    l_star = L_idx[best_row]
    early = [i for i in pool if L_idx[i] <= args.repr_regime_max]
    l_star_early = L_idx[early[int(np.argmin(mag_band[early]))]] if early else None

    has8 = SHIPPED_LAYER in L_idx
    has26 = DEEP_LAYER in L_idx
    r8, r26 = (
        (_row(SHIPPED_LAYER) if has8 else None),
        (_row(DEEP_LAYER) if has26 else None),
    )

    # Head-to-head: the orthogonal-push ratio at 8 vs 26. Lower = gentler = safer.
    if has8 and has26:
        favored = SHIPPED_LAYER if mag_band[r8] < mag_band[r26] else DEEP_LAYER
        ratio = mag_band[r26] / (mag_band[r8] + 1e-12)
        if favored == SHIPPED_LAYER:
            h2h = (
                f"GRADIENT IS CONSISTENT WITH SHIPPED 8 over deep {DEEP_LAYER}: both "
                f"layers are ~orthogonal to FM (cos {cos_band[r8]:+.3f} vs "
                f"{cos_band[r26]:+.3f}), and the orthogonal-push ratio is {ratio:.1f}× "
                f"larger at deep {DEEP_LAYER} (mag {mag_band[r26]:.3f} vs "
                f"{mag_band[r8]:.3f}). CAVEAT: that ratio is driven by ‖g_fm‖ "
                f"collapsing toward the head (‖g_fm‖ {nfm_band[r8]:.1e}→"
                f"{nfm_band[r26]:.1e}), i.e. FM barely constrains the deep output, "
                "so REPA's push lands relatively unopposed there — a plausible "
                "negative-transfer mechanism (the A/B found 8 won), but the metric "
                "is activation-scale-confounded. Treat as suggestive; the decisive, "
                "unconfounded test is param-space interference / held-out FM loss on "
                "the trained adapters (compare_repa_ckpts.py)."
            )
        else:
            h2h = (
                f"GRADIENT FAVORS DEEP {DEEP_LAYER} over shipped 8 on push magnitude "
                f"(mag {mag_band[r26]:.3f} vs {mag_band[r8]:.3f}; cos "
                f"{cos_band[r26]:+.3f} vs {cos_band[r8]:+.3f}). This CONTRADICTS the "
                "training A/B (8 won) — the magnitude proxy does not explain the "
                "outcome here; treat with care."
            )
    else:
        h2h = "head-to-head needs both layer 8 and 26 tapped (use default --layers)."

    # ── artifacts ──────────────────────────────────────────────────────────────
    run_dir = make_run_dir("repa", label=args.label or "layer-grad-conflict")
    np.savez(
        run_dir / "grad_conflict.npz",
        layers=np.array(L_idx),
        sigmas=np.array(S_idx),
        cos=cos_g,
        cos2=cos2_g,
        mag=mag_g,
        g_fm_norm=nfm_g,
        g_repa_norm=nrp_g,
        cos_band=cos_band,
        redundancy_band=cos2_band,
        mag_band=mag_band,
        g_fm_norm_band=nfm_band,
        g_repa_norm_band=nrp_band,
        l_star=l_star,
        shipped_layer=SHIPPED_LAYER,
        deep_layer=DEEP_LAYER,
    )

    with (run_dir / "grad_conflict_by_layer.csv").open("w") as f:
        f.write("layer,cos_band,redundancy_band,mag_band,g_fm_norm,g_repa_norm\n")
        for i, ell in enumerate(L_idx):
            f.write(
                f"{ell},{cos_band[i]:.4f},{cos2_band[i]:.4f},{mag_band[i]:.4f},"
                f"{nfm_band[i]:.4e},{nrp_band[i]:.4e}\n"
            )

    M = ["# REPA layer × σ probe — gradient interaction with the flow-matching loss\n"]
    M.append(
        f"- images: **{n_scored}** ({n_skipped} skipped) · {n_seeds} noise draw(s) "
        f"· base DiT, no adapter · encoder={args.encoder}\n"
        "- ruler: at each block output hₗ, **FM-stress-weighted per-token** "
        "cos(∂L_fm/∂hₗ, ∂L_repa/∂hₗ). **mag = ‖g_repa‖/‖g_fm‖** is the "
        "orthogonal-push ratio (REPA's perturbation relative to FM's).\n"
        f"- band σ∈[{args.band_lo}, {args.band_hi}]"
        f"{' · target=DoG (σ1=min/' + format(args.dog_sigma1_div, 'g') + ')' if args.target_dog else ''}"
        f"{' · spatial_norm' if args.spatial_norm and not args.target_dog else ''}\n"
    )
    M.append("\n## Verdicts\n")
    M.append(
        f"- **ORTHOGONAL EVERYWHERE:** max |cos| over the band = {ortho:.3f} — REPA's "
        "gradient is ~perpendicular to FM's at every depth, so the "
        "'deep-layer = directionally-redundant' reading of the CKA peak does **not** "
        "hold at the gradient level. The discriminator is push *magnitude*, not "
        "direction."
    )
    M.append(f"- **{h2h}**")
    M.append(
        f"- **GENTLEST NON-CONFLICTING l\\*={l_star}** (smallest orthogonal push): "
        f"mag {mag_band[best_row]:.3f}, cos {cos_band[best_row]:+.3f}"
        + (
            f" · repr-regime (≤{args.repr_regime_max}) l\\*={l_star_early}"
            if l_star_early is not None
            else ""
        )
        + "."
    )

    M.append("\n## Band-averaged by layer (sorted shallow→deep)\n")
    M.append(
        "| layer | cos (signed) | redundancy (cos²) | mag ‖g_repa‖/‖g_fm‖ | "
        "‖g_fm‖ | ‖g_repa‖ |"
    )
    M.append("|---|---|---|---|---|---|")
    for i, ell in enumerate(L_idx):
        tag = (
            " ⟵ l*"
            if ell == l_star
            else " (shipped)"
            if ell == SHIPPED_LAYER
            else f" (deep {DEEP_LAYER})"
            if ell == DEEP_LAYER
            else ""
        )
        M.append(
            f"| {ell}{tag} | {cos_band[i]:+.3f} | {cos2_band[i]:.3f} | "
            f"{mag_band[i]:.3f} | {nfm_band[i]:.2e} | {nrp_band[i]:.2e} |"
        )

    M.append("\n## mag (orthogonal-push ratio) by layer × σ (**bold** = band)\n")
    header = "| layer | " + " | ".join(f"σ={s:g}" for s in S_idx) + " |"
    M.append(header)
    M.append("|---|" + "---|" * len(S_idx))
    for i, ell in enumerate(L_idx):
        cells = []
        for j in range(len(S_idx)):
            v = mag_g[i, j]
            txt = "—" if np.isnan(v) else f"{v:.2f}"
            cells.append(f"**{txt}**" if j in band_cols else txt)
        star = " ⟵ l*" if ell == l_star else ""
        M.append(f"| {ell}{star} | " + " | ".join(cells) + " |")

    M.append("\n## Reading it\n")
    M.append(
        "- **cos ≈ 0 at all depths** ⇒ REPA injects a direction the FM loss neither "
        "drives nor opposes — it is non-redundant *directionally* everywhere. So a "
        "high static CKA at deep layers is **not** the same as gradient redundancy.\n"
        "- The damage is **magnitude × depth**: a large orthogonal push (high mag) "
        "to a near-output representation drags it sideways off the FM manifold with "
        "few layers left to re-absorb the perturbation → negative transfer. The "
        "‖g_fm‖ / ‖g_repa‖ columns show which side drives the mag trend.\n"
        "- Local linearization at the base model: ranks layers by instantaneous "
        "interaction, not full-trajectory outcome. The training A/B is ground "
        "truth; this predicts which A/B to run.\n"
    )
    (run_dir / "summary.md").write_text("\n".join(M) + "\n", encoding="utf-8")

    metrics = {
        "n_images": n_scored,
        "n_skipped": n_skipped,
        "num_seeds": n_seeds,
        "encoder": args.encoder,
        "layers": L_idx,
        "sigma_grid": S_idx,
        "band": band,
        "target_dog": bool(args.target_dog),
        "spatial_norm": bool(args.spatial_norm),
        "shipped_layer": SHIPPED_LAYER,
        "deep_layer": DEEP_LAYER,
        "l_star": int(l_star),
        "l_star_early": (int(l_star_early) if l_star_early is not None else None),
        "max_abs_cos_band": float(ortho),
        "cos_band": [float(x) for x in cos_band],
        "redundancy_band": [float(x) for x in cos2_band],
        "mag_band": [float(x) for x in mag_band],
        "g_fm_norm_band": [float(x) for x in nfm_band],
        "g_repa_norm_band": [float(x) for x in nrp_band],
        "cos_shipped": (float(cos_band[r8]) if has8 else None),
        "mag_shipped": (float(mag_band[r8]) if has8 else None),
        "cos_deep": (float(cos_band[r26]) if has26 else None),
        "mag_deep": (float(mag_band[r26]) if has26 else None),
        "head_to_head": h2h,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["summary.md", "grad_conflict_by_layer.csv", "grad_conflict.npz"],
        device=device,
    )

    log.info("\n" + "=" * 72)
    log.info(f"  REPA layer grad-conflict probe → {run_dir}")
    log.info(f"  orthogonal everywhere: max|cos| over band = {ortho:.3f}")
    if has8 and has26:
        log.info(
            f"  shipped  8: cos {cos_band[r8]:+.3f}  mag {mag_band[r8]:.3f}  "
            f"(‖g_fm‖ {nfm_band[r8]:.2e}, ‖g_repa‖ {nrp_band[r8]:.2e})"
        )
        log.info(
            f"  deep    {DEEP_LAYER}: cos {cos_band[r26]:+.3f}  mag {mag_band[r26]:.3f}  "
            f"(‖g_fm‖ {nfm_band[r26]:.2e}, ‖g_repa‖ {nrp_band[r26]:.2e})"
        )
    log.info(
        f"  l* (gentlest): {l_star}  |  l* (repr ≤{args.repr_regime_max}): {l_star_early}"
    )
    log.info(f"  {h2h}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
