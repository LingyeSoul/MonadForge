#!/usr/bin/env python3
"""Compare two trained REPA adapters head-to-head — the decisive, unconfounded
read on the layer-8 vs layer-26 A/B that the base-model probes only predict.

THE PAIR. ``anima_repa4_tenth`` (repa_layer **8**) and ``anima_repa5_tenth``
(repa_layer **26**) are identical runs (same data, dim=16/α=16, 3 epochs,
relational + target_dog) differing *only* in which block the alignment loss
anchors. The CKA probe (``probe_layer_sigma_cka.py``) ranks layer 26 higher; the
training A/B and the gradient probe (``probe_layer_grad_conflict.py``) say 8 is
better. This tool measures the *trained artifacts themselves* on held-out data,
so the verdict needs no proxy.

THREE RULERS:

  1. **Held-out denoising (flow-matching) loss** — the quantity that actually
     matters. For each model (base, repa4, repa5) over the probe set × σ grid,
     ``MSE(v_pred, ε−x0)``. The adapter that denoises real held-out latents
     better at the production σ band IS the better adapter — no confound, the
     direct objective. (FM-val doesn't always track sample quality on Anima — see
     [[project_fm_val_loss_uninformative]] — but a *paired* same-data delta
     between two adapters is a far cleaner contrast than absolute val, and it's
     the loss training optimized.)

  2. **Per-layer CKA-to-PE delta vs base** — did REPA at layer ℓ actually inject
     the alignment it was supposed to, and did anchoring it deep perturb other
     layers? Centered linear CKA of each block's pooled tokens against the image's
     PE-Spatial target, adapter minus base. Closes the loop with the CKA probe:
     it predicted *where alignment is high*; this shows *what training did to it*.

  3. **ΔW structural fingerprint** — per-block Frobenius norm + stable rank of the
     baked LoRA update ``ΔW = (α/r)·U@V`` (from the safetensors directly, no GPU).
     Where does each adapter write, and how concentrated is the update? Anchoring
     the aux loss at a different depth can move the whole adapter's mass.

Run from anima_lora/::

    uv run python bench/repa/compare_repa_ckpts.py \
        --adapters output/ckpt/anima_repa4_tenth.safetensors \
                   output/ckpt/anima_repa5_tenth.safetensors \
        --labels layer8 layer26 --num_samples 64
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402

# Reuse the CKA probe's sidecar resolver so the pair discovery matches.
from bench.repa.probe_layer_sigma_cka import _pe_sidecar  # noqa: E402
from bench.soft_tokens_contrastive.reward_premise_probe import (  # noqa: E402
    EmbCache,
    discover_pairs,
)
from library.io.cache import load_cached_latents  # noqa: E402
from library.training.repa import (  # noqa: E402
    dog_standardize,
    pool_dit_tokens_to_grid,
    resolve_pe_grid,
)
from library.vision.buckets import get_bucket_spec  # noqa: E402

log = logging.getLogger("bench.repa.compare_ckpts")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATA = "post_image_dataset/lora"
DEFAULT_SIGMAS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
DEFAULT_ADAPTERS = [
    "output/ckpt/anima_repa4_tenth.safetensors",
    "output/ckpt/anima_repa5_tenth.safetensors",
]
DEFAULT_LABELS = ["layer8", "layer26"]
# The trained anchor of each default adapter (for the "did it inject alignment at
# its own layer" readout). Pulled from the snapshots.
DEFAULT_TRAINED_LAYER = {"layer8": 8, "layer26": 26}

_BLK = re.compile(r"blocks_(\d+)_")


# ───────────────────────────── ΔW structural analysis ──────────────────────────
def delta_w_by_block(adapter_path: str) -> dict[int, dict[str, float]]:
    """Per-block ΔW stats from the LoRA safetensors (no model build).

    ``ΔW = (alpha/rank)·(up @ down)`` per module; we aggregate Frobenius energy
    and the (norm-weighted mean) stable rank ``‖ΔW‖_F²/‖ΔW‖_2²`` over the modules
    of each DiT block. Stable rank needs only the top singular value
    (``matrix_norm(ord=2)``), so no full SVD.
    """
    sd = load_file(adapter_path)
    downs = {
        k[: -len(".lora_down.weight")]: v
        for k, v in sd.items()
        if k.endswith(".lora_down.weight")
    }
    out: dict[int, dict[str, float]] = {}
    for base, down in downs.items():
        up = sd.get(base + ".lora_up.weight")
        if up is None:
            continue
        m = _BLK.search(base)
        if m is None:
            continue
        blk = int(m.group(1))
        rank = down.shape[0]
        alpha = sd.get(base + ".alpha")
        scale = (float(alpha) / rank) if alpha is not None else 1.0
        dw = (up.float() @ down.float()) * scale  # (out, in)
        fro = float(dw.norm())
        spec = float(torch.linalg.matrix_norm(dw, ord=2))
        srank = (fro * fro) / (spec * spec + 1e-12)
        acc = out.setdefault(blk, {"fro2": 0.0, "srank_w": 0.0, "wsum": 0.0, "n": 0})
        acc["fro2"] += fro * fro
        acc["srank_w"] += srank * fro  # norm-weight the stable rank
        acc["wsum"] += fro
        acc["n"] += 1
    for blk, acc in out.items():
        acc["fro"] = float(np.sqrt(acc["fro2"]))
        acc["srank"] = acc["srank_w"] / (acc["wsum"] + 1e-12)
    return out


def _layer_metrics(dit_stack, pe_raw, pe_dog, eps=1e-8):
    """Vectorized across all L tapped layers in a handful of batched matmuls.

    The per-layer Python loop over ``linear_cka`` launched 28 tiny Gram ops per
    (image, σ) → GPU sat at ~2% util, wall-clock dominated by launch overhead.
    Stacking the layers and using ``bmm`` does the whole sweep GPU-resident.

    ``dit_stack`` ``(L, N, D)`` pooled DiT tokens (all layers share the encoder
    grid N); ``pe_raw`` ``(N, Dr)`` the raw PE target (for centered CKA — the
    confounded matched ruler the CKA probe reports); ``pe_dog`` ``(N, Dd)`` the
    DoG-band-passed target (for the relational Gram **align loss** — the exact
    quantity training optimized). Returns ``(cka[L], align[L])`` as numpy.
    """
    # Centered linear CKA (dual/Gram form) vs the raw PE target.
    xc = dit_stack - dit_stack.mean(dim=1, keepdim=True)
    k = torch.bmm(xc, xc.transpose(1, 2))  # (L, N, N)
    yc = pe_raw - pe_raw.mean(dim=0, keepdim=True)
    lpe = yc @ yc.transpose(0, 1)  # (N, N)
    hsic = (k * lpe.unsqueeze(0)).sum(dim=(1, 2))  # (L,)
    cka = hsic / (k.norm(dim=(1, 2)) * lpe.norm() + eps)
    del xc, k, lpe, yc
    # Relational Gram MSE vs the DoG target = the trained REPA loss (per layer).
    dit_hat = F.normalize(dit_stack, dim=-1)
    g_dit = torch.bmm(dit_hat, dit_hat.transpose(1, 2))  # (L, N, N)
    pe_hat = F.normalize(pe_dog, dim=-1)
    g_pe = pe_hat @ pe_hat.transpose(0, 1)  # (N, N)
    align = ((g_dit - g_pe.unsqueeze(0)) ** 2).mean(dim=(1, 2))  # (L,)
    del dit_hat, g_dit
    return cka.detach().cpu().numpy(), align.detach().cpu().numpy()


@torch.no_grad()
def eval_model(
    anima, store, kept, layers, sigma_grid, patch, spec, device, dtype, seed
):
    """One pass over the probe set → FM loss by σ, and matched-CKA + DoG-Gram
    align loss by (layer, σ), both vectorized across layers."""
    layer_set = set(layers)
    fm = {s: [] for s in sigma_grid}
    cka = {ell: {s: [] for s in sigma_grid} for ell in layers}
    align = {ell: {s: [] for s in sigma_grid} for ell in layers}
    for ai, stem in enumerate(kept):
        rec = store[stem]
        H, W = rec["hw"]
        gh, gw = rec["grid"]
        x0 = rec["x0"].to(device, dtype)
        x0_f = x0.float()
        emb_b = rec["emb"].to(device, dtype)
        pe_raw = rec["pe_tok"].to(device).float()[0]  # (N, Dr)
        pe_dog = rec["pe_dog"].to(device).float()[0]  # (N, Dd)
        pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)
        for s in sigma_grid:
            g = torch.Generator(device=device).manual_seed(seed + ai * 1000)
            eps = torch.randn(x0.shape, generator=g, device=device, dtype=dtype)
            noisy = ((1.0 - s) * x0_f + s * eps.float()).to(dtype)
            t_b = torch.full((1,), float(s), device=device, dtype=dtype)
            target = (eps - x0).float()
            velocity, feats = anima.forward_mini_train_dit(
                noisy,
                t_b,
                emb_b,
                padding_mask=pad,
                skip_pooled_text_proj=True,
                return_block_features=layer_set,
            )
            fm[s].append(float(F.mse_loss(velocity.float(), target)))
            # Pool every tapped layer to the encoder grid, stack → (L, N, D).
            dit_stack = torch.stack(
                [
                    pool_dit_tokens_to_grid(feats[ell], (H, W), patch, gh, gw)[0]
                    for ell in layers
                ]
            )
            cka_l, align_l = _layer_metrics(dit_stack, pe_raw, pe_dog)
            for li, ell in enumerate(layers):
                cka[ell][s].append(float(cka_l[li]))
                align[ell][s].append(float(align_l[li]))
            del dit_stack, feats, velocity
        if device.type == "cuda" and (ai + 1) % 20 == 0:
            torch.cuda.empty_cache()
    return fm, cka, align


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap, vae=False, text_encoder=False)
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--adapters", nargs="+", default=DEFAULT_ADAPTERS)
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="block indices to tap for CKA (default: all DiT blocks).",
    )
    ap.add_argument("--encoder", default="pe_spatial")
    ap.add_argument("--band_lo", type=float, default=0.45)
    ap.add_argument("--band_hi", type=float, default=0.90)
    # DoG target band-pass for the align-loss column — defaults match what the
    # repa4/repa5 runs trained with (repa_target_dog, σ1=min/16, high-pass only).
    ap.add_argument("--dog_sigma1_div", type=float, default=16.0)
    ap.add_argument("--dog_sigma2_div", type=float, default=0.0)
    ap.add_argument("--dog_norm_std", type=float, default=0.0)
    add_common_args(ap)
    args = ap.parse_args()

    if len(args.labels) != len(args.adapters):
        args.labels = [Path(a).stem for a in args.adapters]
    if getattr(args, "compile", False):
        log.warning("comparison runs eager; ignoring --compile.")
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
        raise SystemExit(f"no {args.encoder} sidecars under {args.data_dir}")
    rng = np.random.default_rng(args.seed)
    take = min(args.num_samples, len(pool_list))
    stems = [pool_list[int(i)] for i in rng.choice(len(pool_list), take, replace=False)]

    embs = EmbCache(pairs)
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
        pe = pe.float().unsqueeze(0)
        n_pe = pe.shape[1] - (1 if spec.use_cls else 0)
        lat, _res, _oh, _ow = load_cached_latents(npz_path)
        x0 = lat.unsqueeze(0).unsqueeze(2)
        H, W = x0.shape[-2], x0.shape[-1]
        gh, gw = resolve_pe_grid(spec, n_pe, H, W)
        # CKA matched target = the image's own PE (no DoG — CKA centers anyway, and
        # we want the same ruler the CKA probe's matched column reported).
        pe_tok = (pe[:, 1:, :] if spec.use_cls else pe).contiguous()
        # DoG target = the exact band-passed target the relational REPA loss
        # trained against (filtered once here, as the adapter does each step).
        pe_dog = dog_standardize(
            pe_tok, gh, gw, args.dog_sigma1_div, args.dog_sigma2_div, args.dog_norm_std
        ).contiguous()
        store[stem] = {
            "x0": x0,
            "emb": emb.unsqueeze(0),
            "pe_tok": pe_tok,
            "pe_dog": pe_dog,
            "grid": (gh, gw),
            "hw": (H, W),
        }
    kept = [s for s in stems if s in store]
    if not kept:
        raise SystemExit("no images scored — check caches")
    log.info(f"probe set: {len(kept)} images ({n_skipped} skipped)")

    # ── functional eval: base + each adapter ───────────────────────────────────
    results: dict[str, dict] = {}
    model_order = ["base"] + list(args.labels)
    adapter_of = {lab: path for lab, path in zip(args.labels, args.adapters)}

    for name in model_order:
        adapter = None if name == "base" else adapter_of[name]
        log.info(f"\n=== building model: {name} ===")
        bundle = build_anima(args, adapter=adapter, train_mode=False)
        anima = bundle.anima
        device, dtype = bundle.device, bundle.dtype
        patch = int(anima.patch_spatial)
        n_blocks = len(anima.blocks)
        layers = sorted(args.layers) if args.layers else list(range(n_blocks))
        fm, cka, align = eval_model(
            anima,
            store,
            kept,
            layers,
            sigma_grid,
            patch,
            spec,
            device,
            dtype,
            args.seed,
        )
        results[name] = {"fm": fm, "cka": cka, "align": align, "layers": layers}
        del anima, bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    layers = results["base"]["layers"]
    band_cols = [sigma_grid.index(s) for s in band]

    def fm_band(name):
        arr = np.array([float(np.mean(results[name]["fm"][s])) for s in sigma_grid])
        return arr, float(arr[band_cols].mean())

    def _layer_band(name, key):
        grid = np.array(
            [
                [float(np.mean(results[name][key][ell][s])) for s in sigma_grid]
                for ell in layers
            ]
        )
        return grid, grid[:, band_cols].mean(axis=1)

    fm_arrs = {n: fm_band(n) for n in model_order}
    cka_grids, cka_band = {}, {}
    align_grids, align_band = {}, {}
    for n in model_order:
        cka_grids[n], cka_band[n] = _layer_band(n, "cka")
        align_grids[n], align_band[n] = _layer_band(n, "align")

    # ── ΔW structural ──────────────────────────────────────────────────────────
    dw = {lab: delta_w_by_block(adapter_of[lab]) for lab in args.labels}

    # ── verdict ────────────────────────────────────────────────────────────────
    fm_rank = sorted(args.labels, key=lambda n: fm_arrs[n][1])
    base_band = fm_arrs["base"][1]
    best, worst = fm_rank[0], fm_rank[-1]
    fm_gap = fm_arrs[worst][1] - fm_arrs[best][1]
    fm_verdict = (
        f"HELD-OUT FM (band σ∈[{args.band_lo},{args.band_hi}]): "
        + ", ".join(f"{n} {fm_arrs[n][1]:.4f}" for n in args.labels)
        + f" (base {base_band:.4f}). **{best} denoises held-out latents better** "
        f"(Δ {fm_gap:+.4f} vs {worst})."
    )

    # Did each adapter inject its intended alignment at its own trained layer?
    # The DoG-Gram align loss is the quantity training MINIMIZED, so a NEGATIVE Δ
    # vs base = REPA did its job (raw CKA is the confounded cross-check).
    align_lines = []
    for lab in args.labels:
        tl = DEFAULT_TRAINED_LAYER.get(lab)
        if tl is None or tl not in layers:
            continue
        ri = layers.index(tl)
        dc = cka_band[lab][ri] - cka_band["base"][ri]
        da = align_band[lab][ri] - align_band["base"][ri]
        verdict = "injected" if da < 0 else "did NOT reduce"
        align_lines.append(
            f"{lab} @ trained layer {tl}: DoG-Gram align loss {align_band[lab][ri]:.4f} "
            f"(base {align_band['base'][ri]:.4f}, Δ {da:+.4f} → {verdict}); "
            f"raw CKA Δ {dc:+.3f}"
        )

    # ── artifacts ──────────────────────────────────────────────────────────────
    run_dir = make_run_dir("repa", label=args.label or "compare-ckpts")
    np.savez(
        run_dir / "compare.npz",
        layers=np.array(layers),
        sigmas=np.array(sigma_grid),
        **{f"fm_{n}": fm_arrs[n][0] for n in model_order},
        **{f"cka_{n}": cka_grids[n] for n in model_order},
        **{f"align_{n}": align_grids[n] for n in model_order},
    )

    M = ["# REPA adapter comparison — layer-8 vs layer-26, head-to-head\n"]
    M.append(
        f"- probe set: **{len(kept)}** held-out images ({n_skipped} skipped) · "
        f"encoder={args.encoder} · band σ∈[{args.band_lo}, {args.band_hi}]\n"
        f"- models: base + {', '.join(args.labels)}\n"
    )
    M.append("\n## Verdicts\n")
    M.append(f"- **{fm_verdict}**")
    for line in align_lines:
        M.append(f"- {line}")

    M.append("\n## Held-out flow-matching loss by σ (lower = better denoiser)\n")
    M.append("| σ | " + " | ".join(model_order) + " |")
    M.append("|---|" + "---|" * len(model_order))
    for j, s in enumerate(sigma_grid):
        cells = []
        vals = {n: fm_arrs[n][0][j] for n in model_order}
        best_n = min(args.labels, key=lambda n: vals[n])
        for n in model_order:
            v = f"{vals[n]:.4f}"
            cells.append(f"**{v}**" if n == best_n else v)
        tag = " ⟵ band" if j in band_cols else ""
        M.append(f"| {s:g}{tag} | " + " | ".join(cells) + " |")
    M.append(
        "| **band** | "
        + " | ".join(f"**{fm_arrs[n][1]:.4f}**" for n in model_order)
        + " |"
    )

    M.append(
        "\n## DoG-Gram align loss by layer (band-avg) — Δ vs base in () "
        "[**the trained objective**; lower = more aligned, Δ<0 = REPA worked]\n"
    )
    M.append("| layer | base | " + " | ".join(f"{n} (Δ)" for n in args.labels) + " |")
    M.append("|---|---|" + "---|" * len(args.labels))
    for ri, ell in enumerate(layers):
        cells = [f"{align_band['base'][ri]:.4f}"]
        for lab in args.labels:
            d = align_band[lab][ri] - align_band["base"][ri]
            mark = " ⟸trained" if DEFAULT_TRAINED_LAYER.get(lab) == ell else ""
            cells.append(f"{align_band[lab][ri]:.4f} ({d:+.4f}){mark}")
        M.append(f"| {ell} | " + " | ".join(cells) + " |")

    M.append(
        "\n## CKA-to-PE by layer (band-avg) — Δ vs base in () "
        "[raw/confounded cross-check]\n"
    )
    M.append("| layer | base | " + " | ".join(f"{n} (Δ)" for n in args.labels) + " |")
    M.append("|---|---|" + "---|" * len(args.labels))
    for ri, ell in enumerate(layers):
        cells = [f"{cka_band['base'][ri]:.3f}"]
        for lab in args.labels:
            d = cka_band[lab][ri] - cka_band["base"][ri]
            mark = ""
            if DEFAULT_TRAINED_LAYER.get(lab) == ell:
                mark = " ⟸trained"
            cells.append(f"{cka_band[lab][ri]:.3f} ({d:+.3f}){mark}")
        M.append(f"| {ell} | " + " | ".join(cells) + " |")

    M.append("\n## ΔW per-block — Frobenius norm | stable rank\n")
    M.append(
        "| block | " + " | ".join(f"{lab} ‖ΔW‖ | srank" for lab in args.labels) + " |"
    )
    M.append("|---|" + "---|---|" * len(args.labels))
    all_blocks = sorted({b for lab in args.labels for b in dw[lab]})
    for blk in all_blocks:
        cells = []
        for lab in args.labels:
            a = dw[lab].get(blk)
            if a is None:
                cells += ["—", "—"]
            else:
                cells += [f"{a['fro']:.3f}", f"{a['srank']:.1f}"]
        tl_marks = [lab for lab in args.labels if DEFAULT_TRAINED_LAYER.get(lab) == blk]
        tag = f" ⟸ {','.join(tl_marks)} anchor" if tl_marks else ""
        M.append(f"| {blk}{tag} | " + " | ".join(cells) + " |")

    M.append("\n## Reading it\n")
    M.append(
        "- **Held-out FM loss is the decisive, unconfounded ruler** — the adapter "
        "that denoises real held-out latents better at the production σ band is the "
        "better adapter, full stop. This is the training objective, measured "
        "out-of-sample.\n"
        "- **DoG-Gram align loss is the trained objective** — Δ<0 at the trained "
        "layer means REPA actually injected its intended alignment. An adapter can "
        "drive its align loss down yet lose on FM: alignment bought at the cost of "
        "the objective (the A/B's lesson). Raw CKA is the confounded cross-check "
        "(it includes the low band DoG strips, so it can move the other way).\n"
        "- **ΔW mass shifting toward the anchor block** shows the aux loss "
        "re-routing where the adapter spends capacity.\n"
    )
    (run_dir / "summary.md").write_text("\n".join(M) + "\n", encoding="utf-8")

    metrics = {
        "n_images": len(kept),
        "n_skipped": n_skipped,
        "encoder": args.encoder,
        "labels": list(args.labels),
        "adapters": list(args.adapters),
        "layers": layers,
        "sigma_grid": sigma_grid,
        "band": band,
        "fm_band": {n: fm_arrs[n][1] for n in model_order},
        "fm_by_sigma": {n: [float(x) for x in fm_arrs[n][0]] for n in model_order},
        "fm_rank_best_to_worst": fm_rank,
        "cka_band": {n: [float(x) for x in cka_band[n]] for n in model_order},
        "align_band": {n: [float(x) for x in align_band[n]] for n in model_order},
        "trained_layer": {lab: DEFAULT_TRAINED_LAYER.get(lab) for lab in args.labels},
        "dog_sigma1_div": args.dog_sigma1_div,
        "fm_verdict": fm_verdict,
        "align_lines": align_lines,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["summary.md", "compare.npz"],
        device=torch.device(getattr(args, "device", "cuda")),
    )

    log.info("\n" + "=" * 72)
    log.info(f"  REPA adapter comparison → {run_dir}")
    log.info(f"  {fm_verdict}")
    for line in align_lines:
        log.info(f"  {line}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
