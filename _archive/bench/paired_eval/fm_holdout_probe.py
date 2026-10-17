#!/usr/bin/env python3
"""Does HOLDOUT FM loss rank the plain-vs-base known-difference (where Gram failed)?

Cheap sanity check requested after PGS failed Phase 0 gate 3. FM loss was
demoted as a val signal (`project_fm_val_loss_uninformative`: lower val FM-MSE
≠ better samples; CMMD replaced it) — this is NOT a bid to re-adopt it, only to
see whether the PAIRED HOLDOUT FM-fit signal (a different quantity than the
pooled val FM-MSE that was demoted — it's the same-artist-holdout arm of
`loss_gap.py`) at least reproduces the CMMD-vindicated ranking that token-Gram
could not.

No rendering — forward passes over cached latents + TE embeddings only. Per
same-artist holdout image, re-noise its cached x0 with fixed per-stem probe
noise, query the velocity field under each model, score confidence
R = -log(MSE(eps_hat, eps) + delta). Easiness-cancel with delta_X = R_X - R_base
(paired per stem). Rank the 4 adapters by mean holdout delta; higher = fits the
artist's unseen work better.

CMMD-vindicated ranking on this render set (cmmd_holdout, lower = closer):
  plain 0.182 < reg36_reft64 0.223 < reft64 0.235 < base 0.279 < reg36 0.364.
So a tracking FM signal wants: plain highest delta, reg36 lowest (negative).

CAVEAT baked into the reading: lower holdout FM loss is ALSO the leading edge of
memorization (loss_gap's member gap). A high holdout delta says "fits this
artist", not "generates well" — treat as covariate, never as the quality gate.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._anima import build_anima  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.memorization.loss_gap import (  # noqa: E402
    _load_z0,
    _read_snapshot_knobs,
    build_items,
    confidence,
    setup_strategies,
)
from bench.memorization.probe import _Acc, build_training_args  # noqa: E402
from bench.reft import reft_network  # noqa: E402
from library.anima.strategy import AnimaLatentsCachingStrategy  # noqa: E402
from library.runtime.device import clean_memory_on_device, str_to_dtype  # noqa: E402
from library.training.validation import (  # noqa: E402
    _build_val_crossattn_emb,
    _load_safetensors,
)

CKPT = REPO_ROOT / "output" / "ckpt"
ADAPTERS = [
    "bench_sincos_half_plain",
    "bench_sincos_half_reft64",
    "bench_sincos_half_reg36",
    "bench_sincos_half_reg36_reft64",
]
# CMMD-vindicated holdout ranking (lower cmmd = closer to artist).
CMMD_HOLDOUT = {
    "base": 0.2793,
    "bench_sincos_half_plain": 0.1816,
    "bench_sincos_half_reft64": 0.2354,
    "bench_sincos_half_reg36": 0.3638,
    "bench_sincos_half_reg36_reft64": 0.2232,
}


@torch.no_grad()
def score_any(scored, kind, adapter, targs, shim, sigmas, *, n_probes, delta, dtype):
    """Per-item {sigma: confidence} under one model. ``kind`` in
    {'base','lora','reft'} — 'reft' loads a bare DiT then manually applies the
    out-of-tree bench network (build_anima's loader has no dispatch for it)."""
    if kind == "reft":
        bundle = build_anima(targs, adapter=None, train_mode=False)
        net = reft_network.create_network_from_weights(1.0, adapter, None, None, bundle.anima)
        net.to(device=bundle.device, dtype=dtype)
        net.apply_to(None, bundle.anima, apply_text_encoder=False, apply_unet=True)
        net.eval()
    else:
        bundle = build_anima(targs, adapter=adapter, train_mode=False)
        net = None
    dit, device = bundle.anima, bundle.device
    lat_strategy = AnimaLatentsCachingStrategy(True, 1, True)
    out = []
    try:
        if hasattr(dit, "prepare_block_swap_before_forward"):
            dit.prepare_block_swap_before_forward()
        for n, it in enumerate(scored):
            z0 = _load_z0(lat_strategy, it["lat_npz"], it["bucket_reso"], device, dtype)
            sd = _load_safetensors(it["te_npz"])
            emb = _build_val_crossattn_emb(dit, sd, shim)
            out.append(
                confidence(
                    dit, z0, emb, sigmas, n_probes=n_probes, delta=delta,
                    seed=zlib.crc32(it["stem"].encode()) & 0x7FFFFFFF,
                    device=device, dtype=dtype,
                )
            )
            del z0, emb, sd
            if (n + 1) % 24 == 0:
                print(f"    {n + 1}/{len(scored)} scored ({kind})", flush=True)
    finally:
        if net is not None:
            net.remove()
        del bundle, dit, net
        clean_memory_on_device(shim.device)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--method", default="lora")
    ap.add_argument("--preset", default="default")
    ap.add_argument("--membership_adapter", default="bench_sincos_half_plain")
    ap.add_argument("--num_holdout", type=int, default=64)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--n_probes", type=int, default=4)
    ap.add_argument("--delta", type=float, default=1e-3)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="fm_holdout")
    args = ap.parse_args()

    dtype = str_to_dtype(args.dtype)
    rng = np.random.default_rng(args.seed)
    memb = str(CKPT / f"{args.membership_adapter}.safetensors")

    snap = _read_snapshot_knobs(memb)
    print(f"snapshot knobs: {snap}", flush=True)
    targs = build_training_args(args.method, args.preset, {})
    for k in ("validation_seed", "validation_split", "validation_split_num"):
        if k in snap:
            setattr(targs, k, snap[k])
    # Scope the blueprint to the target artist so build_items only scans ~331
    # sincos images, not the whole 2972-image dataset (the subset reads
    # path_pattern off the namespace; loss_gap's build_items never set it).
    # We only need same-artist holdout here, so no cross-artist pool is lost.
    targs.path_pattern = snap.get("path_pattern")
    run_sr = snap.get("sample_ratio")
    run_shard = snap.get("artists_shard") or None

    from library.runtime.accelerator import prepare_accelerator  # noqa: PLC0415

    acc = prepare_accelerator(targs)
    shim = _Acc(acc.device, dtype)

    te_strategy = setup_strategies(targs)
    member_items = build_items(targs, te_strategy, sample_ratio=run_sr, artists_shard=run_shard)
    full_items = build_items(
        targs, te_strategy, sample_ratio=1.0, artists_shard=None, include_val=True
    )
    member_paths = {it[0].absolute_path for it in member_items}
    member_dirs = {Path(it[0].absolute_path).parent for it in member_items}
    same_holdout = [
        it
        for it in full_items
        if it[0].absolute_path not in member_paths
        and Path(it[0].absolute_path).parent in member_dirs
    ]
    print(f"members={len(member_items)} same-artist holdout={len(same_holdout)}", flush=True)

    if len(same_holdout) > args.num_holdout:
        idx = sorted(rng.choice(len(same_holdout), args.num_holdout, replace=False))
        same_holdout = [same_holdout[int(i)] for i in idx]
    scored = [
        {
            "group": "same_artist",
            "stem": Path(info.absolute_path).stem,
            "artist": Path(info.absolute_path).parent.name,
            "bucket_reso": tuple(info.bucket_reso),
            "te_npz": te_npz,
            "lat_npz": lat_npz,
        }
        for info, te_npz, lat_npz in same_holdout
    ]
    print(f"scoring {len(scored)} holdout items x {1 + len(ADAPTERS)} models", flush=True)

    def _kind(path):
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            mod = (f.metadata() or {}).get("ss_network_module", "")
        return "reft" if mod == reft_network.ReFTBenchNetwork.network_module else "lora"

    def run(kind, adapter):
        return score_any(
            scored, kind, adapter, targs, shim, args.sigmas,
            n_probes=args.n_probes, delta=args.delta, dtype=dtype,
        )

    print("base pass …", flush=True)
    base_scores = run("base", None)
    R_base = np.array([np.mean(list(c.values())) for c in base_scores])

    rows = {}
    per_model_R = {"base": float(R_base.mean())}
    for a in ADAPTERS:
        path = str(CKPT / f"{a}.safetensors")
        kind = _kind(path)
        print(f"{a} pass … (kind={kind})", flush=True)
        sc = run(kind, path)
        R = np.array([np.mean(list(c.values())) for c in sc])
        d = R - R_base  # paired, easiness-cancelled
        # per-sigma deltas for the σ≤0.7 check
        psig = {}
        for si, s in enumerate(args.sigmas):
            rb = np.array([base_scores[i][s] for i in range(len(scored))])
            ra = np.array([sc[i][s] for i in range(len(scored))])
            psig[f"{s:g}"] = float((ra - rb).mean())
        rows[a] = {
            "delta_mean": float(d.mean()),
            "delta_pos_frac": float((d > 0).mean()),
            "per_sigma": psig,
        }
        per_model_R[a] = float(R.mean())

    # ── Ranking comparison ────────────────────────────────────────────────
    fm_rank = sorted(ADAPTERS, key=lambda a: -rows[a]["delta_mean"])  # high delta first
    cmmd_rank = sorted(ADAPTERS, key=lambda a: CMMD_HOLDOUT[a])  # low cmmd first
    # Spearman over the 4 adapters (rank correlation).
    fm_order = {a: i for i, a in enumerate(fm_rank)}
    cmmd_order = {a: i for i, a in enumerate(cmmd_rank)}
    dsq = sum((fm_order[a] - cmmd_order[a]) ** 2 for a in ADAPTERS)
    n = len(ADAPTERS)
    spearman = 1 - 6 * dsq / (n * (n * n - 1))
    plain_wins_base = rows["bench_sincos_half_plain"]["delta_mean"] > 0
    plain_is_top = fm_rank[0] == "bench_sincos_half_plain"

    lines = [
        "# Holdout FM-loss ranking vs the CMMD-vindicated ranking",
        "",
        f"- same-artist holdout n={len(scored)} · sigmas {args.sigmas} · K={args.n_probes}",
        f"- delta = R_adapter − R_base (higher = fits artist's unseen work better)",
        "",
        "| model | holdout Δ (FM confidence gain) | Δ>0 frac | cmmd_holdout ↓ |",
        "|---|---|---|---|",
        f"| base | 0.0000 (ref) | — | {CMMD_HOLDOUT['base']:.4f} |",
    ]
    for a in ADAPTERS:
        lines.append(
            f"| {a} | {rows[a]['delta_mean']:+.4f} | "
            f"{rows[a]['delta_pos_frac']:.2f} | {CMMD_HOLDOUT[a]:.4f} |"
        )
    lines += [
        "",
        f"- FM ranking (best→worst): {' > '.join(a.replace('bench_sincos_half_','') for a in fm_rank)}",
        f"- CMMD ranking (best→worst): {' > '.join(a.replace('bench_sincos_half_','') for a in cmmd_rank)}",
        f"- **Spearman(FM, CMMD) over 4 adapters = {spearman:+.3f}**",
        f"- plain beats base on holdout FM: {plain_wins_base}; plain is FM-top: {plain_is_top}",
        "",
        "Per-sigma holdout Δ (memory: membership/fit signal lives σ≤0.7):",
        "",
        "| model | " + " | ".join(f"σ={s:g}" for s in args.sigmas) + " |",
        "|---|" + "---|" * len(args.sigmas),
    ]
    for a in ADAPTERS:
        lines.append(
            f"| {a.replace('bench_sincos_half_','')} | "
            + " | ".join(f"{rows[a]['per_sigma'][f'{s:g}']:+.4f}" for s in args.sigmas)
            + " |"
        )
    lines += [
        "",
        "Reading: Spearman≈+1 ⇒ holdout FM-fit reproduces the CMMD/eyeball "
        "ranking that token-Gram reversed — a valid generalization COVARIATE "
        "(not a quality gate; lower holdout FM loss is also the memorization "
        "leading edge). Spearman≤0 ⇒ FM-fit is as blind as Gram here.",
        "",
    ]
    report = "\n".join(lines)
    print("\n" + report, flush=True)

    run_dir = make_run_dir("paired_eval", args.label)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics={
            "spearman_fm_vs_cmmd": spearman,
            "plain_wins_base": plain_wins_base,
            "plain_is_top": plain_is_top,
            "per_model_R": per_model_R,
            "rows": rows,
            "fm_rank": fm_rank,
            "cmmd_rank": cmmd_rank,
        },
        artifacts=["report.md"],
        device=acc.device,
    )
    print(f"done → {run_dir}", flush=True)


if __name__ == "__main__":
    main()
