"""AGSM Eq. 51 renoise probe (Phase 3b gate of docs/proposal/soft_tokens_agsm.md).

THE QUESTION. Phase 2 (and Algorithm 1 as we shipped it) evaluates the AGSM
guidance preds at the anchor's own ``(x_t, t)``. The flow derivation (Appendix D,
Eq. 47/51) evaluates them at a **renoised** point one small step toward noise::

    x_{t+Δ} = (1−t−Δ)·x0 + (t+Δ)·ε = x_t + Δ·(ε − x0) = x_t + Δ·v_target   (clamp t+Δ ≤ 1)

The proposal flags a **cheap offline probe before wiring**: compute the per-candidate
PL guidance direction ``Δ_j = v̂_j − Σ_k w_k v̂_k`` and the matched PL weight
``w_matched`` at ``x_t`` vs at ``x_{t+Δ}`` on cached anchors —

  > if they barely differ, the Algorithm-1 collapse loses nothing and 3b is a no-op.

This is that probe (the renoise sibling of ``reward_premise_probe.py``). It reuses
that script's cached-pair discovery, negative sourcing, and embedding cache.

WHAT IT DOES (no training). For ``n`` cached anchors, at each σ and averaged over a
few noise draws, it builds ``x_t`` + the FM target ``v_target = ε − x0`` (train.py
convention), then for each Δ in the sweep also builds the renoised ``x_{t+Δ}``. At
both eval points it runs a **bare-DiT forward per candidate caption** (matched + k
negatives; only ``crossattn_emb`` differs — the ``extra_forwards`` contract) and
forms, exactly as ``agsm_targets`` does:

    reward_j   = −‖v̂_j − v_target‖²            (v_target is FIXED — the regression
                                                 target stays at x_t per the proposal;
                                                 the FM velocity field is ~constant
                                                 along the (x0, ε) line so this is the
                                                 faithful reward at either eval point)
    w          = softmax_j(reward_j / τ)        (PL weights, index 0 = matched)
    baseline   = Σ_j w_j · v̂_j
    Δ_j        = v̂_j − baseline

and compares the two eval points by:

  * **cos(Δ_j)** — cosine similarity of the per-candidate guidance direction between
    ``x_t`` and ``x_{t+Δ}`` (the load-bearing number: Δ is what shifts the AGSM
    target, so if its *direction* is preserved the renoise changes nothing the loss
    sees). Reported for the matched candidate (``Δ⁺``, the kept-bank target) and the
    mean over negatives (``Δ⁻``).
  * **|Δw_matched|** — absolute shift in the matched PL weight (the self-annealing
    scalar). A large shift would mean the renoise re-ranks which caption "wins".

VERDICT (a no-training gate, not a quality claim). Aggregated over the σ grid and the
Δ sweep:

  * **NO-OP** — Δ⁺ cosine ≥ ``--cos_gate`` (default 0.98) and mean |Δw_matched| ≤
    ``--w_gate`` (default 0.02): the Algorithm-1 ``x_t`` collapse loses nothing, so
    wiring 3b is not worth the (clamp/bucketize/extra-arg) complexity. Skip it.
  * **MATTERS** — otherwise: the renoise moves the guidance, so 3b is worth wiring +
    a Phase-3b A/B against the shipped ``x_t`` baseline.

Run from anima_lora/::

    uv run python -m bench.soft_tokens_contrastive.renoise_probe
    uv run python -m bench.soft_tokens_contrastive.renoise_probe \
        --num_samples 24 --contrastive_k 2 --num_seeds 3 --deltas 0.02 0.05 0.0714
    uv run python -m bench.soft_tokens_contrastive.renoise_probe \
        --adapter output/ckpt/anima_soft_tokens_tenth.safetensors --label withbank
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

from bench._anima import add_common_args, build_anima
from bench._common import make_run_dir, write_result
from bench.soft_tokens_contrastive.reward_premise_probe import (
    DEFAULT_DATA,
    DEFAULT_DIT,
    DEFAULT_INDEX,
    DEFAULT_SIGMAS,
    EmbCache,
    discover_pairs,
    hard_negatives,
    load_index,
    shuffled_negatives,
)
from library.io.cache import load_cached_latents

log = logging.getLogger("bench.soft_tokens_contrastive.renoise")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# One t-bucket width at the toml's n_t_buckets=14 (≈0.0714) plus two fixed small
# steps from the proposal ("a fixed small 0.02–0.05; sweep").
DEFAULT_DELTAS = [0.02, 0.05, 0.0714]


def _splice_bank(net, emb: torch.Tensor, t: torch.Tensor) -> None:
    """Prime the soft-token splice for this caption, exactly as inference does
    (generation.py:374). front_of_padding needs the real (unpadded) seqlen so the
    K-token window can't index past S; end_of_sequence ignores it."""
    S = emb.shape[1]
    K = net.num_tokens
    nz = (emb.float().abs().sum(dim=-1) > 0).sum(dim=1)  # (B,)
    seqlens = nz.clamp(max=S - K).to(torch.long)
    net.append_postfix(emb, seqlens, timesteps=t)


def velocity(
    anima,
    net,
    x: torch.Tensor,  # (1,C,1,H,W) dtype — eval-point noised latent
    t: torch.Tensor,  # (1,) dtype — eval-point sigma
    emb: torch.Tensor,  # (1,S,D) dtype — candidate caption crossattn_emb
    pad: torch.Tensor,  # (1,1,H,W) dtype
) -> torch.Tensor:
    """One bare-DiT velocity (1,C,1,H,W) float for a candidate caption at (x, t).

    Unlike ``reward_premise_probe.fm_reward`` (which returns the scalar reward) we
    need the full velocity tensor to form the PL baseline + per-candidate Δ."""
    if net is not None:
        _splice_bank(net, emb, t)
    with torch.no_grad():
        v = anima(x, t, emb, padding_mask=pad).float()
    return v


def _pl_delta(
    vels: torch.Tensor,  # (m, C, H, W) float — candidate velocities, 0 = matched
    v_target: torch.Tensor,  # (C, H, W) float
    tau: float,
) -> tuple[np.ndarray, torch.Tensor]:
    """PL weights + per-candidate guidance Δ_j, mirroring ``agsm_targets``.

    reward_j = −mean((v̂_j − v_target)²); w = softmax(reward/τ);
    baseline = Σ_j w_j v̂_j; Δ_j = v̂_j − baseline. Returns (w (m,), Δ (m,C,H,W))."""
    m = vels.shape[0]
    err = (vels - v_target.unsqueeze(0)).pow(2).reshape(m, -1).mean(dim=1)  # (m,)
    w = torch.softmax(-err / max(tau, 1e-6), dim=0)  # (m,)
    baseline = (w.view(m, 1, 1, 1) * vels).sum(dim=0)  # (C,H,W)
    delta = vels - baseline.unsqueeze(0)  # (m,C,H,W)
    return w.cpu().numpy(), delta


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flattened cosine similarity between two Δ tensors."""
    af, bf = a.reshape(-1), b.reshape(-1)
    denom = af.norm() * bf.norm()
    if float(denom) < 1e-12:
        return float("nan")
    return float((af @ bf) / denom)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dit", default=DEFAULT_DIT)
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument(
        "--adapter",
        default=None,
        help="Optional trained soft-tokens bank, spliced like inference (ψ⁺). "
        "Default: bare base DiT — the eval-point sensitivity is a DiT property, "
        "the bank splice is a small perturbation on top.",
    )
    ap.add_argument("--num_samples", type=int, default=16, help="number of anchors")
    ap.add_argument(
        "--contrastive_k",
        type=int,
        default=2,
        help="negatives per anchor (matches contrastive_k; m = k+1 candidates).",
    )
    ap.add_argument(
        "--num_seeds",
        type=int,
        default=2,
        help="noise draws averaged per (anchor, σ) — single-draw labels are "
        "seed-variance dominated (project_dcw_seed_variance_dominates).",
    )
    ap.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    ap.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=DEFAULT_DELTAS,
        help="renoise step sizes Δ to sweep (x_{t+Δ} = x_t + Δ·v_target).",
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="PL temperature (matches contrastive_tau; affects w but not the "
        "direction comparison much).",
    )
    ap.add_argument(
        "--cos_gate",
        type=float,
        default=0.98,
        help="NO-OP verdict if Δ⁺ cosine (σ×Δ mean) ≥ this AND |Δw_matched| ≤ "
        "--w_gate. Above this the x_t collapse preserves the guidance direction.",
    )
    ap.add_argument("--w_gate", type=float, default=0.02)
    ap.add_argument(
        "--pool",
        choices=["shuffled", "hard"],
        default="shuffled",
        help="negative pool. shuffled = the kill-gate negative; hard = same-artist"
        "/diff-character (needs caption index).",
    )
    add_common_args(p := ap)
    args = p.parse_args()

    sigma_grid = sorted(float(s) for s in args.sigmas)
    deltas = sorted(float(d) for d in args.deltas)
    k = int(args.contrastive_k)
    log.info(f"σ grid: {sigma_grid}  |  Δ sweep: {deltas}  |  k={k}  τ={args.tau}")

    pairs = discover_pairs(args.data_dir)
    pool_list = sorted(pairs)
    log.info(f"{len(pool_list)} cached (latent, TE) pairs under {args.data_dir}")
    index = load_index(args.index)
    if args.pool == "hard" and index is None:
        raise SystemExit("--pool hard needs the caption index; none found")

    rng = np.random.default_rng(args.seed)
    anchor_idx = rng.choice(
        len(pool_list), size=min(args.num_samples, len(pool_list)), replace=False
    )
    anchors = [pool_list[int(i)] for i in anchor_idx]

    bundle = build_anima(args, adapter=None, train_mode=False)
    anima, device, dtype = bundle.anima, bundle.device, bundle.dtype

    net = None
    if args.adapter is not None:
        from networks.methods.soft_tokens import create_network_from_weights

        net, _sd = create_network_from_weights(
            1.0, args.adapter, None, None, anima, for_inference=True
        )
        net.apply_to(text_encoders=None, unet=anima)
        net.load_weights(args.adapter)
        net.to(device=device, dtype=dtype)
        net.eval()
        log.info(f"bank loaded: {args.adapter} (K={net.num_tokens}, ψ⁺ only)")

    embs = EmbCache(pairs)

    # acc[Δ][σ] = lists over anchors of {cos_pos, cos_neg, dw_matched, w_xt, w_xtd}
    acc = {
        d: {
            s: {"cos_pos": [], "cos_neg": [], "dw": [], "w_xt": [], "w_xtd": []}
            for s in sigma_grid
        }
        for d in deltas
    }
    n_skipped = 0

    for ai, anchor in enumerate(anchors):
        npz_path, _te = pairs[anchor]
        lat, _res, _oh, _ow = load_cached_latents(npz_path)
        anchor_emb = embs.get(anchor)
        if anchor_emb is None:
            log.warning(f"  anchor {anchor}: no crossattn_emb, skipping")
            continue

        nrng = np.random.default_rng(args.seed + 1000 + ai)
        if args.pool == "hard":
            negs = hard_negatives(anchor, index, set(pairs), k, nrng)
        else:
            negs = shuffled_negatives(anchor, pool_list, k, nrng)
        neg_embs = [embs.get(s) for s in negs]
        neg_embs = [e for e in neg_embs if e is not None]
        if len(neg_embs) != k:
            n_skipped += 1
            continue

        x0 = lat.to(device, dtype).unsqueeze(0).unsqueeze(2)  # (1,C,1,H,W)
        H, W = x0.shape[-2], x0.shape[-1]
        pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)
        x0_f = x0.float()
        cand_embs = [anchor_emb.to(device, dtype).unsqueeze(0)] + [
            e.to(device, dtype).unsqueeze(0) for e in neg_embs
        ]  # index 0 = matched

        for s in sigma_grid:
            for d in deltas:
                # Seed-averaged Δ/w at each eval point → one comparison per (σ,Δ).
                cos_pos_seeds, cos_neg_seeds, dw_seeds = [], [], []
                w_xt_seeds, w_xtd_seeds = [], []
                for sj in range(max(1, args.num_seeds)):
                    g = torch.Generator(device=device).manual_seed(
                        args.seed + ai * 1000 + sj
                    )
                    eps = torch.randn(x0.shape, generator=g, device=device, dtype=dtype)
                    v_target = (eps.float() - x0_f).squeeze(0).squeeze(1)  # (C,H,W)
                    x_t = ((1.0 - s) * x0_f + s * eps.float()).to(dtype)
                    # Renoise one step toward noise; clamp t+Δ ≤ 1.
                    td = min(1.0, s + d)
                    eff_d = td - s  # actual step after clamp (0 at σ→1)
                    x_td = (x_t.float() + eff_d * v_target.unsqueeze(0).unsqueeze(2)).to(
                        dtype
                    )
                    t_t = torch.full((1,), float(s), device=device, dtype=dtype)
                    t_td = torch.full((1,), float(td), device=device, dtype=dtype)

                    vt_xt = torch.stack(
                        [
                            velocity(anima, net, x_t, t_t, e, pad).squeeze(0).squeeze(1)
                            for e in cand_embs
                        ],
                        dim=0,
                    )  # (m,C,H,W)
                    vt_xtd = torch.stack(
                        [
                            velocity(anima, net, x_td, t_td, e, pad).squeeze(0).squeeze(1)
                            for e in cand_embs
                        ],
                        dim=0,
                    )

                    w_a, d_a = _pl_delta(vt_xt, v_target, args.tau)
                    w_b, d_b = _pl_delta(vt_xtd, v_target, args.tau)
                    cos_pos_seeds.append(_cos(d_a[0], d_b[0]))
                    cos_neg_seeds.append(
                        float(np.mean([_cos(d_a[j], d_b[j]) for j in range(1, k + 1)]))
                    )
                    dw_seeds.append(abs(float(w_a[0] - w_b[0])))
                    w_xt_seeds.append(float(w_a[0]))
                    w_xtd_seeds.append(float(w_b[0]))

                a = acc[d][s]
                a["cos_pos"].append(float(np.nanmean(cos_pos_seeds)))
                a["cos_neg"].append(float(np.nanmean(cos_neg_seeds)))
                a["dw"].append(float(np.mean(dw_seeds)))
                a["w_xt"].append(float(np.mean(w_xt_seeds)))
                a["w_xtd"].append(float(np.mean(w_xtd_seeds)))
        log.info(f"  [{ai + 1}/{len(anchors)}] {anchor} scored")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── aggregate ───────────────────────────────────────────────────────────
    def mean_or_none(xs: list[float]) -> float | None:
        return float(np.mean(xs)) if xs else None

    summary = {}
    for d in deltas:
        summary[d] = {}
        for s in sigma_grid:
            a = acc[d][s]
            summary[d][s] = {
                "cos_pos": mean_or_none(a["cos_pos"]),
                "cos_neg": mean_or_none(a["cos_neg"]),
                "dw_matched": mean_or_none(a["dw"]),
                "w_xt": mean_or_none(a["w_xt"]),
                "w_xtd": mean_or_none(a["w_xtd"]),
                "n": len(a["cos_pos"]),
            }

    # σ×Δ-grid means of the two gate quantities.
    all_cos_pos = [
        summary[d][s]["cos_pos"]
        for d in deltas
        for s in sigma_grid
        if summary[d][s]["cos_pos"] is not None
    ]
    all_dw = [
        summary[d][s]["dw_matched"]
        for d in deltas
        for s in sigma_grid
        if summary[d][s]["dw_matched"] is not None
    ]
    cos_pos_mean = float(np.mean(all_cos_pos)) if all_cos_pos else None
    dw_mean = float(np.mean(all_dw)) if all_dw else None
    verdict = (
        "NO-OP"
        if (
            cos_pos_mean is not None
            and dw_mean is not None
            and cos_pos_mean >= args.cos_gate
            and dw_mean <= args.w_gate
        )
        else "MATTERS"
    )

    # ── render markdown ──────────────────────────────────────────────────────
    L = ["# AGSM Eq. 51 renoise probe (Phase 3b gate)\n"]
    L.append(
        "Does evaluating the AGSM guidance preds at the renoised "
        "`x_{t+Δ}=x_t+Δ·v_target` (Eq. 51) change the per-candidate guidance "
        "`Δ_j` direction or the matched PL weight vs the shipped `x_t` collapse "
        "(Algorithm 1 / Phase 2)? If not, 3b is a no-op.\n"
    )
    L.append(
        f"- anchors: **{len([a for a in anchors])}** (n per cell varies; "
        f"{n_skipped} dropped for thin negatives) · k={k} · "
        f"{max(1, args.num_seeds)} noise draws · σ {sigma_grid} · Δ {deltas} · "
        f"τ={args.tau} · pool=`{args.pool}` · "
        f"bank={'ψ⁺ ' + args.adapter if args.adapter else 'none (base DiT)'}\n"
    )
    L.append(
        f"- gate: Δ⁺ cos (σ×Δ mean) **{_f(cos_pos_mean)}** ≥ {args.cos_gate} AND "
        f"|Δw_matched| **{_f(dw_mean, '.4f')}** ≤ {args.w_gate} → **{verdict}**\n"
    )

    for d in deltas:
        L.append(f"\n## Δ = {d:.4f}\n")
        L.append("| σ | t+Δ | Δ⁺ cos | Δ⁻ cos | w_matched(x_t) | w_matched(x_{t+Δ}) | |Δw| |")
        L.append("|---|---|---|---|---|---|---|")
        for s in sigma_grid:
            r = summary[d][s]
            td = min(1.0, s + d)
            L.append(
                f"| {s:.2f} | {td:.3f} | {_f(r['cos_pos'])} | {_f(r['cos_neg'])} | "
                f"{_f(r['w_xt'])} | {_f(r['w_xtd'])} | {_f(r['dw_matched'], '.4f')} |"
            )

    L.append("\n## Reading it\n")
    L.append(
        "- **Δ⁺ cos ≈ 1 and |Δw| ≈ 0** ⇒ the renoise preserves the guidance the "
        "AGSM loss actually uses — the `x_t` collapse (Algorithm 1) loses nothing, "
        "so wiring 3b (clamp + bucketize t+Δ + an extra EMA eval-point arg) buys "
        "nothing. **Skip 3b.**\n"
        "- **Δ⁺ cos materially < 1 or |Δw| materially > 0** (esp. in the informative "
        "mid/high-σ band where caption-conditioning is discriminable, "
        "`project_agsm_reward_premise_holds`) ⇒ the eval point moves the guidance → "
        "3b is worth wiring + a Phase-3b A/B against the shipped `x_t` baseline.\n"
        "- At σ→1 the clamp `t+Δ≤1` shrinks the effective step toward 0, so cos→1 "
        "there is an artifact of the clamp, not evidence of insensitivity — weight "
        "the verdict on the σ≤0.75 rows.\n"
        "- CAVEAT: a no-training direction/weight comparison, NOT a quality claim. "
        "MATTERS ≠ 3b helps; it only says the collapse is lossy enough to test.\n"
    )

    run_dir = make_run_dir(
        "soft_tokens_contrastive", label=args.label or "renoise-3b"
    )
    (run_dir / "renoise.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    csv = run_dir / "renoise.csv"
    with csv.open("w") as f:
        f.write("delta,sigma,cos_pos,cos_neg,dw_matched,w_xt,w_xtd,n\n")
        for d in deltas:
            for s in sigma_grid:
                r = summary[d][s]
                f.write(
                    f"{d},{s},{r['cos_pos']},{r['cos_neg']},{r['dw_matched']},"
                    f"{r['w_xt']},{r['w_xtd']},{r['n']}\n"
                )

    metrics = {
        "n_anchors": len(anchors),
        "n_skipped": n_skipped,
        "contrastive_k": k,
        "num_seeds": int(max(1, args.num_seeds)),
        "sigma_grid": sigma_grid,
        "deltas": deltas,
        "tau": args.tau,
        "pool": args.pool,
        "adapter": args.adapter,
        "cos_gate": args.cos_gate,
        "w_gate": args.w_gate,
        "delta_pos_cos_mean": None if cos_pos_mean is None else round(cos_pos_mean, 5),
        "dw_matched_mean": None if dw_mean is None else round(dw_mean, 6),
        "verdict": verdict,
        "summary": {
            str(d): {str(s): summary[d][s] for s in sigma_grid} for d in deltas
        },
        "note": (
            "Offline gate for Phase 3b (Eq. 51 renoise). NO-OP ⇒ x_t collapse is "
            "safe, skip 3b. MATTERS ⇒ wire + A/B. Not a quality claim."
        ),
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["renoise.md", "renoise.csv"],
        device=device,
    )

    log.info("\n" + "=" * 70)
    log.info(f"  AGSM renoise probe → {run_dir}")
    log.info(
        f"  Δ⁺ cos (σ×Δ mean) = {_f(cos_pos_mean)}  (≥{args.cos_gate}?)  "
        f"|Δw_matched| = {_f(dw_mean, '.4f')}  (≤{args.w_gate}?)  → {verdict}"
    )
    log.info("  open: renoise.md")
    log.info("=" * 70)


def _f(v: float | None, p: str = ".3f") -> str:
    return "—" if v is None else f"{v:{p}}"


if __name__ == "__main__":
    main()
