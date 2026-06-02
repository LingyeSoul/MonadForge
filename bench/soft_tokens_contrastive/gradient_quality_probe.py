"""AGSM gradient-quality probe — native soft-rank vs detached-PL target regression.

THE QUESTION. Anima's shipped AGSM objective (soft_tokens.py ``agsm_targets`` /
``agsm_losses``) computes a Plackett–Luce ranking of the candidate captions, then
**``.detach()``-es it** into a frozen shifted MSE target. The ordering is computed
and thrown away as a constant — gradient never flows *through* the ranking. That is
the "compute-the-discrete-thing-then-regress" workaround this probe exists to test.

The native-gradient alternative is a differentiable listwise ranking loss
(``_softrank.soft_rank``, the vendored ~20-line core of softtorch) whose gradient
flows through the soft ordering. The pitch is the missing corner of the bracket:

  * InfoNCE (the other shipped objective) DOES push gradient through the ranking
    (softmax-CE = soft top-1) but is **unbounded** — the reason AGSM was adopted.
  * AGSM is **bounded** but detaches the ranking.
  * soft-rank should give InfoNCE-like *direction* with AGSM-like *boundedness*.

So the claim under test is narrow and falsifiable: a differentiable soft-rank loss
points more like "actually improve the matched-vs-best-negative margin" than AGSM's
detached MSE does, AND its gradient stays bounded as the matched caption loses.

WHAT IT DOES (no training). Reuses the ``reward_premise_probe`` scaffold (frozen
base DiT, cached anchors, σ-grid, matched + k negatives sharing one ``(x_t, ε, t)``).
For each operating point it runs one bare-DiT forward per candidate, stacks the
velocities ``V = [v_matched, v_neg₁…v_neg_k]`` as a leaf, and computes
``g = ∂L/∂V`` for four heads on the *same* ``V`` (so ∂V/∂ψ cancels and the
comparison is splice-independent):

  * ``agsm``     — detached-PL shifted-target MSE (the shipped objective; EMA≈live).
  * ``infonce``  — softmax-CE to the matched index (reference).
  * ``softrank`` — ``soft_rank(matched)`` listwise loss (the native-gradient arm).
  * ``ideal``    — ``∂(−margin)/∂V`` where ``margin = r_matched − max_j r_neg_j``;
                   the gradient of the probe's *own* goal metric — the "truth" the
                   training losses are surrogates for.

METRICS (per σ, aggregated):

  1. **alignment** ``cos(g_obj, g_ideal)`` — does this loss point where the margin
     actually improves? The headline. Higher is better.
  2. **boundedness** — sweep the matched error magnitude (s·‖v_matched−v_target‖)
     and report ‖g_matched‖ vs s. InfoNCE should blow up; soft-rank must stay
     sub-linear (the AGSM-retention test).
  3. **near-miss credit** — fraction of negative-branch gradient mass landing on
     the runner-up (highest-reward) negative vs the clear losers. Detached MSE
     spreads it; soft-rank should concentrate it where a reorder is achievable.
  4. **self-anneal** — corr(‖g_matched‖, w_matched). Should be negative: as the
     matched caption wins, the pull relaxes toward plain FM (AGSM's bounded
     fixed point — soft-rank must reproduce it, not over-drive a won case).

GATE (GO for a Tier-B live A/B). softrank alignment ≥ agsm alignment (ideally
≥ infonce) on the informative σ band (≥0.45) AND softrank boundedness sub-linear
(slope ratio vs infonce < 1).

HONEST KILL. This measures gradient *quality*, not signal quality. If agsm's
alignment is already high, the bottleneck is the reward premise / char-tag
coverage (project_soft_tokens_hard_negative_untagged) and a better gradient won't
help — do NOT proceed to Tier B. The reward-premise probe is the prerequisite GO.

Run from anima_lora/::

    uv run python -m bench.soft_tokens_contrastive.gradient_quality_probe
    uv run python -m bench.soft_tokens_contrastive.gradient_quality_probe \
        --num_samples 24 --contrastive_k 2 --num_seeds 3 --label tierA
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

from bench._anima import add_common_args, build_anima
from bench._common import make_run_dir, write_result
from bench.soft_tokens_contrastive._softrank import soft_rank
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

log = logging.getLogger("bench.soft_tokens_contrastive.gradient_quality")
logging.basicConfig(level=logging.INFO, format="%(message)s")

OBJECTIVES = ("agsm", "infonce", "softrank")
# matched-error scale grid for the boundedness sweep: s<1 ⇒ matched fits better
# (wins), s>1 ⇒ matched loses progressively harder. s=1 is the real operating
# point. The boundedness signature lives in the s>1 tail.
BOUND_SCALES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


# ── velocity forward (one candidate) ─────────────────────────────────────────


@torch.no_grad()
def _velocity(anima, noisy, t, emb, pad) -> torch.Tensor:
    """Bare-DiT velocity for one caption — same forward as reward_premise.fm_reward
    but returns the (1,C,1,H,W) float velocity instead of the scalar reward."""
    return anima(noisy, t, emb, padding_mask=pad).float()


def _rewards(V: torch.Tensor, vt: torch.Tensor) -> torch.Tensor:
    """``r_j = −mean((V_j − v_target)²)`` over C·H·W — the InfoNCE/AGSM reward,
    higher = better. ``V`` is (m, …); returns (m,)."""
    return -((V - vt) ** 2).flatten(1).mean(dim=1)


# ── loss heads (all pure functions of the leaf V and the constant target) ─────


def loss_infonce(V, vt, tau):
    return -torch.log_softmax(_rewards(V, vt) / tau, dim=0)[0]


def loss_agsm(V, vt, tau, gp, gn):
    """Replicates agsm_targets/agsm_losses with the EMA shadow taken at its
    ema≈live limit (the probe is frozen, so there is no lagged bank): PL weights
    and shifted targets are detached exactly as the shipped code detaches them."""
    r = _rewards(V, vt)
    w = torch.softmax(r / tau, dim=0).detach()  # (m,) — DETACHED ranking
    shp = (-1,) + (1,) * (V.dim() - 1)
    baseline = (w.view(*shp) * V).sum(dim=0, keepdim=True).detach()  # (1, …)
    tgt_pos = (vt + gp * (V[0:1] - baseline)).detach()
    tgt_neg = (vt - gn * (V[1:] - baseline)).detach()
    l_pos = ((V[0:1] - tgt_pos) ** 2).flatten(1).mean()
    l_neg = ((V[1:] - tgt_neg) ** 2).flatten(1).mean()
    return l_pos + l_neg


def loss_softrank(V, vt, tau):
    """Native-gradient listwise loss: the differentiable rank of the matched
    candidate (index 0) among all candidates — gradient flows THROUGH the soft
    ordering. Bounded in [0, m−1]; → 0 as matched cleanly wins (self-anneal)."""
    r = _rewards(V, vt)  # higher = better
    return soft_rank(r, tau=tau, dim=0)[0]


def neg_margin(V, vt):
    """``−(r_matched − max_j r_neg)`` — the negative of the probe's own goal
    metric. ``∂/∂V`` is the 'ideal' descent direction every loss approximates."""
    r = _rewards(V, vt)
    return -(r[0] - r[1:].max())


def _loss_fn(name, V, vt, *, tau, gp, gn):
    if name == "agsm":
        return loss_agsm(V, vt, tau, gp, gn)
    if name == "infonce":
        return loss_infonce(V, vt, tau)
    if name == "softrank":
        return loss_softrank(V, vt, tau)
    raise ValueError(name)


def _grad(loss_fn, V) -> torch.Tensor:
    leaf = V.detach().requires_grad_(True)
    (g,) = torch.autograd.grad(loss_fn(leaf), leaf)
    return g.detach()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    return float(
        torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dit", default=DEFAULT_DIT)
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--num_samples", type=int, default=16, help="number of anchors")
    ap.add_argument("--contrastive_k", type=int, default=2, help="negatives/anchor")
    ap.add_argument("--num_seeds", type=int, default=2, help="noise draws averaged")
    ap.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    ap.add_argument("--tau", type=float, default=0.5, help="τ for agsm/infonce/softrank")
    ap.add_argument("--agsm_gamma", type=float, default=1.0, help="γ⁺")
    ap.add_argument("--agsm_gamma_neg", type=float, default=0.1, help="γ⁻")
    ap.add_argument(
        "--negative_mode",
        choices=["shuffled", "hard"],
        default="hard",
        help="which negative pool to probe (hard = same-artist/diff-char, the "
        "realistic training negative; shuffled = the kill-gate pool).",
    )
    ap.add_argument(
        "--informative_sigma",
        type=float,
        default=0.45,
        help="σ ≥ this counts toward the gate (x0 resolves by ~0.45).",
    )
    add_common_args(p := ap)
    args = p.parse_args()

    sigma_grid = sorted(float(s) for s in args.sigmas)
    k = int(args.contrastive_k)
    tau, gp, gn = float(args.tau), float(args.agsm_gamma), float(args.agsm_gamma_neg)
    log.info(
        f"σ grid: {sigma_grid} | k={k} | τ={tau} γ⁺={gp} γ⁻={gn} | "
        f"neg={args.negative_mode}"
    )

    pairs = discover_pairs(args.data_dir)
    pool_list = sorted(pairs)
    index = load_index(args.index) if args.negative_mode == "hard" else None
    if args.negative_mode == "hard" and index is None:
        raise SystemExit("hard negatives requested but caption index missing")
    log.info(f"{len(pool_list)} cached (latent, TE) pairs under {args.data_dir}")

    rng = np.random.default_rng(args.seed)
    anchor_idx = rng.choice(
        len(pool_list), size=min(args.num_samples, len(pool_list)), replace=False
    )
    anchors = [pool_list[int(i)] for i in anchor_idx]

    bundle = build_anima(args, adapter=None, train_mode=False)
    anima, device, dtype = bundle.anima, bundle.device, bundle.dtype
    embs = EmbCache(pairs)

    # accumulators
    align = {o: {s: [] for s in sigma_grid} for o in OBJECTIVES}  # cos(g_obj, g_ideal)
    nearmiss = {o: [] for o in OBJECTIVES}  # runner-up gradient-mass fraction
    anneal = {o: {"g": [], "w": []} for o in OBJECTIVES}  # ‖g_matched‖ vs w_matched
    bound = {o: {sc: [] for sc in BOUND_SCALES} for o in OBJECTIVES}  # ‖g_m‖ vs scale
    n_skipped = 0

    for ai, anchor in enumerate(anchors):
        npz_path, _te = pairs[anchor]
        lat, *_ = load_cached_latents(npz_path)
        anchor_emb = embs.get(anchor)
        if anchor_emb is None:
            continue

        nrng = np.random.default_rng(args.seed + 1000 + ai)
        if args.negative_mode == "hard":
            neg_stems = hard_negatives(anchor, index, set(pairs), k, nrng)
        else:
            neg_stems = shuffled_negatives(anchor, pool_list, k, nrng)
        cand = [embs.get(s) for s in neg_stems]
        cand = [e for e in cand if e is not None]
        if len(cand) != k:
            n_skipped += 1
            continue

        x0 = lat.to(device, dtype).unsqueeze(0).unsqueeze(2)  # (1,C,1,H,W)
        H, W = x0.shape[-2], x0.shape[-1]
        pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)
        x0_f = x0.float()
        emb_a = anchor_emb.to(device, dtype).unsqueeze(0)
        cand_embs = [e.to(device, dtype).unsqueeze(0) for e in cand]

        for s in sigma_grid:
            for sj in range(max(1, args.num_seeds)):
                g = torch.Generator(device=device).manual_seed(
                    args.seed + ai * 1000 + sj
                )
                eps = torch.randn(x0.shape, generator=g, device=device, dtype=dtype)
                vt = (eps.float() - x0_f)[0]  # (C,1,H,W) target
                noisy = ((1.0 - s) * x0_f + s * eps.float()).to(dtype)
                t = torch.full((1,), float(s), device=device, dtype=dtype)

                vs = [_velocity(anima, noisy, t, emb_a, pad)[0]]
                vs += [_velocity(anima, noisy, t, ce, pad)[0] for ce in cand_embs]
                V = torch.stack(vs, dim=0)  # (m, C,1,H,W)

                g_ideal = _grad(lambda x: neg_margin(x, vt), V)
                r = _rewards(V, vt)
                w_matched = float(torch.softmax(r / tau, dim=0)[0])
                best_neg = int(1 + torch.argmax(r[1:]).item())  # runner-up index

                for o in OBJECTIVES:
                    g_o = _grad(
                        lambda x, _o=o: _loss_fn(_o, x, vt, tau=tau, gp=gp, gn=gn), V
                    )
                    align[o][s].append(_cos(g_o, g_ideal))
                    # near-miss: negative-branch grad mass on runner-up vs all negs
                    neg_mass = g_o[1:].flatten(1).norm(dim=1)  # (k,)
                    tot = float(neg_mass.sum()) + 1e-12
                    nearmiss[o].append(float(neg_mass[best_neg - 1]) / tot)
                    # self-anneal: matched-slice grad norm vs how much matched wins
                    anneal[o]["g"].append(float(g_o[0].norm()))
                    anneal[o]["w"].append(w_matched)

                # boundedness sweep — no extra forwards, scale the matched error.
                e0 = V[0:1] - vt
                for sc in BOUND_SCALES:
                    Vs = V.clone()
                    Vs[0:1] = vt + sc * e0
                    for o in OBJECTIVES:
                        g_o = _grad(
                            lambda x, _o=o: _loss_fn(_o, x, vt, tau=tau, gp=gp, gn=gn),
                            Vs,
                        )
                        bound[o][sc].append(float(g_o[0].norm()))

        log.info(f"  [{ai + 1}/{len(anchors)}] {anchor} probed")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── aggregate ────────────────────────────────────────────────────────────
    def _m(xs):
        return float(np.mean(xs)) if xs else None

    align_sigma = {
        o: {s: _m(align[o][s]) for s in sigma_grid} for o in OBJECTIVES
    }
    info_band = [s for s in sigma_grid if s >= args.informative_sigma]
    align_band = {
        o: _m([v for s in info_band for v in align[o][s]]) for o in OBJECTIVES
    }
    nearmiss_mean = {o: _m(nearmiss[o]) for o in OBJECTIVES}
    bound_curve = {o: {sc: _m(bound[o][sc]) for sc in BOUND_SCALES} for o in OBJECTIVES}

    def _anneal_r(o):
        gw, ww = anneal[o]["g"], anneal[o]["w"]
        if len(gw) < 3 or np.std(gw) < 1e-12 or np.std(ww) < 1e-12:
            return None
        return float(np.corrcoef(gw, ww)[0, 1])

    anneal_r = {o: _anneal_r(o) for o in OBJECTIVES}

    # boundedness slope ratio: ‖g‖ growth over the s≥1 tail, softrank vs infonce.
    def _tail_slope(o):
        tail = [sc for sc in BOUND_SCALES if sc >= 1.0]
        ys = [bound_curve[o][sc] for sc in tail]
        if any(y is None for y in ys):
            return None
        return float(np.polyfit(tail, ys, 1)[0])

    slope = {o: _tail_slope(o) for o in OBJECTIVES}
    slope_ratio = (
        slope["softrank"] / slope["infonce"]
        if slope["softrank"] is not None
        and slope["infonce"] not in (None, 0.0)
        else None
    )

    sr, ag = align_band["softrank"], align_band["agsm"]
    align_ok = sr is not None and ag is not None and sr >= ag
    bound_ok = slope_ratio is not None and slope_ratio < 1.0
    verdict = "GO" if (align_ok and bound_ok) else "NO-GO"

    # ── render ───────────────────────────────────────────────────────────────
    def f(v, p=".3f"):
        return "—" if v is None else f"{v:{p}}"

    L = ["# AGSM gradient-quality probe — soft-rank vs detached-PL\n"]
    L.append(
        "Does a **native-gradient** soft-rank loss point more like 'improve the "
        "matched-vs-best-negative margin' than the shipped **detached-PL** AGSM "
        "target-regression — while staying bounded? `agsm` detaches the ranking "
        "into a frozen MSE target; `softrank` flows gradient through the soft "
        "ordering; `infonce` is the unbounded reference; `ideal` = ∂(−margin)/∂V.\n"
    )
    L.append(
        f"- anchors **{len(anchors) - n_skipped}** (skipped {n_skipped} for thin "
        f"neg pool) · k={k} · {max(1, args.num_seeds)} seeds · neg=`{args.negative_mode}`\n"
        f"- τ={tau} γ⁺={gp} γ⁻={gn} · informative band σ≥{args.informative_sigma}\n"
        f"- **GATE** (softrank align ≥ agsm on band **AND** bound slope ratio<1): "
        f"**{verdict}**\n"
    )

    L.append("\n## 1. Alignment with ∂(−margin) — `cos(g_obj, g_ideal)`, per σ\n")
    L.append("| σ | " + " | ".join(OBJECTIVES) + " |")
    L.append("|---|" + "---|" * len(OBJECTIVES))
    for s in sigma_grid:
        L.append(
            f"| {s:.2f} | " + " | ".join(f(align_sigma[o][s]) for o in OBJECTIVES) + " |"
        )
    L.append(
        f"| **band≥{args.informative_sigma:.2f}** | "
        + " | ".join(f"**{f(align_band[o])}**" for o in OBJECTIVES)
        + " |"
    )

    L.append("\n## 2. Boundedness — ‖g_matched‖ vs matched-error scale\n")
    L.append("| scale | " + " | ".join(OBJECTIVES) + " |")
    L.append("|---|" + "---|" * len(OBJECTIVES))
    for sc in BOUND_SCALES:
        L.append(
            f"| {sc:.2f}× | "
            + " | ".join(f(bound_curve[o][sc], ".4f") for o in OBJECTIVES)
            + " |"
        )
    L.append(
        "| **tail slope** | "
        + " | ".join(f(slope[o], ".4f") for o in OBJECTIVES)
        + " |"
    )
    L.append(
        f"\nsoftrank/infonce tail-slope ratio = **{f(slope_ratio)}** "
        "(< 1 ⇒ soft-rank stays bounded where InfoNCE blows up).\n"
    )

    L.append("\n## 3. Near-miss credit & 4. self-anneal\n")
    L.append("| obj | runner-up grad-mass frac | corr(‖g_m‖, w_matched) |")
    L.append("|---|---|---|")
    for o in OBJECTIVES:
        L.append(f"| {o} | {f(nearmiss_mean[o])} | {f(anneal_r[o])} |")
    L.append(
        "\n- **near-miss**: higher ⇒ gradient concentrates on the binding "
        "competitor (chance = 1/k); detached-MSE tends to spread it.\n"
        "- **self-anneal**: negative ⇒ the pull relaxes as the matched caption "
        "wins (AGSM's bounded fixed point; soft-rank should reproduce it).\n"
    )
    L.append("\n## Reading it\n")
    L.append(
        "- **GO** ⇒ soft-rank's gradient is better-aligned *and* bounded → worth a "
        "Tier-B live A/B (`contrastive_objective=softrank`, same plumbing, CMMD + "
        "reward_premise rank@1 on the trained bank).\n"
        "- **NO-GO via alignment** ⇒ AGSM's detached target already points the "
        "right way; the native gradient buys nothing here. Bottleneck is the reward "
        "premise / negative coverage, not the gradient — don't build it.\n"
        "- **NO-GO via boundedness** ⇒ soft-rank inherits InfoNCE's blow-up; it "
        "would need the same damping AGSM was invented to avoid.\n"
        "- CAVEAT: gradient *quality*, not *signal* quality. Gate the reward-premise "
        "probe first; a clean gradient through a dead reward is still dead.\n"
    )

    run_dir = make_run_dir(
        "soft_tokens_contrastive", label=args.label or "grad-quality"
    )
    (run_dir / "gradient_quality.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    csv = run_dir / "alignment_per_sigma.csv"
    with csv.open("w") as fcsv:
        fcsv.write("sigma," + ",".join(OBJECTIVES) + "\n")
        for s in sigma_grid:
            fcsv.write(
                f"{s}," + ",".join(str(align_sigma[o][s]) for o in OBJECTIVES) + "\n"
            )

    metrics = {
        "n_anchors": len(anchors) - n_skipped,
        "n_skipped": n_skipped,
        "contrastive_k": k,
        "num_seeds": int(max(1, args.num_seeds)),
        "negative_mode": args.negative_mode,
        "tau": tau,
        "agsm_gamma": gp,
        "agsm_gamma_neg": gn,
        "sigma_grid": sigma_grid,
        "informative_sigma": args.informative_sigma,
        "alignment_per_sigma": align_sigma,
        "alignment_band": align_band,
        "nearmiss_mean": nearmiss_mean,
        "anneal_corr": anneal_r,
        "boundedness_curve": bound_curve,
        "boundedness_tail_slope": slope,
        "softrank_infonce_slope_ratio": slope_ratio,
        "verdict": verdict,
        "note": (
            "Gradient quality, not signal quality. softrank = vendored soft_rank "
            "(softtorch core). Gate reward_premise_probe first."
        ),
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["gradient_quality.md", "alignment_per_sigma.csv"],
        device=device,
    )

    log.info("\n" + "=" * 70)
    log.info(f"  AGSM gradient-quality probe → {run_dir}")
    for o in OBJECTIVES:
        log.info(
            f"  [{o:8s}] align(band≥{args.informative_sigma:.2f})="
            f"{f(align_band[o])}  near-miss={f(nearmiss_mean[o])}  "
            f"anneal_r={f(anneal_r[o])}  tail_slope={f(slope[o], '.4f')}"
        )
    log.info(
        f"  GATE: softrank≥agsm align ({f(sr)}≥{f(ag)}? {align_ok}) & "
        f"slope_ratio={f(slope_ratio)}<1? {bound_ok} → {verdict}"
    )
    log.info("  open: gradient_quality.md")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
