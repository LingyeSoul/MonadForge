#!/usr/bin/env python3
"""Turbo caption-ranking probe (Phase 0 of docs/proposal/turbo_caption_ranking.md).

THE QUESTION. Does DP-DMD distillation preserve the teacher's caption
discriminability? Nothing currently measures turbo prompt-following (CMMD is
distribution match, not text alignment), and prompt-following degradation is a
classic few-step-distill failure mode. The validated reward premise
(``docs/findings/agsm_reward_premise_holds.md``) says relative FM-ranking at a
shared ``(x_t, ε, t)`` is a trustworthy compass for σ ≥ 0.45 — and the 4-step
student operates only at σ ∈ {1.0, 0.9, 0.75, 0.5}, entirely inside that band.

THREE ARMS (no training):

* **0a renoised-real ranking** — the reward_premise_probe construction:
  ``x_t = (1−σ)x0 + σε`` from cached real latents, rank matched vs negative
  captions by ``−‖v(x_t, c_j) − v_target‖²`` with ``v_target = ε − x0``. Run for
  both the bare base DiT (LoRA multiplier 0 — the teacher's backbone) and the
  turbo student (multiplier 1) on the SAME states, so degradation reads as
  student-below-base at matched σ. At σ=1.0, ``x_t = ε`` exactly — the model
  must rank purely from text, the student's step-0 situation.

* **0b anchor ranking (on-trajectory)** — the ``turbo_fei`` lesson: measure at
  the exact state + target the diversity loss trains. From fresh ε, roll the
  teacher CFG anchor ``k_anchor`` steps on the matched caption (the
  ``distill.py`` step-0 construction) → ``v_target = (ε − z_tk)/(1 − t_k)``;
  rank ``−‖v(ε, t=1, c_j) − v_target‖²`` over matched + shuffled negatives, for
  both student and base.

* **0c caption-contrast transfer** — catches the failure 0a can't: a student
  that still *ranks* correctly but with collapsed contrast magnitude. At shared
  states (0a's ``x_t`` at the contrast σ's, plus the student's own ``z1`` after
  its real first step), compare the text-conditioning channel's gain:
  ``ratio = ‖v_S(c_pos) − v_S(c_neg)‖ / ‖v_T_cfg(c_pos) − v_T_cfg(c_neg)‖`` and
  the cosine between the two contrast directions. CFG is affine, so
  ``v_T_cfg(c_pos) − v_T_cfg(c_neg) = α·(v_T(c_pos) − v_T(c_neg))`` exactly —
  the uncond forward cancels and the teacher contrast costs no extra forwards
  beyond the plain cond pair.

GATE (pre-registered in the proposal). DEGRADATION if any of:

* student shuffled rank@1 (0a) < ``--gate_abs_rank1`` (0.93) at any σ ≥ 0.75
  (frozen base reference: 0.993 σ-mean, run 20260529-1157-phase0-agsm);
* student rank@1 more than ``--gate_rel_drop`` (0.05) below the base arm at
  matched σ — applied per-pool across the whole grid (this is the only check
  that bites at σ=0.5, where the absolute threshold was never calibrated);
* contrast ratio < ``--gate_ratio`` (0.7) AND cosine < ``--gate_cos`` (0.8).

NO DEGRADATION → STOP and write the finding ("DP-DMD preserves caption
discriminability"); Phase 1 (soft-rank auxiliary in distill.py) never happens.

Run from anima_lora/::

    uv run python bench/dpdmd/caption_ranking_probe.py \
        --adapter output/ckpt/anima_turbo_N_1250.safetensors
    # over-distilled comparison arm:
    uv run python bench/dpdmd/caption_ranking_probe.py \
        --adapter output/ckpt/<overbaked>.safetensors --label overdistill
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

from bench._anima import (  # noqa: E402
    add_common_args,
    add_model_args,
    build_anima,
)
from bench._common import make_run_dir, write_result  # noqa: E402

# Negative sourcing + cache plumbing are shared with the AGSM reward-premise
# probe — same anchors/pools semantics keeps the frozen reference numbers
# comparable. Import rather than fork.
from bench.soft_tokens_contrastive.reward_premise_probe import (  # noqa: E402
    EmbCache,
    _fmt,
    discover_pairs,
    hard_negatives,
    load_index,
    shuffled_negatives,
)
from library.inference.sampling import get_timesteps_sigmas  # noqa: E402
from library.inference.uncond import (  # noqa: E402
    default_uncond_path,
    load_uncond_crossattn,
    uncond_for_batch,
)
from library.io.cache import load_cached_latents  # noqa: E402

log = logging.getLogger("bench.dpdmd.caption_ranking")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATA = "post_image_dataset/lora"
DEFAULT_INDEX = "post_image_dataset/captions/caption_index.json"
# The 4-step student's operating band {1.0, 0.9, 0.75, 0.5} plus 0.97 (near-pure
# noise, just short of the exact-ε endpoint). All ≥ 0.45, the validated regime.
DEFAULT_SIGMAS = [0.50, 0.75, 0.90, 0.97, 1.00]
# σ's where 0c reads the renoised-real contrast (both are real operating states
# of the 4-step student).
DEFAULT_CONTRAST_SIGMAS = [0.75, 0.90]

ARMS = ("base", "student")  # multiplier 0.0 / 1.0 on the same DiT


def _set_arm(network, arm: str) -> None:
    """Teacher/student toggle via set_multiplier — zeroes the LoRA delta without
    changing control flow, so a compiled DiT keeps one graph family (the
    bench/dpdmd/probe_first_step_anchor.py pattern)."""
    network.set_multiplier(0.0 if arm == "base" else 1.0)


@torch.no_grad()
def _velocity(anima, x_t, t_b, emb, pad) -> torch.Tensor:
    """One DiT forward → fp32 velocity. ``enable_pooled_text_modulation`` is off
    for base-DiT + LoRA, so this is bit-equivalent to the distill loop's
    ``forward_mini_train_dit(..., skip_pooled_text_proj=True)``."""
    return anima(x_t, t_b, emb, padding_mask=pad).float()


def _reward(v: torch.Tensor, v_target: torch.Tensor) -> float:
    """``−mean((v − v_target)²)`` — the FM-ranking reward (reduction matches the
    reward-premise probe; constants cancel in the ranking)."""
    return -float(((v - v_target) ** 2).mean())


def _new_acc(sigmas):
    return {s: {"rank1": [], "mbest": [], "mmean": []} for s in sigmas}


def _push(acc_s, pos: float, negs: list[float]) -> None:
    best = max(negs)
    acc_s["rank1"].append(1.0 if pos > best else 0.0)
    acc_s["mbest"].append(pos - best)
    acc_s["mmean"].append(pos - float(np.mean(negs)))


def _agg(acc_s) -> dict:
    if not acc_s["rank1"]:
        return {"rank1": None, "mbest": None, "mmean": None, "n": 0}
    return {
        "rank1": float(np.mean(acc_s["rank1"])),
        "mbest": float(np.mean(acc_s["mbest"])),
        "mmean": float(np.mean(acc_s["mmean"])),
        "n": len(acc_s["rank1"]),
    }


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
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--uncond", default=str(default_uncond_path()))
    ap.add_argument("--num_samples", type=int, default=24, help="number of anchors")
    ap.add_argument(
        "--contrastive_k",
        type=int,
        default=2,
        help="negatives per anchor per pool (chance rank@1 = 1/(k+1)).",
    )
    ap.add_argument(
        "--num_seeds",
        type=int,
        default=2,
        help="noise draws averaged per (anchor, σ) before the ranking decision.",
    )
    ap.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    ap.add_argument(
        "--contrast_sigmas", type=float, nargs="+", default=DEFAULT_CONTRAST_SIGMAS
    )
    # Distill-schedule knobs — defaults match anima_turbo_N.snapshot.toml.
    ap.add_argument("--student_steps", type=int, default=4)
    ap.add_argument("--k_anchor", type=int, default=6)
    ap.add_argument("--teacher_anchor_steps", type=int, default=12)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--teacher_cfg", type=float, default=4.0)
    # Pre-registered gate.
    ap.add_argument("--gate_abs_rank1", type=float, default=0.93)
    ap.add_argument("--gate_rel_drop", type=float, default=0.05)
    ap.add_argument("--gate_ratio", type=float, default=0.7)
    ap.add_argument("--gate_cos", type=float, default=0.8)
    add_common_args(ap)
    args = ap.parse_args()

    sigma_grid = sorted(float(s) for s in args.sigmas)
    contrast_set = {float(s) for s in args.contrast_sigmas}
    k = int(args.contrastive_k)
    chance = 1.0 / (k + 1)
    log.info(f"σ grid: {sigma_grid}  |  k={k}  (chance rank@1 = {chance:.3f})")

    pairs = discover_pairs(args.data_dir)
    pool_list = sorted(pairs)
    log.info(f"{len(pool_list)} cached (latent, TE) pairs under {args.data_dir}")
    index = load_index(args.index)
    pools = ["shuffled"] + (["hard"] if index is not None else [])

    rng = np.random.default_rng(args.seed)
    anchor_idx = rng.choice(
        len(pool_list), size=min(args.num_samples, len(pool_list)), replace=False
    )
    anchors = [pool_list[int(i)] for i in anchor_idx]

    bundle = build_anima(args, adapter=args.adapter, train_mode=False)
    anima, network = bundle.anima, bundle.network
    device, dtype = bundle.device, bundle.dtype

    uncond_base = load_uncond_crossattn(args.uncond, device=device, dtype=dtype)
    log.info(f"T5('') uncond sidecar: {args.uncond}  shape={tuple(uncond_base.shape)}")

    # Distill grids (4-step default: student σ = [1.0, 0.9, 0.75, 0.5]).
    student_sigmas = get_timesteps_sigmas(args.student_steps, args.flow_shift, "cpu")[
        1
    ].tolist()
    teacher_anchor_sigmas = get_timesteps_sigmas(
        args.teacher_anchor_steps, args.flow_shift, "cpu"
    )[1].tolist()
    t_k_anchor = float(teacher_anchor_sigmas[args.k_anchor])
    s1 = float(student_sigmas[1])  # state after the student's real first step
    log.info(
        f"student σ={['%.3f' % s for s in student_sigmas]}  "
        f"anchor t_k={t_k_anchor:.4f} (teacher step {args.k_anchor}/"
        f"{args.teacher_anchor_steps}, CFG={args.teacher_cfg})"
    )

    embs = EmbCache(pairs)
    n_seeds = max(1, args.num_seeds)

    # acc_xt[arm][pool][σ]: 0a renoised-real ranking.
    acc_xt = {a: {pl: _new_acc(sigma_grid) for pl in pools} for a in ARMS}
    # acc_anchor[arm]: 0b on-trajectory ranking at t=1 (shuffled only — v0).
    acc_anchor = {a: _new_acc([1.0]) for a in ARMS}
    # 0c contrast: per-σ at renoised x_t, plus at the student's own z1.
    con_xt = {s: {"ratio": [], "cos": []} for s in sorted(contrast_set)}
    con_z1 = {"ratio": [], "cos": []}
    n_hard_skipped = 0

    def _contrast(v_s_pos, v_s_neg, v_t_pos, v_t_neg, sink) -> None:
        """contrast_S vs α·contrast_T (the CFG-affine closed form)."""
        c_s = (v_s_pos - v_s_neg).flatten()
        c_t = (args.teacher_cfg * (v_t_pos - v_t_neg)).flatten()
        sink["ratio"].append(float(c_s.norm() / c_t.norm().clamp_min(1e-12)))
        sink["cos"].append(
            float(torch.nn.functional.cosine_similarity(c_s, c_t, dim=0))
        )

    for ai, anchor in enumerate(anchors):
        npz_path, _te = pairs[anchor]
        lat, _res, _oh, _ow = load_cached_latents(npz_path)
        anchor_emb = embs.get(anchor)
        if anchor_emb is None:
            log.warning(f"  anchor {anchor}: no crossattn_emb, skipping")
            continue

        # Negative caption lists — same RNG stream as the reward-premise probe.
        nrng = np.random.default_rng(args.seed + 1000 + ai)
        neg_stems = {"shuffled": shuffled_negatives(anchor, pool_list, k, nrng)}
        if index is not None:
            hn = hard_negatives(anchor, index, set(pairs), k, nrng)
            if len(hn) < k:
                n_hard_skipped += 1
            neg_stems["hard"] = hn
        cand_embs = {}
        for pl in pools:
            es = [embs.get(s) for s in neg_stems[pl]]
            es = [e for e in es if e is not None]
            cand_embs[pl] = es if len(es) == k else None
        if cand_embs["shuffled"] is None:
            log.warning(f"  anchor {anchor}: shuffled pool came up short, skipping")
            continue

        x0 = lat.to(device, dtype).unsqueeze(0).unsqueeze(2)  # (1,C,1,H,W)
        H, W = x0.shape[-2], x0.shape[-1]
        pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)
        x0_f = x0.float()
        emb_a = anchor_emb.to(device, dtype).unsqueeze(0)
        emb_negs = {
            pl: [e.to(device, dtype).unsqueeze(0) for e in cand_embs[pl]]
            if cand_embs[pl] is not None
            else None
            for pl in pools
        }
        c_null = uncond_for_batch(uncond_base, emb_a)

        # ---- 0a renoised-real ranking (+ 0c contrast at the contrast σ's) ----
        for s in sigma_grid:
            # seed-averaged per-candidate reward → one ranking decision per σ.
            r_pos = {a: [] for a in ARMS}
            r_negs = {a: {pl: [[] for _ in range(k)] for pl in pools} for a in ARMS}
            for sj in range(n_seeds):
                g = torch.Generator(device=device).manual_seed(
                    args.seed + ai * 1000 + sj
                )
                eps = torch.randn(x0.shape, generator=g, device=device, dtype=dtype)
                v_target = eps.float() - x0_f
                noisy = ((1.0 - s) * x0_f + s * eps.float()).to(dtype)
                t = torch.full((1,), float(s), device=device, dtype=dtype)
                v_pos = {}
                v_neg0 = {}
                for arm in ARMS:
                    _set_arm(network, arm)
                    v = _velocity(anima, noisy, t, emb_a, pad)
                    v_pos[arm] = v
                    r_pos[arm].append(_reward(v, v_target))
                    for pl in pools:
                        if emb_negs[pl] is None:
                            continue
                        for j, ne in enumerate(emb_negs[pl]):
                            vn = _velocity(anima, noisy, t, ne, pad)
                            if pl == "shuffled" and j == 0:
                                v_neg0[arm] = vn
                            r_negs[arm][pl][j].append(_reward(vn, v_target))
                if s in contrast_set:
                    _contrast(
                        v_pos["student"],
                        v_neg0["student"],
                        v_pos["base"],
                        v_neg0["base"],
                        con_xt[s],
                    )
            for arm in ARMS:
                for pl in pools:
                    if emb_negs[pl] is None:
                        continue
                    _push(
                        acc_xt[arm][pl][s],
                        float(np.mean(r_pos[arm])),
                        [float(np.mean(rn)) for rn in r_negs[arm][pl]],
                    )

        # ---- 0b anchor ranking at t=1 (+ 0c contrast at the student's z1) ----
        r_pos_b = {a: [] for a in ARMS}
        r_negs_b = {a: [[] for _ in range(k)] for a in ARMS}
        for sj in range(n_seeds):
            g = torch.Generator(device=device).manual_seed(
                args.seed + 500_000 + ai * 1000 + sj
            )
            eps = torch.randn(x0.shape, generator=g, device=device, dtype=dtype)

            # Teacher K-step CFG anchor (distill.py step-0 construction).
            _set_arm(network, "base")
            z = eps
            for i in range(args.k_anchor):
                si, s_next = teacher_anchor_sigmas[i], teacher_anchor_sigmas[i + 1]
                t_b = torch.full((1,), float(si), device=device, dtype=dtype)
                v_c = _velocity(anima, z, t_b, emb_a, pad)
                if args.teacher_cfg == 1.0:
                    v = v_c
                else:
                    v_u = _velocity(anima, z, t_b, c_null, pad)
                    v = v_u + args.teacher_cfg * (v_c - v_u)
                z = (z.float() - (si - s_next) * v).to(dtype)
            v_target_b = (eps.float() - z.float()) / (1.0 - t_k_anchor)

            t1 = torch.full((1,), 1.0, device=device, dtype=dtype)
            v_first = None  # student matched velocity at t=1 → z1 for 0c
            for arm in ARMS:
                _set_arm(network, arm)
                v = _velocity(anima, eps, t1, emb_a, pad)
                if arm == "student":
                    v_first = v
                r_pos_b[arm].append(_reward(v, v_target_b))
                for j, ne in enumerate(emb_negs["shuffled"]):
                    r_negs_b[arm][j].append(
                        _reward(_velocity(anima, eps, t1, ne, pad), v_target_b)
                    )

            # 0c at z1: the student's own state after its real first step.
            z1 = (eps.float() - (1.0 - s1) * v_first).to(dtype)
            t_s1 = torch.full((1,), s1, device=device, dtype=dtype)
            ne0 = emb_negs["shuffled"][0]
            _set_arm(network, "student")
            v_s_pos = _velocity(anima, z1, t_s1, emb_a, pad)
            v_s_neg = _velocity(anima, z1, t_s1, ne0, pad)
            _set_arm(network, "base")
            v_t_pos = _velocity(anima, z1, t_s1, emb_a, pad)
            v_t_neg = _velocity(anima, z1, t_s1, ne0, pad)
            _contrast(v_s_pos, v_s_neg, v_t_pos, v_t_neg, con_z1)

        for arm in ARMS:
            _push(
                acc_anchor[arm][1.0],
                float(np.mean(r_pos_b[arm])),
                [float(np.mean(rn)) for rn in r_negs_b[arm]],
            )

        log.info(f"  [{ai + 1}/{len(anchors)}] {anchor} scored")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── aggregate ────────────────────────────────────────────────────────────
    sum_xt = {
        a: {pl: {s: _agg(acc_xt[a][pl][s]) for s in sigma_grid} for pl in pools}
        for a in ARMS
    }
    sum_anchor = {a: _agg(acc_anchor[a][1.0]) for a in ARMS}
    sum_con_xt = {
        s: {
            "ratio": float(np.mean(c["ratio"])) if c["ratio"] else None,
            "cos": float(np.mean(c["cos"])) if c["cos"] else None,
            "n": len(c["ratio"]),
        }
        for s, c in con_xt.items()
    }
    sum_con_z1 = {
        "ratio": float(np.mean(con_z1["ratio"])) if con_z1["ratio"] else None,
        "cos": float(np.mean(con_z1["cos"])) if con_z1["cos"] else None,
        "n": len(con_z1["ratio"]),
    }

    # ── pre-registered gate ──────────────────────────────────────────────────
    reasons: list[str] = []
    gate_sigmas = [s for s in sigma_grid if s >= 0.75]
    for s in gate_sigmas:
        r1 = sum_xt["student"]["shuffled"][s]["rank1"]
        if r1 is not None and r1 < args.gate_abs_rank1:
            reasons.append(
                f"student shuffled rank@1 {r1:.3f} < {args.gate_abs_rank1} at σ={s}"
            )
    for pl in pools:
        for s in sigma_grid:
            rs = sum_xt["student"][pl][s]["rank1"]
            rb = sum_xt["base"][pl][s]["rank1"]
            if rs is not None and rb is not None and (rb - rs) > args.gate_rel_drop:
                reasons.append(
                    f"student {pl} rank@1 {rs:.3f} is {rb - rs:.3f} below base "
                    f"({rb:.3f}) at σ={s}"
                )
    all_ratios = [v for c in con_xt.values() for v in c["ratio"]] + con_z1["ratio"]
    all_cos = [v for c in con_xt.values() for v in c["cos"]] + con_z1["cos"]
    if all_ratios:
        mr, mc = float(np.mean(all_ratios)), float(np.mean(all_cos))
        if mr < args.gate_ratio and mc < args.gate_cos:
            reasons.append(
                f"contrast ratio {mr:.3f} < {args.gate_ratio} with "
                f"cosine {mc:.3f} < {args.gate_cos}"
            )
    degraded = bool(reasons)
    verdict = "DEGRADATION (Phase 1 unlocked)" if degraded else "NO-DEGRADATION (STOP)"

    # 0b is reported, not gated (the proposal's gate is the 0a/0c trio) — but a
    # big student-below-base drop on the trained quantity deserves a flag.
    cautions: list[str] = []
    rb_b, rs_b = sum_anchor["base"]["rank1"], sum_anchor["student"]["rank1"]
    if rb_b is not None and rs_b is not None and (rb_b - rs_b) > args.gate_rel_drop:
        cautions.append(
            f"anchor-ranking (0b): student rank@1 {rs_b:.3f} is "
            f"{rb_b - rs_b:.3f} below base ({rb_b:.3f}) at t=1"
        )

    # ── render markdown ──────────────────────────────────────────────────────
    L = ["# Turbo caption-ranking probe (Phase 0)\n"]
    L.append(
        f"- adapter: `{args.adapter}`\n"
        f"- anchors: **{len(anchors)}** · k={k} negatives/pool · {n_seeds} noise "
        f"draws averaged · σ grid {sigma_grid}\n"
        f"- chance rank@1 = 1/(k+1) = **{chance:.3f}** · frozen base reference "
        f"(run 20260529-1157-phase0-agsm): shuffled 0.993 / hard 0.958\n"
        f"- distill grids: student σ={['%.3f' % s for s in student_sigmas]}, "
        f"anchor t_k={t_k_anchor:.4f}, teacher CFG={args.teacher_cfg}\n"
        f"- **verdict: {verdict}**\n"
    )
    for r in reasons:
        L.append(f"  - GATE: {r}")
    for c in cautions:
        L.append(f"  - CAUTION: {c}")
    if index is not None:
        L.append(
            f"\nhard pool short (<k strict siblings) for {n_hard_skipped}/"
            f"{len(anchors)} anchors → those drop from the hard pool only.\n"
        )

    L.append("\n## 0a — renoised-real ranking (x_t from cached latents)\n")
    for pl in pools:
        L.append(f"\n### {pl} negatives\n")
        L.append("| σ | base rank@1 | student rank@1 | base margin | student margin |")
        L.append("|---|---|---|---|---|")
        for s in sigma_grid:
            b, st = sum_xt["base"][pl][s], sum_xt["student"][pl][s]
            L.append(
                f"| {s:.2f} | {_fmt(b['rank1'])} | {_fmt(st['rank1'])} | "
                f"{_fmt(b['mmean'], '+.4f')} | {_fmt(st['mmean'], '+.4f')} |"
            )

    L.append("\n## 0b — anchor ranking at t=1 (on-trajectory, shuffled pool)\n")
    L.append("| arm | rank@1 | margin vs best neg | margin vs mean neg | n |")
    L.append("|---|---|---|---|---|")
    for arm in ARMS:
        sa = sum_anchor[arm]
        L.append(
            f"| {arm} | {_fmt(sa['rank1'])} | {_fmt(sa['mbest'], '+.4f')} | "
            f"{_fmt(sa['mmean'], '+.4f')} | {sa['n']} |"
        )
    L.append(
        "\nReward target is the teacher K-step CFG anchor velocity "
        "`(ε − z_tk)/(1 − t_k)` — the exact quantity `div_loss` trains. The base "
        "arm calibrates how separable this target is for the backbone itself.\n"
    )

    L.append("\n## 0c — caption-contrast transfer (student vs CFG'd teacher)\n")
    L.append("| state | ratio ‖ΔS‖/‖ΔT‖ | cosine(ΔS, ΔT) | n |")
    L.append("|---|---|---|---|")
    for s in sorted(contrast_set):
        c = sum_con_xt[s]
        L.append(
            f"| x_t @ σ={s:.2f} | {_fmt(c['ratio'])} | {_fmt(c['cos'])} | {c['n']} |"
        )
    L.append(
        f"| z1 @ σ={s1:.2f} | {_fmt(sum_con_z1['ratio'])} | "
        f"{_fmt(sum_con_z1['cos'])} | {sum_con_z1['n']} |"
    )
    L.append(
        "\nratio ≈ 1 with high cosine ⇒ the distilled student kept the teacher's "
        "(CFG-amplified) text-conditioning gain. ratio ≪ 1 ⇒ text channel "
        "attenuated even if ranking survives — the failure 0a can't see.\n"
    )

    L.append("\n## Reading it\n")
    L.append(
        "- Ranking is trustworthy here by construction: the whole grid sits at "
        "σ ≥ 0.45 where the reward premise is validated "
        "(`docs/findings/agsm_reward_premise_holds.md`), and the student's "
        "operating σ's are exactly the gated band.\n"
        "- **NO-DEGRADATION** ⇒ write the negative finding and stop — Phase 1 "
        "(soft-rank auxiliary) never happens.\n"
        "- **DEGRADATION** ⇒ note *where*: σ=1.0/0.97 → the step-0 soft-rank "
        "site as proposed; concentrated at σ≤0.75 instead → the site moves "
        "(re-scope, don't stretch).\n"
        "- Also run an over-distilled checkpoint before concluding — over-baking "
        "is where text response should die first.\n"
    )

    run_dir = make_run_dir("dpdmd", label=args.label or "caption-ranking")
    (run_dir / "caption_ranking.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    csv = run_dir / "caption_ranking.csv"
    with csv.open("w") as f:
        f.write("part,arm,pool,sigma,rank1,margin_best,margin_mean,n\n")
        for arm in ARMS:
            for pl in pools:
                for s in sigma_grid:
                    a = sum_xt[arm][pl][s]
                    f.write(
                        f"0a,{arm},{pl},{s},{a['rank1']},{a['mbest']},"
                        f"{a['mmean']},{a['n']}\n"
                    )
            a = sum_anchor[arm]
            f.write(
                f"0b,{arm},shuffled,1.0,{a['rank1']},{a['mbest']},{a['mmean']},"
                f"{a['n']}\n"
            )

    metrics = {
        "adapter": args.adapter,
        "n_anchors": len(anchors),
        "contrastive_k": k,
        "chance_rank1": round(chance, 4),
        "num_seeds": n_seeds,
        "sigma_grid": sigma_grid,
        "student_sigmas": student_sigmas,
        "t_k_anchor": t_k_anchor,
        "teacher_cfg": args.teacher_cfg,
        "n_hard_skipped": n_hard_skipped,
        "renoised": sum_xt,
        "anchor_ranking": sum_anchor,
        "contrast_xt": sum_con_xt,
        "contrast_z1": sum_con_z1,
        "gate": {
            "abs_rank1": args.gate_abs_rank1,
            "rel_drop": args.gate_rel_drop,
            "ratio": args.gate_ratio,
            "cos": args.gate_cos,
            "reasons": reasons,
            "cautions": cautions,
        },
        "degraded": degraded,
        "verdict": verdict,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["caption_ranking.md", "caption_ranking.csv"],
        device=device,
    )

    log.info("\n" + "=" * 70)
    log.info(f"  Turbo caption-ranking probe → {run_dir}")
    for pl in pools:
        for arm in ARMS:
            r1s = [
                sum_xt[arm][pl][s]["rank1"]
                for s in sigma_grid
                if sum_xt[arm][pl][s]["rank1"] is not None
            ]
            mean_r1 = float(np.mean(r1s)) if r1s else None
            log.info(f"  [0a {arm:7s}] {pl:8s}: rank@1 (σ-mean) {_fmt(mean_r1)}")
    for arm in ARMS:
        log.info(f"  [0b {arm:7s}] anchor-rank@1 {_fmt(sum_anchor[arm]['rank1'])}")
    log.info(
        f"  [0c] ratio (all states) {_fmt(float(np.mean(all_ratios)) if all_ratios else None)}  "
        f"cos {_fmt(float(np.mean(all_cos)) if all_cos else None)}"
    )
    log.info(f"  VERDICT: {verdict}")
    for r in reasons:
        log.info(f"    - {r}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
