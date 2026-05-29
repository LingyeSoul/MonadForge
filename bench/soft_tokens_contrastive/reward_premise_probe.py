"""AGSM reward-premise probe (Phase 0 of docs/proposal/soft_tokens_agsm.md).

THE QUESTION. AGSM's entire alignment signal is the model's own denoising
likelihood — the per-candidate FM error ``r(x_t, c) = −‖v_θ(x_t, c) − v_target‖²``
that the Plackett–Luce weighting ranks across candidate captions. But Anima has a
hard, repeatedly confirmed finding that **FM-MSE does not track quality**
(``project_fm_val_loss_uninformative``). AGSM only survives if the *relative
ordering* of captions for the **same** noised latent is informative even though the
*absolute* MSE is not. This probe is the kill-check the proposal flags as "the
single most important early" one:

  > if matched does not reliably out-rank ``shuffled`` negatives, AGSM's reward
  > premise fails on Anima → stop.

This is the reward-ranking sibling of the structural ``negative_audit.py`` (which
only proves a hard negative *exists*) — it actually scores the velocities.

WHAT IT DOES (no training). For ``n`` cached anchors, at each σ on a grid and
averaged over a few noise draws, it builds the anchor's own ``x_t`` and FM target
``v_target = ε − x0`` (train.py convention), then runs a **bare-DiT forward per
candidate caption** sharing that exact ``(x_t, ε, t)`` — only ``crossattn_emb``
differs, the InfoNCE/AGSM ``extra_forwards`` contract (soft_tokens.py:843). The
candidates are the **matched** caption + ``k`` negatives drawn two ways:

  * ``shuffled`` — random other stems (the proposal's literal kill-gate).
  * ``hard`` — same-artist / different-character siblings via the caption index
    (the realistic training negative; the ``negative_audit`` 68.7%-strict pool).

Reward = ``−‖v − v_target‖²`` (mean over C·H·W — the same reduction the InfoNCE
logit uses, ``_velocities_to_logits``; τ is irrelevant to ranking and dropped).
Reported per σ and aggregated:

  * **rank@1** — fraction of anchors where matched beats *every* negative (random
    baseline = ``1/(k+1)``).
  * **margin** — matched reward minus the best (binding) negative, and minus the
    mean negative, in units of the per-sample FM-MSE.

Optionally repeats the whole thing with a trained soft-tokens bank spliced in
(``--adapter``, the proposal's "with the current bank" arm) to see whether the
existing bank sharpens the ranking — wired exactly like inference
(``append_postfix`` per forward, generation.py:374). LoRA-off is the load-bearing
gate; the bank arm is secondary (and the shipped bank was trained with InfoNCE,
not AGSM, so read it as "does any learned bank help", not "does AGSM help").

GATE. PASS if the LoRA-off **shuffled** rank@1 (aggregated over the σ grid) clears
``--gate_rank1`` (default 0.60, well above the ``1/(k+1)`` chance line) with a
positive mean margin. The ``hard`` numbers are printed for context but do not gate
— a matched caption that beats random text but not a same-artist sibling still
satisfies the premise (the bank is what's supposed to sharpen the hard axis).

HONEST CAVEAT (the same one probe_sigma_signal carries). This measures the
*ranking* property AGSM needs, not final quality. A PASS says the reward premise is
not dead on arrival; it does not promise AGSM beats plain-FM — that is Phase 2's
CMMD A/B. A FAIL is a real stop sign.

Run from anima_lora/::

    uv run python -m bench.soft_tokens_contrastive.reward_premise_probe
    uv run python -m bench.soft_tokens_contrastive.reward_premise_probe \
        --num_samples 24 --contrastive_k 2 --num_seeds 3
    uv run python -m bench.soft_tokens_contrastive.reward_premise_probe \
        --adapter output/ckpt/anima_soft_tokens.safetensors --label withbank
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch

from bench._anima import add_common_args, build_anima
from bench._common import make_run_dir, write_result
from library.io.cache import load_cached_crossattn_emb, load_cached_latents

log = logging.getLogger("bench.soft_tokens_contrastive.reward_premise")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
DEFAULT_DATA = "post_image_dataset/lora"
DEFAULT_INDEX = "post_image_dataset/captions/caption_index.json"
# Mid-band σ where Anima still has caption-discriminable signal: x0 resolves by
# σ≈0.45 (project_sigma_signal_resolves_by_045), and at σ→1 the latent is ~pure
# noise so no caption can explain it — ranking there is expected to be chance.
DEFAULT_SIGMAS = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]

_RES_RE = re.compile(r"_(\d+)x(\d+)_anima\.npz$")


# ── cached-pair discovery (recursive — caches are nested per-artist) ──────────


def discover_pairs(data_dir: str) -> dict[str, tuple[str, str]]:
    """Map ``stem → (latent_npz, te_safetensors)`` for every paired cache.

    ``discover_bucketed_samples`` is non-recursive and the lora cache is nested
    by artist, so we rglob (mirrors probe_sigma_signal.discover_pairs). Returns
    the *whole* pool so negatives can be drawn from stems that aren't anchors.
    """
    pairs: dict[str, tuple[str, str]] = {}
    for p in sorted(Path(data_dir).rglob("*_anima.npz")):
        m = _RES_RE.search(p.name)
        if not m:
            continue
        stem = p.name[: m.start()]
        te = p.parent / f"{stem}_anima_te.safetensors"
        if te.exists():
            pairs[stem] = (str(p), str(te))
    if not pairs:
        raise SystemExit(f"no paired (latent, TE) caches under {data_dir}")
    return pairs


# ── negative sourcing ─────────────────────────────────────────────────────────


def load_index(index_path: str) -> dict | None:
    p = Path(index_path)
    if not p.exists():
        log.warning(f"caption index missing ({index_path}); hard negatives disabled")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def hard_negatives(
    anchor: str, index: dict, pool: set[str], k: int, rng: np.random.Generator
) -> list[str]:
    """Up to ``k`` same-artist / different-character siblings of ``anchor``.

    The option-(c) "strict" hard negative of negative_audit.py: shares an artist
    (style fixed) but carries a non-empty, disjoint character set (content
    differs). Restricted to stems that have a cached (latent, TE) pair. Returns
    [] when none qualify (orphan artist, untagged, or none cached).
    """
    meta = index["image_meta"]
    groups = index["groups"]
    am = meta.get(anchor)
    if am is None:
        return []
    a_chars = set(am.get("character", []))
    if not a_chars:
        return []  # untagged anchor → no genuine content contrast (lenient, skip)
    cands: set[str] = set()
    for artist in am.get("artist", []):
        cands.update(groups["artist"].get(artist, ()))
    cands.discard(anchor)
    strict = [
        s
        for s in cands
        if s in pool
        and meta.get(s, {}).get("character")
        and not (set(meta[s]["character"]) & a_chars)
    ]
    if not strict:
        return []
    rng.shuffle(strict)
    return strict[:k]


def shuffled_negatives(
    anchor: str, pool_list: list[str], k: int, rng: np.random.Generator
) -> list[str]:
    """``k`` random other stems — the proposal's literal kill-gate negative."""
    pick: list[str] = []
    while len(pick) < k and len(pick) < len(pool_list) - 1:
        s = pool_list[int(rng.integers(len(pool_list)))]
        if s != anchor and s not in pick:
            pick.append(s)
    return pick


# ── embedding cache (load each stem's crossattn_emb at most once) ─────────────


class EmbCache:
    def __init__(self, pairs: dict[str, tuple[str, str]]):
        self._pairs = pairs
        self._cache: dict[str, torch.Tensor | None] = {}

    def get(self, stem: str) -> torch.Tensor | None:
        if stem not in self._cache:
            te = self._pairs.get(stem)
            self._cache[stem] = (
                load_cached_crossattn_emb(te[1]) if te is not None else None
            )
        return self._cache[stem]


# ── scoring ────────────────────────────────────────────────────────────────────


def _fmt(v: float | None, p: str = ".3f") -> str:
    """Markdown-cell formatter: em-dash for missing, else ``f"{v:{p}}"``."""
    return "—" if v is None else f"{v:{p}}"


def fm_reward(
    anima,
    soft_tokens_net,
    noisy: torch.Tensor,  # (1,C,1,H,W) dtype
    t: torch.Tensor,  # (1,) dtype
    v_target: torch.Tensor,  # (1,C,1,H,W) float
    emb: torch.Tensor,  # (1,S,D) dtype
    pad: torch.Tensor,  # (1,1,H,W) dtype
) -> float:
    """``−mean((v − v_target)²)`` for one candidate caption (the InfoNCE logit's
    reward, τ dropped). Splices the soft-tokens bank first when present, exactly
    as the inference loop does (generation.py:374)."""
    if soft_tokens_net is not None:
        # Real (unpadded) text length — the front_of_padding splice scatters the
        # K tokens at [seqlen, seqlen+K), so a full-length value would index past
        # the sequence (device-side assert). Cached emb is max-padded with
        # zero-rows at the tail, so seqlen = count of non-zero rows; clamp so the
        # K-token window can't overflow S.
        S = emb.shape[1]
        K = soft_tokens_net.num_tokens
        nz = (emb.float().abs().sum(dim=-1) > 0).sum(dim=1)  # (B,)
        seqlens = nz.clamp(max=S - K).to(torch.long)
        soft_tokens_net.append_postfix(emb, seqlens, timesteps=t)
    with torch.no_grad():
        v = anima(noisy, t, emb, padding_mask=pad).float()
    return -float(((v - v_target) ** 2).mean())


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
        help="Optional trained soft-tokens checkpoint. Default: bare base DiT "
        "(the load-bearing kill-gate arm). With a bank, the probe also reports a "
        "'bank-on' arm spliced exactly like inference.",
    )
    ap.add_argument("--num_samples", type=int, default=16, help="number of anchors")
    ap.add_argument(
        "--contrastive_k",
        type=int,
        default=2,
        help="negatives per anchor per pool (matches contrastive_k; chance "
        "rank@1 = 1/(k+1)).",
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
        "--gate_rank1",
        type=float,
        default=0.60,
        help="PASS threshold on LoRA-off shuffled rank@1 (σ-grid mean).",
    )
    add_common_args(p := ap)
    args = p.parse_args()

    sigma_grid = sorted(float(s) for s in args.sigmas)
    k = int(args.contrastive_k)
    chance = 1.0 / (k + 1)
    log.info(f"σ grid: {sigma_grid}  |  k={k}  (chance rank@1 = {chance:.3f})")

    pairs = discover_pairs(args.data_dir)
    pool_list = sorted(pairs)
    log.info(f"{len(pool_list)} cached (latent, TE) pairs under {args.data_dir}")
    index = load_index(args.index)

    rng = np.random.default_rng(args.seed)
    anchor_idx = rng.choice(
        len(pool_list), size=min(args.num_samples, len(pool_list)), replace=False
    )
    anchors = [pool_list[int(i)] for i in anchor_idx]

    bundle = build_anima(args, adapter=None, train_mode=False)
    anima, device, dtype = bundle.anima, bundle.device, bundle.dtype

    # Optional bank arm — soft_tokens uses its OWN factory + splice (not the LoRA
    # one build_anima drives), so load it by hand and apply_to the live DiT.
    soft_tokens_net = None
    if args.adapter is not None:
        from networks.methods.soft_tokens import create_network_from_weights

        soft_tokens_net, _sd = create_network_from_weights(
            1.0, args.adapter, None, None, anima, for_inference=True
        )
        log.info(
            f"bank splice_position={soft_tokens_net.splice_position!r} "
            "(seqlens derived from non-zero rows of the cached emb)"
        )
        soft_tokens_net.apply_to(text_encoders=None, unet=anima)
        soft_tokens_net.load_weights(args.adapter)
        soft_tokens_net.to(device=device, dtype=dtype)
        soft_tokens_net.eval()
        log.info(f"bank loaded: {args.adapter} (K={soft_tokens_net.num_tokens})")

    embs = EmbCache(pairs)

    # arms: "base" always; "bank" only when an adapter is loaded. Each forward in
    # the bank arm splices via append_postfix; the base arm passes None.
    arms = {"base": None}
    if soft_tokens_net is not None:
        arms["bank"] = soft_tokens_net

    # accumulators[arm][pool][σ] = list over anchors of {rank1, margin_best, margin_mean}
    pools = ["shuffled"] + (["hard"] if index is not None else [])
    acc = {
        a: {
            pl: {s: {"rank1": [], "mbest": [], "mmean": []} for s in sigma_grid}
            for pl in pools
        }
        for a in arms
    }
    n_hard_skipped = 0

    for ai, anchor in enumerate(anchors):
        npz_path, _te = pairs[anchor]
        lat, _res, _oh, _ow = load_cached_latents(npz_path)
        anchor_emb = embs.get(anchor)
        if anchor_emb is None:
            log.warning(f"  anchor {anchor}: no crossattn_emb, skipping")
            continue

        # Build per-pool negative caption lists (deterministic per anchor).
        nrng = np.random.default_rng(args.seed + 1000 + ai)
        neg_stems = {"shuffled": shuffled_negatives(anchor, pool_list, k, nrng)}
        if index is not None:
            hn = hard_negatives(anchor, index, set(pairs), k, nrng)
            if len(hn) < k:
                n_hard_skipped += 1  # thin/absent strict pool for this anchor
            neg_stems["hard"] = hn

        # Pre-load candidate embeddings; drop pools that came up short.
        cand_embs = {}
        for pl in pools:
            es = [embs.get(s) for s in neg_stems[pl]]
            es = [e for e in es if e is not None]
            cand_embs[pl] = es if len(es) == k else None

        x0 = lat.to(device, dtype).unsqueeze(0).unsqueeze(2)  # (1,C,1,H,W)
        H, W = x0.shape[-2], x0.shape[-1]
        pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)
        x0_f = x0.float()
        emb_a = anchor_emb.to(device, dtype).unsqueeze(0)

        for s in sigma_grid:
            # seed-averaged per-candidate reward → one ranking decision per σ.
            for arm_name, net in arms.items():
                for pl in pools:
                    if cand_embs[pl] is None:
                        continue
                    r_pos, r_negs = [], [[] for _ in range(k)]
                    for sj in range(max(1, args.num_seeds)):
                        g = torch.Generator(device=device).manual_seed(
                            args.seed + ai * 1000 + sj
                        )
                        eps = torch.randn(
                            x0.shape, generator=g, device=device, dtype=dtype
                        )
                        v_target = eps.float() - x0_f
                        noisy = ((1.0 - s) * x0_f + s * eps.float()).to(dtype)
                        t = torch.full((1,), float(s), device=device, dtype=dtype)
                        r_pos.append(
                            fm_reward(anima, net, noisy, t, v_target, emb_a, pad)
                        )
                        for j, ne in enumerate(cand_embs[pl]):
                            r_negs[j].append(
                                fm_reward(
                                    anima,
                                    net,
                                    noisy,
                                    t,
                                    v_target,
                                    ne.to(device, dtype).unsqueeze(0),
                                    pad,
                                )
                            )
                    pos = float(np.mean(r_pos))
                    negs = [float(np.mean(rn)) for rn in r_negs]
                    best_neg = max(negs)
                    a = acc[arm_name][pl][s]
                    a["rank1"].append(1.0 if pos > best_neg else 0.0)
                    a["mbest"].append(pos - best_neg)
                    a["mmean"].append(pos - float(np.mean(negs)))
        log.info(f"  [{ai + 1}/{len(anchors)}] {anchor} scored")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── aggregate ───────────────────────────────────────────────────────────
    def agg(arm: str, pl: str, key: str) -> dict:
        per_sigma = {
            s: (float(np.mean(acc[arm][pl][s][key])) if acc[arm][pl][s][key] else None)
            for s in sigma_grid
        }
        vals = [v for v in per_sigma.values() if v is not None]
        return {"per_sigma": per_sigma, "mean": float(np.mean(vals)) if vals else None}

    summary = {}
    for arm in arms:
        summary[arm] = {}
        for pl in pools:
            summary[arm][pl] = {
                "rank1": agg(arm, pl, "rank1"),
                "margin_best": agg(arm, pl, "mbest"),
                "margin_mean": agg(arm, pl, "mmean"),
                "n_anchors": len(acc[arm][pl][sigma_grid[0]]["rank1"]),
            }

    base_shuf_r1 = summary["base"]["shuffled"]["rank1"]["mean"] or 0.0
    base_shuf_mm = summary["base"]["shuffled"]["margin_mean"]["mean"] or 0.0
    verdict = (
        "PASS" if base_shuf_r1 >= args.gate_rank1 and base_shuf_mm > 0.0 else "FAIL"
    )

    # ── render markdown ──────────────────────────────────────────────────────
    L = ["# AGSM reward-premise probe (Phase 0)\n"]
    L.append(
        "Does Anima's own FM error `−‖v−v_target‖²` rank the **matched** caption "
        "above mismatched captions for the *same* noised latent? AGSM's "
        "Plackett–Luce reward needs this even though absolute FM-MSE is "
        "uninformative (`project_fm_val_loss_uninformative`).\n"
    )
    L.append(
        f"- anchors: **{len(anchors)}** · k={k} negatives/pool · "
        f"{max(1, args.num_seeds)} noise draws averaged · σ grid {sigma_grid}\n"
        f"- chance rank@1 = 1/(k+1) = **{chance:.3f}**\n"
        f"- gate (LoRA-off shuffled rank@1 ≥ {args.gate_rank1:.2f} & margin>0): "
        f"**{verdict}**\n"
    )
    if index is not None:
        L.append(
            f"- hard-negative pool came up short (<k strict siblings) for "
            f"{n_hard_skipped}/{len(anchors)} anchors → those drop from the hard "
            f"pool only.\n"
        )

    for arm in arms:
        L.append(
            f"\n## Arm: {arm}"
            + (" (bare base DiT)" if arm == "base" else " (with trained bank)")
            + "\n"
        )
        for pl in pools:
            srow = summary[arm][pl]
            L.append(f"\n### {pl} negatives  (n={srow['n_anchors']})\n")
            L.append("| σ | rank@1 | margin vs best neg | margin vs mean neg |")
            L.append("|---|---|---|---|")
            for s in sigma_grid:
                r1 = srow["rank1"]["per_sigma"][s]
                mb = srow["margin_best"]["per_sigma"][s]
                mm = srow["margin_mean"]["per_sigma"][s]
                L.append(
                    f"| {s:.2f} | {_fmt(r1)} | {_fmt(mb, '+.4f')} | {_fmt(mm, '+.4f')} |"
                )
            L.append(
                f"| **mean** | **{_fmt(srow['rank1']['mean'])}** | "
                f"**{_fmt(srow['margin_best']['mean'], '+.4f')}** | "
                f"**{_fmt(srow['margin_mean']['mean'], '+.4f')}** |"
            )

    L.append("\n## Reading it\n")
    L.append(
        "- **rank@1 > chance with positive margin** at some σ band ⇒ the reward "
        "premise holds — matched text explains the anchor's latent better than "
        "mismatched. AGSM is not dead on arrival.\n"
        "- **rank@1 ≈ chance / negative margin everywhere** ⇒ the premise FAILS "
        "on Anima — the PL reward is noise, stop before Phase 2.\n"
        "- **shuffled** is the kill-gate; **hard** (same-artist/diff-character) is "
        "the realistic training negative — expect it harder (lower rank@1). A bank "
        "arm that lifts hard rank@1 over base is the signal a learned bank sharpens "
        "the axis AGSM trains.\n"
        "- Empirically the margin GROWS with σ (caption-conditioning matters most "
        "when x_t is mostly noise and the model must guess x0 from text); the weak "
        "band is LOW σ, where the near-clean latent already determines the velocity "
        "and all captions score alike. Informative band = mid/high σ (≥0.45, where "
        "x0 resolves — `project_sigma_signal_resolves_by_045`).\n"
        "- CAVEAT: this is the *ranking* property AGSM needs, NOT final quality. "
        "PASS ≠ AGSM beats plain-FM (that's Phase 2's CMMD A/B); FAIL is a real "
        "stop sign.\n"
    )

    run_dir = make_run_dir(
        "soft_tokens_contrastive", label=args.label or "reward-premise"
    )
    (run_dir / "reward_premise.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # per-σ CSV (long form, all arms/pools).
    csv = run_dir / "reward_premise.csv"
    with csv.open("w") as f:
        f.write("arm,pool,sigma,rank1,margin_best,margin_mean,n\n")
        for arm in arms:
            for pl in pools:
                srow = summary[arm][pl]
                for s in sigma_grid:
                    f.write(
                        f"{arm},{pl},{s},"
                        f"{srow['rank1']['per_sigma'][s]},"
                        f"{srow['margin_best']['per_sigma'][s]},"
                        f"{srow['margin_mean']['per_sigma'][s]},"
                        f"{srow['n_anchors']}\n"
                    )

    metrics = {
        "n_anchors": len(anchors),
        "contrastive_k": k,
        "chance_rank1": round(chance, 4),
        "num_seeds": int(max(1, args.num_seeds)),
        "sigma_grid": sigma_grid,
        "gate_rank1": args.gate_rank1,
        "adapter": args.adapter,
        "n_hard_skipped": n_hard_skipped,
        "summary": summary,
        "base_shuffled_rank1_mean": round(base_shuf_r1, 4),
        "base_shuffled_margin_mean": round(base_shuf_mm, 6),
        "verdict": verdict,
        "note": (
            "Ranking property AGSM needs, not quality "
            "(project_fm_val_loss_uninformative). shuffled gates; hard is context. "
            "PASS != AGSM beats plain-FM (Phase 2 CMMD A/B)."
        ),
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["reward_premise.md", "reward_premise.csv"],
        device=device,
    )

    log.info("\n" + "=" * 70)
    log.info(f"  AGSM reward-premise probe → {run_dir}")
    for arm in arms:
        for pl in pools:
            srow = summary[arm][pl]
            r1 = srow["rank1"]["mean"]
            mm = srow["margin_mean"]["mean"]
            log.info(
                f"  [{arm:4s}] {pl:8s}: rank@1 (σ-mean) "
                f"{('%.3f' % r1) if r1 is not None else '  — '}  "
                f"margin_mean {('%+.4f' % mm) if mm is not None else '  — '}  "
                f"(chance {chance:.3f})"
            )
    log.info(
        f"  GATE: LoRA-off shuffled rank@1={base_shuf_r1:.3f} "
        f"(≥{args.gate_rank1:.2f}?) margin={base_shuf_mm:+.4f} (>0?) → {verdict}"
    )
    log.info("  open: reward_premise.md")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
