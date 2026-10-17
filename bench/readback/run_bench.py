#!/usr/bin/env python
"""Phase 0a — tag read-back validity (gate; no training, no new models).

Validates the primitive in ``anime_tools.tagger.readback`` before any
selection/RWR consumer is allowed to trust it. Everything here scores *cached*
tagger PE features over the tagger's own held-out split — no re-encoding, no
renders, seconds on one GPU.

Two controls run entirely on real data (the make-or-break gate axis):

  * SHUFFLED-CAPTION CONTROL (proposal gate ii): per image, does read-back of
    its TRUE content-tag set beat read-back of a random OTHER image's content-tag
    set? Reported as a per-image win-rate (gate: drop under shuffle for >= 90% of
    images) and a population AUROC. This is the base-rate / language-prior
    robustness test — the axis where the per-caption prior does NOT cancel.
  * CROSS-CAPTION DISCRIMINATION: the full N x N read-back matrix R[i, j] =
    readback(image_i, caption_j). Row-wise recall@1 (is the true caption the
    argmax over all candidates for this image?) and column-wise recall@1 (the
    paper's group-relative form: fix the caption, is the true image best of N?).

Both aggregations are reported (mean log σ vs calibrated recall — the proposal's
open question, "pick by gate AUC"), and broken down by content-tag category set
(all content vs general-only). Render-based validity (the in-axis cross-prompt
discrimination on GENERATED images, plus the turbo teacher/student + checkpoint
pair sets) is added by passing ``--render_manifest`` — a manifest produced by
``bench/readback/render_turbo.py`` (which renders the arms). NB the turbo
teacher-gap is a text/pose phenomenon; content-tag read-back is orthogonal to it
by design, so its gate-(i) AUROC is a scope diagnostic, not a gate on the
instrument (see that section of the summary).

READOUT: bench/readback/results/<ts>[-label]/{controls.csv, summary.md,
         result.json}

Run from anima_lora/::

    uv run python bench/readback/run_bench.py --label v3-refit
    uv run python bench/readback/run_bench.py --split train --num_images 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from anime_tools.tagger.readback import (  # noqa: E402
    AGG_LOGSIGMOID,
    AGG_RECALL,
    CONTENT_CATEGORIES,
    TagReadback,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--model_dir",
        default="models/captioners/anima-tagger-dbv4",
        help="Tagger checkpoint dir (model.safetensors + config.json + vocab.json "
        "+ dataset.json [+ thresholds.safetensors]).",
    )
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument(
        "--num_images",
        type=int,
        default=0,
        help="Cap images scored (0 = whole split). The N x N matrix is O(N^2) in "
        "memory; val (~789) is fine, cap large train splits.",
    )
    p.add_argument(
        "--feature_cache_dir",
        default=None,
        help="Feature-cache root override (default: post_image_dataset/anima_tagger).",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--feature_cache_workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default=None, help="Run-dir label suffix.")
    p.add_argument(
        "--render_manifest",
        default=None,
        help="Optional render_manifest.json from render_turbo.py (live path). "
        "Enables gate (i) [teacher>student pair AUROC] and gate (iii) [turbo "
        "checkpoint family spread].",
    )
    p.add_argument(
        "--gate_win_rate",
        type=float,
        default=0.90,
        help="Pre-registered shuffled-caption drop-rate gate (default 0.90).",
    )
    p.add_argument(
        "--gate_auc",
        type=float,
        default=0.80,
        help="Pre-registered per-pair-set AUC gate (default 0.80).",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Cached forward off the sidecar trainer's dbv4 feature cache
# --------------------------------------------------------------------------- #
def collect_logits(
    args: argparse.Namespace, model_dir: Path, cfg_d: dict, split_stems: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.device]:
    """Forward ``split_stems`` through the tagger head off cached features.

    Returns ``(tag_logits [N, n_tags] fp32 cpu, multi_hot [N, n_tags] cpu,
    device)`` in loader-emission order — the two rows always align.
    """
    if cfg_d.get("backend") != "dbv4":
        raise SystemExit(
            "bench/readback: only dbv4-backed tagger checkpoints are supported "
            "(the PE dual-encoder head was removed 2026-08-30)."
        )
    return _collect_logits_dbv4(args, model_dir, cfg_d, split_stems)


def _collect_logits_dbv4(
    args: argparse.Namespace, model_dir: Path, cfg_d: dict, split_stems: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.device]:
    """dbv4 backend: logits off the sidecar trainer's feature cache.

    ``train_sidecar.py`` caches the projected our-vocab probs + the MLP-head
    hidden feature for every ``dataset.json`` stem; sidecar rows are recomputed
    from the hidden feature so the logits match ``AnimaTagger.tag_logits``.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file as st_load

    from anime_tools.tagger.dbv4_backend import (
        UNSUPPORTED_LOGIT,
        SidecarHead,
        probs_to_logits,
    )

    device = torch.device(args.device or "cpu")
    arch = cfg_d["dbv4"]["arch"]
    cache_path = (
        Path(args.feature_cache_dir or "post_image_dataset/anima_tagger")
        / "dbv4"
        / f"{arch}_hidden.safetensors"
    )
    if not cache_path.exists():
        raise SystemExit(
            f"missing {cache_path} — run python -m anime_tools.tagger.cli.train_sidecar "
            f"--cache_only first"
        )
    with safe_open(str(cache_path), "pt") as f:
        cached_stems = json.loads(f.metadata()["stems"])
    cache = st_load(str(cache_path))
    pos = {s: i for i, s in enumerate(cached_stems)}
    rows = torch.tensor(
        [pos[s] for s in split_stems if s in pos and cache["ok"][pos[s]]]
    )
    probs = cache["probs"][rows].float()
    n_tags = probs.shape[1]
    with open(model_dir / "config.json") as f:
        n_sup = json.load(f)["alignment"]
    supported = torch.zeros(n_tags, dtype=torch.bool)
    unmatched = {u["index"] for u in n_sup["unmatched"]}
    supported[[i for i in range(n_tags) if i not in unmatched]] = True
    head = SidecarHead.load(model_dir)
    if head is not None:
        with torch.no_grad():
            bce, _ = head(cache["hidden"][rows].float())
        idx = torch.tensor(head.bce_indices)
        probs[:, idx] = bce.sigmoid()
        supported[idx] = True
    logits = probs_to_logits(probs)
    logits[:, ~supported] = UNSUPPORTED_LOGIT
    with open(model_dir / "dataset.json") as f:
        ds = json.load(f)
    tag_idx = dict(zip(ds["stems"], ds["tag_indices"]))
    kept_stems = [cached_stems[i] for i in rows.tolist()]
    mh = torch.zeros(len(kept_stems), n_tags)
    for n, s in enumerate(kept_stems):
        mh[n, tag_idx[s]] = 1.0
    return logits, mh, device


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def roc_auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """AUROC = P(pos > neg), proper tie handling (the recall agg ties heavily)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    pos = pos[~torch.isnan(pos)].numpy()
    neg = neg[~torch.isnan(neg)].numpy()
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    s = np.concatenate([pos, neg])
    return float(roc_auc_score(y, s))


def matrix_metrics(R: torch.Tensor, seed: int) -> Dict[str, float]:
    """Diagonal-vs-off-diagonal metrics on an N x N read-back matrix.

    ``R[i, j]`` = readback(image_i, caption_j); the diagonal is the true pairing.
    A row/column can be all-nan when an image has no content tags under the
    active category set — those are dropped from every reduction.
    """
    n = R.shape[0]
    diag = R.diagonal().clone()  # [N] true-caption scores
    off_mask = ~torch.eye(n, dtype=torch.bool)
    R_off = R.masked_fill(~off_mask, float("nan"))  # diagonal -> nan

    # Shuffled-caption control: true vs one random OTHER caption per image.
    g = torch.Generator().manual_seed(seed)
    order = torch.stack([torch.randperm(n, generator=g) for _ in range(n)])  # [N,N]
    rand_j = order[:, 0]
    rand_j = torch.where(rand_j == torch.arange(n), order[:, 1], rand_j)  # avoid self
    shuffled = R[torch.arange(n), rand_j]
    valid_s = ~(torch.isnan(diag) | torch.isnan(shuffled))
    win_rate = (
        float((diag[valid_s] > shuffled[valid_s]).float().mean())
        if valid_s.any()
        else float("nan")
    )

    # Row-wise recall@1: true caption is the best of all candidates for image i.
    row_off_max = torch.nan_to_num(R_off, nan=float("-inf")).max(dim=1).values
    valid_r = ~torch.isnan(diag) & torch.isfinite(row_off_max)
    recall1_row = (
        float((diag[valid_r] >= row_off_max[valid_r]).float().mean())
        if valid_r.any()
        else float("nan")
    )
    # mean fraction of other captions the true caption strictly beats, per image
    beat = (diag.unsqueeze(1) > R_off).float()
    beat[torch.isnan(R_off)] = float("nan")
    beat_frac = torch.nanmean(beat, dim=1)
    mean_beat_frac = float(torch.nanmean(beat_frac))

    # Column-wise recall@1 (paper's group-relative form): true image best of N.
    col_off_max = torch.nan_to_num(R_off, nan=float("-inf")).max(dim=0).values
    valid_c = ~torch.isnan(diag) & torch.isfinite(col_off_max)
    recall1_col = (
        float((diag[valid_c] >= col_off_max[valid_c]).float().mean())
        if valid_c.any()
        else float("nan")
    )

    auroc = roc_auc(diag, R_off.reshape(-1))
    return {
        "n": float(n),
        "shuffled_win_rate": win_rate,
        "recall1_row": recall1_row,
        "recall1_col": recall1_col,
        "mean_beat_frac": mean_beat_frac,
        "auroc_true_vs_random": auroc,
        "diag_mean": float(torch.nanmean(diag)),
        "offdiag_mean": float(torch.nanmean(R_off)),
    }


# --------------------------------------------------------------------------- #
# Optional render-pair-manifest (live path) — the proposal's known-ranking sets
# --------------------------------------------------------------------------- #
# Render-manifest (live path) — the proposal's known-ranking pair sets.
# Format is emitted by bench/readback/render_turbo.py:
#   {"prompts": {pi: caption}, "seeds": [...],
#    "arms": {name: {"rank": int|None, "family": str|None, "step": int|None,
#                    "renders": {"p{pi}_s{si}": abs_path}}}}
# --------------------------------------------------------------------------- #
def score_render_manifest(
    rb: TagReadback, manifest: dict, agg: str
) -> Dict[str, object]:
    """Score every render, then compute the render-based validity metrics.

    RENDER CROSS-PROMPT DISCRIMINATION (the in-axis test — the real deployment
    surface): for every render, read-back of its TRUE prompt vs the other prompts
    in the set. The real-data control proves the instrument on training images;
    this proves it survives the shift to GENERATED images, per generator arm.

    Gate (i): for each arm carrying an integer ``rank`` (lower == should adhere
    better, teacher=0, student=1), within each (prompt, seed) cell compare
    read-back across ranked arms → pooled ordered-pair AUROC. NOTE the turbo
    teacher-gap is a text/pose phenomenon; content-tag read-back is near-blind to
    it by design, so a chance-level AUROC here is a scope boundary, not a failure.

    Gate (iii): for each ``family`` (turbo step checkpoints), per-arm mean
    read-back and the between-checkpoint spread vs same-prompt seed jitter.
    """
    from PIL import Image

    prompts: Dict[str, str] = manifest["prompts"]
    arms: Dict[str, dict] = manifest["arms"]
    pids = sorted(prompts.keys(), key=int)
    # content-tag caption mask per prompt, stacked [P, n_tags]
    pmasks = torch.stack(
        [
            rb.caption_mask(
                [t.strip() for t in prompts[pid].replace("_", " ").split(",")]
            )
            for pid in pids
        ]
    )
    pid_to_col = {pid: j for j, pid in enumerate(pids)}

    # score[arm][cell] = read-back of that render against its OWN prompt (diagonal)
    scored: Dict[str, Dict[str, float]] = {}
    # render_disc[arm] = cross-prompt discrimination on that arm's renders
    render_disc: Dict[str, dict] = {}
    for arm_name, arm in arms.items():
        cell_scores: Dict[str, float] = {}
        true_s: List[float] = []  # render vs its true prompt
        rand_s: List[float] = []  # render vs one random other prompt
        r1_hits = 0
        n_r = 0
        g = torch.Generator().manual_seed(1234)
        for cell, path in arm["renders"].items():
            pid = cell.split("_")[0][1:]  # "p3_s1" -> "3"
            logits = rb.image_logits(Image.open(path)).unsqueeze(0)  # [1, n_tags]
            row = rb.readback_from_logits(logits, pmasks, agg=agg)[
                0
            ]  # [P] vs all prompts
            j = pid_to_col[pid]
            cell_scores[cell] = float(row[j])
            true_v = row[j]
            others = torch.cat([row[:j], row[j + 1 :]])
            cell_scores[cell] = float(true_v)
            # recall@1 over prompts + true-vs-random-other
            if not torch.isnan(true_v):
                n_r += 1
                valid_o = others[~torch.isnan(others)]
                if valid_o.numel() and float(true_v) >= float(valid_o.max()):
                    r1_hits += 1
                if valid_o.numel():
                    k = int(torch.randint(0, valid_o.numel(), (1,), generator=g))
                    true_s.append(float(true_v))
                    rand_s.append(float(valid_o[k]))
        scored[arm_name] = cell_scores
        win = [1.0 for t, r in zip(true_s, rand_s) if t > r]
        render_disc[arm_name] = {
            "recall1_over_prompts": (r1_hits / n_r) if n_r else float("nan"),
            "true_vs_random_winrate": (len(win) / len(true_s))
            if true_s
            else float("nan"),
            "auroc_true_vs_other": roc_auc(torch.tensor(true_s), torch.tensor(rand_s)),
            "n_renders": n_r,
        }

    # -- gate (i): ranked-arm pairwise AUROC, pooled over cells --------------
    ranked = {n: a["rank"] for n, a in arms.items() if a.get("rank") is not None}
    pos: List[float] = []
    neg: List[float] = []
    if len(ranked) >= 2:
        cells = set.intersection(*(set(scored[n].keys()) for n in ranked))
        for cell in cells:
            for a_name, a_rank in ranked.items():
                for b_name, b_rank in ranked.items():
                    if a_rank < b_rank:  # a should score higher than b
                        pos.append(scored[a_name][cell])
                        neg.append(scored[b_name][cell])
    gate_i_auc = roc_auc(torch.tensor(pos), torch.tensor(neg)) if pos else float("nan")

    # -- gate (iii): family spread vs seed noise ----------------------------
    families: Dict[str, List[str]] = {}
    for n, a in arms.items():
        if a.get("family"):
            families.setdefault(a["family"], []).append(n)
    family_stats: Dict[str, dict] = {}
    for fam, members in families.items():
        members = sorted(members, key=lambda n: arms[n].get("step") or 0)
        # Per-arm mean over the SAME cell set → cross-prompt variance cancels.
        per_arm_mean = {n: _mean(list(scored[n].values())) for n in members}
        # Between-checkpoint spread: std of those prompt-cancelled per-arm means.
        spread = _std(list(per_arm_mean.values()))
        # Noise floor = SAME-prompt seed jitter (NOT cross-prompt variance): for
        # each (arm, prompt) the std over its seeds, averaged. This is the paired
        # reference the spread must beat to be "not noise".
        seed_std = _mean([_seed_jitter(scored[n]) for n in members])
        # Standard error of each per-arm mean (n cells) — the spread must exceed
        # this for the checkpoint means to be distinguishable.
        n_cells = _mean([float(len(scored[n])) for n in members])
        se_mean = seed_std / (n_cells**0.5) if n_cells else float("nan")
        family_stats[fam] = {
            "members": members,
            "per_arm_mean": per_arm_mean,
            "between_spread": spread,
            "within_seed_std": seed_std,
            "se_of_mean": se_mean,
            "spread_over_noise": (spread / seed_std) if seed_std else float("inf"),
        }

    return {
        "render_discrimination": render_disc,
        "gate_i_pair_auroc": gate_i_auc,
        "n_pairs": len(pos),
        "families": family_stats,
        "per_arm_mean": {n: _mean(list(s.values())) for n, s in scored.items()},
    }


def _seed_jitter(cell_scores: Dict[str, float]) -> float:
    """Same-prompt seed noise: std over seeds within each prompt, averaged.

    Removes cross-prompt variance (different tag sets score wildly differently),
    which is the correct noise floor for the gate-(iii) spread comparison.
    """
    from collections import defaultdict

    by_prompt: Dict[str, List[float]] = defaultdict(list)
    for cell, v in cell_scores.items():
        by_prompt[cell.split("_")[0]].append(v)  # "p3_s1" -> "p3"
    stds = [_std(vs) for vs in by_prompt.values() if len(vs) >= 2]
    return _mean(stds)


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: List[float]) -> float:
    xs = [x for x in xs if x == x]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json") as f:
        cfg_d = json.load(f)
    with open(model_dir / "dataset.json") as f:
        split_stems = json.load(f)["split"][args.split]
    if args.num_images and args.num_images < len(split_stems):
        # deterministic subsample
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(split_stems), generator=g)[: args.num_images]
        split_stems = [split_stems[i] for i in perm.tolist()]

    tag_logits, multi_hot, device = collect_logits(args, model_dir, cfg_d, split_stems)
    n, n_tags = tag_logits.shape
    log.info("collected logits: N=%d n_tags=%d (split=%s)", n, n_tags, args.split)

    from anime_tools.tagger.tagger import AnimaTagger

    rb = TagReadback(
        AnimaTagger(model_dir, device=device), content_categories=CONTENT_CATEGORIES
    )

    # Two content-category sets: full content vs general-only (the discriminative
    # identity tags — character/copyright — make the test easier, so report both).
    category_sets = {
        "content": CONTENT_CATEGORIES,
        "general_only": ("general",),
    }

    rows: List[dict] = []
    metrics: Dict[str, dict] = {}
    for cat_name, cats in category_sets.items():
        rb_c = TagReadback(rb.tagger, content_categories=cats)
        caption_masks = rb_c.content_multi_hot(multi_hot)  # [N, n_tags] bool
        for agg in (AGG_LOGSIGMOID, AGG_RECALL):
            R = rb_c.readback_from_logits(tag_logits, caption_masks, agg=agg)  # [N, N]
            m = matrix_metrics(R, seed=args.seed)
            key = f"{cat_name}/{agg}"
            metrics[key] = m
            rows.append({"category_set": cat_name, "agg": agg, **m})
            log.info(
                "[%s] win_rate=%.3f recall@1(row)=%.3f recall@1(col)=%.3f "
                "AUROC=%.3f (diag=%.3f off=%.3f)",
                key,
                m["shuffled_win_rate"],
                m["recall1_row"],
                m["recall1_col"],
                m["auroc_true_vs_random"],
                m["diag_mean"],
                m["offdiag_mean"],
            )

    pair_metrics: Optional[dict] = None
    if args.render_manifest:
        with open(args.render_manifest) as f:
            rman = json.load(f)
        pair_metrics = {}
        for agg in (AGG_LOGSIGMOID, AGG_RECALL):
            pm = score_render_manifest(rb, rman, agg)
            pair_metrics[agg] = pm
            for arm_name, rd in pm["render_discrimination"].items():
                log.info(
                    "[render-disc/%s] %-9s recall@1/prompts=%.3f win(true>rand)=%.3f "
                    "AUROC=%.3f (n=%d)",
                    agg,
                    arm_name,
                    rd["recall1_over_prompts"],
                    rd["true_vs_random_winrate"],
                    rd["auroc_true_vs_other"],
                    rd["n_renders"],
                )
            log.info(
                "[render/%s] gate(i) pair AUROC=%.3f (n_pairs=%d)",
                agg,
                pm["gate_i_pair_auroc"],
                int(pm["n_pairs"]),
            )
            for fam, fs in pm["families"].items():
                log.info(
                    "  gate(iii) family '%s': spread/noise=%.2f (between=%.4f, "
                    "seed_std=%.4f) means=%s",
                    fam,
                    fs["spread_over_noise"],
                    fs["between_spread"],
                    fs["within_seed_std"],
                    {k: round(v, 3) for k, v in fs["per_arm_mean"].items()},
                )

    # -- gate verdict (pre-registered) ----------------------------------------
    best_win = max(
        (
            m["shuffled_win_rate"]
            for m in metrics.values()
            if m["shuffled_win_rate"] == m["shuffled_win_rate"]
        ),
        default=float("nan"),
    )
    best_auc = max(
        (
            m["auroc_true_vs_random"]
            for m in metrics.values()
            if m["auroc_true_vs_random"] == m["auroc_true_vs_random"]
        ),
        default=float("nan"),
    )
    gate_ii = best_win >= args.gate_win_rate
    gate_disc = best_auc >= args.gate_auc
    verdict = "PASS" if (gate_ii and gate_disc) else "FAIL"

    run_dir = make_run_dir("readback", label=args.label)
    with open(run_dir / "controls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = [
        f"# tag read-back Phase 0a — {args.split} split (N={n})",
        "",
        f"verdict: **{verdict}**  (gate ii shuffled-drop >= {args.gate_win_rate}: "
        f"best win_rate={best_win:.3f} -> {'ok' if gate_ii else 'FAIL'}; "
        f"discrimination AUROC >= {args.gate_auc}: best={best_auc:.3f} -> "
        f"{'ok' if gate_disc else 'FAIL'})",
        "",
        (
            "NOTE: render-based validity (generated images) is in the section below."
            if args.render_manifest
            else "NOTE: gate (i)/(iii) render sets need a prompt/seed manifest — "
            "pass --render_manifest (see render_turbo.py) to add them."
        ),
        "",
        "| category_set | agg | win_rate | recall@1 row | recall@1 col | AUROC | diag | off |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        summary.append(
            f"| {r['category_set']} | {r['agg']} | {r['shuffled_win_rate']:.3f} | "
            f"{r['recall1_row']:.3f} | {r['recall1_col']:.3f} | "
            f"{r['auroc_true_vs_random']:.3f} | {r['diag_mean']:.3f} | "
            f"{r['offdiag_mean']:.3f} |"
        )
    if pair_metrics:
        summary += [
            "",
            "## render-based validity (--render_manifest)",
            "",
            "**In-axis test — cross-prompt discrimination on GENERATED images** "
            "(the deployment surface: read-back of each render's TRUE prompt vs "
            "the other prompts). This is the validity claim; gate (i)/(iii) below "
            "are turbo-specific and test the text/pose axis read-back is blind to.",
            "",
            "| agg | arm | recall@1/prompts | win(true>rand) | AUROC |",
            "|---|---|---|---|---|",
        ]
        for agg, pm in pair_metrics.items():
            for arm_name, rd in pm["render_discrimination"].items():
                summary.append(
                    f"| {agg} | {arm_name} | {rd['recall1_over_prompts']:.3f} | "
                    f"{rd['true_vs_random_winrate']:.3f} | "
                    f"{rd['auroc_true_vs_other']:.3f} |"
                )
        summary += [
            "",
            "### turbo pair set (scope-boundary diagnostics, NOT a gate on the instrument)",
            "",
        ]
        for agg, pm in pair_metrics.items():
            summary.append(
                f"- **{agg}** gate(i) teacher>student pair AUROC = "
                f"**{pm['gate_i_pair_auroc']:.3f}** (n_pairs={int(pm['n_pairs'])}) "
                f"— chance-level: content-tag read-back is ~orthogonal to the "
                f"turbo teacher-gap (text/pose). per-arm mean = "
                f"{ {k: round(v, 3) for k, v in pm['per_arm_mean'].items()} }"
            )
            for fam, fs in pm["families"].items():
                summary.append(
                    f"  - gate(iii) family '{fam}': spread/noise = "
                    f"**{fs['spread_over_noise']:.2f}** "
                    f"(between-ckpt std {fs['between_spread']:.4f} vs same-prompt "
                    f"seed jitter {fs['within_seed_std']:.4f}; SE-of-mean "
                    f"{fs['se_of_mean']:.4f})"
                )
    (run_dir / "summary.md").write_text("\n".join(summary) + "\n")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "verdict": verdict,
            "gate_ii_shuffled_drop": gate_ii,
            "gate_discrimination": gate_disc,
            "best_shuffled_win_rate": best_win,
            "best_auroc": best_auc,
            "controls": metrics,
            "pairs": pair_metrics,
            "content_categories": list(CONTENT_CATEGORIES),
            "n_content_tags": int(rb.content_mask.sum()),
        },
        label=args.label,
        artifacts=[run_dir / "controls.csv", run_dir / "summary.md"],
        device=str(device),
    )
    log.info("verdict: %s  ->  %s", verdict, run_dir)


if __name__ == "__main__":
    main()
