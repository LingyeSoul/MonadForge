#!/usr/bin/env python
"""Tagger ceiling / attribution probe — is the general-tag floor fixable?

``tagger_eval`` showed the deployed tagger floors on the *localized* semantic
slices (pose ~0.25 AP, expression ~0.18, gesture ~0.13, clothing_details,
body_parts, …) while identity slices (artist/character/copyright) are near
solved. Those floor slices all live in the **spatial** routing partition —
they read frozen PE-Spatial tokens through the learned MAP-pool + trunk. This
probe answers the attribution question ``tagger_eval`` cannot:

  Is the floor a HEAD/TRAINING problem (the deployed spatial branch underfits
  the frozen features — cheaply fixable via loss/longer/isolated training) or
  a FEATURE CEILING (frozen PE-Spatial simply does not encode pose/expression
  at the granularity a tagger needs — not fixable without touching the tower)?

Method — re-fit the spatial branch on the *same frozen PE-Spatial features* at
three capacity rungs, score every spatial tag through the threshold-free mean
AP, and stratify by the same KB taxonomy slices:

  * ``deployed``  — the shipped tagger's spatial head (Arm D, the baseline).
  * ``linear``    — fixed mean|max|cls pool → per-tag logistic (Arm L): the
    linear-accessible signal under a simple global pool.
  * ``oracle``    — a fresh MAPHead (same pooling *class* the deployed model
    uses, so no "you destroyed the spatial info" confound) + MLP trunk, trained
    to convergence in ISOLATION (spatial tags only, no rating/people/core
    multitask, class-balanced loss, val early-stop): the strongest head these
    frozen features admit (Arm M).

VERDICT (self-checking — see :func:`classify_ceiling`):

  * The oracle must first PASS a sanity gate: on the slices the deployed model
    already does well (high-AP control slices) the oracle must ≈ match it. If
    the oracle is globally *below* deployed, it is merely undertrained and any
    null result is inconclusive (``INCONCLUSIVE``).
  * Given a well-fit oracle: if it lifts the floor slices materially over
    deployed → ``HEAD_HEADROOM`` (isolate/retrain/relabel the spatial branch).
    If it cannot → ``FEATURE_CEILING`` (the tower is the bound; redirect off the
    tagger-quality line — a valuable don't-re-propose result).

READOUT: bench/tagger_ceiling/results/<ts>[-label]/{per_tag.csv,
         per_slice.csv, summary.md, result.json}

Run from anima_lora/::

    uv run python bench/tagger_ceiling/run_bench.py --label v2 --epochs 40
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested — no model, no disk).
# --------------------------------------------------------------------------- #


def fixed_pool(tokens: torch.Tensor) -> torch.Tensor:
    """``[B, T, D] → [B, 3D]`` via mean|max|cls(row 0) concat.

    The simple global pool the ``linear`` arm probes: mean and max over the
    patch grid plus the encoder's CLS token (row 0). Order [mean, max, cls] is
    fixed so downstream weight columns stay interpretable.
    """
    if tokens.dim() != 3:
        raise ValueError(f"fixed_pool expects [B, T, D], got rank {tokens.dim()}")
    mean = tokens.mean(dim=1)
    mx = tokens.amax(dim=1)
    cls = tokens[:, 0]
    return torch.cat([mean, mx, cls], dim=-1)


def classify_ceiling(
    slice_rows: List[dict],
    *,
    floor_ap: float = 0.22,
    min_support: int = 10,
    sanity_lift_floor: float = -0.03,
    headroom_lift: float = 0.05,
    ceiling_lift: float = 0.02,
) -> dict:
    """Attribution verdict from per-slice ``{deployed, linear, oracle}`` AP rows.

    Splits supported slices into FLOOR (deployed mean_ap < ``floor_ap``) and
    CONTROL (>= ``floor_ap``). Lift is ``oracle - deployed`` (per-slice mean AP),
    aggregated as the support-weighted median over each group.

    Gate 1 (oracle validity): the oracle must not be globally underpowered — its
    median CONTROL lift must be >= ``sanity_lift_floor`` (i.e. it roughly matches
    or beats deployed where deployed is strong). Fail → ``INCONCLUSIVE``.

    Given a valid oracle, the FLOOR median lift decides:
      * >= ``headroom_lift`` → ``HEAD_HEADROOM``
      * <= ``ceiling_lift``  → ``FEATURE_CEILING``
      * in between           → ``MIXED``
    """

    def _median(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    floor, control = [], []
    for r in slice_rows:
        if r.get("n_supported", 0) < min_support or r.get("support_sum", 0) <= 0:
            continue
        lift = r["oracle_ap"] - r["deployed_ap"]
        (floor if r["deployed_ap"] < floor_ap else control).append(lift)

    control_lift = _median(control)
    floor_lift = _median(floor)

    oracle_valid = (not control) or (control_lift >= sanity_lift_floor)
    if not oracle_valid:
        verdict = "INCONCLUSIVE"
    elif not floor:
        verdict = "NO_FLOOR_SLICES"
    elif floor_lift >= headroom_lift:
        verdict = "HEAD_HEADROOM"
    elif floor_lift <= ceiling_lift:
        verdict = "FEATURE_CEILING"
    else:
        verdict = "MIXED"

    return {
        "verdict": verdict,
        "n_floor_slices": len(floor),
        "n_control_slices": len(control),
        "floor_median_lift": floor_lift,
        "control_median_lift": control_lift,
        "oracle_valid": bool(oracle_valid),
        "floor_ap_cut": floor_ap,
    }


# --------------------------------------------------------------------------- #
# Args.
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model_dir", default="models/captioners/anima-tagger-v2")
    p.add_argument(
        "--tag_cache",
        default="models/danbooru_tags_classified.csv",
        help="Danbooru taxonomy KB CSV (drives KB slice assignment).",
    )
    p.add_argument("--feature_cache_dir", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40, help="Oracle training epochs.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--d_hidden", type=int, default=1536)
    p.add_argument("--pool_queries", type=int, default=8)
    p.add_argument("--pool_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--max_train",
        type=int,
        default=None,
        help="Cap train stems for the probe (default: all). Lower for a fast smoke.",
    )
    p.add_argument(
        "--eval_every", type=int, default=4, help="Val-AP check cadence (epochs)."
    )
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default=None)
    p.add_argument("--freq_head_min", type=int, default=1000)
    p.add_argument("--freq_mid_min", type=int, default=200)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run (few epochs, capped train) to exercise the pipeline.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data — materialize frozen PE-Spatial tokens per aspect bucket.
# --------------------------------------------------------------------------- #


def _bucket_index_lists(ds) -> Dict[tuple, List[int]]:
    """Group dataset indices by aux (PE-Spatial) bucket → shape-homogeneous."""
    from collections import defaultdict

    groups: Dict[tuple, List[int]] = defaultdict(list)
    for i, (_, aux_b) in enumerate(ds.buckets):
        groups[aux_b].append(i)
    return dict(groups)


def materialize_spatial(ds, spatial_idx: torch.Tensor):
    """Load every stem's PE-Spatial tokens into per-bucket RAM tensors.

    Returns ``(buckets, mh_spatial)`` where ``buckets`` maps aux-bucket → dict
    with ``tokens`` ``[n, T, D] bf16`` and ``rows`` (dataset indices, ascending),
    and ``mh_spatial`` is the ``[N, n_spatial]`` label matrix in dataset order.
    Reads the spatial sidecar directly (skips the unused PE-Core load).
    """
    from safetensors.torch import load_file as st_load
    from tqdm import tqdm

    groups = _bucket_index_lists(ds)
    mh_spatial = ds.multi_hot[:, spatial_idx].contiguous()
    out: Dict[tuple, dict] = {}
    for bk, idxs in groups.items():
        first = st_load(str(ds.paths_aux[idxs[0]]))["tokens"]
        toks = torch.empty((len(idxs),) + tuple(first.shape), dtype=first.dtype)
        toks[0] = first
        for row, i in enumerate(
            tqdm(idxs[1:], desc=f"spatial {bk}", leave=False), start=1
        ):
            toks[row] = st_load(str(ds.paths_aux[i]))["tokens"]
        out[bk] = {"tokens": toks, "rows": idxs}
    return out, mh_spatial


# --------------------------------------------------------------------------- #
# Arm L — linear probe on the fixed global pool.
# --------------------------------------------------------------------------- #


def _pooled_matrix(buckets, mh_spatial, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate fixed-pool features + aligned labels across all buckets."""
    xs, ys = [], []
    for b in buckets.values():
        toks = b["tokens"]
        # Chunk within the bucket — a whole bucket in fp32 on GPU can be tens of
        # GB (one bucket holds most of the corpus).
        for s in range(0, toks.shape[0], 512):
            chunk = toks[s : s + 512].to(device=device, dtype=torch.float32)
            xs.append(fixed_pool(chunk).cpu())
        ys.append(mh_spatial[b["rows"]])
    return torch.cat(xs, 0), torch.cat(ys, 0)


def fit_linear_probe(
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    n_out: int,
    pos_weight: torch.Tensor,
    device,
    steps: int = 400,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> nn.Module:
    """Full-batch logistic multilabel probe on fixed-pool features."""
    d_in = Xtr.shape[1]
    head = nn.Linear(d_in, n_out).to(device)
    X = Xtr.to(device)
    Y = Ytr.to(device)
    # Standardize features (linear probe is scale-sensitive; the oracle's
    # LayerNorm does this internally).
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True).clamp_min(1e-6)
    X = (X - mu) / sd
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    pw = pos_weight.to(device)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(X), Y, pos_weight=pw)
        loss.backward()
        opt.step()
    head._mu, head._sd = mu, sd  # stash for eval-time standardization
    return head


@torch.no_grad()
def linear_scores(head: nn.Module, Xva: torch.Tensor, device) -> torch.Tensor:
    X = (Xva.to(device) - head._mu) / head._sd
    return head(X).sigmoid().cpu()


# --------------------------------------------------------------------------- #
# Arm M — oracle: fresh MAPHead + MLP trunk, spatial tags only.
# --------------------------------------------------------------------------- #


class SpatialOracle(nn.Module):
    """Fresh spatial head: MAPHead (same pooling class as deployed) + MLP trunk.

    Deliberately at-or-above the deployed spatial branch's capacity so a null
    lift is attributable to the frozen features, not to under-provisioning.
    """

    def __init__(
        self,
        d_in: int,
        n_out: int,
        d_hidden: int,
        n_queries: int,
        n_heads: int,
        dropout: float,
        n_trunk_layers: int = 2,
    ):
        super().__init__()
        from library.captioning.anima_tagger_model import MAPHead

        self.pool = MAPHead(
            d=d_in, n_queries=n_queries, n_heads=n_heads, dropout=dropout
        )
        pool_dim = d_in * (n_queries + 2)  # MAP queries + cls + mean
        layers: List[nn.Module] = [
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_trunk_layers - 1):
            layers += [nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d_hidden, n_out)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            [self.pool(tokens).flatten(1), tokens[:, 0], tokens.mean(dim=1)], dim=-1
        )
        return self.head(self.trunk(pooled))


def _iter_batches(buckets, batch_size, generator=None, shuffle=True):
    """Yield (bucket_key, row_indices) minibatches within each bucket."""
    order = list(buckets.keys())
    for bk in order:
        n = buckets[bk]["tokens"].shape[0]
        perm = (
            torch.randperm(n, generator=generator).tolist()
            if shuffle
            else list(range(n))
        )
        for s in range(0, n, batch_size):
            yield bk, perm[s : s + batch_size]


@torch.no_grad()
def oracle_scores(
    model, buckets, mh_spatial, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Val scores + aligned labels in a fixed (unshuffled) bucket order."""
    model.eval()
    score_chunks, label_chunks = [], []
    for bk, rows in _iter_batches(buckets, 256, shuffle=False):
        toks = buckets[bk]["tokens"][rows].to(device=device, dtype=torch.float32)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(toks).float()
        score_chunks.append(logits.sigmoid().cpu())
        ds_rows = [buckets[bk]["rows"][r] for r in rows]
        label_chunks.append(mh_spatial[ds_rows])
    return torch.cat(score_chunks, 0), torch.cat(label_chunks, 0)


def train_head(
    model,
    args,
    train_buckets,
    mh_tr,
    val_buckets,
    mh_va,
    pos_weight,
    device,
    *,
    epochs=None,
    lr=None,
    label="head",
):
    """Train any spatial head to convergence; return (best-val-AP model, best mean-AP).

    ``pos_weight=None`` → plain (unbalanced) BCE. Val early-stop on spatial mean AP.
    """
    from scripts.anima_tagger.eval_metrics import per_tag_average_precision

    epochs = epochs or args.epochs
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr or args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw = pos_weight.to(device) if pos_weight is not None else None
    gen = torch.Generator().manual_seed(args.seed)

    best_map = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        n_seen, loss_sum = 0, 0.0
        for bk, rows in _iter_batches(train_buckets, args.batch_size, generator=gen):
            toks = train_buckets[bk]["tokens"][rows].to(
                device=device, dtype=torch.float32
            )
            y = mh_tr[[train_buckets[bk]["rows"][r] for r in rows]].to(device)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(toks).float()
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(rows)
            n_seen += len(rows)
        sched.step()
        if (ep + 1) % args.eval_every == 0 or ep == epochs - 1:
            scores, labels = oracle_scores(model, val_buckets, mh_va, device)
            ap = per_tag_average_precision(scores, labels)
            mean_ap = float(ap.nanmean())
            log.info(
                "%s ep %2d/%d  loss %.4f  val mean-AP %.4f",
                label,
                ep + 1,
                epochs,
                loss_sum / max(n_seen, 1),
                mean_ap,
            )
            if mean_ap > best_map:
                best_map = mean_ap
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_map


# --------------------------------------------------------------------------- #
# Arm D — deployed spatial-head val scores (same order as val_buckets).
# --------------------------------------------------------------------------- #


@torch.no_grad()
def deployed_scores(args, cfg_d, model_dir, val_ds, val_buckets, spatial_idx, device):
    """Run the shipped tagger over val in the exact bucket order of val_buckets."""
    from library.captioning.anima_tagger_model import AnimaTaggerConfig, AnimaTaggerHead
    from safetensors.torch import load_file as st_load

    cfg = AnimaTaggerConfig.from_dict(cfg_d["model"])
    model = AnimaTaggerHead(cfg)
    model.load_state_dict(st_load(str(model_dir / "model.safetensors")))
    model.to(device).eval()

    score_chunks, label_chunks = [], []
    autocast = torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    # PE-Core and PE-Spatial bucket independently, so token counts vary within a
    # single aux bucket → forward per-sample (val is small) rather than stacking.
    for bk, rows in _iter_batches(val_buckets, args.batch_size, shuffle=False):
        ds_rows = [val_buckets[bk]["rows"][r] for r in rows]
        for i in ds_rows:
            fc = val_ds._load_one(val_ds.paths[i], val_ds.pool_kind)
            fa = val_ds._load_one(val_ds.paths_aux[i], val_ds.pool_kind_aux)
            fc = fc.unsqueeze(0).to(device=device, dtype=torch.float32)
            fa = fa.unsqueeze(0).to(device=device, dtype=torch.float32)
            with autocast:
                tag_logits, _, _ = model(fc, fa)
            score_chunks.append(tag_logits[:, spatial_idx].float().sigmoid().cpu())
        label_chunks.append(val_ds.multi_hot[ds_rows][:, spatial_idx])
    return torch.cat(score_chunks, 0), torch.cat(label_chunks, 0)


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #


def _build_dataset(args, cfg_d, model_dir, stems):
    from library.captioning.anima_tagger_data import CachedDualDataset, TaggerManifest
    from library.vision.encoders import get_encoder_info
    from scripts.anima_tagger.caches import cache_dir_for, feature_cache_root

    pool_kind = str(cfg_d.get("pool_kind", "map"))
    pool_kind_aux = str(cfg_d.get("pool_kind_aux", "map"))
    encoder = cfg_d.get("encoder", "pe")
    aux_encoder = cfg_d.get("aux_encoder", "pe_spatial")
    root = feature_cache_root(args)
    cache_dir = cache_dir_for(root, pool_kind, encoder)
    cache_dir_aux = cache_dir_for(root, pool_kind_aux, aux_encoder)
    spec = get_encoder_info(encoder).bucket_spec if pool_kind == "map" else None
    spec_aux = (
        get_encoder_info(aux_encoder).bucket_spec if pool_kind_aux == "map" else None
    )
    manifest = TaggerManifest.from_path(model_dir / "dataset.json")
    return CachedDualDataset(
        manifest,
        cache_dir,
        pool_kind,
        spec,
        cache_dir_aux,
        pool_kind_aux,
        spec_aux,
        stems_subset=stems,
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 4)
        args.eval_every = 2
        args.max_train = args.max_train or 512
    torch.manual_seed(args.seed)
    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json") as f:
        cfg_d = json.load(f)
    with open(model_dir / "vocab.json") as f:
        vocab = json.load(f)
    with open(model_dir / "dataset.json") as f:
        splits = json.load(f)["split"]

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    spatial_idx = torch.tensor(cfg_d["model"]["tag_indices_spatial"], dtype=torch.long)
    n_spatial = spatial_idx.numel()

    from library.captioning import tag_rules as tr
    from library.captioning.correction import load_tag_knowledge_base
    from scripts.anima_tagger.eval_metrics import (
        assign_slices,
        per_tag_average_precision,
    )

    # ── Datasets (spatial features materialized into RAM) ──
    train_stems = (
        splits["train"][: args.max_train] if args.max_train else splits["train"]
    )
    train_ds = _build_dataset(args, cfg_d, model_dir, train_stems)
    val_ds = _build_dataset(args, cfg_d, model_dir, splits["val"])
    log.info(
        "train stems=%d  val stems=%d  n_spatial=%d",
        len(train_ds),
        len(val_ds),
        n_spatial,
    )

    train_buckets, mh_tr = materialize_spatial(train_ds, spatial_idx)
    val_buckets, mh_va = materialize_spatial(val_ds, spatial_idx)

    # pos_weight from train spatial positives (class-balanced BCE).
    pos = mh_tr.sum(0).clamp_min(1.0)
    neg = mh_tr.shape[0] - pos
    pos_weight = (neg / pos).clamp(max=100.0)

    # ── Arm D: deployed ──
    log.info("Arm D: forwarding deployed tagger over val ...")
    dep_scores, dep_labels = deployed_scores(
        args, cfg_d, model_dir, val_ds, val_buckets, spatial_idx, device
    )
    ap_dep = per_tag_average_precision(dep_scores, dep_labels)

    # ── Arm L: fixed-pool linear ──
    log.info("Arm L: fitting linear probe on fixed mean|max|cls pool ...")
    Xtr, Ytr = _pooled_matrix(train_buckets, mh_tr, device)
    Xva, Yva = _pooled_matrix(val_buckets, mh_va, device)
    lin = fit_linear_probe(Xtr, Ytr, n_spatial, pos_weight, device)
    ap_lin = per_tag_average_precision(linear_scores(lin, Xva, device), Yva)

    # ── Arm M: oracle ──
    log.info("Arm M: training oracle (MAPHead + MLP, spatial-only) ...")
    d_in = next(iter(train_buckets.values()))["tokens"].shape[-1]
    oracle = SpatialOracle(
        d_in, n_spatial, args.d_hidden, args.pool_queries, args.pool_heads, args.dropout
    ).to(device)
    oracle, best_map = train_head(
        oracle,
        args,
        train_buckets,
        mh_tr,
        val_buckets,
        mh_va,
        pos_weight,
        device,
        label="oracle",
    )
    orc_scores, orc_labels = oracle_scores(oracle, val_buckets, mh_va, device)
    ap_orc = per_tag_average_precision(orc_scores, orc_labels)

    # ── Slice assignment (spatial tags only) ──
    kb = load_tag_knowledge_base(args.tag_cache)
    rules_path = model_dir / "rules.yaml"
    rename_recovery: Dict[str, str] = {}
    if rules_path.exists():
        rename_recovery = {
            tgt: src for src, tgt in tr.load_rules(rules_path).replacements
        }
    slices = assign_slices(
        vocab, kb, rename_recovery, freq_cuts=(args.freq_head_min, args.freq_mid_min)
    )
    spatial_list = spatial_idx.tolist()
    kb_of = [slices.kb_slice[i] for i in spatial_list]
    support = dep_labels.sum(0)  # val positives per spatial tag (order == arms)

    # Per-slice aggregation across arms.
    from collections import defaultdict

    by_slice: Dict[str, List[int]] = defaultdict(list)
    for local_i, key in enumerate(kb_of):
        by_slice[key].append(local_i)

    def _slice_ap(ap: torch.Tensor, idxs) -> float:
        v = ap[torch.tensor(idxs)]
        v = v[~torch.isnan(v)]
        return float(v.mean()) if v.numel() else float("nan")

    slice_rows: List[dict] = []
    for key, idxs in by_slice.items():
        sup = support[torch.tensor(idxs)]
        n_sup = int((sup > 0).sum())
        slice_rows.append(
            {
                "slice": key,
                "n_tags": len(idxs),
                "n_supported": n_sup,
                "support_sum": int(sup.sum()),
                "deployed_ap": _slice_ap(ap_dep, idxs),
                "linear_ap": _slice_ap(ap_lin, idxs),
                "oracle_ap": _slice_ap(ap_orc, idxs),
            }
        )
    slice_rows.sort(
        key=lambda r: r["deployed_ap"] if not math.isnan(r["deployed_ap"]) else 9
    )

    verdict = classify_ceiling(slice_rows)

    def _mean_over_supported(ap: torch.Tensor) -> float:
        m = support > 0
        v = ap[m]
        v = v[~torch.isnan(v)]
        return float(v.mean()) if v.numel() else float("nan")

    headline = {
        "n_val": int(dep_scores.shape[0]),
        "n_spatial_tags": n_spatial,
        "n_supported": int((support > 0).sum()),
        "deployed_mean_ap": _mean_over_supported(ap_dep),
        "linear_mean_ap": _mean_over_supported(ap_lin),
        "oracle_mean_ap": _mean_over_supported(ap_orc),
        "oracle_best_val_map": best_map,
        **verdict,
    }

    # ── Artifacts ──
    run_dir = make_run_dir("tagger_ceiling", label=args.label)
    per_tag_path = run_dir / "per_tag.csv"
    with open(per_tag_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "global_index",
                "name",
                "kb_slice",
                "val_support",
                "deployed_ap",
                "linear_ap",
                "oracle_ap",
            ]
        )
        for local_i, gi in enumerate(spatial_list):
            w.writerow(
                [
                    gi,
                    vocab["tags"][gi]["name"],
                    kb_of[local_i],
                    int(support[local_i]),
                    f"{float(ap_dep[local_i]):.4f}",
                    f"{float(ap_lin[local_i]):.4f}",
                    f"{float(ap_orc[local_i]):.4f}",
                ]
            )
    per_slice_path = run_dir / "per_slice.csv"
    with open(per_slice_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(slice_rows[0].keys()))
        w.writeheader()
        w.writerows(slice_rows)

    def _f(x: float) -> str:
        return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"

    lines = [
        f"# tagger_ceiling — {args.label or model_dir.name} (spatial branch)",
        "",
        f"val N={headline['n_val']}, spatial tags={n_spatial} "
        f"({headline['n_supported']} supported).",
        "",
        f"## VERDICT: **{verdict['verdict']}**",
        "",
        f"- floor slices (deployed AP < {verdict['floor_ap_cut']}): "
        f"{verdict['n_floor_slices']}  |  control slices: {verdict['n_control_slices']}",
        f"- floor median lift (oracle − deployed): **{_f(verdict['floor_median_lift'])}**",
        f"- control median lift (oracle sanity gate): {_f(verdict['control_median_lift'])} "
        f"(oracle_valid={verdict['oracle_valid']})",
        "",
        "## Headline mean AP (supported spatial tags)",
        "",
        f"- deployed: {_f(headline['deployed_mean_ap'])}",
        f"- linear  : {_f(headline['linear_mean_ap'])}",
        f"- oracle  : {_f(headline['oracle_mean_ap'])}",
        "",
        "## Per-slice AP (sorted by deployed AP asc — the floor is on top)",
        "",
        "| slice | tags | sup | deployed | linear | oracle | lift |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in slice_rows:
        if r["n_supported"] < 1:
            continue
        lift = r["oracle_ap"] - r["deployed_ap"]
        lines.append(
            f"| {r['slice']} | {r['n_tags']} | {r['n_supported']} | "
            f"{_f(r['deployed_ap'])} | {_f(r['linear_ap'])} | {_f(r['oracle_ap'])} | "
            f"{lift:+.4f} |"
        )
    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics={**headline, "slices": slice_rows},
        artifacts=[per_tag_path, per_slice_path, summary_path],
        device=device,
    )
    print(f"\nwrote {run_dir}/")
    print("\n".join(lines[4:12]))


if __name__ == "__main__":
    main()
