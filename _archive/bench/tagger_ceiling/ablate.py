#!/usr/bin/env python
"""Tagger ceiling — factor ablation: WHY does a linear probe beat the deployed head?

The ceiling probe found a plain linear head on a global pool of frozen PE-Spatial
beats the deployed spatial branch by ~+0.084 mean AP. Three factors are confounded
in that gap:

  A. ISOLATION      — the probe trains spatial-tags-only; the deployed head shares
                      optimization with the core/rating/people heads (multitask).
  B. CLASS-BALANCE  — the probe uses pos_weight-balanced BCE; the deployed loss does
                      not (group-conditional inactive_neg_weight ≈ 1.0 per gold-check).
  C. ARCHITECTURE   — linear-on-global-pool vs the deployed MAPHead(K=4) + MLP trunk.

This ablation trains a small grid on the SAME materialized frozen features (all
spatial-only, so isolation is held constant = isolated) and reads spatial mean AP:

  cell                         head            pos_weight   isolates
  ----------------------------------------------------------------------------
  deployed (anchor)            deployed-arch   deployed     multitask baseline
  dep_arch__nobal              deployed-arch   OFF          A: isolation (+recipe)
  dep_arch__bal                deployed-arch   ON           B: class-balance
  linear__bal                  linear pool     ON           C: architecture
  linear__nobal                linear pool     OFF          (B on the linear head)

Additive read (each step holds the prior + flips one knob):
  deployed → dep_arch__nobal   = isolation (+ train-recipe) gain
  dep_arch__nobal → dep_arch__bal = class-balance gain (deployed arch)
  dep_arch__bal → linear__bal  = architecture gain (MAP+MLP → linear)

CAVEAT: the deployed→dep_arch__nobal step also folds in train-recipe differences
(epochs / LR / schedule / exact loss) I can't match to the deployed run, so it is
"isolation + recipe", not pure isolation. Phase-0, one seed.

READOUT: bench/tagger_ceiling/results/<ts>[-label]-ablate/{cells.csv, summary.md,
         result.json}

Run from anima_lora/::

    uv run python bench/tagger_ceiling/ablate.py --label v2 --epochs 40
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from bench.tagger_ceiling.run_bench import (  # noqa: E402
    SpatialOracle,
    _build_dataset,
    _pooled_matrix,
    deployed_scores,
    fit_linear_probe,
    linear_scores,
    log,
    materialize_spatial,
    parse_args,
    train_head,
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
    with open(model_dir / "dataset.json") as f:
        splits = json.load(f)["split"]

    from scripts.anima_tagger.eval_metrics import per_tag_average_precision

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    spatial_idx = torch.tensor(cfg_d["model"]["tag_indices_spatial"], dtype=torch.long)
    n_spatial = spatial_idx.numel()

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
    d_in = next(iter(train_buckets.values()))["tokens"].shape[-1]

    pos = mh_tr.sum(0).clamp_min(1.0)
    pos_weight = ((mh_tr.shape[0] - pos) / pos).clamp(max=100.0)

    def mean_ap(scores, labels) -> float:
        ap = per_tag_average_precision(scores, labels)
        return float(ap.nanmean())

    cells: dict[str, float] = {}

    # ── Anchor: deployed head (multitask, as shipped) ──
    log.info("[anchor] deployed head over val ...")
    dep_scores, dep_labels = deployed_scores(
        args, cfg_d, model_dir, val_ds, val_buckets, spatial_idx, device
    )
    cells["deployed"] = mean_ap(dep_scores, dep_labels)

    # Deployed-arch reinit params: MAPHead K=4, single trunk layer, d_hidden 1024
    # (mirrors AnimaTaggerHead's spatial branch); vary only pos_weight.
    def make_dep_arch():
        return SpatialOracle(
            d_in,
            n_spatial,
            d_hidden=1024,
            n_queries=4,
            n_heads=8,
            dropout=args.dropout,
            n_trunk_layers=1,
        ).to(device)

    # ── A: isolation (+recipe) — deployed arch, isolated, no balance ──
    log.info("[dep_arch__nobal] isolated deployed-arch, plain BCE ...")
    m, _ = train_head(
        make_dep_arch(),
        args,
        train_buckets,
        mh_tr,
        val_buckets,
        mh_va,
        None,
        device,
        label="dep_arch__nobal",
    )
    from bench.tagger_ceiling.run_bench import oracle_scores

    s, lb = oracle_scores(m, val_buckets, mh_va, device)
    cells["dep_arch__nobal"] = mean_ap(s, lb)

    # ── B: class-balance — deployed arch, isolated, pos_weight ──
    log.info("[dep_arch__bal] isolated deployed-arch, balanced BCE ...")
    m, _ = train_head(
        make_dep_arch(),
        args,
        train_buckets,
        mh_tr,
        val_buckets,
        mh_va,
        pos_weight,
        device,
        label="dep_arch__bal",
    )
    s, lb = oracle_scores(m, val_buckets, mh_va, device)
    cells["dep_arch__bal"] = mean_ap(s, lb)

    # ── C: architecture — linear global-pool head ──
    Xtr, Ytr = _pooled_matrix(train_buckets, mh_tr, device)
    Xva, Yva = _pooled_matrix(val_buckets, mh_va, device)
    log.info("[linear__bal] linear pool head, balanced ...")
    lin = fit_linear_probe(Xtr, Ytr, n_spatial, pos_weight, device)
    cells["linear__bal"] = mean_ap(linear_scores(lin, Xva, device), Yva)
    log.info("[linear__nobal] linear pool head, plain ...")
    lin0 = fit_linear_probe(Xtr, Ytr, n_spatial, torch.ones(n_spatial), device)
    cells["linear__nobal"] = mean_ap(linear_scores(lin0, Xva, device), Yva)

    # ── Additive decomposition ──
    decomp = {
        "isolation_plus_recipe": cells["dep_arch__nobal"] - cells["deployed"],
        "class_balance": cells["dep_arch__bal"] - cells["dep_arch__nobal"],
        "architecture": cells["linear__bal"] - cells["dep_arch__bal"],
        "total_deployed_to_linear": cells["linear__bal"] - cells["deployed"],
        "class_balance_on_linear": cells["linear__bal"] - cells["linear__nobal"],
    }

    # ── Artifacts ──
    run_dir = make_run_dir("tagger_ceiling", label=(args.label or "") + "-ablate")
    with open(run_dir / "cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "mean_ap"])
        for k, v in cells.items():
            w.writerow([k, f"{v:.4f}"])

    order = [
        "deployed",
        "dep_arch__nobal",
        "dep_arch__bal",
        "linear__bal",
        "linear__nobal",
    ]
    lines = [
        f"# tagger_ceiling ablation — {args.label or model_dir.name} (spatial branch)",
        "",
        f"val N={int(dep_scores.shape[0])}, spatial tags={n_spatial}. "
        f"train={len(train_ds)}, epochs={args.epochs}.",
        "",
        "## Cells (spatial mean AP)",
        "",
        "| cell | mean AP |",
        "|---|---|",
    ]
    for k in order:
        lines.append(f"| {k} | {cells[k]:.4f} |")
    lines += [
        "",
        "## Additive decomposition of the deployed→linear gap",
        "",
        "| factor | ΔAP |",
        "|---|---|",
        f"| A. isolation (+recipe) | {decomp['isolation_plus_recipe']:+.4f} |",
        f"| B. class-balance (dep arch) | {decomp['class_balance']:+.4f} |",
        f"| C. architecture (MAP+MLP → linear) | {decomp['architecture']:+.4f} |",
        f"| **total (deployed → linear)** | **{decomp['total_deployed_to_linear']:+.4f}** |",
        f"| (class-balance on linear head) | {decomp['class_balance_on_linear']:+.4f} |",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=(args.label or "") + "-ablate",
        metrics={
            "cells": cells,
            "decomposition": decomp,
            "n_val": int(dep_scores.shape[0]),
        },
        artifacts=[run_dir / "cells.csv", run_dir / "summary.md"],
        device=device,
    )
    print(f"\nwrote {run_dir}/")
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    main()
