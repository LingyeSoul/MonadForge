"""Per-tag F1-optimal threshold sweep on the val split.

A global 0.5 (wd-tagger's inheritance) under-fires rare tags and over-fires
common ones. This sweeps thresholds in [0.05, 0.95] step 0.05 per tag and
picks the F1-maximizing one. Tags with no positive val examples or zero
achievable F1 keep ``default=0.5`` — they can't be calibrated and the F1
sweep is degenerate, but the floor keeps the head well-formed for inference.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

from .train_common import GroupRouter, eval_split

logger = logging.getLogger(__name__)


def calibrate_thresholds(
    scores: torch.Tensor,        # [N, n_tags] sigmoid probabilities
    targets: torch.Tensor,       # [N, n_tags] multi-hot
    sweep: torch.Tensor,         # [K] candidate thresholds
    default: float = 0.5,
    skip_indices: Optional[torch.Tensor] = None,  # LongTensor of tag indices to leave at default
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tag F1-optimal threshold sweep.

    Returns ``(thresholds[n_tags], best_f1[n_tags])``. Tags with no positives
    in the val split keep ``default`` (they can't be calibrated and the
    F1 sweep is degenerate — a 0.5 floor is harmless and keeps the head
    well-formed for inference). Same fallback for tags whose best
    achievable F1 is 0 (model never predicts them at any threshold).

    ``skip_indices`` is the trainer-side hint that some tags belong to a
    softmax group and shouldn't be sigmoid-thresholded (inference uses
    argmax). Those keep ``default`` and ``best_f1=0``.
    """
    n_tags = scores.shape[1]
    K = sweep.shape[0]
    best_thresh = torch.full((n_tags,), default)
    best_f1 = torch.zeros(n_tags)
    pos_count = targets.sum(dim=0)                              # [n_tags]
    has_pos = pos_count > 0
    if skip_indices is not None and skip_indices.numel() > 0:
        skip_mask = torch.zeros(n_tags, dtype=torch.bool)
        skip_mask[skip_indices.cpu()] = True
        has_pos = has_pos & ~skip_mask
    # Process tag-blocks to keep memory bounded — the dense [N, n_tags, K]
    # tensor would be ~12k × 5k × 19 ≈ 1.1B floats which is too big.
    block_size = 256
    for start in range(0, n_tags, block_size):
        end = min(start + block_size, n_tags)
        s = scores[:, start:end]                                 # [N, b]
        t = targets[:, start:end]
        # [N, b, K] boolean
        pred = s.unsqueeze(-1) > sweep.view(1, 1, K)
        pred_f = pred.float()
        tp = (pred_f * t.unsqueeze(-1)).sum(dim=0)               # [b, K]
        fp = (pred_f * (1 - t).unsqueeze(-1)).sum(dim=0)
        fn = ((1 - pred_f) * t.unsqueeze(-1)).sum(dim=0)
        prec = tp / (tp + fp).clamp_min(1e-8)
        rec = tp / (tp + fn).clamp_min(1e-8)
        f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-8)       # [b, K]
        f1_best, k_best = f1.max(dim=-1)                          # [b]
        thresh_best = sweep[k_best]                               # [b]
        local_has_pos = has_pos[start:end]
        keep = local_has_pos & (f1_best > 0)
        best_f1[start:end] = torch.where(
            keep, f1_best, best_f1[start:end]
        )
        best_thresh[start:end] = torch.where(
            keep, thresh_best, best_thresh[start:end]
        )
    return best_thresh, best_f1


def cmd_calibrate(args: argparse.Namespace) -> None:
    from safetensors.torch import load_file as st_load
    from safetensors.torch import save_file as st_save

    from library.captioning.anima_tagger_data import (
        CachedFeatureDataset,
        TaggerManifest,
    )
    from library.captioning.anima_tagger_model import (
        AnimaTaggerConfig,
        AnimaTaggerHead,
    )

    out_dir = Path(args.out_dir)
    manifest = TaggerManifest.from_path(out_dir / "dataset.json")
    cache_dir = out_dir / ".cache" / f"pooled-{args.encoder}"
    val_ds = CachedFeatureDataset(manifest, cache_dir, stems_subset=manifest.val_stems)

    with open(out_dir / "config.json") as f:
        cfg_d = json.load(f)
    cfg = AnimaTaggerConfig.from_dict(cfg_d["model"])
    model = AnimaTaggerHead(cfg)
    state = st_load(str(out_dir / "model.safetensors"))
    model.load_state_dict(state)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    val_feats = val_ds.features.to(device)
    val_mh = val_ds.multi_hot.to(device)
    # Build the same router the trainer used so we can:
    #   (a) skip softmax-group tags from the per-tag F1 sweep, and
    #   (b) report eval F1 over residual tags only (matching training).
    with open(out_dir / "vocab.json") as f:
        vocab_dict = json.load(f)
    train_mh = val_ds.multi_hot.to(device)  # router only needs the shape; pos_weight is unused here
    router = GroupRouter.from_vocab(vocab_dict, train_mh, device=device)
    if router.is_active() and router.softmax_member_indices is not None:
        all_softmax_idx = router.softmax_member_indices
    else:
        all_softmax_idx = torch.empty(0, dtype=torch.long)

    with torch.no_grad():
        tag_logits, _ = model(val_feats)
        scores = tag_logits.sigmoid().cpu()
    sweep = torch.linspace(0.05, 0.95, 19)
    thresh, f1 = calibrate_thresholds(
        scores, val_mh.cpu(), sweep,
        default=0.5,
        skip_indices=all_softmax_idx,
    )

    st_save(
        {"thresholds": thresh, "val_f1": f1},
        str(out_dir / "thresholds.safetensors"),
    )
    n_active = int((f1 > 0).sum().item())
    logger.info(
        "calibrated %d/%d tags with non-zero F1 at sweep optimum",
        n_active,
        thresh.shape[0],
    )
    logger.info(
        "macro-F1 (calibrated) = %.4f  vs default 0.5 macro-F1 = %.4f",
        f1.mean().item(),
        eval_split(
            model, val_feats, val_mh, val_ds.rating_idx.to(device),
            router=router,
        )["macro_f1"],
    )
    print(f"  thresholds: {out_dir / 'thresholds.safetensors'}")
    print(f"  active tags (F1>0): {n_active} / {thresh.shape[0]}")
    print(f"  calibrated macro-F1: {f1.mean().item():.4f}")
    # Print a sample of low/mid/high thresholds for sanity.
    with open(out_dir / "vocab.json") as f:
        vocab = json.load(f)
    name_of = [t["name"] for t in vocab["tags"]]
    by_thresh = sorted(
        ((thresh[i].item(), f1[i].item(), name_of[i]) for i in range(thresh.shape[0])),
        key=lambda x: x[0],
    )
    print("  sample thresholds (lowest 5 / highest 5):")
    for t, fv, n in by_thresh[:5] + by_thresh[-5:]:
        print(f"    thresh={t:.2f}  f1={fv:.3f}  {n}")
