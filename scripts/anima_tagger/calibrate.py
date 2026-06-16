"""Per-tag F1-optimal threshold sweep on the val split.

A global 0.5 threshold under-fires rare tags and over-fires common ones.
This sweeps thresholds in [0.05, 0.95] step 0.05 per tag and picks the
F1-maximizing one. Tags with no positive val examples or zero
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

from .train_common import GroupRouter

logger = logging.getLogger(__name__)


def calibrate_thresholds(
    scores: torch.Tensor,  # [N, n_tags] sigmoid probabilities
    targets: torch.Tensor,  # [N, n_tags] multi-hot
    sweep: torch.Tensor,  # [K] candidate thresholds
    default: float = 0.5,
    skip_indices: Optional[
        torch.Tensor
    ] = None,  # LongTensor of tag indices to leave at default
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
    pos_count = targets.sum(dim=0)  # [n_tags]
    has_pos = pos_count > 0
    if skip_indices is not None and skip_indices.numel() > 0:
        skip_mask = torch.zeros(n_tags, dtype=torch.bool)
        skip_mask[skip_indices.cpu()] = True
        has_pos = has_pos & ~skip_mask
    # Block over tags to keep memory bounded — the dense [N, n_tags, K] tensor
    # would be ~12k × 5k × 19 ≈ 1.1B floats.
    block_size = 256
    for start in range(0, n_tags, block_size):
        end = min(start + block_size, n_tags)
        s = scores[:, start:end]  # [N, b]
        t = targets[:, start:end]
        # [N, b, K] boolean
        pred = s.unsqueeze(-1) > sweep.view(1, 1, K)
        pred_f = pred.float()
        tp = (pred_f * t.unsqueeze(-1)).sum(dim=0)  # [b, K]
        fp = (pred_f * (1 - t).unsqueeze(-1)).sum(dim=0)
        fn = ((1 - pred_f) * t.unsqueeze(-1)).sum(dim=0)
        prec = tp / (tp + fp).clamp_min(1e-8)
        rec = tp / (tp + fn).clamp_min(1e-8)
        f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-8)  # [b, K]
        f1_best, k_best = f1.max(dim=-1)  # [b]
        thresh_best = sweep[k_best]  # [b]
        local_has_pos = has_pos[start:end]
        keep = local_has_pos & (f1_best > 0)
        best_f1[start:end] = torch.where(keep, f1_best, best_f1[start:end])
        best_thresh[start:end] = torch.where(keep, thresh_best, best_thresh[start:end])
    return best_thresh, best_f1


def cmd_calibrate(args: argparse.Namespace) -> None:
    from safetensors.torch import load_file as st_load
    from safetensors.torch import save_file as st_save

    from library.captioning.anima_tagger_model import (
        AnimaTaggerConfig,
        AnimaTaggerHead,
    )

    from .caches import cache_dir_for, feature_cache_root

    out_dir = Path(args.out_dir)
    feature_root = feature_cache_root(args)

    with open(out_dir / "config.json") as f:
        cfg_d = json.load(f)
    cfg = AnimaTaggerConfig.from_dict(cfg_d["model"])
    model = AnimaTaggerHead(cfg)
    state = st_load(str(out_dir / "model.safetensors"))
    model.load_state_dict(state)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device).eval()

    # Respect the saved config.json so calibrate matches what was trained (no
    # need to re-pass --pool_kind / --encoder / --aux_encoder). CLI args still win.
    pool_kind = str(cfg_d.get("pool_kind", cfg.pool_kind))
    encoder = cfg_d.get("encoder") or args.encoder

    cache_dir = cache_dir_for(feature_root, pool_kind, encoder)
    if not cache_dir.exists():
        raise SystemExit(
            f"missing {cache_dir} — calibrate needs the same cache the "
            f"trainer used (encoder={encoder!r} pool_kind={pool_kind!r}). "
            f"Re-run `--mode build_features --pool_kind={pool_kind} "
            f"--encoder {encoder}` if you deleted it."
        )

    from library.captioning.anima_tagger_data import TaggerManifest

    manifest = TaggerManifest.from_path(out_dir / "dataset.json")

    # Dual-encoder models go through the bucket-loader path.
    from torch.utils.data import DataLoader

    from library.captioning.anima_tagger_data import (
        BucketBatchSampler,
        CachedDualDataset,
        collate_dual_token_batch,
    )
    from library.vision.encoders import get_encoder_info

    spec = get_encoder_info(encoder).bucket_spec if pool_kind == "map" else None
    # Mirror the training-time aux choice from the saved config (no need to
    # re-pass --aux_encoder / --pool_kind_aux). CLI flags still win.
    aux_encoder = args.aux_encoder or cfg_d.get("aux_encoder")
    if not aux_encoder:
        raise SystemExit(
            "config has no recorded aux_encoder and --aux_encoder wasn't given. "
            "Re-pass --aux_encoder pe_spatial."
        )
    pool_kind_aux = args.pool_kind_aux or str(
        cfg_d.get("pool_kind_aux", cfg.pool_kind_aux)
    )
    spec_aux = (
        get_encoder_info(aux_encoder).bucket_spec if pool_kind_aux == "map" else None
    )
    cache_dir_aux = cache_dir_for(feature_root, pool_kind_aux, aux_encoder)
    if not cache_dir_aux.exists():
        raise SystemExit(
            f"missing aux cache {cache_dir_aux} — calibrate needs the same "
            f"cache the trainer used (pool_kind_aux={pool_kind_aux})."
        )
    # Reuse the trainer's per-bucket mmap shards (keyed by kept-stem hash) under
    # <feature_root>/packed rather than thousands of per-stem sidecars; rebuilt
    # on demand. Calibrate only materializes val, never pruning train shards.
    pack_root = feature_root / "packed"
    val_ds = CachedDualDataset(
        manifest,
        cache_dir,
        pool_kind,
        spec,
        cache_dir_aux,
        pool_kind_aux,
        spec_aux,
        stems_subset=manifest.val_stems,
        pack_root=pack_root,
    )
    val_mh = val_ds.multi_hot.to(device)
    sampler = BucketBatchSampler(
        val_ds.buckets, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )
    loader = DataLoader(
        val_ds,
        batch_sampler=sampler,
        num_workers=args.feature_cache_workers,
        collate_fn=collate_dual_token_batch,
        pin_memory=True,
    )
    chunks: list[torch.Tensor] = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            tokens, tokens_aux, _mh, _rate, _people, _bucket = batch
            tokens = tokens.to(device, non_blocking=True)
            tokens_aux = tokens_aux.to(device, non_blocking=True)
            tl, _rl, _pl = model(tokens, tokens_aux)
            chunks.append(tl.float())
    tag_logits = torch.cat(chunks, dim=0)
    # Reorder val_mh to match loader emission (sorted-bucket, then dataset order).
    order_indices: list[int] = []
    for batch_idx_list in sampler:
        order_indices.extend(batch_idx_list)
    val_mh = val_mh[torch.as_tensor(order_indices, device=val_mh.device)]

    scores = tag_logits.sigmoid().cpu()
    val_mh_cpu = val_mh.cpu()
    # Router: skip softmax-group tags from the F1 sweep + report eval F1 over
    # residual tags only (matching training).
    with open(out_dir / "vocab.json") as f:
        vocab_dict = json.load(f)
    router = GroupRouter.from_vocab(vocab_dict, val_mh, device=device)
    if router.is_active() and router.softmax_member_indices is not None:
        all_softmax_idx = router.softmax_member_indices
    else:
        all_softmax_idx = torch.empty(0, dtype=torch.long)
    sweep = torch.linspace(0.05, 0.95, 19)
    thresh, f1 = calibrate_thresholds(
        scores,
        val_mh_cpu,
        sweep,
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
    # Baseline macro-F1 at the default 0.5 threshold (residual tags only when
    # softmax groups are active), from the already-collected logits.
    if all_softmax_idx.numel() > 0:
        keep_mask = torch.ones(scores.shape[1], dtype=torch.bool)
        keep_mask[all_softmax_idx.cpu()] = False
        kept = keep_mask.nonzero(as_tuple=False).squeeze(1)
        baseline_scores = scores.index_select(1, kept)
        baseline_target = val_mh_cpu.index_select(1, kept)
    else:
        baseline_scores = scores
        baseline_target = val_mh_cpu
    pred = (baseline_scores > 0.5).float()
    tp = (pred * baseline_target).sum(dim=0)
    fp = (pred * (1 - baseline_target)).sum(dim=0)
    fn = ((1 - pred) * baseline_target).sum(dim=0)
    prec_b = tp / (tp + fp).clamp_min(1.0)
    rec_b = tp / (tp + fn).clamp_min(1.0)
    f1_b = 2 * prec_b * rec_b / (prec_b + rec_b).clamp_min(1e-8)
    logger.info(
        "macro-F1 (calibrated) = %.4f  vs default 0.5 macro-F1 = %.4f",
        f1.mean().item(),
        f1_b.mean().item(),
    )
    print(f"  thresholds: {out_dir / 'thresholds.safetensors'}")
    print(f"  active tags (F1>0): {n_active} / {thresh.shape[0]}")
    print(f"  calibrated macro-F1: {f1.mean().item():.4f}")
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
