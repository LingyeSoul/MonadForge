"""Frozen-encoder training path.

Reads pre-pooled PE features from ``out_dir/.cache/pooled-<encoder>/``
(built via ``--mode build_features``) and trains only the head + trunk.
Whole train/val sets fit in VRAM at ~50 MB each, so we push them once
and slice batches by index — no DataLoader.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch

from .train_common import (
    GroupRouter,
    compute_grouped_loss,
    eval_split,
    rating_class_weights,
    save_history_plot,
)

logger = logging.getLogger(__name__)


def cmd_train_cached(args: argparse.Namespace) -> None:
    """Default path: head trains on pre-pooled cached PE features. Encoder frozen."""
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
    manifest_path = out_dir / "dataset.json"
    vocab_path = out_dir / "vocab.json"
    cache_dir = out_dir / ".cache" / f"pooled-{args.encoder}"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path} — run --mode build_vocab first.")
    if not vocab_path.exists():
        raise SystemExit(f"missing {vocab_path} — run --mode build_vocab first.")
    if not cache_dir.exists():
        raise SystemExit(
            f"missing {cache_dir} — run --mode build_features first."
        )
    manifest = TaggerManifest.from_path(manifest_path)
    with open(vocab_path) as f:
        vocab_dict = json.load(f)
    train_ds = CachedFeatureDataset(manifest, cache_dir, stems_subset=manifest.train_stems)
    val_ds = CachedFeatureDataset(manifest, cache_dir, stems_subset=manifest.val_stems)
    logger.info(
        "train (cached features): N=%d  val: N=%d  d_in=%d  n_tags=%d  n_ratings=%d",
        len(train_ds),
        len(val_ds),
        train_ds.d_in,
        train_ds.n_tags,
        train_ds.n_ratings,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = AnimaTaggerConfig(
        d_in=train_ds.d_in,
        n_tags=train_ds.n_tags,
        n_ratings=train_ds.n_ratings,
        d_hidden=args.d_hidden,
        dropout=args.dropout,
    )
    model = AnimaTaggerHead(cfg).to(device)

    # All training/val tensors fit in VRAM trivially (~50 MB) — push them
    # once instead of per-batch.
    train_feats = train_ds.features.to(device)
    train_mh = train_ds.multi_hot.to(device)
    train_rate = train_ds.rating_idx.to(device)
    val_feats = val_ds.features.to(device)
    val_mh = val_ds.multi_hot.to(device)
    val_rate = val_ds.rating_idx.to(device)

    router = GroupRouter.from_vocab(vocab_dict, train_mh, device=device)
    rating_w = rating_class_weights(train_rate, train_ds.n_ratings).to(device)
    ce = torch.nn.CrossEntropyLoss(weight=rating_w)
    if router.is_active():
        n_softmax_tags = (
            int(router.softmax_member_indices.numel())
            if router.softmax_member_indices is not None else 0
        )
        logger.info(
            "groups active: %d softmax groups (%d softmax-member tags / %d total)",
            len(router.softmax_groups), n_softmax_tags, train_ds.n_tags,
        )
        for g in router.softmax_groups:
            logger.info(
                "  %-14s mode=%-18s K=%d  escape=%d",
                g.name, g.mode, int(g.tag_indices.numel()),
                int(g.escape_indices.numel()),
            )
    else:
        logger.info("no typed groups — pure BCE on every tag")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    n_train = len(train_ds)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    best_f1 = -1.0
    best_state: Dict[str, torch.Tensor] = {}
    history: List[Dict[str, float]] = []

    from tqdm import tqdm as _tqdm

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, generator=rng)
        ep_loss = 0.0
        ep_tag_loss = 0.0
        ep_rate_loss = 0.0
        n_batches = 0
        n_steps = (n_train + args.batch_size - 1) // args.batch_size
        bar = _tqdm(
            range(0, n_train, args.batch_size),
            total=n_steps,
            desc=f"ep {epoch + 1}/{args.epochs}",
            leave=False,
            unit="step",
        )
        for start in bar:
            idx = perm[start : start + args.batch_size]
            feat = train_feats[idx]
            mh = train_mh[idx]
            rate = train_rate[idx]
            tag_logits, rating_logits = model(feat)
            l_tag, _per_group = compute_grouped_loss(tag_logits, mh, router)
            l_rate = ce(rating_logits, rate)
            loss = l_tag + args.lambda_rating * l_rate
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            ep_tag_loss += l_tag.item()
            ep_rate_loss += l_rate.item()
            n_batches += 1
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                tag=f"{l_tag.item():.4f}",
                rate=f"{l_rate.item():.4f}",
            )
        sched.step()
        avg_loss = ep_loss / max(n_batches, 1)
        avg_tag = ep_tag_loss / max(n_batches, 1)
        avg_rate = ep_rate_loss / max(n_batches, 1)
        val_metrics = eval_split(
            model, val_feats, val_mh, val_rate,
            ce=ce, lambda_rating=args.lambda_rating, router=router,
        )
        logger.info(
            "epoch %2d/%d  loss=%.4f (tag=%.4f rate=%.4f)  "
            "val_loss=%.4f (tag=%.4f rate=%.4f)  "
            "val_f1=%.4f  val_p=%.4f  val_r=%.4f  rate_acc=%.4f  lr=%.2e",
            epoch + 1,
            args.epochs,
            avg_loss,
            avg_tag,
            avg_rate,
            val_metrics["val_loss"],
            val_metrics["val_tag_loss"],
            val_metrics["val_rate_loss"],
            val_metrics["macro_f1"],
            val_metrics["macro_precision"],
            val_metrics["macro_recall"],
            val_metrics["rating_acc"],
            sched.get_last_lr()[0],
        )
        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "tag_loss": avg_tag,
            "rate_loss": avg_rate,
            **val_metrics,
        })
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if not best_state:
        raise SystemExit("no epochs ran — empty training set?")

    # Save best checkpoint + config.
    ckpt_path = out_dir / "model.safetensors"
    cfg_path = out_dir / "config.json"
    history_path = out_dir / "train_history.json"
    st_save(best_state, str(ckpt_path))
    with open(cfg_path, "w") as f:
        json.dump(
            {
                "model": cfg.to_dict(),
                "encoder": args.encoder,
                "d_in": train_ds.d_in,
                "best_val_macro_f1": best_f1,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "lambda_rating": args.lambda_rating,
                "seed": args.seed,
                "pe_lora": False,
            },
            f,
            indent=2,
        )
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    plot_path = out_dir / "train_history.png"
    save_history_plot(history, plot_path)
    logger.info(
        "wrote %s / %s / %s / %s", ckpt_path, cfg_path, history_path, plot_path
    )
    print(f"  best val macro_f1: {best_f1:.4f}")
