#!/usr/bin/env python
"""Sigma-sensitivity probe for the channel_scaling calibration.

Question: the shipped calibration (`channel_stats.safetensors`) pools per-input
channel mean|x| over 5 *uniform* sigmas [0.1,0.3,0.5,0.7,0.9]. Training instead
draws sigma ~ sigmoid(N(0,1)) (logit-normal, mass near 0.5). Does the per-channel
*shape* that the factory turns into a scale vector

    s = mean_abs.pow(alpha); s = s / s.mean()

actually move with sigma? If not, the sigma grid is irrelevant and we keep the
uniform-5 scheme. If yes, matching the training distribution is worth it.

Two measurements (one DiT load, ~100 forwards for 10 samples):
  (A) per-sigma profiles at the 5 fixed points -> cosine vs the uniform-5 pool
      (= exactly what the script ships). Low cosine == sigma-dependent shape.
  (B) a separate pool with sigma drawn from the TRAINING distribution
      sigmoid(scale*randn + bias) -> cosine vs the uniform-5 pool. This is the
      direct "would switching to logit-normal weighting change the shipped
      vector" number.

Cosine is reported at alpha=1.0 (raw mean|x|, the MOST discriminating case --
alpha<1 only compresses toward 1 and raises cosine) and at alpha=0.5 (documented
"sqrt balance"). Decision rule: if min cosine across modules stays >~0.99 at
alpha=1, the calibration is sigma-insensitive; keep uniform-5.
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_lora_input_channels import (  # noqa: E402
    classify_module,
    find_sample_stems,
    load_cached_te,
    load_latent_npz,
)

from bench._anima import DEFAULT_DIT  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dit", default=DEFAULT_DIT)
    p.add_argument("--dataset_dir", default="post_image_dataset/lora")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument(
        "--fixed_sigmas",
        default="0.1,0.3,0.5,0.7,0.9",
        help="The uniform grid the shipped calibration uses.",
    )
    p.add_argument(
        "--train_draws_per_sample",
        type=int,
        default=5,
        help="How many sigma ~ training-dist draws per sample for the matched pool.",
    )
    p.add_argument("--sigmoid_scale", type=float, default=1.0)
    p.add_argument("--sigmoid_bias", type=float, default=0.0)
    return p.parse_args()


# --- per-tag channel-sum accumulators -------------------------------------
class Accum:
    """sum|x| and token count per Linear module, under one tag (a sigma bucket)."""

    def __init__(self):
        self.sum_abs = {}
        self.count = {}

    def add(self, path, x):
        xf = x.detach().to(torch.float32).abs().reshape(-1, x.shape[-1])
        s = xf.sum(dim=0).double().cpu()
        if path not in self.sum_abs:
            self.sum_abs[path] = s
            self.count[path] = xf.shape[0]
        else:
            self.sum_abs[path] += s
            self.count[path] += xf.shape[0]

    def mean_abs(self):
        return {
            k: (self.sum_abs[k] / max(self.count[k], 1)).numpy()
            for k in self.sum_abs
            if self.count[k] > 0
        }


def install_hooks(model, current_tag):
    """current_tag is a 1-element list holding the live Accum to route into."""
    handles = []
    seen = set()

    def make_hook(path):
        def hook(_m, inputs):
            if not inputs:
                return
            current_tag[0].add(path, inputs[0])

        return hook

    for module_path, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        clean = module_path.replace("_orig_mod.", "")
        if clean in seen:
            continue
        seen.add(clean)
        handles.append(module.register_forward_pre_hook(make_hook(clean)))
    logger.info(f"hooked {len(seen)} Linear modules")
    return handles


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def scale_vec(mean_abs, alpha):
    s = np.power(np.clip(mean_abs, 1e-6, None), alpha)
    return s / max(s.mean(), 1e-12)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    stems = find_sample_stems(
        args.dataset_dir, args.num_samples, args.seed, per_artist=False
    )
    stems = stems[: args.num_samples]
    logger.info(f"{len(stems)} samples: {[s[0] for s in stems]}")

    fixed_sigmas = [float(x) for x in args.fixed_sigmas.split(",") if x.strip()]

    logger.info(f"loading DiT {args.dit}")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        loading_device=device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.eval().requires_grad_(False)
    anima.to(device)

    fixed_acc = {sg: Accum() for sg in fixed_sigmas}  # per-sigma
    train_acc = Accum()  # training-dist pool
    live = [None]
    install_hooks(anima, live)

    def forward_one(lat_4d, emb, pad, sigma):
        noise = torch.randn_like(lat_4d)
        sv = torch.tensor(float(sigma), device=device).view(1, 1, 1, 1)
        noisy = ((1.0 - sv) * lat_4d + sv * noise).to(torch.bfloat16).unsqueeze(2)
        t = torch.tensor([float(sigma)], device=device, dtype=torch.bfloat16)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _ = anima(noisy, t, emb, padding_mask=pad)

    with torch.no_grad():
        for stem, npz_path, te_path in stems:
            lat = load_latent_npz(npz_path).to(device)
            lat_4d = lat.unsqueeze(0).float()
            emb = load_cached_te(te_path).to(device, dtype=torch.bfloat16)
            h, w = lat_4d.shape[-2], lat_4d.shape[-1]
            pad = torch.zeros(1, 1, h, w, dtype=torch.bfloat16, device=device)

            for sg in fixed_sigmas:
                live[0] = fixed_acc[sg]
                forward_one(lat_4d, emb, pad, sg)

            # training-distribution draws -> one shared pool
            live[0] = train_acc
            z = rng.standard_normal(args.train_draws_per_sample)
            draws = 1.0 / (1.0 + np.exp(-(args.sigmoid_scale * z + args.sigmoid_bias)))
            for sg in draws:
                forward_one(lat_4d, emb, pad, float(sg))
            logger.info(f"  {stem}: train-draws sigma={np.round(draws,3).tolist()}")

    # --- assemble profiles ---
    fixed_ma = {sg: fixed_acc[sg].mean_abs() for sg in fixed_sigmas}
    train_ma = train_acc.mean_abs()

    # modules present everywhere
    common = set(train_ma)
    for sg in fixed_sigmas:
        common &= set(fixed_ma[sg])
    common = sorted(common)
    logger.info(f"{len(common)} modules common to all buckets")

    # uniform-5 pool = simple mean over fixed sigmas (equal token counts/sigma)
    uniform_ma = {
        k: np.mean([fixed_ma[sg][k] for sg in fixed_sigmas], axis=0) for k in common
    }

    print("\n" + "=" * 80)
    print("SIGMA-SENSITIVITY PROBE  (cosine of scale vector vs uniform-5 pool)")
    print(f"  samples={len(stems)}  fixed_sigmas={fixed_sigmas}")
    print(f"  train-dist: sigmoid({args.sigmoid_scale}*N(0,1)+{args.sigmoid_bias})")
    print("=" * 80)

    for alpha in (1.0, 0.5):
        print(f"\n--- alpha = {alpha} ---")

        # (A) per-sigma stability vs uniform pool
        print("  (A) per-fixed-sigma profile vs uniform-5 pool:")
        for sg in fixed_sigmas:
            cs = np.array(
                [
                    cos(scale_vec(fixed_ma[sg][k], alpha), scale_vec(uniform_ma[k], alpha))
                    for k in common
                ]
            )
            print(
                f"      sigma={sg:<4}  cos: mean={cs.mean():.4f} "
                f"min={cs.min():.4f}  p01={np.percentile(cs,1):.4f}"
            )

        # worst per-sigma module spread: how far apart are the two extreme sigmas?
        lo, hi = fixed_sigmas[0], fixed_sigmas[-1]
        cs_extreme = np.array(
            [cos(scale_vec(fixed_ma[lo][k], alpha), scale_vec(fixed_ma[hi][k], alpha)) for k in common]
        )
        worst = sorted(
            common,
            key=lambda k: cos(
                scale_vec(fixed_ma[lo][k], alpha), scale_vec(fixed_ma[hi][k], alpha)
            ),
        )[:8]
        print(
            f"    sigma={lo} vs sigma={hi} (extremes): mean={cs_extreme.mean():.4f} "
            f"min={cs_extreme.min():.4f}"
        )
        print("      worst-8 modules (cos, group):")
        for k in worst:
            c = cos(scale_vec(fixed_ma[lo][k], alpha), scale_vec(fixed_ma[hi][k], alpha))
            print(f"        {c:.4f}  {classify_module(k):<18} {k}")

        # (B) THE decision number: training-dist pool vs uniform-5 pool
        cs_train = np.array(
            [cos(scale_vec(train_ma[k], alpha), scale_vec(uniform_ma[k], alpha)) for k in common]
        )
        print("  (B) training-dist pool vs uniform-5 pool  <-- would switching change it?")
        print(
            f"      cos: mean={cs_train.mean():.4f}  min={cs_train.min():.4f} "
            f"p01={np.percentile(cs_train,1):.4f}  median={np.median(cs_train):.4f}"
        )
        # per-group breakdown
        by_group = defaultdict(list)
        for k in common:
            by_group[classify_module(k)].append(
                cos(scale_vec(train_ma[k], alpha), scale_vec(uniform_ma[k], alpha))
            )
        for g in sorted(by_group):
            arr = np.array(by_group[g])
            print(f"        {g:<18} n={len(arr):3d}  mean={arr.mean():.4f}  min={arr.min():.4f}")

    print("\nDecision: if (A)/(B) min-cos stay >~0.99 at alpha=1, keep uniform-5.")


if __name__ == "__main__":
    main()
