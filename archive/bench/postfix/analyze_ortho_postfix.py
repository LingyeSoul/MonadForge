#!/usr/bin/env python
"""Diagnostic for ortho-postfix checkpoints (mode=postfix, ortho=true).

Companion to analyze_sigma_tokens.py / analyze_cond_postfix.py — those probe
caption-conditional or σ-conditional behavior. Ortho-postfix is structurally
caption-independent (one shared K×D postfix, like legacy mode=postfix), so
the per-caption / per-σ variance probes are degenerate. What's worth checking
on this variant is that the structural orthogonality survived save/load
roundtrip, that lambda_global is non-trivial (not pinned at zero by the
optimizer), and that the per-slot T5-token NN actually varies across slots
(the K=1-collapse signature for the legacy postfix was identical-top-k-
for-every-slot).

C1 layout (current): single scalar `lambda_global` enforces uniform per-slot
magnitude. Orthogonality target is `‖postfix @ postfix.T - lambda_global² · I‖_F`
(was `diag(lambda_slot²)` in v1). The `lambda_slot` distribution check from
v1 is gone — vacuously satisfied when magnitudes are constant by construction.

Validation criteria (post-C1) from `docs/proposal/orthogonal_postfix.md`:
  • ‖postfix @ postfix.T - lambda_global² · I‖_F < 1e-4
  • lambda_global magnitude non-trivial (|lambda_global| > 1e-3, i.e. the
    optimizer didn't kill the entire postfix)
  • top-k T5 tokens vary across k (not collapsed to identical-top-k)

Usage:
    python archive/bench/postfix/analyze_ortho_postfix.py \\
        --postfix_weight output/ckpt/anima_postfix_ortho_v2.safetensors \\
        --dataset_dir post_image_dataset/lora \\
        --num_captions 256 \\
        --out_json bench/postfix_ortho/results/<run>/analyze.json
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from library.anima import weights as anima_utils
from library.log import setup_logging
from networks.methods import postfix as postfix_anima

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--postfix_weight", default="output/ckpt/anima_postfix_ortho_v2.safetensors",
        help="ortho-postfix safetensors checkpoint",
    )
    p.add_argument(
        "--dataset_dir", default="post_image_dataset/lora",
        help="Directory with <stem>_anima_te.safetensors files (used for the T5 lexicon only)",
    )
    p.add_argument("--num_captions", type=int, default=256,
                   help="Cached TE files to seed the T5 NN-probe lexicon")
    p.add_argument("--min_count", type=int, default=3)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_t5", action="store_true",
                   help="Skip the T5 NN probe (Part B); useful if no cached TE corpus is handy")
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def load_network(weight_path, device):
    network, _ = postfix_anima.create_network_from_weights(
        multiplier=1.0, file=weight_path, ae=None, text_encoders=None, unet=None,
    )
    if not network.ortho:
        raise ValueError(
            f"This bench expects an ortho-postfix checkpoint (ss_ortho=true), got "
            f"ortho={network.ortho!r} mode={network.mode!r}. Use analyze_cond_postfix.py "
            "for cond / cond-timestep, or analyze_sigma_tokens.py for cond-timestep."
        )
    network.load_weights(weight_path)
    network.to(device).eval()
    for p in network.parameters():
        p.requires_grad_(False)
    return network


def find_cached_te(dataset_dir, n, seed):
    files = sorted(glob.glob(os.path.join(dataset_dir, "*_anima_te.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *_anima_te.safetensors in {dataset_dir}")
    rng = np.random.default_rng(seed)
    if len(files) > n:
        idx = rng.choice(len(files), size=n, replace=False)
        files = [files[i] for i in sorted(idx.tolist())]
    return files


def load_te(path):
    sd = load_file(path)
    emb = sd["crossattn_emb_v0"].float()
    ids = sd["t5_input_ids_v0"].long()
    mask = sd["attn_mask_v0"].bool()
    seqlen = int(mask.sum().item())
    return emb, ids, mask, seqlen


def build_lexicon(cached_files, min_count):
    sums, counts, D = {}, {}, None
    for path in cached_files:
        emb, ids, mask, seqlen = load_te(path)
        if seqlen == 0:
            continue
        if D is None:
            D = emb.shape[-1]
        for pos in range(seqlen):
            tok = int(ids[pos].item())
            v = emb[pos]
            if tok in sums:
                sums[tok] += v
                counts[tok] += 1
            else:
                sums[tok] = v.clone()
                counts[tok] = 1
    filt = sorted(t for t, c in counts.items() if c >= min_count)
    if not filt:
        raise RuntimeError(f"no tokens with >= {min_count} occurrences")
    means = torch.stack([sums[t] / counts[t] for t in filt], dim=0)
    cnts = np.array([counts[t] for t in filt], dtype=np.int64)
    return filt, means, cnts


def top_k_nearest(q, lex_norm, k):
    qn = F.normalize(q, dim=-1)
    if qn.dim() == 1:
        qn = qn.unsqueeze(0)
    sims = qn @ lex_norm.T
    cos, idx = sims.topk(k=k, dim=-1)
    return idx.numpy(), cos.numpy()


def fmt_toks(ids, cos, tok):
    return " ".join(
        f"{tok.convert_ids_to_tokens([int(i)])[0]!s}({c:+.3f})"
        for i, c in zip(ids, cos)
    )


def run_cond_ortho_analysis(network, args, K, D, device):
    """cond+ortho path: per-caption postfix(c) — different gates than postfix-mode.

    Materialize postfix(c) for N captions, check
      (a) per-caption ortho residual `‖gram(c) - λ(c)² · I‖_F < 1e-4` (every caption)
      (b) λ(c) distribution across captions: alive count + spread
      (c) cross-caption postfix diversity (the §C diagnostic): mean off-diagonal
          cosine of flattened postfix(c) tensors. Collapsed cond-mode v1 had
          near-1.0 here; structural ortho should disagree across captions.
    """
    n_skew = K * (K - 1) // 2
    cached = find_cached_te(args.dataset_dir, args.num_captions, args.seed)
    logger.info(f"cond+ortho: materializing postfix(c) for {len(cached)} captions")

    pooled_list: list[torch.Tensor] = []
    for path in cached:
        emb, _ids, mask, seqlen = load_te(path)
        if seqlen == 0:
            continue
        # Mean-pool content slots — same as append_postfix's pool.
        pooled_list.append(emb[:seqlen].float().mean(dim=0))
    pooled = torch.stack(pooled_list, dim=0).to(device)  # (N, D)
    N = pooled.shape[0]

    with torch.no_grad():
        cond_out = network.cond_mlp(pooled.to(next(network.cond_mlp.parameters()).dtype))
    cond_out = cond_out.float()
    S_seed = cond_out[:, :n_skew]
    lam_c = cond_out[:, -1]

    S_c = pooled.new_zeros(N, K, K, dtype=torch.float32)
    triu_i = network._S_triu_i.to(device)
    triu_j = network._S_triu_j.to(device)
    S_c[:, triu_i, triu_j] = S_seed
    A = S_c - S_c.transpose(-1, -2)
    eye = torch.eye(K, device=device, dtype=torch.float32)
    R = torch.linalg.solve(eye + A, eye - A)
    basis = network.postfix_basis.float()
    postfix = (R @ basis) * lam_c[:, None, None]  # (N, K, D)

    # (a) per-caption orthogonality residuals
    residuals = []
    for n in range(N):
        gram = postfix[n] @ postfix[n].T
        expected = (lam_c[n].pow(2)) * eye
        residuals.append((gram - expected).norm().item())
    residuals_t = torch.tensor(residuals)
    pc_max = float(residuals_t.max().item())
    pc_mean = float(residuals_t.mean().item())
    pc_pass_frac = float((residuals_t < 1e-4).float().mean().item())
    pc_ortho_pass = pc_pass_frac >= 0.99  # tolerate <1% numerical outliers

    # (b) λ(c) distribution
    abs_lam = lam_c.abs().cpu()
    alive = (abs_lam > 1e-3).sum().item()
    lam_alive_frac = alive / N
    lam_min = float(abs_lam.min().item())
    lam_max = float(abs_lam.max().item())
    lam_mean = float(abs_lam.mean().item())
    lam_std = float(abs_lam.std().item())
    lam_alive_pass = lam_alive_frac >= 0.8

    # (c) cross-caption diversity (§C): pairwise cos of flattened postfix
    flat = postfix.reshape(N, -1)
    flat_n = F.normalize(flat, dim=-1)
    cos_mat = (flat_n @ flat_n.T).cpu()
    off_mask = ~torch.eye(N, dtype=torch.bool)
    off = cos_mat[off_mask]
    cross_cos_mean = float(off.mean().item())
    cross_cos_max = float(off.max().item())
    cross_cos_min = float(off.min().item())
    diversity_pass = cross_cos_mean < 0.95

    print("\n" + "=" * 78)
    print(f"cond+ortho analysis — {os.path.basename(args.postfix_weight)}")
    print("=" * 78)
    print(f"  K={K}  D={D}  basis={network.ortho_basis_kind}  N_captions={N}")
    print("\n  (a) Per-caption structural orthogonality")
    print(f"    ‖gram - λ(c)² · I‖_F   max={pc_max:.3e}   mean={pc_mean:.3e}")
    print(f"    fraction passing < 1e-4: {pc_pass_frac:.3f} → "
          f"{'PASS' if pc_ortho_pass else 'FAIL'}")
    print("\n  (b) λ(c) distribution across captions")
    print(f"    |λ(c)| min={lam_min:.4f}  max={lam_max:.4f}  mean={lam_mean:.4f}  std={lam_std:.4f}")
    print(f"    alive (|λ| > 1e-3): {alive}/{N} = {lam_alive_frac:.2%} → "
          f"{'PASS' if lam_alive_pass else 'FAIL'}")
    print("\n  (c) Cross-caption postfix diversity (§C diagnostic)")
    print(f"    pairwise cosine: mean={cross_cos_mean:+.4f}  min={cross_cos_min:+.4f}  max={cross_cos_max:+.4f}")
    if diversity_pass:
        print("    → DIVERSE (mean < 0.95): captions produce different postfixes. PASS")
    else:
        print("    → COLLAPSED (mean ≥ 0.95): captions produce near-identical postfixes. FAIL")

    payload = {
        "postfix_weight": args.postfix_weight,
        "K": K,
        "D": D,
        "ortho_basis_kind": network.ortho_basis_kind,
        "mode": "cond",
        "ortho_lambda_kind": "per_caption",
        "n_captions": N,
        "per_caption_orthogonality": {
            "residual_max": pc_max,
            "residual_mean": pc_mean,
            "fraction_pass_1e_4": pc_pass_frac,
            "pass": bool(pc_ortho_pass),
        },
        "lambda_distribution": {
            "abs_min": lam_min,
            "abs_max": lam_max,
            "abs_mean": lam_mean,
            "abs_std": lam_std,
            "alive_count": int(alive),
            "alive_fraction": lam_alive_frac,
            "alive_pass": bool(lam_alive_pass),
        },
        "cross_caption_diversity": {
            "cos_mean": cross_cos_mean,
            "cos_min": cross_cos_min,
            "cos_max": cross_cos_max,
            "pass": bool(diversity_pass),
        },
    }
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"wrote {args.out_json}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    logger.info(f"loading ortho-postfix from {args.postfix_weight}")
    network = load_network(args.postfix_weight, device)
    K, D = network.num_postfix_tokens, network.embed_dim
    logger.info(
        f"postfix: mode={network.mode} ortho={network.ortho} "
        f"basis_kind={network.ortho_basis_kind} K={K} D={D}"
    )

    if network.mode == "cond":
        # cond+ortho takes a different path entirely — postfix is per-caption,
        # so the per-slot lambda_global / single ortho residual gates don't apply.
        return run_cond_ortho_analysis(network, args, K, D, device)
    if network.mode != "postfix":
        raise ValueError(
            f"Unsupported network mode={network.mode!r} ortho={network.ortho!r}. "
            "Only postfix-mode ortho (postfix_ortho_v2) and cond-mode ortho "
            "(postfix_ortho_cond_v2) are wired."
        )

    # ----- structural orthogonality (C1: scalar lambda_global) -------------
    with torch.no_grad():
        eff = network._effective_postfix().cpu()  # (K, D), fp32
    lam_g = float(network.lambda_global.detach().float().cpu().item())
    gram = eff @ eff.T
    eye = torch.eye(K)
    expected_gram = (lam_g ** 2) * eye
    diag = torch.diag(gram)
    diag_err = (diag - (lam_g ** 2)).abs().max().item()
    off_diag = gram - torch.diag(diag)
    off_max = off_diag.abs().max().item()
    ortho_resid = (gram - expected_gram).norm().item()
    ortho_pass = ortho_resid < 1e-4

    # |lambda_global| > 1e-3 = optimizer didn't kill the postfix outright.
    # No max/min ratio gate — magnitudes are uniform by construction in C1.
    lam_alive = abs(lam_g) > 1e-3

    print("\n" + "=" * 78)
    print(f"ortho-postfix analysis — {os.path.basename(args.postfix_weight)}")
    print("=" * 78)
    print(f"  K={K}  D={D}  basis={network.ortho_basis_kind}")
    print("\n  Structural orthogonality (after save/load roundtrip)")
    print(f"    ‖postfix @ postfix.T - λ_g² · I‖_F = {ortho_resid:.3e}   "
          f"(target < 1e-4) → {'PASS' if ortho_pass else 'FAIL'}")
    print(f"    diag mismatch max |·|              = {diag_err:.3e}")
    print(f"    off-diagonal max |·|               = {off_max:.3e}")

    print("\n  Global magnitude (uniform per slot by construction)")
    print(f"    lambda_global = {lam_g:+.4f}   (|λ_g| > 1e-3 = alive)")
    print(f"    → {'ALIVE' if lam_alive else 'PINNED-NEAR-ZERO'}")

    # ----- T5 token NN probe (Part B) --------------------------------------
    per_slot_topk = None
    pooled_topk = None
    lex_size = 0
    if not args.skip_t5:
        try:
            logger.info("building T5-token lexicon")
            tokenizer = anima_utils.load_t5_tokenizer()
            cached = find_cached_te(args.dataset_dir, args.num_captions, args.seed)
            logger.info(f"using {len(cached)} cached TE files")
            lex_ids, lex_vecs, _lex_counts = build_lexicon(cached, args.min_count)
            lex_norm = F.normalize(lex_vecs, dim=-1)
            lex_size = len(lex_ids)
            logger.info(f"lexicon: {lex_size} tokens (min_count={args.min_count})")

            # Per-slot NN — the K=1-collapse signature was identical-top-k for
            # every slot. Under structural orthogonality these should disagree.
            # In C1, slot magnitudes are uniform (|λ_g|), so the per-slot |λ|
            # column collapses to a constant; printed once in the header for
            # context and the per-row slot-level magnitude is dropped.
            print("\n  Per-slot T5 NN probe (collapse signature: identical top-1 for every slot)")
            print(f"  uniform magnitude per slot: |lambda_global| = {abs(lam_g):.4f}")
            print(f"  slot   top-{args.top_k} nearest T5 tokens")
            per_slot_topk = []
            top1_set: set[int] = set()
            for slot in range(K):
                idx, cos = top_k_nearest(eff[slot], lex_norm, args.top_k)
                per_slot_topk.append({
                    "slot": slot,
                    "ids": idx[0].tolist(),
                    "tokens": tokenizer.convert_ids_to_tokens([int(x) for x in idx[0]]),
                    "cos": cos[0].tolist(),
                })
                top1_set.add(int(idx[0][0]))
                if slot < 16:
                    print(
                        f"   {slot:3d}  {fmt_toks(idx[0], cos[0], tokenizer)}"
                    )
            distinct_top1 = len(top1_set)
            print(f"\n   distinct top-1 tokens across {K} slots: {distinct_top1} / {K}")
            if distinct_top1 == 1:
                print("   → COLLAPSED: every slot's nearest neighbor is the same token. "
                      "Either ortho param tensor is K-rank but cross-attention reads "
                      "rank-1 (splice-position symmetry — see proposal §B), or λ is "
                      "concentrated on one slot (check distribution above).")
            elif distinct_top1 < K // 4:
                print(f"   → mostly-collapsed ({distinct_top1} unique top-1 across {K} slots).")
            else:
                print(f"   → diverse ({distinct_top1}/{K} unique top-1 tokens across slots).")

            # Pooled-over-slots NN, for comparability with the cond-mode analyzer.
            pooled = eff.mean(dim=0)  # [D]
            pidx, pcos = top_k_nearest(pooled, lex_norm, args.top_k)
            pooled_topk = {
                "ids": pidx[0].tolist(),
                "tokens": tokenizer.convert_ids_to_tokens([int(x) for x in pidx[0]]),
                "cos": pcos[0].tolist(),
            }
            print("\n  Pooled (mean over K slots) NN")
            print(f"    top-{args.top_k}: {fmt_toks(pidx[0], pcos[0], tokenizer)}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"\n  [T5 NN probe skipped: {e}]")

    # ----- JSON ------------------------------------------------------------
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            "postfix_weight": args.postfix_weight,
            "K": K,
            "D": D,
            "ortho_basis_kind": network.ortho_basis_kind,
            "structural_orthogonality": {
                "frobenius_residual": ortho_resid,
                "diag_mismatch_max": diag_err,
                "off_diag_max": off_max,
                "pass": bool(ortho_pass),
            },
            "lambda_global": {
                "value": lam_g,
                "abs_value": abs(lam_g),
                "alive_pass": bool(lam_alive),
            },
            "t5_nn_probe": {
                "lexicon_size": lex_size,
                "per_slot_topk": per_slot_topk,
                "pooled_topk": pooled_topk,
            } if per_slot_topk is not None else None,
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
