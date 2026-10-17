#!/usr/bin/env python3
"""Exact ΔW-average soup of plain-LoRA checkpoints via rank concatenation.

Souping LoRAs *parameterwise* (averaging A's and B's) is wrong twice over:
``avg(B) @ avg(A) != avg(B @ A)``, and with ``down_init="weight_svd"`` the
randomized SVD's per-vector sign ambiguity means row-wise A averaging can
actively cancel rows. So we soup at the ΔW level, which is invariant to any
per-ingredient ``(A, B)`` reparameterization — and a weighted average of
rank-r deltas is *exactly* representable as one rank-``sum(r_i)`` LoRA:

    dW_soup = sum_i w_i * scale_i * up_i @ down_i'
            = [w_1*scale_1*up_1 | w_2*scale_2*up_2 | ...] @ [down_1' ; down_2' ; ...]

(block concatenation — no SVD truncation, no approximation), where
``down_i' = down_i * inv_scale_i`` folds the persisted channel-scaling buffer
back in so the soup applies to raw inputs (mirrors ``LoRAModule.get_weight``),
and the soup's alpha is set to its rank (scale = 1). No ``inv_scale`` keys are
carried, so the loaded soup module takes the raw-input path.

Plain-LoRA checkpoints only (the lora.toml weight_svd / T-LoRA family —
T-LoRA is training-only so its checkpoints are plain). Hydra / chimera /
stacked-experts / ortho key shapes are refused loudly.

Usage
-----
    # uniform soup
    python bench/uncond_soup/soup.py \
        --ckpts output/ckpt/a.safetensors output/ckpt/b.safetensors \
        --out /tmp/soup.safetensors

    # lambda=0.25 interpolation of a pair (LMC probe point)
    python bench/uncond_soup/soup.py \
        --ckpts a.safetensors b.safetensors --weights 0.75 0.25 --out mid.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# Suffixes a plain-LoRA checkpoint may carry per module. Anything else means a
# non-plain variant (or a new format) — fail loudly rather than soup garbage.
_ALLOWED_SUFFIXES = ("lora_down.weight", "lora_up.weight", "alpha", "inv_scale")
# Non-module top-level keys we refuse (registers / routers / adapters change
# inference behavior and cannot be averaged this way).
_REFUSED_TOP_LEVEL = ("register_tokens",)


def _split_key(key: str) -> tuple[str, str]:
    """``lora_unet_..._q_proj.lora_down.weight`` -> (module, suffix)."""
    head, _, _ = key.partition(".")
    return head, key[len(head) + 1 :]


def _group_modules(sd: dict, path: str) -> dict[str, dict[str, torch.Tensor]]:
    modules: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in sd.items():
        if key in _REFUSED_TOP_LEVEL or "." not in key:
            raise ValueError(
                f"{path}: key {key!r} is not a plain-LoRA module key — this "
                "checkpoint carries state the ΔW soup cannot represent."
            )
        module, suffix = _split_key(key)
        if suffix not in _ALLOWED_SUFFIXES:
            raise ValueError(
                f"{path}: unrecognized key suffix {suffix!r} on {module!r} — "
                "only plain LoRA checkpoints (lora_down/lora_up/alpha/inv_scale) "
                "can be souped; Hydra/chimera/stacked-experts/ortho are refused."
            )
        modules.setdefault(module, {})[suffix] = value
    return modules


def _effective_factors(
    mod: dict[str, torch.Tensor], module: str, path: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """(down', up') in float32 such that ΔW = up' @ down' exactly.

    Folds alpha/r scaling and the channel-scaling ``inv_scale`` buffer
    (mirrors ``LoRAModule.get_weight``). Linear (2-D) only.
    """
    try:
        down = mod["lora_down.weight"].to(torch.float32)
        up = mod["lora_up.weight"].to(torch.float32)
    except KeyError as exc:
        raise ValueError(f"{path}: {module} is missing {exc} — truncated file?")
    if down.dim() != 2 or up.dim() != 2:
        raise ValueError(
            f"{path}: {module} has non-Linear factors "
            f"(down {tuple(down.shape)}, up {tuple(up.shape)}) — Conv LoRA is "
            "not supported by the soup (the DiT LoRA family is Linear-only)."
        )
    rank = down.shape[0]
    alpha = mod.get("alpha")
    scale = (float(alpha) / rank) if alpha is not None else 1.0
    inv_scale = mod.get("inv_scale")
    if inv_scale is not None:
        down = down * inv_scale.to(torch.float32).unsqueeze(0)
    return down, up * scale


def soup_state_dicts(
    sds: list[dict],
    weights: list[float] | None = None,
    *,
    paths: list[str] | None = None,
    out_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Build the exact rank-concatenated soup state dict.

    ``weights`` default to uniform 1/N. They are used as-is (not normalized),
    so ``[0.5, 0.5]`` on a pair is both the uniform soup and the lambda=0.5
    LMC midpoint.
    """
    n = len(sds)
    if n < 2:
        raise ValueError("Need >= 2 ingredient checkpoints to soup.")
    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"{len(weights)} weights for {n} checkpoints.")
    paths = paths or [f"<sd {i}>" for i in range(n)]

    grouped = [_group_modules(sd, p) for sd, p in zip(sds, paths)]
    module_names = set(grouped[0])
    for g, p in zip(grouped[1:], paths[1:]):
        if set(g) != module_names:
            missing = module_names.symmetric_difference(g)
            raise ValueError(
                f"Ingredient module sets differ ({p}): {sorted(missing)[:5]}... — "
                "all ingredients must come from the same recipe."
            )

    if out_dtype is None:
        out_dtype = sds[0][next(iter(sds[0]))].dtype

    soup: dict[str, torch.Tensor] = {}
    for module in sorted(module_names):
        downs, ups = [], []
        for g, w, p in zip(grouped, weights, paths):
            down, up = _effective_factors(g[module], module, p)
            downs.append(down)
            ups.append(up * w)
        soup_down = torch.cat(downs, dim=0)
        soup_up = torch.cat(ups, dim=1)
        rank = soup_down.shape[0]
        soup[f"{module}.lora_down.weight"] = soup_down.to(out_dtype)
        soup[f"{module}.lora_up.weight"] = soup_up.to(out_dtype)
        # alpha = rank -> scale 1 (per-ingredient scales already folded into up).
        soup[f"{module}.alpha"] = torch.tensor(float(rank), dtype=torch.float32)
    return soup


def build_soup_file(
    ckpts: list[str], out: str, weights: list[float] | None = None
) -> Path:
    """Load ingredient .safetensors, soup, write ``out`` with updated metadata."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    sds = [load_file(c) for c in ckpts]
    with safe_open(ckpts[0], framework="pt") as f:
        metadata = dict(f.metadata() or {})

    soup = soup_state_dicts(sds, weights, paths=list(ckpts))

    ranks = {v.shape[0] for k, v in soup.items() if k.endswith(".lora_down.weight")}
    rank_str = str(ranks.pop()) if len(ranks) == 1 else "Dynamic"
    metadata["ss_network_dim"] = rank_str
    metadata["ss_network_alpha"] = rank_str
    metadata["ss_soup_ingredients"] = json.dumps(
        {
            "ckpts": [Path(c).name for c in ckpts],
            "weights": weights or [1.0 / len(ckpts)] * len(ckpts),
        }
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(soup, str(out_path), metadata=metadata)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Per-ingredient weights (default uniform 1/N). Used as-is.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = build_soup_file(args.ckpts, args.out, args.weights)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
