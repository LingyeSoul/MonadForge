"""Does static-shape zero-padding leak into the real-token velocity under flash?

Context — `issues.md` #1 and CLAUDE.md's "constant-token bucketing" invariant.
The production path (`configs/base.toml`: `attn_mode="flash"`, `static_token_count=4096`)
zero-pads every bucket's patch sequence up to a fixed token count and runs flash
self-attention *without* a padding mask. The flex `BlockMask` that excludes
padded positions is only built when `attn_mode=="flex"` (`library/anima/models.py`
~L1766). The standing assumption is that the zero-padded patch tokens act as
"attention sinks" the pretrained model tolerates — same contract as the text
padding invariant.

But padded patch tokens are NOT zero where it matters: AdaLN applies a
σ-dependent shift and the Q/K/V projections carry bias, so a zero-input token
still emits non-trivial K/V into self-attention. Whether those leak into the
real tokens' output is an empirical question. This probe measures it.

Method — for a set of real buckets (token counts just under 4096) and a few
synthetic latents (larger gaps), with identical inputs we compute three forwards
of the base DiT:

  * ref       static_token_count=None, flash  — native seq_len, no padding (truth)
  * ref2      same again                       — bf16/kernel nondeterminism FLOOR
  * pad_flash static_token_count=4096, flash   — production path (unmasked padding)
  * pad_flex  static_token_count=4096, flex    — padding masked out (optional, --flex)

`forward_mini_train_dit` strips the static padding before returning, so all four
outputs are (B,16,1,H,W) and compare elementwise on the real tokens. Metric is
relative L2 error vs `ref`, in fp32. The verdict: if `pad_flash`'s error sits at
the `ref2` floor, the sink assumption holds and the gap is harmless; if it rises
with the gap size, padding leaks and issue #1 has teeth.

Run (repo root):
    python -m bench.static_padding.probe_pad_leak
    python -m bench.static_padding.probe_pad_leak --flex --sigmas 0.2 0.5 0.8
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch

# Defensive: allow `python bench/static_padding/probe_pad_leak.py` as well as -m.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima.weights import load_anima_model  # noqa: E402

log = logging.getLogger("bench.static_padding")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
STATIC_TOKENS = 4096
TEXT_LEN = 226  # representative crossattn length; held fixed across all runs

# (label, H_lat, W_lat) — token count = (H_lat/2)*(W_lat/2). VAE-latent dims.
# Real CONSTANT_TOKEN_BUCKETS (just under 4096) plus synthetic large-gap probes.
CONFIGS: list[tuple[str, int, int]] = [
    ("bucket_4096_gap0", 128, 128),   # 1024x1024  — control, gap 0
    ("bucket_4080", 120, 136),        # 960x1088   — gap 16
    ("bucket_4056", 104, 156),        # 832x1248   — gap 40
    ("bucket_4050", 90, 180),         # 720x1440   — gap 46
    ("bucket_4032", 112, 144),        # 896x1152   — gap 64
    ("synthetic_3072_gap1024", 96, 128),
    ("synthetic_2048_gap2048", 64, 128),
    ("synthetic_1024_gap3072", 64, 64),
]


def _make_inputs(H, W, sigma, device, dtype, seed):
    """Identical, seeded inputs reused across every variant for this config."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    lat = torch.randn(1, 16, 1, H, W, generator=g, dtype=torch.float32)
    txt = torch.randn(1, TEXT_LEN, 1024, generator=g, dtype=torch.float32)
    return {
        "x_B_C_T_H_W": lat.to(device, dtype),
        "timesteps_B_T": torch.tensor([sigma], device=device, dtype=dtype),
        "crossattn_emb": txt.to(device, dtype),
        # concat_padding_mask=True → a mask channel is required; all-zero = all valid.
        "padding_mask": torch.zeros(1, 1, H, W, device=device, dtype=dtype),
    }


@torch.no_grad()
def _forward(model, inputs, static, attn_mode, dtype, crossattn_seqlens=None, pad=True):
    model.set_static_token_count(static, pad=pad)
    model.attn_mode = attn_mode
    with torch.autocast("cuda", dtype=dtype):
        out = model.forward_mini_train_dit(
            inputs["x_B_C_T_H_W"],
            inputs["timesteps_B_T"],
            inputs["crossattn_emb"],
            padding_mask=inputs["padding_mask"],
            crossattn_seqlens=crossattn_seqlens,
            skip_pooled_text_proj=True,
        )
    return out.float()


def _rel(a, ref):
    """Relative L2 error, max-abs diff, and cosine — a, ref already fp32."""
    diff = (a - ref).flatten()
    ref_n = ref.flatten().norm().item()
    rel = (diff.norm().item() / ref_n) if ref_n > 0 else float("nan")
    cos = torch.nn.functional.cosine_similarity(
        a.flatten(), ref.flatten(), dim=0
    ).item()
    return rel, diff.abs().max().item(), cos


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dit_path", default=DEFAULT_DIT)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flex", action="store_true", help="also run masked flex pad")
    ap.add_argument("--label", default="pad-leak")
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    log.info("Loading DiT %s (flash, bf16)…", args.dit_path)
    model = load_anima_model(
        device, args.dit_path, attn_mode="flash",
        loading_device=device, dit_weight_dtype=dtype,
    )
    model.to(device)  # move non-persistent buffers (mod-guidance) off CPU
    model.reset_mod_guidance()  # ensure guidance is identity (zeros)
    model.eval()

    rows: list[dict] = []
    for label, H, W in CONFIGS:
        seq_len = (H // 2) * (W // 2)
        gap = STATIC_TOKENS - seq_len
        if gap < 0:
            log.warning("skip %s: seq_len %d > %d", label, seq_len, STATIC_TOKENS)
            continue
        for sigma in args.sigmas:
            inp = _make_inputs(H, W, sigma, device, dtype, args.seed)
            ref = _forward(model, inp, None, "flash", dtype)
            ref2 = _forward(model, inp, None, "flash", dtype)
            pad_flash = _forward(model, inp, STATIC_TOKENS, "flash", dtype)
            # No-pad path: static count set but pad=False → native shape, flash.
            # Should match ref to the kernel floor (it's the same computation).
            nopad_flash = _forward(model, inp, STATIC_TOKENS, "flash", dtype, pad=False)

            floor_rel, floor_max, _ = _rel(ref2, ref)
            f_rel, f_max, f_cos = _rel(pad_flash, ref)
            np_rel, np_max, np_cos = _rel(nopad_flash, ref)
            row = {
                "config": label, "seq_len": seq_len, "gap": gap, "sigma": sigma,
                "floor_rel_l2": floor_rel, "floor_max_abs": floor_max,
                "pad_flash_rel_l2": f_rel, "pad_flash_max_abs": f_max,
                "pad_flash_cos": f_cos,
                "excess_over_floor": f_rel - floor_rel,
                "nopad_flash_rel_l2": np_rel, "nopad_flash_max_abs": np_max,
                "nopad_flash_cos": np_cos,
            }
            if args.flex:
                try:
                    seqlens = torch.tensor([TEXT_LEN], device=device, dtype=torch.int32)
                    pad_flex = _forward(
                        model, inp, STATIC_TOKENS, "flex", dtype, crossattn_seqlens=seqlens
                    )
                    x_rel, x_max, x_cos = _rel(pad_flex, ref)
                    row.update(pad_flex_rel_l2=x_rel, pad_flex_max_abs=x_max,
                               pad_flex_cos=x_cos)
                except Exception as e:  # flex_attention may be unavailable on this arch
                    log.warning("flex failed (%s): %s", label, e)
                    row.update(pad_flex_rel_l2=None)
            rows.append(row)
            log.info(
                "%-26s gap=%-4d σ=%.2f  flash_rel=%.2e (excess=%.2e)  "
                "nopad_rel=%.2e cos=%.6f",
                label, gap, sigma, f_rel, f_rel - floor_rel, np_rel, np_cos,
            )

    # Verdict: padding is harmless when flash excess-over-floor stays negligible.
    excess = [r["excess_over_floor"] for r in rows if r["gap"] > 0]
    worst = max(excess) if excess else 0.0
    gap0 = [r["pad_flash_rel_l2"] for r in rows if r["gap"] == 0]
    # The no-pad path should sit at the kernel floor for every gap — it's the
    # same computation as ref, just reached with static_token_count set + pad=False.
    nopad_excess = [
        r["nopad_flash_rel_l2"] - r["floor_rel_l2"] for r in rows if r["gap"] > 0
    ]
    nopad_worst = max(nopad_excess) if nopad_excess else 0.0
    verdict = "HARMLESS" if worst < 1e-3 else ("LEAKS" if worst > 1e-2 else "MARGINAL")
    nopad_verdict = "CLEAN" if nopad_worst < 1e-3 else "LEAKS"
    log.info("\n=== pad verdict: %s ===", verdict)
    log.info("max pad_flash excess-over-floor (gap>0): %.3e", worst)
    log.info("=== no-pad verdict: %s ===", nopad_verdict)
    log.info("max nopad_flash excess-over-floor (gap>0): %.3e", nopad_worst)
    if gap0:
        log.info("gap=0 control flash_rel (should ~= floor): %.3e", max(gap0))

    run_dir = make_run_dir("static_padding", label=args.label)
    csv_path = run_dir / "per_config.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    write_result(
        run_dir, script=__file__, args=args, label=args.label,
        metrics={
            "verdict": verdict,
            "max_flash_excess_over_floor": worst,
            "nopad_verdict": nopad_verdict,
            "max_nopad_excess_over_floor": nopad_worst,
            "threshold_harmless": 1e-3, "threshold_leaks": 1e-2,
            "n_configs": len(rows),
        },
        artifacts=["per_config.csv"], device=device,
    )
    log.info("wrote %s", run_dir)


if __name__ == "__main__":
    main()
