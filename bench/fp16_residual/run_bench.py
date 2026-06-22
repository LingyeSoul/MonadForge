"""fp16-safe residual accumulation bench (Tier 1.5 numerics gate).

The DiT residual stream exceeds fp16's 65504 ceiling late in the block stack
(``docs/findings/selfflow.md``); under fp16 autocast the Block residual adds,
FinalLayer AdaLN modulate, and final projection can overflow to inf→NaN from
step 0. ``Anima.enable_fp32_residual()`` keeps those numerically sensitive
residual/final-layer operations in fp32 while transformer-block matmuls still
run fp16 under autocast. This bench quantifies:

1. **Correctness**: fp16 + flag-on runs N steps finite; bf16 baseline finite.
   (The >65504 overflow itself only bites on a trained model at large
   resolution — the dispositive overflow regression is the unit test in
   ``tests/test_fp16_residual_safe.py``, which injects a >65504 sum directly
   into ``_residual_add``. This bench validates the *mechanism* wires through a
   real multi-block forward, not the overflow itself.)
2. **Speed**: per-step wall-time for fp16+flag vs bf16 — the flag's fp32
   promotion has a cost, but fp16 overall should still beat bf16 (the matmul
   speedup the flag exists to unlock on V100).

Runs on CPU with a tiny synthetic Anima so it's CI-friendly; for real overflow
reproduction + GPU timing, run on a V100 with a trained checkpoint.

Usage::

    uv run python -m bench.fp16_residual.run_bench [--steps N] [--num_blocks N]
    # or: .venv/Scripts/python.exe -m bench.fp16_residual.run_bench

Drops a ``result.json`` envelope (``bench/_common.py``) into
``bench/fp16_residual/results/<ts>[-<label>]/``.
"""

from __future__ import annotations

import argparse
import time

import torch

from bench._common import make_run_dir, write_result

_FP16_MAX = torch.finfo(torch.float16).max


def _build_tiny_anima(num_blocks: int):
    """A small but real Anima DiT runnable on CPU (mirrors test_native_flatten)."""
    from library.anima.models import Anima

    return Anima(
        max_img_h=16,
        max_img_w=16,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=64,
        num_blocks=num_blocks,
        num_heads=4,
        mlp_ratio=2.0,
        crossattn_emb_channels=64,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        use_llm_adapter=False,
        attn_mode="torch",
    ).eval()


def _make_inputs():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, 4, 4)
    timesteps = torch.tensor([0.5])
    crossattn_emb = torch.randn(1, 8, 64)
    return x, timesteps, crossattn_emb


def _run_steps(model, x, timesteps, emb, dtype, n_steps):
    """Run n_steps forwards; return (all_finite, ms_per_step)."""
    with torch.autocast("cpu", dtype=dtype), torch.no_grad():
        # warmup (3 calls) so timing excludes one-off dispatch overhead
        for _ in range(3):
            model.forward_mini_train_dit(x, timesteps, emb)
        t0 = time.perf_counter()
        last_out = None
        for _ in range(n_steps):
            last_out = model.forward_mini_train_dit(x, timesteps, emb)
        elapsed = time.perf_counter() - t0
    finite = bool(torch.isfinite(last_out).all().item())
    return finite, (elapsed / n_steps) * 1000.0, last_out.dtype


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=20, help="Forward steps per config.")
    p.add_argument("--num_blocks", type=int, default=8, help="DiT block count.")
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Free-form label appended to the run directory name.",
    )
    args = p.parse_args()

    x, timesteps, emb = _make_inputs()

    # --- bf16 baseline ---
    m_bf16 = _build_tiny_anima(args.num_blocks)
    bf16_finite, bf16_ms, bf16_dtype = _run_steps(
        m_bf16, x, timesteps, emb, torch.bfloat16, args.steps
    )

    # --- fp16 flag OFF (the broken path on a trained model) ---
    m_fp16_off = _build_tiny_anima(args.num_blocks)
    for b in m_fp16_off.blocks:
        b.fp32_residual = False
    m_fp16_off.final_layer.fp32_residual = False
    fp16_off_finite, fp16_off_ms, fp16_off_dtype = _run_steps(
        m_fp16_off, x, timesteps, emb, torch.float16, args.steps
    )

    # --- fp16 flag ON (the fix) ---
    m_fp16_on = _build_tiny_anima(args.num_blocks)
    m_fp16_on.enable_fp32_residual()
    fp16_on_finite, fp16_on_ms, fp16_on_dtype = _run_steps(
        m_fp16_on, x, timesteps, emb, torch.float16, args.steps
    )

    metrics = {
        "fp16_max": float(_FP16_MAX),
        "num_blocks": args.num_blocks,
        "steps": args.steps,
        "bf16": {
            "finite": bf16_finite,
            "ms_per_step": round(bf16_ms, 3),
            "out_dtype": str(bf16_dtype),
        },
        "fp16_flag_off": {
            "finite": fp16_off_finite,
            "ms_per_step": round(fp16_off_ms, 3),
            "out_dtype": str(fp16_off_dtype),
            "note": (
                "the broken path on a TRAINED model at large resolution; this "
                "tiny untrained fixture stays finite because default-zero "
                "AdaLN gates don't accumulate the residual past 65504. The "
                "dispositive overflow regression is the unit test."
            ),
        },
        "fp16_flag_on": {
            "finite": fp16_on_finite,
            "ms_per_step": round(fp16_on_ms, 3),
            "out_dtype": str(fp16_on_dtype),
        },
        "speedup_fp16_on_vs_bf16": round(bf16_ms / fp16_on_ms, 3),
        "flag_overhead_fp16_on_vs_off_pct": round(
            (fp16_on_ms - fp16_off_ms) / fp16_off_ms * 100.0, 1
        ),
    }

    run_dir = make_run_dir("fp16_residual", label=args.label)
    write_result(run_dir, script=__file__, args=args, metrics=metrics)

    print(f"=== fp16_residual bench → {run_dir} ===")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
