"""Compiled-graph memory + recompile cost: pad (1 shape) vs no-pad (5 shapes).

The pad-to-static design buys a single torch.compile block graph; the native
(no-pad) path trades that for one graph per distinct CONSTANT_TOKEN_BUCKETS
token-count (5). This bench answers the open question from
`project_static_flash_padding_leaks`: does holding 5 compiled shapes inflate
peak GPU memory enough to matter on a 16 GB card?

For each mode it block-compiles the base DiT, runs a forward at every distinct
bucket shape (no-pad) or the single padded shape (pad), and reports the peak
allocated bytes plus the dynamo recompile count. Forward-only: backward roughly
doubles the activation term uniformly across both modes, so the *delta* between
modes — the thing in question — is what this isolates.

Run (repo root):
    python -m bench.static_padding.bench_compile_mem
    python -m bench.static_padding.bench_compile_mem --inductor_mode reduce-overhead
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima.weights import load_anima_model  # noqa: E402
from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS  # noqa: E402

log = logging.getLogger("bench.static_padding")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
STATIC_TOKENS = 4096
TEXT_LEN = 226


def _distinct_bucket_latents():
    """One (H_lat, W_lat) per distinct token-count in CONSTANT_TOKEN_BUCKETS."""
    seen: dict[int, tuple[int, int]] = {}
    for h, w in CONSTANT_TOKEN_BUCKETS:
        h_lat, w_lat = h // 8, w // 8  # VAE downsample 8x
        tok = (h_lat // 2) * (w_lat // 2)  # patch 2x
        seen.setdefault(tok, (h_lat, w_lat))
    return seen  # {token_count: (H_lat, W_lat)}


def _all_bucket_latents():
    """Every distinct (H_lat, W_lat) resolution — to prove the no-pad block
    graph keys on token-count (collapses to 5), not resolution (17)."""
    out = []
    for h, w in CONSTANT_TOKEN_BUCKETS:
        h_lat, w_lat = h // 8, w // 8
        tok = (h_lat // 2) * (w_lat // 2)
        out.append((tok, (h_lat, w_lat)))
    return out


def _inputs(h_lat, w_lat, sigma, device, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "x_B_C_T_H_W": torch.randn(
            1, 16, 1, h_lat, w_lat, generator=g, dtype=torch.float32
        ).to(device, dtype),
        "timesteps_B_T": torch.tensor([sigma], device=device, dtype=dtype),
        "crossattn_emb": torch.randn(
            1, TEXT_LEN, 1024, generator=g, dtype=torch.float32
        ).to(device, dtype),
        "padding_mask": torch.zeros(1, 1, h_lat, w_lat, device=device, dtype=dtype),
    }


@torch.no_grad()
def _fwd(model, inp, dtype):
    with torch.autocast("cuda", dtype=dtype):
        model.forward_mini_train_dit(
            inp["x_B_C_T_H_W"], inp["timesteps_B_T"], inp["crossattn_emb"],
            padding_mask=inp["padding_mask"], skip_pooled_text_proj=True,
        )


def _run_mode(args, pad: bool, device, dtype):
    import torch._dynamo as _dynamo

    _dynamo.reset()
    _dynamo.utils.counters.clear()
    buckets = _distinct_bucket_latents()
    n_shapes = len(buckets)
    if not pad:
        _dynamo.config.cache_size_limit = max(
            _dynamo.config.cache_size_limit, 2 * n_shapes + 8
        )

    model = load_anima_model(
        device, args.dit_path, attn_mode="flash",
        loading_device=device, dit_weight_dtype=dtype,
    )
    model.to(device)
    model.reset_mod_guidance()
    model.eval()
    model.set_static_token_count(STATIC_TOKENS, pad=pad)
    model.compile_blocks(
        "inductor", mode=(args.inductor_mode if args.inductor_mode != "none" else None)
    )

    torch.cuda.reset_peak_memory_stats(device)
    # No-pad: run ALL 17 resolutions so unique_graphs reveals the collapse to
    # 5 token-counts. Pad: one shape regardless.
    shapes = (
        [(STATIC_TOKENS, next(iter(buckets.values())))] if pad else _all_bucket_latents()
    )
    # Two passes: first triggers compiles, second is steady state.
    for _ in range(2):
        for _tok, (h_lat, w_lat) in shapes:
            _fwd(model, _inputs(h_lat, w_lat, args.sigma, device, dtype), dtype)
            torch.cuda.synchronize(device)

    peak = torch.cuda.max_memory_allocated(device)
    recompiles = _dynamo.utils.counters["stats"].get("unique_graphs", 0)
    del model
    torch.cuda.empty_cache()
    return {
        "mode": "pad" if pad else "no_pad",
        "n_shapes_run": len(shapes),
        "peak_alloc_bytes": peak,
        "peak_alloc_gib": round(peak / 2**30, 3),
        "unique_graphs": recompiles,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dit_path", default=DEFAULT_DIT)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument(
        "--inductor_mode", default="none",
        help="inductor preset ('none', 'default', 'reduce-overhead'). "
        "reduce-overhead pins one CUDAGraph pool per shape — the case where "
        "no-pad's 5 shapes can multiply memory.",
    )
    ap.add_argument("--label", default="compile-mem")
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    log.info("buckets collapse to %d distinct token-counts: %s",
             len(_distinct_bucket_latents()), sorted(_distinct_bucket_latents()))

    results = []
    for pad in (True, False):
        log.info("\n--- compiling mode=%s (inductor_mode=%s) ---",
                 "pad" if pad else "no_pad", args.inductor_mode)
        r = _run_mode(args, pad, device, dtype)
        results.append(r)
        log.info("%s: peak=%.3f GiB, unique_graphs=%d, shapes_run=%d",
                 r["mode"], r["peak_alloc_gib"], r["unique_graphs"], r["n_shapes_run"])

    pad_r, nopad_r = results
    delta_gib = round(nopad_r["peak_alloc_gib"] - pad_r["peak_alloc_gib"], 3)
    log.info("\n=== no-pad peak overhead vs pad: %+.3f GiB ===", delta_gib)

    run_dir = make_run_dir("static_padding", label=args.label)
    write_result(
        run_dir, script=__file__, args=args, label=args.label,
        metrics={
            "inductor_mode": args.inductor_mode,
            "pad": pad_r, "no_pad": nopad_r,
            "nopad_peak_overhead_gib": delta_gib,
        },
        artifacts=[], device=device,
    )
    log.info("wrote %s", run_dir)


if __name__ == "__main__":
    main()
