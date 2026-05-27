"""Per-image inversion of the orthogonal postfix tail (probe entry point).

See ``docs/proposal/postfix_residual_per_image_inversion.md``. This is a
measurement instrument, not a deployable adapter: it optimizes the K-dim
scale vector ``s`` per image so that

    ψ = concat(T5(tags), Q @ diag(s))

minimizes flow-matching loss against the frozen DiT. The output is the
K-vector ``s`` plus an optimization log. Downstream analysis (ceiling, PCA
spectrum, content clustering, lane-discipline, multi-seed functional
equivalence) consumes ``s`` directly.

Two modes:

* ``--image_dir`` (primary, batched): reads cached latents + cached T5
  prefixes from ``post_image_dataset/lora`` (or any dir with
  ``{stem}_*_anima.npz`` + ``{stem}_anima_te.safetensors`` pairs) and inverts
  ``--num_images`` of them.
* ``--image_stem``: single named image inside ``--image_dir``. Useful for
  one-off / debug.

Outputs per image, under ``--output_dir``:

* ``s/{stem}_s.safetensors`` — the (K,) trained vector, fp32.
* ``loss/{stem}.csv`` — per-step optimization log.

The SVD-of-cached-TE basis is computed once for ``(K, kind, seed)`` and
cached at ``--basis_path`` so repeat runs (different seeds / images) skip the
expensive SVD over the corpus.

Usage::

    uv run python scripts/inversion/invert_postfix_tail.py \\
        --dit models/diffusion_models/anima-base-v1.0.safetensors \\
        --image_dir post_image_dataset/lora \\
        --num_images 30 --shuffle --seed 0 \\
        --K 48 --basis svd_te \\
        --basis_path output/probes/postfix_tail/svd_basis_K48.pt \\
        --steps 100 --lr 0.01 --grad_accum 4 \\
        --lambda_zero 0.0 --sigma_min 0.0 \\
        --output_dir output/probes/postfix_tail/v0_first_run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path


import torch  # noqa: E402

from library.anima import weights as anima_utils  # noqa: E402
from library.inference.editing.postfix_inversion import (  # noqa: E402
    TailInversionConfig,
    invert_tail,
    load_cached_prefix,
    load_or_build_basis,
    save_soft_tokens_bank,
    save_tail_s,
)
from library.io.cache import (  # noqa: E402
    discover_cached_images,
    discover_cached_pairs,
    load_cached_latents,
)
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-image inversion of the orthogonal postfix tail "
        "(probe — see docs/proposal/postfix_residual_per_image_inversion.md)"
    )

    # Model
    p.add_argument("--dit", type=str, required=True, help="DiT checkpoint path")
    p.add_argument("--attn_mode", type=str, default="flash", help="Attention backend")
    p.add_argument(
        "--blocks_to_swap",
        type=int,
        default=16,
        help="Number of transformer blocks to swap to CPU (0 = none, "
        "<0 = gradient checkpointing instead)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available)",
    )
    p.add_argument(
        "--compile_blocks",
        action="store_true",
        default=True,
        help="torch.compile each transformer block's _forward (default on). "
        "Incompatible with block swap — silently skipped when "
        "--blocks_to_swap > 0.",
    )
    p.add_argument(
        "--no_compile_blocks",
        dest="compile_blocks",
        action="store_false",
        help="Disable torch.compile (run eager). Use for debugging or when "
        "compile time outweighs runtime gain.",
    )
    p.add_argument(
        "--compile_inductor_mode",
        type=str,
        default=None,
        help="Inductor preset passed through to torch.compile(mode=...). "
        "e.g. 'reduce-overhead' for per-block CUDAGraphs. None = inductor default.",
    )

    # Data
    p.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory with cached latents + cached T5 outputs "
        "(typically post_image_dataset/lora)",
    )
    p.add_argument(
        "--image_stem",
        type=str,
        default=None,
        help="Process only one image by stem (overrides --num_images/--shuffle)",
    )
    p.add_argument(
        "--num_images",
        type=int,
        default=30,
        help="Number of images to invert from --image_dir (default: 30, "
        "the proposal's N≈30–50 lower bound)",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle image order before slicing the first --num_images",
    )
    p.add_argument(
        "--shuffle_seed",
        type=int,
        default=0,
        help="Seed for image-order shuffling (separate from --seed which "
        "drives per-image optimization)",
    )
    p.add_argument(
        "--basis_te_dir",
        type=str,
        default=None,
        help="Directory of cached *_anima_te.safetensors for SVD basis "
        "construction. Defaults to --image_dir.",
    )

    # Basis
    p.add_argument(
        "--K",
        type=int,
        default=48,
        help="Tail length (number of orthonormal slots). Default 48 matches "
        "the companion encoder proposal.",
    )
    p.add_argument(
        "--basis",
        type=str,
        default="svd_te",
        choices=["svd_te", "random"],
        help="Basis kind. 'svd_te' = top-K right singular vectors of cached "
        "T5 corpus; 'random' = QR of a Gaussian.",
    )
    p.add_argument(
        "--basis_path",
        type=str,
        default=None,
        help="Path to cache/load the basis. If file exists and shape matches, "
        "reuses it; otherwise computes and saves.",
    )
    p.add_argument(
        "--basis_seed",
        type=int,
        default=0,
        help="Seed for basis construction (row-shuffle for svd_te, RNG for random)",
    )
    p.add_argument(
        "--svd_num_files",
        type=int,
        default=256,
        help="Number of cached TE files sampled for SVD basis computation",
    )
    p.add_argument(
        "--embed_dim",
        type=int,
        default=1024,
        help="T5-compatible embedding dim (Qwen3 hidden size = 1024)",
    )

    # Parameterization
    p.add_argument(
        "--parameterization",
        type=str,
        default="ortho_tail",
        choices=["ortho_tail", "soft_tokens"],
        help="How to fill the K postfix slots. 'ortho_tail' (default) = K scalars "
        "over a frozen orthonormal Q, one tail shared across all blocks and all t. "
        "'soft_tokens' = SoftREPA bank (per-block × per-t free tokens, no ortho, no "
        "caption-conditioning) to measure the per-image floor with that extra freedom. "
        "soft_tokens ignores --basis*/--K-vs-embed_dim and saves a {tokens, "
        "t_offsets.weight} bank instead of an s-vector.",
    )
    p.add_argument(
        "--st_n_layers",
        type=int,
        default=0,
        help="soft_tokens: number of leading DiT blocks to patch with the splice "
        "hook. 0 (default) = all blocks (max per-block freedom for a ceiling probe).",
    )
    p.add_argument(
        "--st_n_t_buckets",
        type=int,
        default=14,
        help="soft_tokens: number of σ buckets (per-t freedom). Each bucket gets its "
        "own per-layer offset vector. 1 collapses to a single t-independent tail.",
    )
    p.add_argument(
        "--st_init_std",
        type=float,
        default=0.02,
        help="soft_tokens: Gaussian std for the base-token init (t_offsets always "
        "zero-init). Nonzero by default — random per-token init breaks the within-"
        "layer slot symmetry that collapsed the old postfix to K=1. 0.0 = zero-init.",
    )
    p.add_argument(
        "--st_splice_position",
        type=str,
        default="end_of_sequence",
        choices=["end_of_sequence", "front_of_padding"],
        help="soft_tokens: where the K tokens land. Default end_of_sequence matches "
        "the ortho_tail probe (apples-to-apples — only the parameterization differs).",
    )

    # Optimization
    p.add_argument("--steps", type=int, default=50, help="Optimization steps per image")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate (AdamW)")
    p.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
    )
    p.add_argument(
        "--grad_accum", type=int, default=2, help="Gradient accumulation steps"
    )
    p.add_argument(
        "--timesteps_per_step",
        type=int,
        default=1,
        help="Extra σ samples per optimizer update — multiplies into grad_accum "
        "(total micro-iterations = grad_accum × timesteps_per_step). Each "
        "micro-iteration is a separate forward at batch=1, so raising this "
        "trades wall-time for variance reduction without growing VRAM.",
    )
    p.add_argument(
        "--sigma_sampling",
        type=str,
        default="sigmoid",
        choices=["uniform", "sigmoid"],
    )
    p.add_argument("--sigmoid_scale", type=float, default=1.0)
    p.add_argument(
        "--sigma_min",
        type=float,
        default=0.0,
        help="Lower bound for sampled sigmas (P-GRAFT-style low-σ skip — proposal "
        "calls out sigma_min ∈ {0, 0.1, 0.2} as a relevant sweep here)",
    )
    p.add_argument(
        "--sigma_max",
        type=float,
        default=0.25,
        help="Upper bound for sampled sigmas. Set < 1.0 to restrict supervision "
        "to low-σ steps (e.g. 0.25), where the FM target carries more per-image "
        "identity. Must be > --sigma_min.",
    )
    p.add_argument(
        "--lambda_zero",
        type=float,
        default=0.0,
        help="‖s‖² regularization weight. Proposal: primary run uses 0.0; "
        "auxiliary sweep at {0.001, 0.01, 0.1} measures lane-discipline cost.",
    )
    p.add_argument(
        "--init_std",
        type=float,
        default=0.0,
        help="Gaussian std for s init. 0.0 = zero-init (baseline). The "
        "archive script's 0.149 default is here as a documented ablation.",
    )
    p.add_argument("--seed", type=int, default=0, help="Per-image RNG seed")
    p.add_argument("--log_every", type=int, default=5)

    # Variance-reduced FM (AsymFlow §5.2 control variate, adapted per-image)
    p.add_argument(
        "--vr",
        dest="vr_enabled",
        action="store_true",
        help="Use VR-FM loss: per microstep, blend in a no-grad reference forward "
        "at s=0 on FEI-low-passed latents and supervise (y + λ·z)² with λ "
        "estimated online via EMA. (σ, noise, z) tuples are pre-sampled to a "
        "pool of --vr_pool_size to amortize the extra reference forwards.",
    )
    p.add_argument(
        "--vr_pool_size",
        type=int,
        default=32,
        help="VR pool size — # of (σ, noise, z) tuples precomputed per image",
    )
    p.add_argument(
        "--vr_lambda_beta",
        type=float,
        default=0.2,
        help="EMA β for λ. Training default is 0.01; per-image inversion bumps "
        "this to ~0.2 because the horizon is only ~50 microsteps per image.",
    )
    p.add_argument(
        "--vr_fei_sigma_low_div",
        type=float,
        default=4.0,
        help="FEI low-pass divisor (σ_low = min(H_lat, W_lat) / div). Matches "
        "FeRA's bench-validated 8.0 default — aspect-invariant on Anima.",
    )

    # Output
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Probe run directory. Outputs: s/{stem}_s.safetensors, "
        "loss/{stem}.csv, manifest.json",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Reinvert images even if their s file already exists "
        "(default: skip cached)",
    )

    args = p.parse_args()
    if args.K < 1:
        p.error("--K must be >= 1")
    if args.parameterization == "ortho_tail":
        # K orthonormal rows must fit in D-dim space.
        if args.K > args.embed_dim:
            p.error(f"--K ({args.K}) must be <= --embed_dim ({args.embed_dim})")
    else:
        # soft_tokens: K is a sequence-slot count; bound by the tail capacity.
        from library.inference.editing.postfix_inversion import MAX_SEQ_LEN

        if args.K >= MAX_SEQ_LEN:
            p.error(f"--K ({args.K}) must be < MAX_SEQ_LEN ({MAX_SEQ_LEN})")
        if args.st_n_t_buckets < 1:
            p.error("--st_n_t_buckets must be >= 1")
    return args


def _resolve_device(args) -> torch.device:
    if args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pick_images(args) -> list:
    """Discover cached image pairs in --image_dir; optionally filter / shuffle / slice."""
    images = discover_cached_images(args.image_dir)
    if not images:
        # Fall back to cache-only discovery (the post_image_dataset/lora layout
        # where the cache dir is decoupled from the source images).
        images = discover_cached_pairs(args.image_dir)
    if not images:
        raise FileNotFoundError(
            f"No cached latent+TE pairs found under {args.image_dir!r}. "
            "Run `make preprocess` first."
        )

    images = [im for im in images if im.te_path is not None]
    if not images:
        raise FileNotFoundError(
            f"No images under {args.image_dir!r} have cached "
            "_anima_te.safetensors prefixes (run `make preprocess-te`)"
        )

    if args.image_stem is not None:
        matches = [im for im in images if im.stem == args.image_stem]
        if not matches:
            raise ValueError(
                f"--image_stem {args.image_stem!r} not found in {args.image_dir}"
            )
        return matches

    if args.shuffle:
        rng = random.Random(args.shuffle_seed)
        images = list(images)
        rng.shuffle(images)

    if args.num_images is not None and args.num_images > 0:
        images = images[: args.num_images]
    return images


def _load_anima(args, device: torch.device, apply_before_compile=None):
    """Load DiT frozen on device, with the same swap/grad-ckpt switches as
    archive/inversion/invert_reference.py.

    ``apply_before_compile`` (if given) is invoked with the placed-but-uncompiled
    DiT right before the compile step. soft_tokens mode uses it to monkey-patch
    the Block.forward splice hooks first — torch.compile traces those hooks, so
    they MUST be installed before compile_blocks (the build_anima invariant)."""
    is_swapping = args.blocks_to_swap > 0
    grad_ckpt = args.blocks_to_swap < 0
    logger.info(f"Loading DiT: {args.dit}")
    anima = anima_utils.load_anima_model(
        device="cpu" if is_swapping else device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        loading_device="cpu" if is_swapping else device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.to(torch.bfloat16)
    anima.requires_grad_(False)

    if is_swapping:
        logger.info(f"Enabling block swap: {args.blocks_to_swap} blocks to CPU")
        anima.enable_block_swap(args.blocks_to_swap, device)
        anima.move_to_device_except_swap_blocks(device)
        anima.prepare_block_swap_before_forward()
        # block_swap moves weights CPU↔GPU mid-forward; incompatible with any
        # torch.compile mode — leave eager. Apply the splice hooks now anyway.
        if apply_before_compile is not None:
            apply_before_compile(anima)
    else:
        anima.to(device)
        if grad_ckpt:
            logger.info("Enabling gradient checkpointing")
            anima.enable_gradient_checkpointing()
            for block in anima.blocks:  # type: ignore[union-attr]
                block.train()
        if apply_before_compile is not None:
            apply_before_compile(anima)
        if args.compile_blocks:
            # compile_blocks turns on native-shape flattening (each aspect bucket
            # at its real token count, no padding → no flash pad-leak) and compiles
            # block._forward (not .forward) so the unsloth_checkpoint
            # @torch._disable_dynamo decorator doesn't blow the trace under
            # grad_ckpt. Dynamo recompiles once per distinct token count (e.g.
            # 720×1440 vs 1024×1024) — a one-time warmup, not per-step. These span
            # more than the 2 CONSTANT_TOKEN_BUCKETS families, so pre-raise the
            # dynamo cache (compile_blocks' max() won't lower it).
            import torch._dynamo as _dynamo

            _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
            anima.compile_blocks(backend="inductor", mode=args.compile_inductor_mode)
        else:
            logger.info("torch.compile disabled (--no_compile_blocks)")
    return anima


def _build_soft_tokens_net(args, anima, device):
    """Build a SoftTokensNetwork sized for this DiT (fp32 params on device).

    Caller is responsible for ``apply_to`` (done before compile via
    _load_anima's apply_before_compile hook). Params are reset per image inside
    invert_tail, so one network is reused across the whole run."""
    from networks.methods.soft_tokens import SoftTokensNetwork

    n_blocks = len(anima.blocks)  # type: ignore[arg-type]
    n_layers = args.st_n_layers if args.st_n_layers > 0 else n_blocks
    if n_layers > n_blocks:
        raise ValueError(f"--st_n_layers ({n_layers}) > DiT block count ({n_blocks})")
    net = SoftTokensNetwork(
        num_tokens=args.K,
        embed_dim=args.embed_dim,
        n_layers=n_layers,
        n_t_buckets=args.st_n_t_buckets,
        init_std=args.st_init_std,
        splice_position=args.st_splice_position,
        contrastive_weight=0.0,  # contrastive is training-only; off for the probe
    )
    net.to(device)  # keep fp32 params (master weights); hook casts to bf16 per forward
    logger.info(
        f"soft_tokens probe net: {n_layers} layers × K={args.K} × D={args.embed_dim}, "
        f"{args.st_n_t_buckets} t-buckets, splice={args.st_splice_position}, "
        f"init_std={args.st_init_std}"
    )
    return net


def main() -> None:
    args = parse_args()
    device = _resolve_device(args)
    logger.info(f"Device: {device}")

    images = _pick_images(args)
    logger.info(f"Inverting {len(images)} images from {args.image_dir}")

    soft_tokens = args.parameterization == "soft_tokens"

    out_root = Path(args.output_dir)
    # ortho_tail saves an s-vector under s/; soft_tokens saves a bank under bank/.
    bank_dir_name = "bank" if soft_tokens else "s"
    s_dir = out_root / bank_dir_name
    loss_dir = out_root / "loss"
    s_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    # Basis is ortho_tail-only; soft_tokens needs no SVD corpus.
    Q = None
    if not soft_tokens:
        # Pre-build / cache the basis BEFORE the DiT load so its VRAM cost is
        # peaked first (SVD over a 256-file corpus is CPU-bound but allocates).
        basis_te_dir = args.basis_te_dir or args.image_dir
        Q = load_or_build_basis(
            K=args.K,
            D=args.embed_dim,
            kind=args.basis,
            te_cache_dir=basis_te_dir,
            basis_path=args.basis_path,
            svd_num_files=args.svd_num_files,
            seed=args.basis_seed,
        )

    # soft_tokens: build + apply the splice net before compile (build_anima
    # compile-after-apply invariant). One net, reused across images.
    st_holder: dict = {}

    def _apply_st(_anima):
        net = _build_soft_tokens_net(args, _anima, device)
        net.apply_to(text_encoders=None, unet=_anima, apply_unet=True)
        st_holder["net"] = net

    anima = _load_anima(args, device, apply_before_compile=_apply_st if soft_tokens else None)
    soft_tokens_net = st_holder.get("net")

    cfg = TailInversionConfig(
        K=args.K,
        steps=args.steps,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        grad_accum=args.grad_accum,
        timesteps_per_step=args.timesteps_per_step,
        sigma_sampling=args.sigma_sampling,
        sigmoid_scale=args.sigmoid_scale,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        lambda_zero=args.lambda_zero,
        init_std=args.init_std,
        log_every=args.log_every,
        parameterization=args.parameterization,
        vr_enabled=args.vr_enabled,
        vr_pool_size=args.vr_pool_size,
        vr_lambda_beta=args.vr_lambda_beta,
        vr_fei_sigma_low_div=args.vr_fei_sigma_low_div,
    )

    manifest = {
        "K": args.K,
        "embed_dim": args.embed_dim,
        "parameterization": args.parameterization,
        "st_n_layers": (
            (args.st_n_layers if args.st_n_layers > 0 else len(anima.blocks))
            if soft_tokens
            else None
        ),
        "st_n_t_buckets": args.st_n_t_buckets if soft_tokens else None,
        "st_init_std": args.st_init_std if soft_tokens else None,
        "st_splice_position": args.st_splice_position if soft_tokens else None,
        "basis": args.basis,
        "basis_seed": args.basis_seed,
        "basis_path": args.basis_path,
        "svd_num_files": args.svd_num_files,
        "image_dir": args.image_dir,
        "num_images_requested": args.num_images,
        "shuffle": args.shuffle,
        "shuffle_seed": args.shuffle_seed,
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "grad_accum": args.grad_accum,
        "timesteps_per_step": args.timesteps_per_step,
        "sigma_sampling": args.sigma_sampling,
        "sigmoid_scale": args.sigmoid_scale,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "lambda_zero": args.lambda_zero,
        "init_std": args.init_std,
        "vr_enabled": args.vr_enabled,
        "vr_pool_size": args.vr_pool_size,
        "vr_lambda_beta": args.vr_lambda_beta,
        "vr_fei_sigma_low_div": args.vr_fei_sigma_low_div,
        "dit": args.dit,
        "attn_mode": args.attn_mode,
        "compile_blocks": args.compile_blocks,
        "compile_inductor_mode": args.compile_inductor_mode,
        "results": [],
    }

    for i, img in enumerate(images):
        stem = img.stem
        s_path = s_dir / f"{stem}_{'bank' if soft_tokens else 's'}.safetensors"
        loss_path = loss_dir / f"{stem}.csv"

        if s_path.exists() and not args.overwrite:
            logger.info(f"[{i + 1}/{len(images)}] {stem} — already inverted, skipping")
            continue

        logger.info(f"[{i + 1}/{len(images)}] {stem}")

        lat, _, orig_h, orig_w = load_cached_latents(img.npz_path)
        latents = lat.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        prefix_emb = load_cached_prefix(img.te_path, device=device)

        result = invert_tail(
            anima,
            prefix_emb=prefix_emb,
            latents=latents,
            basis_Q=Q,
            config=cfg,
            device=device,
            seed=args.seed,
            log_path=str(loss_path),
            soft_tokens_net=soft_tokens_net,
        )

        common_meta = {
            "ss_source_stem": stem,
            "ss_image_hw": f"{orig_h}x{orig_w}",
            "ss_best_loss": f"{result.best_loss:.6f}",
            "ss_best_fm_loss": f"{result.best_fm_loss:.6f}",
            "ss_best_step": str(result.best_step),
            "ss_final_param_l2": f"{result.final_s_l2:.6f}",
            "ss_steps": str(args.steps),
            "ss_lr": str(args.lr),
            "ss_lambda_zero": str(args.lambda_zero),
            "ss_sigma_min": str(args.sigma_min),
            "ss_sigma_max": str(args.sigma_max),
            "ss_seed": str(args.seed),
        }
        if soft_tokens:
            save_soft_tokens_bank(
                str(s_path),
                result.bank,
                n_layers=soft_tokens_net.n_layers,
                num_tokens=soft_tokens_net.num_tokens,
                embed_dim=soft_tokens_net.embed_dim,
                n_t_buckets=soft_tokens_net.n_t_buckets,
                splice_position=soft_tokens_net.splice_position,
                metadata={**common_meta, "ss_init_std": str(args.st_init_std)},
            )
        else:
            save_tail_s(
                str(s_path),
                result.s,
                K=args.K,
                D=args.embed_dim,
                basis_kind=args.basis,
                metadata={
                    **common_meta,
                    "ss_init_std": str(args.init_std),
                    "ss_basis_kind": args.basis,
                },
            )

        manifest["results"].append(
            {
                "stem": stem,
                "image_hw": [orig_h, orig_w],
                "best_loss": result.best_loss,
                "best_fm_loss": result.best_fm_loss,
                "best_step": result.best_step,
                "final_s_l2": result.final_s_l2,
                "final_lambda_ema": result.final_lambda_ema,
                "s_path": str(s_path.relative_to(out_root)),
                "loss_path": str(loss_path.relative_to(out_root)),
            }
        )
        logger.info(
            f"  saved {s_path.name} (best_loss={result.best_loss:.6f} "
            f"@ step {result.best_step}, fm={result.best_fm_loss:.6f}, "
            f"s‖₂={result.final_s_l2:.3f})"
        )

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest: {manifest_path}")
    logger.info("Done")


if __name__ == "__main__":
    main()
