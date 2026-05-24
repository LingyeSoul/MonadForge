"""SPD fine-tuning LoRA — trajectory adapter for progressive-resolution inference.

Trains a *plain* LoRA on one frozen Anima DiT to follow the stage-specific
straight-line velocity targets of the Spectral Progressive Diffusion (SPD)
multi-resolution trajectory (Xiao et al., arXiv:2605.18736, §4.3, Eq. 11–14).
This is "Case B" of the SPD investigation — see
``docs/proposal/spd_finetune_lora.md``. Output ``output/ckpt/anima_spd.safetensors``
is a normal LoRA: load it through the standard inference path and run it with
the SPD sampler (``--spd``) at the *same* schedule it was trained on.

Models the structure on ``scripts/distill_mod/distill.py`` /
``scripts/distill_turbo.py`` (frozen-DiT + adapter-only + single MSE backward),
but strictly simpler: one adapter, one optimizer.

Usage::

    make exp-spd                                  # defaults from spd.toml
    make exp-spd ARGS="--iterations 2000 --single_prompt_idx 0"   # Phase 0
    make exp-spd PRESET=low_vram                  # block swap + grad ckpt
    make exp-spd ARGS="--torch_compile"           # per-stage static-shape compile

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from tqdm import tqdm  # noqa: E402

from library.anima import weights as anima_utils  # noqa: E402
from library.anima.models import Anima  # noqa: E402
from library.datasets.distill import CachedDataset  # noqa: E402
from networks.lora_anima.factory import create_network  # noqa: E402
from networks.lora_save import save_network_weights  # noqa: E402
from networks.spd import (  # noqa: E402
    spd_schedule_bands,
    spd_stage_target,
)
from library.io.cache import get_latent_resolution  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _flatten(cfg: dict, key_path: str, default):
    """Look up ``a.b.c`` in a nested TOML dict, falling back to ``default``."""
    node = cfg
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def main():
    parser = argparse.ArgumentParser(
        description="SPD fine-tuning LoRA — §4.3 trajectory adapter"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/methods/spd.toml",
        help="Path to the SPD TOML config (CLI flags override TOML values).",
    )
    # CLI overrides — sentinels (None / -1 / -1.0) mean "use the TOML value".
    parser.add_argument("--dit_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=-1.0)
    parser.add_argument(
        "--channel_scaling_alpha",
        type=float,
        default=-1.0,
        help="SmoothQuant-style per-channel input pre-scaling exponent for the LoRA "
        "down projection (0.0 = off / paper-faithful, 0.5 = sqrt balance, 1.0 = full "
        "flatten). Overrides network.channel_scaling_alpha. inv_scale is baked into the "
        "saved weights, so inference needs no extra plumbing.",
    )
    parser.add_argument("--attn_mode", type=str, default=None)
    parser.add_argument(
        "--stages",
        type=float,
        nargs="+",
        default=None,
        help="Ascending resolution scales (last must be 1.0). Overrides schedule.stages.",
    )
    parser.add_argument(
        "--transition_sigmas",
        type=float,
        nargs="+",
        default=None,
        help="σ thresholds to expand to the next stage (len = len(stages)-1). "
        "Overrides schedule.transition_sigmas.",
    )
    parser.add_argument(
        "--sigma_jitter",
        type=float,
        default=-1.0,
        help="±absolute uniform jitter on transition σ each step (R2 robustness). 0 = off.",
    )
    parser.add_argument("--lr", type=float, default=-1.0)
    parser.add_argument("--grad_clip", type=float, default=-1.0)
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=-1,
        help="Micro-steps accumulated per optimizer step (resampling the SPD "
        "stage each one, so updates mix low-/full-res). 1 = off. `iterations` "
        "counts optimizer steps, so wall-clock scales ~linearly with this.",
    )
    parser.add_argument("--warmup", type=float, default=-1.0)
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--grad_ckpt", action="store_true", default=False)
    parser.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    parser.add_argument(
        "--torch_compile",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="torch.compile each block's _forward (dynamic=False). Recompiles "
        "once per distinct (stage x bucket) shape on the flash backend, each at "
        "its real token count (no padding); the dynamo cache limit is raised to "
        "keep every specialization cached. On by default; pass --no-torch_compile "
        "to run eager.",
    )
    parser.add_argument("--dynamo_backend", type=str, default="inductor")
    parser.add_argument(
        "--compile_inductor_mode",
        type=str,
        default=None,
        help="torch.compile inductor preset (e.g. 'reduce-overhead'). "
        "Incompatible with --blocks_to_swap (CUDAGraphs need stable addresses).",
    )
    parser.add_argument("--save_every", type=int, default=-1)
    parser.add_argument("--log_interval", type=int, default=-1)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument(
        "--single_prompt_idx",
        type=int,
        default=None,
        help="Phase 0 overfit mode — pin the dataloader to a single (latent, text) pair.",
    )
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build the schedule + iterate the dataloader without loading the DiT.",
    )
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    def pick(cli_val, toml_key, default):
        if cli_val is not None and cli_val != -1 and cli_val != -1.0:
            return cli_val
        return _flatten(cfg, toml_key, default)

    dit_path = pick(
        args.dit_path, "dit_path", "models/diffusion_models/anima-base-v1.0.safetensors"
    )
    data_dir = pick(args.data_dir, "data_dir", "post_image_dataset/lora")
    output_dir = pick(args.output_dir, "output_dir", "output/ckpt")
    output_name = pick(args.output_name, "output_name", "anima_spd")
    iterations = int(pick(args.iterations, "iterations", 1000))
    batch_size = int(pick(args.batch_size, "batch_size", 1))
    seed = int(pick(args.seed, "seed", 42))

    rank = int(pick(args.rank, "network.rank", 48))
    alpha = float(
        _flatten(cfg, "network.alpha", rank) if args.alpha == -1.0 else args.alpha
    )
    attn_mode = pick(args.attn_mode, "network.attn_mode", "flash")
    channel_scaling_alpha = float(
        pick(args.channel_scaling_alpha, "network.channel_scaling_alpha", 0.0)
    )
    if (
        args.torch_compile
        and args.compile_inductor_mode == "reduce-overhead"
        and (args.blocks_to_swap > 0)
    ):
        logger.warning(
            "compile_inductor_mode='reduce-overhead' (CUDAGraphs) is incompatible "
            "with --blocks_to_swap (block addresses move each step); expect breakage."
        )

    stages = list(
        args.stages
        if args.stages is not None
        else _flatten(cfg, "schedule.stages", [0.5, 1.0])
    )
    transition_sigmas = list(
        args.transition_sigmas
        if args.transition_sigmas is not None
        else _flatten(cfg, "schedule.transition_sigmas", [0.5])
    )
    schedule_label = _flatten(cfg, "schedule.label", "custom")
    sigma_jitter = float(pick(args.sigma_jitter, "schedule.sigma_jitter", 0.0))

    # Schedule sanity — same invariants spd_denoise / spd_schedule_bands assume.
    if not stages or abs(stages[-1] - 1.0) > 1e-9:
        raise ValueError(f"schedule.stages must end at 1.0, got {stages}")
    if any(stages[i] >= stages[i + 1] for i in range(len(stages) - 1)):
        raise ValueError(f"schedule.stages must be strictly ascending, got {stages}")
    if len(transition_sigmas) != len(stages) - 1:
        raise ValueError(
            f"transition_sigmas (len {len(transition_sigmas)}) must be len(stages)-1 "
            f"({len(stages) - 1}); stages={stages}, transition_sigmas={transition_sigmas}"
        )

    lr = float(pick(args.lr, "optim.lr", 1e-4))
    weight_decay = float(_flatten(cfg, "optim.weight_decay", 0.0))
    grad_clip = float(pick(args.grad_clip, "optim.grad_clip", 1.0))
    grad_accum = max(1, int(pick(args.grad_accum, "optim.grad_accum", 1)))
    warmup = float(pick(args.warmup, "optim.warmup", 0.02))

    save_every = int(pick(args.save_every, "io.save_every", 500))
    log_interval = int(pick(args.log_interval, "io.log_interval", 10))
    log_dir = pick(args.log_dir, "io.log_dir", "output/logs/spd")

    torch.manual_seed(seed)

    # --- Schedule bands (data-independent; weights keep marginal-over-t uniform) ---
    bands = spd_schedule_bands(stages, transition_sigmas)
    band_widths = torch.tensor([hi - lo for (lo, hi) in bands], dtype=torch.float64)
    band_widths_f = band_widths.float()  # hoisted for the per-step multinomial
    stage_probs = (band_widths / band_widths.sum()).tolist()
    logger.info(
        "SPD schedule '%s': stages=%s transition_sigmas=%s",
        schedule_label,
        stages,
        transition_sigmas,
    )
    for i, ((lo, hi), p) in enumerate(zip(bands, stage_probs)):
        logger.info(
            "  stage %d  scale=%.3f  query σ∈(%.4f, %.4f)  p=%.3f",
            i,
            stages[i],
            lo,
            hi,
            p,
        )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # --- Dataset (bucket-grouped; one resolution per batch) ---
    dataset = CachedDataset(
        data_dir, batch_size=batch_size, sample_ratio=args.sample_ratio
    )
    if args.single_prompt_idx is not None:
        pinned = args.single_prompt_idx % len(dataset.samples)
        only = dataset.samples[pinned]
        dataset.samples = [only]
        logger.info(
            "single-prompt overfit mode: pinned idx=%d (latent=%s)",
            args.single_prompt_idx,
            os.path.basename(only[0]),
        )

    def _collate(batch):
        return (
            [b[0] for b in batch],
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]),  # pooled — unused
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # bucket-grouped: shuffling would mix resolutions
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate,
    )

    # Generator for stage construction (fresh HF noise per step; seed offset so
    # it's independent of the torch global stream used for stage selection).
    gen = torch.Generator(device=device).manual_seed(seed + 7919)

    if args.dry_run:
        for i, (_idx, lat, te, _pooled) in enumerate(tqdm(dataloader, desc="dry-run")):
            lat = lat.to(device, dtype=dtype)
            x0_full = lat.unsqueeze(2)
            for s in range(len(stages)):
                x0_si, eps_si = spd_stage_target(
                    x0_full, s, stages, transition_sigmas, patch=1, gen=gen
                )
                assert x0_si.shape == eps_si.shape
            if i >= 20:
                break
        logger.info("Dry run OK: stage-target construction + collation clean.")
        return

    # --- Load DiT (frozen) ---
    logger.info("Loading DiT model...")
    model: Anima = anima_utils.load_anima_model(
        device,
        dit_path,
        attn_mode=attn_mode,
        loading_device="cpu" if args.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )
    patch = model.patch_spatial

    # --- Plain LoRA adapter (paper-faithful: no MoE / ortho / T-LoRA / ReFT) ---
    # use_custom_down_autograd: save the bf16 lora_down input and recompute the
    # fp32 cast in backward instead of stashing the fp32 copy. Bitwise-identical
    # on the no-channel-scale path (the default); when channel_scaling_alpha>0 the
    # per-Linear inv_scale is folded into the recomputed down-project (lora.py:97),
    # so it stays correct, just no longer bit-identical to the alpha=0 baseline.
    # Trims LoRA-branch activation memory and avoids a per-Linear bf16 intermediate
    # getting pinned in the CUDA-Graph pool under reduce-overhead. See
    # custom_autograd.py and project-custom-down-autograd-distill-lever.
    if channel_scaling_alpha:
        logger.info(
            "channel_scaling enabled (alpha=%.3g); inv_scale baked at save",
            channel_scaling_alpha,
        )
    network = create_network(
        multiplier=1.0,
        network_dim=rank,
        network_alpha=alpha,
        vae=None,
        text_encoders=[],
        unet=model,
        use_custom_down_autograd=True,
        channel_scaling_alpha=channel_scaling_alpha,
    )
    network.apply_to(
        text_encoders=[], unet=model, apply_text_encoder=False, apply_unet=True
    )

    # Block swap / device placement.
    if args.blocks_to_swap > 0:
        model.enable_block_swap(args.blocks_to_swap, device)
        model.move_to_device_except_swap_blocks(device)
        model.switch_block_swap_for_training()
    else:
        model.to(device)

    if args.grad_ckpt:
        model.enable_gradient_checkpointing(unsloth_offload=True)
        logger.info("gradient checkpointing: on (unsloth CPU offload)")
    else:
        logger.info("gradient checkpointing: off")
    model.train()

    # Freeze base DiT; only the LoRA params train. apply_to add_module'd the
    # LoRA submodules onto the unet, so a wholesale freeze then re-enabling the
    # network's own params leaves exactly the adapter trainable.
    for p in model.parameters():
        p.requires_grad_(False)
    network.to(device=device, dtype=dtype)
    network.prepare_grad_etc(None, model)  # network.requires_grad_(True)

    trainable = [p for p in network.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    logger.info(
        "trainable: %s LoRA params over %d modules",
        f"{n_train:,}",
        len(network.unet_loras),
    )

    # --- Per-shape block compile ---
    # Compile each block's _forward (dynamic=False) and let torch.compile
    # recompile once per distinct (stage x aspect-bucket) shape on the flash
    # backend — each at its real token count, no padding/masking. Raise the
    # dynamo cache limit to cover every (stage x bucket) specialization plus its
    # backward graph so none falls back to eager. Recompiles are a one-time
    # warmup cost, not a correctness issue.
    if args.torch_compile:
        import torch._dynamo as _dynamo

        n_buckets = len({get_latent_resolution(npz) for npz, _te in dataset.samples})
        # SPD runs each stage at a downsampled resolution NOT in
        # CONSTANT_TOKEN_BUCKETS, so distinct shapes = stages × buckets — far more
        # than the 2 full-res families compile_blocks budgets for internally.
        # Pre-raise the limit here; compile_blocks' max() won't lower it. fwd+bwd
        # entries share the one `_forward` bytecode, so give headroom.
        n_shapes = len(stages) * max(1, n_buckets)
        _dynamo.config.cache_size_limit = max(
            _dynamo.config.cache_size_limit, 2 * n_shapes + 8
        )
        model.compile_blocks(args.dynamo_backend, mode=args.compile_inductor_mode)
        logger.info(
            "torch_compile: %d block._forward compiled (backend=%s, mode=%s); "
            "up to %d (stage x bucket) shapes recompile over the first steps "
            "(cache_size_limit=%d).",
            len(model.blocks),
            args.dynamo_backend,
            args.compile_inductor_mode,
            n_shapes,
            _dynamo.config.cache_size_limit,
        )

    # --- Optimizer + warmup→cosine ---
    optimizer = torch.optim.AdamW(
        trainable, lr=lr, weight_decay=weight_decay, fused=torch.cuda.is_available()
    )
    warmup_steps = int(warmup) if warmup >= 1 else int(warmup * iterations)
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-6 / lr, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=iterations - warmup_steps, eta_min=lr * 0.1
                ),
            ],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iterations, eta_min=lr * 0.1
        )

    # --- Logging ---
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_log:
        from datetime import datetime

        run_log = Path(log_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_log))
        writer.add_text(
            "config",
            "  \n".join(
                f"{k}: {v}"
                for k, v in {
                    "schedule_label": schedule_label,
                    "stages": stages,
                    "transition_sigmas": transition_sigmas,
                    "rank": rank,
                    "alpha": alpha,
                    "channel_scaling_alpha": channel_scaling_alpha,
                    "lr": lr,
                    "iterations": iterations,
                    "sigma_jitter": sigma_jitter,
                }.items()
            ),
        )
        logger.info("TensorBoard logs -> %s", run_log)

    def _save(step: int):
        save_path = str(Path(output_dir) / f"{output_name}.safetensors")
        sd = network.state_dict()
        # Keep .inv_scale buffers (per_channel_scaling): the standard write path's
        # bake_inv_scale folds them into lora_down, so dropping them here would
        # silently emit a wrong delta whenever channel_scaling_alpha>0.
        sd = {
            k: v
            for k, v in sd.items()
            if ".lora_" in k or k.endswith(".alpha") or k.endswith(".inv_scale")
        }
        save_network_weights(
            sd,
            file=save_path,
            dtype=torch.bfloat16,
            metadata={
                # R2 / open-question #2: snapshot the schedule so inference can't
                # silently mismatch the geometry the LoRA learned.
                "ss_spd_stages": json.dumps(stages),
                "ss_spd_transition_sigmas": json.dumps(transition_sigmas),
                "ss_spd_schedule_label": str(schedule_label),
                "ss_spd_rank": str(rank),
                "ss_channel_scaling_alpha": str(channel_scaling_alpha),
                "ss_spd_step": str(step),
            },
            save_variant="standard",
        )
        logger.info("saved SPD LoRA → %s  (step %d, %d keys)", save_path, step, len(sd))

    stage_rng = torch.Generator().manual_seed(seed + 1)  # CPU: stage / mode selection

    def _forward_dit(x5, sig_vec, cattn):
        """Single conditional forward at x5's own resolution (adapter on)."""
        pad = torch.zeros(
            x5.shape[0], 1, x5.shape[-2], x5.shape[-1], dtype=dtype, device=device
        )
        if model.blocks_to_swap:
            model.prepare_block_swap_before_forward()
        with torch.autocast("cuda", dtype=dtype):
            return model.forward_mini_train_dit(
                x5, sig_vec, cattn, padding_mask=pad, skip_pooled_text_proj=True
            )

    # --- Training loop ---
    logger.info("Starting SPD distillation: %d iterations", iterations)
    data_iter = [iter(dataloader)]  # boxed so _micro_step can refresh on exhaustion
    progress = tqdm(range(iterations), desc="spd")
    # GPU-side logging accumulators — flushed in one stacked .tolist() at every
    # log_interval, replacing the per-micro-step loss.item() (grad_accum CUDA
    # syncs per optimizer step) and the per-parameter .item() walk in the
    # LoRA-norm logging. Mirrors the accumulator pattern in distill_turbo.py.
    n_stages = len(stages)
    acc_loss = torch.zeros((), device=device)              # Σ step-mean loss
    acc_loss_stage = torch.zeros(n_stages, device=device)  # Σ micro-loss by stage
    acc_stage_cnt = torch.zeros(n_stages, device=device)   # micro-steps by stage
    def _micro_step():
        """One sample → scaled backward. Returns (unscaled_loss_tensor, stage_idx).

        The loss is returned as a *detached GPU tensor* (not ``.item()``) so the
        accumulation in the training loop stays sync-free; grad_accum micro-steps
        would otherwise force that many CUDA syncs per optimizer step.

        Stage is resampled here (not once per optimizer step), so when
        grad_accum > 1 each update averages gradients across the low-res and
        full-res regimes instead of swinging between them — the high CoV in the
        stage losses is regime-switching noise, which accumulation cancels.
        """
        try:
            _idx, latents, crossattn_emb, _pooled = next(data_iter[0])
        except StopIteration:
            data_iter[0] = iter(dataloader)
            _idx, latents, crossattn_emb, _pooled = next(data_iter[0])

        latents = latents.to(device, dtype=dtype, non_blocking=True)
        crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
        B = latents.shape[0]
        x0_full = latents.unsqueeze(2)  # (B, 16, 1, H, W)

        # Optional R2 jitter: perturb the transition σ so the segment geometry is
        # learned as a band, not a point.
        trans = transition_sigmas
        if sigma_jitter > 0.0 and len(transition_sigmas) > 0:
            trans = [
                float(
                    min(
                        0.999,
                        max(0.001, s + (torch.rand(1).item() * 2 - 1) * sigma_jitter),
                    )
                )
                for s in transition_sigmas
            ]

        # Sample one stage for this micro-batch (single-resolution per forward),
        # weighted by band width.
        stage_idx = int(
            torch.multinomial(band_widths_f, 1, generator=stage_rng).item()
        )
        # Bands depend only on the schedule, so reuse the precomputed ones;
        # only jitter (which builds a fresh `trans`) needs a recompute.
        t_lo, t_hi = (
            bands[stage_idx]
            if trans is transition_sigmas
            else spd_schedule_bands(stages, trans)[stage_idx]
        )
        x0_si, eps_si = spd_stage_target(
            x0_full, stage_idx, stages, trans, patch=patch, gen=gen
        )
        # FM training sample + analytic velocity target at scale s_i (Eq. 13–14).
        t = (t_lo + (t_hi - t_lo) * torch.rand(B, device=device)).to(dtype)
        t_e = t.view(B, 1, 1, 1, 1)
        x_t = (1.0 - t_e) * x0_si + t_e * eps_si
        if args.grad_ckpt:  # reentrant checkpoint needs a grad-requiring input
            x_t.requires_grad_()
        v_target = (eps_si - x0_si).float()
        # Native shapes: the forward runs at this stage's real token count (no
        # padding → no flash pad-leak). Flattening is enabled once by
        # compile_blocks above, which traces one graph per (stage × bucket) shape
        # keyed on the real seq_len — nothing per-step to set here.
        pred = _forward_dit(x_t, t, crossattn_emb)
        loss = nn.functional.mse_loss(pred.float(), v_target)
        # Scale so accumulated grads are the *mean* over micro-steps (matches a
        # true batch); LR/grad_clip semantics stay invariant to grad_accum.
        (loss / grad_accum).backward()
        return loss.detach(), stage_idx

    for step in progress:
        step_loss = torch.zeros((), device=device)  # mean micro-loss, GPU-side
        for _ in range(grad_accum):
            micro_loss, stage_idx = _micro_step()
            step_loss = step_loss + micro_loss / grad_accum
            acc_loss_stage[stage_idx] += micro_loss  # python idx → no sync
            acc_stage_cnt[stage_idx] += 1
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        acc_loss.add_(step_loss)

        if (step + 1) % log_interval == 0:
            # LoRA L2 norms: accumulate squared sums on-device and fold them into
            # the single sync below (was one .item() per trainable parameter).
            with torch.no_grad():
                up_sq = torch.zeros((), device=device)
                down_sq = torch.zeros((), device=device)
                for name, p in network.named_parameters():
                    if not p.requires_grad:
                        continue
                    s = p.detach().float().pow(2).sum()
                    if "lora_up" in name:
                        up_sq = up_sq + s
                    elif "lora_down" in name:
                        down_sq = down_sq + s
            # One CUDA sync per log boundary: stack every scalar, read once.
            stage_means = acc_loss_stage / acc_stage_cnt.clamp(min=1)
            packed = torch.cat(
                [
                    (acc_loss / log_interval).reshape(1),
                    up_sq.sqrt().reshape(1),
                    down_sq.sqrt().reshape(1),
                    stage_means,
                    acc_stage_cnt,
                ]
            ).tolist()
            avg, up_norm, down_norm = packed[0], packed[1], packed[2]
            stage_vals = packed[3 : 3 + n_stages]
            stage_cnts = packed[3 + n_stages : 3 + 2 * n_stages]
            cur_lr = scheduler.get_last_lr()[0]  # CPU-side; no sync
            progress.set_postfix(
                loss=f"{avg:.5f}",
                stage=stage_idx,
                lr=f"{cur_lr:.2e}",
                up=f"{up_norm:.3f}",
            )
            if writer is not None:
                writer.add_scalar("train/loss", avg, step + 1)
                writer.add_scalar("train/lr", cur_lr, step + 1)
                writer.add_scalar("train/lora_up_norm", up_norm, step + 1)
                writer.add_scalar("train/lora_down_norm", down_norm, step + 1)
                # Per-stage mean loss over the interval (only stages touched).
                for si in range(n_stages):
                    if stage_cnts[si] > 0:
                        writer.add_scalar(
                            f"train/loss_stage{si}", stage_vals[si], step + 1
                        )
            acc_loss.zero_()
            acc_loss_stage.zero_()
            acc_stage_cnt.zero_()

        if (step + 1) % save_every == 0 or (step + 1) == iterations:
            _save(step + 1)

    if writer is not None:
        writer.close()
    logger.info("SPD distillation complete.")


if __name__ == "__main__":
    main()
