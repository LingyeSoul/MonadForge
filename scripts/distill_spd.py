"""SPD fine-tuning LoRA — trajectory adapter for progressive-resolution inference.

Trains a *plain* LoRA on one frozen Anima DiT to follow the stage-specific
straight-line velocity targets of the Spectral Progressive Diffusion (SPD)
multi-resolution trajectory (Xiao et al., arXiv:2605.18736, §4.3, Eq. 11–14).
This is "Case B" of the SPD investigation — see
``docs/proposal/spd_finetune_lora.md``. Output ``output/ckpt/anima_spd.safetensors``
is a normal LoRA: load it through the standard inference path and run it with
the SPD sampler (``--spd``) at the *same* schedule it was trained on.

The analytic backbone (``--on_policy_ratio 0``) needs **no teacher, no fake-score
network, no adversarial loop, no CFG-bake** — the §4.3 target velocity is analytic.
The only thing that differs from ordinary Anima LoRA training is the *noising
process*: instead of one straight line from a clean latent to white noise at
full resolution, each step regresses ``v_θ`` onto the per-stage segment of the
SPD trajectory at that stage's resolution. The stage-target construction
(``networks.spd.spd_stage_target``) is shared with the SPD sampler so the
train-time stage-entry state matches the sampler's spectral expansion
bit-for-bit (the Phase-0 contract in the proposal).

With ``--on_policy_ratio > 0`` a fraction of steps switch to a DAgger-style
teacher-distillation tail (see ``_onpolicy_loss``): roll the adapter-on prefix to
the *actual* handoff state and distill the tail toward the frozen base model's own
full-res / full-step gold sample for the same seed — the trusted answer at the
visited state. This is the only place a teacher (the frozen base itself, run
expensive + LoRA-off) enters; there is still no separate teacher network or
adversary, and the gold can be cached offline (a future lever).

Models the structure on ``scripts/distill_mod/distill.py`` /
``scripts/distill_turbo.py`` (frozen-DiT + adapter-only + single MSE backward),
but strictly simpler: one adapter, one optimizer.

Usage::

    make exp-spd                                  # defaults from spd.toml
    make exp-spd ARGS="--iterations 2000 --single_prompt_idx 0"   # Phase 0
    make exp-spd PRESET=low_vram                  # block swap + grad ckpt
    make exp-spd ARGS="--torch_compile"           # per-stage static-shape compile

Compile note: SPD trains one resolution per batch, so the constant-token
bucketing invariant (everything padded to 4096) does NOT apply — the block
input shape varies per (stage x aspect-bucket). ``--torch_compile`` compiles
each block's ``_forward`` with ``dynamic=False`` and lets torch.compile
recompile once per distinct (stage x bucket) shape, each at its real token
count on the normal flash backend (no padding, no masking). The dynamo cache
limit is raised to ``~len(stages) * num_buckets`` so every specialization stays
cached instead of falling back to eager. Recompiles are a one-time warmup cost
(seconds per new shape), not a correctness issue.
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
    _snap,
    spd_rollout_to_stage,
    spd_schedule_bands,
    spd_stage_target,
    spectral_expand,
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


def _stage_static_token_counts(
    samples, stages, patch: int, patch_temporal: int = 1, granule: int = 64
) -> list[int]:
    """Per-stage constant token count for static-shape compile (Option B).

    Enumerates every *unique* latent bucket present in the dataset and replays
    the SPD low-pass snap math (``networks.spd.dct_lowpass_init``) to find the
    largest patchified token count any bucket produces at each stage scale, then
    rounds up to a ``granule`` margin. Padding each stage's batch to its own
    count collapses the aspect-bucket axis (many shapes → one per stage) while
    keeping low-res stages cheap — the per-step ``set_static_token_count`` then
    feeds torch.compile exactly ``len(stages)`` distinct shapes.

    Token count mirrors ``forward_mini_train_dit``: ``(T//pt)*(h//p)*(w//p)`` for
    a ``(1, h, w)`` latent grid (Anima images are single-frame, T=1).
    """
    res_set = {get_latent_resolution(npz_path) for npz_path, _te in samples}
    buckets = []
    for res in res_set:
        a, b = res.split("x")
        buckets.append((int(a), int(b)))

    counts: list[int] = []
    for s in stages:
        mx = 0
        for Hl, Wl in buckets:
            if s < 1.0:
                h = min(_snap(Hl * s, patch), Hl)
                w = min(_snap(Wl * s, patch), Wl)
            else:
                h, w = Hl, Wl
            tok = (1 // patch_temporal) if patch_temporal > 1 else 1
            tok *= (h // patch) * (w // patch)
            mx = max(mx, tok)
        counts.append(((mx + granule - 1) // granule) * granule)
    return counts

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
    # --- On-policy handoff tail (proposal.md Idea 2; Phase-0 PASSED 2026-05-21) ---
    parser.add_argument(
        "--on_policy_ratio",
        type=float,
        default=-1.0,
        help="Fraction of steps that train the full-res tail on the *on-policy* handoff "
        "state (prefix rollout → spectral_expand → tail) by distilling toward the "
        "frozen-base teacher's gold sample, vs the analytic backbone. 0 = analytic-only. "
        "The analytic steps still train the prefix (the on-policy rollout is no_grad), "
        "so MIX, don't set to 1.0. "
        "v0: 2-stage schedules only. Works with --torch_compile (plain inductor — "
        "rollout + tail recompile per shape); not with --compile_inductor_mode "
        "reduce-overhead (CUDAGraphs can't span the eval/grad toggle).",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=-1,
        help="Euler steps for the on-policy prefix rollout (coarser than inference is "
        "fine; the rollout only needs a representative handoff state). Default 16.",
    )
    parser.add_argument(
        "--teacher_steps",
        type=int,
        default=-1,
        help="Euler steps for the frozen-base teacher's full-res gold rollout (LoRA "
        "off), which supplies the on-policy clean target. Should be at least as fine "
        "as deployment. Default 24.",
    )
    parser.add_argument(
        "--tail_band",
        type=str,
        default=None,
        choices=["entry", "full"],
        help="On-policy supervision site: 'entry' = the handoff σ̃ only (attacks the "
        "seam, cheapest); 'full' = DAgger across the tail band (roll the tail no_grad "
        "to a random σ in [sigma_floor, σ̃], supervise there — needed to claim the "
        "high-res stage can be shortened). Default 'entry'.",
    )
    parser.add_argument(
        "--sigma_floor",
        type=float,
        default=-1.0,
        help="Lower σ clamp for the 'full' tail-band sampling (avoids the high-variance "
        "(x−x0)/σ target as σ→0). Default 0.1.",
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=-1.0,
        help="Flow-matching σ-schedule shift for the rollout (matches inference). "
        "Default 1.0 (base.toml discrete_flow_shift).",
    )
    parser.add_argument("--lr", type=float, default=-1.0)
    parser.add_argument("--grad_clip", type=float, default=-1.0)
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

    # On-policy handoff tail (Idea 2).
    on_policy_ratio = float(pick(args.on_policy_ratio, "onpolicy.ratio", 0.0))
    rollout_steps = int(pick(args.rollout_steps, "onpolicy.rollout_steps", 16))
    teacher_steps = int(pick(args.teacher_steps, "onpolicy.teacher_steps", 24))
    tail_band = pick(args.tail_band, "onpolicy.tail_band", "entry")
    sigma_floor = float(pick(args.sigma_floor, "onpolicy.sigma_floor", 0.1))
    flow_shift = float(pick(args.flow_shift, "onpolicy.flow_shift", 1.0))

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

    # On-policy v0 limitations (proposal.md Idea 2 file-level plan).
    if on_policy_ratio > 0.0:
        if not (0.0 < on_policy_ratio <= 1.0):
            raise ValueError(
                f"--on_policy_ratio must be in (0,1], got {on_policy_ratio}"
            )
        if on_policy_ratio >= 1.0:
            logger.warning(
                "--on_policy_ratio=1.0 trains *only* on-policy; nothing anchors the prefix "
                "(one LoRA, no module-level prefix/tail split) so it can drift. Prefer ≤0.5."
            )
        if len(stages) != 2:
            raise ValueError(
                f"--on_policy_ratio v0 supports 2-stage schedules only, got stages={stages}."
            )
        if args.torch_compile and args.compile_inductor_mode == "reduce-overhead":
            # Plain inductor is fine: torch.compile recompiles per shape, so the
            # rollout (low-res) and tail (full-res) legs each get their own
            # specialization. reduce-overhead (CUDAGraphs) is the exception: it
            # can't capture across the eval↔train + no_grad↔grad toggle the
            # two-pass step makes, and pins freshly-allocated inputs.
            raise ValueError(
                "--on_policy_ratio with --torch_compile requires plain inductor: drop "
                "--compile_inductor_mode reduce-overhead (CUDAGraphs can't span the "
                "rollout's eval/no_grad ↔ tail train/grad toggle)."
            )

    lr = float(pick(args.lr, "optim.lr", 1e-4))
    weight_decay = float(_flatten(cfg, "optim.weight_decay", 0.0))
    grad_clip = float(pick(args.grad_clip, "optim.grad_clip", 1.0))
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
    if on_policy_ratio > 0.0:
        logger.info(
            "on-policy teacher-distill tail: ratio=%.2f  band=%s  rollout_steps=%d  "
            "teacher_steps=%d  sigma_floor=%.3f  flow_shift=%.2f  "
            "(target (x̃−x0_teacher)/σ̃; teacher = frozen base, LoRA off, full-res)",
            on_policy_ratio,
            tail_band,
            rollout_steps,
            teacher_steps,
            sigma_floor,
            flow_shift,
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
    # fp32 cast in backward instead of stashing the fp32 copy (bitwise-identical
    # for the no-channel-scale path SPD uses). Trims LoRA-branch activation memory
    # and avoids a per-Linear bf16 intermediate getting pinned in the CUDA-Graph
    # pool under --compile_inductor_mode reduce-overhead. See custom_autograd.py.
    network = create_network(
        multiplier=1.0,
        network_dim=rank,
        network_alpha=alpha,
        vae=None,
        text_encoders=[],
        unet=model,
        use_custom_down_autograd=True,
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
        n_shapes = len(stages) * max(1, n_buckets)
        stage_token_counts = _stage_static_token_counts(
            dataset.samples, stages, patch, model.patch_temporal
        )
        # fwd + bwd entries share the one `_forward` bytecode; give headroom.
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
        sd = {k: v for k, v in sd.items() if ".lora_" in k or ".alpha" in k}
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

    def _teacher_denoise(eps_full, cattn, n_steps, fshift):
        """Frozen-base gold rollout: full-res, full-step Euler denoise from ``eps_full``
        with the LoRA *disabled* → the clean sample the expensive (non-SPD) path
        produces for this seed+prompt. Conditional-only + ``skip_pooled_text_proj``,
        matching the student tail forward, so teacher and student live on the same
        conditioning manifold. Caller has already disabled the adapter and set the
        full-res static token count; this just runs the Euler loop (no_grad upstream).
        """
        sig = torch.linspace(1.0, 0.0, n_steps + 1, device=device, dtype=torch.float32)
        sig = (fshift * sig) / (1.0 + (fshift - 1.0) * sig)
        x = eps_full
        for i in range(n_steps):
            s = float(sig[i])
            v = _forward_dit(
                x, x.new_full((x.shape[0],), s, dtype=dtype), cattn
            ).float()
            x = (x.float() + v * (float(sig[i + 1]) - s)).to(dtype)
        return x  # ≈ x0_teacher at σ=0

    def _onpolicy_loss(x0f, cattn, trans):
        """On-policy *teacher-distillation* step (2-stage only). Roll the adapter-on
        SPD prefix from pure noise to the handoff and spectral-expand to full res —
        the exact state the deployed sampler visits — then regress the grad tail
        toward ``(x̃ − x0_teacher)/σ̃``, where ``x0_teacher`` is the frozen base
        model's own full-res / full-step gold sample for the *same* ε and prompt.

        Why the teacher, not the dataset latent: the prefix rolls a fresh ε to *a*
        prompt-consistent sample, never to the specific dataset ``x0`` (Phase-0 probe:
        implied-clean recovery rel_x0_on ≈ 0.83→0.98). Targeting dataset ``x0`` asks
        the tail to bend every trajectory toward one arbitrary latent → mean-regression
        → blur. The frozen base, run the expensive seam-free way from the *same* seed,
        is the trusted "correct answer" at the visited state (DAgger expert), and the
        shared seed keeps its layout aligned with the cheap path's so the tail's job is
        HF/seam correction, not a relayout.
        """
        B = x0f.shape[0]
        H, W = int(x0f.shape[-2]), int(x0f.shape[-1])
        # Shared init: full-res white noise. Lowpassed internally for the prefix
        # rollout; consumed at full res by the teacher → both target the same sample.
        eps_full = torch.randn(
            x0f.shape, generator=gen, device=device, dtype=torch.float32
        ).to(dtype)

        def _roll_v(x5, sig):
            return _forward_dit(
                x5, x5.new_full((x5.shape[0],), float(sig), dtype=dtype), cattn
            )

        # Roll in EVAL mode: the LoRA then takes its inference branch — matching
        # (a) deployment (SPD inference runs eval) and (b) the Phase-0 probe that
        # validated this approach — so the on-policy states are the ones the adapter
        # will actually face, not train-mode (fp32-bottleneck) variants. Also skips
        # the wasted custom-autograd fp32 path under no_grad. Restored to train()
        # before the grad tail forward (where custom autograd + grad-ckpt must be live).
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                # --- Teacher: frozen-base gold (LoRA OFF), full-res from the same ε.
                # The teacher rolls at full res, so pin the full-res static count
                # FIRST. Otherwise a stale low-res count left by the previous step
                # (e.g. an analytic step at stage 0) makes forward_mini_train_dit
                # pad to a target < the real seq_len → negative pad truncates the
                # sequence → the unpad reshape blows up. Same count as the grad
                # tail below, so no extra compile graph.
                if stage_token_counts is not None:
                    model.set_static_token_count(stage_token_counts[-1])
                network.set_enabled(False)
                x0_teacher = _teacher_denoise(eps_full, cattn, teacher_steps, flow_shift)
                network.set_enabled(True)
                if stage_token_counts is not None:
                    model.set_static_token_count(stage_token_counts[0])
                # --- Student: adapter-on cheap SPD prefix → on-policy handoff state.
                # The rollout runs at the stage-0 (pre-expansion) resolution and the
                # tail at full res; torch.compile recompiles per shape on demand.
                x_entry, sig_cross, scale_lo = spd_rollout_to_stage(
                    _roll_v,
                    eps_full,
                    stages,
                    trans,
                    infer_steps=rollout_steps,
                    flow_shift=flow_shift,
                    patch=patch,
                    gen=gen,
                    stop_stage=1,
                )
                if stage_token_counts is not None:
                    model.set_static_token_count(stage_token_counts[-1])
                x_tilde, sig_tilde = spectral_expand(
                    x_entry, sig_cross, scale_lo, 1.0, H, W, patch, gen
                )
                # Tail band: 'entry' supervises at the handoff σ̃ (attacks the seam);
                # 'full' rolls the tail no_grad to a random σ in [sigma_floor, σ̃] and
                # supervises there (DAgger across the band — lets the high-res stage
                # be shortened). Both states are on-policy (adapter-on rollout).
                if tail_band == "full" and sig_tilde > sigma_floor + 1e-4:
                    n_tail = max(2, int(round(rollout_steps * sig_tilde)))
                    tsig = torch.linspace(sig_tilde, 0.0, n_tail + 1, device=device)
                    tsig = (flow_shift * tsig) / (1.0 + (flow_shift - 1.0) * tsig)
                    valid = [
                        k for k in range(1, n_tail) if float(tsig[k]) >= sigma_floor
                    ]
                    stop_k = (
                        valid[int(torch.randint(len(valid), (1,), generator=stage_rng))]
                        if valid
                        else 1
                    )
                    xs = x_tilde
                    for k in range(stop_k):
                        v = _roll_v(xs, float(tsig[k])).float()
                        xs = (
                            xs.float() + v * (float(tsig[k + 1]) - float(tsig[k]))
                        ).to(dtype)
                    x_state, sig_state = xs, float(tsig[stop_k])
                else:
                    x_state, sig_state = x_tilde, sig_tilde
        finally:
            network.set_enabled(True)  # never leave the adapter off for the grad tail
            if was_training:
                model.train()

        # Grad tail forward at the (full-res) on-policy state, distilling toward the
        # frozen-base gold: (x̃ − x0_teacher)/σ̃ — the straight line from the visited
        # state to the sample the expensive path produces for this seed.
        x_state = x_state.detach()
        if args.grad_ckpt:  # reentrant checkpoint needs a grad-requiring input
            x_state.requires_grad_()
        pred = _forward_dit(
            x_state, x_state.new_full((B,), float(sig_state), dtype=dtype), cattn
        )
        v_target = (x_state.detach().float() - x0_teacher.float()) / float(sig_state)
        return nn.functional.mse_loss(pred.float(), v_target)

    # --- Training loop ---
    logger.info("Starting SPD distillation: %d iterations", iterations)
    data_iter = iter(dataloader)
    running = 0.0
    progress = tqdm(range(iterations), desc="spd")
    for step in progress:
        try:
            _idx, latents, crossattn_emb, _pooled = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            _idx, latents, crossattn_emb, _pooled = next(data_iter)

        latents = latents.to(device, dtype=dtype, non_blocking=True)
        crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
        B = latents.shape[0]
        x0_full = latents.unsqueeze(2)  # (B, 16, 1, H, W)

        # Optional R2 jitter: perturb the transition σ so the segment geometry is
        # learned as a band, not a point (shared by both training modes).
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

        # Mode select: on-policy handoff tail (Idea 2) vs analytic backbone. The
        # analytic steps anchor the prefix, so we mix rather than replace.
        on_policy = (
            on_policy_ratio > 0.0
            and float(torch.rand(1, generator=stage_rng).item()) < on_policy_ratio
        )
        if on_policy:
            stage_idx = len(stages) - 1  # logs against the full-res tail
            loss = _onpolicy_loss(x0_full, crossattn_emb, trans)
        else:
            # Sample one stage for the whole batch (single-resolution per step),
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
            # Pad this stage's tokens to its constant count so the compiled blocks
            # see a single shape per stage. No-op when --torch_compile is off.
            if stage_token_counts is not None:
                model.set_static_token_count(stage_token_counts[stage_idx])
            pred = _forward_dit(x_t, t, crossattn_emb)
            loss = nn.functional.mse_loss(pred.float(), v_target)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        loss_v = loss.item()
        running += loss_v
        cur_lr = scheduler.get_last_lr()[0]
        if (step + 1) % log_interval == 0:
            avg = running / log_interval
            running = 0.0
            with torch.no_grad():
                up_sq = 0.0
                down_sq = 0.0
                for name, p in network.named_parameters():
                    if not p.requires_grad:
                        continue
                    s = p.detach().float().pow(2).sum().item()
                    if "lora_up" in name:
                        up_sq += s
                    elif "lora_down" in name:
                        down_sq += s
                up_norm = up_sq**0.5
                down_norm = down_sq**0.5
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
                writer.add_scalar(f"train/loss_stage{stage_idx}", loss_v, step + 1)

        if (step + 1) % save_every == 0 or (step + 1) == iterations:
            _save(step + 1)

    if writer is not None:
        writer.close()
    logger.info("SPD distillation complete.")


if __name__ == "__main__":
    main()
