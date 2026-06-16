"""SPD fine-tuning LoRA — trajectory adapter for progressive-resolution inference.

Trains a *plain* LoRA on one frozen Anima DiT to follow the stage-specific
straight-line velocity targets of the Spectral Progressive Diffusion (SPD)
multi-resolution trajectory (Xiao et al., arXiv:2605.18736, §4.3, Eq. 11–14).
This is "Case B" of the SPD investigation — see
``_archive/proposals/spd_finetune_lora.md``. Output ``output/ckpt/anima_spd.safetensors``
is a normal LoRA: load it through the standard inference path and run it with
the SPD sampler (``--spd``) at the *same* schedule it was trained on.

Models the structure on ``scripts/distill_mod/distill.py`` /
``scripts/distill_turbo/distill.py`` (frozen-DiT + adapter-only + single MSE backward),
but strictly simpler: one adapter, one optimizer.

Usage::

    make exp-spd                                  # defaults from spd.toml
    make exp-spd ARGS="--iterations 2000 --single_prompt_idx 0"   # Phase 0
    make exp-spd PRESET=low_vram                  # block swap + grad ckpt
    make exp-spd ARGS="--torch_compile"           # per-stage static-shape compile

"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
from pathlib import Path


import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from tqdm import tqdm  # noqa: E402

from library.anima import weights as anima_utils  # noqa: E402
from library.anima.models import Anima  # noqa: E402
from library.config.io import toml_get as _flatten  # noqa: E402
from library.datasets.cache import make_cached_collate  # noqa: E402
from library.datasets.cache import CachedDataset  # noqa: E402
from library.runtime.harness import (  # noqa: E402
    compile_dit_blocks_for_pool,
    enable_training_grad_ckpt,
    place_dit_for_training,
)
from library.training.accumulator import ScalarAccumulator  # noqa: E402
from library.training.forward import PadCache, renoise, to_dit_5d  # noqa: E402
from library.training.schedulers import make_warmup_cosine_scheduler  # noqa: E402
from networks.lora_anima.factory import create_network  # noqa: E402
from networks.lora_save import save_network_weights  # noqa: E402
from networks.spd import (  # noqa: E402
    SpdSnrGate,
    _snap,
    dct_lowpass_init,
    measure_dct_power_profile,
    spd_rollout_to_stage,
    spd_schedule_bands,
    spd_stage_target,
    spectral_expand,
)
from library.io.cache import get_latent_resolution, load_cached_latents  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    parser.add_argument(
        "--stage_weights",
        type=float,
        nargs="+",
        default=None,
        help="Per-stage sampling multiplier (len = len(stages)). Stage sampled "
        "∝ (band width × stage_weight) — tilts gradient budget across stages "
        "without disturbing in-band σ density. Omitted/all-ones = band-width "
        "baseline. e.g. 0.3 0.7 leans post-transition. Overrides schedule.stage_weights.",
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
    parser.add_argument(
        "--compile_dynamic_seq",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Collapse the per-(stage x bucket) block graphs into ONE symbolic-seq "
        "graph (mark_dynamic on the seq axis only), mirroring the LoRA-training "
        "compile_dynamic_seq path. The dynamic axis is bounded by the *downsampled* "
        "stage token counts (not just the on-disk full-res latents). Only matters "
        "with --torch_compile. Sentinel None -> TOML (compile_dynamic_seq, default true).",
    )
    parser.add_argument(
        "--activation_memory_budget",
        type=float,
        default=-1.0,
        help="torch.compile partitioner saved-activation fraction (<1.0 recomputes "
        "cheap intermediates in backward to cut peak VRAM; mirrors train.py). "
        "Ignored under --grad_ckpt (CheckpointError otherwise; redundant there). "
        "Sentinel -1 -> TOML (activation_memory_budget, default 1.0 = off).",
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
        "--val_split",
        type=float,
        default=-1.0,
        help="Fraction of each bucket held out (never trained on) for the "
        "CMMD-free analytic-MSE validation signal. 0 = off. Overrides io.val_split.",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=-1,
        help="Run validation every N optimizer steps (+ once at the end). "
        "Defaults to io.val_interval, else save_every.",
    )
    parser.add_argument(
        "--n_val_sigmas",
        type=int,
        default=-1,
        help="Deterministic σ grid points per stage band used by validation "
        "(midpoints of equal sub-intervals). Overrides io.n_val_sigmas (default 4).",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=-1.0,
        help="EMA decay on the LoRA params (e.g. 0.999). 0 = off. When on, the "
        "EMA weights are what gets validated AND saved. Overrides optim.ema_decay.",
    )
    # On-policy (DAgger) stage entry. Analytic entries are a straight line from
    # the true clean low-passed latent, but inference rolls to an imperfect,
    # off-manifold state (exposure bias; bench/spd/probe_onpolicy_handoff.py).
    # --onpolicy instead supervises from the state the adapter-on prefix actually
    # rolls to (spd_rollout_to_stage), targeting the true clean x0. Stage 0 is
    # already on-policy (pure-noise entry), so only stage>0 micro-steps change.
    parser.add_argument(
        "--onpolicy",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Roll the adapter-on prefix to build the stage>0 entry (DAgger). "
        "Validation moves on-policy too, so best-ckpt selection tracks inference.",
    )
    parser.add_argument(
        "--dagger_warmup",
        type=float,
        default=-1.0,
        help="Steps (int >=1) or ratio of iterations (<1) trained fully analytic "
        "before mixing on-policy entries in — the adapter is too random early to "
        "roll a useful prefix. Overrides onpolicy.dagger_warmup (default 0.25).",
    )
    parser.add_argument(
        "--onpolicy_ratio",
        type=float,
        default=-1.0,
        help="Final fraction of stage>0 micro-steps that use the on-policy entry "
        "(ramped in linearly after dagger_warmup). 1.0 = fully on-policy. "
        "Overrides onpolicy.ratio (default 1.0).",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=-1,
        help="Euler steps in the no-grad prefix rollout. Fewer than deploy is fine "
        "(only a plausible on-policy state is needed, not a finished image) and "
        "keeps the extra forwards cheap. Overrides onpolicy.rollout_steps (default 12).",
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=-1.0,
        help="flow_shift for the rollout σ schedule — MUST match the deployed SPD "
        "sampler. Overrides schedule.flow_shift (default 1.0).",
    )
    # SNR-gated (information-aware) loss. Plain velocity MSE supervises bands
    # whose clean coefficient is unrecoverable from x_t (post-expansion HF is
    # fresh noise); the MSE-optimal response there is dataset-mean detail, i.e.
    # learned HF attenuation (blur). The gate weights DCT-domain error by the
    # per-band recoverable fraction (paper Prop. 1 / Eq. 15; see SpdSnrGate).
    parser.add_argument(
        "--snr_gate",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Weight the velocity loss per DCT band by SNR_w(t) so only bands "
        "the input determines are supervised. Sentinel None -> TOML "
        "(snr_gate.enabled). Validation uses the same gated loss, so best-ckpt "
        "selection no longer rewards HF hedging.",
    )
    parser.add_argument(
        "--snr_gate_mode",
        type=str,
        default=None,
        choices=["soft", "hard"],
        help="soft = w=SNR/(1+SNR) (recoverable-variance fraction); hard = "
        "binary 1[t <= t_w] at the Prop.-1 delta-activation time.",
    )
    parser.add_argument(
        "--snr_gate_delta",
        type=float,
        default=-1.0,
        help="Error tolerance delta for the hard gate's activation time "
        "(paper default 0.01). Ignored in soft mode.",
    )
    parser.add_argument(
        "--snr_gate_profile_n",
        type=int,
        default=-1,
        help="Training latents sampled at startup to measure the per-frequency "
        "power profile P_w the gate is built from.",
    )
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
    compile_inductor_mode = pick(
        args.compile_inductor_mode, "compile_inductor_mode", None
    )
    compile_dynamic_seq = bool(
        args.compile_dynamic_seq
        if args.compile_dynamic_seq is not None
        else _flatten(cfg, "compile_dynamic_seq", True)
    )
    activation_memory_budget = float(
        pick(args.activation_memory_budget, "activation_memory_budget", 1.0)
    )
    if (
        args.torch_compile
        and compile_inductor_mode == "reduce-overhead"
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

    # Per-stage sampling multiplier (default all-ones → band-width baseline).
    stage_weights = list(
        args.stage_weights
        if args.stage_weights is not None
        else _flatten(cfg, "schedule.stage_weights", [1.0] * len(stages))
    )
    if len(stage_weights) != len(stages):
        raise ValueError(
            f"schedule.stage_weights (len {len(stage_weights)}) must match "
            f"len(stages) ({len(stages)}); stages={stages}, stage_weights={stage_weights}"
        )
    if any(w < 0 for w in stage_weights) or sum(stage_weights) <= 0:
        raise ValueError(
            f"schedule.stage_weights must be non-negative with positive sum, "
            f"got {stage_weights}"
        )

    lr = float(pick(args.lr, "optim.lr", 1e-4))
    weight_decay = float(_flatten(cfg, "optim.weight_decay", 0.0))
    grad_clip = float(pick(args.grad_clip, "optim.grad_clip", 1.0))
    grad_accum = max(1, int(pick(args.grad_accum, "optim.grad_accum", 1)))
    warmup = float(pick(args.warmup, "optim.warmup", 0.02))

    save_every = int(pick(args.save_every, "io.save_every", 500))
    log_interval = int(pick(args.log_interval, "io.log_interval", 10))
    log_dir = pick(args.log_dir, "io.log_dir", "output/logs/spd")

    val_split = float(pick(args.val_split, "io.val_split", 0.0))
    val_interval = int(pick(args.val_interval, "io.val_interval", save_every))
    n_val_sigmas = max(1, int(pick(args.n_val_sigmas, "io.n_val_sigmas", 4)))
    ema_decay = float(pick(args.ema_decay, "optim.ema_decay", 0.0))

    # On-policy (DAgger) stage-entry config.
    onpolicy = bool(args.onpolicy or _flatten(cfg, "onpolicy.enabled", False))
    flow_shift = float(pick(args.flow_shift, "schedule.flow_shift", 1.0))
    dagger_warmup_raw = float(pick(args.dagger_warmup, "onpolicy.dagger_warmup", 0.25))
    dagger_warmup = (
        int(dagger_warmup_raw)
        if dagger_warmup_raw >= 1
        else int(dagger_warmup_raw * iterations)
    )
    onpolicy_ratio = float(pick(args.onpolicy_ratio, "onpolicy.ratio", 1.0))
    rollout_steps = int(pick(args.rollout_steps, "onpolicy.rollout_steps", 12))

    # SNR-gated loss config.
    snr_gate_enabled = bool(
        args.snr_gate
        if args.snr_gate is not None
        else _flatten(cfg, "snr_gate.enabled", False)
    )
    snr_gate_mode = pick(args.snr_gate_mode, "snr_gate.mode", "soft")
    snr_gate_delta = float(pick(args.snr_gate_delta, "snr_gate.delta", 0.01))
    snr_gate_profile_n = int(pick(args.snr_gate_profile_n, "snr_gate.profile_n", 64))
    snr_gate_n_bins = int(_flatten(cfg, "snr_gate.n_bins", 128))
    if onpolicy and len(stages) < 2:
        logger.warning(
            "onpolicy set but schedule has one stage → no prefix to roll; ignoring."
        )
        onpolicy = False
    if onpolicy:
        logger.info(
            "on-policy DAgger: warmup=%d steps, ratio→%.2f, rollout_steps=%d, flow_shift=%.3g",
            dagger_warmup,
            onpolicy_ratio,
            rollout_steps,
            flow_shift,
        )

    # Phase-0 overfit mode has a single pinned sample → nothing to hold out.
    if args.single_prompt_idx is not None and val_split > 0.0:
        logger.warning(
            "single_prompt_idx set → disabling validation (no held-out data)."
        )
        val_split = 0.0

    torch.manual_seed(seed)

    # Schedule bands (data-independent; weights keep marginal-over-t uniform).
    bands = spd_schedule_bands(stages, transition_sigmas)
    band_widths = torch.tensor([hi - lo for (lo, hi) in bands], dtype=torch.float64)
    # Stage sampling weight = band width × per-stage multiplier. The band-width
    # factor keeps σ marginally-uniform within each sampled stage (paper U(0,1));
    # stage_weights tilt mass across stages without touching in-band σ density.
    stage_w = torch.tensor(stage_weights, dtype=torch.float64)
    stage_sample_w = band_widths * stage_w
    stage_sample_w_f = stage_sample_w.float()  # hoisted for the per-step multinomial
    stage_probs = (stage_sample_w / stage_sample_w.sum()).tolist()
    logger.info(
        "SPD schedule '%s': stages=%s transition_sigmas=%s stage_weights=%s",
        schedule_label,
        stages,
        transition_sigmas,
        stage_weights,
    )
    for i, ((lo, hi), p) in enumerate(zip(bands, stage_probs)):
        logger.info(
            "  stage %d  scale=%.3f  query σ∈(%.4f, %.4f)  w=%.3g  p=%.3f",
            i,
            stages[i],
            lo,
            hi,
            stage_weights[i],
            p,
        )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Dataset (bucket-grouped, one resolution per batch). CachedDataset carves a
    # deterministic per-bucket val slice (seeded by validation_seed) that never
    # overlaps train, mirroring the LoRA pipeline.
    dataset = CachedDataset(
        data_dir,
        batch_size=batch_size,
        sample_ratio=args.sample_ratio,
        split="train",
        validation_split=val_split,
        validation_seed=seed,
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

    # Stacking collate (pooled-text slot returned but unused by SPD). Shared with
    # the val loader; pickle-safe under the Windows/spawn DataLoader start method.
    collate_fn = make_cached_collate()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # bucket-grouped: shuffling would mix resolutions
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Held-out val loader — same batch_size as train so the compiled (stage ×
    # bucket × B) block graphs are reused (a different B would force recompiles
    # and could blow the dynamo cache budget sized below).
    val_loader = None
    if val_split > 0.0:
        val_dataset = CachedDataset(
            data_dir,
            batch_size=batch_size,
            sample_ratio=args.sample_ratio,
            split="val",
            validation_split=val_split,
            validation_seed=seed,
        )
        if len(val_dataset) == 0:
            logger.warning(
                "val_split=%.3g produced 0 held-out samples (dataset too small "
                "for per-bucket carving at this batch_size); validation disabled.",
                val_split,
            )
        else:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_fn,
            )

    # Generator for stage construction (fresh HF noise per step; seed offset so
    # it's independent of the torch global stream used for stage selection).
    gen = torch.Generator(device=device).manual_seed(seed + 7919)

    if args.dry_run:
        for i, (_idx, lat, te, _pooled) in enumerate(tqdm(dataloader, desc="dry-run")):
            lat = lat.to(device, dtype=dtype)
            x0_full = to_dit_5d(lat)
            for s in range(len(stages)):
                x0_si, eps_si = spd_stage_target(
                    x0_full, s, stages, transition_sigmas, patch=1, gen=gen
                )
                assert x0_si.shape == eps_si.shape
            if i >= 20:
                break
        logger.info("Dry run OK: stage-target construction + collation clean.")
        return

    # SNR gate: orthonormal-DCT per-coefficient power, radially binned — measured
    # not power-law-fitted (anime latents carry line-art HF a 2-param fit smooths
    # over). Train split only; a dataset statistic, so a small sample suffices.
    snr_gate = None
    if snr_gate_enabled:
        prof_rng = random.Random(seed + 31)
        prof_paths = [npz for (npz, _te) in dataset.samples]
        prof_rng.shuffle(prof_paths)
        prof_paths = prof_paths[: max(1, snr_gate_profile_n)]
        profile = measure_dct_power_profile(
            (load_cached_latents(p)[0].to(device, torch.float32) for p in prof_paths),
            n_bins=snr_gate_n_bins,
        )
        snr_gate = SpdSnrGate(profile, mode=snr_gate_mode, delta=snr_gate_delta)
        logger.info(
            "SNR gate ON (mode=%s%s): P_w profile from %d latents, %d bins; "
            "P[lowest bin]=%.3g, P[median bin]=%.3g, P[last bin]=%.3g",
            snr_gate_mode,
            f", delta={snr_gate_delta}" if snr_gate_mode == "hard" else "",
            len(prof_paths),
            snr_gate_n_bins,
            float(profile[0]),
            float(profile[snr_gate_n_bins // 2]),
            float(profile[-1]),
        )

    logger.info("Loading DiT model...")
    model: Anima = anima_utils.load_anima_model(
        device,
        dit_path,
        attn_mode=attn_mode,
        loading_device="cpu" if args.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )
    patch = model.patch_spatial

    # Plain LoRA adapter (paper-faithful: no MoE / ortho / T-LoRA).
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
        channel_scaling_alpha=channel_scaling_alpha,
    )
    network.apply_to(
        text_encoders=[], unet=model, apply_text_encoder=False, apply_unet=True
    )

    place_dit_for_training(model, device, blocks_to_swap=args.blocks_to_swap)

    enable_training_grad_ckpt(model, enabled=args.grad_ckpt)
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

    # SPD runs each stage at a DOWNSAMPLED resolution not in CONSTANT_TOKEN_BUCKETS
    # (dct_lowpass_init snaps each latent dim to _snap(dim*scale, patch)), so real
    # forward token counts span more than the 2 full-res families and run *below*
    # them. Enumerate every (stage × on-disk bucket) token count so both compile
    # paths size to the true shape set.
    if args.torch_compile:
        # Distinct (stage × bucket) token counts in the cached pool — each image's
        # full-res latent downsampled per stage scale. This derivation is coupled
        # to SPD's multi-stage schedule, so it stays here; the budget → cache
        # isolation → block-compile glue is the shared library helper.
        stage_bucket_tokens: set[int] = set()
        for npz, _te in dataset.samples:
            w_lat, h_lat = (int(v) for v in get_latent_resolution(npz).split("x"))
            for s in stages:
                h = min(_snap(h_lat * s, patch), h_lat) if s < 1.0 else h_lat
                w = min(_snap(w_lat * s, patch), w_lat) if s < 1.0 else w_lat
                stage_bucket_tokens.add((h // patch) * (w // patch))
        pc = compile_dit_blocks_for_pool(
            model,
            stage_bucket_tokens,
            enabled=True,
            dynamic_seq=compile_dynamic_seq,
            backend=args.dynamo_backend,
            mode=compile_inductor_mode,
            activation_memory_budget=activation_memory_budget,
            grad_ckpt=args.grad_ckpt,
            logger=logger,
        )
        logger.info(
            "torch_compile: %d block._forward compiled (backend=%s, mode=%s, "
            "dynamic_seq=%s); %d distinct (stage x bucket) token counts in %s.",
            len(model.blocks),
            args.dynamo_backend,
            compile_inductor_mode,
            compile_dynamic_seq,
            pc.n_shapes,
            (
                f"seq_range={pc.seq_range} (one symbolic graph)"
                if compile_dynamic_seq
                else "static per-shape graphs"
            ),
        )

    optimizer = torch.optim.AdamW(
        trainable, lr=lr, weight_decay=weight_decay, fused=torch.cuda.is_available()
    )

    # EMA (optional): the shadow tracks a decaying average, so saved/validated
    # weights sit near the sweet spot without hand-picking an iteration. copy_
    # into live params (stable addresses) keeps it cudagraph-safe under reduce-overhead.
    ema_shadow = [p.detach().clone() for p in trainable] if ema_decay > 0.0 else None
    if ema_shadow is not None:
        logger.info(
            "EMA enabled (decay=%.5f); EMA weights are validated + saved.", ema_decay
        )

    @contextlib.contextmanager
    def _ema_weights():
        """Temporarily swap the EMA shadow into the live params (no-op if off)."""
        if ema_shadow is None:
            yield
            return
        backup = [p.detach().clone() for p in trainable]
        with torch.no_grad():
            for p, s in zip(trainable, ema_shadow):
                p.data.copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(trainable, backup):
                    p.data.copy_(b)

    warmup_steps = int(warmup) if warmup >= 1 else int(warmup * iterations)
    scheduler = make_warmup_cosine_scheduler(
        optimizer, iterations, lr, warmup_steps=warmup_steps
    )

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
                    "stage_weights": stage_weights,
                    "rank": rank,
                    "alpha": alpha,
                    "channel_scaling_alpha": channel_scaling_alpha,
                    "lr": lr,
                    "iterations": iterations,
                    "sigma_jitter": sigma_jitter,
                    "val_split": val_split,
                    "val_interval": val_interval,
                    "n_val_sigmas": n_val_sigmas,
                    "ema_decay": ema_decay,
                    "snr_gate": snr_gate_mode if snr_gate is not None else "off",
                    "snr_gate_delta": snr_gate_delta,
                }.items()
            ),
        )
        logger.info("TensorBoard logs -> %s", run_log)

    def _save(step: int):
        save_path = str(Path(output_dir) / f"{output_name}.safetensors")
        with _ema_weights():
            sd = {k: v.detach().clone() for k, v in network.state_dict().items()}
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
                "ss_spd_stage_weights": json.dumps(stage_weights),
                "ss_spd_schedule_label": str(schedule_label),
                "ss_spd_rank": str(rank),
                "ss_channel_scaling_alpha": str(channel_scaling_alpha),
                "ss_spd_step": str(step),
                "ss_spd_onpolicy": str(onpolicy),
                "ss_spd_flow_shift": str(flow_shift),
                "ss_spd_snr_gate": snr_gate_mode if snr_gate is not None else "off",
                "ss_spd_snr_gate_delta": str(snr_gate_delta),
            },
            save_variant="standard",
        )
        logger.info("saved SPD LoRA → %s  (step %d, %d keys)", save_path, step, len(sd))

    stage_rng = torch.Generator().manual_seed(seed + 1)  # CPU: stage / mode selection

    # Pad mask recycled across forwards — a fresh allocation each call would hand
    # the compiled forward a new input address (hostile to reduce-overhead CUDA
    # graphs). cudagraph step-marking stays decoupled (once per optimizer step /
    # val pass), so _forward_dit is hand-rolled, not run_mini_train_forward
    # (which marks per forward).
    pad_cache = PadCache(dtype)

    def _forward_dit(x5, sig_vec, cattn):
        """Single conditional forward at x5's own resolution (adapter on)."""
        pad = pad_cache.get(x5)
        if model.blocks_to_swap:
            model.prepare_block_swap_before_forward()
        with torch.autocast("cuda", dtype=dtype):
            return model.forward_mini_train_dit(
                x5, sig_vec, cattn, padding_mask=pad, skip_pooled_text_proj=True
            )

    def _band_edges(stage_idx, trans):
        """(t_lo, t_hi) query band — precomputed unless σ-jitter built a fresh `trans`."""
        return (
            bands[stage_idx]
            if trans is transition_sigmas
            else spd_schedule_bands(stages, trans)[stage_idx]
        )

    def _stage_entry(x0_full, cattn, stage_idx, trans, gen_, use_onpolicy):
        """Build ``(x0_si, eps_si, t_lo, t_hi)`` for a stage.

        Analytic (default): ``spd_stage_target`` — straight line from the true
        clean LL. On-policy (stage>0): roll the adapter-on prefix from pure noise
        to the entry of ``stage_idx`` (``spd_rollout_to_stage``, no-grad), expand
        to this stage's grid, and recover the FM-consistent effective noise so the
        velocity target ``eps_si − x0_si`` still points at the *true* clean x0 —
        from the off-manifold state inference actually visits. The rollout is
        detached, so gradients flow only through the supervised forward later.
        """
        if not (use_onpolicy and stage_idx > 0):
            x0_si, eps_si = spd_stage_target(
                x0_full, stage_idx, stages, trans, patch=patch, gen=gen_
            )
            t_lo, t_hi = _band_edges(stage_idx, trans)
            return x0_si, eps_si, t_lo, t_hi

        s_hi = stages[stage_idx]
        H_full, W_full = int(x0_full.shape[-2]), int(x0_full.shape[-1])
        x0_si = dct_lowpass_init(x0_full, s_hi, patch) if s_hi < 1.0 else x0_full
        init_noise = torch.randn(
            x0_full.shape, generator=gen_, device=device, dtype=dtype
        )

        def _vfn(x5, sig):
            sig_vec = torch.full((x5.shape[0],), float(sig), device=device, dtype=dtype)
            return _forward_dit(x5, sig_vec, cattn)

        x_entry_lo, sigma_cross, scale_lo = spd_rollout_to_stage(
            _vfn,
            init_noise,
            stages,
            trans,
            infer_steps=rollout_steps,
            flow_shift=flow_shift,
            patch=patch,
            gen=gen_,
            stop_stage=stage_idx,
        )
        x_tilde, t_tilde = spectral_expand(
            x_entry_lo, sigma_cross, scale_lo, s_hi, H_full, W_full, patch, gen_
        )
        t_lo = trans[stage_idx] if stage_idx < len(stages) - 1 else 0.0
        # Degenerate crossing (rollout fell below the band) → analytic fallback.
        if t_tilde <= t_lo + 1e-6:
            x0_si, eps_si = spd_stage_target(
                x0_full, stage_idx, stages, trans, patch=patch, gen=gen_
            )
            lo, hi = _band_edges(stage_idx, trans)
            return x0_si, eps_si, lo, hi
        eps_si = (x_tilde.float() - (1.0 - t_tilde) * x0_si.float()) / t_tilde
        return x0_si, eps_si.to(dtype), t_lo, float(t_tilde)

    def _onpolicy_active(step):
        """Per-step probability a stage>0 micro-step uses the on-policy entry."""
        if not onpolicy or step < dagger_warmup:
            return 0.0
        ramp = (step - dagger_warmup) / max(1, iterations - dagger_warmup)
        return onpolicy_ratio * min(1.0, ramp)

    logger.info("Starting SPD distillation: %d iterations", iterations)
    # Under reduce-overhead (CUDA graphs), grad_accum keeps the previous step's
    # autograd outputs alive when the next step's forward begins, so inductor
    # skips the cudagraph fast path ("outputs from a previous step still require
    # backward"). Marking the step boundary lets the cudagraph tree recycle its
    # static pool each optimizer step. No-op when cudagraphs aren't active.
    cudagraph_step = bool(
        args.torch_compile and compile_inductor_mode == "reduce-overhead"
    )
    data_iter = [iter(dataloader)]  # boxed so _micro_step can refresh on exhaustion
    progress = tqdm(range(iterations), desc="spd")
    n_stages = len(stages)
    # Named GPU-side accumulators flushed in one CUDA sync per log boundary
    # (library.training.accumulator), replacing per-micro-step loss.item() syncs
    # and the per-parameter LoRA-norm .item() walk. Keys: "loss"; per-stage
    # "stage_loss"/"stage_cnt" (width n_stages); "gate_w"/"ungated" (SNR gate on);
    # "up_sq"/"down_sq" added just before flush so they ride the same sync.
    acc = ScalarAccumulator(device)

    # Fixed RNG for validation: reseeded each eval so the ε field (and hence the
    # analytic target) is identical across checkpoints → val/loss is a pure
    # function of the weights, directly comparable step-to-step. (Training's `gen`
    # advances freely instead, so each epoch sees a fresh ε per image — the target
    # is an expectation there, which is the regularizing behaviour we want.)
    val_gen = torch.Generator(device=device)

    @torch.no_grad()
    def _validate():
        """Deterministic held-out velocity-MSE over every stage × a fixed σ grid.

        Sweeps the *full* validation set, all stages, at ``n_val_sigmas`` fixed
        band-midpoints per stage — no sampling, no PE-Core, same memory footprint
        as one training micro-step (won't OOM where CMMD does). When ``onpolicy``
        is on the stage>0 entry is the rolled prefix state (same exposure-bias
        geometry as inference), so best-ckpt selection tracks the deployed sampler
        rather than the analytic line. Returns (overall_mse,
        per_stage_mse[n_stages], per_stage_count[n_stages]).
        """
        val_gen.manual_seed(seed + 104729)
        sums = torch.zeros(n_stages, device=device)
        cnts = torch.zeros(n_stages, device=device)
        if cudagraph_step:
            torch.compiler.cudagraph_mark_step_begin()
        with _ema_weights():
            for _idx, latents, crossattn_emb, _pooled in val_loader:
                latents = latents.to(device, dtype=dtype, non_blocking=True)
                crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
                B = latents.shape[0]
                x0_full = to_dit_5d(latents)
                for stage_idx in range(n_stages):
                    # Entry (and ε) drawn once per (batch, stage), reused across σ.
                    x0_si, eps_si, t_lo, t_hi = _stage_entry(
                        x0_full,
                        crossattn_emb,
                        stage_idx,
                        transition_sigmas,
                        val_gen,
                        onpolicy,
                    )
                    v_target = (eps_si - x0_si).float()
                    for k in range(n_val_sigmas):
                        frac = (k + 0.5) / n_val_sigmas
                        t = torch.full(
                            (B,),
                            t_lo + (t_hi - t_lo) * frac,
                            device=device,
                            dtype=dtype,
                        )
                        x_t = renoise(x0_si, t, eps_si)
                        pred = _forward_dit(x_t, t, crossattn_emb)
                        if snr_gate is not None:
                            v_loss, _ = snr_gate.gated_mse(
                                pred,
                                v_target,
                                t,
                                (int(x0_full.shape[-2]), int(x0_full.shape[-1])),
                            )
                            sums[stage_idx] += v_loss
                        else:
                            sums[stage_idx] += nn.functional.mse_loss(
                                pred.float(), v_target
                            )
                        cnts[stage_idx] += 1
        overall = sums.sum() / cnts.sum().clamp(min=1)
        return overall, sums / cnts.clamp(min=1), cnts

    def _micro_step(step):
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
        x0_full = to_dit_5d(latents)  # (B, 16, 1, H, W)

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
            torch.multinomial(stage_sample_w_f, 1, generator=stage_rng).item()
        )
        # On-policy entry for stage>0 with annealed probability (DAgger): roll the
        # adapter-on prefix from pure noise instead of the analytic straight line,
        # so the LoRA trains on the off-manifold state inference visits. Decided
        # per micro-step (not per optimizer step) so grad_accum keeps mixing
        # analytic/on-policy and the low-/full-res regimes. _stage_entry returns
        # the matching query band (on-policy t_hi = the rollout's aligned σ̃).
        use_op = stage_idx > 0 and (
            float(torch.rand(1, generator=stage_rng).item()) < _onpolicy_active(step)
        )
        x0_si, eps_si, t_lo, t_hi = _stage_entry(
            x0_full, crossattn_emb, stage_idx, trans, gen, use_op
        )
        # FM training sample + analytic velocity target at scale s_i (Eq. 13–14).
        t = (t_lo + (t_hi - t_lo) * torch.rand(B, device=device)).to(dtype)
        x_t = renoise(x0_si, t, eps_si)
        if args.grad_ckpt:  # reentrant checkpoint needs a grad-requiring input
            x_t.requires_grad_()
        v_target = (eps_si - x0_si).float()
        # Native shapes: the forward runs at this stage's real token count (no
        # padding → no flash pad-leak). Flattening is enabled once by
        # compile_blocks above, which traces one graph per (stage × bucket) shape
        # keyed on the real seq_len — nothing per-step to set here.
        pred = _forward_dit(x_t, t, crossattn_emb)
        if snr_gate is not None:
            # Information-aware loss: DCT-domain error weighted by the per-band
            # recoverable fraction at this t. The plain MSE is kept (detached)
            # for the train/loss_ungated comparison curve.
            loss, gate_w = snr_gate.gated_mse(
                pred, v_target, t, (int(x0_full.shape[-2]), int(x0_full.shape[-1]))
            )
            acc.add("gate_w", gate_w)
            acc.add("ungated", nn.functional.mse_loss(pred.detach().float(), v_target))
        else:
            loss = nn.functional.mse_loss(pred.float(), v_target)
        # Scale so accumulated grads are the *mean* over micro-steps (matches a
        # true batch); LR/grad_clip semantics stay invariant to grad_accum.
        (loss / grad_accum).backward()
        return loss.detach(), stage_idx

    # When validation is on, save only on val/loss improvement (best-ckpt-only,
    # like distill-mod) instead of overwriting every save_every steps. With
    # val off, fall back to the step-cadence save below.
    best_val_loss = float("inf")
    for step in progress:
        if cudagraph_step:
            torch.compiler.cudagraph_mark_step_begin()
        step_loss = torch.zeros((), device=device)  # mean micro-loss, GPU-side
        for _ in range(grad_accum):
            micro_loss, stage_idx = _micro_step(step)
            step_loss = step_loss + micro_loss / grad_accum
            acc.add_at(
                "stage_loss", stage_idx, micro_loss, width=n_stages
            )  # python idx → no sync
            acc.add_at("stage_cnt", stage_idx, 1.0, width=n_stages)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        acc.add("loss", step_loss)
        if ema_shadow is not None:
            with torch.no_grad():
                for s, p in zip(ema_shadow, trainable):
                    s.mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)

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
            acc.add("up_sq", up_sq)
            acc.add("down_sq", down_sq)
            # One CUDA sync per log boundary: read every accumulator by name.
            # Per-key reductions (mean over the interval, sqrt of squared-norm
            # sums) run on the returned floats — no further sync.
            m = acc.flush_reset()
            n_micro = log_interval * grad_accum  # micro-steps per log interval
            avg = m["loss"] / log_interval
            up_norm = m["up_sq"] ** 0.5
            down_norm = m["down_sq"] ** 0.5
            gate_w_mean = m.get("gate_w", 0.0) / n_micro
            ungated_mean = m.get("ungated", 0.0) / n_micro
            stage_cnts = m["stage_cnt"]
            stage_vals = [
                ls / c if c > 0 else 0.0 for ls, c in zip(m["stage_loss"], stage_cnts)
            ]
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
                if snr_gate is not None:
                    # Effective supervised fraction + the plain MSE the gated
                    # loss replaced (diverging curves = the gate is doing work).
                    writer.add_scalar("train/gate_w", gate_w_mean, step + 1)
                    writer.add_scalar("train/loss_ungated", ungated_mean, step + 1)
                # Per-stage mean loss over the interval (only stages touched).
                for si in range(n_stages):
                    if stage_cnts[si] > 0:
                        writer.add_scalar(
                            f"train/loss_stage{si}", stage_vals[si], step + 1
                        )

        # Held-out analytic-MSE validation (CMMD-free overfit signal).
        improved = False
        v_overall = None
        if val_loader is not None and (
            (step + 1) % val_interval == 0 or (step + 1) == iterations
        ):
            val_overall, val_stage, val_cnt = _validate()
            packed = torch.cat([val_overall.reshape(1), val_stage, val_cnt]).tolist()
            v_overall = packed[0]
            v_stage = packed[1 : 1 + n_stages]
            v_cnt = packed[1 + n_stages : 1 + 2 * n_stages]
            logger.info(
                "val @ step %d: loss=%.6f  %s",
                step + 1,
                v_overall,
                "  ".join(
                    f"stage{si}={v_stage[si]:.6f}(n={int(v_cnt[si])})"
                    for si in range(n_stages)
                ),
            )
            if writer is not None:
                writer.add_scalar("val/loss", v_overall, step + 1)
                for si in range(n_stages):
                    if v_cnt[si] > 0:
                        writer.add_scalar(f"val/loss_stage{si}", v_stage[si], step + 1)
            if v_overall < best_val_loss:
                best_val_loss = v_overall
                improved = True

        # Save: with validation on, only overwrite the checkpoint when val/loss
        # improves (keep the best, like distill-mod). With val off, fall back to
        # the step-cadence save.
        if val_loader is not None:
            should_save = improved
        else:
            should_save = (step + 1) % save_every == 0 or (step + 1) == iterations
        if should_save:
            _save(step + 1)
        elif v_overall is not None:
            logger.info(
                "skipped save at step %d: val=%.6f >= best=%.6f",
                step + 1,
                v_overall,
                best_val_loss,
            )

    if writer is not None:
        writer.close()
    logger.info("SPD distillation complete.")


if __name__ == "__main__":
    main()
