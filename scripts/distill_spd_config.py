"""SPD fine-tuning config: argparser + CLI/TOML resolver → frozen ``SpdConfig``.

Mirrors the ``distill_turbo/config.py`` precedent (sectioned TOML read raw, CLI
flags override via sentinel defaults, resolved values frozen into a dataclass the
loop never reaches back through). Extracted out of ``scripts/distill_spd.py`` so
the ~230-line argparser and the precedence/schedule-sanity logic stop living
inside the training loop and become independently testable.

Lives as a sibling module (not a ``scripts/distill_spd/`` package) so the
``python scripts/distill_spd.py`` invocation path is unchanged.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from library.config.io import toml_get
from library.config.resolved import load_toml, pick

__all__ = ["build_argparser", "resolve_config", "load_toml", "SpdConfig"]

logger = logging.getLogger(__name__)


def build_argparser() -> argparse.ArgumentParser:
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
    return parser


@dataclass(frozen=True)
class SpdConfig:
    # Paths / IO
    dit_path: str
    data_dir: str
    output_dir: str
    output_name: str
    log_dir: str
    save_every: int
    log_interval: int

    # Run shape
    iterations: int
    batch_size: int
    seed: int

    # LoRA
    rank: int
    alpha: float
    attn_mode: str
    channel_scaling_alpha: float

    # Schedule
    stages: list
    transition_sigmas: list
    stage_weights: list
    schedule_label: str
    sigma_jitter: float

    # Optimizer
    lr: float
    weight_decay: float
    grad_clip: float
    grad_accum: int
    warmup: float

    # Validation
    val_split: float
    val_interval: int
    n_val_sigmas: int
    ema_decay: float

    # On-policy (DAgger)
    onpolicy: bool
    flow_shift: float
    dagger_warmup: int
    onpolicy_ratio: float
    rollout_steps: int

    # SNR gate
    snr_gate_enabled: bool
    snr_gate_mode: str
    snr_gate_delta: float
    snr_gate_profile_n: int
    snr_gate_n_bins: int

    # Compile
    compile_inductor_mode: "str | None"
    compile_dynamic_seq: bool
    activation_memory_budget: float


def resolve_config(args: argparse.Namespace, cfg: dict) -> SpdConfig:
    """Apply CLI/TOML/default precedence + schedule sanity checks → ``SpdConfig``.

    Reads a couple of raw ``args`` flags (``torch_compile`` / ``blocks_to_swap`` /
    ``single_prompt_idx``) for config-time warnings/disabling, but every value the
    training loop consumes comes back through the frozen dataclass.
    """
    dit_path = pick(
        args.dit_path,
        cfg,
        "dit_path",
        "models/diffusion_models/anima-base-v1.0.safetensors",
    )
    data_dir = pick(args.data_dir, cfg, "data_dir", "post_image_dataset/lora")
    output_dir = pick(args.output_dir, cfg, "output_dir", "output/ckpt")
    output_name = pick(args.output_name, cfg, "output_name", "anima_spd")
    iterations = int(pick(args.iterations, cfg, "iterations", 1000))
    batch_size = int(pick(args.batch_size, cfg, "batch_size", 1))
    seed = int(pick(args.seed, cfg, "seed", 42))

    rank = int(pick(args.rank, cfg, "network.rank", 48))
    alpha = float(pick(args.alpha, cfg, "network.alpha", rank))
    attn_mode = pick(args.attn_mode, cfg, "network.attn_mode", "flash")
    channel_scaling_alpha = float(
        pick(args.channel_scaling_alpha, cfg, "network.channel_scaling_alpha", 0.0)
    )
    compile_inductor_mode = pick(
        args.compile_inductor_mode, cfg, "compile_inductor_mode", None
    )
    compile_dynamic_seq = bool(
        pick(args.compile_dynamic_seq, cfg, "compile_dynamic_seq", True)
    )
    activation_memory_budget = float(
        pick(args.activation_memory_budget, cfg, "activation_memory_budget", 1.0)
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

    stages = list(pick(args.stages, cfg, "schedule.stages", [0.5, 1.0]))
    transition_sigmas = list(
        pick(args.transition_sigmas, cfg, "schedule.transition_sigmas", [0.5])
    )
    schedule_label = toml_get(cfg, "schedule.label", "custom")
    sigma_jitter = float(pick(args.sigma_jitter, cfg, "schedule.sigma_jitter", 0.0))

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
        pick(args.stage_weights, cfg, "schedule.stage_weights", [1.0] * len(stages))
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

    lr = float(pick(args.lr, cfg, "optim.lr", 1e-4))
    weight_decay = float(toml_get(cfg, "optim.weight_decay", 0.0))
    grad_clip = float(pick(args.grad_clip, cfg, "optim.grad_clip", 1.0))
    grad_accum = max(1, int(pick(args.grad_accum, cfg, "optim.grad_accum", 1)))
    warmup = float(pick(args.warmup, cfg, "optim.warmup", 0.02))

    save_every = int(pick(args.save_every, cfg, "io.save_every", 500))
    log_interval = int(pick(args.log_interval, cfg, "io.log_interval", 10))
    log_dir = pick(args.log_dir, cfg, "io.log_dir", "output/logs/spd")

    val_split = float(pick(args.val_split, cfg, "io.val_split", 0.0))
    val_interval = int(pick(args.val_interval, cfg, "io.val_interval", save_every))
    n_val_sigmas = max(1, int(pick(args.n_val_sigmas, cfg, "io.n_val_sigmas", 4)))
    ema_decay = float(pick(args.ema_decay, cfg, "optim.ema_decay", 0.0))

    # On-policy (DAgger) stage-entry config.
    onpolicy = bool(args.onpolicy or toml_get(cfg, "onpolicy.enabled", False))
    flow_shift = float(pick(args.flow_shift, cfg, "schedule.flow_shift", 1.0))
    dagger_warmup_raw = float(
        pick(args.dagger_warmup, cfg, "onpolicy.dagger_warmup", 0.25)
    )
    dagger_warmup = (
        int(dagger_warmup_raw)
        if dagger_warmup_raw >= 1
        else int(dagger_warmup_raw * iterations)
    )
    onpolicy_ratio = float(pick(args.onpolicy_ratio, cfg, "onpolicy.ratio", 1.0))
    rollout_steps = int(pick(args.rollout_steps, cfg, "onpolicy.rollout_steps", 12))

    # SNR-gated loss config.
    snr_gate_enabled = bool(pick(args.snr_gate, cfg, "snr_gate.enabled", False))
    snr_gate_mode = pick(args.snr_gate_mode, cfg, "snr_gate.mode", "soft")
    snr_gate_delta = float(pick(args.snr_gate_delta, cfg, "snr_gate.delta", 0.01))
    snr_gate_profile_n = int(
        pick(args.snr_gate_profile_n, cfg, "snr_gate.profile_n", 64)
    )
    snr_gate_n_bins = int(toml_get(cfg, "snr_gate.n_bins", 128))
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

    return SpdConfig(
        dit_path=dit_path,
        data_dir=data_dir,
        output_dir=output_dir,
        output_name=output_name,
        log_dir=log_dir,
        save_every=save_every,
        log_interval=log_interval,
        iterations=iterations,
        batch_size=batch_size,
        seed=seed,
        rank=rank,
        alpha=alpha,
        attn_mode=attn_mode,
        channel_scaling_alpha=channel_scaling_alpha,
        stages=stages,
        transition_sigmas=transition_sigmas,
        stage_weights=stage_weights,
        schedule_label=schedule_label,
        sigma_jitter=sigma_jitter,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        grad_accum=grad_accum,
        warmup=warmup,
        val_split=val_split,
        val_interval=val_interval,
        n_val_sigmas=n_val_sigmas,
        ema_decay=ema_decay,
        onpolicy=onpolicy,
        flow_shift=flow_shift,
        dagger_warmup=dagger_warmup,
        onpolicy_ratio=onpolicy_ratio,
        rollout_steps=rollout_steps,
        snr_gate_enabled=snr_gate_enabled,
        snr_gate_mode=snr_gate_mode,
        snr_gate_delta=snr_gate_delta,
        snr_gate_profile_n=snr_gate_profile_n,
        snr_gate_n_bins=snr_gate_n_bins,
        compile_inductor_mode=compile_inductor_mode,
        compile_dynamic_seq=compile_dynamic_seq,
        activation_memory_budget=activation_memory_budget,
    )
