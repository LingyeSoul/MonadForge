"""Turbo distillation config: TOML loader + argparser + CLI/TOML resolver.

The resolved knobs are returned as a ``TurboConfig`` frozen dataclass so the
training loop never reaches back into ``args``/``cfg`` mid-step.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

# Python 3.11+; fall back to ``tomli`` if needed.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TOML helpers
# ---------------------------------------------------------------------------


def load_turbo_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _flatten(cfg: dict, key_path: str, default):
    """Look up ``a.b.c`` in a nested TOML dict, falling back to ``default``."""
    node = cfg
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _pick(cli_val: Any, cfg: dict, toml_key: str, default: Any) -> Any:
    """CLI override (non-sentinel) wins, else TOML, else default.

    Sentinels are: ``None`` (explicitly unset), ``-1`` (int default), ``-1.0``
    (float default). They mean "use the TOML/default value".
    """
    if cli_val is not None and cli_val != -1 and cli_val != -1.0:
        return cli_val
    return _flatten(cfg, toml_key, default)


# ---------------------------------------------------------------------------
# Argparser
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turbo Anima — Decoupled DMD2 distillation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/methods/turbo.toml",
        help="Path to the turbo TOML config (CLI flags override TOML values).",
    )
    # CLI overrides — every TOML key has a matching flag. Default sentinels
    # (None / -1.0) mean "use the TOML value".
    parser.add_argument("--dit_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--student_rank", type=int, default=-1)
    parser.add_argument("--fake_rank", type=int, default=-1)
    parser.add_argument(
        "--use_custom_down_autograd",
        action="store_true",
        default=None,
        help="Memory-saving down-projection autograd (skips fp32 input save). "
        "Default: read from TOML (top-level scalar), else off.",
    )
    parser.add_argument(
        "--no_use_custom_down_autograd",
        dest="use_custom_down_autograd",
        action="store_false",
    )
    parser.add_argument(
        "--use_masked_loss",
        action="store_true",
        default=None,
        help="Apply the per-image foreground mask to the student DMD2 gradient "
        "(masked-out latents get zero student push). Fake/critic loss is "
        "unaffected. Default: read from TOML (top-level scalar), else off.",
    )
    parser.add_argument(
        "--no_use_masked_loss",
        dest="use_masked_loss",
        action="store_false",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Mask root for --use_masked_loss (default: TOML mask_dir, else "
        "post_image_dataset/masks). Mirrors data_dir's subdir layout.",
    )
    parser.add_argument("--student_lr", type=float, default=-1.0)
    parser.add_argument("--fake_lr", type=float, default=-1.0)
    parser.add_argument(
        "--fake_steps_per_student_step",
        type=int,
        default=-1,
        help="Number of fake (DM regularizer) updates per student step. "
        "Standard DMD2 practice keeps the fake ahead of the moving x_pred "
        "distribution; >1 gives the fake extra SGD iterations on resampled "
        "(τ, ε) noise against the same x_pred.detach(). Default: TOML "
        "(optim.fake_steps_per_student_step, default 1).",
    )
    parser.add_argument(
        "--fake_warmup_steps",
        type=int,
        default=-1,
        help="Fake-only (critic head-start) updates run BEFORE the main loop. "
        "The student LR warmup finishes at ~0.02·iterations, so the student "
        "starts full-strength steps while the zero-init fake/critic LoRA is "
        "still ≈ the teacher → a large, misaligned delta_dm and an early "
        "grad_signal_rms spike (~step 50). Pre-training the fake net against the "
        "student's (init ≈ teacher) x_pred distribution calibrates it first. "
        "The fake scheduler IS stepped during warmup (the main-loop scheduler "
        "is sized over iterations + fake_warmup_steps so the 2%% LR warmup "
        "overlaps the head-start and the fake enters the main loop at full LR). "
        "Default: TOML (optim.fake_warmup_steps, default 0 = off).",
    )
    parser.add_argument(
        "--student_steps",
        type=int,
        default=-1,
        help="Sampler step count baked into the student",
    )
    parser.add_argument(
        "--dm_x0_norm",
        dest="dm_x0_norm",
        action="store_const",
        const=True,
        default=None,
        help="DMD per-sample x0-space magnitude normalization (policy 'b'): "
        "grad_dm = τ·Δ_dm / clamp(τ·mean|v_real|, norm_floor). Because the denom "
        "≈ τ·mean|v_real|, the τ CANCELS across the bulk → ≈ no-τ, magnitude-"
        "normalized. This REPLACES the default τ-damping (policy 'a'); it does NOT "
        "stack with it (that would be policy 'c'). A/B lever — see "
        "docs/proposal/dmd2_decoupled_improvements.md §2B.",
    )
    parser.add_argument(
        "--norm_floor",
        type=float,
        default=-1.0,
        help="clamp_min for the x0-norm denominator (latent scale); only active "
        "with --dm_x0_norm.",
    )
    # Mean-variance reg (lever B / paper Eq. 7; proposal §3.B / S2). Pulls each
    # generated image's (μ_i, σ²_i) toward the real-latent target — clamps the
    # variance inflation that is the over-bake's oversaturation.
    parser.add_argument(
        "--mean_var_weight",
        type=float,
        default=-1.0,
        help="Weight on the Eq.7 mean-variance KL added to the student loss. "
        "0 disables; S2 uses ~0.01–0.05. The target stats are read from TOML "
        "([mean_var].mu_t/sigma2_t), or measured exactly in a one-pass scan over "
        "the real latents when sigma2_t <= 0. Default: TOML (mean_var.weight, "
        "default 0).",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--attn_mode", type=str, default="flash")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        default=True,
        help="Compile block._forward. Off by default — multiple forwards per step "
        "are not yet validated under cudagraphs; turn on once Phase 0 is green.",
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

    # ---- DP-DMD (diversity-preserved DMD; arXiv 2602.03139) ----
    # The student is a genuine N-step rollout: step 1 supervised toward a teacher
    # K-step anchor (diversity), detached, then DMD on x_θ over steps 2..N
    # (quality). See docs/experimental/dpdmd.md.
    parser.add_argument(
        "--k_anchor",
        type=int,
        default=-1,
        help="DP-DMD: teacher steps rolled to the diversity anchor (their K). "
        "Default: TOML (dpdmd.k_anchor, default 5).",
    )
    parser.add_argument(
        "--teacher_anchor_steps",
        type=int,
        default=-1,
        help="DP-DMD: teacher σ-grid the K anchor is counted against. Default: "
        "TOML (dpdmd.teacher_anchor_steps, default 28).",
    )
    parser.add_argument(
        "--div_weight",
        type=float,
        default=-1.0,
        help="DP-DMD: λ on the first-step diversity loss. Default: TOML "
        "(dpdmd.div_weight, default 0.05).",
    )
    parser.add_argument(
        "--detach_after_first",
        dest="detach_after_first",
        action="store_const",
        const=True,
        default=None,
        help="DP-DMD: stop-grad after the diversity-supervised first step (the "
        "load-bearing detach; keep True except for A/B). Default: TOML "
        "(dpdmd.detach_after_first, default true).",
    )
    parser.add_argument(
        "--no_detach_after_first",
        dest="detach_after_first",
        action="store_false",
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=-1.0,
        help="DP-DMD: σ-schedule shift for the student/teacher Euler grids "
        "(matches inference). Default: TOML (sampling.flow_shift, default 3.0).",
    )
    return parser


# ---------------------------------------------------------------------------
# Resolved config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurboConfig:
    # Paths / IO
    dit_path: str
    data_dir: str
    output_dir: str
    output_name: str
    log_dir: str
    save_every: int
    log_interval: int
    no_log: bool

    # Run shape
    iterations: int
    batch_size: int
    seed: int
    sample_ratio: float
    single_prompt_idx: int | None

    # LoRA stacks
    student_rank: int
    fake_rank: int
    student_alpha: float
    fake_alpha: float
    attn_mode: str
    use_custom_down_autograd: bool

    # Masked loss
    use_masked_loss: bool
    mask_dir: str

    # DP-DMD knobs
    k_anchor: int
    teacher_anchor_steps: int
    div_weight: float
    detach_after_first: bool
    flow_shift: float

    # DMD core
    student_steps: int
    teacher_cfg: float
    dm_x0_norm: bool
    norm_floor: float

    # Mean-variance reg (lever B / Eq. 7)
    mean_var_weight: float
    mv_mu_t: float
    mv_sigma2_t: float
    mv_calib_batches: int

    # Optimizer + scheduler
    student_lr: float
    fake_lr: float
    fake_steps_per_student_step: int
    fake_warmup_steps: int
    weight_decay: float
    grad_clip: float

    # Sampling distribution
    t_distribution: str
    sigmoid_scale: float

    # Runtime
    blocks_to_swap: int
    grad_ckpt: bool
    torch_compile: bool


def resolve_config(args: argparse.Namespace, cfg: dict) -> TurboConfig:
    """Apply CLI/TOML/default precedence and run sanity checks."""

    # Paths
    dit_path = _pick(
        args.dit_path, cfg, "dit_path",
        "models/diffusion_models/anima-base-v1.0.safetensors",
    )
    data_dir = _pick(args.data_dir, cfg, "data_dir", "post_image_dataset/lora")
    output_dir = _pick(args.output_dir, cfg, "output_dir", "output/ckpt")
    output_name = _pick(args.output_name, cfg, "output_name", "anima_turbo")
    log_dir = _pick(args.log_dir, cfg, "io.log_dir", "output/logs/turbo")
    save_every = int(_pick(args.save_every, cfg, "io.save_every", 1000))
    log_interval = int(_pick(args.log_interval, cfg, "io.log_interval", 2))

    # Run shape
    iterations = int(_pick(args.iterations, cfg, "iterations", 20000))
    batch_size = int(_pick(args.batch_size, cfg, "batch_size", 1))
    seed = int(_pick(args.seed, cfg, "seed", 42))

    # LoRA stacks
    student_rank = int(_pick(args.student_rank, cfg, "network.student_rank", 48))
    fake_rank = int(_pick(args.fake_rank, cfg, "network.fake_rank", 48))
    student_alpha = float(_flatten(cfg, "network.student_alpha", student_rank))
    fake_alpha = float(_flatten(cfg, "network.fake_alpha", fake_rank))
    attn_mode = _pick(args.attn_mode, cfg, "network.attn_mode", "flash")
    # use_custom_down_autograd lives at TOML top level (matches the LoRA family's
    # config layout in methods/lora.toml). CLI flag wins when set explicitly.
    if args.use_custom_down_autograd is None:
        use_custom_down_autograd = bool(_flatten(cfg, "use_custom_down_autograd", False))
    else:
        use_custom_down_autograd = bool(args.use_custom_down_autograd)

    # Masked loss
    if args.use_masked_loss is None:
        use_masked_loss = bool(_flatten(cfg, "use_masked_loss", False))
    else:
        use_masked_loss = bool(args.use_masked_loss)
    mask_dir = _pick(args.mask_dir, cfg, "mask_dir", "post_image_dataset/masks")

    # DMD core
    student_steps = int(_pick(args.student_steps, cfg, "dmd.student_steps", 4))
    teacher_cfg = float(_flatten(cfg, "dmd.teacher_cfg", 4.0))
    # DM-branch gradient policy: (a) τ-damping [default] vs (b) DMD per-sample
    # x0-space magnitude normalization. Alternative policies, not additive;
    # (b) ≈ "drop the τ-weight, magnitude-normalize."
    dm_x0_norm = bool(_pick(args.dm_x0_norm, cfg, "dmd.dm_x0_norm", False))
    norm_floor = float(_pick(args.norm_floor, cfg, "dmd.norm_floor", 0.05))

    # DP-DMD — diversity-anchor knobs.
    k_anchor = int(_pick(args.k_anchor, cfg, "dpdmd.k_anchor", 5))
    teacher_anchor_steps = int(
        _pick(args.teacher_anchor_steps, cfg, "dpdmd.teacher_anchor_steps", 28)
    )
    div_weight = float(_pick(args.div_weight, cfg, "dpdmd.div_weight", 5e-2))
    if args.detach_after_first is None:
        detach_after_first = bool(_flatten(cfg, "dpdmd.detach_after_first", True))
    else:
        detach_after_first = bool(args.detach_after_first)
    flow_shift = float(_pick(args.flow_shift, cfg, "sampling.flow_shift", 3.0))

    # Mean-variance reg (lever B / Eq. 7). weight=0 disables. Target stats are
    # pinned (sigma2_t > 0) or measured exactly in a one-pass scan over the real
    # latents (sigma2_t <= 0). The target is a static dataset statistic — a global
    # scalar (μ, σ²) over the cached training latents — so an exact pre-pass beats
    # the old running EMA (no decay lag, no batch wobble, deterministic).
    # `calib_batches` caps that scan (0 = full pass; the global scalar converges
    # in a few hundred images, so a cap is a cheap-but-near-exact knob).
    mean_var_weight = float(_pick(args.mean_var_weight, cfg, "mean_var.weight", 0.0))
    mv_mu_t = float(_flatten(cfg, "mean_var.mu_t", 0.0))
    mv_sigma2_t = float(_flatten(cfg, "mean_var.sigma2_t", -1.0))
    mv_calib_batches = int(_flatten(cfg, "mean_var.calib_batches", 0))

    # Optimizer
    student_lr = float(_pick(args.student_lr, cfg, "optim.student_lr", 1e-5))
    fake_lr = float(_pick(args.fake_lr, cfg, "optim.fake_lr", 1e-5))
    fake_steps_per_student_step = int(
        _pick(
            args.fake_steps_per_student_step,
            cfg,
            "optim.fake_steps_per_student_step",
            1,
        )
    )
    fake_warmup_steps = int(
        _pick(args.fake_warmup_steps, cfg, "optim.fake_warmup_steps", 0)
    )
    weight_decay = float(_flatten(cfg, "optim.weight_decay", 0.0))
    grad_clip = float(_flatten(cfg, "optim.grad_clip", 1.0))

    # Sampling
    t_distribution = _flatten(cfg, "sampling.t_distribution", "uniform")
    sigmoid_scale = float(_flatten(cfg, "sampling.sigmoid_scale", 1.0))

    # ----- Validation -----
    if student_steps < 2:
        raise ValueError(
            f"DP-DMD requires dmd.student_steps >= 2 (got {student_steps}): step 1 "
            "is diversity-supervised + detached, so at least one further step must "
            "carry the DMD loss."
        )
    if not (1 <= k_anchor < teacher_anchor_steps):
        raise ValueError(
            f"dpdmd.k_anchor={k_anchor} must satisfy 1 <= k_anchor < "
            f"teacher_anchor_steps={teacher_anchor_steps}."
        )
    if div_weight < 0.0:
        raise ValueError(f"dpdmd.div_weight={div_weight}: must be >= 0")
    if flow_shift <= 0.0:
        raise ValueError(f"sampling.flow_shift={flow_shift}: must be > 0")
    if not detach_after_first:
        logger.warning(
            "detach_after_first=False: the mode-seeking DMD gradient can override "
            "the diversity mapping (their Fig 5). A/B only — keep True for "
            "production."
        )
    if t_distribution not in ("uniform", "sigmoid"):
        raise ValueError(
            f"sampling.t_distribution={t_distribution!r}: expected 'uniform' or 'sigmoid'"
        )
    if fake_rank < student_rank:
        logger.warning(
            f"fake_rank={fake_rank} < student_rank={student_rank}: DM regularizer "
            "has less capacity than the student — proposal R1 risk amplified. "
            "Consider bumping fake_rank to 2 x student_rank."
        )
    if norm_floor <= 0.0:
        raise ValueError(f"dmd.norm_floor={norm_floor}: must be > 0 (latent scale)")
    if fake_steps_per_student_step < 1:
        raise ValueError(
            f"optim.fake_steps_per_student_step={fake_steps_per_student_step}: must be ≥ 1"
        )
    if args.single_prompt_idx is not None and batch_size != 1:
        # single-prompt mode slices the dataset to one sample. With drop_last=True
        # and batch_size > 1 the dataloader yields zero batches and the loop
        # silently no-ops.
        raise ValueError(
            f"--single_prompt_idx requires batch_size=1 (got {batch_size}). "
            "Single-prompt overfit mode pins the dataset to one sample; a "
            "batch_size > 1 dataloader with drop_last=True would yield zero batches."
        )
    if mean_var_weight < 0.0:
        raise ValueError(f"mean_var.weight={mean_var_weight}: must be ≥ 0")
    if mean_var_weight > 0.0:
        mv_auto = mv_sigma2_t <= 0.0
        logger.info(
            f"mean-variance reg ENABLED (Eq.7): weight={mean_var_weight}, target="
            + (
                "exact one-pass over real latents"
                + (
                    " (full pass)"
                    if mv_calib_batches <= 0
                    else f" (≤{mv_calib_batches} batches)"
                )
                if mv_auto
                else f"fixed μ_t={mv_mu_t}, σ²_t={mv_sigma2_t}"
            )
        )
    logger.info(
        "DM gradient policy: "
        + (
            f"(b) x0-norm, norm_floor={norm_floor} — τ cancels, ≈ magnitude-normalized"
            if dm_x0_norm
            else "(a) τ-damping [default]"
        )
    )
    logger.info(
        "DP-DMD: first-step diversity anchor "
        f"k_anchor={k_anchor}/{teacher_anchor_steps} teacher steps, "
        f"div_weight={div_weight}, detach_after_first={detach_after_first}, "
        f"student N={student_steps} @ flow_shift={flow_shift}, "
        f"teacher_cfg={teacher_cfg}."
    )

    return TurboConfig(
        dit_path=dit_path,
        data_dir=data_dir,
        output_dir=output_dir,
        output_name=output_name,
        log_dir=log_dir,
        save_every=save_every,
        log_interval=log_interval,
        no_log=bool(args.no_log),
        iterations=iterations,
        batch_size=batch_size,
        seed=seed,
        sample_ratio=float(args.sample_ratio),
        single_prompt_idx=args.single_prompt_idx,
        student_rank=student_rank,
        fake_rank=fake_rank,
        student_alpha=student_alpha,
        fake_alpha=fake_alpha,
        attn_mode=attn_mode,
        use_custom_down_autograd=use_custom_down_autograd,
        use_masked_loss=use_masked_loss,
        mask_dir=mask_dir,
        k_anchor=k_anchor,
        teacher_anchor_steps=teacher_anchor_steps,
        div_weight=div_weight,
        detach_after_first=detach_after_first,
        flow_shift=flow_shift,
        student_steps=student_steps,
        teacher_cfg=teacher_cfg,
        dm_x0_norm=dm_x0_norm,
        norm_floor=norm_floor,
        mean_var_weight=mean_var_weight,
        mv_mu_t=mv_mu_t,
        mv_sigma2_t=mv_sigma2_t,
        mv_calib_batches=mv_calib_batches,
        student_lr=student_lr,
        fake_lr=fake_lr,
        fake_steps_per_student_step=fake_steps_per_student_step,
        fake_warmup_steps=fake_warmup_steps,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        t_distribution=t_distribution,
        sigmoid_scale=sigmoid_scale,
        blocks_to_swap=int(args.blocks_to_swap),
        grad_ckpt=bool(args.grad_ckpt),
        torch_compile=bool(args.torch_compile),
    )


def snapshot_toml_text(c: TurboConfig, *, source_config: str | None = None) -> str:
    """Render the fully-resolved turbo config as a provenance TOML snapshot.

    Unlike :func:`tb_config_text` (a TB summary of a hand-picked subset), this
    dumps *every* resolved field — CLI overrides folded in — so the run log dir
    becomes a self-contained record of "this run + the config that produced it".
    It's the turbo analogue of the ``<output_name>.snapshot.toml`` that
    ``train.py`` writes for the LoRA family (the bespoke turbo config never went
    through that path).
    """
    import dataclasses

    import tomlkit

    doc = tomlkit.document()
    doc.add(tomlkit.comment("Anima turbo distillation — resolved config snapshot"))
    if source_config:
        doc.add(tomlkit.comment(f"source config: {source_config}"))
    doc.add(tomlkit.nl())
    for k, v in dataclasses.asdict(c).items():
        if v is None:
            # tomlkit has no null; record the field as unset rather than drop it.
            doc.add(tomlkit.comment(f"{k} = (unset)"))
        else:
            doc[k] = v
    return tomlkit.dumps(doc)


def tb_config_text(c: TurboConfig) -> str:
    """Formatted TensorBoard config summary (same key set as v1)."""
    pairs = {
        "k_anchor": c.k_anchor,
        "teacher_anchor_steps": c.teacher_anchor_steps,
        "div_weight": c.div_weight,
        "detach_after_first": c.detach_after_first,
        "flow_shift": c.flow_shift,
        "student_rank": c.student_rank,
        "fake_rank": c.fake_rank,
        "student_steps": c.student_steps,
        "teacher_cfg": c.teacher_cfg,
        "fake_warmup_steps": c.fake_warmup_steps,
        "student_lr": c.student_lr,
        "fake_lr": c.fake_lr,
        "fake_steps_per_student_step": c.fake_steps_per_student_step,
        "iterations": c.iterations,
        "batch_size": c.batch_size,
        "t_distribution": c.t_distribution,
        "mean_var_weight": c.mean_var_weight,
        "use_masked_loss": c.use_masked_loss,
        "data_dir": c.data_dir,
        "dit_path": c.dit_path,
    }
    return "  \n".join(f"{k}: {v}" for k, v in pairs.items())
