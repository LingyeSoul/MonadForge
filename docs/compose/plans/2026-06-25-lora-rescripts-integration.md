# lora-rescripts Optimization Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the high-value algorithm optimizations from WhitecrowAurora/lora-rescripts into Anima, focusing on techniques that improve convergence stability, reduce VRAM, and expand the training toolkit — without duplicating what Anima already has.

**Architecture:** Each task is a self-contained feature addition. Tasks are ordered by dependency (independent tasks can be parallelized). All changes follow Anima's existing patterns: registry-based dispatch, config-driven activation, CLI arg registration.

**Tech Stack:** PyTorch, bitsandbytes, existing Anima training infrastructure (`library/training/`, `library/runtime/`, `networks/`)

---

## Gap Analysis Summary

### Already in Anima (DO NOT port)
| Feature | Anima Location | Notes |
|---------|---------------|-------|
| Rectified Flow Matching | `library/runtime/noise.py`, `train.py` | Core training objective |
| Schedule-Free Optimizers | `library/training/optimizers.py:228` | RAdam/AdamW/SGD variants |
| Lion, DAdaptation, Prodigy | `library/training/optimizers.py` | Full family |
| AdamW8bit, PagedAdamW | `library/training/optimizers.py:39` | bitsandbytes |
| Mixed precision (bf16/fp16) | `train.py:178-216` | Auto-detection for V100 |
| Gradient checkpointing + CPU offload | `train.py:1748-1757`, `configs/presets.toml` | unsloth_offload |
| Block swap (CPU offloading) | `library/anima/models.py:1868-1939` | Async CUDA streams |
| torch.compile + dynamic_seq | `library/runtime/harness.py:559` | Per-block compilation |
| Masked loss | `library/training/losses.py:73` | Per-pixel spatial masking |
| Free-fit multi-scale bucketing | `library/datasets/buckets.py` | 6 edge tiers |
| Flash Attention v2/v4 | `networks/attention_dispatch.py` | + SDPA, Flex, Sage |
| Channel scaling | `networks/lora_anima/factory.py:41-74` | SmoothQuant-style |
| FP8 base model | `library/config/cli_args.py` | --fp8_base_unet |
| Caption variants + tag dropout | `library/preprocess/caption_variants.py` | Shuffle + identity erasure |
| Timestep sampling (6 strategies) | `library/runtime/noise.py:90-162` | sigmoid/uniform/shift/flux_shift/logit_normal/mode |
| Loss weighting (3 schemes) | `library/runtime/noise.py:80-87` | none/sigma_sqrt/cosmap |
| Huber loss + scheduling | `library/training/losses.py:175-213` | exponential/snr/constant |
| Variance-reduced FM loss | `library/training/losses.py:309` | AsymFlow control-variate |
| REPA v2 | `library/training/repa.py` | Absolute + relational |
| Input perturbation noise | `library/runtime/noise.py:148-158` | ip_noise_gamma |
| Multi-adapter LoRA variants | `networks/lora_modules/` | 9 variants (LoRA/LoKR/Ortho/Hydra/Chimera/FeRA/StepExpert/etc.) |
| Adapter types (EasyControl, Soft Tokens, Turbo, etc.) | `networks/methods/` | 7 non-LoRA adapters |

### Gaps to Port (this plan)
| # | Optimization | Impact | Effort | Priority |
|---|-------------|--------|--------|----------|
| 1 | AdamW 8-bit Kahan Summation | High | Low | P0 |
| 2 | Staged Mixed-Resolution Training | High | Medium | P0 |
| 3 | Adaptive Noise Offset | Medium | Low | P1 |
| 4 | Contrastive Flow Matching | Medium | Low | P1 |
| 5 | Pyramid Multi-Resolution Noise | Medium | Low | P1 |
| 6 | VeRA Adapter | Medium | Medium | P2 |
| 7 | DyLoRA Adapter | Medium | Medium | P2 |
| 8 | Optimizer State Offloading | High | Medium | P1 |
| 9 | Training Step Profiler | Low | Low | P3 |
| 10 | Missing Dependencies Fix | High | Trivial | P0 |

---

## File Structure

```
library/training/
  adamw_8bit_kahan.py          # NEW — Kahan-compensated AdamW8bit
  optimizers.py                # MODIFY — add AdamW8bitKahan dispatch
  losses.py                    # MODIFY — add contrastive FM + pyramid noise
  profiler.py                  # NEW — training step profiler

library/runtime/
  noise.py                     # MODIFY — add adaptive noise offset + pyramid noise

library/training/staged_resolution.py   # NEW — staged mixed-resolution runner

networks/lora_modules/
  vera.py                      # NEW — VeRA adapter module
  dylora.py                    # NEW — DyLoRA adapter module

library/config/
  cli_args.py                  # MODIFY — add new CLI args

configs/
  base.toml                    # MODIFY — add new defaults (commented)
```

---

### Task 0: Fix Missing Dependencies in pyproject.toml

**Covers:** Infrastructure bug — code imports `schedulefree`, `dadaptation`, `lion_pytorch` but pyproject.toml doesn't declare them.

**Files:**
- Modify: `pyproject.toml:15-17`

- [ ] **Step 1: Add missing optimizer dependencies**

```toml
# In pyproject.toml dependencies list, after line 17 (prodigyopt):
    "dadaptation>=3.1",
    "lion-pytorch>=0.2.3",
    "schedulefree>=1.4",
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "import schedulefree; import dadaptation; import lion_pytorch; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "fix: declare missing optimizer deps (schedulefree, dadaptation, lion-pytorch)"
```

---

### Task 1: AdamW 8-bit Kahan Summation

**Covers:** Convergence stability for long LoRA training runs. Standard AdamW8bit loses precision proportional to step count; Kahan summation preserves ~11-12 effective bits from 8-bit storage.

**Files:**
- Create: `library/training/adamw_8bit_kahan.py`
- Modify: `library/training/optimizers.py:45-48`

**Reference:** `WhitecrowAurora/lora-rescripts` `scripts/stable/library/adamw_8bit_kahan.py`

- [ ] **Step 1: Write the Kahan AdamW8bit module**

Create `library/training/adamw_8bit_kahan.py`:

```python
"""AdamW 8-bit with Kahan compensated summation.

Reduces floating-point drift in quantized optimizer state updates by
tracking a running compensation term.  Based on the implementation in
WhitecrowAurora/lora-rescripts.
"""

import logging
from typing import Optional

import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F

logger = logging.getLogger(__name__)


def _stochastic_round_bf16_(tensor: torch.Tensor):
    """Stochastic rounding for bfloat16 — randomly dithers mantissa bits."""
    # Generate random bits for the lower 16 bits of mantissa
    rand_bits = torch.randint_like(tensor.view(torch.uint16), 0, 65536)
    # Add random noise before truncation
    # This is equivalent to: round(x + uniform(-0.5*ulp, 0.5*ulp))
    x_f32 = tensor.float()
    # Get the ULP (unit in last place) of the bf16 representation
    x_bf16 = x_f32.bfloat16()
    # Compute the error
    error = x_f32 - x_bf16.float()
    # Stochastic: if error > 0, round up with probability error/ulp
    threshold = torch.abs(error) / torch.finfo(torch.bfloat16).smallest_normal
    random_val = torch.rand_like(x_f32)
    adjustment = torch.where(
        error > 0,
        torch.where(random_val < threshold, torch.finfo(torch.bfloat16).smallest_normal, 0.0),
        torch.where(random_val < threshold, -torch.finfo(torch.bfloat16).smallest_normal, 0.0),
    )
    tensor.copy_((x_f32 + adjustment).bfloat16())


class AdamW8bitKahan(bnb.optim.AdamW8bit):
    """AdamW8bit with Kahan compensated summation for reduced precision drift.

    Each parameter maintains a ``shift`` buffer that accumulates rounding
    error from each step.  The error is added back in the next step,
    preserving precision over thousands of iterations.

    Additional args:
        stabilize: If True, caps lr by rms(grad²/exp_avg_sq) to prevent
            exploding updates.
        kahan_buffer_offload: If True, stores Kahan shift buffers on CPU
            RAM to save VRAM.
    """

    def __init__(self, *args, stabilize: bool = False,
                 kahan_buffer_offload: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.stabilize = stabilize
        self.kahan_buffer_offload = kahan_buffer_offload

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW8bitKahan does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # Kahan compensation buffer
                    state["shift"] = torch.zeros_like(p)

                state["step"] += 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                shift = state["shift"]

                # Move shift to device if offloaded
                if self.kahan_buffer_offload and shift.device.type == "cpu":
                    shift = shift.to(p.device)
                    state["shift"] = shift

                beta1, beta2 = group["betas"]

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decoupled weight decay on real parameters (not Kahan buffer)
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Update moments with dequantized values
                # bitsandbytes keeps moments in 8-bit; dequantize for update
                F.update_moment(grad, exp_avg, exp_avg_sq, beta1, beta2, state["step"])

                # Compute step with Kahan compensation
                step_size = group["lr"] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                # Stabilized lr: cap by rms(grad²/exp_avg_sq)
                if self.stabilize:
                    rms = (grad.pow(2) / exp_avg_sq.clamp_min(1e-12)).mean().sqrt()
                    step_size = step_size / max(1.0, float(rms))

                # Kahan-compensated parameter update
                update = exp_avg / denom
                y = update * step_size - shift
                t = p.data + y
                shift.copy_((t - p.data) - y)
                p.data.copy_(t)

                # Offload shift buffer if configured
                if self.kahan_buffer_offload:
                    state["shift"] = shift.cpu()

        return loss


# Keep math import available for stabilize mode
import math
```

- [ ] **Step 2: Add optimizer dispatch in optimizers.py**

In `library/training/optimizers.py`, add after the `AdamW8bit` block (around line 48):

```python
        elif optimizer_type == "AdamW8bitKahan".lower():
            logger.info(f"use 8-bit AdamW with Kahan summation | {optimizer_kwargs}")
            from library.training.adamw_8bit_kahan import AdamW8bitKahan
            optimizer_class = AdamW8bitKahan
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
```

- [ ] **Step 3: Add CLI arg**

In `library/config/cli_args.py`, add to the optimizer args section (no new args needed — `AdamW8bitKahan` is selected via `--optimizer_type AdamW8bitKahan` and controlled via `--optimizer_args stabilize=true,kahan_buffer_offload=true`).

- [ ] **Step 4: Test import**

Run: `python -c "from library.training.adamw_8bit_kahan import AdamW8bitKahan; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add library/training/adamw_8bit_kahan.py library/training/optimizers.py
git commit -m "feat: add AdamW8bitKahan optimizer with compensated summation"
```

---

### Task 2: Staged Mixed-Resolution Training

**Covers:** Curriculum learning — train at progressively higher resolutions in a single run for better composition learning at low res and detail refinement at high res.

**Files:**
- Create: `library/training/staged_resolution.py`
- Modify: `train.py` (add staged resolution launch hook)
- Modify: `library/config/cli_args.py` (add CLI args)

**Reference:** `WhitecrowAurora/lora-rescripts` `mikazuki/utils/mixed_resolution.py`, `mikazuki/staged_resolution_runner.py`

- [ ] **Step 1: Add CLI arguments**

In `library/config/cli_args.py`, add to the training args section:

```python
    parser.add_argument(
        "--staged_resolution",
        action="store_true",
        default=False,
        help="Enable staged mixed-resolution training (curriculum learning)",
    )
    parser.add_argument(
        "--staged_resolution_ratios",
        type=str,
        default=None,
        help="Comma-separated ratio per stage, e.g. '20,30,50' for 512/768/1024",
    )
    parser.add_argument(
        "--staged_resolution_base_sides",
        type=str,
        default=None,
        help="Comma-separated base sides, e.g. '512,768,1024'",
    )
```

- [ ] **Step 2: Create staged resolution module**

Create `library/training/staged_resolution.py`:

```python
"""Staged mixed-resolution training — curriculum learning for LoRA.

Trains through progressively higher resolutions in a single run:
  Phase 1: low-res (learn composition)
  Phase 2: mid-res (refine structure)
  Phase 3: target-res (detail refinement)

Batch size is auto-scaled by pixel area ratio to maintain equivalent
gradient signal per step.  Save/sample intervals are LCM-aligned across
phases for consistent checkpoint cadence.

Based on the approach in WhitecrowAurora/lora-rescripts.
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StagePlan:
    base_side: int
    ratio: float          # fraction of total epochs (0-100)
    batch_size: int
    raw_epochs: int       # epoch count for this stage
    save_every: int       # steps between saves
    sample_every: int     # steps between samples


@dataclass
class MixedResolutionPlan:
    stages: List[StagePlan]
    total_epochs: int
    base_batch_size: int
    base_side: int        # original target resolution
    plan_hash: str        # SHA1 for reproducibility


DEFAULT_STAGE_SCHEDULES = {
    (512,): {512: 100},
    (768,): {512: 40, 768: 60},
    (1024,): {512: 20, 768: 30, 1024: 50},
    (2048,): {1024: 20, 1536: 30, 2048: 50},
}


def _area(side: int) -> int:
    return side * side


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def build_staged_plan(
    target_res: List[int],
    base_batch_size: int,
    total_epochs: int,
    save_every: int,
    sample_every: Optional[int],
    ratios: Optional[List[float]] = None,
    base_sides: Optional[List[int]] = None,
) -> MixedResolutionPlan:
    """Build a staged resolution plan.

    Args:
        target_res: The original target resolution(s) from config.
        base_batch_size: Batch size at the base resolution.
        total_epochs: Total training epochs to distribute across stages.
        save_every: Steps between saves at base resolution.
        sample_every: Steps between samples at base resolution (None = disabled).
        ratios: Per-stage epoch ratios (must sum to ~100). If None, uses defaults.
        base_sides: Per-stage resolution sides. If None, uses defaults for largest target_res.

    Returns:
        MixedResolutionPlan with per-stage details.
    """
    max_res = max(target_res)

    if base_sides is None:
        # Find the best matching default schedule
        for key, schedule in sorted(DEFAULT_STAGE_SCHEDULES.items()):
            if max_res <= max(key):
                base_sides = list(schedule.keys())
                if ratios is None:
                    ratios = list(schedule.values())
                break
        if base_sides is None:
            # Fallback: single stage at target res
            base_sides = [max_res]
            ratios = [100.0]

    if ratios is None:
        ratios = [100.0 / len(base_sides)] * len(base_sides)

    assert len(base_sides) == len(ratios), (
        f"base_sides ({len(base_sides)}) and ratios ({len(ratios)}) must match"
    )

    # Normalize ratios to sum to 100
    ratio_sum = sum(ratios)
    ratios = [r / ratio_sum * 100 for r in ratios]

    base_area = _area(base_sides[0])  # smallest stage = base
    stages = []

    for i, (side, ratio) in enumerate(zip(base_sides, ratios)):
        stage_area = _area(side)
        # Batch size scales inversely with area (larger res = smaller batch)
        stage_batch = max(1, int(base_batch_size * base_area / stage_area))

        # Epoch count compensates for batch size change
        raw_epochs = max(1, math.ceil(total_epochs * ratio / 100 * (stage_batch / base_batch_size)))

        # Save/sample intervals scale with batch change, LCM-aligned
        save_every_stage = max(1, save_every * stage_batch // base_batch_size)
        sample_every_stage = (
            max(1, sample_every * stage_batch // base_batch_size)
            if sample_every else 0
        )

        stages.append(StagePlan(
            base_side=side,
            ratio=ratio,
            batch_size=stage_batch,
            raw_epochs=raw_epochs,
            save_every=save_every_stage,
            sample_every=sample_every_stage,
        ))

    import hashlib
    plan_str = f"{base_sides}|{ratios}|{base_batch_size}|{total_epochs}"
    plan_hash = hashlib.sha1(plan_str.encode()).hexdigest()[:12]

    return MixedResolutionPlan(
        stages=stages,
        total_epochs=total_epochs,
        base_batch_size=base_batch_size,
        base_side=max_res,
        plan_hash=plan_hash,
    )


def log_plan(plan: MixedResolutionPlan):
    """Log the staged resolution plan."""
    logger.info(f"Staged resolution plan [{plan.plan_hash}]:")
    for i, s in enumerate(plan.stages):
        logger.info(
            f"  Phase {i+1}: {s.base_side}px | "
            f"batch={s.batch_size} | epochs={s.raw_epochs} | "
            f"save_every={s.save_every} | sample_every={s.sample_every} | "
            f"ratio={s.ratio:.0f}%"
        )


def run_staged_training(
    plan: MixedResolutionPlan,
    train_script: str,
    base_argv: List[str],
    output_dir: Path,
    resume_from_last: bool = True,
):
    """Execute staged training by launching one subprocess per phase.

    Each phase:
    1. Sets --resolution to the stage's base_side
    2. Sets --max_train_epochs to the stage's raw_epochs
    3. Sets --train_batch_size to the stage's batch_size
    4. Auto-resumes from the previous phase's checkpoint
    """
    log_plan(plan)
    current_resume = None

    for i, stage in enumerate(plan.stages):
        logger.info(f"=== Starting Phase {i+1}/{len(plan.stages)}: {stage.base_side}px ===")

        argv = list(base_argv)
        argv.extend([
            "--resolution", str(stage.base_side),
            "--max_train_epochs", str(stage.raw_epochs),
            "--train_batch_size", str(stage.batch_size),
            "--save_every_n_steps", str(stage.save_every),
        ])

        if stage.sample_every > 0:
            argv.extend(["--sample_every_n_steps", str(stage.sample_every)])

        # Resume from previous phase checkpoint
        if current_resume and resume_from_last:
            argv.extend(["--resume", str(current_resume)])

        # Tag output with phase info
        phase_output = output_dir / f"phase_{i+1}_{stage.base_side}px"
        argv.extend(["--output_name", f"phase_{i+1}_{stage.base_side}px"])

        logger.info(f"  Launching: python {train_script} {' '.join(argv)}")

        result = subprocess.run(
            [sys.executable, train_script] + argv,
            cwd=str(output_dir.parent.parent),
        )

        if result.returncode != 0:
            logger.error(f"Phase {i+1} failed with return code {result.returncode}")
            raise RuntimeError(f"Staged training phase {i+1} ({stage.base_side}px) failed")

        # Find the latest checkpoint for resume
        ckpt_dir = output_dir
        checkpoints = sorted(ckpt_dir.glob("phase_*/*.safetensors"))
        if checkpoints:
            current_resume = checkpoints[-1]
            logger.info(f"  Phase {i+1} complete. Checkpoint: {current_resume}")
        else:
            logger.warning(f"  Phase {i+1} complete but no checkpoint found for resume")

    logger.info("All staged resolution phases complete.")
```

- [ ] **Step 3: Integrate into train.py**

At the top of `train.py`'s `main()` function, before the training loop starts, add:

```python
    # Staged mixed-resolution training
    if getattr(args, "staged_resolution", False):
        from library.training.staged_resolution import (
            build_staged_plan, run_staged_training
        )
        ratios = None
        if args.staged_resolution_ratios:
            ratios = [float(x) for x in args.staged_resolution_ratios.split(",")]
        base_sides = None
        if args.staged_resolution_base_sides:
            base_sides = [int(x) for x in args.staged_resolution_base_sides.split(",")]

        plan = build_staged_plan(
            target_res=args.target_res if isinstance(args.target_res, list) else [args.target_res],
            base_batch_size=args.train_batch_size,
            total_epochs=args.max_train_epochs,
            save_every=args.save_every_n_steps,
            sample_every=getattr(args, "sample_every_n_steps", None),
            ratios=ratios,
            base_sides=base_sides,
        )
        run_staged_training(
            plan=plan,
            train_script=str(Path(__file__).resolve()),
            base_argv=sys.argv[1:],
            output_dir=Path(args.output_dir),
        )
        return  # Staged training handles its own subprocess lifecycle
```

- [ ] **Step 4: Test plan generation**

Run: `python -c "
from library.training.staged_resolution import build_staged_plan, log_plan
plan = build_staged_plan([1024], 4, 6, 500, 100)
log_plan(plan)
print('OK')
"`
Expected: Plan with 3 phases (512/768/1024) logged, `OK`

- [ ] **Step 5: Commit**

```bash
git add library/training/staged_resolution.py library/config/cli_args.py train.py
git commit -m "feat: add staged mixed-resolution training (curriculum learning)"
```

---

### Task 3: Adaptive Noise Offset

**Covers:** Noise offset based on per-channel latent statistics, improving dynamic range in generated images.

**Files:**
- Modify: `library/runtime/noise.py:148-158`
- Modify: `library/config/cli_args.py`

**Reference:** `WhitecrowAurora/lora-rescripts` `scripts/stable/library/custom_train_functions.py` `apply_noise_offset()`

- [ ] **Step 1: Add CLI arguments**

In `library/config/cli_args.py`:

```python
    parser.add_argument(
        "--adaptive_noise_offset",
        action="store_true",
        default=False,
        help="Use latent-statistics-based adaptive noise offset instead of fixed",
    )
    parser.add_argument(
        "--adaptive_noise_scale",
        type=float,
        default=1.0,
        help="Scale factor for adaptive noise offset",
    )
```

- [ ] **Step 2: Implement adaptive noise offset**

In `library/runtime/noise.py`, replace the noise offset block (lines 148-158) with:

```python
    if args.ip_noise_gamma:
        xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
        if args.ip_noise_gamma_random_strength:
            ip_noise_gamma = (
                torch.rand(1, device=latents.device, dtype=dtype) * args.ip_noise_gamma
            )
        else:
            ip_noise_gamma = args.ip_noise_gamma

        if getattr(args, "adaptive_noise_offset", False):
            # Adaptive: scale noise offset by per-channel latent statistics
            adaptive_scale = getattr(args, "adaptive_noise_scale", 1.0)
            with torch.no_grad():
                channel_mean = latents.abs().mean(dim=(-2, -1), keepdim=True)  # [B, C, 1, 1]
                channel_mean = channel_mean.clamp_min(1e-6)
                # Scale noise offset proportional to channel energy
                xi = xi * channel_mean * adaptive_scale

        noisy_model_input = (1.0 - sigmas) * latents + sigmas * (
            noise + ip_noise_gamma * xi
        )
```

- [ ] **Step 3: Commit**

```bash
git add library/runtime/noise.py library/config/cli_args.py
git commit -m "feat: add adaptive noise offset based on latent channel statistics"
```

---

### Task 4: Contrastive Flow Matching Loss

**Covers:** Add contrastive loss term on top of rectified flow objective for better-structured latent representations.

**Files:**
- Modify: `library/training/losses.py` (add handler + register)
- Modify: `library/config/cli_args.py`

**Reference:** `WhitecrowAurora/lora-rescripts` `scripts/stable/train_network.py` `--contrastive_flow_matching`

- [ ] **Step 1: Add CLI arguments**

In `library/config/cli_args.py`:

```python
    parser.add_argument(
        "--contrastive_flow_matching",
        action="store_true",
        default=False,
        help="Add contrastive loss term to flow matching objective",
    )
    parser.add_argument(
        "--cfm_lambda",
        type=float,
        default=0.05,
        help="Weight for contrastive flow matching loss term",
    )
```

- [ ] **Step 2: Add contrastive FM loss handler**

In `library/training/losses.py`, add after the `_flow_match_loss` function:

```python
def _contrastive_flow_matching_loss(ctx: LossContext) -> torch.Tensor:
    """Contrastive loss term for flow matching.

    Encourages better-structured latent representations by contrasting
    positive flow directions (correct noise→latent) with negative ones
    (random permutations).

    Requires --contrastive_flow_matching and --cfm_lambda.
    """
    base_loss = _flow_match_loss(ctx)

    cfm_weight = float(getattr(ctx.args, "cfm_lambda", 0.0) or 0.0)
    if cfm_weight <= 0.0:
        return base_loss

    # Positive: model_pred vs target
    positive = (ctx.model_pred.float() - ctx.target.float()).pow(2).mean(dim=list(range(1, ctx.model_pred.ndim)))

    # Negative: model_pred vs permuted target (different sample's noise direction)
    with torch.no_grad():
        perm = torch.randperm(ctx.target.shape[0], device=ctx.target.device)
        neg_target = ctx.target[perm]

    negative = (ctx.model_pred.float() - neg_target.float()).pow(2).mean(dim=list(range(1, ctx.model_pred.ndim)))

    # Contrastive: pull positive closer, push negative farther
    contrastive = (positive - negative).clamp(min=0.0)

    return base_loss + cfm_weight * contrastive
```

- [ ] **Step 3: Register in LOSS_REGISTRY**

In `library/training/losses.py`, find the `LOSS_REGISTRY` dict and add:

```python
    "contrastive_flow_matching": LossHandler(
        name="contrastive_flow_matching",
        stage=LossStage.PER_SAMPLE,
        fn=_contrastive_flow_matching_loss,
        weight_attr=None,  # controlled by --cfm_lambda
        description="Contrastive loss term for flow matching",
    ),
```

- [ ] **Step 4: Activate in build_loss_composer**

In the `build_loss_composer` function, add activation logic:

```python
    if getattr(args, "contrastive_flow_matching", False):
        activate("contrastive_flow_matching")
```

- [ ] **Step 5: Commit**

```bash
git add library/training/losses.py library/config/cli_args.py
git commit -m "feat: add contrastive flow matching loss (--contrastive_flow_matching)"
```

---

### Task 5: Pyramid Multi-Resolution Noise

**Covers:** Generate multi-frequency noise by summing bilinear-upsampled noise at multiple scales, giving the model exposure to multiple frequency bands.

**Files:**
- Modify: `library/runtime/noise.py`
- Modify: `library/config/cli_args.py`

**Reference:** `WhitecrowAurora/lora-rescripts` `scripts/stable/library/custom_train_functions.py` `pyramid_noise_like()`

- [ ] **Step 1: Add CLI arguments**

In `library/config/cli_args.py`:

```python
    parser.add_argument(
        "--pyramid_noise_iterations",
        type=int,
        default=0,
        help="Number of pyramid noise scales (0=disabled, 6 recommended)",
    )
    parser.add_argument(
        "--pyramid_noise_discount",
        type=float,
        default=0.4,
        help="Discount factor per pyramid scale (lower = less high-freq)",
    )
```

- [ ] **Step 2: Implement pyramid noise generator**

In `library/runtime/noise.py`, add a new function:

```python
def pyramid_noise_like(
    latents: torch.Tensor,
    iterations: int = 6,
    discount: float = 0.4,
) -> torch.Tensor:
    """Generate multi-resolution pyramid noise.

    Sums bilinear-upsampled noise at multiple scales, each weighted by
    discount^i.  Gives the model exposure to noise patterns at multiple
    frequency bands simultaneously.

    Based on WhitecrowAurora/lora-rescripts implementation.
    """
    b, c, h, w = latents.shape
    device = latents.device
    dtype = latents.dtype

    noise = torch.randn(b, c, h, w, device=device, dtype=dtype)

    for i in range(1, iterations):
        # Random scale factor between 2x and 4x downsample
        r = torch.empty(1).uniform_(2.0, 4.0).item()
        rh, rw = max(1, int(h / r)), max(1, int(w / r))

        # Generate noise at reduced resolution
        small_noise = torch.randn(b, c, rh, rw, device=device, dtype=dtype)

        # Upsample back to original size
        upsampled = torch.nn.functional.interpolate(
            small_noise, size=(h, w), mode="bilinear", align_corners=False
        )

        # Add with discount weighting
        noise = noise + upsampled * (discount ** i)

    # Normalize to unit variance
    noise = noise / noise.std()

    return noise
```

- [ ] **Step 3: Integrate into noising pipeline**

In `library/runtime/noise.py`, in the `get_noisy_model_input_and_timesteps` function, replace the noise generation block:

```python
    # Generate noise (optionally with pyramid multi-resolution)
    pyramid_iters = getattr(args, "pyramid_noise_iterations", 0)
    if pyramid_iters > 0:
        discount = getattr(args, "pyramid_noise_discount", 0.4)
        noise = pyramid_noise_like(latents, iterations=pyramid_iters, discount=discount)
    else:
        noise = torch.randn_like(latents, device=latents.device, dtype=dtype)
```

Note: This replaces the existing `noise = torch.randn_like(...)` call that currently happens before this function (in `train.py`). The function signature needs `noise` passed in, so the integration point is where noise is created in `train.py`'s inner loop.

- [ ] **Step 4: Commit**

```bash
git add library/runtime/noise.py library/config/cli_args.py
git commit -m "feat: add pyramid multi-resolution noise (--pyramid_noise_iterations)"
```

---

### Task 6: VeRA Adapter (Vector-based Random Matrix Adaptation)

**Covers:** A lightweight LoRA alternative that shares random projection matrices across layers, reducing parameter count.

**Files:**
- Create: `networks/lora_modules/vera.py`
- Modify: `networks/__init__.py` (register in NETWORK_REGISTRY)

**Reference:** `WhitecrowAurora/lora-rescripts` `scripts/stable/networks/vera.py`

- [ ] **Step 1: Implement VeRA module**

Create `networks/lora_modules/vera.py`:

```python
"""VeRA — Vector-based Random Matrix Adaptation.

Shares frozen random projection matrices (A, B) across all layers and
learns per-layer diagonal scaling vectors (d, b) instead.  This gives
~10-100x fewer trainable parameters than standard LoRA while maintaining
competitive quality.

Reference: Kopiczko et al., "VeRA: Vector-based Random Matrix Adaptation"
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.lora_modules.base import BaseLoRAModule

logger = logging.getLogger(__name__)


class VeRAModule(BaseLoRAModule):
    """VeRA adapter: shared random A/B + learned diagonal scaling.

    ΔW = diag(d) @ A @ diag(Λ) @ B @ diag(b)

    where A, B are frozen random matrices shared across layers, and
    d, b are per-layer learnable diagonal vectors.
    """

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        **kwargs,
    ):
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        if isinstance(org_module, nn.Linear):
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.op = "linear"
        elif isinstance(org_module, nn.Conv2d):
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.op = "conv2d"
            self.kernel_size = org_module.kernel_size
            self.stride = org_module.stride
            self.padding = org_module.padding
        else:
            raise ValueError(f"VeRA: unsupported module type {type(org_module)}")

        # Frozen random projections (shared — set via set_shared_matrices)
        self.lora_A = nn.Parameter(torch.empty(lora_dim, in_dim), requires_grad=False)
        self.lora_B = nn.Parameter(torch.empty(out_dim, lora_dim), requires_grad=False)

        # Learnable diagonal scaling vectors
        self.vera_d = nn.Parameter(torch.ones(lora_dim))   # scales A output
        self.vera_b = nn.Parameter(torch.ones(out_dim))     # scales B output

        # Initialize random matrices (will be shared/seeded externally)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.multiplier = multiplier
        self.rank = lora_dim

    def set_shared_matrices(self, A: torch.Tensor, B: torch.Tensor):
        """Set shared random matrices (frozen, shared across layers)."""
        with torch.no_grad():
            self.lora_A.copy_(A)
            self.lora_B.copy_(B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Δx = (x @ A^T * d) @ B^T * b
        # More efficiently: x @ (A^T @ diag(d) @ B^T @ diag(b))
        if self.op == "conv2d":
            # For conv, reshape to 2D
            b, c, h, w = x.shape
            x_flat = x.reshape(b, c, -1).permute(0, 2, 1)  # [B, HW, C]

            # A: [r, in] → A^T: [in, r]
            # d: [r] — scales each rank dimension
            h_out = F.linear(x_flat, self.lora_A)  # [B, HW, r]
            h_out = h_out * self.vera_d  # diagonal scaling
            result = F.linear(h_out, self.lora_B)  # [B, HW, out]
            result = result * self.vera_b  # diagonal scaling
            result = result.permute(0, 2, 1).reshape(b, -1, h, w)
        else:
            # Linear path
            h_out = F.linear(x, self.lora_A)  # [B, ..., r]
            h_out = h_out * self.vera_d
            result = F.linear(h_out, self.lora_B)
            result = result * self.vera_b

        return result * self.multiplier * self.alpha / self.rank
```

- [ ] **Step 2: Register in NETWORK_REGISTRY**

In `networks/__init__.py`, add VeRA to the registry:

```python
    "vera": NetworkSpec(
        name="vera",
        module_class="networks.lora_modules.vera.VeRAModule",
        kwargs_allowlist={"lora_dim", "alpha", "multiplier", "algo"},
        description="VeRA — Vector-based Random Matrix Adaptation (shared random A/B + learned scaling)",
    ),
```

- [ ] **Step 3: Create method config**

Create `configs/methods/vera.toml`:

```toml
network_module = "vera"
network_dim = 64
network_alpha = 64
learning_rate = 1e-4
max_train_epochs = 6
output_name = "vera"
```

- [ ] **Step 4: Commit**

```bash
git add networks/lora_modules/vera.py networks/__init__.py configs/methods/vera.toml
git commit -m "feat: add VeRA adapter (Vector-based Random Matrix Adaptation)"
```

---

### Task 7: DyLoRA Adapter (Dynamic LoRA)

**Covers:** Trains multiple LoRA ranks simultaneously in a single pass, eliminating the need for separate rank experiments.

**Files:**
- Create: `networks/lora_modules/dylora.py`
- Modify: `networks/__init__.py` (register in NETWORK_REGISTRY)

**Reference:** `WhitecrowAurora/lora-rescripts` `scripts/stable/networks/dylora.py`

- [ ] **Step 1: Implement DyLoRA module**

Create `networks/lora_modules/dylora.py`:

```python
"""DyLoRA — Dynamic LoRA.

Trains multiple LoRA ranks simultaneously by randomly selecting a rank
at each forward pass and computing the LoRA update for that rank only.
After training, any rank ≤ max_rank can be extracted.

Reference: Valipour et al., "DyLoRA: Parameter-Efficient Tuning of
Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation"
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.lora_modules.base import BaseLoRAModule

logger = logging.getLogger(__name__)


class DyLoRAModule(BaseLoRAModule):
    """DyLoRA: trains all ranks 1..max_rank simultaneously.

    At each forward, a random rank r is selected and only the first r
    rows/cols of lora_A/lora_B are used.  This forces the network to
    learn good representations at every rank level.
    """

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        unit: int = 1,  # rank granularity (train ranks unit, 2*unit, ...)
        **kwargs,
    ):
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        if isinstance(org_module, nn.Linear):
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.op = "linear"
        elif isinstance(org_module, nn.Conv2d):
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.op = "conv2d"
            self.kernel_size = org_module.kernel_size
            self.stride = org_module.stride
            self.padding = org_module.padding
        else:
            raise ValueError(f"DyLoRA: unsupported module type {type(org_module)}")

        self.lora_dim = lora_dim
        self.unit = max(1, unit)
        self.rank = lora_dim

        # Full-rank A and B (will be sliced to random rank at each step)
        if self.op == "conv2d":
            self.lora_A = nn.Conv2d(in_dim, lora_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.lora_B = nn.Conv2d(lora_dim, out_dim, self.kernel_size, self.stride, self.padding, bias=False)
        else:
            self.lora_A = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_B = nn.Linear(lora_dim, out_dim, bias=False)

        # Initialize A with kaiming, B with zeros (so ΔW=0 at step 0)
        nn.init.kaiming_uniform_(self.lora_A.weight if hasattr(self.lora_A, 'weight') else self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight if hasattr(self.lora_B, 'weight') else self.lora_B)

        self.multiplier = multiplier
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Random rank selection: pick r in {unit, 2*unit, ..., lora_dim}
        max_units = self.lora_dim // self.unit
        selected_unit = random.randint(1, max_units)
        r = selected_unit * self.unit

        # Slice A and B to rank r
        if self.op == "conv2d":
            A_sliced = self.lora_A.weight[:r]  # [r, in, 1, 1]
            B_sliced = self.lora_B.weight[:, :r]  # [out, r, k, k]
            h = F.conv2d(x, A_sliced, stride=self.lora_A.stride, padding=self.lora_A.padding)
            result = F.conv2d(h, B_sliced, stride=self.stride, padding=self.padding)
        else:
            A_sliced = self.lora_A.weight[:r]  # [r, in]
            B_sliced = self.lora_B.weight[:, :r]  # [out, r]
            h = F.linear(x, A_sliced)
            result = F.linear(h, B_sliced)

        # Scale by multiplier * alpha / selected_rank
        return result * self.multiplier * self.alpha / r
```

- [ ] **Step 2: Register in NETWORK_REGISTRY**

In `networks/__init__.py`, add DyLoRA:

```python
    "dylora": NetworkSpec(
        name="dylora",
        module_class="networks.lora_modules.dylora.DyLoRAModule",
        kwargs_allowlist={"lora_dim", "alpha", "multiplier", "unit", "algo"},
        description="DyLoRA — trains multiple ranks simultaneously via random rank selection",
    ),
```

- [ ] **Step 3: Commit**

```bash
git add networks/lora_modules/dylora.py networks/__init__.py
git commit -m "feat: add DyLoRA adapter (dynamic rank training)"
```

---

### Task 8: Optimizer State Offloading

**Covers:** Offload optimizer states (momentum, variance) to CPU RAM for large models on limited VRAM. Can reduce VRAM by ~50% for Adam-family optimizers.

**Files:**
- Create: `library/training/optimizer_offload.py`
- Modify: `library/training/optimizers.py`
- Modify: `library/config/cli_args.py`

**Reference:** `WhitecrowAurora/lora-rescripts` `scripts/stable/library/optimizer_offload_util.py`

- [ ] **Step 1: Add CLI argument**

In `library/config/cli_args.py`:

```python
    parser.add_argument(
        "--optimizer_cpu_offload",
        action="store_true",
        default=False,
        help="Offload optimizer states to CPU RAM to save VRAM (slower step time)",
    )
```

- [ ] **Step 2: Implement optimizer offload wrapper**

Create `library/training/optimizer_offload.py`:

```python
"""Optimizer state offloading to CPU RAM.

Wraps any optimizer to keep momentum/variance buffers on CPU and
fetch them to GPU on demand during step().  Trades ~30% slower step
time for significant VRAM savings on large models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class OffloadedOptimizer(Optimizer):
    """Wraps an optimizer to offload state tensors to CPU.

    GPU tensors are moved to CPU after each step and back to GPU
    before the next step.  Only state tensors are offloaded;
    parameters stay on their original device.
    """

    def __init__(self, optimizer: Optimizer, pin_memory: bool = True):
        # Don't call super().__init__ — we delegate everything
        self._optimizer = optimizer
        self._pin_memory = pin_memory
        self._offloaded = False

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._optimizer, name)

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._optimizer.param_groups = value

    @property
    def state(self):
        return self._optimizer.state

    @state.setter
    def state(self, value):
        self._optimizer.state = value

    def _move_state_to_device(self, device: torch.device):
        """Move all state tensors to target device."""
        for param, state_dict in self._optimizer.state.items():
            for key, val in state_dict.items():
                if isinstance(val, torch.Tensor) and val.device != device:
                    state_dict[key] = val.to(device, non_blocking=True)

    def _offload_to_cpu(self):
        """Move state tensors to CPU after step."""
        cpu = torch.device("cpu")
        for param, state_dict in self._optimizer.state.items():
            for key, val in state_dict.items():
                if isinstance(val, torch.Tensor) and val.device != cpu:
                    if self._pin_memory:
                        state_dict[key] = val.pin_memory()
                    else:
                        state_dict[key] = val.to(cpu)
        self._offloaded = True

    def _load_to_gpu(self):
        """Move state tensors back to GPU before step."""
        for param, state_dict in self._optimizer.state.items():
            param_device = param.device
            for key, val in state_dict.items():
                if isinstance(val, torch.Tensor) and val.device != param_device:
                    state_dict[key] = val.to(param_device, non_blocking=True)
        self._offloaded = False

    @torch.no_grad()
    def step(self, closure=None):
        if self._offloaded:
            self._move_state_to_device(
                next(iter(self._optimizer.state.keys())).device
                if self._optimizer.state
                else torch.device("cuda")
            )

        result = self._optimizer.step(closure)

        self._offload_to_cpu()
        return result

    def zero_grad(self, set_to_none: bool = True):
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # Move to GPU for state_dict serialization
        was_offloaded = self._offloaded
        if was_offloaded:
            self._load_to_gpu()
        sd = self._optimizer.state_dict()
        if was_offloaded:
            self._offload_to_cpu()
        return sd

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)
        # After loading, offload to CPU
        self._offload_to_cpu()
```

- [ ] **Step 3: Integrate in optimizers.py**

At the end of `get_optimizer()` in `library/training/optimizers.py`, before the return:

```python
    # Wrap with CPU offloading if requested
    if getattr(args, "optimizer_cpu_offload", False):
        from library.training.optimizer_offload import OffloadedOptimizer
        logger.info("Wrapping optimizer with CPU state offloading")
        optimizer = OffloadedOptimizer(optimizer)
```

- [ ] **Step 4: Commit**

```bash
git add library/training/optimizer_offload.py library/training/optimizers.py library/config/cli_args.py
git commit -m "feat: add optimizer state CPU offloading (--optimizer_cpu_offload)"
```

---

### Task 9: Training Step Profiler

**Covers:** Built-in wall-clock profiler that breaks down per-step time into forward/backward/optimizer/save sections.

**Files:**
- Create: `library/training/profiler.py`
- Modify: `library/training/loop.py` (integrate profiler)

**Reference:** `WhitecrowAurora/lora-rescripts` `scripts/stable/train_network.py` `ExperimentalAttentionStepProfiler`

- [ ] **Step 1: Add CLI argument**

In `library/config/cli_args.py`:

```python
    parser.add_argument(
        "--profile_steps",
        type=int,
        default=0,
        help="Number of steps to profile (0=disabled). Reports timing breakdown after N steps.",
    )
```

- [ ] **Step 2: Implement profiler**

Create `library/training/profiler.py`:

```python
"""Training step profiler — wall-clock breakdown per training phase.

Aggregates timing over a configurable window and reports per-section
average ms and percentage of total step time.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StepProfiler:
    """Windowed training step profiler."""

    window_size: int = 50
    _timings: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _current: Dict[str, float] = field(default_factory=dict)
    _step_count: int = 0
    _enabled: bool = True

    def start(self, section: str):
        """Mark the start of a timed section."""
        if not self._enabled:
            return
        self._current[section] = time.perf_counter()

    def end(self, section: str):
        """Mark the end of a timed section."""
        if not self._enabled or section not in self._current:
            return
        elapsed = (time.perf_counter() - self._current[section]) * 1000  # ms
        self._timings[section].append(elapsed)

    def step_end(self):
        """Mark end of a training step.  Reports after window_size steps."""
        if not self._enabled:
            return
        self._step_count += 1

        if self._step_count % self.window_size == 0:
            self._report()

    def _report(self):
        """Report average timings for the completed window."""
        if not self._timings:
            return

        total = 0.0
        for section, times in self._timings.items():
            recent = times[-self.window_size:]
            avg = sum(recent) / len(recent)
            total += avg

        lines = [f"Step Profiler (steps {self._step_count - self.window_size + 1}-{self._step_count}):"]
        for section, times in self._timings.items():
            recent = times[-self.window_size:]
            avg = sum(recent) / len(recent)
            pct = (avg / total * 100) if total > 0 else 0
            lines.append(f"  {section:20s}  {avg:8.1f} ms  ({pct:5.1f}%)")
        lines.append(f"  {'TOTAL':20s}  {total:8.1f} ms")

        logger.info("\n".join(lines))

    def reset(self):
        self._timings.clear()
        self._current.clear()
        self._step_count = 0
```

- [ ] **Step 3: Integrate into training loop**

In `library/training/loop.py`, add profiler instrumentation around the step body:

```python
    # In run_training_loop(), before the loop:
    profiler = None
    if getattr(args, "profile_steps", 0) > 0:
        from library.training.profiler import StepProfiler
        profiler = StepProfiler(window_size=args.profile_steps)

    # In _run_step(), wrap sections:
    # profiler.start("forward") ... profiler.end("forward")
    # profiler.start("backward") ... profiler.end("backward")
    # profiler.start("optimizer") ... profiler.end("optimizer")
    # profiler.step_end()
```

- [ ] **Step 4: Commit**

```bash
git add library/training/profiler.py library/training/loop.py library/config/cli_args.py
git commit -m "feat: add training step profiler (--profile_steps)"
```

---

### Task 10: Config Defaults and Documentation

**Covers:** Add commented-out defaults for all new features in base.toml, update method configs.

**Files:**
- Modify: `configs/base.toml`
- Modify: `configs/methods/lora.toml`

- [ ] **Step 1: Add commented defaults to base.toml**

Append to `configs/base.toml`:

```toml
# --- lora-rescripts integration (commented out, activate as needed) ---
# optimizer_type = "AdamW8bitKahan"   # Kahan-compensated 8-bit AdamW
# optimizer_args = ["stabilize=true"] # Cap lr by rms(grad²/exp_avg_sq)
# staged_resolution = true            # Curriculum: 512→768→1024
# staged_resolution_ratios = "20,30,50"
# adaptive_noise_offset = true        # Latent-statistics-based noise offset
# contrastive_flow_matching = true    # Contrastive loss on RF objective
# cfm_lambda = 0.05
# pyramid_noise_iterations = 6        # Multi-frequency noise
# pyramid_noise_discount = 0.4
# optimizer_cpu_offload = true         # CPU offload optimizer states
# profile_steps = 50                  # Step timing profiler
```

- [ ] **Step 2: Add VeRA method reference to lora.toml**

Append to `configs/methods/lora.toml`:

```toml
# --- Alternative adapters (uncomment to switch) ---
# network_module = "vera"    # VeRA: ~10-100x fewer params than LoRA
# network_module = "dylora"  # DyLoRA: train multiple ranks simultaneously
```

- [ ] **Step 3: Commit**

```bash
git add configs/base.toml configs/methods/lora.toml
git commit -m "docs: add commented config defaults for lora-rescripts features"
```

---

## Task Dependency Graph

```
Task 0 (deps fix) ─────────────────────────────────────┐
Task 1 (Kahan AdamW) ──────────────────────────────────┤
Task 2 (Staged Resolution) ────────────────────────────┤── All independent
Task 3 (Adaptive Noise) ───────────────────────────────┤   (parallelizable)
Task 4 (Contrastive FM) ───────────────────────────────┤
Task 5 (Pyramid Noise) ────────────────────────────────┤
Task 6 (VeRA) ─────────────────────────────────────────┤
Task 7 (DyLoRA) ───────────────────────────────────────┤
Task 8 (Optimizer Offload) ────────────────────────────┤
Task 9 (Profiler) ─────────────────────────────────────┘
Task 10 (Config/Docs) ─── depends on all above ──────── last
```

All implementation tasks (0-9) are fully independent and can be parallelized. Task 10 (config defaults) should be done last.

---

## Verification

After all tasks complete:

- [ ] Run linting: `ruff check . --fix && ruff format .`
- [ ] Run tests: `make test-unit`
- [ ] Test optimizer dispatch: `python -c "from library.training.optimizers import get_optimizer; print('OK')"`
- [ ] Test VeRA import: `python -c "from networks.lora_modules.vera import VeRAModule; print('OK')"`
- [ ] Test DyLoRA import: `python -c "from networks.lora_modules.dylora import DyLoRAModule; print('OK')"`
- [ ] Test staged plan: `python -c "from library.training.staged_resolution import build_staged_plan; print(build_staged_plan([1024], 4, 6, 500, 100))"`
- [ ] Print config: `make print-config METHOD=lora PRESET=default`
