#!/usr/bin/env python
"""APEX temporal-shift training probe — does ``t_fake = t + Δt`` drive useful gradient?

What this probe answers
-----------------------
``probe_temporal_shift.py`` showed that ``v_fake`` from a ±0.05 temporal shift
has MSE in cond_diag's ballpark — same supervision *strength*. But APEX's
theoretical guarantee (Section 3.3, Fisher-divergence-aligned gradient with
constant weight ``w ≡ 1``) leans on ``c_fake`` being an *independent estimator*
of ``p_fake``'s velocity. Temporal-shift breaks that assumption: the perturbed
forward is the same network at a slightly displaced ``t``, not an independent
branch. Magnitude similarity does not imply direction similarity in score
space.

So this probe runs a short training loop with t-shift as the adversarial
signal, and watches whether L_mix actually descends below the pure-FM baseline
on a held-out batch. Falsifies temporal-shift cheaply before committing to a
6-epoch APEX-scale run.

What the probe drops vs. APEX
-----------------------------
APEX has 3 forwards per step: real (Forward 1), shifted target for L_mix
(Forward 2), and shifted-on-fake-trajectory for L_fake (Forward 3, which
trains ``c_fake`` to be a useful adversary). Temporal-shift has nothing to
train — Δt is a fixed scalar — so:

  - Forward 2 swaps c-shift for t-shift: ``F_θ(x_t, t+Δt, c)`` as the
    ``v_fake`` stop-grad target for L_mix.
  - Forward 3 / L_fake / ConditionShift module: skipped entirely.

Net cost: 2 forwards/step (vs APEX's 3), 0 perturbation params (vs APEX's
diag/full ConditionShift), single-term loss ``λ_c · L_mix`` (no λ_p · L_fake).

Decision rule (what this probe does and does not gate)
------------------------------------------------------
This probe gates on **training stability and v_fake aliveness**, NOT on
sample quality. Quality validation requires generation + eval (T3).

PASS (graduate to T3, full training run):
  - ``L_mix`` curve descends smoothly and stays bounded (no NaN, no spike).
  - ``v_fake_divergence`` stays in 1e-3 to 1e-1 (where APEX cond_diag sits
    at warmstart per probe_temporal_shift.py). Don't let it collapse to 0
    (perturbation absorbed by adapter — degenerate fixed-point) or explode
    above ~0.2 (perturbation gone into dt_big regime, training-side noise).
  - ``grad_norm`` does not blow up.

FAIL (do not run T3):
  - NaN, divergence, oscillation in L_mix.
  - v_fake_divergence collapses to 0 within 500 steps (degenerate).

The headline ratio ``L_mix / L_fm_obs`` is reported but is NOT a quality
proxy: when ``v_real ≈ v_fake`` (small Δt perturbation, warmstart base),
T_mix is mechanically between v_data and v_real, so L_mix < L_fm is
arithmetic, not evidence. Use the ratio only to check the perturbation
is doing *something* (ratio close to 1.0 → t-shift is invisible).

Setup
-----
  - DiT loaded with the warmstart **merged in** (matches apex.toml's
    "merge then train fresh adapter" pattern, see configs/methods/apex.toml).
  - A small fresh LoRA (rank 8 by default) on the 56 cross-attn Linears
    (q_proj + kv_proj across 28 blocks). ~2M trainable params — plenty
    for a probe, small enough to fit at batch=1 with two forwards.
  - One bucket only (default: 0832x1248 — 1182 cached samples in the
    standard LoRA dataset). Static shape → no recompile churn.

Outputs ``bench/apex/results/<YYYYMMDD-HHMM>[-<label>]/``::

    result.json       — schedule, summary stats, ratio L_mix / L_fm_baseline
    loss.csv          — per-step losses + telemetry
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import weights as anima_weights  # noqa: E402
from library.training.apex_loss import apex_schedule_weights  # noqa: E402
from networks.lora_modules.lora import LoRAModule  # noqa: E402
from networks.methods.apex import ConditionShift  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


# ----- data: cached latents + TE embeddings (one bucket only) ---------------


def _list_bucket_files(cache_dir: Path, bucket: str):
    """Find (latent_path, te_path) pairs whose filename suffix matches bucket.
    bucket format: '0832x1248' (exactly as it appears in the npz filename)."""
    latents = sorted(cache_dir.glob(f"*_{bucket}_anima.npz"))
    pairs: list[tuple[Path, Path]] = []
    for lp in latents:
        stem = lp.name.removesuffix(f"_{bucket}_anima.npz")
        tp = cache_dir / f"{stem}_anima_te.safetensors"
        if tp.exists():
            pairs.append((lp, tp))
    return pairs


def load_batch(pairs, idxs, *, device, dtype):
    """Stack a batch of (latent, crossattn_emb) from the cached pair list."""
    import numpy as np
    from safetensors import safe_open

    lats, embs = [], []
    for i in idxs:
        lp, tp = pairs[i]
        with np.load(lp) as d:
            key = next(k for k in d.files if k.startswith("latents_"))
            lats.append(torch.from_numpy(d[key]).to(device=device, dtype=dtype))
        with safe_open(str(tp), framework="pt") as f:
            # crossattn_emb_v0 — same selection rule as probe_temporal_shift.py.
            embs.append(f.get_tensor("crossattn_emb_v0").to(device=device, dtype=dtype))
    x_0 = torch.stack(lats, dim=0).unsqueeze(2)  # [B, C, 1, H, W]
    c = torch.stack(embs, dim=0)                  # [B, S, D]
    return x_0, c


# ----- DiT forward (training-aware: NOT decorated with no_grad) -------------


def dit_forward(model, x_5d, t_b, c, padding_mask):
    return model.forward_mini_train_dit(
        x_B_C_T_H_W=x_5d,
        timesteps_B_T=t_b,
        crossattn_emb=c,
        padding_mask=padding_mask,
    )


# ----- adapter setup --------------------------------------------------------


_TARGET_PAT = re.compile(r"^blocks\.\d+\.cross_attn\.(q_proj|kv_proj)$")


def attach_probe_lora(model, *, rank: int, alpha: float, device, dtype):
    """Wrap every cross_attn q_proj/kv_proj Linear with a fresh LoRAModule.

    Returns the ModuleList holding the adapters (already in train mode and
    moved to (device, dtype)). Trainable parameter list is ``module_list.parameters()``.
    """
    adapters: list[LoRAModule] = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and _TARGET_PAT.match(name):
            lora_name = "lora_unet_" + name.replace(".", "_")
            adapters.append(
                LoRAModule(
                    lora_name,
                    mod,
                    multiplier=1.0,
                    lora_dim=rank,
                    alpha=alpha,
                )
            )
    if not adapters:
        raise SystemExit("attach_probe_lora: no cross-attn targets matched")
    for a in adapters:
        a.apply_to()
    bank = torch.nn.ModuleList(adapters).to(device=device, dtype=dtype)
    bank.train()
    for p in bank.parameters():
        p.requires_grad_(True)
    return bank


# ----- main -----------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dit", default="models/diffusion_models/anima-preview3-base.safetensors")
    p.add_argument(
        "--warmstart",
        default="output/ckpt/anima-tlora-0507-12.safetensors",
        help="LoRA merged into DiT at load time. APEX-style frozen base.",
    )
    p.add_argument("--cache-dir", default="post_image_dataset/lora")
    p.add_argument(
        "--bucket",
        default="0832x1248",
        help="Latent-bucket suffix to filter on; one bucket → static shape.",
    )
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--rank", type=int, default=8, help="Probe LoRA rank.")
    p.add_argument("--alpha", type=float, default=8.0, help="Probe LoRA alpha.")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument(
        "--mode",
        choices=("temporal", "cond_diag", "combined"),
        default="temporal",
        help=(
            "Perturbation form for Forward 2 (the v_fake target):\n"
            "  temporal  — anima(x_t, t+Δt, c)        — t-shift only (default).\n"
            "  cond_diag — anima(x_t, t,    c_fake)   — APEX c-shift only "
            "(ConditionShift mode=diag, init_a=-0.5, init_b=0, frozen).\n"
            "  combined  — anima(x_t, t+Δt, c_fake)   — both perturbations "
            "in one forward (Option A; same 2-forwards/step cost as temporal).\n"
            "Cross-strategy cosine probe (probe_temporal_shift.py) showed c-shift "
            "and t-shift point in roughly orthogonal directions in output-delta "
            "space — combining is justified by that measurement, but only T3 "
            "tells you whether it improves NFE=1 quality."
        ),
    )
    p.add_argument(
        "--dt",
        type=float,
        default=-0.05,
        help="Temporal shift Δt for the v_fake target. probe_temporal_shift.py "
        "showed -0.05 is more orthogonal to c-shift than +0.05 (cond_diag × "
        "dt_neg_small ≈ 0 vs +0.31 for dt_pos_small) and avoids the upper-clamp "
        "artifact at t=0.98. Only used when mode ∈ {temporal, combined}.",
    )
    p.add_argument(
        "--apex-lambda",
        type=float,
        default=0.5,
        help="Inner T_mix mixing coefficient (paper Eq. 23). Matches apex.toml.",
    )
    p.add_argument(
        "--apex-lambda-c",
        type=float,
        default=1.0,
        help="Outer L_mix weight (paper Eq. 25 lam_c). Constant.",
    )
    p.add_argument(
        "--rampup-ratio",
        type=float,
        default=0.10,
        help="Linear ramp of inner lambda over first rampup_ratio*steps.",
    )
    p.add_argument(
        "--anchor-ratio",
        type=float,
        default=0.05,
        help="Per-batch fraction with lam_inner=0 (pure FM anchor, EMF "
        "Theorem 4.3 validity). Matches apex.toml's 0.05.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print + record every N steps (CSV always records every step).",
    )
    p.add_argument(
        "--grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Per-block gradient checkpointing — matches apex.toml's pattern. "
        "On 16 GB cards this is required to fit 2 forwards on the 0832x1248 bucket.",
    )
    p.add_argument(
        "--cpu-offload-checkpointing",
        action="store_true",
        help="Pair with --grad-checkpoint to offload activations to CPU.",
    )
    p.add_argument(
        "--compile-blocks",
        action="store_true",
        default=True,
        help="torch.compile each block._forward — reduces peak VRAM by cutting "
        "cross-block fragmentation. Compile-time cost is paid once at the first step.",
    )
    p.add_argument(
        "--compile-mode",
        default=None,
        help="torch.compile mode for --compile-blocks (e.g. 'reduce-overhead').",
    )
    p.add_argument("--latent-clamp-min", type=float, default=0.02)
    p.add_argument("--latent-clamp-max", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--label", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16

    # ----- DiT (with warm-start MERGED — frozen base) -----------------------
    lora_weights_list = None
    if args.warmstart:
        ws = load_file(str(_resolve(args.warmstart)))
        lora_weights_list = [{k: v for k, v in ws.items() if k.startswith("lora_unet_")}]
        print(f"warm-start: merging {args.warmstart} ({len(lora_weights_list[0])} keys)")
    print(f"loading DiT from {args.dit}")
    model = anima_weights.load_anima_model(
        device,
        str(_resolve(args.dit)),
        "torch",
        True,
        "cpu",
        dtype,
        lora_weights_list=lora_weights_list,
        lora_multipliers=[1.0] if lora_weights_list else None,
    )
    # Frozen base, but train()-mode so gradient_checkpointing fires (it gates on
    # ``self.training and self.gradient_checkpointing`` in models.py:1188).
    # The LoRA bank below also needs train mode to take the training fast-path.
    model.requires_grad_(False)
    model.train()
    model.to(device)
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing(
            cpu_offload=args.cpu_offload_checkpointing,
            unsloth_offload=False,
        )
        print(
            f"gradient checkpointing: ON "
            f"(cpu_offload={args.cpu_offload_checkpointing})"
        )
    if args.compile_blocks:
        model.compile_blocks(backend="inductor", mode=args.compile_mode)

    # ----- fresh probe LoRA on cross_attn (trainable) -----------------------
    bank = attach_probe_lora(model, rank=args.rank, alpha=args.alpha, device=device, dtype=dtype)
    n_params = sum(p.numel() for p in bank.parameters())
    print(f"probe LoRA: {len(bank)} adapters, {n_params/1e6:.2f}M params, rank={args.rank}")

    optimizer = torch.optim.AdamW(bank.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # ----- ConditionShift (frozen) for cond_diag / combined modes ----------
    # No L_fake in this probe → (a, b) have nothing to train them, so freeze
    # at the shipped APEX init. mode="diag" matches apex.toml; at uniform init
    # this is bit-equivalent to mode="scalar" (apex-0506 §"Synthetic q" note).
    condshift = None
    if args.mode in ("cond_diag", "combined"):
        ctx_dim = model.blocks[0].cross_attn.context_dim
        condshift = ConditionShift(
            dim=ctx_dim, mode="diag", init_a=-0.5, init_b=0.0,
        ).to(device=device, dtype=dtype)
        condshift.requires_grad_(False)
        condshift.eval()
        print(
            f"ConditionShift: dim={ctx_dim}, mode=diag, init_a=-0.5, init_b=0.0 (frozen)"
        )

    # ----- data ------------------------------------------------------------
    cache_dir = _resolve(args.cache_dir)
    pairs = _list_bucket_files(cache_dir, args.bucket)
    if len(pairs) < args.batch_size:
        raise SystemExit(
            f"need ≥{args.batch_size} cached pairs in bucket {args.bucket}, "
            f"found {len(pairs)} under {cache_dir}"
        )
    print(f"bucket {args.bucket}: {len(pairs)} cached pairs")
    rng = torch.Generator(device="cpu").manual_seed(int(args.seed))
    # Probe latent shape → static padding_mask once.
    sample_x0, _ = load_batch(pairs, [0], device=device, dtype=dtype)
    _, _, _, H, W = sample_x0.shape
    padding_mask = torch.zeros(args.batch_size, 1, H, W, device=device, dtype=dtype)
    print(f"latent shape per sample: ({sample_x0.shape[1]}, {H}, {W})")

    # ----- schedule --------------------------------------------------------
    rampup_steps = max(1, int(round(args.rampup_ratio * args.steps)))
    print(
        f"schedule: lam_inner ramps 0 → {args.apex_lambda} over {rampup_steps} steps "
        f"(no warmup; anchor_ratio={args.anchor_ratio})"
    )

    # ----- training loop ---------------------------------------------------
    csv_rows: list[dict] = []
    t_clamp_min, t_clamp_max = args.latent_clamp_min, args.latent_clamp_max
    t_eps = 1e-3
    t0 = time.time()
    for step in range(args.steps):
        # Sample batch indices
        idxs = torch.randint(0, len(pairs), (args.batch_size,), generator=rng).tolist()
        x_0, c = load_batch(pairs, idxs, device=device, dtype=dtype)  # [B,C,1,H,W], [B,S,D]

        # Sample t, build flow-matching pair
        t = torch.rand(args.batch_size, device=device, dtype=torch.float32)
        t = t.clamp(t_clamp_min, t_clamp_max).to(dtype)
        t_b5 = t.view(-1, 1, 1, 1, 1)
        z = torch.randn_like(x_0)
        x_t = t_b5 * z + (1.0 - t_b5) * x_0
        v_data_5d = (z - x_0).detach()

        # Forward 1: real (grad on)
        v_real = dit_forward(model, x_t, t, c, padding_mask)

        # Forward 2: v_fake target (no grad). Dispatch on --mode:
        #   temporal  : perturb t only      (anima(x_t, t+Δt, c))
        #   cond_diag : perturb c only      (anima(x_t, t,    c_fake))
        #   combined  : perturb t and c     (anima(x_t, t+Δt, c_fake)) — Option A
        with torch.no_grad():
            if args.mode in ("temporal", "combined"):
                t_eff = (t.float() + args.dt).clamp(t_clamp_min, t_clamp_max).to(dtype)
            else:
                t_eff = t
            if args.mode in ("cond_diag", "combined"):
                c_eff = condshift(c)
            else:
                c_eff = c
            v_fake = dit_forward(model, x_t, t_eff, c_eff, padding_mask)
            v_fake_divergence = float(((v_fake - v_real.detach()) ** 2).mean().item())

        # Schedule + per-batch anchor
        lam_inner_eff, _ = apex_schedule_weights(
            step=step,
            warmup_steps=0,
            rampup_steps=rampup_steps,
            lam_inner_target=float(args.apex_lambda),
            lam_f_target=0.0,  # no L_fake in temporal-shift
        )
        if args.anchor_ratio > 0.0 and lam_inner_eff > 0.0:
            n_anchor = int(round(args.batch_size * args.anchor_ratio))
            n_anchor = max(0, min(args.batch_size, n_anchor))
            anchor_mask_b = torch.zeros(args.batch_size, device=device, dtype=dtype)
            if n_anchor > 0:
                perm = torch.randperm(args.batch_size, device=device, generator=None)
                anchor_mask_b[perm[:n_anchor]] = 1.0
        else:
            anchor_mask_b = torch.zeros(args.batch_size, device=device, dtype=dtype)
        lam_inner_per = (lam_inner_eff * (1.0 - anchor_mask_b)).view(-1, 1, 1, 1, 1)

        # T_mix in velocity space (paper Eq. 23 after Prop. 3 conversion)
        T_mix = (1.0 - lam_inner_per) * v_data_5d + lam_inner_per * v_fake
        T_mix = T_mix.detach()

        # L_mix = MSE(v_real, T_mix.sg) — paper Eq. 24
        L_mix = (v_real.float() - T_mix.float()).pow(2).mean()

        # Reference: pure FM loss on the same batch (kill-criterion baseline).
        # No grad — just observe.
        with torch.no_grad():
            L_fm_obs = (v_real.detach().float() - v_data_5d.float()).pow(2).mean().item()

        loss = float(args.apex_lambda_c) * L_mix
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Grad norm before step for telemetry
        with torch.no_grad():
            sq = 0.0
            for prm in bank.parameters():
                if prm.grad is not None:
                    sq += float(prm.grad.detach().pow(2).sum().item())
            grad_norm = math.sqrt(sq)
        optimizer.step()

        csv_rows.append({
            "step": step,
            "loss": float(loss.item()),
            "L_mix": float(L_mix.item()),
            "L_fm_obs": float(L_fm_obs),
            "v_fake_divergence": v_fake_divergence,
            "lam_inner_eff": float(lam_inner_eff),
            "grad_norm": grad_norm,
            "t_mean": float(t.float().mean().item()),
        })
        if step % args.log_every == 0 or step == args.steps - 1:
            print(
                f"  step {step:4d}  loss={loss.item():.5f}  L_mix={L_mix.item():.5f}  "
                f"L_fm_obs={L_fm_obs:.5f}  v_fake_div={v_fake_divergence:.5f}  "
                f"lam={lam_inner_eff:.3f}  |g|={grad_norm:.3f}"
            )

    elapsed = time.time() - t0
    print(f"\nfinished {args.steps} steps in {elapsed:.1f}s")

    # ----- summary ----------------------------------------------------------
    def tail_mean(key: str, frac: float = 0.2) -> float:
        n = max(1, int(len(csv_rows) * frac))
        return sum(r[key] for r in csv_rows[-n:]) / n

    summary = {
        "L_mix_first10_mean": sum(r["L_mix"] for r in csv_rows[:10]) / max(1, min(10, len(csv_rows))),
        "L_mix_last20pct_mean": tail_mean("L_mix"),
        "L_fm_first10_mean": sum(r["L_fm_obs"] for r in csv_rows[:10]) / max(1, min(10, len(csv_rows))),
        "L_fm_last20pct_mean": tail_mean("L_fm_obs"),
        "v_fake_div_last20pct_mean": tail_mean("v_fake_divergence"),
        "v_fake_div_first10_mean": sum(r["v_fake_divergence"] for r in csv_rows[:10])
        / max(1, min(10, len(csv_rows))),
        "grad_norm_last20pct_mean": tail_mean("grad_norm"),
        "elapsed_s": elapsed,
        "steps_per_s": args.steps / max(1e-6, elapsed),
        "trainable_params_M": n_params / 1e6,
    }
    # Headline ratio: how much did L_mix descend below the FM-on-the-same-batch line?
    # < 1.0 means t-shift drove v_real toward the (1-λ)·v_data + λ·v_fake target
    # *better* than v_real would have tracked v_data alone. The kill criterion.
    summary["L_mix_ratio_to_fm_obs"] = (
        summary["L_mix_last20pct_mean"] / summary["L_fm_last20pct_mean"]
        if summary["L_fm_last20pct_mean"] > 0
        else float("nan")
    )

    # ----- artifacts --------------------------------------------------------
    if args.label:
        label = args.label
    elif args.mode == "cond_diag":
        label = "cond-diag-train"
    elif args.mode == "combined":
        label = f"combined-train-dt{args.dt:+.2f}"
    else:
        label = f"temporal-train-dt{args.dt:+.2f}"
    run_dir = make_run_dir("apex", label=label)

    csv_path = run_dir / "loss.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"summary": summary, "n_steps": args.steps, "rampup_steps": rampup_steps},
        label=label,
        artifacts=["loss.csv"],
        device=device,
    )

    bar = "=" * 72
    print()
    print(bar)
    if args.mode == "cond_diag":
        print(f"APEX training probe  mode=cond_diag  (c-shift only)")
    elif args.mode == "combined":
        print(f"APEX training probe  mode=combined  (Δt={args.dt:+.2f}, c-shift)")
    else:
        print(f"APEX training probe  mode=temporal  (Δt={args.dt:+.2f})")
    print(f"  steps           : {args.steps}  rampup: {rampup_steps}")
    print(f"  L_mix           : {summary['L_mix_first10_mean']:.5f}  →  {summary['L_mix_last20pct_mean']:.5f}")
    print(f"  L_fm_obs        : {summary['L_fm_first10_mean']:.5f}  →  {summary['L_fm_last20pct_mean']:.5f}")
    print(f"  L_mix / L_fm    : {summary['L_mix_ratio_to_fm_obs']:.3f}  (sanity: ~1 = t-shift invisible; not a quality proxy)")
    print(f"  v_fake_div      : {summary['v_fake_div_first10_mean']:.5f}  →  {summary['v_fake_div_last20pct_mean']:.5f}")
    print(f"  grad_norm tail  : {summary['grad_norm_last20pct_mean']:.4f}")
    print(f"  artifacts       → {run_dir}")
    print(bar)


if __name__ == "__main__":
    main()
