#!/usr/bin/env python3
"""Three-arm LoRA-init parameterization probe: Kaiming vs SVD-Down vs OrthoInit.

THE QUESTION (original). When ``student_ortho_init=true`` was switched on for
turbo (DP-DMD), the generator GAN loss climbed and the checkpoint went to
garbage. Hypothesis (no code bug): OrthoInit's λ-first parameterization changes
the early effective-ΔW step relative to plain LoRA, and DP-DMD's adversarial
loop has a narrow stability band. The original two-arm probe put a NUMBER on
that and *falsified the "hot start" direction*: OrthoInit's step-1 ‖ΔW‖ is ~25×
SMALLER than plain LoRA's (cold start, not hot) — see
``results/20260621-1420-ortho-init-step``.

THE QUESTION (this extension — Phase 0 of _archive/proposals/svd_down_lora_init.md).
If OrthoInit's defect is a narrow cold-start tangent (only its r λ-amplitudes
get gradient at init), can we keep plain LoRA's WIDE first-step tangent (the
whole d_out×r up-projection) while still seeding the *input* basis from W₀'s
principal right singular vectors? That is **SVD-Down LoRA**:

    A_0 = V_rᵀ / sqrt(3),   B_0 = 0   (still ordinary LoRA after init)

where V_r = top-r right singular vectors of W₀ and the 1/sqrt(3) matches the
expected row-norm of the current ``kaiming_uniform_(a=sqrt(5))`` down init, so a
"win" cannot be a disguised larger step.

WHAT IT MEASURES. Build the SAME frozen W₀ as three adapters at one operating
point and run an identical forward+backward+AdamW step on each:

  1. ``lora``      — Kaiming-down plain LoRA (current default).
  2. ``svd_down``  — normalized SVD-Down LoRA (the proposal).
  3. ``ortho``     — current P·diag(λ)·Q OrthoInit.

Per arm, per step: ‖ΔW‖_F, loss, step-0 factor-gradient split, and the cosine
of the step-1 ΔW with the **ideal full-FT descent direction** −∂L/∂W₀ (the
direction full fine-tuning would move on step 1). Run BOTH an isotropic input
arm (``align=0`` — pure parameterization effect) and a W₀-aligned input arm
(``align=1`` — activations concentrated where W₀ has gain, so V_r's prior can
bite) and print the four Phase-0 gates.

PHASE-0 GATES (svd_down must pass all to advance to training):
  G1  zero-output at init                         (init ‖ΔW‖ == 0)
  G2  step-1 gradient lives only in lora_up        (down grad ≈ 0, like LoRA)
  G3  step-1 ‖ΔW‖ within 0.5×–2× plain LoRA        (no disguised step size)
  G4  cos(ΔW,ideal) improves in the ALIGNED arm    without hurting the random arm

This is a parameterization-level numeric fact, independent of dtype/model —
fp32, synthetic power-law W₀ by default (no model load).

Run::

    python bench/turbo/probe_ortho_init_step.py
    python bench/turbo/probe_ortho_init_step.py --rank 64 --alpha 180 --lr 1e-5
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench._common import make_run_dir, write_result  # noqa: E402
from networks.lora_modules.lora import LoRAModule  # noqa: E402
from networks.lora_modules.ortho import OrthoInitLoRAModule  # noqa: E402

ARMS = ("lora", "svd_down", "ortho")


def _power_law_weight(out_dim: int, in_dim: int, decay: float, gen: torch.Generator):
    """W₀ = U diag(s) Vᵀ with a power-law spectrum (realistic trained-weight σ)."""
    k = min(out_dim, in_dim)
    U, _ = torch.linalg.qr(torch.randn(out_dim, k, generator=gen))
    V, _ = torch.linalg.qr(torch.randn(in_dim, k, generator=gen))
    idx = torch.arange(1, k + 1, dtype=torch.float32)
    s = idx ** (-decay)
    s = s / s[0]
    W = (U * s.unsqueeze(0)) @ V.T
    return W.contiguous(), V, s  # V: (in, k) right singular vectors


def _make_input(B, V, s, align_power, gen):
    """x lives in W₀'s input subspace with energy following the spectrum^align.

    align_power=0 → isotropic (pure parameterization effect); 1 → activations
    concentrated where W₀ has gain (the realistic regime, where V_r bites).
    """
    in_dim, k = V.shape
    w = (s ** align_power).unsqueeze(0)  # (1, k)
    coeffs = torch.randn(B, k, generator=gen) * w
    return coeffs @ V.T  # (B, in)


def _dW_lora(m) -> torch.Tensor:
    return (m.multiplier * m.scale) * (m.lora_up.weight.data @ m.lora_down.weight.data)


def _dW_ortho(m) -> torch.Tensor:
    # ΔW = scale · P_init @ diag(λ) @ Q_init  (mask ≡ 1 here).
    return (m.multiplier * m.scale) * (
        (m.P_init.data * m.lambda_layer.data) @ m.Q_init.data
    )


def _pnames(kind):
    return ["P_init", "Q_init", "lambda"] if kind == "ortho" else ["down", "up"]


def _build(kind, org_linear, rank, alpha, device):
    """Construct one adapter arm on the shared frozen W₀.

    ``svd_down`` is a plain ``LoRAModule`` whose ``lora_down`` is overwritten by
    V_rᵀ/sqrt(3) — using the SAME randomized-SVD construction the proposal ships
    (``networks/lora_modules/ortho.py``), so the probe exercises the real code
    path, not an idealized exact SVD.
    """
    if kind == "ortho":
        m = OrthoInitLoRAModule(
            "probe", org_linear, multiplier=1.0, lora_dim=rank, alpha=alpha
        )
        params = [m.P_init, m.Q_init, m.lambda_layer]
        dW = _dW_ortho
    else:
        m = LoRAModule("probe", org_linear, multiplier=1.0, lora_dim=rank, alpha=alpha)
        params = [m.lora_down.weight, m.lora_up.weight]
        dW = _dW_lora
    m.to(device)
    if kind == "svd_down":
        W = org_linear.weight.data.float().to(device)
        q = min(rank + 6, min(W.shape))
        _, _, V = torch.svd_lowrank(W, q=q, niter=2)  # V: (in, q)
        with torch.no_grad():
            v_r = (V[:, :rank].T / math.sqrt(3)).to(m.lora_down.weight.dtype)
            m.lora_down.weight.copy_(v_r)
    # Bypass apply_to() (no monkey-patch); forward only needs org_forward set,
    # and the optimizer gets ONLY the adapter params (never the frozen W₀).
    m.org_forward = org_linear
    m.train()
    for p in params:
        p.requires_grad_(True)
    return m, params, dW


def _run_one(kind, org_linear, x, y, ideal, rank, alpha, lr, n_steps, device):
    """Return per-step ‖ΔW‖, step-0 grad split, init ‖ΔW‖, and ΔW↔ideal cosines."""
    m, params, dW = _build(kind, org_linear, rank, alpha, device)
    opt = torch.optim.AdamW(params, lr=lr)
    init_dW = float(dW(m).norm())  # G1: must be exactly 0
    norms, losses, grad_split, cos1 = [], [], None, None
    for step in range(n_steps):
        opt.zero_grad()
        out = m(x)
        loss = 0.5 * ((out - y) ** 2).mean()
        loss.backward()
        if step == 0:
            grad_split = {n: float(p.grad.norm()) for n, p in zip(_pnames(kind), params)}
        opt.step()
        d = dW(m)
        norms.append(float(d.norm()))
        losses.append(float(loss.detach()))
        if step == 0:
            cos1 = _cos(d.flatten(), ideal)
    cos_final = _cos(dW(m).flatten(), ideal)
    return {
        "init_dW": init_dW,
        "dW": norms,
        "loss": losses,
        "grad_split": grad_split,
        "cos_step1": cos1,
        "cos_final": cos_final,
    }


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if float(na) < 1e-30 or float(nb) < 1e-30:
        return 0.0
    return float((a @ b) / (na * nb))


def _ideal_direction(org, x, y):
    """−∂L/∂W₀ at ΔW=0: the rank-full descent direction full FT would take."""
    Wg = org.weight.detach().clone().requires_grad_(True)
    out = torch.nn.functional.linear(x, Wg)
    loss = 0.5 * ((out - y) ** 2).mean()
    loss.backward()
    return (-Wg.grad.detach()).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dim", type=int, default=2048)
    ap.add_argument("--out-dim", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=180.0, help="turbo student_alpha")
    ap.add_argument("--lr", type=float, default=1e-5, help="turbo student_lr")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--spectrum-decay", type=float, default=1.0, help="σ_i ∝ i^-decay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)  # svd_lowrank uses the global RNG
    gen = torch.Generator().manual_seed(args.seed)

    W, V, s = _power_law_weight(args.out_dim, args.in_dim, args.spectrum_decay, gen)
    org = torch.nn.Linear(args.in_dim, args.out_dim, bias=False)
    org.weight.data.copy_(W)
    org.weight.requires_grad_(False)
    org.to(device)

    # Two input regimes: aligned (V_r prior can bite) and isotropic (control).
    regimes = {}
    for name, align in (("aligned", 1.0), ("isotropic", 0.0)):
        x = _make_input(args.batch, V, s, align, gen).to(device)
        with torch.no_grad():
            y = org(x) + 0.1 * torch.randn(
                args.batch, args.out_dim, generator=gen
            ).to(device)
        ideal = _ideal_direction(org, x, y)
        arms = {
            k: _run_one(
                k, org, x, y, ideal, args.rank, args.alpha, args.lr, args.n_steps, device
            )
            for k in ARMS
        }
        regimes[name] = arms

    # --- Phase-0 gate evaluation (svd_down) -------------------------------
    al, iso = regimes["aligned"], regimes["isotropic"]
    sd_a, lo_a = al["svd_down"], al["lora"]
    sd_i, lo_i = iso["svd_down"], iso["lora"]
    ratio_a = sd_a["dW"][0] / max(lo_a["dW"][0], 1e-30)
    ratio_i = sd_i["dW"][0] / max(lo_i["dW"][0], 1e-30)
    gates = {
        "G1_zero_init": (sd_a["init_dW"] == 0.0 and sd_i["init_dW"] == 0.0),
        "G2_grad_up_only": (
            sd_a["grad_split"]["down"] == 0.0 and sd_i["grad_split"]["down"] == 0.0
        ),
        "G3_step1_norm_0p5_2x": (0.5 <= ratio_a <= 2.0 and 0.5 <= ratio_i <= 2.0),
        "G4_aligned_cos_gain_no_iso_harm": (
            sd_a["cos_step1"] > lo_a["cos_step1"]
            and sd_i["cos_step1"] >= lo_i["cos_step1"] - 0.02
        ),
    }
    phase0_pass = all(gates.values())

    run_dir = make_run_dir("turbo", label=args.label or "svd-down-phase0")

    # trajectory CSV (aligned regime, all three arms)
    traj = run_dir / "dW_trajectory_aligned.csv"
    with traj.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", *(f"{k}_dW" for k in ARMS)])
        for i in range(args.n_steps):
            w.writerow([i + 1, *(f"{al[k]['dW'][i]:.6e}" for k in ARMS)])

    metrics = {
        "phase0_pass": phase0_pass,
        "gates": gates,
        "step1_dW_ratio_svd_down_over_lora": {"aligned": ratio_a, "isotropic": ratio_i},
        "regimes": {
            name: {
                k: {
                    "init_dW": arms[k]["init_dW"],
                    "dW_step1": arms[k]["dW"][0],
                    "dW_final": arms[k]["dW"][-1],
                    "loss_step1": arms[k]["loss"][0],
                    "loss_final": arms[k]["loss"][-1],
                    "grad_split_step0": arms[k]["grad_split"],
                    "cos_ideal_step1": arms[k]["cos_step1"],
                    "cos_ideal_final": arms[k]["cos_final"],
                }
                for k in ARMS
            }
            for name, arms in regimes.items()
        },
    }

    write_result(
        run_dir, script=__file__, args=args, metrics=metrics, device=device,
        artifacts=[traj], label=args.label or "svd-down-phase0",
    )

    print(f"\n=== SVD-Down LoRA — Phase 0 (rank={args.rank} alpha={args.alpha} "
          f"lr={args.lr:g} W₀ decay={args.spectrum_decay}) ===")
    for name, arms in regimes.items():
        print(f"\n[{name} input]")
        print(f"  {'arm':>9} {'init‖ΔW‖':>10} {'step1‖ΔW‖':>11} "
              f"{'cos→ideal_1':>12} {'cos→ideal_N':>12} {'grad(down/P)':>13}")
        for k in ARMS:
            a = arms[k]
            g0 = list(a["grad_split"].values())[0]
            print(f"  {k:>9} {a['init_dW']:10.3e} {a['dW'][0]:11.3e} "
                  f"{a['cos_step1']:12.4f} {a['cos_final']:12.4f} {g0:13.3e}")
    print(f"\n  step-1 ‖ΔW‖ ratio svd_down/lora:  aligned={ratio_a:.3f}  "
          f"isotropic={ratio_i:.3f}")
    print("\n  Phase-0 gates:")
    for g, ok in gates.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {g}")
    print(f"\n  PHASE 0: {'PASS' if phase0_pass else 'FAIL'}")
    print(f"\nwrote {run_dir}/result.json")


if __name__ == "__main__":
    main()
