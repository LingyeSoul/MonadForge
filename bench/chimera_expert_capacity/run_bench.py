#!/usr/bin/env python
"""Per-expert capability of the frozen-Cayley chimera levers.

Question: do ``expert_basis_mult`` (over-complete pool + Stiefel select) and
``expert_diag`` (per-expert diagonal) make each ortho expert MORE expressive
than the canonical r-slice — without freeing the bases (the ``use_ortho_init``
collapse mode the user observed at ~4k steps)?

Probe: fix a non-uniform routing gate, then fit a synthetic target ΔW by Adam
on the module's trainable params only (the frozen bases stay frozen). Lower
residual = a richer reachable ΔW per expert. We report the fit residual for:
  * m=1, diag off  — baseline (rotation within a fixed r-slice; magnitude = λ)
  * m=1, diag on   — + per-expert singular spectrum
  * m=2, diag on   — + trainable r-dim subspace within a 2r pool (Stiefel)

Cross-expert orthogonality of the fitted ups is reported alongside, to confirm
the expander does not buy expressivity by collapsing experts together.

CPU, a few hundred Adam steps — runs in seconds. Drops the standard envelope.
"""

from __future__ import annotations

import argparse
from unittest import mock

import torch

from bench._common import make_run_dir, write_result
from networks.lora_modules.chimera import ChimeraHydraLoRAModule

OUT_F, IN_F, R, K_C, K_F = 128, 64, 4, 2, 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default=None)
    return p.parse_args()


def _build(basis_mult, diag, seed):
    torch.manual_seed(seed)
    base = torch.nn.Linear(IN_F, OUT_F, bias=False).to(torch.bfloat16)
    base.weight.requires_grad_(False)
    with mock.patch("torch.cuda.is_available", return_value=False):
        m = ChimeraHydraLoRAModule(
            "m",
            base,
            multiplier=1.0,
            lora_dim=R,
            alpha=R,
            num_experts_content=K_C,
            num_experts_freq=K_F,
            lambda_init=0.1,
            expert_basis_mult=basis_mult,
            expert_diag=diag,
        )
    m.apply_to()
    # Fixed non-uniform gate so the centered combine has live per-expert terms.
    m.set_content_routing_weights(torch.tensor([[0.7, 0.3]]))
    m.set_freq_routing_weights(torch.tensor([[0.6, 0.4]]))
    return m


def _delta_pred(m, eye):
    """ΔW_pred (out, in) = (module(I) - base(I))^T."""
    out = m(eye) - m.org_forward(eye)  # (in, out)
    return out.float().T


def _cross_cos(m):
    """Max cross-expert column cosine of the content ups (0 = disjoint)."""
    with torch.no_grad():
        eye = torch.eye(IN_F, dtype=torch.bfloat16)
        _ = _delta_pred(m, eye)  # warm the rotation
    # Reconstruct effective content ups from the live params.
    r = R
    skew = m.S_p_c
    A = skew - skew.transpose(-2, -1)
    eye_m = torch.eye(m._M)
    R_p = torch.linalg.solve(eye_m + A, eye_m - A)  # (K_c, M, M)
    P = (m.P_bases_c.float() @ R_p)[..., :r]  # (K_c, out, r)
    if hasattr(m, "sigma_c"):
        P = P * m.sigma_c.detach().unsqueeze(1)
    cols = P.permute(0, 2, 1).reshape(K_C * r, OUT_F)
    cols = cols / cols.norm(dim=1, keepdim=True).clamp_min(1e-8)
    g = cols @ cols.T
    block = torch.zeros_like(g, dtype=torch.bool)
    for k in range(K_C):
        block[k * r : (k + 1) * r, k * r : (k + 1) * r] = True
    return g[~block].abs().max().item()


def _fit(m, target, steps, lr):
    eye = torch.eye(IN_F, dtype=torch.bfloat16)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    t0 = (_delta_pred(m, eye) - target).pow(2).mean().item()
    for _ in range(steps):
        opt.zero_grad()
        loss = (_delta_pred(m, eye) - target).pow(2).mean()
        loss.backward()
        opt.step()
    return t0, loss.item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    # Target: the effective ΔW of a randomly-rotated over-complete (m=2, diag)
    # teacher. By construction it lives in the experts' 2r pools using directions
    # OUTSIDE the canonical r-slice — representable by an m=2 student, only
    # partially by an m=1 student. This isolates the subspace+magnitude DOF the
    # levers add (a full-rank random target is unreachable by any rank-K·r
    # adapter, so it can't separate the configs).
    # Same seed as the students ⇒ identical frozen U/V bases, so the m=2
    # student shares the teacher's pools and the contrast is purely the
    # parameterization's reach, not a different base weight.
    teacher = _build(2, True, args.seed)
    with torch.no_grad():
        for p in (teacher.S_p_c, teacher.S_p_f, teacher.S_q_c, teacher.S_q_f):
            p.copy_(torch.randn_like(p) * 0.6)  # large rotation → off the r-slice
        teacher.sigma_c.copy_(1.0 + torch.randn_like(teacher.sigma_c) * 0.4)
        teacher.sigma_f.copy_(1.0 + torch.randn_like(teacher.sigma_f) * 0.4)
        teacher.lambda_c.copy_(torch.randn_like(teacher.lambda_c) * 0.4)
        teacher.lambda_f.copy_(torch.randn_like(teacher.lambda_f) * 0.4)
        target = _delta_pred(teacher, torch.eye(IN_F, dtype=torch.bfloat16)).detach()

    configs = [
        ("m1_nodiag", 1, False),
        ("m1_diag", 1, True),
        ("m2_diag", 2, True),
    ]
    results = {}
    for name, mult, diag in configs:
        m = _build(mult, diag, args.seed)
        t0, tf = _fit(m, target, args.steps, args.lr)
        results[name] = {
            "init_mse": t0,
            "final_mse": tf,
            "rel_residual": tf / t0,
            "cross_expert_cos_max": _cross_cos(m),
            "M": int(m._M),
        }
        print(
            f"{name:12s} final_mse={tf:.3e} rel={tf / t0:.3f} "
            f"cross_cos={results[name]['cross_expert_cos_max']:.3f}"
        )

    base_mse = results["m1_nodiag"]["final_mse"]
    metrics = {
        "configs": results,
        "diag_gain": base_mse / results["m1_diag"]["final_mse"],
        "overcomplete_gain": base_mse / results["m2_diag"]["final_mse"],
        "steps": args.steps,
    }
    run_dir = make_run_dir("chimera_expert_capacity", label=args.label)
    write_result(
        run_dir,
        script=__file__,
        args=vars(args),
        metrics=metrics,
        device="cpu",
    )
    print(
        f"\ndiag gain × {metrics['diag_gain']:.2f}, "
        f"overcomplete+diag gain × {metrics['overcomplete_gain']:.2f}"
    )
    print(f"wrote {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
