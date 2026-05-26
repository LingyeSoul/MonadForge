#!/usr/bin/env python3
"""Representation-space dispersive probe for soft tokens (proposal variant (b)).

Phase 0 of the contrastive proposal FAILED the hard-negative gate (29% strict;
`negative_audit.py`). Variant (b) is the *negative-free* alternative: instead of
building hard negatives we keep dispersing the bank, but in the **representation
space the model actually discriminates the soft tokens in** — the frozen
cross-attention K projection (`net.blocks.{k}.cross_attn.{k_proj,k_norm}`) — not
in raw bank-parameter space the way the shipped `bank_dispersive_loss` does.

This probe tests the one precondition that decides whether (b) is worth wiring:

  **Does parameter-space dispersion actually separate the soft tokens in K
  representation space, or does `k_proj` re-correlate them?**

Random high-D vectors are near-orthogonal in *both* spaces, so an init-from-noise
test is vacuous. We therefore simulate the failure mode the dispersive exists to
fix — **slot collapse** — by initialising the K per-layer tokens as tiny
perturbations around a shared per-layer direction (param-space cosine ≈ 1). Then
two arms each run the *same* bounded dispersive form, differing only in the space
the distance is measured in:

  * ``param``  — disperse raw bank vectors (what `bank_dispersive_loss` does).
  * ``repr``   — disperse the K-projected vectors ``k_norm(k_proj · token)``.

For each arm we log both the param-space and the repr-space mean |cosine| of the
K slots (per layer, averaged). The gate:

  * ``param`` arm drives **repr** |cos| ≈ as low as the ``repr`` arm does
    → param-space dispersion already separates tokens where it matters →
    **(b) redundant**, keep the shipped param-space loss, Phase 1 stays as-is.
  * ``param`` arm leaves **repr** |cos| high while the ``repr`` arm collapses it
    → `k_proj` re-correlates param-separated tokens → **(b) is the lever**, wire
    a ``dispersive_space="representation"`` knob.

Pure linear algebra on the frozen `k_proj` matrices + a tiny bank — **no DiT
forward, no data, no negatives.** Loads only the cross-attn K weights off disk.

Run from anima_lora/::

    uv run python -m bench.soft_tokens_contrastive.repr_dispersive_probe
    uv run python -m bench.soft_tokens_contrastive.repr_dispersive_probe \
        --steps 800 --include_v --no_k_norm
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from bench._common import make_run_dir, write_result

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIT = REPO_ROOT / "models/diffusion_models/anima-base-v1.0.safetensors"

# Match networks/methods/soft_tokens.py + configs/methods/soft_tokens.toml.
DEFAULT_N_LAYERS = 10
DEFAULT_K = 4
DEFAULT_D = 1024


# ─────────────────────────────────────────────────────── representation map


def load_cross_attn_kv(
    dit_path: Path, n_layers: int, *, include_v: bool, include_k_norm: bool
):
    """Read frozen cross-attn K (and optionally V) projections off disk.

    Returns a list of per-layer dicts: ``W`` (D_out, D_in), optional per-head
    ``norm`` weight (head_dim,), ``head_dim``. ``W`` stacks k_proj (and v_proj
    when ``include_v``) row-wise so the representation is the concatenation of
    keys (and values). ``head_dim`` is inferred from the k_norm weight; the
    per-head RMSNorm is applied only to the K block (it is what gates attention).
    """
    from safetensors import safe_open

    layers = []
    with safe_open(str(dit_path), framework="pt") as f:
        keys = set(f.keys())
        for k in range(n_layers):
            base = f"net.blocks.{k}.cross_attn"
            kp = f"{base}.k_proj.weight"
            if kp not in keys:
                raise KeyError(
                    f"{kp} not in {dit_path.name}; expected split cross-attn proj "
                    f"(got e.g. {sorted(x for x in keys if base in x)[:4]})"
                )
            Wk = f.get_tensor(kp).float()  # (inner, D_in)
            knorm = (
                f.get_tensor(f"{base}.k_norm.weight").float()
                if include_k_norm and f"{base}.k_norm.weight" in keys
                else None
            )
            head_dim = int(knorm.shape[0]) if knorm is not None else Wk.shape[0]
            entry = {"Wk": Wk, "k_norm": knorm, "head_dim": head_dim}
            if include_v:
                entry["Wv"] = f.get_tensor(f"{base}.v_proj.weight").float()
            layers.append(entry)
    return layers


def _rms_per_head(x: torch.Tensor, head_dim: int, weight: torch.Tensor) -> torch.Tensor:
    """Per-head RMSNorm over head_dim, scaled by ``weight`` — Anima's k_norm.

    x: (K, inner). Returns (K, inner).
    """
    K, inner = x.shape
    n_heads = inner // head_dim
    xh = x.view(K, n_heads, head_dim)
    xh = xh * torch.rsqrt(xh.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    xh = xh * weight.view(1, 1, head_dim)
    return xh.reshape(K, inner)


def represent(tokens_layer: torch.Tensor, entry: dict) -> torch.Tensor:
    """Project a layer's (K, D_in) tokens into cross-attn K(+V) repr space."""
    rep = tokens_layer @ entry["Wk"].t()  # (K, inner)
    if entry["k_norm"] is not None:
        rep = _rms_per_head(rep, entry["head_dim"], entry["k_norm"])
    if "Wv" in entry:
        v = tokens_layer @ entry["Wv"].t()
        rep = torch.cat([rep, v], dim=-1)
    return rep


# ───────────────────────────────────────────────────────────── dispersive


def _mean_abs_cos(z: torch.Tensor) -> torch.Tensor:
    """Mean |cosine| over the K(K-1)/2 pairs of rows. 1=collapsed, 0=dispersed."""
    if z.shape[0] < 2:
        return z.new_zeros(())
    zn = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
    gram = zn @ zn.t()
    n = gram.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=gram.device)
    return gram[iu[0], iu[1]].abs().mean()


def _dispersive(z: torch.Tensor, form: str, tau: float) -> torch.Tensor:
    """Bounded dispersive on a (K, D') slab. Mirrors soft_tokens.py forms."""
    if z.shape[0] < 2:
        return z.new_zeros(())
    if form == "cosine_sq":
        zn = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
        gram = zn @ zn.t()
        n = gram.shape[0]
        iu = torch.triu_indices(n, n, offset=1, device=gram.device)
        return gram[iu[0], iu[1]].pow(2).mean()
    if form == "normalized_pdist":
        zn = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
        d_sq = torch.pdist(zn, p=2).pow(2)
        return torch.logsumexp(-d_sq / tau, dim=-1) - math.log(float(d_sq.numel()))
    raise ValueError(f"unknown form {form!r}")


# ─────────────────────────────────────────────────────────── spectral stat


def _proj_spectrum(W: torch.Tensor) -> tuple[float, float]:
    """(participation-ratio effective rank, top-1 singular energy share) of W."""
    s = torch.linalg.svdvals(W.float())
    s2 = s.pow(2)
    eff_rank = float((s.sum().pow(2) / s2.sum()).item())  # participation ratio
    top1 = float((s2[0] / s2.sum()).item())
    return eff_rank, top1


# ───────────────────────────────────────────────────────────────── arms


def run_arm(space: str, layers, args, device) -> tuple[list[dict], dict]:
    """Optimise a collapsed bank under ``space``-measured dispersive.

    Returns per-log-point rows and a final-metrics dict. Every step we minimise
    the dispersive in ``space`` ∈ {param, repr} but *always log both* the
    param-space and repr-space mean |cos|, so the cross-space leakage is visible.
    """
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    n_layers = len(layers)
    # Collapsed init: K slots per layer = shared direction + tiny noise → the
    # slot-collapse failure mode the dispersive exists to undo.
    shared = torch.randn(n_layers, 1, args.d, generator=g) * args.init_std
    noise = torch.randn(n_layers, args.k, args.d, generator=g) * (
        args.init_std * args.collapse_noise
    )
    tokens = (shared + noise).to(device).requires_grad_(True)

    opt = torch.optim.Adam([tokens], lr=args.lr)
    Wks = [layer for layer in layers]

    def cos_pair() -> tuple[float, float]:
        pc, rc = [], []
        with torch.no_grad():
            for li in range(n_layers):
                tl = tokens[li]
                pc.append(_mean_abs_cos(tl))
                rc.append(_mean_abs_cos(represent(tl, Wks[li])))
        return (
            float(torch.stack(pc).mean()),
            float(torch.stack(rc).mean()),
        )

    log_every = max(1, args.steps // args.log_points)
    rows: list[dict] = []
    p0, r0 = cos_pair()
    rows.append({"step": 0, "param_abs_cos": p0, "repr_abs_cos": r0})

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = tokens.new_zeros(())
        for li in range(n_layers):
            tl = tokens[li]
            z = tl if space == "param" else represent(tl, Wks[li])
            loss = loss + _dispersive(z, args.form, args.tau)
        loss = loss / n_layers
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == args.steps:
            pc, rc = cos_pair()
            rows.append({"step": step, "param_abs_cos": pc, "repr_abs_cos": rc})

    pf, rf = cos_pair()
    metrics = {
        "space": space,
        "param_abs_cos_init": p0,
        "repr_abs_cos_init": r0,
        "param_abs_cos_final": pf,
        "repr_abs_cos_final": rf,
    }
    return rows, metrics


# ───────────────────────────────────────────────────────────────── main


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dit", type=Path, default=DEFAULT_DIT)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--log_points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--d", type=int, default=DEFAULT_D)
    parser.add_argument("--init_std", type=float, default=0.02)
    parser.add_argument(
        "--collapse_noise",
        type=float,
        default=0.05,
        help="per-slot noise as a fraction of init_std (smaller = tighter collapse)",
    )
    parser.add_argument(
        "--form",
        choices=("normalized_pdist", "cosine_sq"),
        default="cosine_sq",
        help="bounded dispersive form (matches soft_tokens.py)",
    )
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument(
        "--include_v", action="store_true", help="repr = [K; V] not K only"
    )
    parser.add_argument(
        "--no_k_norm",
        dest="include_k_norm",
        action="store_false",
        help="skip per-head k_norm RMSNorm in the repr map",
    )
    parser.add_argument("--label", type=str, default="reprprobe")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.set_defaults(include_k_norm=True)
    args = parser.parse_args()

    if not args.dit.exists():
        raise SystemExit(f"missing DiT checkpoint {args.dit}")
    device = torch.device(args.device)
    run_dir = make_run_dir("soft_tokens_contrastive", label=args.label)

    layers = load_cross_attn_kv(
        args.dit,
        args.n_layers,
        include_v=args.include_v,
        include_k_norm=args.include_k_norm,
    )
    layers = [
        {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in layer.items()}
        for layer in layers
    ]

    # Projection spectral structure — explains the result.
    eff_ranks, top1s = [], []
    for layer in layers:
        er, t1 = _proj_spectrum(layer["Wk"])
        eff_ranks.append(er)
        top1s.append(t1)
    eff_rank = sum(eff_ranks) / len(eff_ranks)
    top1 = sum(top1s) / len(top1s)

    print(
        f"repr-dispersive probe: {args.n_layers} layers, K={args.k}, D={args.d}, "
        f"form={args.form}, k_norm={args.include_k_norm}, +V={args.include_v}"
    )
    print(
        f"  k_proj spectrum: eff_rank≈{eff_rank:.0f}/{layers[0]['Wk'].shape[1]}, "
        f"top-1 energy {top1 * 100:.1f}%"
    )
    print(f"  output: {run_dir}")

    fieldnames = ["arm", "step", "param_abs_cos", "repr_abs_cos"]
    csv_path = run_dir / "per_step.csv"
    arm_metrics: dict[str, dict] = {}
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for space in ("param", "repr"):
            rows, m = run_arm(space, layers, args, device)
            arm_metrics[space] = m
            for r in rows:
                writer.writerow({"arm": space, **r})
            print(
                f"  [{space:5s}] param|cos {m['param_abs_cos_init']:.3f}→"
                f"{m['param_abs_cos_final']:.3f}   "
                f"repr|cos {m['repr_abs_cos_init']:.3f}→{m['repr_abs_cos_final']:.3f}"
            )

    # ── verdict ────────────────────────────────────────────────────────────
    repr_param_arm = arm_metrics["param"]["repr_abs_cos_final"]
    repr_repr_arm = arm_metrics["repr"]["repr_abs_cos_final"]
    # How much repr-space separation the param-space objective leaves on the
    # table vs. dispersing directly in repr space.
    leakage = repr_param_arm - repr_repr_arm
    # (b) is load-bearing when param-space dispersion both (i) fails to drive
    # repr |cos| low in absolute terms and (ii) is clearly beaten by the repr arm.
    b_load_bearing = repr_param_arm > 0.30 and leakage > 0.10
    verdict = "WIRE_B" if b_load_bearing else "B_REDUNDANT"

    metrics = {
        "form": args.form,
        "include_k_norm": args.include_k_norm,
        "include_v": args.include_v,
        "kproj_eff_rank": round(eff_rank, 2),
        "kproj_in_dim": layers[0]["Wk"].shape[1],
        "kproj_top1_energy": round(top1, 4),
        "repr_abs_cos_param_arm": round(repr_param_arm, 4),
        "repr_abs_cos_repr_arm": round(repr_repr_arm, 4),
        "repr_separation_leakage": round(leakage, 4),
        "param_abs_cos_param_arm": round(
            arm_metrics["param"]["param_abs_cos_final"], 4
        ),
        "param_abs_cos_repr_arm": round(arm_metrics["repr"]["param_abs_cos_final"], 4),
        "verdict": verdict,
        "arms": arm_metrics,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=["per_step.csv"],
        device=device,
        label=args.label,
    )

    print()
    print(
        f"  param-arm leaves repr|cos={repr_param_arm:.3f}; "
        f"repr-arm reaches {repr_repr_arm:.3f} (leakage {leakage:+.3f})"
    )
    print(
        f"  VERDICT: {verdict} — "
        + (
            "k_proj re-correlates param-dispersed tokens; disperse in repr space."
            if b_load_bearing
            else "param-space dispersion already separates tokens in K space; "
            "variant (b) buys little — keep the shipped loss."
        )
    )
    print(f"  wrote {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
