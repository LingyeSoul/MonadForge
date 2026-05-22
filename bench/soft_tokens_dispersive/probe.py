"""Parameter-only probe for the soft-tokens bank dispersive regularizer.

No DiT, no data, no FM signal — we just allocate the bank exactly as
``SoftTokensNetwork.__init__`` does (n_layers × K × D base tokens, plus the
n_t_buckets × n_layers × D t-offsets) and run an optimizer that minimizes the
dispersive term *alone*. The point is to see what each parameterization does
to the bank on its own, before paying for a real training run.

Per step we log: variant loss, mean bank vector norm, mean pairwise cosine
(K-axis on base tokens, averaged across layers), and min / mean pairwise
distance². A healthy "don't collapse" prior should reach a bounded loss,
push mean cosine off 1 toward 0 (or some non-degenerate value), and not let
norms grow unboundedly.

Variants compared:

- ``current``                — ``log(mean(exp(-d²/τ)))``, τ as currently shipped (0.5).
                                Should run to −∞ with unbounded norm growth.
- ``current_tau5``           — same shape, τ=5.
- ``current_tau50``          — same shape, τ=50.
- ``normalized_pdist``       — unit-normalize each vector first, then current
                                form with τ=0.5. d² ∈ [0, 4], norms can't run.
- ``cosine_abs``             — mean |cos(v_i, v_j)| over pairs. Bounded [0, 1].
- ``cosine_sq``              — mean cos(v_i, v_j)² over pairs. Bounded [0, 1].
- ``hinge``                  — mean max(0, ε − d²) over pairs. Bounded below by 0.

Run from anima_lora/::

    uv run python -m bench.soft_tokens_dispersive.probe
    uv run python -m bench.soft_tokens_dispersive.probe --steps 1000 --lr 3e-4
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass

import torch

from bench._common import make_run_dir, write_result


# Match networks/methods/soft_tokens.py defaults + configs/methods/soft_tokens.toml.
DEFAULT_N_LAYERS = 10
DEFAULT_K = 4
DEFAULT_N_T_BUCKETS = 14
DEFAULT_D = 1024
DEFAULT_INIT_STD = 0.02
DEFAULT_TAU = 0.5
DEFAULT_HINGE_EPS = 1.0


# ─────────────────────────────────────────────────────────── loss variants


def _pdist_sq(z: torch.Tensor) -> torch.Tensor:
    """Squared pairwise distances on the leading axis of a 2D tensor."""
    return torch.pdist(z, p=2).pow(2)


def _logmeanexp_neg(d_sq: torch.Tensor, tau: float) -> torch.Tensor:
    """``log(mean(exp(-d²/τ)))`` via logsumexp − log N, numerically stable."""
    if d_sq.numel() == 0:
        return d_sq.new_zeros(())
    return torch.logsumexp(-d_sq / tau, dim=-1) - math.log(float(d_sq.numel()))


def _mean_over_axes(per_term_fn, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Apply ``per_term_fn`` to each (K, D) layer slab of tokens AND each
    (n_buckets, D) layer slab of offsets, then average.

    tokens: (n_layers, K, D); offsets: (n_buckets, n_layers, D).
    Mirrors ``SoftTokensNetwork.bank_dispersive_loss`` axis structure.
    """
    n_layers = tokens.shape[0]
    terms = []
    if tokens.shape[1] >= 2:
        for k in range(n_layers):
            terms.append(per_term_fn(tokens[k]))
    if offsets.shape[0] >= 2:
        for k in range(n_layers):
            terms.append(per_term_fn(offsets[:, k, :]))
    if not terms:
        return tokens.new_zeros(())
    return torch.stack(terms).mean()


def loss_current(tokens, offsets, *, tau):
    return _mean_over_axes(lambda z: _logmeanexp_neg(_pdist_sq(z), tau), tokens, offsets)


def loss_normalized_pdist(tokens, offsets, *, tau):
    def _term(z):
        zn = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
        return _logmeanexp_neg(_pdist_sq(zn), tau)

    return _mean_over_axes(_term, tokens, offsets)


def _abs_cosine_pairs(z: torch.Tensor) -> torch.Tensor:
    zn = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
    gram = zn @ zn.t()
    n = gram.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=gram.device)
    return gram[iu[0], iu[1]].abs()


def _sq_cosine_pairs(z: torch.Tensor) -> torch.Tensor:
    zn = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
    gram = zn @ zn.t()
    n = gram.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=gram.device)
    return gram[iu[0], iu[1]].pow(2)


def loss_cosine_abs(tokens, offsets):
    return _mean_over_axes(lambda z: _abs_cosine_pairs(z).mean(), tokens, offsets)


def loss_cosine_sq(tokens, offsets):
    return _mean_over_axes(lambda z: _sq_cosine_pairs(z).mean(), tokens, offsets)


def loss_hinge(tokens, offsets, *, eps):
    def _term(z):
        d_sq = _pdist_sq(z)
        if d_sq.numel() == 0:
            return z.new_zeros(())
        return torch.clamp(eps - d_sq, min=0.0).mean()

    return _mean_over_axes(_term, tokens, offsets)


# ──────────────────────────────────────────────────────────── probe loop


@dataclass
class Variant:
    name: str
    fn: callable
    note: str


def build_variants(tau: float, hinge_eps: float) -> list[Variant]:
    return [
        Variant(
            "current_tau0_5",
            lambda t, o: loss_current(t, o, tau=tau),
            f"shipped form, τ={tau}",
        ),
        Variant(
            "current_tau5",
            lambda t, o: loss_current(t, o, tau=5.0),
            "shipped form, τ=5",
        ),
        Variant(
            "current_tau50",
            lambda t, o: loss_current(t, o, tau=50.0),
            "shipped form, τ=50",
        ),
        Variant(
            "normalized_pdist_tau0_5",
            lambda t, o: loss_normalized_pdist(t, o, tau=tau),
            "unit-normalize then pdist, τ=0.5 (d² ∈ [0,4])",
        ),
        Variant(
            "cosine_abs",
            loss_cosine_abs,
            "mean |cos(v_i, v_j)| over pairs (bounded [0, 1])",
        ),
        Variant(
            "cosine_sq",
            loss_cosine_sq,
            "mean cos(v_i, v_j)² over pairs (bounded [0, 1])",
        ),
        Variant(
            "hinge",
            lambda t, o: loss_hinge(t, o, eps=hinge_eps),
            f"mean max(0, ε−d²), ε={hinge_eps} (bounded ≥ 0)",
        ),
    ]


def init_bank(args, device, dtype, generator):
    tokens = torch.empty(
        args.n_layers, args.k, args.d, device=device, dtype=dtype
    )
    tokens.normal_(mean=0.0, std=args.init_std, generator=generator)
    # t_offsets is zero-init in the live network — matches SoftTokensNetwork.
    offsets = torch.zeros(
        args.n_buckets, args.n_layers, args.d, device=device, dtype=dtype
    )
    # Small jitter so cosine isn't NaN at step 0.
    offsets.normal_(mean=0.0, std=args.init_std * 1e-3, generator=generator)
    tokens.requires_grad_(True)
    offsets.requires_grad_(True)
    return tokens, offsets


@torch.no_grad()
def diagnose(tokens: torch.Tensor, offsets: torch.Tensor) -> dict[str, float]:
    """Variant-agnostic diagnostics on the *bank* state."""
    n_layers = tokens.shape[0]
    bank_norm = tokens.flatten(1).norm(dim=-1).mean().item()
    offset_norm = (
        offsets.permute(1, 0, 2).flatten(1).norm(dim=-1).mean().item()
    )

    cos_per_layer = []
    d_min_per_layer = []
    d_mean_per_layer = []
    if tokens.shape[1] >= 2:
        for k in range(n_layers):
            z = tokens[k]
            zn = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
            gram = zn @ zn.t()
            n = gram.shape[0]
            iu = torch.triu_indices(n, n, offset=1, device=gram.device)
            cos_per_layer.append(gram[iu[0], iu[1]].mean().item())
            d_sq = _pdist_sq(z)
            d_min_per_layer.append(d_sq.min().item())
            d_mean_per_layer.append(d_sq.mean().item())

    return {
        "bank_norm": bank_norm,
        "offset_norm": offset_norm,
        "tokens_mean_cos": float(sum(cos_per_layer) / max(len(cos_per_layer), 1)),
        "tokens_min_d_sq": float(min(d_min_per_layer)) if d_min_per_layer else 0.0,
        "tokens_mean_d_sq": float(sum(d_mean_per_layer) / max(len(d_mean_per_layer), 1)),
    }


def run_variant(variant: Variant, args, device, dtype):
    """Run one variant. Returns (per_step rows, final metrics)."""
    generator = torch.Generator(device=device).manual_seed(args.seed)
    tokens, offsets = init_bank(args, device, dtype, generator)
    optim = torch.optim.Adam([tokens, offsets], lr=args.lr)

    rows = []
    log_every = max(1, args.steps // args.log_points)
    for step in range(args.steps + 1):
        loss = variant.fn(tokens, offsets)
        loss_val = float(loss.detach().item())

        if step % log_every == 0 or step == args.steps:
            diag = diagnose(tokens, offsets)
            rows.append({"step": step, "loss": loss_val, **diag})

        if step == args.steps:
            break

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    final = rows[-1]
    init = rows[0]

    # Boundedness heuristic: did loss / norm change at a small rate over the
    # last 25% of steps, or are they still moving?
    tail_start = max(0, int(len(rows) * 0.75))
    tail = rows[tail_start:]
    loss_tail_slope = (tail[-1]["loss"] - tail[0]["loss"]) / max(len(tail) - 1, 1)
    norm_tail_slope = (
        tail[-1]["bank_norm"] - tail[0]["bank_norm"]
    ) / max(len(tail) - 1, 1)

    metrics = {
        "loss_init": init["loss"],
        "loss_final": final["loss"],
        "loss_tail_slope_per_log_point": loss_tail_slope,
        "bank_norm_init": init["bank_norm"],
        "bank_norm_final": final["bank_norm"],
        "bank_norm_tail_slope_per_log_point": norm_tail_slope,
        "tokens_mean_cos_init": init["tokens_mean_cos"],
        "tokens_mean_cos_final": final["tokens_mean_cos"],
        "tokens_min_d_sq_final": final["tokens_min_d_sq"],
        "tokens_mean_d_sq_final": final["tokens_mean_d_sq"],
        "offset_norm_final": final["offset_norm"],
    }
    return rows, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_points", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n_buckets", type=int, default=DEFAULT_N_T_BUCKETS)
    parser.add_argument("--d", type=int, default=DEFAULT_D)
    parser.add_argument("--init_std", type=float, default=DEFAULT_INIT_STD)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--hinge_eps", type=float, default=DEFAULT_HINGE_EPS)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32  # bank is tiny; fp32 keeps diagnostics clean
    run_dir = make_run_dir("soft_tokens_dispersive", label=args.label or "probe")

    variants = build_variants(tau=args.tau, hinge_eps=args.hinge_eps)
    print(f"Probe: {len(variants)} variants × {args.steps} steps on {device}")
    print(f"Bank: n_layers={args.n_layers}, K={args.k}, n_buckets={args.n_buckets}, D={args.d}")
    print(f"Output: {run_dir}")
    print()

    all_metrics: dict[str, dict] = {}
    fieldnames = [
        "variant", "step", "loss", "bank_norm", "offset_norm",
        "tokens_mean_cos", "tokens_min_d_sq", "tokens_mean_d_sq",
    ]
    csv_path = run_dir / "per_step.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in variants:
            rows, metrics = run_variant(v, args, device, dtype)
            all_metrics[v.name] = {"note": v.note, **metrics}
            for r in rows:
                writer.writerow({"variant": v.name, **r})
            print(
                f"{v.name:28s} "
                f"loss {metrics['loss_init']:+9.3f} → {metrics['loss_final']:+9.3f}  "
                f"norm {metrics['bank_norm_init']:.3f} → {metrics['bank_norm_final']:.3f}  "
                f"cos {metrics['tokens_mean_cos_init']:+.3f} → {metrics['tokens_mean_cos_final']:+.3f}  "
                f"min_d² {metrics['tokens_min_d_sq_final']:.3e}"
            )

    print()
    print("Verdict heuristics:")
    print("  - 'unbounded' if loss tail-slope is large negative AND norm grows")
    print("  - 'collapsed' if final mean_cos > 0.9 or final min_d² < 1e-4")
    print("  - 'dispersed' if final mean_cos near 0 and norm did not blow up")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=all_metrics,
        artifacts=["per_step.csv"],
        device=device,
        label=args.label,
    )
    print()
    print(f"Wrote {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
