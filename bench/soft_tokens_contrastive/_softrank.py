"""Dependency-free differentiable soft-rank / soft-sort (vendored subset).

This is the ~20-line core of what ``softtorch`` (a-paulus/softtorch, Apache-2.0,
arXiv:2603.08824) packages as ``st.rank`` / ``st.sort`` / ``st.topk`` — vendored
here so the AGSM gradient-quality probe can A/B a *native-gradient* listwise
ranking loss against the current detached-Plackett–Luce target-regression WITHOUT
taking softtorch's full dependency footprint (numba + POT + torchopt). If Tier-A
(this probe) and a Tier-B live A/B both win, swapping to the real package is a
one-line change; until then this keeps the experiment self-contained.

The relaxations are the standard ones:

  * ``soft_rank``  — rank_i ≈ Σ_{j≠i} σ((s_j − s_i)/τ). The differentiable count
    of "how many entries out-score i" (pairwise-sigmoid relaxation of the rank
    operator). Bounded in [0, n−1]; τ→0 recovers the hard integer rank.
  * ``soft_sort``  — a NeuralSort/SoftSort-style row-softmax permutation applied
    to the values (Prillo & Eisenschlos, "SoftSort", ICML 2020), provided for
    completeness; the probe only needs ``soft_rank``.

Both are deterministic (no Gumbel sampling) and temperature-controlled, which is
exactly the property that lets a ranking loss stay *bounded* — the contrast that
matters versus InfoNCE's unbounded negative push (the whole reason AGSM exists).
"""

from __future__ import annotations

import torch


def soft_rank(scores: torch.Tensor, tau: float = 0.1, dim: int = -1) -> torch.Tensor:
    """Differentiable rank of each entry (0 = best/highest score).

    ``rank_i = Σ_{j≠i} sigmoid((s_j − s_i) / tau)`` — the soft count of entries
    that out-score ``i``. Higher ``scores`` ⇒ lower (better) rank. Shape-preserving
    along ``dim``. As ``tau → 0`` this converges to the integer rank; larger ``tau``
    spreads gradient onto near-ties (the near-miss credit a detached argsort can't
    deliver).
    """
    s = scores.transpose(dim, -1)
    diff = s.unsqueeze(-1) - s.unsqueeze(-2)  # diff[..., i, j] = s_i − s_j
    # σ((s_j − s_i)/τ) summed over j (the diagonal j=i contributes σ(0)=0.5 and is
    # subtracted off so a unique max gets rank 0).
    r = torch.sigmoid(-diff / max(float(tau), 1e-6)).sum(dim=-1) - 0.5
    return r.transpose(dim, -1)


def soft_sort(values: torch.Tensor, tau: float = 0.1, dim: int = -1) -> torch.Tensor:
    """SoftSort-style descending soft-sort of ``values`` along ``dim``.

    Returns values in (soft) descending order via a row-softmax permutation
    ``P = softmax(−|sort(v)ᵀ − v| / tau)`` applied to ``v`` (Prillo & Eisenschlos,
    ICML 2020). Provided for completeness / parity with ``softtorch.sort``; the
    AGSM probe uses ``soft_rank`` only.
    """
    v = values.transpose(dim, -1)
    sorted_desc = torch.sort(v, dim=-1, descending=True).values
    pairwise = -(sorted_desc.unsqueeze(-1) - v.unsqueeze(-2)).abs()
    perm = torch.softmax(pairwise / max(float(tau), 1e-6), dim=-1)
    out = perm @ v.unsqueeze(-1)
    return out.squeeze(-1).transpose(dim, -1)
