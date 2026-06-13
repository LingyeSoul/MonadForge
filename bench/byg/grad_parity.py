#!/usr/bin/env python3
"""BYG gradient-parity probe: ``unsloth_offload_checkpointing`` vs plain checkpoint.

BYG runs the DiT blocks through one of two activation checkpointers (selected by
``unsloth_offload_checkpointing`` in ``configs/methods/byg.toml`` /
``_make_byg_block_forward``):

    unsloth=True  -> library.anima.models.unsloth_checkpoint   (REENTRANT)
    unsloth=False -> torch.utils.checkpoint(use_reentrant=False) (NON-reentrant)

The reentrant path has a well-known footgun: it only builds an autograd node
when **at least one explicit input tensor requires grad**. When the only
grad-requiring tensors are the *closed-over* LoRA parameters (not passed as
checkpoint args), the node is never created and those params receive **no
gradient** — silently. ``use_reentrant=False`` tracks closed-over params and
does not have this problem.

That maps directly onto BYG's three grad-bearing passes (networks/methods/byg.py
``BYGMethodAdapter.compute_loss``), classified by whether *any* block input
requires grad:

    Forward  -> L_prior : target=y_t (detached), source=x (frozen)      -> NEITHER
    Identity -> L_id    : target=x_t (no grad),  source=x (frozen)      -> NEITHER
    Cycle    -> L_cycle : target=x_t (no grad),  source=y_hyb (STE)     -> source DOES

Prediction: under the reentrant (unsloth) path the LoRA params get ZERO grad in
the Forward and Identity scenarios (so L_prior / L_id never train the adapter),
while the Cycle scenario trains because its source input requires grad. The
non-reentrant path trains the params in all three. This script reproduces the
mechanism on a faithful two-stream mock (shared params, cross-stream attention,
source threaded block->block, two outputs — mirroring ``_two_stream_inner``) so
it runs on CPU in milliseconds and does not contend with a live GPU training run.

Run:  python bench/byg/grad_parity.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

# bench/ is not an installed package — bootstrap repo root onto sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library.anima.models import unsloth_checkpoint  # noqa: E402


class TwoStreamBlock(nn.Module):
    """Faithful mock of BYG's ``_two_stream_inner``.

    Shared ``qkv``/``out``/``mlp`` linears stand in for the LoRA-patched block
    linears: the *target* stream runs extended self-attention over
    ``[target_k; src_k]`` and the *source* stream runs its own self-attention,
    both through the same parameters — so every param is exercised by both
    streams, exactly like the real two-stream block. ``inner`` is what gets
    handed to the checkpointer; it takes the two streams as args and returns
    both (the only closed-over grad tensors are the module params).
    """

    def __init__(self, d: int, is_last: bool = False):
        super().__init__()
        self.d = d
        self.is_last = is_last
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.mlp = nn.Linear(d, d, bias=False)

    def inner(self, target, src):
        scale = 1.0 / math.sqrt(self.d)
        tq, tk, tv = self.qkv(target).chunk(3, dim=-1)
        sq, sk, sv = self.qkv(src).chunk(3, dim=-1)

        # target attends over [target; src] (extended self-attn)
        k = torch.cat([tk, sk], dim=1)
        v = torch.cat([tv, sv], dim=1)
        a = torch.softmax(tq @ k.transpose(-1, -2) * scale, dim=-1) @ v
        target = target + self.out(a)
        target = target + self.mlp(F.relu(target))

        if not self.is_last:
            sa = torch.softmax(sq @ sk.transpose(-1, -2) * scale, dim=-1) @ sv
            src = src + self.out(sa)
            src = src + self.mlp(F.relu(src))
        return target, src


def build_blocks(n: int, d: int, seed: int = 0) -> nn.ModuleList:
    g = torch.Generator().manual_seed(seed)
    blocks = nn.ModuleList(
        [TwoStreamBlock(d, is_last=(i == n - 1)) for i in range(n)]
    )
    # deterministic init (double precision for exact comparison)
    for p in blocks.parameters():
        p.data = torch.empty_like(p, dtype=torch.float64).uniform_(-0.1, 0.1, generator=g)
    return blocks


def run(blocks, target0, src0, variant: str):
    """One forward+backward through the chained blocks; return param grads."""
    for p in blocks.parameters():
        p.grad = None
    t, s = target0.clone(), src0.clone()
    t.requires_grad_(target0.requires_grad)
    s.requires_grad_(src0.requires_grad)

    for blk in blocks:
        if variant == "unsloth":
            t, s = unsloth_checkpoint(blk.inner, t, s)
        elif variant == "nonreentrant":
            t, s = torch_checkpoint(blk.inner, t, s, use_reentrant=False)
        elif variant == "eager":
            t, s = blk.inner(t, s)
        else:
            raise ValueError(variant)

    loss = (t.float() ** 2).mean() + (s.float() ** 2).mean()
    # Under reentrant checkpointing with no grad-requiring input, the output has
    # no grad_fn at all — backward would raise. Treat that as "grad fully
    # dropped" (the strongest form of the failure) rather than crashing.
    no_grad_path = not loss.requires_grad
    if not no_grad_path:
        loss.backward()
    grads = {
        name: (p.grad.detach().clone() if p.grad is not None else None)
        for name, p in blocks.named_parameters()
    }
    return float(loss.detach()), grads, no_grad_path


def grad_total(grads) -> float:
    tot = 0.0
    for g in grads.values():
        if g is not None:
            tot += float(g.abs().sum())
    return tot


def max_divergence(ga, gb) -> float:
    m = 0.0
    for name in ga:
        a, b = ga[name], gb[name]
        if a is None and b is None:
            continue
        if a is None:
            m = max(m, float(b.abs().max()))
        elif b is None:
            m = max(m, float(a.abs().max()))
        else:
            m = max(m, float((a - b).abs().max()))
    return m


SCENARIOS = [
    # (label, BYG pass, target_requires_grad, source_requires_grad)
    ("Forward  -> L_prior", "y_t detached, x frozen", False, False),
    ("Identity -> L_id   ", "x_t no-grad,  x frozen", False, False),
    ("Cycle    -> L_cycle", "x_t no-grad,  y_hyb STE", False, True),
    ("sanity (both grad) ", "both inputs require grad", True, True),
]


def main():
    torch.manual_seed(0)
    n, d, B, L = 4, 16, 2, 5
    base = torch.randn(B, L, d, dtype=torch.float64)
    src_base = torch.randn(B, L, d, dtype=torch.float64)

    print(f"{'scenario':<22} {'inputs':<24} {'eager':>10} {'nonreent':>10} "
          f"{'unsloth':>10}   verdict")
    print("-" * 92)

    any_fail = False
    for label, inputs, tg, sg in SCENARIOS:
        blocks = build_blocks(n, d, seed=1)

        target0 = base.clone().requires_grad_(tg)
        src0 = src_base.clone().requires_grad_(sg)

        _, g_eager, _ = run(blocks, target0, src0, "eager")
        _, g_nonre, _ = run(blocks, target0, src0, "nonreentrant")
        _, g_unsl, unsl_no_path = run(blocks, target0, src0, "unsloth")

        te, tn, tu = grad_total(g_eager), grad_total(g_nonre), grad_total(g_unsl)

        # reference is eager autograd; non-reentrant must match it
        ref_div = max_divergence(g_eager, g_nonre)
        unsl_div = max_divergence(g_eager, g_unsl)

        unsl_dropped = (tu < 1e-12 and te > 1e-9) or unsl_no_path
        ok = unsl_div < 1e-6 and not unsl_dropped
        if not ok:
            any_fail = True
        drop_msg = "FAIL: unsloth drops ALL param grad" + (
            " (loss has no grad_fn)" if unsl_no_path else ""
        )
        verdict = "OK" if ok else (
            drop_msg if unsl_dropped
            else f"FAIL: diverges (max {unsl_div:.2e})"
        )
        # non-reentrant sanity
        if ref_div > 1e-6:
            verdict += f"  [!! non-reentrant also diverges {ref_div:.2e}]"

        print(f"{label:<22} {inputs:<24} {te:>10.4f} {tn:>10.4f} {tu:>10.4f}   {verdict}")

    print("-" * 92)
    print(
        "\nInterpretation:\n"
        "  'eager'/'nonreent' = correct gradient (closed-over params trained).\n"
        "  'unsloth' grad-total == 0 on a scenario  =>  that BYG loss does NOT\n"
        "  train the LoRA under unsloth_offload_checkpointing=true.\n"
    )
    if any_fail:
        print("RESULT: PARITY BROKEN — unsloth_offload_checkpointing is unsafe for BYG\n"
              "        (silently kills grad on passes with no grad-requiring input:\n"
              "         Forward/L_prior and Identity/L_id). Use the non-reentrant path.")
    else:
        print("RESULT: parity holds — unsloth path is safe for BYG.")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
