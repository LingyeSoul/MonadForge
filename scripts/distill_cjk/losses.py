"""The four distillation objectives compared by the Phase-2b G2 ablation.

``attn``  Set-level readout through the DiT's own cross-attention geometry,
          zero-pad sink included. The faithful one — see ``attn_bank`` for why
          the interface is a *set plus a length*, not a sequence.
``span``  Tag-level supervision using the alignment that composition makes
          free (each EN tag ↔ one JA tag, same order). Local gradient: an ext
          row is supervised by *its own tag* rather than by a sequence
          average, which is the sample-efficiency lever at ~10⁴ pairs. Also the
          only loss that can carry a per-span trust weight, so the 34% of tag
          occurrences that come from `mt_unverified` can be demoted instead of
          dropped.
``flat``  Position-wise cosine over the padded outputs. Control. Well-defined
          only insofar as the two arms' token counts coincide; it compares
          position i of the EN teacher with position i of the JA student.
``pool``  Mean-pooled sequence statistic. Weakest control.

Every loss returns ``1 - cosine`` (per-sample, then averaged) so the arms are
on one scale and the G2 cross-tab is readable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.distill_cjk.attn_bank import readout


def zero_pads(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mirror the inference boundary: non-mask rows are exactly zero."""
    return x * mask.unsqueeze(-1).to(x.dtype)


def _cos_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - cosine over everything but dim 0 (per-sample, then mean)."""
    a = a.reshape(a.shape[0], -1).float()
    b = b.reshape(b.shape[0], -1).float()
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()


def _pad_to(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[1] >= length:
        return x[:, :length]
    return F.pad(x, (0, 0, 0, length - x.shape[1]))


def loss_flat(student, s_mask, teacher, t_mask) -> torch.Tensor:
    L = max(student.shape[1], teacher.shape[1])
    return _cos_loss(
        _pad_to(zero_pads(student, s_mask), L), _pad_to(zero_pads(teacher, t_mask), L)
    )


def loss_pool(student, s_mask, teacher, t_mask) -> torch.Tensor:
    s = zero_pads(student, s_mask).sum(1) / s_mask.sum(1, keepdim=True).clamp_min(1)
    t = zero_pads(teacher, t_mask).sum(1) / t_mask.sum(1, keepdim=True).clamp_min(1)
    return _cos_loss(s, t)


def _segment_mean(values: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
    """Mean of ``values`` grouped by ``seg`` — one index_add, not a Python loop."""
    out = values.new_zeros(n_seg, values.shape[-1])
    out.index_add_(0, seg, values)
    count = values.new_zeros(n_seg).index_add_(
        0, seg, torch.ones_like(seg, dtype=values.dtype)
    )
    return out / count.clamp_min(1).unsqueeze(-1)


def loss_span(student, teacher, pack) -> torch.Tensor:
    """Weighted mean of ``1 - cos`` over aligned (EN tag, JA tag) span pools.

    ``pack`` is the flattened index set built by ``data.collate`` — pooling all
    spans in the batch at once keeps the loop compute-bound instead of
    launch-bound (a per-span Python loop was ~1600 kernels per step).
    """
    if not pack:
        return student.new_zeros((), dtype=torch.float32)
    n = pack["n_spans"]
    s = _segment_mean(
        student.reshape(-1, student.shape[-1]).float()[pack["s_flat"]],
        pack["s_seg"],
        n,
    )
    t = _segment_mean(
        teacher.reshape(-1, teacher.shape[-1]).float()[pack["t_flat"]],
        pack["t_seg"],
        n,
    )
    w = pack["w"]
    return (w * (1.0 - F.cosine_similarity(s, t, dim=-1))).sum() / w.sum().clamp_min(
        1e-8
    )


def loss_attn(student, s_mask, teacher, t_mask, probes, seq_total: int = 512):
    out = student.new_zeros((), dtype=torch.float32)
    for probe in probes:
        rs = readout(zero_pads(student, s_mask), s_mask, probe, seq_total=seq_total)
        rt = readout(zero_pads(teacher, t_mask), t_mask, probe, seq_total=seq_total)
        out = out + _cos_loss(rs, rt)
    return out / max(len(probes), 1)


def compute(
    weights: dict[str, float],
    *,
    student,
    s_mask,
    teacher,
    t_mask,
    spans=None,
    probes=None,
    seq_total: int = 512,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum plus a per-term readout for the log / result envelope."""
    parts: dict[str, torch.Tensor] = {}
    if "flat" in weights:
        parts["flat"] = loss_flat(student, s_mask, teacher, t_mask)
    if "pool" in weights:
        parts["pool"] = loss_pool(student, s_mask, teacher, t_mask)
    if "span" in weights:
        if spans is None:
            raise ValueError("span loss selected but the batch carries no spans")
        parts["span"] = loss_span(student, teacher, spans)
        # `student` may be zero-padded per position but the pooled spans index
        # into the flattened view, so the mask is already respected by
        # construction — the pack only ever addresses non-pad positions.
    if "attn" in weights:
        if not probes:
            raise ValueError("attn loss selected but the probe bank is empty")
        parts["attn"] = loss_attn(
            student, s_mask, teacher, t_mask, probes, seq_total=seq_total
        )
    total = sum(weights[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
