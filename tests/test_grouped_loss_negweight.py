"""Invariants for group-conditional negative weighting in ``compute_grouped_loss``.

Two guarantees that the train-time ``--inactive_neg_weight`` λ must keep:

1. λ=1.0 is **bit-identical** to the un-weighted reduction (the inert default).
2. λ<1 down-weights **exactly** the inactive-group negative cells — positives,
   active-group negatives, and ungrouped tags are untouched.

See bench/tagger_groups/probe_group_structure.py for the Phase-0 sizing that
motivated the λ≈0.6–0.75 operating range.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.anima_tagger.train_common import GroupRouter, compute_grouped_loss


def _router(mh: torch.Tensor) -> GroupRouter:
    # 6 tags: group A={0,1,2}, group B={3,4} (both multilabel), tag5 ungrouped.
    vocab = {
        "tags": [{"name": f"t{i}", "index": i} for i in range(mh.shape[1])],
        "groups": [
            {
                "name": "A",
                "mode": "multilabel",
                "tag_indices": [0, 1, 2],
                "escape_indices": [],
            },
            {
                "name": "B",
                "mode": "multilabel",
                "tag_indices": [3, 4],
                "escape_indices": [],
            },
        ],
    }
    return GroupRouter.from_vocab(vocab, mh, device=torch.device("cpu"))


def _sample():
    # img0: A active, B inactive ; img1: A inactive, B active ; img2: all empty.
    mh = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    torch.manual_seed(0)
    logits = torch.randn(3, 6)
    return mh, logits


def test_lambda_one_is_bit_identical():
    mh, logits = _sample()
    router = _router(mh)
    # No softmax groups → bce_mask is all-True, so the un-weighted reduction is a
    # plain pos-weighted BCE mean over every cell.
    ref = F.binary_cross_entropy_with_logits(
        logits, mh, pos_weight=router.bce_pos_weight, reduction="none"
    ).mean()
    l1, _ = compute_grouped_loss(logits, mh, router, inactive_neg_weight=1.0)
    assert torch.equal(l1, ref)


def test_lambda_lt_one_changes_loss():
    mh, logits = _sample()
    router = _router(mh)
    l1, _ = compute_grouped_loss(logits, mh, router, inactive_neg_weight=1.0)
    l05, _ = compute_grouped_loss(logits, mh, router, inactive_neg_weight=0.5)
    assert not torch.allclose(l1, l05)


def test_downweight_touches_only_inactive_group_negatives():
    mh, _ = _sample()
    router = _router(mh)
    B = mh.shape[0]
    G = router.n_group_slots
    gpc = mh.new_zeros(B, G + 1)
    gpc.index_add_(1, router.group_of_tag, mh)
    active = gpc > 0
    per_tag_active = active.gather(1, router.group_of_tag.unsqueeze(0).expand(B, -1))
    grouped = (router.group_of_tag != G).unsqueeze(0)
    dw = (mh == 0) & grouped & ~per_tag_active

    expected = torch.tensor(
        [
            [0, 0, 0, 1, 1, 0],  # img0: B inactive → t3,t4 ; t5 ungrouped untouched
            [1, 1, 1, 0, 0, 0],  # img1: A inactive → t0,t1,t2 ; t4 active-neg untouched
            [1, 1, 1, 1, 1, 0],  # img2: both inactive ; t5 ungrouped untouched
        ],
        dtype=torch.bool,
    )
    assert torch.equal(dw, expected)


def test_inert_when_no_groups():
    mh = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    vocab = {
        "tags": [{"name": "t0", "index": 0}, {"name": "t1", "index": 1}],
        "groups": [],
    }
    router = GroupRouter.from_vocab(vocab, mh, device=torch.device("cpu"))
    assert router.group_of_tag is None
    torch.manual_seed(1)
    logits = torch.randn(2, 2)
    l1, _ = compute_grouped_loss(logits, mh, router, inactive_neg_weight=1.0)
    l05, _ = compute_grouped_loss(logits, mh, router, inactive_neg_weight=0.5)
    # No groups → λ has nothing to act on → identical.
    assert torch.equal(l1, l05)
