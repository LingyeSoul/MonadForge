"""MOSS trainer: the per-step caption choice (CPU, no model)."""

import random

import torch

from library.moss.train import choose_embeds


def _rows(n):
    return torch.zeros(n, 4)


FULL, PLAIN, EMPTY = _rows(10), _rows(14), _rows(3)
VARIANTS = {1: _rows(6), 2: _rows(17), 3: _rows(9)}


def _rngs(seed=0):
    return dict(
        plain_rng=random.Random(seed + 1),
        drop_rng=random.Random(seed + 2),
        variant_rng=random.Random(seed + 3),
    )


def test_worst_case_is_the_longest_caption_and_never_dropped():
    loaded = []

    def load(k):
        loaded.append(k)
        return VARIANTS[k]

    rngs = _rngs()
    x, kind = choose_embeds(
        FULL,
        PLAIN,
        3,
        load,
        EMPTY,
        plain_rate=1.0,
        variant_rate=1.0,
        dropout=1.0,
        worst_case=True,
        **rngs,
    )
    assert (kind, x.shape[0]) == ("v2", 17)
    assert loaded == [1, 2, 3]
    # The fit step draws nothing, so the run after it is the run without it.
    fresh = _rngs()
    assert [r.random() for r in rngs.values()] == [r.random() for r in fresh.values()]

    x, kind = choose_embeds(FULL, PLAIN, worst_case=True)
    assert (kind, x.shape[0]) == ("plain", 14)
    x, kind = choose_embeds(_rows(20), PLAIN, worst_case=True)
    assert kind == "full"


def _kinds(n=4000, **kw):
    rngs = _rngs()
    counts = {}
    for _ in range(n):
        _, kind = choose_embeds(
            FULL, PLAIN, 3, VARIANTS.__getitem__, EMPTY, **kw, **rngs
        )
        kind = "variant" if kind.startswith("v") else kind
        counts[kind] = counts.get(kind, 0) + 1
    return {k: v / n for k, v in counts.items()}


def test_rates_compose():
    """Variants draw only on steps the plain caption did not take, so their share
    is (1 - plain_rate) · variant_rate; dropout then takes its share of all."""
    share = _kinds(plain_rate=0.2, variant_rate=0.5)
    assert abs(share["plain"] - 0.2) < 0.03
    assert abs(share["variant"] - 0.8 * 0.5) < 0.03
    assert abs(share["full"] - 0.8 * 0.5) < 0.03
    share = _kinds(plain_rate=0.0, variant_rate=1.0, dropout=0.25)
    assert abs(share["empty"] - 0.25) < 0.03
    assert abs(share["variant"] - 0.75) < 0.03 and "plain" not in share


def test_missing_candidates_draw_nothing():
    rngs = _rngs()
    for _ in range(5):
        x, kind = choose_embeds(
            FULL, plain_rate=1.0, variant_rate=1.0, dropout=1.0, **rngs
        )
        assert kind == "full" and x is FULL
    fresh = _rngs()
    assert [r.random() for r in rngs.values()] == [r.random() for r in fresh.values()]
