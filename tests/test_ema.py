"""library.training.ema — promoted from the sr loops' identical copies."""

import torch
from torch import nn

from library.training.ema import ema_update, make_ema


def _model():
    m = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    return m


def test_make_ema_frozen_eval_copy():
    model = _model()
    ema = make_ema(model)
    assert not ema.training
    assert all(not p.requires_grad for p in ema.parameters())
    # independent copy: mutating the source must not touch the EMA
    with torch.no_grad():
        model[0].weight.add_(1.0)
    assert not torch.equal(ema[0].weight, model[0].weight)


def test_ema_update_lerp_and_buffer_copy():
    model = _model()
    ema = make_ema(model)
    with torch.no_grad():
        model[0].weight.zero_().add_(1.0)
        model[1].running_mean.add_(3.0)
    before = ema[0].weight.clone()
    ema_update(ema, model, decay=0.9)
    expected = before.lerp(model[0].weight, 0.1)
    assert torch.allclose(ema[0].weight, expected)
    # buffers are copied outright, not lerped
    assert torch.equal(ema[1].running_mean, model[1].running_mean)
    # decay=1.0 is a no-op on params
    snap = ema[0].weight.clone()
    ema_update(ema, model, decay=1.0)
    assert torch.equal(ema[0].weight, snap)
