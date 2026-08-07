"""Model-EMA helpers shared by the SR/RSD finetune loops."""

import copy

import torch


def make_ema(model):
    """Return a frozen eval-mode deep copy of ``model`` to track as its EMA."""
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update(ema, model, decay):
    """Lerp EMA parameters toward ``model`` and copy its buffers."""
    for ema_parameter, model_parameter in zip(ema.parameters(), model.parameters()):
        ema_parameter.lerp_(model_parameter.detach(), 1 - decay)
    for ema_buffer, model_buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(model_buffer)
