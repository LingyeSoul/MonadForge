"""INT8 ConvRot decoder linears for the Qwen3.5-9B TE.

Each linear's weight is rotated per 256-column group by a normalised regular
Hadamard ``h`` (ConvRot, arXiv 2512.03673): ``W_rot = W·blockdiag(h)``, then
round-to-nearest int8 with one fp32 ``weight_scale`` per output row (amax / 127).
``h`` is symmetric and its own inverse, so the same group rotation maps the input
into the weight's basis and maps a rotated weight back. The tensors and their
``comfy_quant`` are ComfyUI's ``int8_tensorwise`` + ``convrot`` layout (matching
comfy_kitchen's quantizer).

:class:`Int8ConvRotLinear` runs it two ways: ``act=False`` (weight-only) matmuls
the rotated input against the dequantised weight in bf16; ``act=True`` (W8A8)
calls ``comfy_kitchen.int8_linear(convrot=True)``, which also rounds the rotated
activation to int8 per token (faster, plus that rounding's error).
``library/moss/fp8.py`` writes the file (``quantize_te(fmt="int8_convrot")``) and
loads it back. The catalog file (``moss_te_int8``, the default
``text_encoder=int8``) is ``te_gptq.py --smooth 0.5 --act_aware``: its RMSNorm
weights carry SmoothQuant scales."""

from __future__ import annotations

import torch
import torch.nn.functional as F

GROUP = 256
INT8_MAX = 127.0
QUANT_CONF = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}


def rotate(x: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """``x`` (…, K) times ``blockdiag(h)`` over its last dim; its own inverse."""
    from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight

    shape = x.shape
    return rotate_int8_convrot_weight(x.reshape(-1, shape[-1]), group).reshape(shape)


def qdq(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``w`` (out, …) on the int8 grid at per-row ``scale`` (out,), dequantised."""
    s = scale.reshape(-1, *[1] * (w.dim() - 1))
    return (w / s).round().clamp(-INT8_MAX - 1, INT8_MAX) * s


def row_scale(w_rot: torch.Tensor) -> torch.Tensor:
    return w_rot.abs().amax(1).clamp(min=1e-12) / INT8_MAX


def quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(int8 rotated weight, fp32 (out, 1) scale)``, round to nearest; fp32 math,
    on the GPU when there is one."""
    dev = torch.device("cuda") if torch.cuda.is_available() else w.device
    w_rot = rotate(w.to(dev, torch.float32))
    scale = row_scale(w_rot)
    q = (w_rot / scale[:, None]).round().clamp(-INT8_MAX - 1, INT8_MAX)
    return q.to(torch.int8).cpu(), scale[:, None].cpu()


class Int8ConvRotLinear(torch.nn.Module):
    """A bias-free Linear holding the rotated int8 weight and its per-row scale."""

    def __init__(
        self, weight: torch.Tensor, scale: torch.Tensor, group: int, act: bool
    ):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale", scale.float().reshape(-1, 1))
        self.group, self.act = group, act

    def forward(self, x):
        if self.act:
            import comfy_kitchen

            return comfy_kitchen.int8_linear(
                x,
                self.weight,
                self.weight_scale,
                out_dtype=x.dtype,
                convrot=self.group > 0,
                convrot_groupsize=self.group or GROUP,
            )
        w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype)
        return F.linear(rotate(x, self.group) if self.group else x, w)
