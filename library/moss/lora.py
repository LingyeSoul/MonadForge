"""A LoRA on the MOSS-Image DiT whose parameters live outside ``layers``.

Same construction as ``library/qwen21/lora.py``, for the same reason: the block
swapper moves every ``.weight`` under a swapped block, so an adapter nested in
the block (peft) would page its trainable weights to the CPU while their
``.grad`` stays on the card. :class:`LoRANetwork` holds the down/up pairs and
stays resident; each target ``nn.Linear`` gets an instance-level ``forward``
that adds their output.

Default targets are the six bias-free linears of every block,
``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`` (42 × 6 = 252). The fused
``ffn_up_gate`` gets one adapter over its full 2·ffn_dim output, so the hidden
and the gate halves share the rank.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.moss.requests import TrainRequest

DEFAULT_TARGETS = TrainRequest.TARGETS
SAVED_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class LoRAAdapter(nn.Module):
    """``scale · multiplier · up(down(x))``, added to the base linear's output.

    Built fp32 for training (the master weights AdamW updates); the rank GEMMs
    run in the base output's dtype, so an fp32 adapter never lifts a
    ``(T, 3072)`` activation to fp32. ``multiplier == 0`` returns the base output
    untouched — the base model exactly, not a zeroed delta.
    """

    def __init__(self, in_features, out_features, rank, alpha, dtype):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, dtype=dtype)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        self.multiplier = 1.0

    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0.0:
            return base_out
        work = base_out.dtype
        lx = F.linear(x.to(work), self.down.weight.to(work))
        delta = F.linear(lx, self.up.weight.to(work))
        return base_out + delta * (self.scale * self.multiplier)


class LoRANetwork(nn.Module):
    """The adapters for one transformer, held outside it.

    ``apply_to`` patches the targets and must run before any ``torch.compile``
    of the blocks (compile traces the patched forward).
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        alpha: float | None = None,
        targets: str = DEFAULT_TARGETS,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        pattern = re.compile(targets)
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.targets = targets
        self.num_blocks = len(model.layers)
        self.adapters = nn.ModuleDict()
        self._targets: list[tuple[str, nn.Linear]] = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or not pattern.fullmatch(name):
                continue
            self.adapters[name.replace(".", "_")] = LoRAAdapter(
                module.in_features, module.out_features, rank, self.alpha, dtype
            )
            self._targets.append((name, module))
        if not self._targets:
            raise ValueError(f"no nn.Linear matched {targets!r}")
        self._applied = False

    @property
    def target_names(self) -> list[str]:
        return [name for name, _ in self._targets]

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def apply_to(self) -> int:
        """Patch every target's ``forward``. Returns the count."""
        if self._applied:
            raise RuntimeError("already applied")
        for name, module in self._targets:
            module.forward = _patched_forward(module, self.adapters[name.replace(".", "_")])
        self._applied = True
        return len(self._targets)

    def set_multiplier(self, multiplier: float) -> None:
        """Scale every adapter; ``0.0`` is the base model exactly, so an A/B
        needs neither a reload nor a second model."""
        for adapter in self.adapters.values():
            adapter.multiplier = float(multiplier)

    def restore(self) -> None:
        """Un-patch the targets (the adapters stay built)."""
        for _name, module in self._targets:
            module.__dict__.pop("forward", None)
        self._applied = False

    def metadata(self, **extra) -> dict[str, str]:
        meta = {
            "base_model": "MOSS-Image",
            "rank": str(self.rank),
            "alpha": str(self.alpha),
            "targets": self.targets,
            # A LoRA is depth-specific: module names carry the block index.
            "ss_num_blocks": str(self.num_blocks),
        }
        meta.update({k: str(v) for k, v in extra.items()})
        return meta

    def save(self, path, dtype: str = "bf16", **extra_meta) -> None:
        from safetensors.torch import save_file

        state = {
            k: v.detach().to("cpu", SAVED_DTYPES[dtype]).contiguous()
            for k, v in self.state_dict().items()
        }
        save_file(state, str(path), metadata=self.metadata(lora_dtype=dtype, **extra_meta))


def _patched_forward(module: nn.Linear, adapter: LoRAAdapter):
    base = module.__class__.forward

    def forward(x: torch.Tensor) -> torch.Tensor:
        return adapter(x, base(module, x))

    return forward


def load_network(model: nn.Module, path, dtype=None) -> tuple[LoRANetwork, dict]:
    """Rebuild the network a ``train.py`` checkpoint describes, and load it.
    Rank, alpha and targets come from the file's metadata; a depth mismatch
    (``ss_num_blocks``) raises instead of silently dropping blocks."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        meta = handle.metadata() or {}
        state = {key: handle.get_tensor(key) for key in handle.keys()}
    missing = [k for k in ("rank", "alpha", "targets") if k not in meta]
    if missing:
        raise ValueError(f"{path}: checkpoint metadata lacks {missing}")
    depth = meta.get("ss_num_blocks")
    if depth is not None and int(depth) != len(model.layers):
        raise ValueError(
            f"{path}: trained on a {depth}-block DiT, this one has {len(model.layers)}"
        )
    network = LoRANetwork(
        model,
        rank=int(meta["rank"]),
        alpha=float(meta["alpha"]),
        targets=meta["targets"],
        dtype=dtype or SAVED_DTYPES.get(meta.get("lora_dtype", ""), torch.float32),
    )
    network.load_state_dict(state)
    return network, meta


def load_for_inference(transformer: nn.Module, path, multiplier: float = 1.0) -> LoRANetwork:
    """Load a LoRA onto ``transformer`` (hooked, not merged), on its device, at
    ``multiplier``. Call before compiling the blocks. ``network.set_multiplier``
    switches strength later; ``0.0`` is the base model exactly, ``restore()``
    takes the hooks off."""
    network, _meta = load_network(transformer, path)
    network.to(next(transformer.parameters()).device)
    network.requires_grad_(False)
    network.apply_to()
    network.set_multiplier(multiplier)
    return network
