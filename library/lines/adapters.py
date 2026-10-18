"""Adapters for the non-Anima lines, with their parameters outside the blocks.

``peft``'s ``add_adapter`` nests ``lora_A``/``lora_B`` inside the module they
adapt, so they would land inside the block list — and ``ModelOffloader`` swaps a
block by walking its ``named_modules()`` and moving every ``.weight`` it finds.
The trainable weights would ride to the CPU while their ``.grad`` stayed on the
card. Here the adapters are held by :class:`LoRANetwork`, which stays resident,
and each target ``nn.Linear`` gets an instance-level ``forward`` that adds their
output (Anima's ``networks/`` does the same). Patching ``forward`` rather than
replacing the module keeps the swapper's name→shape matching intact.

A line subclasses :class:`LoRANetwork` with its ``BASE_MODEL`` name, its block
list attribute (``BLOCKS``) and its default target regex
(``library/{moss,qwen21}/lora.py``). Files are saved in the kohya layout
ComfyUI's LoRA loader reads — ``lora_unet_<module_name>.{lora_down,lora_up}.weight``
plus a ``.alpha`` tensor (LyCORIS keys keep their own names under the same
prefix) — so a file is tied to its line's module names. ``from_file`` also
reads the earlier ``adapters.<module_name>.{down,up}.weight`` layout.

``down_init="weight_svd"`` seeds ``down`` with ``V_rᵀ/√3`` from the base weight's
SVD, as Anima's SVD-Down (``docs/methods/svd-down-lora.md``). The Kaiming draw
still runs first, so the global RNG stream (and the training noise drawn after
it) is the same under either init.

``algo`` picks the adapter: ``lora``, or the LyCORIS ``lokr`` / ``loha``. Their
init, scale and ΔW math come from ``lycoris.functional`` (pinned ``backend="torch"``
so dynamo traces them inside the compiled blocks), and their tensors carry
LyCORIS's key names (``lokr_w1``, ``hada_w1_a``, …, plus an ``alpha`` buffer).
LoKr applies in bypass (grouped GEMMs on the tokens; ΔW is never built). LoHa
rebuilds ΔW each forward and runs one GEMM against ``W + ΔW``, LyCORIS's own
LoHa path, so the delta is rounded to the base weight's bf16 grid in the forward.

``channel_scaling_alpha`` > 0 (``lora`` only) is Anima's channel scaling
(``docs/optimizations/channel_scaling.md``): ``down[:, c]`` is stored multiplied by
``s[c] = mean|x[c]|^α`` (mean-normalized) and the forward uses ``down · s⁻¹``, so
the output is unchanged and only Adam's per-channel step moves. The scale goes on
the ``(rank, in)`` weight instead of the ``(T, in)`` activation — same product, same
gradient, no activation-sized op. ``save`` writes ``down · s⁻¹``: the file is a plain
LoRA. ``mean|x|`` comes from a calibration file keyed by target name
(``load_channel_stats``).
"""

from __future__ import annotations

import math
import re
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

SAVED_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
DOWN_INITS = ("kaiming", "weight_svd")
ALGOS = ("lora", "lokr", "loha")
FILE_PREFIX = "lora_unet_"
_LORA_FILE_KEYS = {"down.weight": "lora_down.weight", "up.weight": "lora_up.weight"}
_LORA_STATE_KEYS = {v: k for k, v in _LORA_FILE_KEYS.items()}


class LoRAAdapter(nn.Module):
    """``scale · multiplier · up(down(x))``, added to the base linear's output.

    Built fp32 for training (the master weights AdamW updates — bf16's 8-bit
    mantissa rounds away updates below ~0.4 % of a weight). The rank GEMMs run
    in the base output's dtype, so an fp32 adapter never lifts a ``(T, hidden)``
    activation to fp32; same policy as ``networks/lora_modules/base.py``.
    ``multiplier == 0`` returns the base output untouched — the base model
    exactly, not a zeroed delta.
    """

    def __init__(self, in_features, out_features, rank, alpha, dtype):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, dtype=dtype)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        self.multiplier = 1.0
        self.register_buffer("inv_scale", None, persistent=False)

    @torch.no_grad()
    def absorb_channel_scale(self, scale: torch.Tensor) -> None:
        """Store ``down · s`` and run ``down · s⁻¹``: same output, Adam's step on
        input channel ``c`` reweighted by ``s[c]``."""
        scale = scale.to(self.down.weight.device, torch.float32)
        self.down.weight.mul_(scale.to(self.down.weight.dtype))
        self.inv_scale = scale.reciprocal()

    def down_weight(self) -> torch.Tensor:
        """The down projection the forward applies (what a plain LoRA file holds)."""
        w = self.down.weight
        return w if self.inv_scale is None else w * self.inv_scale.to(w.dtype)

    @torch.no_grad()
    def init_down_weight_svd(self, base: nn.Linear) -> None:
        """``down = V_rᵀ / √3``: the base weight's top-r right singular vectors,
        at the Kaiming default's expected row norm. Computed on the card when
        there is one (an fp32 eigh of the in_features² Gram per linear)."""
        from networks.lora_modules.lora import _top_right_singular_vectors

        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight = base.weight.detach().to(device, torch.float32)
        basis = _top_right_singular_vectors(weight, self.down.out_features)
        self.down.weight.copy_(basis.T / math.sqrt(3.0))

    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0.0:
            return base_out
        work = base_out.dtype
        lx = F.linear(x.to(work), self.down_weight().to(work))
        delta = F.linear(lx, self.up.weight.to(work))
        return base_out + delta * (self.scale * self.multiplier)

    def patched(self, module: nn.Linear, base, x: torch.Tensor) -> torch.Tensor:
        return self(x, base(module, x))


class LoKrAdapter(nn.Module):
    """LyCORIS LoKr: ``ΔW = scale · kron(w1, w2)``, ``out = a·b``, ``in = c·d``.

    ``w1`` (a, c) is the small full factor; ``w2`` (b, d) is ``w2_a @ w2_b`` at
    ``rank`` when the rank is under half its larger side, else a full matrix
    (LyCORIS's huge-dim convention), and with both halves full the scale is 1
    whatever alpha says. ``factor`` caps ``a`` / ``c`` (-1 = near-square).
    """

    def __init__(self, base: nn.Linear, rank, alpha, factor, dtype):
        super().__init__()
        from lycoris.functional import lokr

        # tucker=False: weight_gen's default inits a conv-only t2 that a
        # Linear never allocates (None).
        w1, _w1a, _w1b, w2, w2a, w2b, _t = lokr.weight_gen(
            base.weight, rank, tucker=False, factor=factor
        )
        self.lokr_w1 = nn.Parameter(w1.to(dtype))
        if w2 is not None:
            self.lokr_w2 = nn.Parameter(w2.to(dtype))
            alpha = rank
        else:
            self.lokr_w2_a = nn.Parameter(w2a.to(dtype))
            self.lokr_w2_b = nn.Parameter(w2b.to(dtype))
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.scale = alpha / rank
        self.multiplier = 1.0

    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0.0:
            return base_out
        from lycoris.functional import lokr

        work = base_out.dtype
        full = hasattr(self, "lokr_w2")
        delta = lokr.kron_bypass(
            x.to(work),
            self.lokr_w1.to(work),
            None,
            None,
            self.lokr_w2.to(work) if full else None,
            None if full else self.lokr_w2_a.to(work),
            None if full else self.lokr_w2_b.to(work),
            None,
            scale=self.scale * self.multiplier,
            backend="torch",
        )
        return base_out + delta

    def patched(self, module: nn.Linear, base, x: torch.Tensor) -> torch.Tensor:
        return self(x, base(module, x))


class LoHaAdapter(nn.Module):
    """LyCORIS LoHa: ``ΔW = scale · (w1_a @ w1_b) ⊙ (w2_a @ w2_b)``, each pair at
    ``rank`` (an effective rank up to rank²). ``hada_w1_a`` starts at zero, so
    ΔW does too."""

    def __init__(self, base: nn.Linear, rank, alpha, dtype):
        super().__init__()
        from lycoris.functional import loha

        w1d, w1u, w2d, w2u, _t1, _t2 = loha.weight_gen(base.weight, rank)
        self.hada_w1_a = nn.Parameter(w1u.to(dtype))
        self.hada_w1_b = nn.Parameter(w1d.to(dtype))
        self.hada_w2_a = nn.Parameter(w2u.to(dtype))
        self.hada_w2_b = nn.Parameter(w2d.to(dtype))
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        # HadaWeight saves its gamma for backward, so it must be a tensor; a
        # device-resident 1 keeps the Python-float scale outside it.
        self.register_buffer("_unit", torch.ones(()), persistent=False)
        self.scale = alpha / rank
        self.multiplier = 1.0

    def delta_weight(self, dtype: torch.dtype) -> torch.Tensor:
        from lycoris.functional import loha

        hada = loha.diff_weight(
            self.hada_w1_b.to(dtype),
            self.hada_w1_a.to(dtype),
            self.hada_w2_b.to(dtype),
            self.hada_w2_a.to(dtype),
            None,
            None,
            gamma=self._unit.to(dtype),
            backend="torch",
        )
        return hada * (self.scale * self.multiplier)

    def patched(self, module: nn.Linear, base, x: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0.0:
            return base(module, x)
        weight = module.weight
        merged = weight + self.delta_weight(weight.dtype)
        return F.linear(x.to(weight.dtype), merged, module.bias)


class LoRANetwork(nn.Module):
    """The adapters for one transformer, held outside it.

    ``apply_to`` patches the targets and must run before any ``torch.compile``
    of the blocks (compile traces the patched forward), the same ordering
    ``build_anima`` encodes for Anima.
    """

    BASE_MODEL: ClassVar[str] = ""
    BLOCKS: ClassVar[str] = ""
    DEFAULT_TARGETS: ClassVar[str] = ""

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        alpha: float | None = None,
        targets: str | None = None,
        dtype: torch.dtype = torch.float32,
        down_init: str = "kaiming",
        algo: str = "lora",
        factor: int = -1,
        channel_stats: dict[str, torch.Tensor] | None = None,
        channel_scaling_alpha: float = 0.0,
    ):
        super().__init__()
        if down_init not in DOWN_INITS:
            raise ValueError(
                f"down_init must be one of {DOWN_INITS}, not {down_init!r}"
            )
        if algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}, not {algo!r}")
        if channel_scaling_alpha > 0 and algo != "lora":
            raise ValueError("channel scaling applies to algo='lora' only")
        if channel_scaling_alpha > 0 and channel_stats is None:
            raise ValueError("channel_scaling_alpha > 0 needs channel_stats")
        targets = targets or self.DEFAULT_TARGETS
        pattern = re.compile(targets)
        self.algo = algo
        self.factor = factor
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.targets = targets
        self.down_init = down_init
        self.channel_scaling_alpha = float(channel_scaling_alpha)
        self.num_blocks = self._num_blocks(model)
        self.adapters = nn.ModuleDict()
        self._targets: list[tuple[str, nn.Linear]] = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or not pattern.fullmatch(name):
                continue
            if algo == "lokr":
                adapter = LoKrAdapter(module, rank, self.alpha, factor, dtype)
            elif algo == "loha":
                adapter = LoHaAdapter(module, rank, self.alpha, dtype)
            else:
                adapter = LoRAAdapter(
                    module.in_features, module.out_features, rank, self.alpha, dtype
                )
                if down_init == "weight_svd":
                    adapter.init_down_weight_svd(module)
                if self.channel_scaling_alpha > 0:
                    if name not in channel_stats:
                        raise ValueError(f"no channel stats for {name}")
                    adapter.absorb_channel_scale(
                        channel_scale(channel_stats[name], self.channel_scaling_alpha)
                    )
            self.adapters[name.replace(".", "_")] = adapter
            self._targets.append((name, module))
        if not self._targets:
            raise ValueError(f"no nn.Linear matched {targets!r}")
        self._applied = False

    @classmethod
    def _num_blocks(cls, model: nn.Module) -> int | None:
        blocks = getattr(model, cls.BLOCKS, None) if cls.BLOCKS else None
        return None if blocks is None else len(blocks)

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
            module.forward = _patched_forward(
                module, self.adapters[name.replace(".", "_")]
            )
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
            "base_model": self.BASE_MODEL,
            "algo": self.algo,
            "rank": str(self.rank),
            "alpha": str(self.alpha),
            "targets": self.targets,
        }
        if self.num_blocks is not None:
            # A LoRA is depth-specific: module names carry the block index.
            meta["ss_num_blocks"] = str(self.num_blocks)
        if self.algo == "lora":
            meta["down_init"] = self.down_init
            if self.channel_scaling_alpha > 0:
                # Provenance only: the saved down is already ``down · s⁻¹``.
                meta["channel_scaling_alpha"] = str(self.channel_scaling_alpha)
        if self.algo == "lokr":
            meta["factor"] = str(self.factor)
        meta.update({k: str(v) for k, v in extra.items()})
        return meta

    def save(self, path, dtype: str = "bf16", **extra_meta) -> None:
        from safetensors.torch import save_file

        state = {}
        for key, value in self.state_dict().items():
            name, rest = key.removeprefix("adapters.").split(".", 1)
            stem = FILE_PREFIX + name
            if self.algo == "lora" and rest == "down.weight":
                value = self.adapters[name].down_weight()
                state[f"{stem}.alpha"] = torch.tensor(self.alpha)
            state[f"{stem}.{_LORA_FILE_KEYS.get(rest, rest)}"] = value
        state = {
            k: v.detach().to("cpu", SAVED_DTYPES[dtype]).contiguous()
            for k, v in state.items()
        }
        save_file(
            state, str(path), metadata=self.metadata(lora_dtype=dtype, **extra_meta)
        )

    @classmethod
    def from_file(cls, model: nn.Module, path, dtype=None) -> tuple[LoRANetwork, dict]:
        """Rebuild the network a trainer checkpoint describes, and load it.
        Algo, rank, alpha and targets come from the file's metadata (no ``algo``
        = a LoRA from before LyCORIS); a depth mismatch (``ss_num_blocks``)
        raises instead of silently dropping blocks."""
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            meta = handle.metadata() or {}
            state = {key: handle.get_tensor(key) for key in handle.keys()}
        missing = [k for k in ("rank", "alpha", "targets") if k not in meta]
        if missing:
            raise ValueError(f"{path}: checkpoint metadata lacks {missing}")
        depth, have = meta.get("ss_num_blocks"), cls._num_blocks(model)
        if depth is not None and have is not None and int(depth) != have:
            raise ValueError(
                f"{path}: trained on a {depth}-block DiT, this one has {have}"
            )
        network = cls(
            model,
            rank=int(meta["rank"]),
            alpha=float(meta["alpha"]),
            targets=meta["targets"],
            dtype=dtype or SAVED_DTYPES.get(meta.get("lora_dtype", ""), torch.float32),
            algo=meta.get("algo", "lora"),
            factor=int(meta.get("factor", -1)),
        )
        network.load_state_dict(_state_from_file(state, network.algo))
        return network, meta

    @classmethod
    def for_inference(
        cls, transformer: nn.Module, path, multiplier: float = 1.0
    ) -> LoRANetwork:
        """Load a LoRA onto ``transformer`` (hooked, not merged), on its device, at
        ``multiplier``. Call before compiling the blocks. ``set_multiplier``
        switches strength later; ``0.0`` is the base model exactly, ``restore()``
        takes the hooks off."""
        network, _meta = cls.from_file(transformer, path)
        network.to(next(transformer.parameters()).device)
        network.requires_grad_(False)
        network.apply_to()
        network.set_multiplier(multiplier)
        return network


def load_channel_stats(path) -> dict[str, torch.Tensor]:
    """Per-target ``mean|x|`` input vectors, keyed by qualified module name."""
    from safetensors.torch import load_file

    return load_file(str(path))


def channel_scale(mean_abs: torch.Tensor, alpha: float) -> torch.Tensor:
    """``s = mean|x|^α``, normalized to ``s.mean() == 1`` (Anima's
    ``_load_channel_scales``)."""
    s = mean_abs.float().clamp_min(1e-6).pow(alpha)
    return s / s.mean()


def _state_from_file(state: dict, algo: str) -> dict:
    """File keys → ``state_dict`` keys; the pre-kohya layout passes through."""
    out = {}
    for key, value in state.items():
        if not key.startswith(FILE_PREFIX):
            out[key] = value
            continue
        name, rest = key.removeprefix(FILE_PREFIX).split(".", 1)
        if algo == "lora":
            if rest == "alpha":
                continue
            rest = _LORA_STATE_KEYS[rest]
        out[f"adapters.{name}.{rest}"] = value
    return out


def _patched_forward(module: nn.Linear, adapter: nn.Module):
    base = module.__class__.forward

    def forward(x: torch.Tensor) -> torch.Tensor:
        return adapter.patched(module, base, x)

    return forward
