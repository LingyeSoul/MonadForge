"""ReFT bench network — a train.py-compatible adapter, kept bench-local.

Drives the `bench/reft/plan.md` Phase-1' arms without restoring ReFT into the
live tree (the INTEGRATION.md re-wiring stays gated on this bench's verdict).
Selected via ``--network_module bench.reft.reft_network`` + ``--network_args``:

  reft_dim=64 reft_layers=last_8 reft_positions=all num_registers=0
      → ReFT-only arm (LoReFT residual-stream edit on every token, the
        shipped-era semantics; math mirrors bench/reft/impl/reft.py).
  reft_layers=none num_registers=36 insert_block=8
      → registers-only control (K learnable DSR registers, no ReFT).
  reft_dim=64 reft_layers=last_8 reft_positions=registers num_registers=36
      → the synergy arm: ReFT interventions restricted to the K register
        positions — position-selective ReFT with the registers as the only
        fixed-identity tokens in the sequence (register *content*
        parameterized per-block, DSR Theory-2 style).

Composition notes:

* Registers ride the shared ``networks.register_injection.RegisterInjector``
  (same machinery as ``--method register`` and the lora-family
  ``num_registers`` knob): concat at ``insert_block``, rope-exempt, stripped
  before unpatchify. ``extra_seq_tokens`` is exposed for train.py's
  dynamic-seq widening.
* ReFT wraps ``Block.forward`` (an instance-attribute monkey-patch), so the
  edit runs eagerly *outside* the compiled ``block._forward`` — compile-safe
  the same way forward hooks are (networks/CLAUDE.md §compile_blocks).
* ``reft_positions="registers"`` edits only the trailing K rows of the
  native-flatten sequence ``(B, 1, seq+K, 1, D)``; wrapping is restricted to
  blocks >= insert_block (earlier blocks never see the registers).
* Compute dtype = the block output's dtype (org_forwarded policy —
  tests/test_lora_dtype_policy.py), weights cast per call.
* Orthogonality of R is kept by the shipped-era ``‖RRᵀ−I‖²`` penalty riding
  the generic ``_ortho_reg_weight`` loss hook (library/training/losses.py).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.methods.base import AdapterNetworkBase
from networks.register_injection import RegisterInjector

logger = logging.getLogger(__name__)


def _parse_layers(spec, n_blocks: int) -> list[int]:
    """``"all"``, ``"none"``, ``"last_N"``, ``"a-b"``, comma list, or JSON list."""
    if spec is None or spec == "all":
        return list(range(n_blocks))
    if isinstance(spec, (list, tuple)):
        return sorted(int(x) for x in spec)
    s = str(spec).strip()
    if s in ("none", ""):
        return []
    if s == "all":
        return list(range(n_blocks))
    if s.startswith("last_"):
        n = int(s[len("last_") :])
        return list(range(max(0, n_blocks - n), n_blocks))
    if s.startswith("["):
        return sorted(int(x) for x in json.loads(s))
    if "-" in s and "," not in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return sorted(int(x) for x in s.split(",") if x != "")


class PositionalReFT(nn.Module):
    """LoReFT edit on a Block's output: ``h + Rᵀ(W_s·h + b) · scale · mult``.

    Standalone re-derivation of ``bench/reft/impl/reft.py`` (same math, same
    zero-init/no-op-at-step-0 property) with two bench deltas: the wrapped
    Block is stashed via ``object.__setattr__`` so it never enters this
    module's params/state_dict (register-method frozen-base trick), and
    ``register_only=True`` confines the edit to the trailing K register rows.
    """

    def __init__(
        self,
        name: str,
        block: nn.Module,
        embed_dim: int,
        reft_dim: int,
        alpha: Optional[float] = None,
        multiplier: float = 1.0,
        register_only: bool = False,
        num_registers: int = 0,
    ) -> None:
        super().__init__()
        self.name = name
        object.__setattr__(self, "_block", block)
        self.reft_dim = int(reft_dim)
        self.multiplier = float(multiplier)
        self.register_only = bool(register_only)
        self.K = int(num_registers)

        # R: orthonormal rows spanning the intervention subspace (QR init →
        # the ortho penalty starts at exactly 0).
        r = torch.randn(embed_dim, self.reft_dim)
        q, _ = torch.linalg.qr(r)
        self.rotate = nn.Parameter(q.T.contiguous())  # (reft_dim, embed_dim)

        # ΔW within R's subspace; zero-init → delta = 0 at step 0.
        self.source_w = nn.Parameter(torch.zeros(self.reft_dim, embed_dim))
        self.source_b = nn.Parameter(torch.zeros(self.reft_dim))

        a = self.reft_dim if alpha is None or float(alpha) == 0.0 else float(alpha)
        self.scale = a / self.reft_dim

        self._applied = False
        self._org_forward = None

    def apply_to(self) -> None:
        if self._applied:
            return
        self._org_forward = self._block.forward
        self._block.forward = self.forward
        self._applied = True

    def remove(self) -> None:
        if not self._applied:
            return
        self._block.forward = self._org_forward
        self._org_forward = None
        self._applied = False

    def _edit(self, h: torch.Tensor) -> torch.Tensor:
        # Compute dtype = the block output's dtype (org_forwarded policy).
        w = self.source_w.to(h.dtype)
        b = self.source_b.to(h.dtype)
        r = self.rotate.to(h.dtype)
        delta = F.linear(h, w, b)
        return F.linear(delta, r.T) * (self.multiplier * self.scale)

    def forward(self, *args, **kwargs):
        h = self._org_forward(*args, **kwargs)
        if not self.register_only:
            return h + self._edit(h)
        # Registers are the trailing K rows of the native-flatten sequence
        # (B, 1, seq+K, 1, D) — the injector forces native-flatten, so dim 2
        # is always the seq axis here.
        if self.K <= 0 or h.shape[2] <= self.K:
            return h
        tail = h[:, :, -self.K :]
        return torch.cat([h[:, :, : -self.K], tail + self._edit(tail)], dim=2)

    def regularization(self) -> torch.Tensor:
        r = self.rotate
        eye = torch.eye(self.reft_dim, device=r.device, dtype=r.dtype)
        return torch.sum((r @ r.T - eye) ** 2)


class ReFTBenchNetwork(AdapterNetworkBase):
    """ReFT (+ optional DSR registers) on a frozen Anima DiT — bench arms only."""

    network_module = "bench.reft.reft_network"
    network_spec = "reft_bench"
    mergeable = False

    def __init__(
        self,
        unet,
        reft_dim: int = 64,
        reft_alpha: Optional[float] = None,
        reft_layers="last_8",
        reft_positions: str = "all",
        num_registers: int = 0,
        insert_block: int = 8,
        init_std: float = 0.02,
        register_lr_scale: float = 100.0,
        ortho_reg_weight: float = 1e-4,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if reft_positions not in ("all", "registers"):
            raise ValueError(f"reft_positions must be all|registers, got {reft_positions!r}")
        object.__setattr__(self, "_dit", unet)
        self.multiplier = float(multiplier)
        self.K = int(num_registers)
        self.insert_block = int(insert_block)
        self.D = int(unet.model_channels)
        self.reft_dim = int(reft_dim)
        self.reft_positions = reft_positions
        self.register_lr_scale = float(register_lr_scale)
        # Generic hook read by library/training/losses.py::_ortho_reg_loss.
        self._ortho_reg_weight = float(ortho_reg_weight)

        n_blocks = len(unet.blocks)
        block_ids = _parse_layers(reft_layers, n_blocks)
        if reft_positions == "registers":
            if self.K <= 0:
                raise ValueError("reft_positions=registers requires num_registers > 0")
            # Blocks < insert_block never see the registers; the LAST block's
            # register edit is stripped before unpatchify and provably gets
            # zero gradient (verified on the 2026-07-05 smoke run).
            dropped = [
                b for b in block_ids if b < self.insert_block or b == n_blocks - 1
            ]
            if dropped:
                logger.warning(
                    "reft_positions=registers: dropping dead blocks %s "
                    "(< insert_block %d, or the stripped final block)",
                    dropped,
                    self.insert_block,
                )
            block_ids = [b for b in block_ids if b not in dropped]
        self.reft_block_ids = block_ids

        if self.K > 0:
            self.register_tokens = nn.Parameter(torch.randn(self.K, self.D) * init_std)
            self._injector = RegisterInjector(
                num_registers=self.K,
                insert_block=self.insert_block,
                get_scaled_tokens=lambda: self.register_tokens * self.multiplier,
            )
        else:
            self._injector = None
        # train.py widens the compile dynamic-seq MAX bound by this.
        self.extra_seq_tokens = self.K

        self.refts = nn.ModuleDict(
            {
                str(bi): PositionalReFT(
                    f"reft_unet_blocks_{bi}",
                    unet.blocks[bi],
                    embed_dim=int(getattr(unet.blocks[bi], "x_dim", 0) or self.D),
                    reft_dim=self.reft_dim,
                    alpha=reft_alpha,
                    multiplier=self.multiplier,
                    register_only=(reft_positions == "registers"),
                    num_registers=self.K,
                )
                for bi in block_ids
            }
        )
        if not self.refts and self.K <= 0:
            raise ValueError("nothing to train: reft_layers=none and num_registers=0")
        logger.info(
            "ReFTBenchNetwork: reft_dim=%d layers=%s positions=%s K=%d insert_block=%d "
            "trainable=%d",
            self.reft_dim,
            block_ids,
            reft_positions,
            self.K,
            self.insert_block,
            sum(p.numel() for p in self.parameters()),
        )

        self._applied = False

    # -- trainer-facing protocol ----------------------------------------------
    def set_multiplier(self, multiplier: float) -> None:
        super().set_multiplier(multiplier)
        for reft in self.refts.values():
            reft.multiplier = float(multiplier)

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        del text_encoder_lr
        lr = unet_lr or default_lr
        groups, descriptions = [], []
        reft_params = [p for m in self.refts.values() for p in m.parameters()]
        if reft_params:
            groups.append({"params": reft_params, "lr": lr})
            descriptions.append("reft_bench.reft")
        if self.K > 0:
            # Fresh embeddings competing with a baked-in sink attractor — own
            # (higher) lr group, same rationale as the register method.
            groups.append(
                {"params": [self.register_tokens], "lr": lr * self.register_lr_scale}
            )
            descriptions.append("reft_bench.registers")
        return groups, descriptions

    def get_ortho_regularization(self) -> torch.Tensor:
        total = None
        for reft in self.refts.values():
            reg = reft.regularization()
            total = reg if total is None else total + reg
        if total is None:
            return torch.zeros((), device=self.register_tokens.device)
        return total / max(len(self.refts), 1)

    def metadata_fields(self) -> dict[str, str]:
        return {
            "ss_reft_dim": str(self.reft_dim),
            "ss_reft_scale": str(next(iter(self.refts.values())).scale)
            if self.refts
            else "1.0",
            "ss_reft_layers": json.dumps(self.reft_block_ids),
            "ss_reft_positions": self.reft_positions,
            "ss_num_registers": str(self.K),
            "ss_insert_block": str(self.insert_block),
            "ss_ortho_reg_weight": str(self._ortho_reg_weight),
            "ss_model_channels": str(self.D),
            "ss_num_blocks": str(len(self._dit.blocks)),
        }

    # Relocation-crossover readouts (adoption-gate telemetry, no-grad).
    @property
    def last_reg_ratio(self) -> Optional[float]:
        return self._injector.last_reg_ratio if self._injector else None

    @property
    def last_patch_sink_ratio(self) -> Optional[float]:
        return self._injector.last_patch_sink_ratio if self._injector else None

    # -- apply / remove --------------------------------------------------------
    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        del text_encoders, apply_text_encoder
        if self._applied or not apply_unet:
            return
        if self._injector is not None:
            self._injector.apply(unet)
        for reft in self.refts.values():
            reft.apply_to()
        self._applied = True

    def remove(self) -> None:
        if not self._applied:
            return
        for reft in self.refts.values():
            reft.remove()
        if self._injector is not None:
            self._injector.remove()
        self._applied = False


# ---------------------------------------------------------------------------
# Factories (train.py contract — mirrors networks/methods/register.py)
# ---------------------------------------------------------------------------
def _as_float_or_none(v) -> Optional[float]:
    if v is None or v == "" or str(v).lower() == "none":
        return None
    return float(v)


def _build(unet, kwargs) -> ReFTBenchNetwork:
    return ReFTBenchNetwork(
        unet,
        reft_dim=int(kwargs.get("reft_dim", 64)),
        reft_alpha=_as_float_or_none(kwargs.get("reft_alpha")),
        reft_layers=kwargs.get("reft_layers", "last_8"),
        reft_positions=str(kwargs.get("reft_positions", "all")),
        num_registers=int(kwargs.get("num_registers", 0)),
        insert_block=int(kwargs.get("insert_block", 8)),
        init_std=float(kwargs.get("init_std", 0.02)),
        register_lr_scale=float(kwargs.get("register_lr_scale", 100.0)),
        ortho_reg_weight=float(kwargs.get("ortho_reg_weight", 1e-4)),
    )


def create_network(
    multiplier,
    network_dim,
    network_alpha,
    vae,
    text_encoders,
    unet,
    neuron_dropout=None,
    **kwargs,
):
    # network_dim/network_alpha come from the lora method TOML the arm rides on
    # (rank/alpha of the LoRA family) — deliberately ignored here; the ReFT
    # capacity knob is reft_dim in --network_args.
    del network_dim, network_alpha, vae, text_encoders, neuron_dropout
    net = _build(unet, kwargs)
    net.set_multiplier(float(multiplier))
    return net


def create_network_from_weights(
    multiplier,
    file,
    ae,
    text_encoders,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    """Rebuild the network shape from ``ss_*`` metadata, then load."""
    del ae, text_encoders, for_inference
    meta: dict[str, str] = {}
    if weights_sd is None:
        from safetensors import safe_open

        weights_sd = {}
        with safe_open(file, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            for k in f.keys():
                weights_sd[k] = f.get_tensor(k)
    reft_dim = int(meta.get("ss_reft_dim", kwargs.get("reft_dim", 64)))
    scale = float(meta.get("ss_reft_scale", 1.0))
    net = ReFTBenchNetwork(
        unet,
        reft_dim=reft_dim,
        reft_alpha=scale * reft_dim,
        reft_layers=json.loads(meta["ss_reft_layers"])
        if "ss_reft_layers" in meta
        else kwargs.get("reft_layers", "last_8"),
        reft_positions=str(
            meta.get("ss_reft_positions", kwargs.get("reft_positions", "all"))
        ),
        num_registers=int(meta.get("ss_num_registers", kwargs.get("num_registers", 0))),
        insert_block=int(meta.get("ss_insert_block", kwargs.get("insert_block", 8))),
    )
    missing, unexpected = net.load_state_dict(weights_sd, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("sshs_")]
    if missing or unexpected:
        raise RuntimeError(
            f"reft_bench checkpoint mismatch: missing={missing} unexpected={unexpected}"
        )
    net.set_multiplier(float(multiplier))
    return net
