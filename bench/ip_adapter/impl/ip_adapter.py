"""IP-Adapter network module for Anima (decoupled cross-attention).

Architecture (adapter-only — DiT frozen):
  reference image
      -> PE-Core vision encoder (live, eval, no_grad)        [B, T_pe, 1024]
      -> small Perceiver resampler                            [B, K, 1024]
      -> per-block (28) to_k_ip / to_v_ip Linear(1024, 2048)  [B, K, n_h, d_h]
      -> SDPA(text_q, ip_k, ip_v)
      -> add scale * ip_attn_out to existing text cross-attention output

Init:
  - Per-block ``ip_gate`` scalar at 0 — THIS is what guarantees step 0 ≡
    baseline DiT: the patched cross-attn adds ``gate * scale * ip_out`` to
    the text path, so with gate=0 the IP contribution is exactly zero
    regardless of K/V weight magnitude. The optimizer opens each gate as it
    learns to use the IP path.
  - to_k_ip / to_v_ip default (PyTorch Kaiming-uniform) — the gate alone
    handles step-0 equivalence, so K/V can be at normal magnitude once the
    gate opens. ``ip_init_std`` is exposed as a defense-in-depth override
    but isn't required: the earlier "small std acts as a soft gate" theory
    was wrong (the DiT's ``v_norm`` is Identity, so V was never normalized,
    and small-std weights grew fast enough under Adam that the IP/text
    ratio still hit ~3.7 by step 2).
  - set_ip_tokens deliberately does NOT run the IP K/V through the
    cross_attn RMSNorms: ``v_norm`` is a no-op, but ``k_norm`` would
    concentrate the IP-side softmax on a single token instead of averaging
    — keeping K small lets uniform attention smear ip_out across all slots.
  - Resampler queries init at N(0, 0.15) (img2emb's PerceiverResampler default).

Hooking:
  apply_to() monkey-patches each Block.cross_attn.forward. The patched closure
  runs the existing text cross-attn via attention_dispatch.dispatch_attention(),
  then computes a decoupled SDPA over IP K/V and adds scale * ip_out before the
  output projection. Lives on the instance, so gradient-checkpointing rerolls
  inside Block._forward see the same patch.

Train-time contract:
  Caller invokes network.set_ip_tokens(ip_tokens) ONCE per batch before the
  DiT forward. set_ip_tokens precomputes per-block (K, V) in SDPA layout
  ([B, n_h, K, d_h]) plus a per-block ``gate * scale`` scalar so the patched
  cross-attn forward is a single SDPA call followed by a single multiply.
  Pass ``None`` (or call clear_ip_tokens) for unconditional passes / CFG
  dropout.

Inference VRAM:
  After set_ip_tokens, the PE encoder is dead weight (K/V live on the
  cross-attn modules). _setup_ip_adapter calls release_encoder() to drop
  ``_pe_inner`` / ``_pe_lora`` so PE-Core's ~600 MB-2 GB doesn't sit on
  the GPU through the diffusion loop.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.log import setup_logging
from library.training.method_adapter import (
    MethodAdapter,
    SetupCtx,
    StepCtx,
    ValidationBaseline,
)
from library.vision import (
    VisionEncoderBundle,
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)
from library.vision.encoders import get_encoder_info
from library.vision.resampler import PerceiverResampler
from networks.methods.base import AdapterNetworkBase
from networks.methods.ip_adapter_pe_lora import inject_pe_lora

setup_logging()
logger = logging.getLogger(__name__)


# Anima DiT defaults — see library/anima/models.py:Anima.__init__
DEFAULT_NUM_BLOCKS = 28
DEFAULT_HIDDEN_SIZE = 2048  # query_dim
DEFAULT_NUM_HEADS = 16
DEFAULT_HEAD_DIM = DEFAULT_HIDDEN_SIZE // DEFAULT_NUM_HEADS  # 128
DEFAULT_CONTEXT_DIM = (
    1024  # crossattn_emb_channels — also matches PE-Core d_enc
)
DEFAULT_NUM_IP_TOKENS = 16
DEFAULT_RESAMPLER_HEADS = 8
DEFAULT_RESAMPLER_LAYERS = 2
DEFAULT_ENCODER_NAME = "pe"
# None => leave nn.Linear's default Kaiming-uniform init in place. Step-0 ==
# baseline DiT is enforced by the per-block `ip_gate` scalar (init 0), not by
# shrinking K/V — small init was the old "soft gate" trick, misdiagnosed
# (v_norm is Identity, so ip_init_std alone can't pin step-0 to baseline).
# Set `ip_init_std` explicitly in the toml for a defense-in-depth override.
DEFAULT_IP_INIT_STD: Optional[float] = None


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    del vae, text_encoders, neuron_dropout, network_alpha
    num_ip_tokens = network_dim if network_dim is not None else DEFAULT_NUM_IP_TOKENS

    encoder_name = kwargs.get("encoder", DEFAULT_ENCODER_NAME)
    encoder_dim = int(kwargs.get("encoder_dim", DEFAULT_CONTEXT_DIM))
    resampler_layers = int(kwargs.get("resampler_layers", DEFAULT_RESAMPLER_LAYERS))
    resampler_heads = int(kwargs.get("resampler_heads", DEFAULT_RESAMPLER_HEADS))
    raw_ip_init_std = kwargs.get("ip_init_std", DEFAULT_IP_INIT_STD)
    ip_init_std = float(raw_ip_init_std) if raw_ip_init_std is not None else None
    ip_scale = float(kwargs.get("ip_scale", 1.0))
    # None => global adapter LR. e.g. "gate_lr=1e-3" opens the gate faster.
    raw_gate_lr = kwargs.get("gate_lr", None)
    gate_lr = float(raw_gate_lr) if raw_gate_lr is not None else None
    # Optional precomputed PE centroid sidecar (see IPAdapterNetwork.__init__
    # comment on ip_centroid), subtracted from every token before the resampler.
    ip_centroid_path = kwargs.get("ip_centroid_path", None) or None

    # PE-Core LoRA prototype: network owns the PE encoder and trains LoRA on
    # every resblock (qkv + attn_out + mlp); forces live PE encoding.
    pe_lora_enabled = _parse_bool(kwargs.get("pe_lora_enabled", False))
    pe_lora_rank = int(kwargs.get("pe_lora_rank", 16))
    pe_lora_alpha = float(kwargs.get("pe_lora_alpha", 16.0))
    raw_pe_lora_lr = kwargs.get("pe_lora_lr", None)
    pe_lora_lr = float(raw_pe_lora_lr) if raw_pe_lora_lr is not None else None
    pe_lora_qkv = _parse_bool(kwargs.get("pe_lora_qkv", True))
    pe_lora_attn_out = _parse_bool(kwargs.get("pe_lora_attn_out", True))
    pe_lora_mlp = _parse_bool(kwargs.get("pe_lora_mlp", True))
    # -1 (default) adapts every PE resblock; positive N adapts only the last N,
    # concentrating capacity on the high-level semantic layers.
    pe_lora_layer_from = int(kwargs.get("pe_lora_layer_from", -1))

    num_blocks = (
        getattr(unet, "num_blocks", DEFAULT_NUM_BLOCKS)
        if unet is not None
        else DEFAULT_NUM_BLOCKS
    )
    hidden_size = (
        getattr(unet, "model_channels", DEFAULT_HIDDEN_SIZE)
        if unet is not None
        else DEFAULT_HIDDEN_SIZE
    )
    num_heads = (
        getattr(unet, "num_heads", DEFAULT_NUM_HEADS)
        if unet is not None
        else DEFAULT_NUM_HEADS
    )
    context_dim = DEFAULT_CONTEXT_DIM

    network = IPAdapterNetwork(
        num_ip_tokens=num_ip_tokens,
        encoder_name=encoder_name,
        encoder_dim=encoder_dim,
        context_dim=context_dim,
        num_blocks=num_blocks,
        hidden_size=hidden_size,
        num_heads=num_heads,
        resampler_layers=resampler_layers,
        resampler_heads=resampler_heads,
        ip_init_std=ip_init_std,
        ip_scale=ip_scale,
        gate_lr=gate_lr,
        multiplier=multiplier,
        pe_lora_enabled=pe_lora_enabled,
        pe_lora_rank=pe_lora_rank,
        pe_lora_alpha=pe_lora_alpha,
        pe_lora_lr=pe_lora_lr,
        pe_lora_qkv=pe_lora_qkv,
        pe_lora_attn_out=pe_lora_attn_out,
        pe_lora_mlp=pe_lora_mlp,
        pe_lora_layer_from=pe_lora_layer_from,
    )
    if ip_centroid_path:
        network.load_centroid_from_file(ip_centroid_path)
    return network


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


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
    del ae, text_encoders, for_inference
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    metadata = {}
    if file is not None and os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            metadata = f.metadata() or {}

    encoder_name = kwargs.get("encoder") or metadata.get(
        "ss_encoder", DEFAULT_ENCODER_NAME
    )
    encoder_dim = int(metadata.get("ss_encoder_dim", DEFAULT_CONTEXT_DIM))
    context_dim = int(metadata.get("ss_context_dim", DEFAULT_CONTEXT_DIM))
    num_ip_tokens = int(metadata.get("ss_num_ip_tokens", DEFAULT_NUM_IP_TOKENS))
    num_blocks = int(metadata.get("ss_num_blocks", DEFAULT_NUM_BLOCKS))
    hidden_size = int(metadata.get("ss_hidden_size", DEFAULT_HIDDEN_SIZE))
    num_heads = int(metadata.get("ss_num_heads", DEFAULT_NUM_HEADS))
    resampler_layers = int(
        metadata.get("ss_resampler_layers", DEFAULT_RESAMPLER_LAYERS)
    )
    resampler_heads = int(metadata.get("ss_resampler_heads", DEFAULT_RESAMPLER_HEADS))
    ip_scale = float(kwargs.get("ip_scale") or metadata.get("ss_ip_scale", 1.0))

    pe_lora_enabled = _parse_bool(metadata.get("ss_pe_lora_enabled", False))
    pe_lora_rank = int(metadata.get("ss_pe_lora_rank", 16))
    pe_lora_alpha = float(metadata.get("ss_pe_lora_alpha", 16.0))
    pe_lora_qkv = _parse_bool(metadata.get("ss_pe_lora_qkv", True))
    pe_lora_attn_out = _parse_bool(metadata.get("ss_pe_lora_attn_out", True))
    pe_lora_mlp = _parse_bool(metadata.get("ss_pe_lora_mlp", True))
    pe_lora_layer_from = int(metadata.get("ss_pe_lora_layer_from", -1))

    network = IPAdapterNetwork(
        num_ip_tokens=num_ip_tokens,
        encoder_name=encoder_name,
        encoder_dim=encoder_dim,
        context_dim=context_dim,
        num_blocks=num_blocks,
        hidden_size=hidden_size,
        num_heads=num_heads,
        resampler_layers=resampler_layers,
        resampler_heads=resampler_heads,
        ip_init_std=DEFAULT_IP_INIT_STD,
        ip_scale=ip_scale,
        gate_lr=None,  # inference path; no optimizer
        multiplier=multiplier,
        pe_lora_enabled=pe_lora_enabled,
        pe_lora_rank=pe_lora_rank,
        pe_lora_alpha=pe_lora_alpha,
        pe_lora_lr=None,
        pe_lora_qkv=pe_lora_qkv,
        pe_lora_attn_out=pe_lora_attn_out,
        pe_lora_mlp=pe_lora_mlp,
        pe_lora_layer_from=pe_lora_layer_from,
    )
    return network, weights_sd


class IPAdapterNetwork(AdapterNetworkBase):
    network_module = "networks.methods.ip_adapter"
    network_spec = "ip_adapter"

    def __init__(
        self,
        *,
        num_ip_tokens: int,
        encoder_name: str,
        encoder_dim: int,
        context_dim: int,
        num_blocks: int,
        hidden_size: int,
        num_heads: int,
        resampler_layers: int,
        resampler_heads: int,
        ip_init_std: Optional[float],
        ip_scale: float,
        gate_lr: Optional[float] = None,
        multiplier: float = 1.0,
        pe_lora_enabled: bool = False,
        pe_lora_rank: int = 16,
        pe_lora_alpha: float = 16.0,
        pe_lora_lr: Optional[float] = None,
        pe_lora_qkv: bool = True,
        pe_lora_attn_out: bool = True,
        pe_lora_mlp: bool = True,
        pe_lora_layer_from: int = -1,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
            )
        self.num_ip_tokens = num_ip_tokens
        self.encoder_name = encoder_name
        self.encoder_dim = encoder_dim
        self.context_dim = context_dim
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.resampler_layers = resampler_layers
        self.resampler_heads = resampler_heads
        self.ip_init_std = ip_init_std
        self.ip_scale = ip_scale
        # None => gate uses the global adapter LR. Set explicitly to override.
        self.gate_lr = gate_lr
        self.multiplier = multiplier
        self.pe_lora_enabled = pe_lora_enabled
        self.pe_lora_rank = pe_lora_rank
        self.pe_lora_alpha = pe_lora_alpha
        self.pe_lora_lr = pe_lora_lr
        self.pe_lora_qkv = pe_lora_qkv
        self.pe_lora_attn_out = pe_lora_attn_out
        self.pe_lora_mlp = pe_lora_mlp
        self.pe_lora_layer_from = pe_lora_layer_from

        # Resampler: PE patch tokens -> K compact image tokens in context-dim space.
        # d_out=context_dim so to_k_ip/to_v_ip can read directly without another proj.
        self.resampler = PerceiverResampler(
            d_enc=encoder_dim,
            d_model=context_dim,
            n_heads=resampler_heads,
            n_slots=num_ip_tokens,
            n_layers=resampler_layers,
            d_out=context_dim,
        )

        # Dataset-mean centroid in PE feature space, subtracted from every
        # token before the resampler: pooled PE features sit on a low-rank
        # sub-space with high cross-pair similarity, so the resampler is
        # hunting a small per-image delta on top of a strong shared-aesthetic
        # background — subtracting the centroid gives it that delta directly.
        # Init zero => no-op until a centroid is loaded; round-trips through
        # state_dict so inference inherits the same shift.
        self.register_buffer(
            "ip_centroid", torch.zeros(encoder_dim), persistent=True
        )
        # "is centroid populated?" bit. None => recompute on next encode (one-
        # time GPU->CPU sync); load_state_dict invalidates it.
        self._centroid_active: Optional[bool] = False

        # Per-block IP projections. Bias=False matches the existing kv_proj.
        self.to_k_ip = nn.ModuleList(
            [nn.Linear(context_dim, hidden_size, bias=False) for _ in range(num_blocks)]
        )
        self.to_v_ip = nn.ModuleList(
            [nn.Linear(context_dim, hidden_size, bias=False) for _ in range(num_blocks)]
        )
        # None: leave PyTorch's default Kaiming-uniform init — the gate
        # handles step-0 equivalence, so K/V can be at normal magnitude.
        if ip_init_std is not None:
            for proj in list(self.to_k_ip) + list(self.to_v_ip):
                nn.init.normal_(proj.weight, std=ip_init_std)

        # Per-block learned scalar gate, init 0 — decouples "step 0 =
        # baseline" from the K/V weight scale (ip_init_std alone can't
        # enforce it; see module docstring). Optimizer learns when/where to
        # open the IP path.
        #
        # Stored as a ParameterList of 0-d Parameters (not a single
        # Parameter[num_blocks]) so apply_to can hand each block's gate
        # directly to its cross_attn as ``cross_attn._ip_gate`` — no
        # Python-int block_idx indexing, which would make torch.compile
        # specialize a separate graph per block (same rationale as the
        # diagnostic accumulators below).
        self.ip_gate = nn.ParameterList(
            [nn.Parameter(torch.zeros(())) for _ in range(num_blocks)]
        )

        # Populated by apply_to() — references to the DiT cross-attn Attention
        # modules. Held as a plain list (NOT nn.ModuleList) so PyTorch doesn't
        # treat the DiT as a child of this network.
        self._cross_attn_modules: list[nn.Module] = []
        self._original_forwards: list = []
        self._patched: bool = False

        # Diagnostic accumulators: per-block running sum of ‖ip_out‖/‖text_result‖
        # plus call count. Stored as 0-d tensors ON THE cross_attn MODULES (not
        # here) so the patched forward reads them via ``orig_attn`` rather than
        # a Python-int block_idx (torch.compile graph-per-block trap again).
        # Allocated lazily on ``set_diagnostics_enabled(True)``; read via
        # ``diagnostic_summary()``.
        self._diag_enabled: bool = False

        # PE-Core encoder + LoRA. When enabled, the network owns the encoder so
        # FM gradients flow back through PE; base PE params are frozen, only
        # the LoRA ModuleDict trains. PE base params live under
        # ``self._pe_inner.*`` and are EXCLUDED from save_weights (loaded
        # fresh from the .pt checkpoint each time).
        self._pe_inner: Optional[nn.Module] = None
        self._pe_lora: Optional[nn.ModuleDict] = None
        self._pe_bundle_info = None
        if pe_lora_enabled:
            self._build_pe_with_lora()

        total = sum(p.numel() for p in self.parameters())
        init_desc = f"std={ip_init_std}" if ip_init_std is not None else "default"
        gate_lr_desc = f"lr={gate_lr}" if gate_lr is not None else "lr=global"
        logger.info(
            f"IPAdapterNetwork: K={num_ip_tokens}, encoder={encoder_name}, "
            f"context_dim={context_dim}, num_blocks={num_blocks}, "
            f"hidden={hidden_size}/{num_heads}h, ip_init={init_desc}, "
            f"gate=per-block (init 0, {gate_lr_desc}), ip_scale={ip_scale}, "
            f"params={total / 1e6:.1f}M"
        )

    # ------------------------------------------------------------ apply / hook

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        del text_encoders, apply_text_encoder
        if not apply_unet:
            return
        if self._patched:
            logger.warning("IPAdapterNetwork.apply_to called twice — skipping")
            return
        if unet is None or not hasattr(unet, "blocks"):
            raise ValueError("apply_to requires the Anima DiT (unet) with .blocks")
        if len(unet.blocks) != self.num_blocks:
            raise ValueError(
                f"DiT has {len(unet.blocks)} blocks, IP-Adapter expects {self.num_blocks}. "
                "Re-create the network with matching num_blocks."
            )

        from networks import (
            attention_dispatch as anima_attention,
        )  # local: avoid import cycle at file load

        for idx, block in enumerate(unet.blocks):
            cross_attn = block.cross_attn
            if cross_attn.is_selfattn:
                raise RuntimeError(
                    f"block[{idx}].cross_attn unexpectedly self-attention"
                )
            if cross_attn.context_dim != self.context_dim:
                raise ValueError(
                    f"block[{idx}].cross_attn context_dim {cross_attn.context_dim} "
                    f"!= IP context_dim {self.context_dim}"
                )
            if (
                cross_attn.n_heads != self.num_heads
                or cross_attn.head_dim != self.head_dim
            ):
                raise ValueError(
                    f"block[{idx}].cross_attn heads/head_dim mismatch: "
                    f"({cross_attn.n_heads}, {cross_attn.head_dim}) vs "
                    f"({self.num_heads}, {self.head_dim})"
                )
            self._cross_attn_modules.append(cross_attn)
            self._original_forwards.append(cross_attn.forward)
            cross_attn._ip_k_cached = None
            cross_attn._ip_v_cached = None
            cross_attn._ip_gate_scale_cached = None
            cross_attn._ip_gate = self.ip_gate[idx]  # 0-d Parameter, init 0
            cross_attn._ip_diag_ratio_sum = (
                None  # 0-d float32, set by set_diagnostics_enabled
            )
            cross_attn._ip_diag_count = None  # 0-d int64
            cross_attn.forward = _make_patched_forward(
                cross_attn, self, idx, anima_attention
            )

        self._patched = True
        logger.info(
            f"IP-Adapter: patched cross-attn forward on {len(self._cross_attn_modules)} blocks"
        )

    def remove_from(self):
        for cross_attn, orig in zip(self._cross_attn_modules, self._original_forwards):
            cross_attn.forward = orig
            cross_attn._ip_k_cached = None
            cross_attn._ip_v_cached = None
            cross_attn._ip_gate_scale_cached = None
            cross_attn._ip_gate = None
            cross_attn._ip_diag_ratio_sum = None
            cross_attn._ip_diag_count = None
        self._cross_attn_modules.clear()
        self._original_forwards.clear()
        self._patched = False

    # ------------------------------------------------------------ runtime API

    def encode_ip_tokens(self, image_features: torch.Tensor) -> torch.Tensor:
        """Run the resampler on raw vision-encoder features.

        Args:
            image_features: [B, T_pe, encoder_dim] from PE-Core.
        Returns:
            [B, K, context_dim]
        """
        # Lazy-resolve the centroid bit so subsequent encodes skip the
        # GPU→CPU sync. load_centroid_from_file / load_weights flip it.
        if self._centroid_active is None:
            self._centroid_active = bool(self.ip_centroid.abs().sum().item() > 0)
        if self._centroid_active:
            image_features = image_features - self.ip_centroid.to(
                dtype=image_features.dtype, device=image_features.device
            )
        return self.resampler(image_features)

    def load_centroid_from_file(self, path: str) -> None:
        """Load a precomputed PE centroid sidecar into ``ip_centroid``.

        Sidecar layout: a safetensors file with a single ``"centroid"`` tensor
        of shape ``[encoder_dim]`` (fp32 on disk; cast to the buffer's dtype
        on load). Produced by ``scripts/preprocess/cache_pe_encoder.py --centroid_only``.
        """
        from safetensors.torch import load_file

        sd = load_file(path)
        if "centroid" not in sd:
            raise KeyError(
                f"PE centroid sidecar {path} has no 'centroid' key; keys={list(sd)}"
            )
        c = sd["centroid"]
        if c.shape != (self.encoder_dim,):
            raise ValueError(
                f"PE centroid shape mismatch: got {tuple(c.shape)}, "
                f"expected ({self.encoder_dim},)"
            )
        self.ip_centroid.copy_(c.to(self.ip_centroid.dtype))
        self._centroid_active = True
        logger.info(
            f"IP-Adapter: loaded PE centroid from {path} "
            f"(‖centroid‖={float(self.ip_centroid.norm()):.3f})"
        )

    def set_ip_tokens(self, ip_tokens: Optional[torch.Tensor]) -> None:
        """Pre-compute per-block (K, V) and cache on each cross_attn.

        Pass ``None`` (or call ``clear_ip_tokens``) for unconditional / CFG
        dropout passes. K/V are stored in SDPA layout ``[B, n_h, K, d_h]`` so
        the patched forward feeds them straight to SDPA with no per-step
        transpose; ``gate * effective_scale`` is also pre-cached per block
        (still carries grad back to ``ip_gate`` at training time).
        """
        if not self._patched:
            raise RuntimeError("set_ip_tokens called before apply_to")
        if ip_tokens is None:
            self.clear_ip_tokens()
            return
        if ip_tokens.dim() != 3 or ip_tokens.shape[-1] != self.context_dim:
            raise ValueError(
                f"ip_tokens must be [B, K, {self.context_dim}], got {tuple(ip_tokens.shape)}"
            )

        # all to_k_ip/to_v_ip share dtype, so one cast covers every block
        proj_dtype = self.to_k_ip[0].weight.dtype
        ip_in = ip_tokens.to(proj_dtype)
        n_h, d_h = self.num_heads, self.head_dim
        eff_scale = self.ip_scale * self.multiplier
        for idx, cross_attn in enumerate(self._cross_attn_modules):
            k = self.to_k_ip[idx](ip_in)  # [B, K, hidden]
            v = self.to_v_ip[idx](ip_in)
            # SDPA layout; contiguous() materializes the small K cache once
            # here instead of letting SDPA re-stride per call.
            k = k.unflatten(-1, (n_h, d_h)).transpose(1, 2).contiguous()
            v = v.unflatten(-1, (n_h, d_h)).transpose(1, 2).contiguous()
            # Intentionally skip cross_attn.k_norm/v_norm: v_norm is Identity
            # anyway, but k_norm (RMSNorm) would concentrate the IP-side
            # softmax on a single token instead of averaging across K_ip
            # slots. The step-0-baseline invariant comes from ip_gate, not
            # this bypass.
            cross_attn._ip_k_cached = k
            cross_attn._ip_v_cached = v
            # pre-cast (gate * scale); still a tensor, so autograd flows to ip_gate
            cross_attn._ip_gate_scale_cached = (
                self.ip_gate[idx] * eff_scale
            ).to(proj_dtype)

    def clear_ip_tokens(self) -> None:
        for cross_attn in self._cross_attn_modules:
            cross_attn._ip_k_cached = None
            cross_attn._ip_v_cached = None
            cross_attn._ip_gate_scale_cached = None

    def get_effective_scale(self) -> float:
        return self.ip_scale * self.multiplier

    # ------------------------------------------------------------ PE-LoRA path

    def _build_pe_with_lora(self) -> None:
        """Instantiate PE base + inject LoRA. Called from __init__ when enabled."""
        from library.models.pe import build_pe_vision

        info = get_encoder_info(self.encoder_name)
        if self.encoder_name != "pe":
            raise ValueError(
                f"PE-LoRA: unknown encoder_name={self.encoder_name!r} (only 'pe' is supported)"
            )
        if info.d_enc != self.encoder_dim:
            raise ValueError(
                f"PE-LoRA: registry d_enc {info.d_enc} != encoder_dim {self.encoder_dim}"
            )

        ckpt_path = info.default_model_id()
        pe = build_pe_vision("PE-Core-L14-336")
        pe.load_pe_checkpoint(ckpt_path, verbose=True)
        pe.requires_grad_(False)  # base PE frozen; LoRA params trainable

        self._pe_inner = pe
        self._pe_lora = inject_pe_lora(
            pe,
            rank=self.pe_lora_rank,
            alpha=self.pe_lora_alpha,
            target_qkv=self.pe_lora_qkv,
            target_attn_out=self.pe_lora_attn_out,
            target_mlp=self.pe_lora_mlp,
            layer_from=self.pe_lora_layer_from,
        )
        self._pe_bundle_info = info  # for bucket_spec lookup at encode time

    def release_encoder(self) -> None:
        """Drop the owned PE encoder + LoRA after the reference image has been
        encoded into ``ip_k`` / ``ip_v`` caches.

        Inference-only: the PE encoder runs exactly once (in
        ``_setup_ip_adapter``); freeing it claws back VRAM (~600 MB bf16 for
        PE-Core-L14-336) for the diffusion forward. After this call
        ``encode_images`` / ``encode_ip_tokens`` will fail — do not call it
        during training.
        """
        had_inner = self._pe_inner is not None
        had_lora = self._pe_lora is not None
        self._pe_inner = None
        self._pe_lora = None
        self._pe_bundle_info = None
        if (had_inner or had_lora) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode_images(self, images_minus1to1: torch.Tensor) -> torch.Tensor:
        """Run the owned PE encoder on [B, 3, H, W] in [-1, 1]; gradient flows
        through the injected LoRA. Returns ``[B, T_pe, encoder_dim]``.

        Dataloader is per-batch homogeneous, so resizes to one bucket-pixels
        target and runs a single PE forward. Mirrors
        ``library.vision.encoder.encode_pe_from_imageminus1to1(same_bucket=True)``
        but without its VisionEncoderBundle indirection / ``torch.no_grad``.
        """
        if not self.pe_lora_enabled or self._pe_inner is None:
            raise RuntimeError("encode_images requires pe_lora_enabled=True")
        from library.vision.buckets import pick_bucket

        device = next(self._pe_lora.parameters()).device
        dtype = next(self._pe_lora.parameters()).dtype
        spec = self._pe_bundle_info.bucket_spec
        x = images_minus1to1.to(device=device, dtype=dtype)
        h, w = x.shape[-2:]
        h_p, w_p = pick_bucket(h, w, spec)
        target_h, target_w = h_p * spec.patch, w_p * spec.patch
        if (h, w) != (target_h, target_w):
            x = F.interpolate(
                x, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        feats, _pooled = self._pe_inner.encode(x)  # [B, T_pe, encoder_dim]
        return feats

    # ------------------------------------------------------------ diagnostics

    def set_diagnostics_enabled(
        self, enabled: bool, device: Optional[torch.device] = None
    ) -> None:
        """Toggle per-step accumulation of ‖scale·ip_out‖/‖text_result‖.

        Buffers are lazy-allocated on first enable, as 0-d tensors on each
        cross_attn module (so the patched forward never indexes by block_idx).
        NOT registered as nn.Buffers — pure runtime state, not in ``state_dict``.
        """
        if not self._patched:
            raise RuntimeError("set_diagnostics_enabled called before apply_to")
        if enabled:
            if device is None:
                device = next(self.to_k_ip[0].parameters()).device
            for cross_attn in self._cross_attn_modules:
                if cross_attn._ip_diag_ratio_sum is None:
                    cross_attn._ip_diag_ratio_sum = torch.zeros(
                        (), dtype=torch.float32, device=device
                    )
                    cross_attn._ip_diag_count = torch.zeros(
                        (), dtype=torch.int64, device=device
                    )
        self._diag_enabled = bool(enabled)

    def reset_diagnostics(self) -> None:
        for cross_attn in self._cross_attn_modules:
            if cross_attn._ip_diag_ratio_sum is not None:
                cross_attn._ip_diag_ratio_sum.zero_()
                cross_attn._ip_diag_count.zero_()

    def diagnostic_summary(self, *, reset: bool = True, log: bool = True) -> dict:
        """Return per-block ‖to_k_ip‖, ‖to_v_ip‖, mean ip/text ratio.

        ``log=True`` prints aggregate stats plus a few representative
        per-block lines (first/middle/last); the full per-block tensors are
        always returned in the dict.
        """
        with torch.no_grad():
            k_norms = torch.tensor(
                [
                    self.to_k_ip[i].weight.detach().float().norm().item()
                    for i in range(self.num_blocks)
                ]
            )
            v_norms = torch.tensor(
                [
                    self.to_v_ip[i].weight.detach().float().norm().item()
                    for i in range(self.num_blocks)
                ]
            )
            sums_list = []
            counts_list = []
            for cross_attn in self._cross_attn_modules:
                if cross_attn._ip_diag_ratio_sum is not None:
                    sums_list.append(
                        cross_attn._ip_diag_ratio_sum.detach().float().cpu()
                    )
                    counts_list.append(cross_attn._ip_diag_count.detach().float().cpu())
                else:
                    sums_list.append(torch.zeros(()))
                    counts_list.append(torch.zeros(()))
            if sums_list:
                sums = torch.stack(sums_list)
                counts = torch.stack(counts_list)
                ratios = torch.where(
                    counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums)
                )
                total_calls = int(counts.sum().item())
            else:
                ratios = torch.zeros(self.num_blocks)
                total_calls = 0

        if log:

            def _stats(t: torch.Tensor) -> str:
                return f"min={t.min().item():.3e} mean={t.mean().item():.3e} max={t.max().item():.3e}"

            logger.info(
                f"[IP-Adapter diag] params: ‖to_k_ip‖ {_stats(k_norms)} | ‖to_v_ip‖ {_stats(v_norms)}"
            )
            logger.info(
                f"[IP-Adapter diag] runtime: ‖scale·ip_out‖/‖text_result‖ {_stats(ratios)} "
                f"(N={total_calls} calls across {self.num_blocks} blocks)"
            )
            sample_idx = [0, self.num_blocks // 2, self.num_blocks - 1]
            for i in sample_idx:
                logger.info(
                    f"[IP-Adapter diag]   block[{i}]: ‖k‖={k_norms[i].item():.3e} "
                    f"‖v‖={v_norms[i].item():.3e} ratio={ratios[i].item():.3e}"
                )

        if reset:
            self.reset_diagnostics()

        return {
            "to_k_ip_norm": k_norms,
            "to_v_ip_norm": v_norms,
            "ip_text_ratio": ratios,
            "num_calls": total_calls,
        }

    def metrics(self, ctx) -> dict[str, float]:
        """Step-cadence tfevents scalars for the IP path. Single CPU sync.

        Always emits ``ip_adapter/to_{k,v}_ip_norm/{mean,max}`` (pure weight
        reads, no forward overhead). Emits ``ip_adapter/ip_text_ratio/{mean,max}``
        only while ``_diag_enabled`` is True; each call resets the
        accumulator so the scalar reflects the window since the previous log
        step, not a running epoch mean.

        Reading these: flat ``to_k_ip_norm/mean`` near init => projections
        aren't growing (resampler+KV undertrained). ``ip_text_ratio/mean``
        < 1e-2 => IP signal too small to nudge cross-attn output. Both
        growing but validation ``no_ip`` delta stays <= 0 => projections are
        moving in a direction that doesn't help held-out loss.
        """
        del ctx
        if not self._patched:
            return {}
        with torch.no_grad():
            k_norms = torch.stack(
                [self.to_k_ip[i].weight.float().norm() for i in range(self.num_blocks)]
            )
            v_norms = torch.stack(
                [self.to_v_ip[i].weight.float().norm() for i in range(self.num_blocks)]
            )
            # signed mean keeps the gate's sign visible (it can go negative);
            # abs-mean/max tracks how far it has opened either direction
            gates = torch.stack([self.ip_gate[i].float() for i in range(self.num_blocks)])
            agg = [
                k_norms.mean(), k_norms.max(),
                v_norms.mean(), v_norms.max(),
                gates.mean(), gates.abs().mean(), gates.abs().max(),
            ]

            ratio_keys: list[str] = []
            if self._diag_enabled:
                sums = []
                counts = []
                for cross_attn in self._cross_attn_modules:
                    if cross_attn._ip_diag_ratio_sum is None:
                        continue
                    sums.append(cross_attn._ip_diag_ratio_sum.float())
                    counts.append(cross_attn._ip_diag_count.float())
                if sums:
                    sums_t = torch.stack(sums)
                    counts_t = torch.stack(counts).clamp_min(1)
                    ratios = sums_t / counts_t
                    agg.extend([ratios.mean(), ratios.max()])
                    ratio_keys = [
                        "ip_adapter/ip_text_ratio/mean",
                        "ip_adapter/ip_text_ratio/max",
                    ]
                    for cross_attn in self._cross_attn_modules:
                        if cross_attn._ip_diag_ratio_sum is not None:
                            cross_attn._ip_diag_ratio_sum.zero_()
                            cross_attn._ip_diag_count.zero_()
            vals = torch.stack(agg).cpu().tolist()

        keys = [
            "ip_adapter/to_k_ip_norm/mean",
            "ip_adapter/to_k_ip_norm/max",
            "ip_adapter/to_v_ip_norm/mean",
            "ip_adapter/to_v_ip_norm/max",
            "ip_adapter/ip_gate/mean",
            "ip_adapter/ip_gate/abs_mean",
            "ip_adapter/ip_gate/abs_max",
            *ratio_keys,
        ]
        return dict(zip(keys, vals))

    # ------------------------------------------------------------ trainer hooks

    def prepare_grad_etc(self, text_encoder, unet):
        super().prepare_grad_etc(text_encoder, unet)
        if self.pe_lora_enabled and self._pe_inner is not None:
            # Re-freeze the base PE; only the injected LoRA params train.
            for p in self._pe_inner.parameters():
                p.requires_grad_(False)

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        del text_encoder_lr
        lr = unet_lr or default_lr
        # gate is 28 scalars; defaults to the global LR (Adam's adaptive rates
        # usually handle it) but can move too slowly at the global LR, hence
        # the override.
        gate_lr = self.gate_lr if self.gate_lr is not None else lr
        params = [
            {"params": list(self.resampler.parameters()), "lr": lr},
            {
                "params": list(self.to_k_ip.parameters())
                + list(self.to_v_ip.parameters()),
                "lr": lr,
            },
            {"params": list(self.ip_gate.parameters()), "lr": gate_lr},
        ]
        descriptions = ["ip_resampler", "ip_kv_proj", "ip_gate"]
        if self.pe_lora_enabled and self._pe_lora is not None:
            pe_lora_lr = self.pe_lora_lr if self.pe_lora_lr is not None else lr
            params.append(
                {"params": list(self._pe_lora.parameters()), "lr": pe_lora_lr}
            )
            descriptions.append("ip_pe_lora")
        return params, descriptions

    # ------------------------------------------------------------ I/O

    def state_dict_for_save(self, dtype):
        # drop the frozen PE base; it's loaded fresh from the .pt checkpoint
        return {
            k: v.detach().cpu().to(dtype)
            for k, v in self.state_dict().items()
            if not k.startswith("_pe_inner.")
        }

    def metadata_fields(self) -> dict[str, str]:
        meta: dict[str, str] = {
            "ss_num_ip_tokens": str(self.num_ip_tokens),
            "ss_encoder": self.encoder_name,
            "ss_encoder_dim": str(self.encoder_dim),
            "ss_context_dim": str(self.context_dim),
            "ss_num_blocks": str(self.num_blocks),
            "ss_hidden_size": str(self.hidden_size),
            "ss_num_heads": str(self.num_heads),
            "ss_resampler_layers": str(self.resampler_layers),
            "ss_resampler_heads": str(self.resampler_heads),
            "ss_ip_scale": str(self.ip_scale),
            "ss_pe_lora_enabled": str(self.pe_lora_enabled),
        }
        if self.pe_lora_enabled:
            meta["ss_pe_lora_rank"] = str(self.pe_lora_rank)
            meta["ss_pe_lora_alpha"] = str(self.pe_lora_alpha)
            meta["ss_pe_lora_qkv"] = str(self.pe_lora_qkv)
            meta["ss_pe_lora_attn_out"] = str(self.pe_lora_attn_out)
            meta["ss_pe_lora_mlp"] = str(self.pe_lora_mlp)
            meta["ss_pe_lora_layer_from"] = str(self.pe_lora_layer_from)
        return meta

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            sd = load_file(file)
        else:
            sd = torch.load(file, map_location="cpu")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        self._centroid_active = None  # force a re-check on next encode_ip_tokens
        # PE base params are intentionally absent (see state_dict_for_save);
        # filter them so the warning only flags real gaps.
        real_missing = [m for m in missing if not m.startswith("_pe_inner.")]
        if real_missing or unexpected:
            logger.warning(
                f"IPAdapterNetwork.load_state_dict: missing={real_missing}, unexpected={unexpected}"
            )
        else:
            logger.info(f"Loaded IP-Adapter weights from {file} ({len(sd)} tensors)")


# ----------------------------------------------------------------- patched forward


def _make_patched_forward(
    orig_attn, ip_net: "IPAdapterNetwork", block_idx: int, anima_attention
):
    """Build a closure that replaces ``Attention.forward`` for one block's cross-attn.

    Mirrors the original ``Attention.forward`` logic but additionally
    computes a decoupled SDPA over the cached IP K/V and adds
    ``scale * ip_attn_out`` to the text attention output before ``output_proj``.
    """

    def patched_forward(x, attn_params, context, rope_cos_sin=None):
        q, k, v = orig_attn.compute_qkv(x, context, rope_cos_sin)
        if q.dtype != v.dtype:
            if (
                not attn_params.supports_fp32 or attn_params.requires_same_dtype
            ) and torch.is_autocast_enabled():
                target_dtype = v.dtype
                q = q.to(target_dtype)
                k = k.to(target_dtype)
        text_qkv = [q, k, v]
        text_result = anima_attention.dispatch_attention(
            text_qkv, attn_params=attn_params
        )

        ip_k = orig_attn._ip_k_cached
        ip_v = orig_attn._ip_v_cached
        if ip_k is not None and ip_v is not None:
            # q: [B, S, n_h, d_h]; ip_k/ip_v: [B_ip, n_h, K, d_h] (SDPA layout,
            # cached by set_ip_tokens). Broadcast B_ip=1 -> B for inference,
            # where the same reference image conditions every CFG branch.
            B = q.shape[0]
            if ip_k.shape[0] == 1 and B > 1:
                ip_k = ip_k.expand(B, -1, -1, -1)
                ip_v = ip_v.expand(B, -1, -1, -1)
            elif ip_k.shape[0] != B:
                raise RuntimeError(
                    f"IP-Adapter K/V batch {ip_k.shape[0]} does not match q batch {B}"
                )
            q_sdpa = q.transpose(1, 2)  # [B, n_h, S, d_h]
            ip_out = F.scaled_dot_product_attention(q_sdpa, ip_k, ip_v)
            # Back to [B, S, n_h*d_h] to match text_result's flattened layout.
            ip_out = ip_out.transpose(1, 2).reshape(text_result.shape)
            # per-block 0-init gate, pre-multiplied by effective scale and
            # pre-cast in set_ip_tokens; still carries grad to ip_gate. At
            # step 0 this is exactly zero regardless of K/V magnitude.
            gate_scale = orig_attn._ip_gate_scale_cached
            ip_contribution = gate_scale * ip_out
            diag_sum = orig_attn._ip_diag_ratio_sum
            if ip_net._diag_enabled and diag_sum is not None:
                # detached, on-device update via the captured `orig_attn`
                # object — never a Python-int block_idx (torch.compile trap)
                with torch.no_grad():
                    ip_norm = ip_contribution.float().norm()
                    txt_norm = text_result.float().norm().clamp_min(1e-12)
                    diag_sum.add_(ip_norm / txt_norm)
                    orig_attn._ip_diag_count.add_(1)
            text_result = text_result + ip_contribution

        return orig_attn.output_dropout(orig_attn.output_proj(text_result))

    return patched_forward


# ----------------------------------------------------------------- trainer integration


class IPAdapterMethodAdapter(MethodAdapter):
    """Bridges IP-Adapter into AnimaTrainer's adapter dispatch.

    Owns the live PE-Core vision encoder (when not running off pre-cached
    features) so the trainer no longer needs an ``_vision_bundle`` field.

    Setup: validate the network exposes set_ip_tokens / encode_ip_tokens,
    enforce the cache_latents/ip_features_cache_to_disk constraint, load the
    live PE encoder when needed, enable runtime diagnostics.
    Step: encode the reference image (cached features or live PE) and prime
    per-block ``ip_k`` / ``ip_v`` on the network with whole-batch CFG dropout.
    Epoch end: dump ‖to_k_ip‖ / ‖to_v_ip‖ / ‖ip_out‖ / ‖text_result‖ ratios."""

    name = "ip_adapter"

    def __init__(self) -> None:
        self._vision_bundle: Optional[VisionEncoderBundle] = None
        self._epochs_completed: int = 0
        # Set by the validation-baseline sweep to force ip_tokens=None on the
        # next prime_for_forward (measures DiT-only behavior); restored after.
        self._force_drop_ip_tokens: bool = False
        # Set by the shuffled_ref baseline to source IP tokens from
        # ``batch["ip_features_shuffled"]`` instead of the matched-distinct
        # reference, isolating identity-specificity. None => primary reference.
        self._ref_override: Optional[str] = None

    def on_network_built(self, ctx: SetupCtx) -> None:
        args = ctx.args
        accelerator = ctx.accelerator
        net = ctx.network
        if not (hasattr(net, "set_ip_tokens") and hasattr(net, "encode_ip_tokens")):
            raise ValueError(
                "--use_ip_adapter requires a network module with set_ip_tokens / "
                "encode_ip_tokens (e.g. networks.methods.ip_adapter)."
            )
        pe_lora = getattr(net, "pe_lora_enabled", False)
        if pe_lora:
            # PE-LoRA needs gradient-flowing live encoding; cached features
            # can't track a moving encoder.
            if getattr(args, "ip_features_cache_to_disk", False):
                raise ValueError(
                    "PE-LoRA: ip_features_cache_to_disk=true is incompatible with "
                    "pe_lora_enabled=true (cached features can't track a moving "
                    "encoder). Set ip_features_cache_to_disk=false."
                )
            # cache_latents is fine (VAE is frozen, independent of PE); train.py
            # sets force_load_images_for_ip so the dataloader also surfaces
            # batch["images"]. subset.random_crop must stay False so the live
            # reload matches the cached latent's deterministic crop.
            accelerator.print(
                f"IP-Adapter + PE-LoRA: encoder owned by network "
                f"(rank={getattr(net, 'pe_lora_rank', None)}, "
                f"alpha={getattr(net, 'pe_lora_alpha', None)}, "
                f"image_drop_p={getattr(args, 'ip_image_drop_p', 0.1)}, "
                f"cache_latents={getattr(args, 'cache_latents', False)})"
            )
        else:
            cache_features = getattr(args, "ip_features_cache_to_disk", False)
            # cache_latents=true + live PE is supported via force_load_images_for_ip
            if cache_features:
                accelerator.print(
                    f"IP-Adapter: reading cached vision features "
                    f"(encoder={getattr(args, 'ip_encoder', 'pe')}, "
                    f"image_drop_p={getattr(args, 'ip_image_drop_p', 0.1)}) — "
                    "vision encoder NOT loaded."
                )
            else:
                self._vision_bundle = load_pe_encoder(
                    accelerator.device,
                    name=getattr(args, "ip_encoder", "pe"),
                    dtype=torch.bfloat16,
                )
                accelerator.print(
                    f"IP-Adapter: loaded vision encoder {self._vision_bundle.name} "
                    f"(d_enc={self._vision_bundle.d_enc}, "
                    f"image_drop_p={getattr(args, 'ip_image_drop_p', 0.1)})"
                )
        diag_epochs = int(getattr(args, "ip_diagnostics_epochs", 1) or 0)
        if hasattr(net, "set_diagnostics_enabled") and diag_epochs > 0:
            net.set_diagnostics_enabled(True, device=accelerator.device)
            if accelerator.is_main_process:
                net.diagnostic_summary(reset=True, log=True)

    def prime_for_forward(
        self, ctx: StepCtx, batch, latents: torch.Tensor, *, is_train: bool
    ) -> None:
        args = ctx.args
        accelerator = ctx.accelerator
        network = ctx.network
        if not hasattr(network, "set_ip_tokens"):
            return

        if self._force_drop_ip_tokens:
            network.set_ip_tokens(None)
            return

        drop_p = float(getattr(args, "ip_image_drop_p", 0.1) or 0.0)
        if is_train and drop_p > 0.0 and random.random() < drop_p:
            network.set_ip_tokens(None)
            return

        pe_lora = getattr(network, "pe_lora_enabled", False)
        feature_key = (
            "ip_features_shuffled"
            if self._ref_override == "shuffled"
            else "ip_features"
        )
        cached = batch.get(feature_key) if isinstance(batch, dict) else None
        if self._ref_override == "shuffled" and cached is None:
            # no shuffled reference for this val item; drop IP so the baseline still runs
            network.set_ip_tokens(None)
            return
        if pe_lora:
            images = batch.get("images") if isinstance(batch, dict) else None
            if images is None:
                raise RuntimeError(
                    "PE-LoRA expected batch['images'] but got None — set "
                    "cache_latents=false in the IP-Adapter config."
                )
            # no torch.no_grad: gradient flows through PE-LoRA (base PE frozen)
            ip_features = network.encode_images(images.to(accelerator.device)).to(
                ctx.weight_dtype
            )
        elif cached is not None:
            ip_features = cached.to(accelerator.device, dtype=ctx.weight_dtype)
        else:
            if self._vision_bundle is None:
                raise RuntimeError(
                    "IP-Adapter has no feature source: --ip_features_cache_to_disk "
                    "is off and the live vision encoder isn't loaded."
                )
            images = batch.get("images") if isinstance(batch, dict) else None
            if images is None:
                raise RuntimeError(
                    "IP-Adapter expected batch['images'] but got None — re-check "
                    "cache_latents=false in the IP-Adapter config, or set "
                    "ip_features_cache_to_disk=true with `make preprocess-pe`."
                )
            with torch.no_grad():
                feats_list = encode_pe_from_imageminus1to1(
                    self._vision_bundle,
                    images.to(accelerator.device),
                    same_bucket=True,  # dataloader bucketing guarantees per-batch shape
                )
                ip_features = torch.stack(feats_list, dim=0).to(
                    ctx.weight_dtype
                )  # [B, T_pe, d_enc]
        ip_tokens = network.encode_ip_tokens(ip_features)
        network.set_ip_tokens(ip_tokens)

    def validation_baselines(self) -> list[ValidationBaseline]:
        # Validation primary forward uses the matched_distinct reference (a
        # held-out different image of the target's identity — the deployment
        # condition). Two baselines isolate what the IP path contributes,
        # both relative to that primary:
        #   no_ip        — ip_tokens=None. delta>0 => the IP path helps at all.
        #   shuffled_ref — reference swapped for an unrelated image. delta>0
        #                  => the help is identity-specific, not "any image".
        # Phase-1 gate: matched_distinct beats both (both deltas positive).
        # NB: FM-MSE deltas are necessary-not-sufficient on Anima — confirm
        # with the CMMD/exp-test-ip ladder before trusting a win.
        def _no_ip_enter() -> None:
            self._force_drop_ip_tokens = True

        def _no_ip_exit() -> None:
            self._force_drop_ip_tokens = False

        def _shuffled_enter() -> None:
            self._ref_override = "shuffled"

        def _shuffled_exit() -> None:
            self._ref_override = None

        return [
            ValidationBaseline(name="no_ip", enter=_no_ip_enter, exit=_no_ip_exit),
            ValidationBaseline(
                name="shuffled_ref", enter=_shuffled_enter, exit=_shuffled_exit
            ),
        ]

    def on_epoch_end(self, ctx: StepCtx) -> None:
        # dump per-block norms + ip/text ratio for the finished epoch, then reset
        net = ctx.accelerator.unwrap_model(ctx.network)
        if hasattr(net, "diagnostic_summary"):
            net.diagnostic_summary(reset=True, log=True)
        # diagnostics add 56 fp32 norm reductions per step; keep them on only
        # for the warm-up window, then drop the overhead
        self._epochs_completed += 1
        diag_epochs = int(getattr(ctx.args, "ip_diagnostics_epochs", 1) or 0)
        if (
            self._epochs_completed >= diag_epochs
            and hasattr(net, "set_diagnostics_enabled")
            and getattr(net, "_diag_enabled", False)
        ):
            net.set_diagnostics_enabled(False)
            if ctx.accelerator.is_main_process:
                logger.info(
                    f"[IP-Adapter diag] disabled after {self._epochs_completed} epoch(s) "
                    f"(--ip_diagnostics_epochs={diag_epochs}); per-block norm overhead removed"
                )
