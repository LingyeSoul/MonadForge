"""ControlNet-LLLite network module for Anima — lightweight lateral injection.

Architecture (adapter-only — DiT frozen):

  conditioning image (pre-cached VAE latent, 4D [B, C, H/8, W/8])
      -> zero_conv_in (C → hidden_size, init weight=0)          [B, hidden, H/8, W/8]
      -> expand to 5D + patchify to match DiT token layout       [B, T, H', W', D]

  Per Anima Block (patched ``Block.forward``):

      original Block._forward(x, ...) → x_out
      + zero_conv_blocks[i](cond_feat) * scale                  lateral injection
      → x_out_patched

  where ``cond_feat`` is the pre-processed conditioning features
  passed through block[i]'s zero-conv projection.

Key properties:

  - Step-0 baseline equivalence: all zero_conv weights initialized to 0,
    so injection contributes nothing at init → identical to base DiT.
  - Zero convolutions are the trainable parameters; the DiT stays frozen.
  - Conditioning latent is pre-cached on disk via the standard
    ``cache_latents`` pipeline with a distinct suffix (_anima_cond.npz).
  - Composes with LoRA training (use_moe_style etc.) — they modify
    different paths (LoRA modifies weights, ControlNet modifies activations).

Train-time contract:

  Caller sets ``network.set_cond_latent(cond_latent)`` ONCE per batch
  before the DiT forward. Pass ``None`` (or call ``clear_cond_latent``)
  for unconditional / CFG-dropout passes — patched Block.forward then
  falls through to the baseline.

Inference:

  Load the conditioning image, VAE-encode to latent, call
  ``network.set_cond_latent(latent)`` before each denoising step.
  The patched blocks automatically inject the conditioning signal.
"""

from __future__ import annotations

import logging
import random
from typing import ClassVar, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.training.method_adapter import (
    MethodAdapter,
    SetupCtx,
    StepCtx,
)
from networks.methods.base import AdapterNetworkBase

logger = logging.getLogger(__name__)

# ── defaults ────────────────────────────────────────────────────────

DEFAULT_HIDDEN_SIZE = 2048
DEFAULT_NUM_BLOCKS = 28
DEFAULT_COND_DIM = 16  # VAE latent channels for Anima
DEFAULT_INJECTION_SCALE = 1.0


# ── zero-init conv ──────────────────────────────────────────────────


def _zero_conv(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Conv2d:
    """Create a zero-initialized convolution layer.

    At initialization, this contributes nothing to the output, ensuring
    step-0 baseline equivalence with the unmodified DiT.
    """
    conv = nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
    )
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


# ── network class ───────────────────────────────────────────────────


class ControlNetLLLiteNetwork(AdapterNetworkBase):
    """ControlNet-LLLite: per-block zero-conv lateral injection of conditioning features."""

    network_module: ClassVar[str] = "networks.methods.controlnet_lllite"
    network_spec: ClassVar[str] = "controlnet_lllite"
    mergeable: ClassVar[bool] = False  # not foldable into DiT weights

    def __init__(
        self,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        cond_channels: int = DEFAULT_COND_DIM,
        injection_scale: float = DEFAULT_INJECTION_SCALE,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.cond_channels = cond_channels
        self._injection_scale = injection_scale

        # Input projection: VAE latent channels → hidden_size
        # Operates on 2D spatial features (C, H, W)
        self.zero_conv_in = _zero_conv(cond_channels, hidden_size)

        # Per-block zero-conv injection: hidden_size → hidden_size
        self.zero_conv_blocks = nn.ModuleList(
            [_zero_conv(hidden_size, hidden_size) for _ in range(num_blocks)]
        )

        # ── runtime state (not saved) ──
        self._cond_feat: Optional[torch.Tensor] = None  # pre-processed cond features
        self._dit: Optional[nn.Module] = None
        self._block_modules: list[nn.Module] = []
        self._original_block_forwards: list = []
        self._patched: bool = False

    # ── conditioning API ────────────────────────────────────────────

    def set_cond_latent(self, cond_latent: Optional[torch.Tensor]) -> None:
        """Set the conditioning latent for the current batch.

        Args:
            cond_latent: 4D VAE latent [B, C, H/8, W/8] or None for dropout.
        """
        if cond_latent is None:
            self._cond_feat = None
            return
        # Pre-process through the input zero-conv to get cond features
        # that match the hidden_size dimension
        self._cond_feat = self.zero_conv_in(
            cond_latent.to(dtype=self.zero_conv_in.weight.dtype)
        )

    def clear_cond_latent(self) -> None:
        """Clear the conditioning latent (unconditional pass)."""
        self._cond_feat = None

    # ── trainer lifecycle ───────────────────────────────────────────

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        """Monkey-patch each DiT block to inject conditioning features."""
        del text_encoders, apply_text_encoder
        if not apply_unet:
            return
        if self._patched:
            logger.warning("ControlNetLLLiteNetwork.apply_to called twice — skipping")
            return
        if unet is None or not hasattr(unet, "blocks"):
            raise ValueError("apply_to requires the Anima DiT (unet) with .blocks")
        if len(unet.blocks) != self.num_blocks:
            raise ValueError(
                f"DiT has {len(unet.blocks)} blocks, ControlNet expects {self.num_blocks}. "
                "Re-create the network with matching num_blocks."
            )

        object.__setattr__(self, "_dit", unet)

        for idx, block in enumerate(unet.blocks):
            self._block_modules.append(block)
            self._original_block_forwards.append(block.forward)
            block._controlnet_net = self
            block.forward = _make_patched_block_forward(block, idx, self)

        self._patched = True
        logger.info(f"ControlNet-LLLite: patched {len(self._block_modules)} blocks")

    def remove_from(self):
        """Restore original block forwards."""
        for block, orig in zip(self._block_modules, self._original_block_forwards):
            block.forward = orig
            if hasattr(block, "_controlnet_net"):
                del block._controlnet_net
            if hasattr(block, "_controlnet_block_idx"):
                del block._controlnet_block_idx
        self._block_modules.clear()
        self._original_block_forwards.clear()
        object.__setattr__(self, "_dit", None)
        self._patched = False
        self._cond_feat = None

    # ── metadata ────────────────────────────────────────────────────

    def metadata_fields(self) -> dict[str, str]:
        return {
            "ss_num_blocks": str(self.num_blocks),
            "ss_hidden_size": str(self.hidden_size),
            "ss_cond_channels": str(self.cond_channels),
            "ss_injection_scale": str(self._injection_scale),
        }


# ── patched block forward ───────────────────────────────────────────


def _make_patched_block_forward(
    block: nn.Module,
    block_idx: int,
    controlnet: ControlNetLLLiteNetwork,
):
    """Create a patched Block.forward that injects conditioning features.

    The patched forward runs the original block, then adds the zero-conv
    projected conditioning features scaled by ``injection_scale``.

    Handles both eager (real 5D) and compile (native-flatten fake-5D) shapes.
    Under ``compile_blocks()``, the DiT reshapes ``(B, T, H, W, D)`` to
    ``(B, 1, seq_len, 1, D)`` before entering blocks. The patched forward
    detects this via ``T == 1 and W == 1`` and reshapes the conditioning
    projection accordingly.

    If no conditioning latent is set (unconditional / CFG-dropout), the
    original forward runs unchanged — zero overhead.
    """
    original_forward = block.forward
    # Capture per-block zero_conv directly to avoid Python-int specialization
    # under torch.compile (dynamo creates separate graphs per int constant).
    zero_conv = controlnet.zero_conv_blocks[block_idx]

    def patched_forward(
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params,
        rope_cos_sin=None,
        adaln_lora_B_T_3D=None,
    ) -> torch.Tensor:
        # Run the original block
        x_out = original_forward(
            x_B_T_H_W_D,
            emb_B_T_D,
            crossattn_emb,
            attn_params,
            rope_cos_sin=rope_cos_sin,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
        )

        # Inject conditioning if available
        cond_feat = controlnet._cond_feat
        if cond_feat is not None:
            scale = controlnet._injection_scale

            # Project conditioning features through per-block zero-conv
            cond_proj = zero_conv(cond_feat)  # [B, D, Hc, Wc]

            # Detect native-flatten mode: under compile_blocks(), the DiT
            # reshapes (B, T, H, W, D) → (B, 1, seq_len, 1, D).
            # In this case T=1 and W=1, with seq_len in the H dimension.
            T_dim = x_out.shape[1]
            W_dim = x_out.shape[3]
            is_flat = T_dim == 1 and W_dim == 1

            if is_flat:
                # Native-flatten mode: x_out is (B, 1, seq_len, 1, D)
                # cond_proj is (B, D, Hc, Wc) — reshape to match flat layout
                seq_len = x_out.shape[2]
                # Flatten spatial dims of cond_proj to match seq_len
                cond_flat = cond_proj.flatten(2)  # [B, D, Hc*Wc]
                cond_flat = cond_flat.transpose(1, 2)  # [B, Hc*Wc, D]
                if cond_flat.shape[1] != seq_len:
                    # Interpolate along sequence dimension if needed
                    cond_flat = cond_flat.transpose(1, 2).unsqueeze(-1)  # [B, D, S, 1]
                    cond_flat = F.interpolate(
                        cond_flat,
                        size=(seq_len, 1),
                        mode="bilinear",
                        align_corners=False,
                    )
                    cond_flat = cond_flat.squeeze(-1).transpose(1, 2)  # [B, seq_len, D]
                # Reshape to fake-5D: (B, 1, seq_len, 1, D)
                cond_proj_5d = cond_flat.unsqueeze(1).unsqueeze(
                    3
                )  # [B, 1, seq_len, 1, D]
                x_out = x_out + scale * cond_proj_5d
            else:
                # Eager mode: x_out is (B, T, H, W, D) with real spatial dims
                H, W = x_out.shape[2], x_out.shape[3]
                if cond_proj.shape[2:] != (H, W):
                    cond_proj = F.interpolate(
                        cond_proj, size=(H, W), mode="bilinear", align_corners=False
                    )
                # Reshape to [B, T, H, W, D] — broadcast T=1
                cond_proj = cond_proj.permute(0, 2, 3, 1).unsqueeze(
                    1
                )  # [B, 1, H, W, D]
                x_out = x_out + scale * cond_proj

        return x_out

    return patched_forward


# ── module-level factory functions (required by train.py) ───────────


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> ControlNetLLLiteNetwork:
    """Create a ControlNet-LLLite network from CLI args.

    Called by train.py when ``--network_module networks.methods.controlnet_lllite``.
    """
    del vae, text_encoders, neuron_dropout  # unused

    injection_scale = float(kwargs.get("injection_scale", DEFAULT_INJECTION_SCALE))
    cond_channels = int(kwargs.get("cond_channels", DEFAULT_COND_DIM))

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

    return ControlNetLLLiteNetwork(
        num_blocks=num_blocks,
        hidden_size=hidden_size,
        cond_channels=cond_channels,
        injection_scale=injection_scale,
        multiplier=multiplier,
    )


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
    """Load a ControlNet-LLLite network from saved weights.

    Called by train.py when ``--dim_from_weights`` is set or for inference.
    """
    del for_inference  # unused; present for interface parity

    if weights_sd is None:
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            metadata = f.metadata() or {}
            weights_sd = {k: f.get_tensor(k) for k in f.keys()}
    else:
        # Extract metadata from the weights file
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            metadata = f.metadata() or {}

    num_blocks = int(metadata.get("ss_num_blocks", DEFAULT_NUM_BLOCKS))
    hidden_size = int(metadata.get("ss_hidden_size", DEFAULT_HIDDEN_SIZE))
    cond_channels = int(metadata.get("ss_cond_channels", DEFAULT_COND_DIM))
    injection_scale = float(metadata.get("ss_injection_scale", DEFAULT_INJECTION_SCALE))

    network = ControlNetLLLiteNetwork(
        num_blocks=num_blocks,
        hidden_size=hidden_size,
        cond_channels=cond_channels,
        injection_scale=injection_scale,
        multiplier=multiplier,
    )
    return network, weights_sd


# ── trainer integration ─────────────────────────────────────────────


class ControlNetMethodAdapter(MethodAdapter):
    """Bridges ControlNet-LLLite into AnimaTrainer's adapter dispatch.

    Setup: validate the network exposes set_cond_latent / clear_cond_latent.
    Step: prime the conditioning latent from the batch before the DiT forward,
    with whole-batch CFG dropout.
    """

    name = "controlnet_lllite"

    def on_network_built(self, ctx: SetupCtx) -> None:
        net = ctx.network
        if not (hasattr(net, "set_cond_latent") and hasattr(net, "clear_cond_latent")):
            raise ValueError(
                "--use_controlnet requires a network module with set_cond_latent / "
                "clear_cond_latent (e.g. networks.methods.controlnet_lllite)."
            )
        ctx.accelerator.print(
            f"ControlNet-LLLite: lateral injection enabled "
            f"(drop_p={getattr(ctx.args, 'controlnet_drop_p', 0.1)}, "
            f"scale={net._injection_scale})"
        )

    def prime_for_forward(
        self, ctx: StepCtx, batch, latents: torch.Tensor, *, is_train: bool
    ) -> None:
        args = ctx.args
        network = ctx.network
        if not hasattr(network, "set_cond_latent"):
            return

        # CFG dropout: zero conditioning with probability drop_p
        drop_p = float(getattr(args, "controlnet_drop_p", 0.1) or 0.0)
        if is_train and drop_p > 0.0 and random.random() < drop_p:
            network.set_cond_latent(None)
            return

        # Load conditioning latent from the batch
        cond_latent = batch.get("conditioning_latent", None)
        if cond_latent is not None:
            cond_latent = cond_latent.to(ctx.accelerator.device, dtype=ctx.weight_dtype)
            network.set_cond_latent(cond_latent)
        else:
            network.set_cond_latent(None)
