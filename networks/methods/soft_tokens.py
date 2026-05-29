# Per-layer time-indexed soft tokens — SoftREPA-style parameterization
# (Lee et al., NeurIPS 2025, arXiv:2503.08250). DiT is frozen; trains a bank of
# K continuous vectors per (layer, t-bucket), spliced into crossattn_emb at each
# block independently. Anima's DiT is cross-attention (not MM-DiT), so
# crossattn_emb doesn't evolve through blocks — no strip/re-prepend needed.
#
# Splice: end-of-sequence overwrite of the K zero-padding tail slots, keeping
# crossattn_emb shape static for torch.compile. Zero-padded positions are
# cross-attention sinks (Anima text-encoder padding invariant), so writing into
# them gives the tokens attention mass without changing seqlen.
#
# vs postfix.py: postfix splices once at the cached adapter output; soft tokens
# splice per-block via monkey-patched Block.forward (ReFT-pattern).
#
# Inference: library/inference/generation.py + networks/spectrum.py call
# append_postfix(..., timesteps=t) per CFG branch before each forward. Spectrum
# cached steps skip the blocks, so soft tokens no-op there (composes with
# --spectrum).

import os
from typing import Optional

import torch
import torch.nn as nn

from library.log import setup_logging
from library.training.method_adapter import (
    ForwardArtifacts,
    MethodAdapter,
    StepCtx,
)
from networks.methods.base import AdapterNetworkBase

import logging

setup_logging()
logger = logging.getLogger(__name__)

# Anima cached crossattn_emb dimension (Qwen3 hidden size, post LLM-adapter).
DEFAULT_EMBED_DIM = 1024

# Contrastive negative-sourcing modes (docs/proposal/soft_tokens_contrastive.md):
# ``shuffled`` = unrelated cached-TE negative; ``jaccard`` = shuffled but logit
# down-weighted by caption tag-overlap; ``hard`` = same-artist/different-character
# sibling (falls back to shuffled for orphan artists); ``hard_backoff`` = tiered
# same-artist → same-copyright → shuffled (copyright tier rescues hard's fallback).
CONTRASTIVE_MODES = ("shuffled", "jaccard", "hard", "hard_backoff")

# Contrastive objective, sharing the extra-forward plumbing:
#   ``infonce`` — SoftREPA InfoNCE over cached-TE negatives (default).
#   ``agsm``    — Alignment-Guided Score Matching: bounded target-shift
#                 ``v_target ± γ_z·Ã(t)·Δ_j`` with per-candidate Plackett–Luce
#                 guidance ``Δ_j = v̂_ema_j − Σ_k w_k v̂_ema_k`` read off an EMA of
#                 the bank's own preds (docs/proposal/soft_tokens_agsm.md, Phase 2).
CONTRASTIVE_OBJECTIVES = ("infonce", "agsm")


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
    num_tokens = network_dim if network_dim is not None else 4
    embed_dim = int(kwargs.get("embed_dim", DEFAULT_EMBED_DIM))
    n_layers = int(kwargs.get("n_layers", 10))
    n_t_buckets = int(kwargs.get("n_t_buckets", 100))
    init_std = float(kwargs.get("init_std", 0.02))
    splice_position = kwargs.get("splice_position", "end_of_sequence")
    contrastive_weight = float(kwargs.get("contrastive_weight", 0.0))
    contrastive_k = int(kwargs.get("contrastive_k", 1))
    contrastive_negative_mode = str(kwargs.get("contrastive_negative_mode", "shuffled"))
    contrastive_tau = float(kwargs.get("contrastive_tau", 0.5))
    contrastive_warmup_ratio = float(kwargs.get("contrastive_warmup_ratio", 0.1))
    contrastive_jaccard_alpha = float(kwargs.get("contrastive_jaccard_alpha", 1.0))
    contrastive_every_n = int(kwargs.get("contrastive_every_n", 1))
    contrastive_objective = str(kwargs.get("contrastive_objective", "infonce"))
    agsm_gamma = float(kwargs.get("agsm_gamma", 0.5))
    _agsm_gamma_neg = kwargs.get("agsm_gamma_neg")
    agsm_gamma_neg = None if _agsm_gamma_neg is None else float(_agsm_gamma_neg)
    agsm_ema_decay = float(kwargs.get("agsm_ema_decay", 0.99))
    agsm_dual_bank = str(kwargs.get("agsm_dual_bank", "false")).lower() in (
        "true",
        "1",
        "yes",
    )
    network = SoftTokensNetwork(
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_t_buckets=n_t_buckets,
        init_std=init_std,
        splice_position=splice_position,
        contrastive_weight=contrastive_weight,
        contrastive_k=contrastive_k,
        contrastive_negative_mode=contrastive_negative_mode,
        contrastive_tau=contrastive_tau,
        contrastive_warmup_ratio=contrastive_warmup_ratio,
        contrastive_jaccard_alpha=contrastive_jaccard_alpha,
        contrastive_every_n=contrastive_every_n,
        contrastive_objective=contrastive_objective,
        agsm_gamma=agsm_gamma,
        agsm_gamma_neg=agsm_gamma_neg,
        agsm_ema_decay=agsm_ema_decay,
        agsm_dual_bank=agsm_dual_bank,
        multiplier=multiplier,
    )
    return network


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
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    tokens = weights_sd.get("tokens")
    t_offsets = weights_sd.get("t_offsets.weight")
    if tokens is None or t_offsets is None:
        raise ValueError(
            f"soft_tokens weight file must contain 'tokens' and 't_offsets.weight' "
            f"(got keys: {list(weights_sd.keys())[:8]})"
        )
    # Dual-bank (AGSM Phase 3a) checkpoints carry a leading branch axis on
    # ``tokens`` (n_banks, n_layers, K, D); single-bank stay 3D. Inference uses
    # ONLY ψ⁺ (branch 0) — Appendix H: injecting ψ⁻ at inference over-suppresses
    # detail — so build a single-bank net and let ``load_weights`` slice the
    # branch-0 weights. Resume (for_inference=False) keeps both banks.
    if tokens.dim() == 4:
        file_n_banks, n_layers, num_tokens, embed_dim = tokens.shape
    else:
        file_n_banks = 1
        n_layers, num_tokens, embed_dim = tokens.shape
    build_dual = (file_n_banks > 1) and (not for_inference)
    n_t_buckets = t_offsets.shape[0]
    # Splice position is a runtime knob, not learned — read from metadata, CLI
    # kwargs win for post-hoc overrides.
    metadata_splice = None
    if file is not None and os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            meta = f.metadata() or {}
            metadata_splice = meta.get("ss_splice_position")
    splice_position = kwargs.get(
        "splice_position", metadata_splice or "end_of_sequence"
    )
    network = SoftTokensNetwork(
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_t_buckets=n_t_buckets,
        init_std=0.0,  # weights are loaded; init_std doesn't matter
        splice_position=splice_position,
        # Contrastive is training-only (extra forwards, no learned params) — off
        # on the inference path.
        contrastive_weight=0.0,
        agsm_dual_bank=build_dual,
        multiplier=multiplier,
    )
    return network, weights_sd


class SoftTokensNetwork(AdapterNetworkBase):
    """Per-layer time-indexed soft tokens.

    Parameters:
      - tokens: (n_layers, K, D) — base per-layer tokens, small-std init.
      - t_offsets: Embedding(n_t_buckets, n_layers * D) — per-(t_bucket, layer)
        broadcast offset (one D-vector applied to every token in the layer).
        Zero-init so step 0 reproduces the un-time-conditioned base tokens.

    Param count: n_layers·K·D + n_t_buckets·n_layers·D
    With defaults (n_layers=10, K=4, D=1024, n_t_buckets=100): 40k + 1.0M ≈ 1.05M.
    """

    network_module = "networks.methods.soft_tokens"
    network_spec = "soft_tokens"

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        n_layers: int = 10,
        n_t_buckets: int = 100,
        init_std: float = 0.02,
        splice_position: str = "end_of_sequence",
        contrastive_weight: float = 0.0,
        contrastive_k: int = 1,
        contrastive_negative_mode: str = "shuffled",
        contrastive_tau: float = 0.5,
        contrastive_warmup_ratio: float = 0.1,
        contrastive_jaccard_alpha: float = 1.0,
        contrastive_every_n: int = 1,
        contrastive_objective: str = "infonce",
        agsm_gamma: float = 0.5,
        agsm_gamma_neg: Optional[float] = None,
        agsm_ema_decay: float = 0.99,
        agsm_dual_bank: bool = False,
        multiplier: float = 1.0,
    ):
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        # Upper-bound check against actual block count happens in apply_to().
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if n_t_buckets <= 0:
            raise ValueError(f"n_t_buckets must be positive, got {n_t_buckets}")
        if splice_position not in ("front_of_padding", "end_of_sequence"):
            raise ValueError(
                f"splice_position must be 'front_of_padding' or 'end_of_sequence', "
                f"got {splice_position!r}"
            )
        if contrastive_negative_mode not in CONTRASTIVE_MODES:
            raise ValueError(
                f"contrastive_negative_mode must be one of {CONTRASTIVE_MODES}, "
                f"got {contrastive_negative_mode!r}"
            )
        if contrastive_objective not in CONTRASTIVE_OBJECTIVES:
            raise ValueError(
                f"contrastive_objective must be one of {CONTRASTIVE_OBJECTIVES}, "
                f"got {contrastive_objective!r}"
            )
        if contrastive_weight > 0.0:
            if contrastive_k < 1:
                raise ValueError(f"contrastive_k must be >= 1, got {contrastive_k}")
            if contrastive_objective == "infonce" and contrastive_tau <= 0.0:
                raise ValueError(
                    f"contrastive_tau must be positive, got {contrastive_tau}"
                )
            if contrastive_objective == "agsm" and not (0.0 < agsm_ema_decay < 1.0):
                raise ValueError(
                    f"agsm_ema_decay must be in (0, 1), got {agsm_ema_decay}"
                )

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_t_buckets = n_t_buckets
        self.splice_position = splice_position
        self.multiplier = multiplier

        # Contrastive objective (training-only; extra forwards live in
        # ``SoftTokensMethodAdapter``). ``_contrastive_target_weight`` gates
        # composer activation; ``_contrastive_weight`` is the live warmup-held
        # value the loss handler multiplies by.
        self._contrastive_target_weight = float(contrastive_weight)
        self._contrastive_warmup_ratio = float(contrastive_warmup_ratio)
        self._contrastive_tau = float(contrastive_tau)
        self.contrastive_k = int(contrastive_k)
        self.contrastive_negative_mode = str(contrastive_negative_mode)
        self.contrastive_jaccard_alpha = float(contrastive_jaccard_alpha)
        self._contrastive_weight = (
            0.0
            if self._contrastive_warmup_ratio > 0.0
            else self._contrastive_target_weight
        )
        # Cadence: run the contrastive negative forwards only every Nth optimizer
        # step. NOT auto-scaled — effective strength is (weight × 1/N); co-tune
        # ``contrastive_weight`` to hold it constant. 1 = every step (default).
        # ``_contrastive_fire_this_step`` is recomputed each step by
        # ``step_contrastive_warmup`` on the optimizer-step index.
        self._contrastive_every_n = max(1, int(contrastive_every_n))
        self._contrastive_fire_this_step = True

        # AGSM target-shift objective (docs/proposal/soft_tokens_agsm.md, Phase 2).
        # ``contrastive_objective`` selects the loss math in the adapter. Phase 2
        # is single-bank (ψ⁺=ψ⁻=this bank, only crossattn_emb differs), Ã(t)=1;
        # Δ is read off an EMA shadow of the bank's own preds (self-distillation).
        self.contrastive_objective = str(contrastive_objective)
        # Dual token banks (AGSM Phase 3a, paper §3.3): branch 0 = ψ⁺ (spliced on
        # the anchor + positive value passes + ALL inference), branch 1 = ψ⁻
        # (negative value passes, training-only). Single bank (n_banks=1) is
        # bit-identical to Phase 2 and keeps the 3D on-disk format. Only meaningful
        # under the agsm objective, but the param layout is set here at construction.
        self.agsm_dual_bank = bool(agsm_dual_bank)
        self.n_banks = 2 if self.agsm_dual_bank else 1
        self._agsm_gamma = float(agsm_gamma)
        # γ⁻ — the negative-branch guidance scale (paper's γ⁻; SD3/flow used
        # (γ⁺, γ⁻) = (1, 0.1), a deliberately weaker negative push). Falls back
        # to the symmetric γ⁺ when unset for back-compat.
        self._agsm_gamma_neg = (
            float(agsm_gamma) if agsm_gamma_neg is None else float(agsm_gamma_neg)
        )
        self._agsm_ema_decay = float(agsm_ema_decay)
        # EMA shadow of the bank, lazily cloned from the live params and refreshed
        # per optimizer step. Plain attributes (not buffers/params) so they stay
        # out of state_dict and carry no gradient.
        self._tokens_ema: Optional[torch.Tensor] = None
        self._t_offsets_ema: Optional[torch.Tensor] = None

        # Single bank: (n_layers, K, D) — unchanged 3D shape + on-disk format.
        # Dual bank: (2, n_layers, K, D) with a leading branch axis (ψ⁺, ψ⁻),
        # independently initialized so the two guidance regions don't start tied.
        if self.n_banks == 1:
            self.tokens = nn.Parameter(
                torch.randn(n_layers, num_tokens, embed_dim) * init_std
            )
        else:
            self.tokens = nn.Parameter(
                torch.randn(self.n_banks, n_layers, num_tokens, embed_dim) * init_std
            )
        # Per-(bucket, [branch], layer) D-vector offset, broadcast across the
        # K-token axis at lookup (one D-vector per layer per bucket, not K).
        # Zero-init = identity perturbation at step 0. Dual stacks the branches in
        # the column axis (bank-major: [ψ⁺ layers | ψ⁻ layers]) so the ψ⁺ slice is
        # the first n_layers·D columns.
        self.t_offsets = nn.Embedding(
            n_t_buckets, self.n_banks * n_layers * embed_dim
        )
        nn.init.zeros_(self.t_offsets.weight)

        # Step-scoped state set by append_postfix() per forward and consumed by
        # the per-block hooks. Plain attributes (recreated each step).
        # _step_seqlens only used for front_of_padding splice.
        self._step_layer_tokens: Optional[torch.Tensor] = None  # (n_layers, B, K, D)
        self._step_seqlens: Optional[torch.Tensor] = None  # (B,) int

        # Kept so apply_to() could un-monkey-patch later (unused but cheap).
        self._block_refs: list[nn.Module] = []
        self._original_forwards: list = []

        n_token_params = self.tokens.numel()
        n_offset_params = self.t_offsets.weight.numel()
        bank_note = (
            "dual bank (ψ⁺/ψ⁻, ψ⁺-only at inference)"
            if self.n_banks == 2
            else "single bank"
        )
        logger.info(
            f"SoftTokensNetwork: {n_layers} layers × {num_tokens} tokens × dim {embed_dim}, "
            f"{n_t_buckets} t-buckets, splice={splice_position}, {bank_note} → "
            f"{n_token_params + n_offset_params} params "
            f"({n_token_params} base + {n_offset_params} t-offset)"
        )

    # Sentinel for train.py's ``hasattr(network, "append_postfix")`` branch —
    # makes it call append_postfix(..., timesteps=...) per step, which we use
    # only to compute the step-scoped tokens (crossattn_emb passes through; the
    # splice happens in the per-block hooks below).
    @property
    def num_postfix_tokens(self) -> int:
        return self.num_tokens

    def _bucketize(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map sigma in [0, 1] (Anima convention) to integer buckets [0, n_t_buckets).

        Outside-range values are clamped, so callers don't need to pre-clamp.
        """
        t = timesteps.detach().float().flatten()
        idx = torch.floor(t * self.n_t_buckets).long()
        return idx.clamp(min=0, max=self.n_t_buckets - 1)

    def append_postfix(
        self,
        crossattn_emb: torch.Tensor,
        crossattn_seqlens: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-step layer tokens and cache them for the block hooks.

        Returns crossattn_emb unchanged — splice happens per-block in the hooks
        installed by ``apply_to()``. We just piggyback on train.py's existing
        per-step trainer hook to receive timesteps.
        """
        if timesteps is None:
            raise ValueError(
                "soft_tokens requires timesteps (per-step) — train.py and the "
                "inference loop (library/inference/generation.py, "
                "networks/spectrum.py) both pass this per CFG branch each step"
            )
        # Anchor / inference forward always splices ψ⁺ (branch 0). The negative
        # ψ⁻ branch is only ever spliced from the training adapter's value passes.
        self._set_step_tokens(timesteps, crossattn_seqlens, use_ema=False, branch=0)
        return crossattn_emb

    def _layer_tokens_from(
        self,
        tokens: torch.Tensor,
        t_offsets_weight: torch.Tensor,
        timesteps: torch.Tensor,
        branch: int = 0,
    ) -> torch.Tensor:
        """Per-step (n_layers, B, K, D) tokens for a given (tokens, t_offsets)
        pair. Factored out of ``append_postfix`` so the same math runs against
        either the live bank or its EMA shadow (AGSM Δ forwards).

        ``branch`` selects the bank (0=ψ⁺, 1=ψ⁻) for dual-bank nets; ignored
        (always ψ⁺) when ``n_banks == 1`` so the single-bank path is unchanged."""
        B = int(timesteps.detach().flatten().shape[0])
        bucket_idx = self._bucketize(timesteps)  # (B,)
        offsets_full = nn.functional.embedding(bucket_idx, t_offsets_weight)
        if self.n_banks == 1:
            # (B, n_layers * D) → (B, n_layers, D)
            offsets = offsets_full.view(B, self.n_layers, self.embed_dim)
            base = tokens  # (n_layers, K, D)
        else:
            # (B, n_banks * n_layers * D) → pick this branch → (B, n_layers, D)
            offsets = offsets_full.view(
                B, self.n_banks, self.n_layers, self.embed_dim
            )[:, branch]
            base = tokens[branch]  # (n_layers, K, D)
        # (n_layers, K, D) → (1, n_layers, K, D), broadcast over batch + over K.
        per_step = base.unsqueeze(0) + offsets.unsqueeze(2)  # (B, n_layers, K, D)
        # Transpose to (n_layers, B, K, D) for cheap per-layer indexing in the
        # block hook closure.
        return per_step.transpose(0, 1).contiguous()

    def _set_step_tokens(
        self,
        timesteps: torch.Tensor,
        crossattn_seqlens: Optional[torch.Tensor],
        use_ema: bool = False,
        branch: int = 0,
    ) -> None:
        """Compute + cache the per-step layer tokens read by the block hooks.

        ``use_ema=True`` primes the splice from the EMA shadow bank instead of
        the live params (no grad — the AGSM guidance direction Δ is detached).
        ``branch`` selects ψ⁺ (0) / ψ⁻ (1) for dual-bank nets.
        """
        if use_ema:
            self._ensure_bank_ema()
            tokens, t_offsets_weight = self._tokens_ema, self._t_offsets_ema
        else:
            tokens, t_offsets_weight = self.tokens, self.t_offsets.weight
        self._step_layer_tokens = self._layer_tokens_from(
            tokens, t_offsets_weight, timesteps, branch=branch
        )
        # front_of_padding needs per-sample seqlens at hook time; end_of_sequence
        # ignores them. Cache regardless so the hook doesn't have to know which
        # mode is active (the splice branch reads or skips).
        self._step_seqlens = (
            crossattn_seqlens.detach().to(torch.long)
            if crossattn_seqlens is not None
            else None
        )

    def _make_block_hook(self, layer_idx: int, org_forward):
        """Closure that splices layer_idx's tokens into crossattn_emb tail.

        Block.forward signature (from library/anima/models.py:1179):
          forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, attn_params,
                  rope_cos_sin=None, adaln_lora_B_T_3D=None)
        """
        K = self.num_tokens
        splice_position = self.splice_position
        net = self  # capture network for state lookup

        def hook(
            x_B_T_H_W_D,
            emb_B_T_D,
            crossattn_emb,
            attn_params,
            *args,
            **kwargs,
        ):
            step_tokens = net._step_layer_tokens
            if step_tokens is not None:
                # (B, K, D) for this layer. Cast to crossattn dtype/device.
                layer_tok = step_tokens[layer_idx].to(
                    dtype=crossattn_emb.dtype, device=crossattn_emb.device
                )
                S = crossattn_emb.shape[1]
                if S < K:
                    raise RuntimeError(
                        f"crossattn_emb seqlen {S} < num_tokens {K}; cannot splice"
                    )
                if splice_position == "end_of_sequence":
                    # Overwrite the K tail (zero-padding) slots. torch.cat
                    # preserves autograd through both branches.
                    crossattn_emb = torch.cat(
                        [crossattn_emb[:, : S - K, :], layer_tok], dim=1
                    )
                else:  # front_of_padding
                    # Place K tokens at [seqlens[i], seqlens[i]+K) per sample —
                    # displaces the strongest sinks. scatter() preserves grad
                    # on the written values.
                    seqlens = net._step_seqlens
                    if seqlens is None:
                        raise RuntimeError(
                            "front_of_padding splice requires crossattn_seqlens; "
                            "trainer must pass it to append_postfix()"
                        )
                    offsets = seqlens.to(crossattn_emb.device).unsqueeze(
                        1
                    ) + torch.arange(K, device=crossattn_emb.device)  # (B, K)
                    D = crossattn_emb.shape[-1]
                    idx = offsets.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
                    crossattn_emb = crossattn_emb.scatter(1, idx, layer_tok)
            return org_forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                attn_params,
                *args,
                **kwargs,
            )

        return hook

    def apply_to(
        self,
        text_encoders,
        unet,
        apply_text_encoder=True,
        apply_unet=True,
    ):
        """Monkey-patch the first n_layers DiT blocks with the splice hook."""
        blocks = getattr(unet, "blocks", None)
        if blocks is None:
            raise RuntimeError("unet has no .blocks attribute — not an Anima DiT?")
        if len(blocks) < self.n_layers:
            raise RuntimeError(
                f"unet has {len(blocks)} blocks but n_layers={self.n_layers}"
            )
        self._block_refs = []
        self._original_forwards = []
        for k in range(self.n_layers):
            block = blocks[k]
            org_forward = block.forward
            block.forward = self._make_block_hook(k, org_forward)
            self._block_refs.append(block)
            self._original_forwards.append(org_forward)
        logger.info(
            f"soft_tokens: monkey-patched first {self.n_layers} of {len(blocks)} "
            f"DiT blocks (end-of-sequence splice, K={self.num_tokens})"
        )

    # ── Standard adapter API ────────────────────────────────────────────

    def get_trainable_params(self):
        return [self.tokens, self.t_offsets.weight]

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        del text_encoder_lr
        lr = unet_lr or default_lr
        params = [{"params": self.get_trainable_params(), "lr": lr}]
        descriptions = ["soft_tokens(tokens+t_offsets)"]
        return params, descriptions

    def state_dict_for_save(self, dtype):
        return {
            "tokens": self.tokens.detach().clone().cpu().to(dtype),
            "t_offsets.weight": self.t_offsets.weight.detach().clone().cpu().to(dtype),
        }

    def metadata_fields(self) -> dict[str, str]:
        return {
            "ss_num_tokens": str(self.num_tokens),
            "ss_embed_dim": str(self.embed_dim),
            "ss_n_layers": str(self.n_layers),
            "ss_n_t_buckets": str(self.n_t_buckets),
            "ss_splice_position": self.splice_position,
            # Contrastive objective is training-only (no learned params), but
            # stamp the config for run provenance.
            "ss_contrastive_weight": str(self._contrastive_target_weight),
            "ss_contrastive_k": str(self.contrastive_k),
            "ss_contrastive_negative_mode": self.contrastive_negative_mode,
            "ss_contrastive_tau": str(self._contrastive_tau),
            "ss_contrastive_warmup_ratio": str(self._contrastive_warmup_ratio),
            "ss_contrastive_every_n": str(self._contrastive_every_n),
            "ss_contrastive_objective": self.contrastive_objective,
            "ss_agsm_gamma": str(self._agsm_gamma),
            "ss_agsm_gamma_neg": str(self._agsm_gamma_neg),
            "ss_agsm_ema_decay": str(self._agsm_ema_decay),
            # Dual-bank (Phase 3a) provenance. ``ss_n_banks`` is also the load-side
            # signal: inference reads only the ψ⁺ slice regardless, but a 2-bank
            # stamp tells a resume to rebuild both branches.
            "ss_agsm_dual_bank": "true" if self.agsm_dual_bank else "false",
            "ss_n_banks": str(self.n_banks),
        }

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        if "tokens" not in weights_sd or "t_offsets.weight" not in weights_sd:
            raise ValueError(
                f"Missing required keys in soft_tokens checkpoint "
                f"(got: {list(weights_sd.keys())[:8]})"
            )
        tok, toff = self._select_load_weights(
            weights_sd["tokens"], weights_sd["t_offsets.weight"]
        )
        self.tokens.data.copy_(tok)
        self.t_offsets.weight.data.copy_(toff)
        logger.info(
            f"Loaded soft_tokens weights: tokens={tuple(self.tokens.shape)}, "
            f"t_offsets={tuple(self.t_offsets.weight.shape)} (n_banks={self.n_banks})"
        )

    def _select_load_weights(
        self, file_tokens: torch.Tensor, file_t_offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconcile a checkpoint's bank layout with this net's ``n_banks``.

        - file 4D + this net single (inference of a dual checkpoint): slice the
          ψ⁺ branch (index 0) — Appendix H keeps ψ⁻ training-only.
        - file 3D + this net single: pass through (the Phase-2 path).
        - file 4D + this net dual (resume): pass through.
        - file 3D + this net dual: a single-bank checkpoint can't seed both
          branches unambiguously → hard error.
        """
        file_dual = file_tokens.dim() == 4
        if self.n_banks == 1:
            if not file_dual:
                return file_tokens, file_t_offsets
            # ψ⁺ slice: tokens[0]; t_offsets first n_layers·D columns (bank-major).
            n_cols = self.n_layers * self.embed_dim
            return file_tokens[0], file_t_offsets[:, :n_cols]
        # dual net
        if not file_dual:
            raise ValueError(
                "agsm_dual_bank=True but the checkpoint has a single (3D) bank — "
                "a single-bank checkpoint cannot initialize both ψ⁺/ψ⁻ branches; "
                "train dual from scratch or resume a dual checkpoint."
            )
        return file_tokens, file_t_offsets

    def metrics(self, ctx) -> dict[str, float]:
        """TensorBoard bank-state collapse/divergence diagnostics.

        ``tokens_mean_cos`` ~0 = orthogonal (good), ~1 = slot collapse;
        ``tokens_mean_norm`` blowing up = magnitude diverging;
        ``offset_mean_norm`` ~0 = t-offset buckets not training (check LR).
        """
        del ctx
        out: dict[str, float] = {}

        # Diagnostics are on ψ⁺ (branch 0), the bank kept at inference. For dual
        # banks also surface the ψ⁻ magnitude so a diverging negative branch is
        # visible. Single bank: tokens is already (L, K, D).
        tokens_all = self.tokens.detach()
        if self.n_banks > 1:
            out["soft_tokens/tokens_neg_mean_norm"] = float(
                tokens_all[1].flatten(1).norm(dim=-1).mean().item()
            )
            tokens_psi = tokens_all[0]  # (L, K, D)
        else:
            tokens_psi = tokens_all  # (L, K, D)

        # Batched over the layer axis → 3 host syncs for the whole bank.
        if self.num_tokens >= 2 and self.n_layers > 0:
            tokens = tokens_psi  # (L, K, D)
            K = tokens.shape[1]
            iu = torch.triu_indices(K, K, offset=1, device=tokens.device)
            # Mean pairwise cos over all (layer, pair) — equal pair count per
            # layer so this equals the mean of per-layer means.
            zn = torch.nn.functional.normalize(tokens, dim=-1, eps=1e-8)
            gram = zn @ zn.transpose(1, 2)  # (L, K, K)
            out["soft_tokens/tokens_mean_cos"] = float(
                gram[:, iu[0], iu[1]].mean().item()
            )
            # Squared pairwise distances ‖a‖²+‖b‖²−2a·b; clamp subtraction round-off.
            sq = tokens.pow(2).sum(-1)  # (L, K)
            d_sq = (
                sq.unsqueeze(2) + sq.unsqueeze(1) - 2.0 * (tokens @ tokens.transpose(1, 2))
            ).clamp_min(0.0)  # (L, K, K)
            out["soft_tokens/tokens_min_d_sq"] = float(
                d_sq[:, iu[0], iu[1]].min().item()
            )
            out["soft_tokens/tokens_mean_norm"] = float(
                tokens.flatten(1).norm(dim=-1).mean().item()
            )
        # t_offsets.weight is (n_t_buckets, n_banks·n_layers·D); report the ψ⁺
        # (branch 0) per-(layer) offset norm, matching the single-bank metric.
        offsets_psi = self.t_offsets.weight.detach().view(
            self.n_t_buckets, self.n_banks, self.n_layers, self.embed_dim
        )[:, 0]
        out["soft_tokens/offset_mean_norm"] = float(
            offsets_psi.permute(1, 0, 2).flatten(1).norm(dim=-1).mean().item()
        )
        return out

    def step_contrastive_warmup(
        self, global_step: int, max_train_steps: int, accum: int = 1
    ) -> None:
        """Activate the contrastive objective past its warmup window and decide
        whether the negatives fire this step.

        Warmup: ``_contrastive_weight`` holds at 0 for the first
        ``_contrastive_warmup_ratio`` of steps, then flips to the target (no-op
        when target is 0) — lets plain FM shape a non-degenerate bank before the
        contrast pulls on it.

        Cadence: ``global_step`` is the micro-batch counter; the firing decision
        is taken on the optimizer-step index (``global_step // accum``) so every
        micro-batch in an accumulation window agrees (else partial, accum-coupled
        contrastive grads).
        """
        every_n = int(self._contrastive_every_n)
        accum = max(1, int(accum))
        optimizer_step = int(global_step) // accum
        self._contrastive_fire_this_step = (
            every_n <= 1 or optimizer_step % every_n == 0
        )

        target = float(self._contrastive_target_weight)
        ratio = float(self._contrastive_warmup_ratio)
        if target <= 0.0:
            return
        if ratio <= 0.0 or max_train_steps <= 0:
            self._contrastive_weight = target
            return
        warmup_steps = int(max_train_steps * ratio)
        self._contrastive_weight = 0.0 if global_step < warmup_steps else target

    def contrastive_loss(
        self,
        v_pos: torch.Tensor,
        v_neg: torch.Tensor,
        v_target: torch.Tensor,
        neg_penalty: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """SoftREPA-style InfoNCE over cached-TE negatives (B=1-adapted).

        Each forward shares the same ``(x_t, ε, t)`` and spliced soft tokens;
        only ``crossattn_emb`` differs (matched vs. mismatched text). The logit
        for a forward is the negative flow-matching error against the shared
        velocity target, scaled by τ::

            ℓ_* = -‖v_* − v_target‖² / τ            (mean over C·H·W per sample)
            L   = -log( exp(ℓ_pos) / Σ_{pos,neg} exp(ℓ_*) )

        Gradient flows to the soft tokens (via every ``v_*``) to make the
        matched text explain the anchor's latent better than mismatched text —
        i.e. to sharpen the cross-attention's text discrimination.

        ``neg_penalty`` (the ``jaccard`` mode's ``α·s``, shape ``(B, k)``) is
        subtracted from each negative logit before the softmax — a negative that
        shares tags with the anchor becomes a less-surprising mismatch and pulls
        less gradient. ``None`` ⇒ plain InfoNCE (``shuffled`` / ``hard``).

        Shapes
        ------
        v_pos, v_target : ``(B, C, H, W)``
        v_neg           : ``(B, k, C, H, W)``
        neg_penalty     : ``(B, k)`` or None

        Returns
        -------
        (loss_scalar, diagnostics) where diagnostics carries the contrastive
        accuracy (pos beats every negative) and the mean pos−neg logit gap for
        TensorBoard.
        """
        logit_pos, logits_neg = self._velocities_to_logits(
            v_pos, v_neg, v_target, neg_penalty
        )
        loss = self._infonce_from_logits(logit_pos, logits_neg)
        return loss, self._contrastive_diagnostics(logit_pos, logits_neg)

    def _velocities_to_logits(
        self,
        v_pos: torch.Tensor,
        v_neg: torch.Tensor,
        v_target: torch.Tensor,
        neg_penalty: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample FM-error logits for the InfoNCE softmax.

        Differentiable in every velocity arg, so the caller can ``.detach()``
        whichever branch it drops from the graph — this is how the grad-cache
        path splits ∂L/∂v_pos and ∂L/∂v_neg without duplicating the τ/penalty
        math. Returns ``(logit_pos (B,), logits_neg (B, k))``.
        """
        tau = float(self._contrastive_tau)
        vp = v_pos.float()
        vt = v_target.float()
        vn = v_neg.float()
        B = vp.shape[0]
        k = vn.shape[1]
        # Per-sample mean-squared FM error → logit. Reduce over all non-batch
        # dims so τ has a stable scale across resolutions.
        pos_err = (vp - vt).pow(2).reshape(B, -1).mean(dim=1)  # (B,)
        logit_pos = -pos_err / tau  # (B,)
        vt_exp = vt.unsqueeze(1)  # (B, 1, C, H, W)
        neg_err = (vn - vt_exp).pow(2).reshape(B, k, -1).mean(dim=2)  # (B, k)
        logits_neg = -neg_err / tau  # (B, k)
        if neg_penalty is not None:
            # jaccard mode: down-weight tag-overlapping negatives.
            logits_neg = logits_neg - neg_penalty.to(logits_neg.dtype)
        return logit_pos, logits_neg

    @staticmethod
    def _infonce_from_logits(
        logit_pos: torch.Tensor, logits_neg: torch.Tensor
    ) -> torch.Tensor:
        # InfoNCE: -log softmax of the positive over {pos, neg_1..k}.
        all_logits = torch.cat([logit_pos.unsqueeze(1), logits_neg], dim=1)  # (B, 1+k)
        return (-logit_pos + torch.logsumexp(all_logits, dim=1)).mean()

    @staticmethod
    def _contrastive_diagnostics(
        logit_pos: torch.Tensor, logits_neg: torch.Tensor
    ) -> dict[str, float]:
        with torch.no_grad():
            acc = (logit_pos.unsqueeze(1) > logits_neg).all(dim=1).float().mean()
            gap = (logit_pos - logits_neg.mean(dim=1)).mean()
        return {
            "contrastive_acc": float(acc.item()),
            "contrastive_logit_gap": float(gap.item()),
        }

    # ── AGSM target-shift objective (docs/proposal/soft_tokens_agsm.md) ──────

    def _ensure_bank_ema(self) -> None:
        """Lazily clone the EMA shadow off the live bank (device/dtype-matched)."""
        if self._tokens_ema is None:
            self._tokens_ema = self.tokens.detach().clone()
            self._t_offsets_ema = self.t_offsets.weight.detach().clone()

    def update_bank_ema(self) -> None:
        """``ema ← decay·ema + (1−decay)·live`` for both bank tensors.

        Called once per optimizer step (adapter gates on ``sync_gradients``);
        no-op unless AGSM is active. The shadow lags the live bank by one step,
        which keeps Δ off the bank's instantaneous output — the decoupling that
        makes the target a bounded fixed point.
        """
        if self.contrastive_objective != "agsm":
            return
        self._ensure_bank_ema()
        d = float(self._agsm_ema_decay)
        self._tokens_ema.mul_(d).add_(self.tokens.detach(), alpha=1.0 - d)
        self._t_offsets_ema.mul_(d).add_(
            self.t_offsets.weight.detach(), alpha=1.0 - d
        )

    @staticmethod
    def _agsm_pl_weights(
        ema_all: torch.Tensor, v_target: torch.Tensor, tau: float
    ) -> torch.Tensor:
        """Plackett–Luce candidate weights (Algorithm 1 line 10 / Eq. 13, 40).

        ``ema_all`` is the EMA-bank velocity for every candidate caption stacked
        on dim 1 (index 0 = matched, 1..k = mismatched), ``(B, m, C, H, W)``. The
        implicit reward is the negative mean-squared FM error per candidate (same
        scale/reduction as the InfoNCE logit), and ``w = softmax_j(r_j / τ)``.
        When the matched caption explains the latent best, ``w_matched → 1`` which
        drives the guidance Δ → 0 — the self-annealing normalized correction that
        *bounds* AGSM (paper §3.3). Returns ``(B, m)``, detached.
        """
        B, m = ema_all.shape[0], ema_all.shape[1]
        vt = v_target.float().unsqueeze(1)  # (B, 1, C, H, W)
        reward = -(ema_all.float() - vt).pow(2).reshape(B, m, -1).mean(dim=2)  # (B, m)
        return torch.softmax(reward / max(float(tau), 1e-6), dim=1).detach()

    def agsm_targets(
        self,
        v_pos_ema: torch.Tensor,
        v_neg_ema: torch.Tensor,
        v_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Bounded PL-weighted target velocities (Algorithm 1 lines 10–12, Eq. 17).

        Single-bank Phase 2: the matched + mismatched EMA preds all come off the
        one shadow bank (only the spliced caption differs), so a single
        PL-weighted baseline ``Σ_j w_j v̂_ema_j`` (over **all** candidates incl.
        the matched one) serves both branches. Per-candidate guidance
        ``Δ_j = v̂_ema_j − baseline`` then shifts each target by its own sign/scale::

            tgt⁺   = v_target + γ⁺·(v̂⁺_ema   − baseline)
            tgt⁻_j = v_target − γ⁻·(v̂⁻_ema_j − baseline)

        All outputs are detached (constants for the regression in ``agsm_losses``).
        Mirrors Eq. 17 with Ã(t)=1; the ``v = ε − x₀`` mapping makes the ε-target
        shift an exact v-target shift (proposal §"flow-matching mapping").

        Shapes: v_pos_ema, v_target ``(B, C, H, W)``; v_neg_ema ``(B, k, C, H, W)``.
        Returns ``(tgt_pos (B,C,H,W), tgt_neg (B,k,C,H,W), diag)``.
        """
        vp = v_pos_ema.float()
        vn = v_neg_ema.float()
        vt = v_target.float()
        ema_all = torch.cat([vp.unsqueeze(1), vn], dim=1)  # (B, m=k+1, C, H, W)
        w = self._agsm_pl_weights(ema_all, vt, self._contrastive_tau)  # (B, m)
        baseline = (w.view(*w.shape, 1, 1, 1) * ema_all).sum(dim=1)  # (B, C, H, W)
        gp = float(self._agsm_gamma)
        gn = float(self._agsm_gamma_neg)
        delta_pos = vp - baseline  # (B, C, H, W)
        delta_neg = vn - baseline.unsqueeze(1)  # (B, k, C, H, W)
        tgt_pos = (vt + gp * delta_pos).detach()
        tgt_neg = (vt.unsqueeze(1) - gn * delta_neg).detach()
        with torch.no_grad():
            all_delta = torch.cat([delta_pos.unsqueeze(1), delta_neg], dim=1)
            diag = {
                "w_matched": float(w[:, 0].mean().item()),
                "delta_norm": float(all_delta.flatten(2).norm(dim=-1).mean().item()),
            }
        return tgt_pos, tgt_neg, diag

    @staticmethod
    def agsm_losses(
        v_pos: torch.Tensor,
        v_neg: torch.Tensor,
        tgt_pos: torch.Tensor,
        tgt_neg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounded target-shift MSE losses against precomputed (detached) targets.

            L⁺ = ‖ v_pos   − tgt_pos   ‖²
            L⁻ = mean_j ‖ v_neg_j − tgt_neg_j ‖²

        ``tgt_pos`` / ``tgt_neg`` are constants (from ``agsm_targets``), so each
        term has a fixed point — the AGSM fix for InfoNCE's unbounded negative
        divergence. Gradient flows through whichever of ``v_pos`` / ``v_neg`` the
        caller leaves live (the grad-cache split in ``after_backward``).

        Shapes: v_pos, tgt_pos ``(B, C, H, W)``; v_neg, tgt_neg ``(B, k, C, H, W)``.
        Returns ``(l_pos, l_neg)`` scalars; the caller sums them.
        """
        B = v_pos.shape[0]
        k = v_neg.shape[1]
        l_pos = (v_pos.float() - tgt_pos).pow(2).reshape(B, -1).mean(dim=1).mean()
        l_neg = (
            (v_neg.float() - tgt_neg).pow(2).reshape(B, k, -1).mean(dim=2).mean()
        )
        return l_pos, l_neg


class SoftTokensMethodAdapter(MethodAdapter):
    """Runs the contrastive negative forwards for soft tokens.

    Each negative is one extra DiT forward sharing the anchor's ``(x_t, ε, t)``
    and spliced tokens, swapping only ``crossattn_emb`` — the ``extra_forwards``
    contract. Two objectives share the plumbing, selected by the network's
    ``contrastive_objective``:

      - ``infonce`` — SoftREPA InfoNCE over the negatives (k forwards).
      - ``agsm`` — bounded target-shift ``v_target ± γ_z·Δ_j`` with per-candidate
        Plackett–Luce guidance off the bank's EMA shadow. Adds EMA value passes
        (matched + each mismatched) → ~(2k+1) forwards. Same grad-cache split;
        only the loss math differs.

    Wiring: ``prime_for_forward`` stashes ``batch["neg_crossattn_emb"]`` (train
    only); ``extra_forwards`` returns the scalar under ``"soft_tokens_contrastive"``
    (composer applies warmup-gated weight); ``after_backward`` replays the
    deferred ∂L/∂v_neg + refreshes the AGSM EMA. Negatives absent outside training
    → forwards skipped, so val FM-MSE stays clean.
    """

    name = "soft_tokens_contrastive"

    def __init__(self) -> None:
        self._neg_crossattn: Optional[torch.Tensor] = None
        self._neg_jaccard: Optional[torch.Tensor] = None
        self._last_metrics: dict[str, float] = {}
        # Block-swap grad-cache state: when block swapping is active the
        # negative backward can't share the anchor's forward/backward cycle, so
        # it's deferred to ``after_backward``. ``None`` when no replay is queued.
        self._pending_gradcache: Optional[dict] = None

    def prime_for_forward(
        self, ctx: StepCtx, batch, latents: torch.Tensor, *, is_train: bool
    ) -> None:
        del ctx, latents
        if not is_train or not isinstance(batch, dict):
            self._neg_crossattn = None
            self._neg_jaccard = None
            return
        self._neg_crossattn = batch.get("neg_crossattn_emb")
        # Per-negative tag-overlap Jaccard (B, k), present only in jaccard mode.
        self._neg_jaccard = batch.get("neg_jaccard")

    def extra_forwards(self, ctx: StepCtx, primary: ForwardArtifacts) -> Optional[dict]:
        if not primary.is_train:
            return None
        neg = self._neg_crossattn
        if neg is None:
            return None
        net = ctx.accelerator.unwrap_model(ctx.network)
        if float(getattr(net, "_contrastive_target_weight", 0.0) or 0.0) <= 0.0:
            return None
        # Warmup gate: while ``_contrastive_weight`` is held at 0 (first
        # ``_contrastive_warmup_ratio`` of training) the loss is multiplied by 0
        # downstream and ``after_backward`` is already skipped — so the k negative
        # DiT value forwards below would be pure waste. Skip the whole block.
        if float(getattr(net, "_contrastive_weight", 0.0) or 0.0) <= 0.0:
            self._pending_gradcache = None
            return None
        # Cadence gate: skip the whole contrastive block (no_grad value pass +
        # after_backward replay) on non-firing steps. The flag is set per step
        # by ``step_contrastive_warmup`` on the optimizer-step clock.
        if not getattr(net, "_contrastive_fire_this_step", True):
            self._pending_gradcache = None
            return None

        device = primary.noisy_model_input.device
        ce_dtype = primary.crossattn_emb.dtype
        neg = neg.to(device)  # (B, k, S, D)
        k = neg.shape[1]
        # Dual bank (Phase 3a): negatives splice ψ⁻ (branch 1); the anchor +
        # matched EMA stay on ψ⁺ (branch 0). Single bank → branch 0 throughout
        # (bit-identical to Phase 2).
        neg_branch = 1 if getattr(net, "n_banks", 1) > 1 else 0

        v_pos = primary.model_pred.squeeze(2)  # (B, C, H, W) — live anchor graph
        # Rectified-flow velocity target — same as train.py's primary target.
        v_target = primary.noise - primary.latents  # (B, C, H, W)
        timesteps = primary.timesteps
        base_kw = dict(primary.forward_kwargs)
        neg_penalty = self._neg_penalty(net, device)

        # Snapshot the anchor's splice state; the negative value passes mutate the
        # per-step buffers, so restore afterwards. (The anchor's autograd graph
        # holds tensor references, so these attribute writes don't affect it.)
        anchor_tokens = net._step_layer_tokens
        anchor_seqlens = net._step_seqlens

        dit = ctx.accelerator.unwrap_model(primary.anima_call)

        # ── Gradient caching, split so the negative DiT forward NEVER overlaps the
        # anchor backward. Naive checkpoint-and-recompute OOMs (recompute fires
        # during ``accelerator.backward`` with the anchor graph still live → two
        # forwards resident) and crashes under block swap (recompute re-enters
        # ``_run_blocks`` against the offloader's end-of-forward layout). Instead
        # split ∂L_con/∂θ into two partials, each its own clean forward/backward:
        #   • ∂L/∂v_pos — rides the anchor's FM backward: the returned loss uses
        #     ``logit_pos`` (live graph) + *detached* negative logits.
        #   • ∂L/∂v_neg — deferred to ``after_backward`` (anchor freed, offloader
        #     head-resident). Negatives forwarded once here under no_grad for their
        #     values + cached ``g_neg``, then replayed there.
        # ``prepare_block_swap_before_forward`` is a no-op at blocks_to_swap=0, so
        # one path serves swap and no-swap (no-swap still peaks at one graph).

        agsm = getattr(net, "contrastive_objective", "infonce") == "agsm"

        # Negative velocity values under no_grad (no graph retained). Each forward
        # is bracketed by a block-swap reset (no-op when not swapping) so the
        # offloader stays in the anchor's end-of-forward state. AGSM also forwards
        # the EMA shadow bank (matched + each mismatched) for the detached Δ.
        v_neg_vals = []
        v_pos_ema = None
        v_neg_ema_vals = []
        with torch.no_grad():
            if agsm:
                dit.prepare_block_swap_before_forward(free_cache=False)
                v_pos_ema = self._bank_forward(
                    net, primary.anima_call,
                    primary.noisy_model_input, primary.padding_mask,
                    base_kw, timesteps, ce_dtype, primary.crossattn_emb,
                    use_ema=True,
                )
            for j in range(k):
                dit.prepare_block_swap_before_forward(free_cache=False)
                v_neg_vals.append(
                    self._bank_forward(
                        net, primary.anima_call,
                        primary.noisy_model_input, primary.padding_mask,
                        base_kw, timesteps, ce_dtype, neg[:, j],
                        branch=neg_branch,
                    )
                )
                if agsm:
                    dit.prepare_block_swap_before_forward(free_cache=False)
                    v_neg_ema_vals.append(
                        self._bank_forward(
                            net, primary.anima_call,
                            primary.noisy_model_input, primary.padding_mask,
                            base_kw, timesteps, ce_dtype, neg[:, j],
                            use_ema=True, branch=neg_branch,
                        )
                    )
        net._step_layer_tokens = anchor_tokens
        net._step_seqlens = anchor_seqlens
        v_neg = torch.stack(v_neg_vals, dim=1)  # (B, k, C, H, W), detached values

        live = float(getattr(net, "_contrastive_weight", 0.0) or 0.0)

        if agsm:
            # PL-weighted bounded targets from the EMA shadow's own preds (all
            # detached). L⁺ rides the anchor's FM backward (grad via live v_pos);
            # L⁻'s grad is deferred against the per-candidate tgt_neg constants.
            v_neg_ema = torch.stack(v_neg_ema_vals, dim=1)  # (B, k, C, H, W)
            tgt_pos, tgt_neg, agsm_diag = net.agsm_targets(
                v_pos_ema, v_neg_ema, v_target
            )
            l_pos, l_neg_val = net.agsm_losses(v_pos, v_neg, tgt_pos, tgt_neg)
            loss = l_pos + l_neg_val
            self._record_agsm_metrics(net, loss, l_pos, l_neg_val, agsm_diag)
            if live > 0.0:
                v_neg_leaf = v_neg.detach().requires_grad_(True)
                _, l_neg_leaf = net.agsm_losses(
                    v_pos.detach(), v_neg_leaf, tgt_pos, tgt_neg
                )
                (g_neg,) = torch.autograd.grad(l_neg_leaf, v_neg_leaf)
                self._pending_gradcache = self._build_gradcache(
                    net, dit, primary, base_kw, timesteps, neg, ce_dtype,
                    g_neg.detach(), live, anchor_tokens, anchor_seqlens,
                    neg_branch,
                )
            else:
                self._pending_gradcache = None
            return {"soft_tokens_contrastive": loss}

        # Composer-side loss: grad only via v_pos (negatives are constants).
        logit_pos, logits_neg = net._velocities_to_logits(
            v_pos, v_neg, v_target, neg_penalty
        )
        loss = net._infonce_from_logits(logit_pos, logits_neg)
        diag = net._contrastive_diagnostics(logit_pos.detach(), logits_neg.detach())
        self._record_metrics(net, loss, diag)

        if live > 0.0:
            # Cache ∂L/∂v_neg with v_pos held constant (no DiT forward — tiny head).
            v_neg_leaf = v_neg.detach().requires_grad_(True)
            lp_d, ln_leaf = net._velocities_to_logits(
                v_pos.detach(), v_neg_leaf, v_target, neg_penalty
            )
            g_loss = net._infonce_from_logits(lp_d, ln_leaf)
            (g_neg,) = torch.autograd.grad(g_loss, v_neg_leaf)
            self._pending_gradcache = self._build_gradcache(
                net, dit, primary, base_kw, timesteps, neg, ce_dtype,
                g_neg.detach(), live, anchor_tokens, anchor_seqlens,
                neg_branch,
            )
        else:
            self._pending_gradcache = None
        return {"soft_tokens_contrastive": loss}

    @staticmethod
    def _build_gradcache(
        net, dit, primary, base_kw, timesteps, neg, ce_dtype,
        g_neg, weight, anchor_tokens, anchor_seqlens, neg_branch,
    ) -> dict:
        """Pack the deferred ∂L/∂v_neg replay state. Objective-agnostic — the
        replay in ``after_backward`` just pushes the cached ``g_neg`` back through
        each negative's (live-bank) forward, so InfoNCE and AGSM share it.
        ``neg_branch`` is the bank the negative spliced on (ψ⁻ under dual bank) so
        the replay forward routes grad to the same params."""
        return {
            "net": net,
            "dit": dit,
            "anima_call": primary.anima_call,
            "noisy_model_input": primary.noisy_model_input,
            "padding_mask": primary.padding_mask,
            "timesteps": timesteps,
            "base_kw": base_kw,
            "neg": neg,
            "ce_dtype": ce_dtype,
            "g_neg": g_neg,
            "weight": weight,
            "anchor_tokens": anchor_tokens,
            "anchor_seqlens": anchor_seqlens,
            "neg_branch": neg_branch,
        }

    def after_backward(self, ctx: StepCtx) -> None:
        """Replay the cached contrastive negatives after the FM backward.

        The anchor graph is freed (and under swap the offloader head-resident),
        so each negative re-forward + backward peaks at a single graph. The cached
        ``weight·g_neg`` accumulates onto the FM grads on the same params (no
        ``zero_grad`` between here and the optimizer step). Single-process only —
        a manual backward inside ``accelerator.accumulate`` would need DDP no-sync
        handling under multi-GPU.

        Also refreshes the AGSM bank-EMA shadow once per optimizer step (gated on
        ``sync_gradients``), regardless of the cadence gate — the bank moves under
        FM every step, so its slow shadow must track it.
        """
        net_for_ema = ctx.accelerator.unwrap_model(ctx.network)
        if (
            getattr(net_for_ema, "contrastive_objective", "infonce") == "agsm"
            and getattr(ctx.accelerator, "sync_gradients", True)
        ):
            net_for_ema.update_bank_ema()

        pend = self._pending_gradcache
        if pend is None:
            return
        self._pending_gradcache = None

        net = pend["net"]
        dit = pend["dit"]
        accel = ctx.accelerator
        # Match accelerate's 1/N loss scaling so contrastive grads land on the
        # same scale as the FM grads it accumulates alongside.
        accum = max(1, int(getattr(accel, "gradient_accumulation_steps", 1) or 1))
        scale = pend["weight"] / accum
        neg = pend["neg"]
        k = neg.shape[1]
        ts = pend["timesteps"]
        ce_dtype = pend["ce_dtype"]
        g_neg = pend["g_neg"]

        neg_branch = pend.get("neg_branch", 0)
        with accel.autocast(), torch.enable_grad():
            for j in range(k):
                dit.prepare_block_swap_before_forward(free_cache=False)
                v_neg_j = self._bank_forward(
                    net, pend["anima_call"],
                    pend["noisy_model_input"], pend["padding_mask"],
                    pend["base_kw"], ts, ce_dtype, neg[:, j],
                    branch=neg_branch,
                )
                grad_j = (scale * g_neg[:, j]).to(v_neg_j.dtype)
                torch.autograd.backward(v_neg_j, grad_tensors=grad_j)
        net._step_layer_tokens = pend["anchor_tokens"]
        net._step_seqlens = pend["anchor_seqlens"]

    @staticmethod
    def _bank_forward(
        net, anima_call, noisy_model_input, padding_mask,
        base_kw, timesteps, ce_dtype, text_emb, use_ema: bool = False,
        branch: int = 0,
    ) -> torch.Tensor:
        """One DiT forward conditioned on ``text_emb`` → velocity (B, C, H, W).

        Re-primes the per-block soft-token splice for this text and runs the
        frozen DiT with the anchor's (x_t, ε, t). ``use_ema=True`` splices the
        EMA shadow bank instead of the live params (AGSM Δ value passes — no
        grad). ``branch`` picks ψ⁺ (0) / ψ⁻ (1) for dual-bank nets (no-op when
        single bank). Returns the squeezed 4D velocity.
        """
        text_emb = text_emb.to(dtype=ce_dtype)
        # front_of_padding needs per-sample seqlens (non-zero rows of the
        # zero-padded crossattn_emb); end_of_sequence ignores them, so skip the
        # abs-sum reduction there.
        if net.splice_position == "front_of_padding":
            seqlens = (text_emb.abs().sum(dim=-1) > 0).sum(dim=-1).to(torch.int32)
        else:
            seqlens = None
        net._set_step_tokens(timesteps, seqlens, use_ema=use_ema, branch=branch)
        kw_j = dict(base_kw)
        if "pooled_text_override" in kw_j:
            kw_j["pooled_text_override"] = text_emb.max(dim=1).values
        return anima_call(
            noisy_model_input, timesteps, text_emb, padding_mask=padding_mask, **kw_j
        ).squeeze(2)

    def _neg_penalty(self, net, device) -> Optional[torch.Tensor]:
        """jaccard mode: α·s subtracted from each negative logit (s = caption
        tag-overlap surfaced by the dataset). ``None`` for shuffled / hard."""
        if (
            getattr(net, "contrastive_negative_mode", "shuffled") == "jaccard"
            and self._neg_jaccard is not None
        ):
            alpha = float(getattr(net, "contrastive_jaccard_alpha", 1.0) or 0.0)
            return alpha * self._neg_jaccard.to(device).float()
        return None

    def _record_metrics(self, net, loss, diag) -> None:
        live = float(getattr(net, "_contrastive_weight", 0.0) or 0.0)
        loss_val = float(loss.detach().item())
        self._last_metrics = {
            "reg/soft_tokens_contrastive": loss_val,
            "reg/soft_tokens_contrastive_weighted": live * loss_val,
            "reg/soft_tokens_contrastive_lambda_live": live,
            "soft_tokens/contrastive_acc": diag["contrastive_acc"],
            "soft_tokens/contrastive_logit_gap": diag["contrastive_logit_gap"],
        }

    def _record_agsm_metrics(self, net, loss, l_pos, l_neg, diag) -> None:
        """AGSM term diagnostics. ``agsm_delta_norm`` is the mean per-sample L2 of
        the (per-candidate) guidance direction — near 0 ⇒ matched/mismatched preds
        collapsed (no signal); growing ⇒ the EMA bank distinguishes them.
        ``agsm_w_matched`` is the Plackett–Luce weight on the matched caption — as
        it → 1 the normalized correction self-anneals (Δ → 0, paper §3.3), so a
        rising ``w_matched`` with bounded losses is the healthy AGSM signature.
        ``agsm_l_pos`` / ``agsm_l_neg`` should both sit at a bounded steady state
        rather than ``l_neg`` diverging (the SoftREPA failure AGSM removes)."""
        live = float(getattr(net, "_contrastive_weight", 0.0) or 0.0)
        loss_val = float(loss.detach().item())
        self._last_metrics = {
            "reg/soft_tokens_contrastive": loss_val,
            "reg/soft_tokens_contrastive_weighted": live * loss_val,
            "reg/soft_tokens_contrastive_lambda_live": live,
            "soft_tokens/agsm_l_pos": float(l_pos.detach().item()),
            "soft_tokens/agsm_l_neg": float(l_neg.detach().item()),
            "soft_tokens/agsm_delta_norm": diag["delta_norm"],
            "soft_tokens/agsm_w_matched": diag["w_matched"],
        }

    def metrics(self, ctx) -> dict:
        del ctx
        return dict(self._last_metrics)
