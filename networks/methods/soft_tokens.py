# Per-layer time-indexed soft tokens — SoftREPA-style parameterization (without
# the contrastive loss). DiT is frozen; trains a small bank of K continuous
# vectors per (layer, t-bucket) pair, prepended (via end-of-sequence overwrite)
# to crossattn_emb at each block independently.
#
# Reference: Lee et al., "Aligning Text to Image in Diffusion Models is Easier
# Than You Think" (NeurIPS 2025) — arXiv:2503.08250. We adopt only their
# parameterization (per-layer × per-t soft tokens), trained under plain flow-
# matching loss; the contrastive InfoNCE objective is intentionally skipped.
# In Anima the DiT is cross-attention (not joint-stream MM-DiT like SD3), so
# crossattn_emb does not evolve through blocks — each block independently
# receives crossattn_emb extended with its own layer tokens. No strip/re-prepend
# dance is needed.
#
# Splice strategy: end-of-sequence overwrite of zero-padding (K tail slots),
# preserving static crossattn_emb shape so `_run_blocks` torch.compile stays
# happy. Zero-padded positions act as cross-attention sinks (see Anima's text-
# encoder padding invariant), so writing tokens into them gives them attention
# mass without changing seqlen.
#
# Why a separate module from postfix.py: postfix splices once at the cached
# adapter output (training-time and inference-time). Soft tokens splice per-
# block via monkey-patched Block.forward (ReFT-pattern), a fundamentally
# different surface.
#
# Inference path: the per-step splice runs from inside the denoising loop —
# library/inference/generation.py + networks/spectrum.py call append_postfix(...,
# timesteps=t) per CFG branch before each forward, mirroring the training-side
# trainer hook. On Spectrum cached steps the blocks don't fire, so soft tokens
# silently no-op for those steps (composes freely with --spectrum).

import os
from typing import Optional

import torch
import torch.nn as nn

from library.log import setup_logging
from networks.methods.base import AdapterNetworkBase

import logging

setup_logging()
logger = logging.getLogger(__name__)

# Anima cached crossattn_emb dimension (Qwen3 hidden size, post LLM-adapter).
DEFAULT_EMBED_DIM = 1024

# Bank-dispersive regularizer form. See ``SoftTokensNetwork.bank_dispersive_loss``.
# Probe at ``bench/soft_tokens_dispersive/probe.py`` (2026-05-19) found ``current``
# structurally unbounded — the optimizer drives bank norms arbitrarily large to
# push log(mean(exp(-d²/τ))) toward −∞. The bounded forms (cosine_sq, cosine_abs,
# normalized_pdist, hinge) all saturate within ~500 steps without FM counterforce.
DISPERSIVE_FORMS = (
    "current",            # log(mean(exp(-pdist²/τ))) on raw bank vectors. Unbounded below.
    "normalized_pdist",   # same form on unit-normalized vectors. Bounded; magnitude untouched.
    "cosine_sq",          # mean cos(v_i, v_j)² over pairs. Bounded [0, 1], smooth.
    "cosine_abs",         # mean |cos(v_i, v_j)| over pairs. Bounded [0, 1], non-smooth at 0.
    "hinge",              # mean max(0, ε − pdist²) over pairs. Bounded ≥ 0.
)
DEFAULT_DISPERSIVE_FORM = "current"  # back-compat default; new configs should override.
DEFAULT_DISPERSIVE_HINGE_EPS = 1.0


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
    bank_dispersive_weight = float(kwargs.get("bank_dispersive_weight", 0.0))
    bank_dispersive_warmup_ratio = float(
        kwargs.get("bank_dispersive_warmup_ratio", 0.1)
    )
    bank_dispersive_tau = float(kwargs.get("bank_dispersive_tau", 0.5))
    dispersive_form = str(kwargs.get("dispersive_form", DEFAULT_DISPERSIVE_FORM))
    dispersive_hinge_eps = float(
        kwargs.get("dispersive_hinge_eps", DEFAULT_DISPERSIVE_HINGE_EPS)
    )
    network = SoftTokensNetwork(
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_t_buckets=n_t_buckets,
        init_std=init_std,
        splice_position=splice_position,
        bank_dispersive_weight=bank_dispersive_weight,
        bank_dispersive_warmup_ratio=bank_dispersive_warmup_ratio,
        bank_dispersive_tau=bank_dispersive_tau,
        dispersive_form=dispersive_form,
        dispersive_hinge_eps=dispersive_hinge_eps,
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
    n_layers, num_tokens, embed_dim = tokens.shape
    n_t_buckets = t_offsets.shape[0]
    # Splice position is a runtime knob, not learned — read from metadata if
    # present, otherwise default. CLI kwargs win for post-hoc overrides.
    metadata_splice = None
    if file is not None and os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            meta = f.metadata() or {}
            metadata_splice = meta.get("ss_splice_position")
    # Pull metadata for the dispersive form + hinge ε too — both are saved
    # by ``get_metadata`` and should round-trip on load. CLI/TOML kwargs still
    # win for post-hoc overrides.
    metadata_form = None
    metadata_hinge_eps = None
    if file is not None and os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            meta = f.metadata() or {}
            metadata_form = meta.get("ss_dispersive_form")
            metadata_hinge_eps = meta.get("ss_dispersive_hinge_eps")
    splice_position = kwargs.get(
        "splice_position", metadata_splice or "end_of_sequence"
    )
    bank_dispersive_weight = float(kwargs.get("bank_dispersive_weight", 0.0))
    bank_dispersive_warmup_ratio = float(
        kwargs.get("bank_dispersive_warmup_ratio", 0.1)
    )
    bank_dispersive_tau = float(kwargs.get("bank_dispersive_tau", 0.5))
    dispersive_form = str(
        kwargs.get("dispersive_form", metadata_form or DEFAULT_DISPERSIVE_FORM)
    )
    dispersive_hinge_eps = float(
        kwargs.get(
            "dispersive_hinge_eps",
            metadata_hinge_eps if metadata_hinge_eps is not None
            else DEFAULT_DISPERSIVE_HINGE_EPS,
        )
    )
    network = SoftTokensNetwork(
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_t_buckets=n_t_buckets,
        init_std=0.0,  # weights are loaded; init_std doesn't matter
        splice_position=splice_position,
        bank_dispersive_weight=bank_dispersive_weight,
        bank_dispersive_warmup_ratio=bank_dispersive_warmup_ratio,
        bank_dispersive_tau=bank_dispersive_tau,
        dispersive_form=dispersive_form,
        dispersive_hinge_eps=dispersive_hinge_eps,
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
        bank_dispersive_weight: float = 0.0,
        bank_dispersive_warmup_ratio: float = 0.1,
        bank_dispersive_tau: float = 0.5,
        dispersive_form: str = DEFAULT_DISPERSIVE_FORM,
        dispersive_hinge_eps: float = DEFAULT_DISPERSIVE_HINGE_EPS,
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
        if bank_dispersive_tau <= 0.0:
            raise ValueError(
                f"bank_dispersive_tau must be positive, got {bank_dispersive_tau}"
            )
        if dispersive_form not in DISPERSIVE_FORMS:
            raise ValueError(
                f"dispersive_form must be one of {DISPERSIVE_FORMS}, "
                f"got {dispersive_form!r}"
            )
        if dispersive_hinge_eps <= 0.0:
            raise ValueError(
                f"dispersive_hinge_eps must be positive, got {dispersive_hinge_eps}"
            )

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_t_buckets = n_t_buckets
        self.splice_position = splice_position
        self.multiplier = multiplier

        # Parameter-space dispersive regularizer over the K and n_t_buckets axes
        # (Wang & He, "Diffuse and Disperse", arXiv:2506.09027). Adapted to a
        # B=1 regime by dispersing along the bank's intrinsic axes — not the
        # batch axis the paper uses — which lets the term fire every step
        # regardless of which timestep bucket was sampled. ``_bank_dispersive_weight``
        # is the live value (held at 0 during warmup); ``_bank_dispersive_target_weight``
        # is the post-warmup target. Composer activation gates on the target.
        self._bank_dispersive_target_weight = float(bank_dispersive_weight)
        self._bank_dispersive_warmup_ratio = float(bank_dispersive_warmup_ratio)
        self._bank_dispersive_tau = float(bank_dispersive_tau)
        self._dispersive_form = str(dispersive_form)
        self._dispersive_hinge_eps = float(dispersive_hinge_eps)
        # Live weight: starts at 0 if warmup is on, jumps to target after.
        self._bank_dispersive_weight = (
            0.0
            if self._bank_dispersive_warmup_ratio > 0.0
            else self._bank_dispersive_target_weight
        )

        self.tokens = nn.Parameter(
            torch.randn(n_layers, num_tokens, embed_dim) * init_std
        )
        # Per-(bucket, layer) D-vector offset. Broadcast across the K-token axis
        # at lookup so the bucket only has to learn one D-vector per layer per
        # bucket (not K). Zero-init = identity perturbation at step 0.
        self.t_offsets = nn.Embedding(n_t_buckets, n_layers * embed_dim)
        nn.init.zeros_(self.t_offsets.weight)

        # Step-scoped state set by append_postfix() once per forward pass and
        # consumed by the per-block hooks installed in apply_to(). Kept as a
        # plain attribute (not a buffer) — recreated each step, no need to
        # persist or move with .to(). _step_seqlens is only populated for
        # front_of_padding splice; end_of_sequence ignores it.
        self._step_layer_tokens: Optional[torch.Tensor] = None  # (n_layers, B, K, D)
        self._step_seqlens: Optional[torch.Tensor] = None  # (B,) int

        # Reverse-bookkeeping for apply_to(): keep references so we could
        # un-monkey-patch later (currently unused but cheap to track).
        self._block_refs: list[nn.Module] = []
        self._original_forwards: list = []

        n_token_params = self.tokens.numel()
        n_offset_params = self.t_offsets.weight.numel()
        if bank_dispersive_weight > 0.0:
            if dispersive_form == "hinge":
                form_args = f"ε={dispersive_hinge_eps}"
            elif dispersive_form in ("current", "normalized_pdist"):
                form_args = f"τ={bank_dispersive_tau}"
            else:
                form_args = "—"
            disp_note = (
                f", bank_dispersive(form={dispersive_form}, λ={bank_dispersive_weight}, "
                f"warmup={bank_dispersive_warmup_ratio}, {form_args})"
            )
        else:
            disp_note = ""
        logger.info(
            f"SoftTokensNetwork: {n_layers} layers × {num_tokens} tokens × dim {embed_dim}, "
            f"{n_t_buckets} t-buckets, splice={splice_position}{disp_note} → "
            f"{n_token_params + n_offset_params} params "
            f"({n_token_params} base + {n_offset_params} t-offset)"
        )

    # Sentinel attribute so train.py's ``hasattr(network, "append_postfix")``
    # branch picks us up: train.py will then call append_postfix(..., timesteps=...)
    # at the right point in the forward loop, which we use only to compute the
    # step-scoped per-layer tokens. The crossattn_emb passes through unchanged
    # — splicing happens inside the per-block hooks below.
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
                "soft_tokens requires timesteps (per-step) — train.py passes "
                "this automatically; inference path is not yet wired up"
            )
        B = crossattn_emb.shape[0]
        bucket_idx = self._bucketize(timesteps)  # (B,)
        # (B, n_layers * D) → (B, n_layers, D) → (B, n_layers, 1, D)
        offsets = self.t_offsets(bucket_idx).view(B, self.n_layers, self.embed_dim)
        # (n_layers, K, D) → (1, n_layers, K, D), broadcast over batch.
        base = self.tokens.unsqueeze(0)
        # (B, n_layers, K, D); offset broadcasts across K.
        per_step = base + offsets.unsqueeze(2)
        # Transpose to (n_layers, B, K, D) for cheap per-layer indexing in the
        # block hook closure.
        self._step_layer_tokens = per_step.transpose(0, 1).contiguous()
        # front_of_padding needs per-sample seqlens at hook time; end_of_sequence
        # ignores them. Cache regardless so the hook doesn't have to know which
        # mode is active (the splice branch reads or skips).
        self._step_seqlens = (
            crossattn_seqlens.detach().to(torch.long)
            if crossattn_seqlens is not None
            else None
        )
        return crossattn_emb

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
            "ss_bank_dispersive_weight": str(self._bank_dispersive_target_weight),
            "ss_bank_dispersive_warmup_ratio": str(self._bank_dispersive_warmup_ratio),
            "ss_bank_dispersive_tau": str(self._bank_dispersive_tau),
            "ss_dispersive_form": self._dispersive_form,
            "ss_dispersive_hinge_eps": str(self._dispersive_hinge_eps),
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
        self.tokens.data.copy_(weights_sd["tokens"])
        self.t_offsets.weight.data.copy_(weights_sd["t_offsets.weight"])
        logger.info(
            f"Loaded soft_tokens weights: tokens={tuple(self.tokens.shape)}, "
            f"t_offsets={tuple(self.t_offsets.weight.shape)}"
        )

    def metrics(self, ctx) -> dict[str, float]:
        """TensorBoard signals for the bank-dispersive regularizer.

        Two groups:

        ``reg/*`` — regularizer-side values. ``loss`` is the raw dispersive
        term recomputed under no_grad (the trainer wraps ``metrics()`` in
        ``torch.no_grad``); ``loss_weighted`` is λ · loss; ``lambda_live`` is
        the warmup-gated weight actually applied this step. Watching
        ``lambda_live`` jump at the warmup boundary confirms the gate fired.

        ``soft_tokens/*`` — bank-state diagnostics, computed on the K base
        tokens averaged over layers (the same axis the probe at
        ``bench/soft_tokens_dispersive/`` scored variants on). Read these as
        a collapse / divergence detector:
          - ``tokens_mean_cos`` near 0  → bank is orthogonal-ish (good).
          - ``tokens_mean_cos`` near 1  → collapse (bad — regularizer is dead).
          - ``tokens_mean_norm`` blowing up → unbounded form is winning
            (only possible with ``dispersive_form="current"``).
          - ``offset_mean_norm`` staying ~0 → t-offset buckets aren't training
            (FM gradient isn't reaching them; check warmup + LR).
        """
        del ctx
        out: dict[str, float] = {}
        target = float(self._bank_dispersive_target_weight)
        live = float(self._bank_dispersive_weight)
        if target > 0.0:
            # collect_metrics wraps this in no_grad; safe to recompute.
            loss_val = float(self.bank_dispersive_loss().item())
            out["reg/soft_tokens_dispersive"] = loss_val
            out["reg/soft_tokens_dispersive_weighted"] = live * loss_val
            out["reg/soft_tokens_dispersive_lambda_live"] = live

        # Bank-state diagnostics always logged when there are ≥ 2 tokens per
        # layer to take pairs from — cheap, and the most useful signal for
        # "is the regularizer doing anything".
        if self.num_tokens >= 2 and self.n_layers > 0:
            tokens = self.tokens.detach()
            # Pairwise cos / d² per layer, averaged across layers.
            cos_sum = 0.0
            d_min = float("inf")
            for k in range(self.n_layers):
                z = tokens[k]
                zn = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
                gram = zn @ zn.t()
                n = gram.shape[0]
                iu = torch.triu_indices(n, n, offset=1, device=gram.device)
                cos_sum += float(gram[iu[0], iu[1]].mean().item())
                d_sq = torch.pdist(z, p=2).pow(2)
                if d_sq.numel():
                    d_min = min(d_min, float(d_sq.min().item()))
            out["soft_tokens/tokens_mean_cos"] = cos_sum / self.n_layers
            out["soft_tokens/tokens_min_d_sq"] = (
                d_min if d_min != float("inf") else 0.0
            )
            out["soft_tokens/tokens_mean_norm"] = float(
                tokens.flatten(1).norm(dim=-1).mean().item()
            )
        out["soft_tokens/offset_mean_norm"] = float(
            self.t_offsets.weight.detach()
            .view(self.n_t_buckets, self.n_layers, self.embed_dim)
            .permute(1, 0, 2)
            .flatten(1)
            .norm(dim=-1)
            .mean()
            .item()
        )
        return out

    def step_bank_dispersive_warmup(
        self, global_step: int, max_train_steps: int
    ) -> None:
        """Activate the bank-dispersive regularizer once training crosses the
        warmup window. Step function (mirroring ``step_balance_loss_warmup``
        on LoRANetwork): ``_bank_dispersive_weight`` holds at 0 during the
        first ``_bank_dispersive_warmup_ratio`` of steps, then flips to
        ``_bank_dispersive_target_weight``. No-op when target is 0.

        The bank starts near-zero (``init_std=0.02`` on tokens, zero-init on
        t_offsets); pushing for K-axis / bucket-axis dispersion before
        gradients have shaped *any* structure would fight the FM signal.
        Letting denoise loss develop a non-degenerate bank first, then turning
        on the dispersive push, keeps the regularizer's intent ("don't let
        slots collapse to each other") aligned with the FM trajectory.
        """
        target = float(self._bank_dispersive_target_weight)
        ratio = float(self._bank_dispersive_warmup_ratio)
        if target <= 0.0:
            return
        if ratio <= 0.0 or max_train_steps <= 0:
            self._bank_dispersive_weight = target
            return
        warmup_steps = int(max_train_steps * ratio)
        self._bank_dispersive_weight = 0.0 if global_step < warmup_steps else target

    def bank_dispersive_loss(self) -> torch.Tensor:
        """Parameter-space dispersive regularizer over the bank's K and
        n_t_buckets axes. Form selected by ``_dispersive_form``.

        All five forms compute one per-layer scalar from each ``(K, D)`` slab
        of base tokens (6 pairs per layer at default K=4) and from each
        ``(n_buckets, D)`` slab of t-offsets (91 pairs per layer at default
        n_t_buckets=14), then average across the per-layer terms.

        Forms (see DISPERSIVE_FORMS docstring above):
          - ``current``           — paper-style ``log(mean(exp(-d²/τ)))``.
                                    Bench probe at
                                    ``bench/soft_tokens_dispersive/`` showed this
                                    is unbounded below: the optimizer drives bank
                                    norms arbitrarily large to push the loss to
                                    −∞ even without FM signal. Kept for back-compat.
          - ``normalized_pdist``  — same form on unit-normalized vectors. d² ∈
                                    [0, 4] so the loss saturates near a finite
                                    floor. Gradient flows to direction only;
                                    magnitude is left to the FM signal.
          - ``cosine_sq``         — ``mean(cos²)`` over pairs. Bounded [0, 1],
                                    smooth gradient at orthogonal pairs.
          - ``cosine_abs``        — ``mean(|cos|)`` over pairs. Bounded [0, 1],
                                    non-smooth at the orthogonal floor.
          - ``hinge``             — ``mean(max(0, ε - d²))`` over pairs. Bounded
                                    ≥ 0; saturates entirely once every pair
                                    exceeds the ε floor.

        Both axes are fixed dimensions of the bank parameters, so the loss is
        batch-size independent (works at B=1) and provides gradient signal to
        every (layer, k) and (layer, bucket) parameter every step — including
        the t-buckets that the timestep sampler didn't draw this step.

        Returns a scalar averaged across the per-layer terms. The trainer
        multiplies by ``_bank_dispersive_weight`` (warmup-gated, see
        ``step_bank_dispersive_warmup``) on the loss-handler side.
        """
        form = self._dispersive_form
        tau = float(self._bank_dispersive_tau)
        hinge_eps = float(self._dispersive_hinge_eps)

        def _pdist_sq(z: torch.Tensor) -> torch.Tensor:
            return torch.pdist(z, p=2).pow(2)

        def _logmeanexp_neg(d_sq: torch.Tensor) -> torch.Tensor:
            # log(mean(exp(-d²/τ))) via logsumexp − log(N).
            return torch.logsumexp(-d_sq / tau, dim=-1) - torch.log(
                d_sq.new_tensor(float(d_sq.numel()))
            )

        def _cos_pairs(z: torch.Tensor) -> torch.Tensor:
            zn = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
            gram = zn @ zn.t()
            n = gram.shape[0]
            iu = torch.triu_indices(n, n, offset=1, device=gram.device)
            return gram[iu[0], iu[1]]

        def _term(z: torch.Tensor) -> torch.Tensor:
            if z.shape[0] < 2:
                return z.new_zeros(())
            if form == "current":
                return _logmeanexp_neg(_pdist_sq(z))
            if form == "normalized_pdist":
                zn = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
                return _logmeanexp_neg(_pdist_sq(zn))
            if form == "cosine_sq":
                return _cos_pairs(z).pow(2).mean()
            if form == "cosine_abs":
                return _cos_pairs(z).abs().mean()
            if form == "hinge":
                return torch.clamp(hinge_eps - _pdist_sq(z), min=0.0).mean()
            raise RuntimeError(f"unhandled dispersive_form {form!r}")  # validated in __init__

        total = self.tokens.new_zeros(())
        n_terms = 0
        if self.num_tokens >= 2:
            for k in range(self.n_layers):
                total = total + _term(self.tokens[k])
                n_terms += 1
        if self.n_t_buckets >= 2:
            offsets = self.t_offsets.weight.view(
                self.n_t_buckets, self.n_layers, self.embed_dim
            )
            for k in range(self.n_layers):
                total = total + _term(offsets[:, k, :])
                n_terms += 1
        if n_terms == 0:
            return total
        return total / float(n_terms)
