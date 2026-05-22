"""Anima soft-token (SoftREPA-parameterization) live inference for ComfyUI.

Splices per-layer, per-timestep-bucket learned soft tokens into the
T5-compatible ``crossattn_emb`` **inside** the first ``n_layers`` DiT blocks —
the same surface anima_lora's trainer / reference inference monkey-patch
(``networks/methods/soft_tokens.py``). Unlike postfix (one splice on
``diffusion_model.forward``), soft tokens splice independently at each block,
so we install a per-block ``forward_pre_hook`` that rewrites the block's
``crossattn_emb`` positional argument.

Hook-not-override invariant (same as Hydra/ReFT, see CLAUDE.md): we never
replace ``block.forward``. A ``forward_pre_hook`` on each block's
``_forward_pre_hooks`` returns a modified args tuple, leaving ``forward``
untouched so ComfyUI's cast-weights walker and ``unpatch_model`` keep working.

Two pieces of shared state per load, written once per denoising step:
  - a ``diffusion_model._forward_pre_hooks`` pre-hook records the per-step
    sigma and precomputes the ``(n_layers, B, K, D)`` token bank for the step;
  - each block pre-hook indexes its layer's slice and splices it.

Sigma convention: ComfyUI's FLOW model sampling passes ``timesteps = sigma *
1000`` (``ModelSamplingDiscreteFlow``, multiplier 1000) to the diffusion model,
whereas anima_lora trains the t-bucket index on sigma in ``[0, 1]``
(``train.py`` draws ``[0,1]``-scaled timesteps; ``inference.generation`` divides
by 1000 before ``append_postfix``). So we divide ``args[1]`` by 1000 to recover
the ``[0, 1]`` sigma the bucketizer expects.

Bank format (``ss_*`` metadata + two tensors, see
``SoftTokensNetwork.state_dict_for_save`` / ``metadata_fields``):
  - ``tokens``           : ``(n_layers, K, D)`` base per-layer tokens.
  - ``t_offsets.weight`` : ``(n_t_buckets, n_layers * D)`` per-(bucket, layer)
    D-vector offset, broadcast across the K-token axis at lookup.
"""

import logging
from collections import OrderedDict
from typing import Dict

import torch

logger = logging.getLogger(__name__)

# Anima FLOW sampling multiplier (ModelSamplingDiscreteFlow). ComfyUI hands the
# diffusion model ``sigma * _FLOW_MULTIPLIER``; the t-bucket index trains on the
# raw ``[0, 1]`` sigma, so we divide it back out.
_FLOW_MULTIPLIER = 1000.0


class _SoftTokenBank:
    """Parsed soft-token checkpoint: the two tensors + shape/runtime knobs."""

    __slots__ = (
        "tokens",
        "t_offsets",
        "n_layers",
        "num_tokens",
        "embed_dim",
        "n_t_buckets",
        "splice_position",
    )

    def __init__(
        self,
        tokens: torch.Tensor,
        t_offsets: torch.Tensor,
        splice_position: str,
    ):
        self.tokens = tokens  # (n_layers, K, D)
        self.t_offsets = t_offsets  # (n_t_buckets, n_layers * D)
        self.n_layers, self.num_tokens, self.embed_dim = tokens.shape
        self.n_t_buckets = t_offsets.shape[0]
        self.splice_position = splice_position


# Cache: path -> _SoftTokenBank (parse the safetensors once per path).
_bank_cache: Dict[str, _SoftTokenBank] = {}


def load_soft_tokens(file_path: str) -> _SoftTokenBank:
    """Parse a soft-token safetensors file once, cache by path.

    Requires ``tokens`` and ``t_offsets.weight`` tensors (the only two keys
    ``SoftTokensNetwork`` saves). ``splice_position`` is a runtime knob read
    from ``ss_splice_position`` metadata, defaulting to ``end_of_sequence``.
    """
    if file_path in _bank_cache:
        return _bank_cache[file_path]

    from safetensors import safe_open
    from safetensors.torch import load_file

    weights_sd = load_file(file_path)
    tokens = weights_sd.get("tokens")
    t_offsets = weights_sd.get("t_offsets.weight")
    if tokens is None or t_offsets is None:
        raise ValueError(
            f"soft_tokens file must contain 'tokens' and 't_offsets.weight' "
            f"(got keys: {list(weights_sd.keys())[:8]})"
        )
    if tokens.dim() != 3:
        raise ValueError(
            f"soft_tokens 'tokens' must be (n_layers, K, D); got {tuple(tokens.shape)}"
        )
    n_layers, _, embed_dim = tokens.shape
    if t_offsets.dim() != 2 or t_offsets.shape[1] != n_layers * embed_dim:
        raise ValueError(
            f"soft_tokens 't_offsets.weight' must be (n_t_buckets, n_layers*D="
            f"{n_layers * embed_dim}); got {tuple(t_offsets.shape)}"
        )

    splice_position = "end_of_sequence"
    with safe_open(file_path, framework="pt") as f:
        meta = f.metadata() or {}
        splice_position = meta.get("ss_splice_position") or splice_position
    if splice_position not in ("end_of_sequence", "front_of_padding"):
        raise ValueError(
            f"ss_splice_position must be 'end_of_sequence' or 'front_of_padding', "
            f"got {splice_position!r}"
        )

    bank = _SoftTokenBank(tokens.float(), t_offsets.float(), splice_position)
    _bank_cache[file_path] = bank
    logger.info(
        f"Loaded soft_tokens: {bank.n_layers} layers x {bank.num_tokens} tokens "
        f"x dim {bank.embed_dim}, {bank.n_t_buckets} t-buckets, "
        f"splice={bank.splice_position} from {file_path}"
    )
    return bank


def _make_step_pre_hook(state: dict, bank: _SoftTokenBank, strength: float):
    """diffusion_model pre-hook: record sigma + precompute the step's token bank.

    Reads ``args[1]`` (timesteps = sigma * 1000), divides back to ``[0, 1]``,
    bucketizes per-sample, looks up the per-(bucket, layer) offset, and writes
    ``state["step_tokens"]`` of shape ``(n_layers, B, K, D)`` scaled by
    ``strength`` for the per-block hooks to index. dynamo-disabled like the
    Hydra/chimera pre-hooks — the bucketize + embedding gather are eager Python
    on a tiny tensor and never need to trace into a compiled graph.
    """

    @torch._dynamo.disable
    def step_pre_hook(module, args):
        if len(args) < 2 or args[1] is None:
            return
        sigma = (args[1].detach().float().flatten() / _FLOW_MULTIPLIER).clamp(0.0, 1.0)
        device = sigma.device
        # Move the bank to the model device once, then reuse across steps.
        if state.get("_device") != device:
            state["_tokens"] = bank.tokens.to(device)
            state["_t_offsets"] = bank.t_offsets.to(device)
            state["_device"] = device
        tokens = state["_tokens"]  # (n_layers, K, D), fp32
        t_offsets = state["_t_offsets"]  # (n_t_buckets, n_layers*D), fp32

        bucket = (
            torch.floor(sigma * bank.n_t_buckets).long().clamp(0, bank.n_t_buckets - 1)
        )  # (B,)
        B = bucket.shape[0]
        # (B, n_layers*D) -> (B, n_layers, D) -> (B, n_layers, 1, D)
        offsets = t_offsets[bucket].view(B, bank.n_layers, bank.embed_dim)
        # (1, n_layers, K, D) + (B, n_layers, 1, D) -> (B, n_layers, K, D)
        per_step = tokens.unsqueeze(0) + offsets.unsqueeze(2)
        if strength != 1.0:
            per_step = per_step * strength
        # (n_layers, B, K, D) for cheap per-layer indexing in the block hooks.
        state["step_tokens"] = per_step.transpose(0, 1).contiguous()

    return step_pre_hook


def _make_block_pre_hook(layer_idx: int, state: dict, splice_position: str, K: int):
    """Per-block pre-hook: splice this layer's tokens into ``crossattn_emb``.

    Block forward is called positionally as ``block(x, emb, crossattn_emb,
    **block_kwargs)`` (comfy ``predict2.Block.forward``), so ``args[2]`` is the
    cross-attention text embedding. We return a new args tuple with that slot
    rewritten; ``forward`` itself is untouched (hook-not-override invariant).

    ``end_of_sequence`` overwrites the K tail (zero-padding) slots — a static
    slice+cat, shape-preserving. ``front_of_padding`` derives per-sample
    seqlens from the non-zero mask of ``crossattn_emb`` (same as the postfix
    splice) and scatters the tokens right after the real text tokens.
    """

    @torch._dynamo.disable
    def block_pre_hook(module, args):
        step_tokens = state.get("step_tokens")
        if step_tokens is None or len(args) < 3 or args[2] is None:
            return
        ctx = args[2]
        layer_tok = step_tokens[layer_idx].to(dtype=ctx.dtype, device=ctx.device)
        # Broadcast the bank across the batch when sigma was shape (1,) but the
        # block sees a CFG-doubled context.
        if layer_tok.shape[0] == 1 and ctx.shape[0] != 1:
            layer_tok = layer_tok.expand(ctx.shape[0], -1, -1)
        S = ctx.shape[1]
        if S < K:
            raise RuntimeError(
                f"crossattn_emb seqlen {S} < num_tokens {K}; cannot splice soft tokens"
            )
        if splice_position == "end_of_sequence":
            new_ctx = torch.cat([ctx[:, : S - K, :], layer_tok], dim=1)
        else:  # front_of_padding
            D = ctx.shape[-1]
            seqlens = (ctx.abs().sum(dim=-1) > 0).long().sum(dim=-1)  # (B,)
            offsets = seqlens.unsqueeze(1) + torch.arange(K, device=ctx.device)
            offsets = offsets.clamp(max=S - 1)
            idx = offsets.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
            new_ctx = ctx.scatter(1, idx, layer_tok)
        return (args[0], args[1], new_ctx) + tuple(args[3:])

    return block_pre_hook


def _merge_object_patch(model, key: str, hook) -> None:
    """Append ``hook`` to the OrderedDict at ``key``, composing with any prior
    object-patch on the same hook dict (so chaining adapter/postfix nodes plus
    soft tokens preserves every node's hooks)."""
    # get_model_object resolves the full dotted key through object_patches /
    # backups before falling back to the live attribute, so a prior patch on
    # the same hook dict is preserved.
    base = model.get_model_object(key)
    new_hooks = OrderedDict(base)
    new_hooks[id(hook)] = hook
    model.add_object_patch(key, new_hooks)


def apply_soft_tokens(model, file_path: str, strength: float) -> bool:
    """Install the soft-token live-splice hooks on ``model`` (already a clone).

    Returns True if applied. A no-op (returns False) at ``strength == 0``.
    """
    if strength == 0:
        return False

    bank = load_soft_tokens(file_path)
    dit = model.get_model_object("diffusion_model")
    blocks = getattr(dit, "blocks", None)
    if blocks is None:
        raise RuntimeError(
            "diffusion_model has no .blocks — soft tokens need an Anima/cosmos DiT."
        )
    if len(blocks) < bank.n_layers:
        raise RuntimeError(
            f"diffusion_model has {len(blocks)} blocks but the soft-token bank "
            f"declares n_layers={bank.n_layers}."
        )

    # Per-load shared state. The diffusion_model pre-hook writes step_tokens
    # each step; every block pre-hook reads it.
    state: dict = {}
    step_hook = _make_step_pre_hook(state, bank, strength)
    _merge_object_patch(model, "diffusion_model._forward_pre_hooks", step_hook)

    for k in range(bank.n_layers):
        block_hook = _make_block_pre_hook(
            k, state, bank.splice_position, bank.num_tokens
        )
        _merge_object_patch(
            model, f"diffusion_model.blocks.{k}._forward_pre_hooks", block_hook
        )

    logger.info(
        f"soft_tokens: installed splice pre-hooks on first {bank.n_layers} blocks "
        f"(K={bank.num_tokens}, splice={bank.splice_position}, strength={strength})"
    )
    return True
