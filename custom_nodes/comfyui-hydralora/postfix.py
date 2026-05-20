"""Anima prefix / postfix / cond context-splicing for ComfyUI.

Splices learned vectors into the T5-compatible crossattn embedding **after**
the LLM adapter runs and after its pad-to-512 step — i.e. the same space as
anima_lora's training and reference inference. Positive-only routing via
``cond_or_uncond`` from ``transformer_options`` preserves CFG.

Hook-not-override invariant (same as Hydra / ReFT / soft_tokens, see CLAUDE.md):
we never replace ``diffusion_model.forward``. The model already runs the LLM
adapter inside its own ``forward`` and hands the SAME post-adapter
``crossattn_emb`` to every block, so we install a per-block ``forward_pre_hook``
(registered ``with_kwargs`` to read ``cond_or_uncond``) that rewrites the
block's ``crossattn_emb`` positional arg. Replacing ``forward`` strands the
DiT's own Linears (e.g. ``x_embedder``) on CPU under ComfyUI's cast-weights /
dynamic-VRAM staging walk — the hook leaves ``forward`` intact so staging and
``unpatch_model`` keep working.

Modes auto-detected from safetensors keys/metadata:
  - prefix : learned vectors prepended; last K padding slots trimmed
  - postfix: static learned vectors spliced after real text tokens
  - cond   : caption-conditional + structurally-orthogonal postfix. An MLP
             (LayerNorm → Linear → GELU → Linear) over maxabs-pooled content
             emits a Cayley rotation seed S(c) + magnitude λ(c); the postfix is
             ``Cayley(S(c) − S(c)ᵀ) @ ortho_basis · λ(c)``. Mirrors
             ``networks/methods/postfix.py`` (cond+ortho, post-commit e989d64).
"""

import logging
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _build_cond_mlp(embed_dim: int, hidden_dim: int, num_tokens: int) -> nn.Sequential:
    """cond+ortho head: LayerNorm → Linear → GELU → Linear.

    The last Linear emits ``K(K-1)/2 + 1`` scalars per caption: a strict
    upper-triangular Cayley rotation seed (K(K-1)/2 entries) plus one magnitude
    λ(c). Matches ``PostfixNetwork.__init__`` (cond branch).
    """
    n_out = num_tokens * (num_tokens - 1) // 2 + 1
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, n_out),
    )


# Cache: path -> (mode, payload, num_tokens, splice_position)
# payload is a Tensor for prefix/postfix, (nn.Sequential, basis, K, embed_dim) for cond.
_weight_cache: Dict[str, Tuple[str, object, int, str]] = {}


def load_postfix(file_path: str) -> Tuple[str, object, int, str]:
    """Parse a prefix/postfix/cond safetensors file once, cache by path."""
    if file_path in _weight_cache:
        return _weight_cache[file_path]

    from safetensors import safe_open
    from safetensors.torch import load_file

    weights_sd = load_file(file_path)

    metadata_mode: Optional[str] = None
    metadata_splice: Optional[str] = None
    with safe_open(file_path, framework="pt") as f:
        meta = f.metadata() or {}
        metadata_mode = meta.get("ss_mode")
        metadata_splice = meta.get("ss_splice_position")

    has_cond = any(k.startswith("cond_mlp.") for k in weights_sd)
    splice_position = metadata_splice or "end_of_sequence"

    if has_cond or metadata_mode == "cond":
        # cond+ortho layout: LayerNorm(0) → Linear(1) → GELU(2) → Linear(3).
        # First Linear at cond_mlp.1, last Linear at cond_mlp.3 (outputs
        # K(K-1)/2 + 1). K and embed_dim come from the frozen ortho_basis.
        w1 = weights_sd.get("cond_mlp.1.weight")
        w3 = weights_sd.get("cond_mlp.3.weight")
        if w1 is None or w3 is None:
            raise ValueError(
                "cond mode requires cond_mlp.1.weight and cond_mlp.3.weight "
                f"(got keys: {[k for k in weights_sd if 'cond_mlp' in k]}). "
                "Legacy 2-layer cond checkpoints (no LayerNorm, no ortho_basis) "
                "are no longer supported — retrain in mode='cond'."
            )
        basis = weights_sd.get("ortho_basis")
        if basis is None:
            raise ValueError(
                "cond mode requires an 'ortho_basis' tensor (legacy non-ortho "
                "cond checkpoints are no longer loadable — retrain in mode='cond')."
            )
        basis = basis.float()  # (K, D), row-orthonormal; keep fp32 for the Cayley solve
        num_tokens, embed_dim = basis.shape
        hidden_dim = w1.shape[0]
        if w1.shape[1] != embed_dim:
            raise ValueError(
                f"cond_mlp input dim {w1.shape[1]} != ortho_basis dim {embed_dim}"
            )
        expected_n_out = num_tokens * (num_tokens - 1) // 2 + 1
        if w3.shape[0] != expected_n_out:
            raise ValueError(
                f"cond_mlp last-layer dim {w3.shape[0]} != expected {expected_n_out} "
                f"for K={num_tokens}"
            )

        mlp = _build_cond_mlp(embed_dim, hidden_dim, num_tokens)
        mlp_sd = {
            k[len("cond_mlp.") :]: v
            for k, v in weights_sd.items()
            if k.startswith("cond_mlp.")
        }
        missing, unexpected = mlp.load_state_dict(mlp_sd, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"cond_mlp load mismatch: missing={missing}, unexpected={unexpected}"
            )
        mlp.eval()
        for p in mlp.parameters():
            p.requires_grad_(False)
        result = (
            "cond",
            (mlp, basis, num_tokens, embed_dim),
            num_tokens,
            splice_position,
        )
    elif "prefix_embeds" in weights_sd:
        embeds = weights_sd["prefix_embeds"]
        result = ("prefix", embeds, embeds.shape[0], splice_position)
    elif "postfix_embeds" in weights_sd:
        embeds = weights_sd["postfix_embeds"]
        result = ("postfix", embeds, embeds.shape[0], splice_position)
    else:
        raise ValueError(
            f"Unsupported postfix file (keys: {list(weights_sd.keys())[:10]}). "
            f"Expected 'prefix_embeds', 'postfix_embeds', or 'cond_mlp.*'."
        )

    _weight_cache[file_path] = result
    logger.info(
        f"Loaded {result[0]} weights: {result[2]} tokens from {file_path} "
        f"(splice={splice_position})"
    )
    return result


def _prepend_prefix(ctx: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
    K = prefix.shape[0]
    B, S, _ = ctx.shape
    prefix = (
        prefix.unsqueeze(0).expand(B, -1, -1).to(dtype=ctx.dtype, device=ctx.device)
    )
    return torch.cat([prefix, ctx[:, : S - K, :]], dim=1)


def _splice_postfix(
    ctx: torch.Tensor, postfix: torch.Tensor, splice_position: str
) -> torch.Tensor:
    B, S, D = ctx.shape
    K = postfix.shape[1]
    postfix = postfix.to(dtype=ctx.dtype, device=ctx.device)
    if splice_position == "end_of_sequence":
        return torch.cat([ctx[:, : S - K, :], postfix], dim=1)
    mask = ctx.abs().sum(dim=-1) > 0
    seqlens = mask.long().sum(dim=-1)
    offsets = seqlens.unsqueeze(1) + torch.arange(K, device=ctx.device)
    offsets = offsets.clamp(max=S - 1)
    idx = offsets.unsqueeze(-1).expand(-1, -1, D)
    return ctx.scatter(1, idx, postfix)


def _apply_cfg(
    ctx: torch.Tensor, mode: str, payload, splice_position: str, strength: float
) -> torch.Tensor:
    if strength == 0:
        return ctx
    if mode == "prefix":
        return _prepend_prefix(ctx, payload * strength)
    if mode == "cond":
        mlp, basis, K, embed_dim = payload
        mlp.to(device=ctx.device, dtype=torch.float32)
        basis = basis.to(device=ctx.device, dtype=torch.float32)
        B = ctx.shape[0]

        # Maxabs-pool over content (non-padding) slots, sign preserved — mirrors
        # PostfixNetwork.append_postfix. Mean-pool drags every caption onto the
        # T5 DC cone (cos μ≈0.84); maxabs keeps caption-distinct signal (cos≈0.22).
        # Padding rows are zero, so set their |·| to -1 to lose every argmax.
        content = ctx.abs().sum(dim=-1) > 0  # (B, S) bool
        abs_emb = ctx.float().abs().masked_fill(~content.unsqueeze(-1), -1.0)
        idx = abs_emb.argmax(dim=1, keepdim=True)  # (B, 1, D)
        pooled = ctx.float().gather(dim=1, index=idx).squeeze(1)  # (B, D)

        with torch.no_grad():
            cond_out = mlp(pooled)  # (B, K(K-1)/2 + 1)

        # Reconstruct postfix(c) = Cayley(S(c) − S(c)ᵀ) @ basis · λ(c). λ_init is
        # already baked into the last Linear's bias, so no extra term here.
        n_skew = K * (K - 1) // 2
        S_seed = cond_out[:, :n_skew].float()
        lam_c = cond_out[:, -1].float()
        triu = torch.triu_indices(K, K, offset=1, device=ctx.device)
        S_c = pooled.new_zeros(B, K, K, dtype=torch.float32)
        S_c[:, triu[0], triu[1]] = S_seed
        A = S_c - S_c.transpose(-1, -2)
        eye = torch.eye(K, device=ctx.device, dtype=torch.float32)
        R = torch.linalg.solve(eye + A, eye - A)  # Cayley rotation (B, K, K)
        postfix = torch.matmul(R, basis) * lam_c[:, None, None]  # (B, K, D), fp32
        postfix = (postfix * strength).to(ctx.dtype)
        return _splice_postfix(ctx, postfix, splice_position)
    # static postfix
    B = ctx.shape[0]
    postfix = (payload * strength).unsqueeze(0).expand(B, -1, -1)
    return _splice_postfix(ctx, postfix, splice_position)


def _cond_rows(B: int, cond_or_uncond) -> list:
    """Row indices of the positive (cond) batch elements.

    ``cond_or_uncond`` is ComfyUI's per-group kind list (0=cond, 1=uncond) for
    the rows packed into this forward; absent (separate cond/uncond passes that
    don't tag, or no CFG) → every row. Mirrors the pre-rewrite forward-override
    routing so outputs are unchanged for setups where the node already worked.
    """
    if cond_or_uncond:
        per_group = max(B // len(cond_or_uncond), 1)
        return [
            j
            for i, kind in enumerate(cond_or_uncond)
            if kind == 0
            for j in range(i * per_group, (i + 1) * per_group)
        ]
    return list(range(B))


def _make_block_pre_hook(mode, payload, splice_position, strength):
    """Per-block ``with_kwargs`` pre-hook that splices the postfix into
    ``crossattn_emb`` for the positive rows only.

    The block is called positionally as ``block(x, emb, crossattn_emb,
    **block_kwargs)`` (comfy ``predict2.Block.forward``), so ``args[2]`` is the
    post-adapter cross-attention text embedding and ``kwargs['transformer_options']``
    carries ``cond_or_uncond``. We return a new ``(args, kwargs)`` with the
    ``crossattn_emb`` slot rewritten; ``forward`` itself is untouched.

    dynamo-disabled like the soft-token / Hydra hooks — the pool + Cayley solve
    are eager Python on tiny tensors and never need to trace into a compiled graph.
    """

    @torch._dynamo.disable
    def block_pre_hook(module, args, kwargs):
        if len(args) < 3 or args[2] is None:
            return None
        ctx = args[2]
        to = kwargs.get("transformer_options") or {}
        rows = _cond_rows(ctx.shape[0], to.get("cond_or_uncond"))
        if not rows:
            return None
        idx = torch.tensor(rows, device=ctx.device, dtype=torch.long)
        sub = ctx.index_select(0, idx)
        sub = _apply_cfg(sub, mode, payload, splice_position, strength)
        new_ctx = ctx.index_copy(0, idx, sub)
        return (args[0], args[1], new_ctx) + tuple(args[3:]), kwargs

    return block_pre_hook


def _merge_pre_hook(model, key: str, hook, with_kwargs: bool = False) -> None:
    """Append ``hook`` to the ``_forward_pre_hooks`` OrderedDict at ``key``,
    composing with any prior object-patch on the same dict (so chaining
    adapter / postfix / soft-token nodes preserves every node's hooks).

    When ``with_kwargs`` is set, also register the hook id in the sibling
    ``_forward_pre_hooks_with_kwargs`` map so ``Module._call_impl`` passes
    ``kwargs`` to it (PyTorch keys the with-kwargs dispatch on the hook id).
    """
    base = model.get_model_object(key)
    new_hooks = OrderedDict(base)
    new_hooks[id(hook)] = hook
    model.add_object_patch(key, new_hooks)
    if with_kwargs:
        wk_key = key + "_with_kwargs"
        wk_base = model.get_model_object(wk_key)
        new_wk = OrderedDict(wk_base)
        new_wk[id(hook)] = True
        model.add_object_patch(wk_key, new_wk)


def apply_postfix(model, file_path: str, strength: float) -> bool:
    """Install per-block postfix-splice pre-hooks on ``model`` (already a
    clone). Returns True if applied; a no-op (False) at ``strength == 0``.

    The model runs the LLM adapter inside its own ``forward`` and passes the
    same post-adapter ``crossattn_emb`` to every block, so we hook all blocks —
    splicing one block's input doesn't propagate to the others.
    """
    if strength == 0:
        return False

    mode, payload, _, splice_position = load_postfix(file_path)
    dit = model.get_model_object("diffusion_model")
    blocks = getattr(dit, "blocks", None)
    if blocks is None:
        raise RuntimeError(
            "diffusion_model has no .blocks — postfix needs an Anima/cosmos DiT."
        )

    for k in range(len(blocks)):
        hook = _make_block_pre_hook(mode, payload, splice_position, strength)
        _merge_pre_hook(
            model,
            f"diffusion_model.blocks.{k}._forward_pre_hooks",
            hook,
            with_kwargs=True,
        )

    logger.info(
        f"postfix: installed {mode} splice pre-hooks on {len(blocks)} blocks "
        f"(splice={splice_position}, strength={strength})"
    )
    return True
