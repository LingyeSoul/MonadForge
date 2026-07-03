"""Split fused attention projections into per-component Linears (LoKR-only).

Background
----------
The Anima DiT model (``library/anima/models.py``) defines self-attention with
a single fused ``qkv_proj`` Linear (out = 3*inner_dim) and cross-attention
with a fused ``kv_proj`` Linear (out = 2*inner_dim). ComfyUI checkpoints — and
therefore ComfyUI inference — store these as separate ``q_proj`` / ``k_proj``
/ ``v_proj`` Linears.

For plain LoRA this is a non-issue: training on the fused Linear produces a
single ``lora_up.weight`` whose rows are concatenated q/k/v outputs, and
``defuse_standard_qkv`` (``lora_modules/lora.py``) chunks them at save time
back into per-component keys.

For **LoKR** it is unsalvageable on the save side: ``ΔW = kron(w1, w2)``
factors the output dimension as ``out_a * out_b``, so the q/k/v output bands
are interleaved across the Kronecker product and cannot be sliced at any
clean boundary. The result is that a LoKR file saved from fused training
carries ``…_self_attn_qkv_proj.lokr_*`` keys, which ComfyUI cannot map to its
split ``to_q``/``to_k``/``to_v`` modules — every attention key is reported
``lora key not loaded`` and the LoRA is silently inert.

Fix (this module)
-----------------
Split the fused projection **before** the LoRA network attaches modules, but
only for the LoKR variant. The split replaces the fused Linear attribute on
the ``Attention`` module with ``n`` independent ``nn.Linear`` children named
``q_proj`` / ``k_proj`` / ``v_proj`` (self-attn) or ``k_proj`` / ``v_proj``
(cross-attn). Each child's ``weight`` is a ``narrow`` **view** into the
original fused weight — zero-copy, so the frozen base parameters are not
duplicated — wrapped as a non-trainable ``nn.Parameter``.

Because the split children are ordinary ``nn.Linear`` modules under the same
container classes (``Block``), the existing ``create_modules`` walker in
``LoRANetwork`` discovers them naturally and builds one LoKR module per
component. ``state_dict()`` then emits
``lora_unet_blocks_0_self_attn_q_proj.lokr_w*`` keys directly — the exact
layout ComfyUI expects.

The ``Attention.compute_qkv`` method is monkey-patched on each affected
instance so it calls the (LoRA-wrapped) per-component Linears and
concatenates, producing the same fused ``3*inner_dim`` / ``2*inner_dim``
output the original method did. Other variants (Hydra/Ortho/Chimera/plain
LoRA) are untouched: they keep the fused Linear and rely on
``defuse_standard_qkv`` at save.

Non-goals
---------
* Does not change ``library/anima/models.py`` — the model definition stays
  fused; the split is a training-time view, only for LoKR.
* Does not handle ``merge_to_dit`` — LoKR baking reconstructs the full delta
  per component (``LoKRModule.get_weight``) and adds it into the child's
  narrow view, which writes through to the fused weight's storage in place.
  See ``tests/test_lokr_split_attn.py`` for the regression pin.
"""

from __future__ import annotations

import logging
import types
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from networks.attn_fuse import ATTN_FUSE_SPECS, AttnFuseSpec

logger = logging.getLogger(__name__)


# Fragment of the lora_name / attribute path that identifies the fused
# projection's Linear attribute on the ``Attention`` module. The lora-name
# fragment ``AttnFuseSpec.fused_frag`` is e.g. ``"self_attn_qkv_proj"`` (it
# includes the attn_type prefix because the lora_name spans the whole path),
# but the actual ``nn.Linear`` attribute on ``Attention`` is just the trailing
# ``"qkv_proj"`` / ``"kv_proj"`` — i.e. ``f"{fused_letters}_proj"``.
_FUSED_ATTR_SUFFIXES: Tuple[str, ...] = tuple(
    f"{spec.fused_letters}_proj" for spec in ATTN_FUSE_SPECS
)
# e.g. ("qkv_proj", "kv_proj")


def _spec_for_attr(attr_name: str) -> Optional[AttnFuseSpec]:
    """Return the AttnFuseSpec whose fused-projection attr name matches."""
    for spec in ATTN_FUSE_SPECS:
        if attr_name == f"{spec.fused_letters}_proj":
            return spec
    return None


def _make_split_linears(
    fused: nn.Linear,
    component_letters: Tuple[str, ...],
) -> List[Tuple[str, nn.Linear]]:
    """Build per-component Linears whose weights are narrow views of *fused*.

    The fused weight has shape ``[sum(component_out_dims), in_features]`` with
    each component occupying a contiguous ``out_features`` block along dim 0
    (q, then k, then v — the order ``Attention.compute_qkv`` unbinds in, and
    the order ``attn_fuse.ATTN_FUSE_SPECS`` fixes as load-bearing).

    Each split Linear reuses the fused weight's storage via ``narrow`` (zero
    copy) and is wrapped as a non-trainable ``nn.Parameter`` so the frozen
    base stays frozen. All components of one fused projection share the same
    ``in_features`` (they see the same Linear input).
    """
    fused_w = fused.weight
    out_total = fused.out_features
    in_features = fused.in_features
    n = len(component_letters)
    if out_total % n != 0:
        raise ValueError(
            f"Fused projection out_features {out_total} is not divisible by "
            f"component count {n} ({component_letters}); cannot split."
        )
    component_out = out_total // n

    children: List[Tuple[str, nn.Linear]] = []
    for i, letter in enumerate(component_letters):
        # narrow view along dim 0 (output rows). contiguous=False keeps it a
        # view; .contiguous() would copy and break the zero-storage invariant
        # — but Linear needs a 2D weight, and narrow on a contiguous parent
        # along dim 0 stays 2D so it's fine.
        w_view = fused_w.narrow(0, i * component_out, component_out)
        child = nn.Linear(in_features, component_out, bias=fused.bias is not None)
        # Replace the freshly-initialized weight with the fused-weight view.
        # requires_grad=False: the DiT base is frozen during LoRA training.
        child.weight = nn.Parameter(w_view, requires_grad=False)
        if child.bias is not None:
            # Fused bias (if present) is also laid out [q*, k*, v*] along dim 0.
            b_view = fused.bias.narrow(0, i * component_out, component_out)
            child.bias = nn.Parameter(b_view, requires_grad=False)
        else:
            # nn.Linear created a bias above; drop it to match the fused Linear.
            child.bias = None
        children.append((f"{letter}_proj", child))
    return children


def _split_one_attention(attn_module: nn.Module) -> int:
    """Replace fused qkv_proj/kv_proj on a single Attention with split children.

    Returns the number of fused projections split (0, 1, or rarely 2).
    Mutates *attn_module* in place: deletes the fused attribute, registers
    per-component ``nn.Linear`` children, and monkey-patches
    ``compute_qkv`` so it dispatches through the (LoRA-wrapped) children and
    concatenates — preserving the original fused output shape.

    Safe to call on attention modules that have no fused projection (no-op).
    """
    split_count = 0
    # Snapshot the fused attrs present so we can iterate then delete safely.
    fused_attrs: List[Tuple[str, AttnFuseSpec, nn.Linear]] = []
    for attr_name in dir(attn_module):
        if attr_name not in _FUSED_ATTR_SUFFIXES:
            continue
        child = getattr(attn_module, attr_name, None)
        if not isinstance(child, nn.Linear):
            continue
        spec = _spec_for_attr(attr_name)
        if spec is None:
            continue
        fused_attrs.append((attr_name, spec, child))

    if not fused_attrs:
        return 0

    # Cross-attention has BOTH a separate ``q_proj`` (kept) and a fused
    # ``kv_proj`` (split into k/v). Track which component letters are split so
    # the patched compute_qkv knows how to reassemble q/k/v.
    # Map: attn_type -> list of (letter, was_split)
    # The split children register under ``{letter}_proj``; the unsplit q_proj
    # on cross-attn already exists under ``q_proj``.
    split_letters_by_type: dict[str, List[str]] = {}

    for attr_name, spec, fused in fused_attrs:
        children = _make_split_linears(fused, spec.component_letters)
        # Register children on the Attention module directly (not under a
        # container) so the lora_name path is ``…_self_attn_q_proj`` with no
        # extra ``qkv_proj`` segment.
        for child_name, child in children:
            setattr(attn_module, child_name, child)
        # Drop the fused Linear so its stale weight no longer appears in
        # named_modules() (the walker would otherwise still see it).
        delattr(attn_module, attr_name)
        split_letters_by_type.setdefault(spec.attn_type, []).extend(
            spec.component_letters
        )
        split_count += 1

    if split_count > 0:
        _patch_compute_qkv(attn_module, split_letters_by_type)

    return split_count


def _patch_compute_qkv(
    attn_module: nn.Module,
    split_letters_by_type: dict[str, List[str]],
) -> None:
    """Monkey-patch ``compute_qkv`` to dispatch through split children.

    The patched method preserves the original signature and output contract:
    it returns ``(q, k, v)`` with each shaped ``[..., n_heads, head_dim]``,
    after QK-norm / RoPE. The only change is that q/k/v now come from
    per-component Linears (which may be LoRA-wrapped) instead of unbinding a
    fused projection.

    Two paths:

    * **self_attn split**: q,k,v all come from split children. ``x`` is fed
      to each; the fused path's ``qkv_proj(x).unflatten((3, n_heads,
      head_dim)).unbind(-3)`` becomes per-component ``q_proj(x).unflatten((
      n_heads, head_dim))``.
    * **cross_attn kv split**: q still comes from the existing ``q_proj``
      (which reads ``x`` in the original); k,v come from split children fed
      ``context``.
    """
    is_selfattn = getattr(attn_module, "is_selfattn", True)
    n_heads = attn_module.n_heads
    head_dim = attn_module.head_dim
    q_norm = attn_module.q_norm
    k_norm = attn_module.k_norm
    v_norm = attn_module.v_norm

    if is_selfattn:
        # Self-attn: q,k,v all split, all read ``x``.
        def compute_qkv(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ) -> tuple:
            q = self.q_proj(x).unflatten(-1, (n_heads, head_dim))
            k = self.k_proj(x).unflatten(-1, (n_heads, head_dim))
            v = self.v_proj(x).unflatten(-1, (n_heads, head_dim))
            q = q_norm(q)
            k = k_norm(k)
            v = v_norm(v)
            if rope_cos_sin is not None:
                from library.anima.models import apply_rotary_pos_emb_qk

                q, k = apply_rotary_pos_emb_qk(
                    q, k, rope_cos_sin, tensor_format=self.qkv_format
                )
            return q, k, v

    else:
        # Cross-attn: q reads x via existing q_proj; k,v split from kv_proj,
        # reading context.
        def compute_qkv(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ) -> tuple:
            q = self.q_proj(x).unflatten(-1, (n_heads, head_dim))
            k = self.k_proj(context).unflatten(-1, (n_heads, head_dim))
            v = self.v_proj(context).unflatten(-1, (n_heads, head_dim))
            q = q_norm(q)
            k = k_norm(k)
            v = v_norm(v)
            return q, k, v

    attn_module.compute_qkv = types.MethodType(compute_qkv, attn_module)


def split_fused_projections(model: nn.Module) -> int:
    """Walk *model* and split every fused attention projection in place.

    Returns the total number of fused Linears split across the model. Safe to
    call on a model with no fused projections (returns 0). Idempotent: a
    second call finds no ``qkv_proj`` / ``kv_proj`` attrs and returns 0.

    Discovery is duck-typed: any submodule exposing a ``qkv_proj`` or
    ``kv_proj`` attribute that is an ``nn.Linear`` is treated as a fused
    attention projection. This avoids a hard import of ``library.anima.models``
    (which would couple this helper to the heavy DiT model definition and
    break under class-rebinding in tests) while still being precise — the
    attribute names are unique to attention in the Anima DiT.

    Intended to run once on the DiT (``unet``) after model load and before
    ``LoRANetwork.create_modules``, gated on ``use_lokr``.
    """
    total = 0
    for _name, module in model.named_modules():
        # Duck-typed: any module with a fused qkv/kv Linear attribute.
        if any(
            isinstance(getattr(module, attr, None), nn.Linear)
            for attr in _FUSED_ATTR_SUFFIXES
        ):
            total += _split_one_attention(module)
    if total:
        logger.info(
            f"split_fused_projections: split {total} fused qkv/kv projection(s) "
            "into per-component Linears for LoKR training"
        )
    return total
