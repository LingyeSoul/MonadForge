"""NetworkSpec registry for LoRA adapter-method dispatch.

Replaces the flag-cascade in ``networks.lora_anima.create_network`` with a
declarative map. Each entry pairs an adapter variant name with the module
class it instantiates and a ``save_variant`` label consumed by
``networks.lora_save``.

Flag precedence (evaluated top to bottom, first match wins):

    use_moe_style="independent_A"        → stacked_experts_global_fei
    use_moe_style="shared_A" + use_ortho → ortho_hydra
    use_moe_style="shared_A"             → hydra
    use_hydra (legacy) + use_ortho       → ortho_hydra
    use_hydra (legacy)                   → hydra
    use_dora                             → dora
    use_ortho                            → ortho
    (none)                               → lora

Ambiguous combinations (``use_dora`` + ``use_ortho``, ``use_hydra`` + ``use_moe_style``,
…) raise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type

from networks.lora_deprecated import DoRAModule
from networks.lora_modules import (
    HydraLoRAModule,
    LoRAModule,
    OrthoHydraLoRAExpModule,
    OrthoLoRAExpModule,
    StackedExpertsLoRAModule,
)


@dataclass(frozen=True)
class NetworkSpec:
    """Descriptor for one adapter variant.

    Attributes:
        name: Stable identifier stamped on the network and written to
            metadata as ``ss_network_spec``. Also the key into
            ``NETWORK_REGISTRY``.
        module_class: Concrete ``LoRAModule`` subclass the network will
            instantiate per target module.
        save_variant: Label keyed into ``networks.lora_save.SAVE_HANDLERS``
            — selects the serialization pipeline for this variant.
        kwarg_flags: Tuple of kwargs this variant consumes beyond
            ``SHARED_KWARG_FLAGS``. Combined with the shared set by
            ``all_network_kwargs()`` to populate argparse schema and
            forward TOML-level args into ``create_network``. Single
            source of truth for what keys train.py recognizes.
        post_init: Optional hook run after the network is built; receives
            ``(network, kwargs)``. Used for variant-specific attribute
            attachment (e.g. hydra balance loss weight).
    """

    name: str
    module_class: Type
    save_variant: str = "standard"
    kwarg_flags: Tuple[str, ...] = ()
    post_init: Optional[Callable[[Any, Mapping[str, Any]], None]] = None


# Kwargs every LoRA-family variant consumes in ``create_network``: core
# targeting knobs + cross-cutting add-ons (ReFT, channel scaling,
# LoRA+, T-LoRA). Cross-cutting because these compose on top of any
# variant rather than belonging to a single one.
SHARED_KWARG_FLAGS: Tuple[str, ...] = (
    # Core network targeting / knobs
    "train_llm_adapter",
    "exclude_patterns",
    "include_patterns",
    "layer_start",
    "layer_end",
    "rank_dropout",
    "module_dropout",
    "verbose",
    # Regex-driven per-module rank / lr overrides
    "network_reg_dims",
    "network_reg_lrs",
    # HydraLoRA router (+ σ-router MLP) LR multiplier on top of unet_lr / reg_lr
    "network_router_lr_scale",
    # LoRA+
    "loraplus_lr_ratio",
    "loraplus_unet_lr_ratio",
    "loraplus_text_encoder_lr_ratio",
    # T-LoRA (timestep-dependent rank masking)
    "use_timestep_mask",
    "min_rank",
    "alpha_rank_scale",
    # Per-channel input pre-scaling (SmoothQuant-style)
    "per_channel_scaling",
    "channel_stats_path",
    "channel_scaling_alpha",
    # Memory-saving down-projection autograd (classic LoRA only; bitwise-equal grads)
    "use_custom_down_autograd",
    # Variant selectors (read by resolve_network_spec)
    "use_dora",
    "use_hydra",
    "use_ortho",
    # New three-axis routing config (see plan2.md §three-axis-config). Drives
    # `LoRANetworkCfg.from_kwargs` translation; `resolve_network_spec` also
    # dispatches on `use_moe_style="independent_A"` → `stacked_experts_global_fei`.
    "use_moe_style",
    "route_per_layer",
    "router_source",
    # GlobalRouter knobs (consumed only when route_per_layer=False).
    "router_hidden_dim",
    "router_tau",
    # FECL knobs (FeRA auxiliary loss; opt-in via fera_fecl_weight > 0).
    "fera_fecl_weight",
    "fera_num_bands",
    # ReFT add-on (composes with any variant)
    "add_reft",
    "reft_dim",
    "reft_alpha",
    "reft_layers",
    # REPA-style auxiliary alignment (composes with any variant). The factory
    # ``_maybe_attach_repa_head`` reads ``use_repa`` to decide whether to attach
    # the head, then sizes/weights it from the rest. Selection of the hooked
    # block index and the loss weight live as top-level argparse flags
    # (``--repa_layer``, ``--repa_weight``) — the network module doesn't need
    # them, only the adapter / loss handler does.
    "use_repa",
    "repa_dit_dim",
    "repa_hidden_dim",
    "repa_encoder_dim",
    "repa_lr_scale",
)


def _post_init_hydra(network: Any, kwargs: Mapping[str, Any]) -> None:
    blw = kwargs.get("balance_loss_weight")
    target = float(blw) if blw is not None else 0.01
    warmup = kwargs.get("balance_loss_warmup_ratio")
    warmup_ratio = float(warmup) if warmup is not None else 0.0
    network._balance_loss_target_weight = target
    network._balance_loss_warmup_ratio = warmup_ratio
    # Hold the balance penalty at 0 during the warmup window so the router can
    # specialize before load-balancing kicks in; flipped to `target` by
    # LoRANetwork.step_balance_loss_warmup once global_step crosses the ratio.
    network._balance_loss_weight = 0.0 if warmup_ratio > 0.0 else target
    network._use_hydra = True
    # FECL weight surface: ``_fera_fecl_loss`` reads
    # ``ctx.network.fecl_weight``. Mirror the cfg value here (and fall back
    # to the kwarg when no cfg is present — keeps unit tests that hand a
    # network without a full cfg working).
    cfg_weight = getattr(getattr(network, "cfg", None), "fera_fecl_weight", None)
    if cfg_weight is not None:
        network.fecl_weight = float(cfg_weight)
    else:
        network.fecl_weight = float(kwargs.get("fera_fecl_weight", 0.0) or 0.0)


_HYDRA_KWARG_FLAGS: Tuple[str, ...] = (
    "num_experts",
    "balance_loss_weight",
    "balance_loss_warmup_ratio",
    "expert_init_std",
    "expert_warmup_ratio",
    "expert_warmup_k",
    "expert_best_warmup_ratio",
    # Layer filter — concentrates MoE routers on cross-attn + MLP
    "hydra_router_layers",
    # σ-conditional router add-on (hydra-only)
    "use_sigma_router",
    "sigma_router_layers",
    "sigma_feature_dim",
    "sigma_hidden_dim",
    "per_bucket_balance_weight",
    "num_sigma_buckets",
    "specialize_experts_by_sigma_buckets",
    "sigma_bucket_boundaries",
    # FEI-conditional router (FeRA-style content-aware routing, hydra-only)
    "use_fei_router",
    "fei_feature_dim",
    "fei_sigma_low_div",
    "fei_router_layers",
)


NETWORK_REGISTRY: Dict[str, NetworkSpec] = {
    "lora": NetworkSpec(
        name="lora",
        module_class=LoRAModule,
        save_variant="standard",
    ),
    "ortho": NetworkSpec(
        name="ortho",
        module_class=OrthoLoRAExpModule,
        save_variant="ortho_to_lora",
    ),
    "hydra": NetworkSpec(
        name="hydra",
        module_class=HydraLoRAModule,
        save_variant="hydra_moe",
        kwarg_flags=_HYDRA_KWARG_FLAGS,
        post_init=_post_init_hydra,
    ),
    "ortho_hydra": NetworkSpec(
        name="ortho_hydra",
        module_class=OrthoHydraLoRAExpModule,
        save_variant="ortho_hydra_to_hydra",
        kwarg_flags=_HYDRA_KWARG_FLAGS,
        post_init=_post_init_hydra,
    ),
    # FeRA paper-faithful: independent-A stacked experts, single network-level
    # router fed by FEI(z_t). See plan2.md §three-axis-config — selected via
    # ``use_moe_style="independent_A"``. Save handler stub: task #4 wires the
    # real serialization path; until then this spec is reachable only via the
    # gui-methods/fera.toml variant (also a task #6 deliverable).
    "stacked_experts_global_fei": NetworkSpec(
        name="stacked_experts_global_fei",
        module_class=StackedExpertsLoRAModule,
        save_variant="stacked_experts_global_fei",
        kwarg_flags=_HYDRA_KWARG_FLAGS,
        post_init=_post_init_hydra,
    ),
    "dora": NetworkSpec(
        name="dora",
        module_class=DoRAModule,
        save_variant="standard",
    ),
}


def all_network_kwargs() -> Tuple[str, ...]:
    """Return the union of shared + per-variant kwargs, sorted.

    Single source of truth for train.py — populates the argparse schema
    and the TOML → ``net_kwargs`` forwarding list, so adding a new kwarg
    to a ``NetworkSpec`` (or to ``SHARED_KWARG_FLAGS``) automatically
    makes it visible to training without touching train.py.
    """
    merged: set[str] = set(SHARED_KWARG_FLAGS)
    for spec in NETWORK_REGISTRY.values():
        merged.update(spec.kwarg_flags)
    return tuple(sorted(merged))


def _parse_bool_flag(kwargs: Mapping[str, Any], key: str) -> bool:
    v = kwargs.get(key, False)
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() == "true"


def resolve_network_spec(kwargs: Mapping[str, Any]) -> NetworkSpec:
    """Resolve which NetworkSpec to instantiate from create_network kwargs.

    Precedence is deterministic and documented in the module docstring.
    Raises on mutually-exclusive combinations.

    Honors the new ``use_moe_style`` axis (plan2.md): ``"independent_A"``
    routes to ``stacked_experts_global_fei`` (FeRA); ``"shared_A"`` plus
    ``use_ortho`` routes to ``ortho_hydra``; bare ``"shared_A"`` routes to
    ``hydra``. Legacy ``use_hydra=true`` is still accepted and treated as
    ``use_moe_style="shared_A"``.
    """
    use_hydra = _parse_bool_flag(kwargs, "use_hydra")
    use_dora = _parse_bool_flag(kwargs, "use_dora")
    use_ortho = _parse_bool_flag(kwargs, "use_ortho")

    # New axis: ``use_moe_style`` = False / "shared_A" / "independent_A".
    raw_moe = kwargs.get("use_moe_style")
    if isinstance(raw_moe, str):
        moe_style = raw_moe.strip()
        if moe_style.lower() in ("false", "none", ""):
            moe_style = ""
    elif raw_moe is False or raw_moe is None:
        moe_style = ""
    else:
        raise ValueError(
            f"use_moe_style={raw_moe!r}: expected False, 'shared_A', or 'independent_A'."
        )
    if moe_style not in ("", "shared_A", "independent_A"):
        raise ValueError(
            f"use_moe_style={raw_moe!r}: expected False, 'shared_A', or 'independent_A'."
        )

    # Reject ambiguous combinations early so the user gets a clear message
    # instead of silently-picked winner.
    if use_dora and (use_hydra or use_ortho or moe_style):
        raise ValueError(
            "use_dora is mutually exclusive with use_hydra / use_ortho / use_moe_style"
        )
    if moe_style and use_hydra:
        raise ValueError(
            "use_hydra (legacy) and use_moe_style cannot both be set — pick one."
        )

    if moe_style == "independent_A":
        return NETWORK_REGISTRY["stacked_experts_global_fei"]
    if moe_style == "shared_A":
        return (
            NETWORK_REGISTRY["ortho_hydra"] if use_ortho else NETWORK_REGISTRY["hydra"]
        )
    if use_hydra and use_ortho:
        return NETWORK_REGISTRY["ortho_hydra"]
    if use_hydra:
        return NETWORK_REGISTRY["hydra"]
    if use_dora:
        return NETWORK_REGISTRY["dora"]
    if use_ortho:
        return NETWORK_REGISTRY["ortho"]
    return NETWORK_REGISTRY["lora"]


__all__ = [
    "NetworkSpec",
    "NETWORK_REGISTRY",
    "SHARED_KWARG_FLAGS",
    "all_network_kwargs",
    "resolve_network_spec",
]
