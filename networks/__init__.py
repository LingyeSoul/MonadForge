"""NetworkSpec registry for LoRA adapter-method dispatch.

Replaces the flag-cascade in ``networks.lora_anima.create_network`` with a
declarative map. Each entry pairs an adapter variant name with the module
class it instantiates and a ``save_variant`` label consumed by
``networks.lora_save``.

Flag precedence (evaluated top to bottom, first match wins):

    use_chimera_hydra                    → chimera_hydra
    use_moe_style="independent_A"        → stacked_experts_global_fei
    use_moe_style="shared_A" + use_ortho → ortho_hydra
    use_moe_style="shared_A"             → hydra
    use_ortho_init                       → ortho_init
    use_ortho                            → ortho
    (none)                               → lora

The legacy ``use_hydra`` / ``use_sigma_router`` / ``use_fei_router``
kwargs were retired in plan2 task #6 — see ``LoRANetworkCfg.from_kwargs``
for the rejection message. ``use_dora`` was retired alongside the
``lora_deprecated`` module; DoRA is no longer trained, saved, or loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type

from networks.lora_modules import (
    ChimeraHydraLoRAModule,
    HydraLoRAModule,
    LoRAModule,
    OrthoHydraLoRAModule,
    OrthoInitLoRAModule,
    OrthoLoRAModule,
    StackedExpertsLoRAModule,
    StepExpertLoRAModule,
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
        post_init: Optional hook run after the network is built; receives
            ``(network, kwargs)``. Used for variant-specific attribute
            attachment (e.g. hydra balance loss weight).
    """

    name: str
    module_class: Type
    save_variant: str = "standard"
    post_init: Optional[Callable[[Any, Mapping[str, Any]], None]] = None


# Single flat allowlist of every TOML key the LoRA family forwards into
# ``create_network``. It passes these keys through the config-schema validator
# and tells ``train.py`` which keys to copy off ``args`` into ``net_kwargs``.
# Closed mirror of what ``LoRANetworkCfg.from_kwargs`` (+ ``_post_init_hydra``)
# read via ``kwargs.get(...)`` — keep in sync: a new ``kwargs.get("foo")`` there
# means adding ``"foo"`` here, or the kwarg is inert and fails the config test.
NETWORK_KWARGS: frozenset[str] = frozenset(
    {
        "train_llm_adapter",
        "exclude_patterns",
        "include_patterns",
        "layer_start",
        "layer_end",
        "rank_dropout",
        "module_dropout",
        "verbose",
        "network_reg_dims",
        "network_reg_lrs",
        "network_router_lr_scale",
        "loraplus_lr_ratio",
        "loraplus_unet_lr_ratio",
        "loraplus_text_encoder_lr_ratio",
        "use_timestep_mask",
        "min_rank",
        "alpha_rank_scale",
        # Per-channel input pre-scaling, gated by alpha (0.0 off; 0.5 sqrt
        # balance; 1.0 flatten). Calib: networks/calibration/channel_stats.safetensors.
        "channel_scaling_alpha",
        # DEPRECATED no-op (fp32-bottleneck path removed 2026-06-10); kept so old
        # snapshot TOMLs replay. The factory logs and ignores it.
        "use_custom_down_autograd",
        # Three-axis routing + variant selectors (read by resolve_network_spec /
        # LoRANetworkCfg.from_kwargs). use_ortho/use_ortho_init are mutually
        # exclusive; use_moe_style="independent_A" → stacked_experts_global_fei.
        "use_ortho",
        "use_ortho_init",
        "ortho_init_std",
        "use_moe_style",
        "route_per_layer",
        "router_source",
        # GlobalRouter knobs (consumed only when route_per_layer=False).
        "router_hidden_dim",
        "router_tau",
        # FECL: FeRA auxiliary loss, opt-in via fera_fecl_weight > 0.
        "fera_fecl_weight",
        "fera_num_bands",
        "num_experts",
        "balance_loss_weight",
        "balance_loss_warmup_ratio",
        "expert_init_std",
        # OrthoHydra centered-gate init: recenter gate to (g_e - 1/E) + zero-init
        # router + start λ at ortho_lambda_init, so the router gets a step-0
        # gradient while ΔW stays 0.
        "ortho_centered_gate",
        "ortho_lambda_init",
        # Scopes which Linears participate in routed adaptation.
        "router_targets",
        # σ-conditional router (router_source="sigma").
        "sigma_feature_dim",
        "per_bucket_balance_weight",
        "num_sigma_buckets",
        "specialize_experts_by_sigma_buckets",
        "sigma_bucket_boundaries",
        # FEI-conditional router (router_source="fei").
        "fei_feature_dim",
        "fei_sigma_low_div",
        # Step-expert (turbo DP-DMD student): K step-indexed up-heads on the
        # shared down-proj. Presence (>1) selects the step_expert spec.
        "step_expert_K",
        # ChimeraHydra dual-pool routing.
        "use_chimera_hydra",
        "num_experts_content",
        "num_experts_freq",
        # Per-pool balance weights. Fall back to balance_loss_weight when unset.
        "balance_w_content",
        "balance_w_freq",
        # FreqRouter init magnitude (non-zero so the freq pool differentiates at step 0).
        "freq_router_init_std",
        # Per-modality LN on FreqRouter input. Active only when both FEI and σ
        # blocks are on — equalizes variance so the σ block doesn't overpower the FEI simplex.
        "freq_router_layer_norm",
        # Freq-pool routing mode: "learned" (FreqRouter MLP) or "fei" (hardwire
        # π_f = normalize(FEI ** (1/τ)), no params; requires num_experts_freq == fei_feature_dim).
        "freq_router_mode",
        "freq_router_tau",
        # Per-pool router LR multipliers — stack on top of network_router_lr_scale.
        "network_content_router_lr_scale",
        "network_freq_router_lr_scale",
        # Parameterless LN on the ContentRouter's pooled crossattn_emb input.
        "content_router_layer_norm",
        # ContentRouter output-layer init magnitude (default 0.0 = zero-init).
        "content_router_init_std",
        # Centered-gate λ init for BOTH chimera pools (always-on).
        "chimera_lambda_init",
        # Per-expert capability levers (frozen-Cayley chimera; distill away).
        "chimera_expert_basis_mult",
        "chimera_expert_diag",
        # REPA v2 auxiliary alignment loss (docs/experimental/repa.md). Off by default.
        "use_repa",
        "repa_mode",  # "relational" (Gram, no head) | "absolute" (patchwise + head)
        "repa_weight",
        "repa_layer",
        "repa_encoder",  # vision-encoder registry name (default pe_spatial)
        "repa_lr_scale",  # head LR multiplier (absolute mode)
        "repa_anneal_steps",  # hard cutoff: (0,1] = fraction of run, >1 = opt steps
        "repa_spatial_norm",  # iREPA target-side spatial standardization (relational)
        # Signed timestep reweighting (0 = uniform): >0 emphasizes high noise. See repa.py.
        "repa_timestep_weighting",
        # Lever-3 gate diagnostic: probe alignment-gradient heatmap every N
        # micro-steps (0 = off); dumps <output_name>_repa_grad_heatmap.npz.
        "repa_grad_heatmap",
        # REPA-DoG target band-pass (docs/proposal/repa_dog_target.md): broader
        # low-band strip than spatial_norm's DC removal. Off by default.
        "repa_target_dog",  # false = off (no-op); true ⇒ DoG band-pass the target
        "repa_dog_sigma1_div",  # σ₁ = min(gh,gw)/div (outer, broad low band removed)
        "repa_dog_sigma2_div",  # 0 ⇒ σ₂ off (low-band strip only); >div1 ⇒ band-pass
        "repa_dog_norm_std",  # 0 ⇒ empirical std (matches spatial_norm); >0 = fixed
    }
)


def _post_init_hydra(network: Any, kwargs: Mapping[str, Any]) -> None:
    blw = kwargs.get("balance_loss_weight")
    target = float(blw) if blw is not None else 0.01
    warmup = kwargs.get("balance_loss_warmup_ratio")
    warmup_ratio = float(warmup) if warmup is not None else 0.0
    network._balance_loss_target_weight = target
    network._balance_loss_warmup_ratio = warmup_ratio
    # Hold the balance penalty at 0 during warmup so the router can specialize
    # first; flipped to `target` by LoRANetwork.step_balance_loss_warmup.
    network._balance_loss_weight = 0.0 if warmup_ratio > 0.0 else target
    network._use_hydra = True
    # Mirror cfg.fera_fecl_weight to network.fecl_weight (where _fera_fecl_loss
    # reads it); fall back to the kwarg when no cfg is present (unit tests).
    cfg_weight = getattr(getattr(network, "cfg", None), "fera_fecl_weight", None)
    if cfg_weight is not None:
        network.fecl_weight = float(cfg_weight)
    else:
        network.fecl_weight = float(kwargs.get("fera_fecl_weight", 0.0) or 0.0)

    # ChimeraHydra: stamp the chimera flag + per-pool balance weights for
    # ``get_balance_loss``; falls back to the shared ``balance_loss_weight``
    # (OrthoHydra default) when per-pool weights are unset.
    cfg = getattr(network, "cfg", None)
    if cfg is not None and getattr(cfg, "use_chimera_hydra", False):
        network._use_chimera_hydra = True
        w_c = cfg.balance_w_content if cfg.balance_w_content is not None else target
        w_f = cfg.balance_w_freq if cfg.balance_w_freq is not None else target
        network._balance_w_content = float(w_c)
        # Hardwired-FEI freq gate has no router params (fixed function of z_t),
        # so a balance penalty on it is constant w.r.t. trained params (zero
        # gradient) — force w_f=0 to keep the loss scalar honest.
        if str(getattr(cfg, "freq_router_mode", "learned")).lower() == "fei":
            network._balance_w_freq = 0.0
        else:
            network._balance_w_freq = float(w_f)
    else:
        network._use_chimera_hydra = False


NETWORK_REGISTRY: Dict[str, NetworkSpec] = {
    "lora": NetworkSpec(
        name="lora",
        module_class=LoRAModule,
        save_variant="standard",
    ),
    "ortho": NetworkSpec(
        name="ortho",
        module_class=OrthoLoRAModule,
        save_variant="ortho_to_lora",
    ),
    # OrthoInit: trainable top-r SVD init of W0 (no Cayley/frozen subspace).
    # Distills to standard LoRA at save (sqrt-split λ → down/up), so the on-disk
    # form matches a distilled OrthoLoRA.
    "ortho_init": NetworkSpec(
        name="ortho_init",
        module_class=OrthoInitLoRAModule,
        save_variant="ortho_to_lora",
    ),
    "hydra": NetworkSpec(
        name="hydra",
        module_class=HydraLoRAModule,
        save_variant="hydra_moe",
        post_init=_post_init_hydra,
    ),
    "ortho_hydra": NetworkSpec(
        name="ortho_hydra",
        module_class=OrthoHydraLoRAModule,
        save_variant="ortho_hydra_to_hydra",
        post_init=_post_init_hydra,
    ),
    # ChimeraHydra: dual-pool additive routing on the OrthoHydra Cayley
    # parameterization (docs/proposal/chimera_hydra.md). Save distills the Cayley
    # params to the Hydra-MoE layout in a ``*_chimera.safetensors`` sibling; load
    # goes through ``HydraLoRAModule`` with ``num_experts_content > 0``, so the
    # Cayley class is training-only (resume loses the orthogonal parameterization,
    # matching the OrthoHydra → Hydra trade-off).
    "chimera_hydra": NetworkSpec(
        name="chimera_hydra",
        module_class=ChimeraHydraLoRAModule,
        save_variant="chimera_hydra_moe",
        post_init=_post_init_hydra,
    ),
    # Step-expert: shared down-proj + K step-indexed up-heads, hard-selected by
    # diffusion step (no router). Turbo DP-DMD student only; kept-live at
    # inference (K heads can't fold into one DiT weight), so save is bespoke
    # (TurboDMDNetwork.save_student, not save_network_weights). Selected via step_expert_K.
    "step_expert": NetworkSpec(
        name="step_expert",
        module_class=StepExpertLoRAModule,
        save_variant="step_expert",
    ),
    # FeRA paper-faithful: independent-A stacked experts, single network-level
    # router fed by FEI(z_t). Selected via ``use_moe_style="independent_A"``.
    "stacked_experts_global_fei": NetworkSpec(
        name="stacked_experts_global_fei",
        module_class=StackedExpertsLoRAModule,
        save_variant="stacked_experts_global_fei",
        post_init=_post_init_hydra,
    ),
}


def all_network_kwargs() -> Tuple[str, ...]:
    """Return the LoRA-family TOML allowlist (``NETWORK_KWARGS``), sorted.

    Single source of truth for train.py — populates the argparse schema and
    the TOML → ``net_kwargs`` forwarding list, so adding a key to
    ``NETWORK_KWARGS`` automatically makes it visible to training without
    touching train.py.
    """
    return tuple(sorted(NETWORK_KWARGS))


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

    Honors the ``use_moe_style`` axis (plan2.md §three-axis-config):
    ``"independent_A"`` routes to ``stacked_experts_global_fei`` (FeRA);
    ``"shared_A"`` plus ``use_ortho`` routes to ``ortho_hydra``; bare
    ``"shared_A"`` routes to ``hydra``. The legacy ``use_hydra`` kwarg was
    retired in plan2 task #6 — ``LoRANetworkCfg.from_kwargs`` raises if a
    TOML still carries it.

    ``use_chimera_hydra=True`` short-circuits to the chimera variant. The
    chimera config requires ``use_moe_style="shared_A"`` semantics under
    the hood (OrthoHydra parameterization), but uses K_c + K_f instead of
    a single ``num_experts`` — the user only sets the chimera flag.
    """
    use_ortho = _parse_bool_flag(kwargs, "use_ortho")
    use_ortho_init = _parse_bool_flag(kwargs, "use_ortho_init")
    use_chimera = _parse_bool_flag(kwargs, "use_chimera_hydra")
    if use_ortho and use_ortho_init:
        raise ValueError(
            "use_ortho and use_ortho_init are mutually exclusive: ortho freezes "
            "the SVD basis (Cayley-rotates within it); ortho_init trains the SVD "
            "basis (no cap). Pick one."
        )
    if use_chimera:
        # OrthoInit composes with chimera (use_ortho_init swaps each pool's
        # frozen Cayley basis for trainable SVD-seeded bases via cfg.use_ortho_init);
        # same spec either way — distills to the same *_chimera.safetensors layout.
        return NETWORK_REGISTRY["chimera_hydra"]

    # Step-expert short-circuits when step_expert_K > 1; K==1 collapses to plain LoRA.
    raw_step_K = kwargs.get("step_expert_K")
    if raw_step_K is not None and int(raw_step_K) > 1:
        return NETWORK_REGISTRY["step_expert"]

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

    if use_ortho_init and moe_style:
        raise NotImplementedError(
            "use_ortho_init does not yet compose with use_moe_style — the "
            "orthoinit MoE pool is a separate family member (not implemented)."
        )
    if moe_style == "independent_A":
        return NETWORK_REGISTRY["stacked_experts_global_fei"]
    if moe_style == "shared_A":
        return (
            NETWORK_REGISTRY["ortho_hydra"] if use_ortho else NETWORK_REGISTRY["hydra"]
        )
    if use_ortho_init:
        return NETWORK_REGISTRY["ortho_init"]
    if use_ortho:
        return NETWORK_REGISTRY["ortho"]
    return NETWORK_REGISTRY["lora"]


__all__ = [
    "NetworkSpec",
    "NETWORK_REGISTRY",
    "NETWORK_KWARGS",
    "all_network_kwargs",
    "resolve_network_spec",
]
