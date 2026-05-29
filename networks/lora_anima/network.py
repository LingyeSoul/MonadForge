# LoRANetwork: the module-assembly / training-orchestration core of the LoRA
# adapter stack for Anima. Targets DiT blocks (and optionally text-encoder
# attention) with pluggable per-module classes supplied by a NetworkSpec.

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import torch

from library.log import setup_logging
from networks import NETWORK_REGISTRY, NetworkSpec, lora_save
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.loading import (
    _parse_reft_layers,
    _refuse_split_hydra_keys,
    _refuse_split_stacked_experts_keys,
    _refuse_unfused_attn_lora_keys,
    _stack_lora_ups,
)
from networks.lora_modules import (
    ChimeraHydraInferenceModule,
    ChimeraHydraLoRAModule,
    HydraLoRAModule,
    LoRAModule,
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
    ReFTModule,
    StackedExpertsLoRAModule,
    _sigma_sinusoidal_features,
)
from networks.lora_modules.router_state import _fei_temperature
from networks.lora_anima.network_metrics import _NetworkMetricsMixin

# Routers live in routers.py; re-exported here so existing imports
# (``from networks.lora_anima.network import GlobalRouter`` / ``FreqRouter`` /
# ``ContentRouter`` / ``CROSSATTN_EMB_DIM``) keep resolving.
from networks.lora_anima.routers import (  # noqa: F401
    CROSSATTN_EMB_DIM,
    ContentRouter,
    FreqRouter,
    GlobalRouter,
)

setup_logging()
logger = logging.getLogger(__name__)

_BLOCK_IDX_RE = re.compile(r"blocks\.(\d+)\.")


class LoRANetwork(_NetworkMetricsMixin, torch.nn.Module):
    # Target modules: DiT blocks, embedders, final layer. embedders and final layer are excluded by default.
    ANIMA_TARGET_REPLACE_MODULE = [
        "Block",
        "PatchEmbed",
        "TimestepEmbedding",
        "FinalLayer",
    ]
    # Target modules: LLM Adapter blocks
    ANIMA_ADAPTER_TARGET_REPLACE_MODULE = ["LLMAdapterTransformerBlock"]
    # Target modules for text encoder (Qwen3)
    TEXT_ENCODER_TARGET_REPLACE_MODULE = [
        "Qwen3Attention",
        "Qwen3MLP",
        "Qwen3SdpaAttention",
        "Qwen3FlashAttention2",
    ]

    LORA_PREFIX_ANIMA = "lora_unet"  # ComfyUI compatible
    LORA_PREFIX_TEXT_ENCODER = "lora_te"  # Qwen3

    def __init__(
        self,
        text_encoders: list,
        unet,
        cfg: LoRANetworkCfg,
        *,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Mutable runtime state — explicitly NOT in cfg. ``set_multiplier`` and
        # ``set_loraplus_lr_ratio`` write these post-construction; per-step
        # diagnostics (hit counters, σ caches) accumulate during training.
        self.multiplier = multiplier
        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None
        self._channel_scale_misses: List[str] = []
        self._channel_scale_hits: int = 0
        self._sigma_router_hits: int = 0
        self._hydra_router_hits: int = 0
        self._hydra_router_misses: int = 0
        self._last_sigma: Optional[torch.Tensor] = None
        # Hydra up-weight grad-norm snapshot (T-LoRA / σ-bucket conflict
        # diagnostic). Filled by ``capture_up_grad_stats`` between backward
        # and ``optimizer.zero_grad``; consumed by the ``hydra_up_grad``
        # metric. Values stay on-device until ``get_up_grad_stats`` runs the
        # D2H — capture happens every sync step but the metric only reads on
        # log steps, so the sync was the per-step bottleneck.
        self._last_up_grad_stats: Dict[str, object] = {}
        # Per-step cache for ``get_router_stats`` — both the progress-bar
        # postfix and the metrics layer call it on log steps. Cleared in
        # ``clear_step_caches`` so the next forward recomputes.
        self._router_stats_cache: Optional[Dict[str, object]] = None
        # Separate cache for the chimera dual-pool router stats — different
        # reduction (mean gates per pool, not argmax-histogram) and different
        # entropy normalization (per-pool log(K_pool)). Same lifecycle.
        self._chimera_router_stats_cache: Optional[Dict[str, object]] = None

        # Local aliases read by the closure body and the post-closure ReFT block.
        module_class = cfg.module_class
        modules_dim = cfg.modules_dim
        modules_alpha = cfg.modules_alpha
        dropout = cfg.dropout
        rank_dropout = cfg.rank_dropout
        module_dropout = cfg.module_dropout
        verbose = cfg.verbose
        alpha = cfg.alpha
        lora_dim = cfg.lora_dim
        train_llm_adapter = cfg.train_llm_adapter
        add_reft = cfg.add_reft
        reft_dim = cfg.reft_dim
        reft_alpha = cfg.reft_alpha
        reft_layers = cfg.reft_layers

        # Unified routing scope. ``cfg.router_targets`` is the single regex
        # that governs which Linears participate in routed adaptation (Hydra
        # MoE leaves + σ-feature concat + FEI-feature concat all share it).
        # From-weights path supplies an explicit name set per router family
        # (different families may have different module memberships in older
        # checkpoints); when present, the explicit set wins over the regex.
        _router_re = re.compile(cfg.router_targets) if cfg.router_targets else None

        self._sigma_router_names = (
            set(cfg.sigma_router_names) if cfg.sigma_router_names else None
        )
        self._sigma_router_re = (
            _router_re
            if (
                cfg.router_source == "sigma"
                and _router_re is not None
                and self._sigma_router_names is None
            )
            else None
        )

        self._fei_router_names = (
            set(cfg.fei_router_names) if cfg.fei_router_names else None
        )
        self._fei_router_re = (
            _router_re
            if (
                cfg.router_source == "fei"
                and _router_re is not None
                and self._fei_router_names is None
            )
            else None
        )
        self._fei_router_hits = 0
        # Modules built with ``use_global_router=True`` (shared_A +
        # ``route_per_layer=False``): the per-layer router is skipped and gates
        # arrive via the network-level ``GlobalRouter``. Counted separately
        # from ``_fei_router_hits`` because the per-layer FEI cat is bypassed.
        self._global_router_hits = 0
        # Retained as a network attr (library/inference/adapters.py reads it
        # via getattr); derived from cfg.router_source.
        self.use_fei_router = cfg.router_source == "fei"
        self.use_sigma_router = cfg.router_source == "sigma"
        # Shared-A Hydra layout + network-level router (FEI-on-Hydra global).
        # Toggle for the per-module construction loop below; lets Hydra /
        # OrthoHydra modules skip ``self.router`` and consume gates from the
        # ``GlobalRouter`` instead. Mirrors the FeRA (independent_A) routing
        # location without changing the underlying Hydra parameter layout.
        self._use_global_router_for_hydra = (
            cfg.use_moe_style == "shared_A"
            and not cfg.route_per_layer
            and cfg.router_source != "none"
        )

        # Per-module HydraLoRA gating. Matching modules get the Hydra class;
        # non-matching modules fall back to plain LoRA / OrthoLoRA so MoE
        # capacity is concentrated where specialization is actually learnable.
        # Fresh path: regex over `original_name`. From-weights path: explicit
        # name set detected from checkpoint keys. Explicit set wins. None on
        # both = apply MoE everywhere (legacy).
        self._hydra_router_names = (
            set(cfg.hydra_router_names) if cfg.hydra_router_names else None
        )
        self._hydra_router_re = (
            _router_re
            if (_router_re is not None and self._hydra_router_names is None)
            else None
        )

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        else:
            logger.info(
                f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}"
            )
            logger.info(
                f"neuron dropout: p={dropout}, rank dropout: p={rank_dropout}, module dropout: p={module_dropout}"
            )

        def str_to_re_patterns(patterns: Optional[List[str]]) -> List[re.Pattern]:
            re_patterns = []
            if patterns is not None:
                for pattern in patterns:
                    try:
                        re_pattern = re.compile(pattern)
                    except re.error as e:
                        logger.error(f"Invalid pattern '{pattern}': {e}")
                        continue
                    re_patterns.append(re_pattern)
            return re_patterns

        exclude_re_patterns = str_to_re_patterns(cfg.exclude_patterns)
        include_re_patterns = str_to_re_patterns(cfg.include_patterns)

        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
            default_dim: Optional[int] = None,
        ) -> Tuple[List[LoRAModule], List[str]]:
            prefix = (
                self.LORA_PREFIX_ANIMA if is_unet else self.LORA_PREFIX_TEXT_ENCODER
            )

            # First pass: collect candidate modules
            candidates = []
            for name, module in root_module.named_modules():
                if (
                    target_replace_modules is None
                    or module.__class__.__name__ in target_replace_modules
                ):
                    if target_replace_modules is None:
                        module = root_module

                    for child_name, child_module in module.named_modules():
                        is_linear = isinstance(child_module, torch.nn.Linear)
                        is_conv2d = isinstance(child_module, torch.nn.Conv2d)
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            original_name = (name + "." if name else "") + child_name
                            # Strip torch.compile wrapper from module path
                            original_name = original_name.replace("_orig_mod.", "")
                            lora_name = f"{prefix}.{original_name}".replace(".", "_")

                            # exclude/include filter
                            excluded = any(
                                pattern.fullmatch(original_name)
                                for pattern in exclude_re_patterns
                            )
                            included = any(
                                pattern.fullmatch(original_name)
                                for pattern in include_re_patterns
                            )
                            if excluded and not included:
                                if verbose:
                                    logger.info(f"exclude: {original_name}")
                                continue

                            # layer range filter: skip blocks outside [layer_start, layer_end)
                            if is_unet and (
                                cfg.layer_start is not None or cfg.layer_end is not None
                            ):
                                block_match = _BLOCK_IDX_RE.match(original_name)
                                if block_match:
                                    block_idx = int(block_match.group(1))
                                    if (
                                        cfg.layer_start is not None
                                        and block_idx < cfg.layer_start
                                    ):
                                        if verbose:
                                            logger.info(
                                                f"layer_range exclude: {original_name} (block {block_idx} < {cfg.layer_start})"
                                            )
                                        continue
                                    if (
                                        cfg.layer_end is not None
                                        and block_idx >= cfg.layer_end
                                    ):
                                        if verbose:
                                            logger.info(
                                                f"layer_range exclude: {original_name} (block {block_idx} >= {cfg.layer_end})"
                                            )
                                        continue

                            dim = None
                            alpha_val = None

                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha_val = modules_alpha[lora_name]
                            else:
                                if cfg.reg_dims is not None:
                                    for reg, d in cfg.reg_dims.items():
                                        if re.fullmatch(reg, original_name):
                                            dim = d
                                            alpha_val = alpha
                                            logger.info(
                                                f"Module {original_name} matched with regex '{reg}' -> dim: {dim}"
                                            )
                                            break
                                if dim is None:
                                    if is_linear or is_conv2d_1x1:
                                        dim = (
                                            default_dim
                                            if default_dim is not None
                                            else lora_dim
                                        )
                                        alpha_val = alpha

                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1:
                                    candidates.append(
                                        (
                                            lora_name,
                                            None,
                                            None,
                                            None,
                                            original_name,
                                            True,
                                        )
                                    )  # skipped
                                continue

                            candidates.append(
                                (
                                    lora_name,
                                    child_module,
                                    dim,
                                    alpha_val,
                                    original_name,
                                    False,
                                )
                            )

                    if target_replace_modules is None:
                        break

            # Second pass: create LoRA modules with progress bar
            from tqdm import tqdm

            loras = []
            skipped = []
            non_skipped = [
                (ln, cm, d, a, on) for ln, cm, d, a, on, skip in candidates if not skip
            ]
            skipped = [ln for ln, cm, d, a, on, skip in candidates if skip]

            label = (
                "DiT"
                if is_unet
                else f"TE{text_encoder_idx + 1}"
                if text_encoder_idx is not None
                else "model"
            )
            for lora_name, child_module, dim, alpha_val, original_name in tqdm(
                non_skipped, desc=f"Creating {label} LoRA", leave=False
            ):
                # Per-module class resolution: when the network's nominal class
                # is Hydra (MoE), narrow it to only the layers in the hydra
                # filter. Non-matching layers fall back to plain LoRA /
                # OrthoLoRA so router overhead + balance-loss pressure are
                # concentrated on sites where specialization is learnable.
                effective_module_class = module_class
                if (
                    module_class
                    in (
                        HydraLoRAModule,
                        OrthoHydraLoRAModule,
                        ChimeraHydraLoRAModule,
                        ChimeraHydraInferenceModule,
                    )
                    and is_unet
                ):
                    if self._hydra_router_names is not None:
                        hydra_on = lora_name in self._hydra_router_names
                    elif self._hydra_router_re is not None:
                        hydra_on = bool(self._hydra_router_re.search(original_name))
                    else:
                        hydra_on = True
                    if hydra_on:
                        self._hydra_router_hits += 1
                    else:
                        self._hydra_router_misses += 1
                        if module_class is HydraLoRAModule:
                            effective_module_class = LoRAModule
                        elif module_class is ChimeraHydraInferenceModule:
                            # Load path. Unrouted leg was saved as plain LoRA
                            # (OrthoLoRA distilled to ``.lora_down.weight`` +
                            # ``.lora_up.weight`` at save time — see
                            # ``_convert_ortho_to_lora``).
                            effective_module_class = LoRAModule
                        else:
                            # Train path (ChimeraHydraLoRAModule) and
                            # OrthoHydra: unrouted leg uses the OrthoLoRA
                            # Cayley parameterization.
                            effective_module_class = OrthoLoRAModule

                extra_kwargs = {}
                if effective_module_class == OrthoLoRAModule:
                    pass  # no extra kwargs — SVD init reads from org_module directly
                elif effective_module_class == ChimeraHydraLoRAModule:
                    # Pool split is the chimera's only constructor surface;
                    # σ/FEI feature dims are 0 by design (the network-level
                    # FreqRouter owns those axes — see chimera.py module
                    # docstring). The pool sum must equal cfg.num_experts
                    # by ``LoRANetworkCfg.from_kwargs`` invariant.
                    extra_kwargs["num_experts_content"] = cfg.num_experts_content
                    extra_kwargs["num_experts_freq"] = cfg.num_experts_freq
                    extra_kwargs["lambda_init"] = cfg.chimera_lambda_init
                elif effective_module_class == ChimeraHydraInferenceModule:
                    # Inference (free-form) twin of the chimera training
                    # class. Same constructor surface — both pool sizes
                    # arrive from the chimera-stamped metadata via
                    # ``cfg.from_weights``.
                    extra_kwargs["num_experts_content"] = cfg.num_experts_content
                    extra_kwargs["num_experts_freq"] = cfg.num_experts_freq
                elif effective_module_class == OrthoHydraLoRAModule:
                    extra_kwargs["num_experts"] = cfg.num_experts
                    extra_kwargs["centered_gate"] = cfg.ortho_centered_gate
                    extra_kwargs["lambda_init"] = cfg.ortho_lambda_init
                    if self._use_global_router_for_hydra:
                        extra_kwargs["use_global_router"] = True
                        self._global_router_hits += 1
                elif effective_module_class == HydraLoRAModule:
                    extra_kwargs["num_experts"] = cfg.num_experts
                    # Runtime parity for OrthoHydra checkpoints distilled with
                    # ortho_centered_gate (inert for plain Hydra / chimera —
                    # the module gates it on single-pool num_experts_content==0).
                    extra_kwargs["centered_gate"] = cfg.ortho_centered_gate
                    if cfg.expert_init_std > 0.0:
                        extra_kwargs["expert_init_std"] = cfg.expert_init_std
                    if self._use_global_router_for_hydra:
                        extra_kwargs["use_global_router"] = True
                        self._global_router_hits += 1
                    if cfg.use_chimera_hydra:
                        # Dual-pool runtime form (load path from a distilled
                        # chimera checkpoint — see factory.py is_chimera_hydra
                        # branch). HydraLoRAModule narrows its router to K_c
                        # outputs and registers _freq_routing_weights for the
                        # network-level FreqRouter broadcast. σ/FEI feature
                        # dims must stay 0 here — FreqRouter owns those axes.
                        # Chimera content routing is always the network-level
                        # ContentRouter (crossattn_emb).
                        extra_kwargs["num_experts_content"] = cfg.num_experts_content
                        extra_kwargs["use_global_content_router"] = True
                elif effective_module_class == StackedExpertsLoRAModule:
                    # Independent-A (FeRA). Gates arrive via the network-level
                    # ``GlobalRouter`` through the shared ``_routing_weights``
                    # buffer — no per-Linear router knob to set. ``num_experts``
                    # must match ``cfg.num_experts`` (and therefore the
                    # GlobalRouter's output width) or the routing-weight
                    # broadcast inside ``forward`` shape-mismatches.
                    extra_kwargs["num_experts"] = cfg.num_experts
                    extra_kwargs["ortho"] = cfg.use_ortho
                    if cfg.use_ortho:
                        extra_kwargs["ortho_init_std"] = cfg.ortho_init_std

                # Hard σ-band expert partition: applied to every Hydra/
                # OrthoHydra module (independent of the σ-feature router
                # regex). Each module owns the partition; the network-level
                # ``set_sigma`` propagates ``_sigma`` to enable per-step band
                # selection. Validation (E % N == 0) lives in cfg parsing.
                if (
                    cfg.specialize_experts_by_sigma_buckets
                    and effective_module_class
                    in (HydraLoRAModule, OrthoHydraLoRAModule)
                    and is_unet
                ):
                    extra_kwargs["specialize_experts_by_sigma_buckets"] = True
                    extra_kwargs["num_sigma_buckets"] = cfg.num_sigma_buckets
                    if cfg.sigma_bucket_boundaries is not None:
                        extra_kwargs["sigma_bucket_boundaries"] = (
                            cfg.sigma_bucket_boundaries
                        )

                # σ-conditional router: only widen the router input with
                # sinusoidal(σ) features on modules whose name matches the
                # layer filter (cross_attn.q / self_attn.qkv by default — see
                # B0 pre-analysis in timestep-hydra.md). From-weights path uses
                # an explicit name set; fresh-from-kwargs path uses a regex
                # over original_name. Gated on the effective class so a
                # hydra-excluded module can't pick up σ either. Skipped under
                # ``use_global_router`` — the network-level router consumes
                # the routing signal once and the per-Linear cat is dead.
                if (
                    cfg.router_source == "sigma"
                    and effective_module_class
                    in (
                        HydraLoRAModule,
                        OrthoHydraLoRAModule,
                    )
                    and is_unet
                    and not self._use_global_router_for_hydra
                ):
                    if self._sigma_router_names is not None:
                        enable = lora_name in self._sigma_router_names
                    elif self._sigma_router_re is not None:
                        enable = bool(self._sigma_router_re.search(original_name))
                    else:
                        enable = True
                    if enable:
                        extra_kwargs["sigma_feature_dim"] = cfg.sigma_feature_dim
                        self._sigma_router_hits += 1

                # FEI-conditional router (FeRA-style). Same gating as σ —
                # widen the router input with the per-sample FEI simplex on
                # modules whose name matches the layer filter. The FEI tensor
                # itself is computed once per step in the train/inference loop
                # and propagated via ``LoRANetwork.set_fei``. Skipped under
                # ``use_global_router`` — the GlobalRouter reads FEI directly
                # at the network level and per-Linear cat is dead.
                if (
                    cfg.router_source == "fei"
                    and effective_module_class
                    in (
                        HydraLoRAModule,
                        OrthoHydraLoRAModule,
                    )
                    and is_unet
                    and not self._use_global_router_for_hydra
                ):
                    if self._fei_router_names is not None:
                        enable_fei = lora_name in self._fei_router_names
                    elif self._fei_router_re is not None:
                        enable_fei = bool(self._fei_router_re.search(original_name))
                    else:
                        enable_fei = True
                    if enable_fei:
                        extra_kwargs["fei_feature_dim"] = cfg.fei_feature_dim
                        self._fei_router_hits += 1

                # Per-channel scaling is DiT-only: the bench script hooks DiT
                # linears, text encoder activations are never calibrated.
                if cfg.channel_scales_dict is not None and is_unet:
                    _cs = cfg.channel_scales_dict.get(lora_name)
                    if _cs is not None:
                        extra_kwargs["channel_scale"] = _cs
                        self._channel_scale_hits += 1
                    else:
                        self._channel_scale_misses.append(lora_name)

                lora = effective_module_class(
                    lora_name,
                    child_module,
                    self.multiplier,
                    dim,
                    alpha_val,
                    dropout=dropout,
                    rank_dropout=rank_dropout,
                    module_dropout=module_dropout,
                    **extra_kwargs,
                )
                lora.original_name = original_name
                loras.append(lora)

            return loras, skipped

        # Create LoRA for text encoders (Qwen3 - typically not trained for Anima)
        # Skip for OrthoLoRA since SVD init is expensive and TE modules are discarded in apply_to anyway
        self.text_encoder_loras: List[LoRAModule] = []
        skipped_te = []
        if text_encoders is not None and module_class not in (
            OrthoLoRAModule,
            OrthoHydraLoRAModule,
            ChimeraHydraLoRAModule,
            ChimeraHydraInferenceModule,
        ):
            for i, text_encoder in enumerate(text_encoders):
                if text_encoder is None:
                    continue
                logger.info(f"create LoRA for Text Encoder {i + 1}:")
                te_loras, te_skipped = create_modules(
                    False,
                    i,
                    text_encoder,
                    LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE,
                )
                logger.info(
                    f"create LoRA for Text Encoder {i + 1}: {len(te_loras)} modules."
                )
                self.text_encoder_loras.extend(te_loras)
                skipped_te += te_skipped

        # Create LoRA for DiT blocks
        target_modules = list(LoRANetwork.ANIMA_TARGET_REPLACE_MODULE)
        if train_llm_adapter:
            target_modules.extend(LoRANetwork.ANIMA_ADAPTER_TARGET_REPLACE_MODULE)

        self.unet_loras: List[LoRAModule]
        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)

        logger.info(f"create LoRA for Anima DiT: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_te + skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(f"dim (rank) is 0, {len(skipped)} LoRA modules are skipped:")
            for name in skipped:
                logger.info(f"\t{name}")

        if cfg.channel_scales_dict is not None:
            logger.info(
                f"channel_scaling: {self._channel_scale_hits} DiT modules "
                f"received calibration-based input scaling"
            )
            if self._channel_scale_misses:
                logger.warning(
                    f"channel_scaling: {len(self._channel_scale_misses)} DiT modules "
                    f"have no calibration stats (first: {self._channel_scale_misses[:3]}). "
                    f"These will train without input rebalancing — regenerate the vendored "
                    f"calibration with `python bench/channel_stats/analyze_lora_input_channels.py "
                    f"--per_artist --dump_channel_stats networks/calibration/channel_stats.safetensors` "
                    f"if this is unexpected."
                )

        # Create ReFT modules on the DiT residual stream (block outputs), following
        # Wu et al. (2024) §3.3 — one intervention per selected block, not per
        # internal Linear. Selection is controlled by ``reft_layers``.
        self.unet_refts: List[ReFTModule] = []
        self.text_encoder_refts: List[ReFTModule] = []
        if add_reft:
            dit_blocks = getattr(unet, "blocks", None)
            if dit_blocks is None or len(dit_blocks) == 0:
                raise ValueError(
                    "add_reft=True but DiT has no .blocks attribute to wrap. "
                    "Block-level ReFT requires a transformer with a `blocks` ModuleList."
                )
            num_blocks = len(dit_blocks)
            selected_indices = _parse_reft_layers(reft_layers, num_blocks)

            reft_alpha_value = reft_alpha if reft_alpha is not None else alpha
            for idx in selected_indices:
                block = dit_blocks[idx]
                block_embed_dim = getattr(block, "x_dim", None)
                if block_embed_dim is None:
                    raise ValueError(
                        f"Block {idx} ({type(block).__name__}) has no `x_dim`; "
                        "cannot infer embed_dim for ReFT."
                    )
                reft_name = f"reft_unet_blocks_{idx}"
                reft = ReFTModule(
                    reft_name,
                    block,
                    embed_dim=block_embed_dim,
                    multiplier=multiplier,
                    reft_dim=reft_dim,
                    alpha=reft_alpha_value,
                    dropout=dropout,
                    module_dropout=module_dropout,
                )
                reft.original_name = f"blocks.{idx}"
                self.unet_refts.append(reft)
            logger.info(
                f"create ReFT for Anima DiT: {len(self.unet_refts)}/{num_blocks} "
                f"blocks (reft_dim={reft_dim}, layers={reft_layers!r})"
            )

        # assertion: no duplicate names
        names = set()
        for lora in (
            self.text_encoder_loras
            + self.unet_loras
            + self.text_encoder_refts
            + self.unet_refts
        ):
            assert lora.lora_name not in names, (
                f"duplicated lora name: {lora.lora_name}"
            )
            names.add(lora.lora_name)

        # Alias each sigma-aware module's ``_sigma`` / ``_sigma_features``
        # buffer to a single network-level shared tensor. ``set_sigma`` then
        # updates the shared tensor in place once and every aliased module
        # buffer sees the new value through shared storage — instead of
        # ~56 per-module ``copy_`` calls per training step.
        self._wire_shared_sigma_buffers()
        self._wire_shared_fei_buffers()
        self._wire_shared_routing_buffers()
        self._wire_shared_freq_routing_buffers()
        self._wire_shared_content_routing_buffers()

        # Build the network-level GlobalRouter when the cfg selects MoE
        # without per-Linear routers. The input dim is derived from the
        # routing signal: ``"fei"`` → ``fei_feature_dim`` simplex,
        # ``"sigma"`` → ``sigma_feature_dim`` sinusoidal features.
        # Routing-aware modules: ``independent_A`` (StackedExperts) always
        # consume the broadcast gates; ``shared_A`` (Hydra / OrthoHydra)
        # consumes them when built with ``use_global_router=True``.
        self.global_router: Optional[GlobalRouter] = None
        # ``use_crossattn_router`` advertises to the train / inference call
        # sites that they must fire ``set_crossattn_routing`` with the pooled
        # text tensor each forward (parallel to chimera's ``use_content_router``
        # but broadcasting to the standard ``_routing_weights`` slot).
        self.use_crossattn_router: bool = False
        if cfg.use_moe_style is not False and not cfg.route_per_layer:
            router_layer_norm = False
            if cfg.router_source == "fei":
                router_input_dim = int(cfg.fei_feature_dim)
            elif cfg.router_source == "sigma":
                router_input_dim = int(cfg.sigma_feature_dim)
            elif cfg.router_source == "crossattn_emb":
                # Pooled post-LLM-adapter text feature (the DiT's cross-attn
                # K/V). LN on by default — wide T5-space variance budget.
                router_input_dim = CROSSATTN_EMB_DIM
                router_layer_norm = True
            else:
                router_input_dim = 0
            if router_input_dim > 0 and cfg.num_experts > 1:
                self.global_router = GlobalRouter(
                    input_dim=router_input_dim,
                    num_experts=int(cfg.num_experts),
                    hidden_dim=int(cfg.router_hidden_dim),
                    tau=float(cfg.router_tau),
                    apply_layer_norm=router_layer_norm,
                )
                self.use_crossattn_router = cfg.router_source == "crossattn_emb"
                logger.info(
                    f"GlobalRouter: source={cfg.router_source!r}, "
                    f"input_dim={router_input_dim}, "
                    f"num_experts={cfg.num_experts}, "
                    f"hidden={cfg.router_hidden_dim}, τ={cfg.router_tau:.2f}, "
                    f"LN={router_layer_norm}, "
                    f"routing-aware modules={len(self._routing_aware_loras)}"
                )

        # ChimeraHydra FreqRouter: one per network, broadcasts ``π_f`` over
        # the freq pool of every chimera module. Input is
        # ``concat(FEI, sinusoidal-σ-features)`` — owned by the freq router
        # exclusively (the per-layer content router never sees σ/FEI). Built
        # only when at least one chimera module was actually constructed; the
        # router_targets regex can narrow the chimera class to a subset of
        # layers (others fall back to OrthoLoRA).
        self.freq_router: Optional[FreqRouter] = None
        # Freq-pool routing mode. "learned" builds the FreqRouter MLP below;
        # "fei" leaves freq_router=None and broadcasts the FEI simplex directly
        # in set_fei (no params, no σ-features, no freq balance loss).
        self.freq_router_mode: str = str(
            getattr(cfg, "freq_router_mode", "learned")
        ).lower()
        self.freq_router_tau: float = float(getattr(cfg, "freq_router_tau", 1.0))
        if cfg.use_chimera_hydra and self._chimera_aware_loras:
            if self.freq_router_mode == "fei":
                # Hardwired-FEI gate: π_f = normalize(FEI ** (1/τ)). The FEI
                # band-simplex IS the routing distribution, so K_f must equal
                # the band count (validated in from_kwargs; re-assert here for
                # the warm-start / from_weights path that bypasses from_kwargs).
                if int(cfg.num_experts_freq) != int(cfg.fei_feature_dim):
                    raise ValueError(
                        "freq_router_mode='fei' requires num_experts_freq == "
                        f"fei_feature_dim (got K_f={cfg.num_experts_freq}, "
                        f"fei_feature_dim={cfg.fei_feature_dim})."
                    )
                # Still fire set_fei every step — that's where the FEI simplex
                # is broadcast to each chimera module's _freq_routing_weights.
                self.use_fei_router = True
                logger.info(
                    "ChimeraHydra freq pool: HARDWIRED FEI gate "
                    f"(K_f={cfg.num_experts_freq} = fei bands, τ={self.freq_router_tau:.2f}, "
                    "no learned router / no σ-features / no freq balance loss), "
                    f"chimera modules={len(self._chimera_aware_loras)}"
                )
            else:
                freq_input_dim = int(cfg.fei_feature_dim) + int(cfg.sigma_feature_dim)
                if freq_input_dim <= 0:
                    raise ValueError(
                        "use_chimera_hydra=True requires fei_feature_dim + "
                        f"sigma_feature_dim > 0 for the FreqRouter input (got "
                        f"FEI={cfg.fei_feature_dim}, σ={cfg.sigma_feature_dim})."
                    )
                # Chimera is always centered-gate: the freq pool's cold-start
                # is broken by the disjoint P_bases_f·λ_f residual (not router
                # noise), so zero-init the router for an exactly-uniform π_f at
                # step 0 → ΔW_f=0. (The "zero-init is a fixed point" warning in
                # FreqRouter's docstring only applies to the non-centered
                # additive composition, which chimera no longer supports.)
                freq_init_std = 0.0
                self.freq_router = FreqRouter(
                    input_dim=freq_input_dim,
                    num_freq_experts=int(cfg.num_experts_freq),
                    hidden_dim=int(cfg.router_hidden_dim),
                    tau=float(cfg.router_tau),
                    init_std=freq_init_std,
                    fei_dim=int(cfg.fei_feature_dim),
                    sigma_dim=int(cfg.sigma_feature_dim),
                    apply_layer_norm=bool(cfg.freq_router_layer_norm),
                )
                # Force the per-step conditioning hook to fire set_fei every
                # step (router_conditioning.py reads this flag). Chimera ties
                # σ + FEI together for the freq router input, so the set_fei
                # path is where we re-fire FreqRouter.
                self.use_fei_router = True
                logger.info(
                    f"ChimeraHydra FreqRouter: input_dim={freq_input_dim} "
                    f"(FEI={cfg.fei_feature_dim} + σ={cfg.sigma_feature_dim}), "
                    f"K_f={cfg.num_experts_freq}, hidden={cfg.router_hidden_dim}, "
                    f"τ={cfg.router_tau:.2f}, init_std={cfg.freq_router_init_std}, "
                    f"LN={self.freq_router.apply_layer_norm}, "
                    f"chimera modules={len(self._chimera_aware_loras)}"
                )

        # ChimeraHydra ContentRouter: network-level twin of FreqRouter for
        # the content pool, fed the pooled ``crossattn_emb``. Built whenever a
        # chimera module exists — content routing is always network-level (the
        # per-Linear router was removed). π_c flows through the broadcast
        # ``_content_routing_weights`` slot. ``use_content_router=True``
        # advertises to the train / inference call sites that they must thread
        # ``crossattn_emb`` through ``set_content``.
        self.content_router: Optional[ContentRouter] = None
        self.use_content_router: bool = False
        if cfg.use_chimera_hydra and self._chimera_aware_loras:
            # Always centered-gate zero-init (same as the FreqRouter above) —
            # the content pool's disjoint P_bases_c·λ_c residual breaks
            # symmetry, so a uniform π_c at step 0 keeps ΔW_c=0.
            self.content_router = ContentRouter(
                input_dim=CROSSATTN_EMB_DIM,
                num_content_experts=int(cfg.num_experts_content),
                hidden_dim=int(cfg.router_hidden_dim),
                tau=float(cfg.router_tau),
                init_std=0.0,
                apply_layer_norm=bool(cfg.content_router_layer_norm),
            )
            self.use_content_router = True
            logger.info(
                f"ChimeraHydra ContentRouter: input_dim={CROSSATTN_EMB_DIM} "
                f"(pooled crossattn_emb), K_c={cfg.num_experts_content}, "
                f"hidden={cfg.router_hidden_dim}, τ={cfg.router_tau:.2f}, "
                f"LN={cfg.content_router_layer_norm}, "
                f"chimera modules={len(self._chimera_aware_loras)}"
            )

    def _wire_shared_sigma_buffers(self) -> None:
        """Replace each HydraLoRA / OrthoHydraLoRA module's ``_sigma`` and
        ``_sigma_features`` buffers with references to a single network-level
        tensor (per sigma_feature_dim for the features). Modules then read the
        same tensor object as their own attribute, so an in-place ``copy_`` on
        the network's shared buffer flows to every module without a Python
        propagation loop.

        Run once at the end of ``__init__`` — before any forward fires, so
        Dynamo / cudagraphs capture the aliased data pointer on first compile
        and never see a per-module pointer-mismatch event.
        """
        sigma_loras: List[torch.nn.Module] = []
        by_dim: Dict[int, List[torch.nn.Module]] = {}
        for lora in self.unet_loras + self.text_encoder_loras:
            if "_sigma" not in lora._buffers:
                continue
            sigma_loras.append(lora)
            d = int(getattr(lora, "sigma_feature_dim", 0))
            if d > 0 and "_sigma_features" in lora._buffers:
                by_dim.setdefault(d, []).append(lora)
        self._sigma_aware_loras = sigma_loras
        self._sigma_aware_loras_by_dim = by_dim
        if not sigma_loras:
            self._shared_sigma = None
            self._shared_sigma_features: Dict[int, torch.Tensor] = {}
            return

        # Pick the first module's placeholder buffer as the canonical shared
        # tensor; rebind every other module's buffer to the same object. The
        # placeholder is shape (1,) / (1, dim) — set_sigma replaces it with a
        # full-shape tensor on the first call (and re-aliases at the same time).
        shared_sigma = sigma_loras[0]._buffers["_sigma"]
        for lora in sigma_loras:
            lora._buffers["_sigma"] = shared_sigma
        self._shared_sigma = shared_sigma

        self._shared_sigma_features = {}
        for dim, loras in by_dim.items():
            shared_feat = loras[0]._buffers["_sigma_features"]
            for lora in loras:
                lora._buffers["_sigma_features"] = shared_feat
            self._shared_sigma_features[dim] = shared_feat

    def _wire_shared_fei_buffers(self) -> None:
        """Replace each FEI-aware module's ``_fei`` buffer with a single
        network-level shared tensor (per FEI feature dim).

        Mirrors ``_wire_shared_sigma_buffers``. ``set_fei`` writes to one
        shared buffer per dim; aliased module ``_fei`` buffers see the
        update through shared storage. The aliasing-recovery dance from
        ``set_sigma`` (rebind whenever shape or device drift breaks the
        identity) applies here too — ``Module._apply`` (``.to(device)``)
        independently reallocates buffers and silently breaks the link if
        we don't identity-check. See ``[[project_set_sigma_aliasing_bug]]``.
        """
        fei_loras: List[torch.nn.Module] = []
        by_dim: Dict[int, List[torch.nn.Module]] = {}
        for lora in self.unet_loras + self.text_encoder_loras:
            d = int(getattr(lora, "fei_feature_dim", 0))
            if d <= 0:
                continue
            if "_fei" not in lora._buffers:
                continue
            fei_loras.append(lora)
            by_dim.setdefault(d, []).append(lora)
        self._fei_aware_loras = fei_loras
        self._fei_aware_loras_by_dim = by_dim
        if not fei_loras:
            self._shared_fei: Dict[int, torch.Tensor] = {}
            return

        # One shared placeholder per dim — ``set_fei`` rebinds to full-shape
        # ``(B, dim)`` on first call.
        self._shared_fei = {}
        for dim, loras in by_dim.items():
            shared_feat = loras[0]._buffers["_fei"]
            for lora in loras:
                lora._buffers["_fei"] = shared_feat
            self._shared_fei[dim] = shared_feat

    def _wire_shared_broadcast_buffer(
        self, buffer_name: str, aware_attr: str, shared_attr: str
    ) -> None:
        """Alias every module carrying ``buffer_name`` to one shared ``(1, E)``
        tensor — the broadcast scaffold behind the routing / content / freq
        gate buffers.

        Each module registers a ``(1, E)`` uniform placeholder; this pass picks
        the first as canonical and rebinds the rest so one ``set_*`` per step
        propagates by reference. All such modules share one ``num_experts`` by
        construction, so no per-dim split (unlike ``_shared_fei``). Empty case
        records ``[]`` / ``None`` so ``set_*`` / ``clear_*`` no-op cleanly.
        """
        loras = [
            lora
            for lora in self.unet_loras + self.text_encoder_loras
            if buffer_name in lora._buffers
        ]
        setattr(self, aware_attr, loras)
        canonical = loras[0]._buffers[buffer_name] if loras else None
        for lora in loras:
            lora._buffers[buffer_name] = canonical
        setattr(self, shared_attr, canonical)

    def _wire_shared_routing_buffers(self) -> None:
        self._wire_shared_broadcast_buffer(
            "_routing_weights", "_routing_aware_loras", "_shared_routing_weights"
        )

    def _wire_shared_content_routing_buffers(self) -> None:
        self._wire_shared_broadcast_buffer(
            "_content_routing_weights",
            "_content_aware_loras",
            "_shared_content_routing_weights",
        )

    def _wire_shared_freq_routing_buffers(self) -> None:
        self._wire_shared_broadcast_buffer(
            "_freq_routing_weights",
            "_chimera_aware_loras",
            "_shared_freq_routing_weights",
        )

    def prepare_network(self, args):
        if getattr(args, "lora_fp32_accumulation", False):
            logger.warning(
                "--lora_fp32_accumulation is deprecated and has no effect; "
                "fp32 accumulation is now unconditional in LoRA/Hydra/ReFT "
                "bottleneck matmuls. Remove the flag from your config."
            )

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier
        for reft in self.text_encoder_refts + self.unet_refts:
            reft.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def fuse_weights(self):
        """Merge all LoRA deltas into base model weights for zero-overhead inference."""
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.fuse_weight()

    def unfuse_weights(self):
        """Remove all LoRA deltas from base model weights."""
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.unfuse_weight()

    def set_timestep_mask(self, timesteps: torch.Tensor, max_timestep: float = 1.0):
        """Compute and set timestep-dependent rank mask on all modules."""
        if not self.cfg.use_timestep_mask:
            return

        max_rank = self.cfg.lora_dim
        # Reuse a single GPU-resident mask to avoid ~200 CPU→GPU transfers per step
        mask = getattr(self, "_shared_timestep_mask", None)
        if mask is None or mask.device != timesteps.device:
            mask = torch.zeros(1, max_rank, device=timesteps.device)
            self._shared_timestep_mask = mask
            self._timestep_mask_arange = torch.arange(max_rank, device=timesteps.device)
            for lora in self.text_encoder_loras + self.unet_loras:
                lora._timestep_mask = mask

        # Compute threshold r entirely on device — avoids GPU→CPU .item() sync and
        # keeps the effective rank as a tensor so the mask build stays static-shape.
        t = timesteps.float().mean()
        frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
        r = (
            frac.pow(self.cfg.alpha_rank_scale) * (max_rank - self.cfg.min_rank)
            + self.cfg.min_rank
        )
        r = r.clamp(max=float(max_rank))
        mask.copy_((self._timestep_mask_arange < r).to(mask.dtype).unsqueeze(0))

    def set_reft_timestep_mask(
        self, timesteps: torch.Tensor, max_timestep: float = 1.0
    ):
        """Compute and set timestep-dependent mask on ReFT modules."""
        if not self.cfg.use_timestep_mask:
            return
        refts = self.text_encoder_refts + self.unet_refts
        if not refts:
            return
        reft_dim = self.cfg.reft_dim

        mask = getattr(self, "_shared_reft_mask", None)
        if mask is None or mask.device != timesteps.device:
            mask = torch.zeros(1, reft_dim, device=timesteps.device)
            self._shared_reft_mask = mask
            self._reft_mask_arange = torch.arange(reft_dim, device=timesteps.device)
            for reft in refts:
                reft._timestep_mask = mask

        t = timesteps.float().mean()
        frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
        r = frac.pow(self.cfg.alpha_rank_scale) * (reft_dim - 1) + 1
        r = r.clamp(max=float(reft_dim))
        mask.copy_((self._reft_mask_arange < r).to(mask.dtype).unsqueeze(0))

    def clear_timestep_mask(self):
        """Restore full-rank masks on every LoRA / ReFT module.

        Each module's ``_timestep_mask`` is a Tensor by construction (default
        all-ones buffer at init, rebound to the shared live-updated mask when
        ``set_timestep_mask`` runs). Clearing fills the shared masks with ones
        in place — modules that were rebound immediately see the neutral mask
        via the shared reference; modules with local defaults are already
        neutral. Never set to None: the always-a-Tensor invariant is what
        keeps the adapter forward free of a None-vs-Tensor guard under
        ``torch.compile``.
        """
        shared = getattr(self, "_shared_timestep_mask", None)
        if shared is not None:
            shared.fill_(1.0)
        shared_reft = getattr(self, "_shared_reft_mask", None)
        if shared_reft is not None:
            shared_reft.fill_(1.0)

    def set_sigma(self, sigmas: torch.Tensor) -> None:
        """Stash per-sample σ on every HydraLoRA module whose router accepts σ.

        Mirrors ``set_timestep_mask`` — one call per step, propagated by the
        shared-buffer aliasing from ``_wire_shared_sigma_buffers`` (one in-place
        ``copy_`` per tensor, no per-module loop). Write in place, not by
        rebinding: rebinding every step changes the data pointer and forces a
        cudagraph re-record under reduce-overhead. Pointer only changes on the
        first call (placeholder → full shape) or a batch-shape change.

        Aliasing-recovery: ``Module._apply`` (``.to(device)``) reallocates each
        buffer independently and orphans ``_shared_sigma``, so every call
        identity-checks the canonical module buffer and rebinds if broken —
        else the ``copy_`` writes to a stale CPU tensor and every module keeps
        reading zeros. Manifested only at B=1. See [[set_sigma_aliasing_bug]].
        """
        sigmas = sigmas.detach()
        self._last_sigma = sigmas
        # Either path needs per-module ``_sigma``: σ-feature concat router
        # (sigma_feature_dim>0) and hard σ-band expert partition. Skip the
        # propagation entirely when neither is configured.
        if not (
            self.cfg.router_source == "sigma"
            or self.cfg.specialize_experts_by_sigma_buckets
        ):
            return
        sigma_loras = self._sigma_aware_loras
        if not sigma_loras:
            return

        # Canonical = the live buffer on the first sigma-aware module. After
        # ``network.to(device)`` this is the GPU-allocated tensor; before any
        # device move it's still the CPU placeholder from
        # ``_wire_shared_sigma_buffers``.
        canonical = sigma_loras[0]._buffers["_sigma"]
        cast = sigmas.to(dtype=canonical.dtype, device=canonical.device)
        # Rebind whenever (a) the shared attribute lost identity with the
        # canonical (e.g. ``.to()`` rebinding broke aliasing) or (b) the
        # shape changed (placeholder → full batch). Both branches need to
        # re-alias every module so the next call's fast path actually
        # propagates.
        needs_rebind = (
            self._shared_sigma is not canonical or canonical.shape != cast.shape
        )
        if needs_rebind:
            new_sigma = cast.detach().clone()
            for lora in sigma_loras:
                lora._buffers["_sigma"] = new_sigma
            self._shared_sigma = new_sigma
            shared_sigma = new_sigma
        else:
            canonical.copy_(cast)
            shared_sigma = canonical

        for dim, loras in self._sigma_aware_loras_by_dim.items():
            canonical_feat = loras[0]._buffers["_sigma_features"]
            feat = _sigma_sinusoidal_features(shared_sigma, dim).detach()
            cast_feat = feat.to(
                dtype=canonical_feat.dtype, device=canonical_feat.device
            )
            feat_needs_rebind = (
                self._shared_sigma_features.get(dim) is not canonical_feat
                or canonical_feat.shape != cast_feat.shape
            )
            if feat_needs_rebind:
                new_feat = cast_feat.clone()
                for lora in loras:
                    lora._buffers["_sigma_features"] = new_feat
                self._shared_sigma_features[dim] = new_feat
            else:
                canonical_feat.copy_(cast_feat)

    def clear_sigma(self) -> None:
        """Reset cached σ to zeros (eval / validation / inference teardown).

        Never None: ``_sigma`` stays a Tensor so ``_compute_gate`` has no
        None-vs-Tensor guard to recompile on. Zero in place (stable cudagraph
        pointer) on the *live* canonical buffer, re-aliasing if ``Module._apply``
        broke the link — same recovery as ``set_sigma``.
        """
        self._last_sigma = None
        if not self._sigma_aware_loras:
            return
        sigma_loras = self._sigma_aware_loras
        canonical = sigma_loras[0]._buffers["_sigma"]
        if self._shared_sigma is not canonical:
            for lora in sigma_loras:
                lora._buffers["_sigma"] = canonical
            self._shared_sigma = canonical
        canonical.zero_()
        for dim, loras in self._sigma_aware_loras_by_dim.items():
            canonical_feat = loras[0]._buffers["_sigma_features"]
            if self._shared_sigma_features.get(dim) is not canonical_feat:
                for lora in loras:
                    lora._buffers["_sigma_features"] = canonical_feat
                self._shared_sigma_features[dim] = canonical_feat
            zero_feat = _sigma_sinusoidal_features(canonical, dim)
            cast_feat = zero_feat.to(
                dtype=canonical_feat.dtype, device=canonical_feat.device
            )
            if canonical_feat.shape == cast_feat.shape:
                canonical_feat.copy_(cast_feat)
            else:
                new_feat = cast_feat.detach().clone()
                for lora in loras:
                    lora._buffers["_sigma_features"] = new_feat
                self._shared_sigma_features[dim] = new_feat

    def set_fei(self, fei: torch.Tensor) -> None:
        """Stash per-sample FEI ``[B, fei_dim]`` on every FEI-aware module.

        Parallel to ``set_sigma`` — one call per training/inference step.
        Same shared-buffer aliasing recovery: identity-check ``self._shared_fei``
        against the canonical module's live buffer, rebind on shape change
        or after ``Module._apply`` orphans the link
        (``[[project_set_sigma_aliasing_bug]]``).

        ``fei`` must be ``(B, fei_feature_dim)`` matching
        ``cfg.fei_feature_dim`` (default 2 for the simplex). Caller is the
        train/inference loop running ``library.runtime.fei.compute_fei_2band``
        on ``z_t`` once per step.

        When ``cfg.route_per_layer=False`` and a ``GlobalRouter`` is wired,
        the router fires on the fresh FEI and its gates are broadcast to
        every routing-aware module via ``set_routing_weights`` in the same
        call — one entry point for the FeRA-style global-router path.
        """
        fei = fei.detach()
        # Fast-path: if there are no per-Linear FEI consumers, no global
        # router, and no chimera FreqRouter needing FEI, nothing to do.
        has_per_layer_fei = bool(getattr(self, "_fei_aware_loras", None))
        global_fei_router = (
            self.global_router
            if (
                self.global_router is not None
                and self.cfg.router_source == "fei"
                and not self.cfg.route_per_layer
            )
            else None
        )
        chimera_freq_router = (
            self.freq_router
            if (
                getattr(self, "freq_router", None) is not None
                and getattr(self, "_chimera_aware_loras", None)
            )
            else None
        )
        # Hardwired-FEI freq pool: no router module, broadcast the simplex
        # directly. freq_router is None in this mode, so it needs its own flag.
        chimera_fei_active = bool(
            self.cfg.use_chimera_hydra
            and getattr(self, "_chimera_aware_loras", None)
            and getattr(self, "freq_router_mode", "learned") == "fei"
        )
        if not (
            has_per_layer_fei
            or global_fei_router is not None
            or chimera_freq_router is not None
            or chimera_fei_active
        ):
            return
        if not (
            self.use_fei_router
            or global_fei_router is not None
            or chimera_freq_router is not None
            or chimera_fei_active
        ):
            return

        # Per-layer FEI broadcast (legacy path — FEI-on-Hydra Phase 1).
        if has_per_layer_fei:
            # Group loras by their feature dim — every fei-aware module
            # currently in our network shares the same dim (cfg-level), but
            # the loop is robust to a future per-layer dim override.
            for dim, loras in self._fei_aware_loras_by_dim.items():
                canonical = loras[0]._buffers["_fei"]
                cast = fei.to(dtype=canonical.dtype, device=canonical.device)
                if cast.dim() == 1:
                    cast = cast.unsqueeze(0)
                if cast.shape[-1] != dim:
                    raise ValueError(
                        f"set_fei: fei.shape[-1]={cast.shape[-1]} != fei_feature_dim={dim}"
                    )
                current_shared = self._shared_fei.get(dim)
                needs_rebind = (
                    current_shared is not canonical or canonical.shape != cast.shape
                )
                if needs_rebind:
                    new_fei = cast.detach().clone()
                    for lora in loras:
                        lora._buffers["_fei"] = new_fei
                    self._shared_fei[dim] = new_fei
                else:
                    canonical.copy_(cast)

        # Global router (FeRA-style): fire on fresh FEI and broadcast gates.
        # Router runs WITH grad so the autograd path ``L_denoise → y_t →
        # α_{t,m} → g_φ`` (FeRA eq. 6-7, 11) reaches the GlobalRouter params.
        # ``set_routing_weights`` reassigns each expert module's buffer slot
        # to the live ``gates`` tensor (no detach, no in-place copy).
        if global_fei_router is not None:
            gates = global_fei_router(fei)
            self.set_routing_weights(gates)

        # ChimeraHydra FreqRouter: input is concat(FEI, sinusoidal-σ-features).
        # σ already arrived through ``set_sigma`` (which fires before
        # ``set_fei`` in ``apply_router_conditioning``); the freq router lives
        # at network level and computes its features fresh each step rather
        # than relying on per-module shared σ-feature buffers (chimera modules
        # are built with ``sigma_feature_dim=0`` since the freq router owns
        # the σ axis exclusively).
        if chimera_freq_router is not None:
            sigma = self._last_sigma
            if sigma is None:
                raise RuntimeError(
                    "ChimeraHydra FreqRouter requires set_sigma to fire before "
                    "set_fei within the same step (apply_router_conditioning "
                    "preserves this order — check custom call sites)."
                )
            sigma_dim = int(self.cfg.sigma_feature_dim)
            sigma_feat = _sigma_sinusoidal_features(sigma, sigma_dim)
            # Match the FEI tensor's device/dtype and batch axis. Both should
            # share the same B by construction (one σ per sample, one FEI per
            # sample), so a straight cat is correct.
            fei_cast = fei.to(device=sigma_feat.device, dtype=sigma_feat.dtype)
            if fei_cast.dim() == 1:
                fei_cast = fei_cast.unsqueeze(0)
            router_in = torch.cat([fei_cast, sigma_feat], dim=-1)
            freq_gates = chimera_freq_router(router_in)
            self.set_freq_routing_weights(freq_gates)

        # ChimeraHydra hardwired-FEI freq pool: π_f = normalize(FEI ** (1/τ)).
        # No learned router, no σ-feature input — the FEI band-simplex IS the
        # gate (K_f == fei bands, enforced at construction). π_f is detached
        # (FEI is detached above) and carries no grad_fn: there are no router
        # params to receive gradient, and the freq experts learn through their
        # own weights — a fixed gate analogous to T-LoRA's timestep mask.
        elif chimera_fei_active:
            fei_cast = fei.float()
            if fei_cast.dim() == 1:
                fei_cast = fei_cast.unsqueeze(0)
            pi_f = _fei_temperature(fei_cast, float(self.freq_router_tau))
            self.set_freq_routing_weights(pi_f)

    def clear_fei(self) -> None:
        """Reset cached FEI to zeros without rebinding pointers.

        Same in-place-zero pattern as ``clear_sigma`` — keeps cudagraph
        data pointers stable. Re-establishes aliasing if ``Module._apply``
        broke it since the last call.
        """
        if not getattr(self, "_fei_aware_loras", None):
            return
        for dim, loras in self._fei_aware_loras_by_dim.items():
            canonical = loras[0]._buffers["_fei"]
            current_shared = self._shared_fei.get(dim)
            if current_shared is not canonical:
                for lora in loras:
                    lora._buffers["_fei"] = canonical
                self._shared_fei[dim] = canonical
            canonical.zero_()

    def _broadcast_gate(
        self,
        weights: torch.Tensor,
        aware_attr: str,
        buffer_name: str,
        shared_attr: str,
    ) -> None:
        """Slot-assign a ``(B, E)`` gate tensor to every module's ``buffer_name``.

        Assigns the SAME live ``weights`` reference (NO detach, NO copy_) so the
        buffer carries the router's grad_fn — that autograd path
        (``L_denoise → y_t → α → router params``, FeRA eq. 7) is what trains the
        router. cudagraph pointer stability is deliberately traded away here:
        gates are a tiny ``(B, E)`` tensor and the gradient path is the point.
        """
        loras = getattr(self, aware_attr, None)
        if not loras:
            return
        canonical_buf = loras[0]._buffers[buffer_name]
        w = weights.to(dtype=canonical_buf.dtype, device=canonical_buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        for lora in loras:
            setattr(lora, buffer_name, w)  # buffer slot reassign, grad_fn kept
        setattr(self, shared_attr, w)

    def _reset_gate(self, aware_attr: str, buffer_name: str, shared_attr: str) -> None:
        """Reset a broadcast gate buffer to uniform ``1/E`` in place.

        Pointer stays stable for cudagraph capture; re-aliases if
        ``Module._apply`` (``.to(device)``) broke the shared link.
        """
        loras = getattr(self, aware_attr, None)
        if not loras:
            return
        canonical = loras[0]._buffers[buffer_name]
        if getattr(self, shared_attr) is not canonical:
            for lora in loras:
                lora._buffers[buffer_name] = canonical
            setattr(self, shared_attr, canonical)
        E = int(canonical.shape[-1])
        canonical.fill_(1.0 / max(E, 1))

    def set_routing_weights(self, weights: torch.Tensor) -> None:
        """Broadcast a ``(B, E)`` gate tensor to every routing-aware module.

        Fired internally by ``set_fei`` (GlobalRouter, FEI source) or externally
        by inference callers pushing pre-computed gates. See ``_broadcast_gate``.
        """
        self._broadcast_gate(
            weights,
            "_routing_aware_loras",
            "_routing_weights",
            "_shared_routing_weights",
        )

    def clear_routing_weights(self) -> None:
        """Reset GlobalRouter gates to uniform ``1/E`` (between steps / teardown)."""
        self._reset_gate(
            "_routing_aware_loras", "_routing_weights", "_shared_routing_weights"
        )

    def set_crossattn_routing(self, crossattn_emb: torch.Tensor) -> None:
        """Fire the network-level GlobalRouter on a pooled text vector.

        Used when ``cfg.router_source="crossattn_emb"`` (route_per_layer=False).
        ``crossattn_emb`` is the post-LLM-adapter text feature tensor — either
        ``(B, L, D)`` (raw, the GlobalRouter pools) or ``(B, D)`` (pre-pooled).
        No-op when no crossattn GlobalRouter is wired.

        Router runs WITH grad so ``L_denoise → y_t → α → GlobalRouter params``
        is intact; broadcast through :meth:`set_routing_weights` (the same
        ``_routing_weights`` slot the σ/FEI global router writes — the Hydra /
        stacked-experts modules need no crossattn-specific buffer).

        Call BEFORE each forward, separately for cond / uncond branches at
        inference — gates depend on the caption, so the two branches route
        differently (parallel to chimera's ``set_content``).
        """
        if self.global_router is None or not getattr(
            self, "use_crossattn_router", False
        ):
            return
        gates = self.global_router(crossattn_emb)
        self.set_routing_weights(gates)

    def set_freq_routing_weights(self, weights: torch.Tensor) -> None:
        """Broadcast ``π_f`` from the FreqRouter to every chimera module's
        ``_freq_routing_weights`` (``_compute_gate`` reads it for the
        ``[π_c | π_f]`` concat). See ``_broadcast_gate``."""
        self._broadcast_gate(
            weights,
            "_chimera_aware_loras",
            "_freq_routing_weights",
            "_shared_freq_routing_weights",
        )

    def clear_freq_routing_weights(self) -> None:
        """Reset chimera freq gates to uniform ``1/K_f`` in place."""
        self._reset_gate(
            "_chimera_aware_loras",
            "_freq_routing_weights",
            "_shared_freq_routing_weights",
        )

    def set_content(self, crossattn_emb: torch.Tensor) -> None:
        """Fire the network-level ContentRouter on a pooled text vector.

        ``crossattn_emb`` is the post-LLM-adapter text feature tensor —
        either ``(B, L, D)`` (raw, this method pools) or ``(B, D)``
        (pre-pooled by the caller). No-op when the network has no
        ContentRouter (chimera off).

        Router runs WITH grad so ``L_denoise → out_c → π_c → ContentRouter
        params`` is intact. Slot-assigned through
        :meth:`set_content_routing_weights`, same broadcast contract as
        ``set_freq_routing_weights`` / ``set_routing_weights``.
        """
        if self.content_router is None:
            return
        if not getattr(self, "_content_aware_loras", None):
            return
        gates = self.content_router(crossattn_emb)
        self.set_content_routing_weights(gates)

    def set_content_routing_weights(self, weights: torch.Tensor) -> None:
        """Broadcast ``π_c`` from the ContentRouter to every chimera module's
        ``_content_routing_weights``. Externally callable for inference paths
        that pre-compute gates. See ``_broadcast_gate``."""
        self._broadcast_gate(
            weights,
            "_content_aware_loras",
            "_content_routing_weights",
            "_shared_content_routing_weights",
        )

    def clear_content_routing_weights(self) -> None:
        """Reset chimera content gates to uniform ``1/K_c`` in place."""
        self._reset_gate(
            "_content_aware_loras",
            "_content_routing_weights",
            "_shared_content_routing_weights",
        )

    def clear_step_caches(self) -> None:
        """Drop per-step tensor references (``_last_gate``) and invalidate
        memoized router-stats caches between training steps.

        Called unconditionally from the training loop before each forward,
        for two reasons:

        (1) ``_last_gate`` caches a tensor produced inside the compiled
        forward — under ``torch.compile(mode='reduce-overhead')`` that tensor
        lives in the inductor cudagraph memory pool. Holding a Python
        reference across the step boundary prevents ``cudagraph_trees`` from
        reclaiming pool memory and silently demotes the run to the eager
        fallback path. Call must precede ``cudagraph_mark_step_begin()``.

        (2) ``_router_stats_cache`` / ``_chimera_router_stats_cache`` memoize
        per-step router diagnostics so the progress-bar postfix and the TB
        logging layer share one D2H sync. Without per-step invalidation
        these freeze at their first computed values — and on runs without
        cudagraph mode (``_cudagraph_mark_step=False``) the invalidation has
        no other trigger, so TB shows the same usage/entropy on every log
        step.

        ``_sigma`` is intentionally *not* cleared: it's rebound by
        ``set_sigma`` before every forward, the caller passes a tensor from
        outside the compiled region (the flow-matching sampler's ``timesteps``,
        not a pool-allocated intermediate), and keeping it a Tensor at all
        times is what lets the adapter ``_compute_gate`` drop the None-vs-
        Tensor guard under ``torch.compile``.

        Safe to call unconditionally — consumers (balance loss, router stats)
        read ``_last_gate`` only within the step that wrote it.
        """
        self._last_sigma = None
        self._router_stats_cache = None
        self._chimera_router_stats_cache = None
        for lora in self.unet_loras + self.text_encoder_loras:
            if hasattr(lora, "_last_gate"):
                lora._last_gate = None
        # Drop the GlobalRouter's per-step transients for the same reason —
        # ``_last_gates`` / ``_last_input`` are detached tensors that may live
        # in the inductor cudagraph memory pool; holding a Python reference
        # across the step boundary blocks pool reclamation.
        if self.global_router is not None:
            self.global_router._last_gates = None
            self.global_router._last_input = None
            self.global_router._last_fei = None
        # Same treatment for the chimera FreqRouter.
        if getattr(self, "freq_router", None) is not None:
            self.freq_router._last_gates = None
            self.freq_router._last_input = None
        # …and the chimera ContentRouter (network-level content-pool variant).
        if getattr(self, "content_router", None) is not None:
            self.content_router._last_gates = None
            self.content_router._last_input = None

    @staticmethod
    def _strip_orig_mod_keys(state_dict):
        """Strip torch.compile '_orig_mod_' from state_dict keys for compat with old checkpoints."""
        new_sd = {}
        for key, val in state_dict.items():
            new_key = re.sub(r"(?<=_)_orig_mod_", "", key)
            new_sd[new_key] = val
        return new_sd

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        state_dict = self._strip_orig_mod_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        # Stack per-expert hydra ups into fused lora_up_weight (training form).
        # Also stacks per-expert ``.lora_downs.{i}.weight`` for the
        # StackedExperts (independent-A) layout — no-op for Hydra.
        weights_sd = _stack_lora_ups(weights_sd)
        # Refuse split stacked-experts first (its discriminator is per-expert
        # ``lora_down_weight`` 3-D, which the hydra refuser would otherwise
        # short-circuit on the absent shared ``lora_down.weight``).
        weights_sd = _refuse_split_stacked_experts_keys(weights_sd)
        # Refuse split hydra attn keys BEFORE the regular refuser: hydra splits
        # carry no lora_up.weight, so the regular path would skip them anyway,
        # but running hydra first means any non-hydra attention still goes
        # through the normal code path cleanly.
        weights_sd = _refuse_split_hydra_keys(weights_sd)
        # Refuse unfused attn projections (inverse of save_weights defusing).
        weights_sd = _refuse_unfused_attn_lora_keys(weights_sd)

        self._reabsorb_baked_inv_scale(weights_sd)

        info = self.load_state_dict(weights_sd, False)
        return info

    def _reabsorb_baked_inv_scale(self, weights_sd: Dict[str, torch.Tensor]) -> None:
        """Resume guard for baked (inv_scale-folded) checkpoints.

        ``save_network_weights`` now bakes ``inv_scale`` into ``lora_down`` and
        drops the key (see ``lora.bake_inv_scale``), so a baked checkpoint
        carries a raw-input ``down`` and no ``inv_scale``. On *resume*
        (``create_network`` with ``channel_scaling_alpha>0`` → modules build an
        ``inv_scale`` buffer ``1/s_norm`` and bake ``s_norm`` into their init
        ``down``), ``load_state_dict`` would overwrite ``down`` with the raw
        delta while the buffer survives — so the forward ``x*inv_scale @ down``
        would apply ``1/s_norm`` with nothing absorbing it. Re-absorb here: move
        the incoming raw ``down`` back into training space (``down *= s_norm``)
        and re-inject the buffer's ``inv_scale`` so the round trip is exact.

        No-op for inference (modules built without channel scaling) and for
        legacy checkpoints that still carry ``inv_scale`` (the key is present,
        so we leave both ``down`` and the buffer to load straight through).
        """
        for lora in self.unet_loras + self.text_encoder_loras:
            if not getattr(lora, "_has_channel_scale", False):
                continue
            name = lora.lora_name
            down_key = f"{name}.lora_down.weight"
            if f"{name}.inv_scale" in weights_sd or down_key not in weights_sd:
                continue
            inv_scale = lora.inv_scale  # (in,) fp32, == 1/s_norm
            down = weights_sd[down_key]
            s_norm = (
                inv_scale.to(device=down.device, dtype=torch.float)
                .clamp_min(1e-12)
                .reciprocal()
            )
            weights_sd[down_key] = (down.to(torch.float) * s_norm.unsqueeze(0)).to(
                down.dtype
            )
            weights_sd[f"{name}.inv_scale"] = inv_scale.clone()

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(
                f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules"
            )
        else:
            self.text_encoder_loras = []
            self.text_encoder_refts = []

        if apply_unet:
            logger.info(f"enable LoRA for DiT: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []
            self.unet_refts = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        # ReFT wraps each selected DiT Block's forward, so the chain is:
        #   Block.__call__ -> ReFT.forward -> original Block.forward
        #   (inside which LoRA-wrapped Linears still fire normally).
        for reft in self.text_encoder_refts + self.unet_refts:
            reft.apply_to()
            self.add_module(reft.lora_name, reft)

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_ANIMA):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for DiT")
        else:
            self.unet_loras = []

        # Pre-group checkpoint keys by LoRA module prefix (avoid O(modules * keys) scan)
        # Keys are "{module_name}.{param}" where module_name has no dots (dots → underscores)
        grouped_sd: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in weights_sd.items():
            prefix, dot, suffix = key.partition(".")
            if not dot:
                continue
            if prefix not in grouped_sd:
                grouped_sd[prefix] = {}
            grouped_sd[prefix][suffix] = value

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = grouped_sd.get(lora.lora_name, {})
            if sd_for_lora:
                lora.merge_to(sd_for_lora, dtype, device)

        logger.info("weights are merged")

    def set_loraplus_lr_ratio(
        self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio
    ):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(
            f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}"
        )
        logger.info(
            f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}"
        )

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        if text_encoder_lr is None or (
            isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0
        ):
            text_encoder_lr = [default_lr]
        elif isinstance(text_encoder_lr, float) or isinstance(text_encoder_lr, int):
            text_encoder_lr = [float(text_encoder_lr)]
        elif len(text_encoder_lr) == 1:
            pass  # already a list with one element

        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}, "router": {}}
            reg_groups = {}
            reg_lrs_list = (
                list(self.cfg.reg_lrs.items()) if self.cfg.reg_lrs is not None else []
            )
            router_scale = float(self.cfg.router_lr_scale)
            # Chimera content-router multiplier (stacks on router_scale). The
            # per-Linear ``router.*`` group below collects chimera's content
            # router params (chimera modules own the only per-Linear
            # ``router.*`` in their network). Off-by-default for non-chimera
            # runs so plain Hydra is unaffected.
            content_router_scale = (
                float(self.cfg.content_router_lr_scale)
                if getattr(self.cfg, "use_chimera_hydra", False)
                else 1.0
            )
            router_lr_mult = router_scale * content_router_scale

            def _is_router_param(pname: str) -> bool:
                # named_parameters() yields top-level names like "router.weight"
                # — no leading dot. σ features live inside router.weight now
                # (columns [lora_dim:] of the weight), so there's a single path.
                return pname.startswith("router.")

            for lora in loras:
                matched_reg_lr = None
                for i, (regex_str, reg_lr) in enumerate(reg_lrs_list):
                    if re.fullmatch(regex_str, lora.original_name):
                        matched_reg_lr = (i, reg_lr)
                        logger.info(
                            f"Module {lora.original_name} matched regex '{regex_str}' -> LR {reg_lr}"
                        )
                        break

                for name, param in lora.named_parameters():
                    is_router = _is_router_param(name)
                    if matched_reg_lr is not None:
                        reg_idx, reg_lr = matched_reg_lr
                        group_key = f"reg_lr_{reg_idx}"
                        if group_key not in reg_groups:
                            reg_groups[group_key] = {
                                "lora": {},
                                "plus": {},
                                "router": {},
                                "lr": reg_lr,
                            }
                        if is_router:
                            reg_groups[group_key]["router"][
                                f"{lora.lora_name}.{name}"
                            ] = param
                        elif loraplus_ratio is not None and (
                            "lora_up" in name
                            or "p_layer" in name
                            or "learned_source" in name
                        ):
                            reg_groups[group_key]["plus"][
                                f"{lora.lora_name}.{name}"
                            ] = param
                        else:
                            reg_groups[group_key]["lora"][
                                f"{lora.lora_name}.{name}"
                            ] = param
                        continue

                    if is_router:
                        param_groups["router"][f"{lora.lora_name}.{name}"] = param
                    elif loraplus_ratio is not None and (
                        "lora_up" in name
                        or "p_layer" in name
                        or "learned_source" in name
                    ):
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for group_key, group in reg_groups.items():
                reg_lr = group["lr"]
                for key in ("lora", "plus", "router"):
                    param_data = {"params": group[key].values()}
                    if len(param_data["params"]) == 0:
                        continue
                    if key == "plus":
                        param_data["lr"] = (
                            reg_lr * loraplus_ratio
                            if loraplus_ratio is not None
                            else reg_lr
                        )
                    elif key == "router":
                        param_data["lr"] = reg_lr * router_lr_mult
                    else:
                        param_data["lr"] = reg_lr
                    if (
                        param_data.get("lr", None) == 0
                        or param_data.get("lr", None) is None
                    ):
                        logger.info("NO LR skipping!")
                        continue
                    params.append(param_data)
                    desc = f"reg_lr_{group_key.split('_')[-1]}"
                    descriptions.append(
                        desc
                        + (
                            " plus"
                            if key == "plus"
                            else (" router" if key == "router" else "")
                        )
                    )

            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}
                if len(param_data["params"]) == 0:
                    continue
                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    elif key == "router":
                        param_data["lr"] = lr * router_lr_mult
                    else:
                        param_data["lr"] = lr
                if (
                    param_data.get("lr", None) == 0
                    or param_data.get("lr", None) is None
                ):
                    logger.info("NO LR skipping!")
                    continue
                params.append(param_data)
                descriptions.append(
                    "plus" if key == "plus" else ("router" if key == "router" else "")
                )
            return params, descriptions

        if self.text_encoder_loras:
            loraplus_ratio = (
                self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio
            )
            te1_loras = [
                lora
                for lora in self.text_encoder_loras
                if lora.lora_name.startswith(self.LORA_PREFIX_TEXT_ENCODER)
            ]
            if len(te1_loras) > 0:
                logger.info(
                    f"Text Encoder 1 (Qwen3): {len(te1_loras)} modules, LR {text_encoder_lr[0]}"
                )
                params, descriptions = assemble_params(
                    te1_loras, text_encoder_lr[0], loraplus_ratio
                )
                all_params.extend(params)
                lr_descriptions.extend(
                    ["textencoder 1" + (" " + d if d else "") for d in descriptions]
                )

        if self.unet_loras:
            params, descriptions = assemble_params(
                self.unet_loras,
                unet_lr if unet_lr is not None else default_lr,
                self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(
                ["unet" + (" " + d if d else "") for d in descriptions]
            )

        if self.text_encoder_refts:
            params, descriptions = assemble_params(
                self.text_encoder_refts,
                text_encoder_lr[0],
                self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(
                ["reft textencoder" + (" " + d if d else "") for d in descriptions]
            )

        if self.unet_refts:
            params, descriptions = assemble_params(
                self.unet_refts,
                unet_lr if unet_lr is not None else default_lr,
                self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(
                ["reft unet" + (" " + d if d else "") for d in descriptions]
            )

        # HydraLoRA per-module routers are submodules of HydraLoRAModule instances,
        # so they are already captured by the unet_loras param group above.

        # GlobalRouter (route_per_layer=False) lives on the network, not on
        # per-Linear LoRA modules, so the assemble_params loop above misses it.
        # Add it explicitly with the same router_lr_scale convention used for
        # per-Linear routers (unet_lr × router_lr_scale).
        if getattr(self, "global_router", None) is not None:
            gr_params = list(self.global_router.parameters())
            if len(gr_params) > 0:
                router_scale = float(self.cfg.router_lr_scale)
                base_lr = unet_lr if unet_lr is not None else default_lr
                if base_lr is None or base_lr == 0:
                    logger.info("GlobalRouter: no base LR, skipping param group")
                else:
                    gr_lr = float(base_lr) * router_scale
                    all_params.append({"params": gr_params, "lr": gr_lr})
                    lr_descriptions.append("global router")
                    logger.info(
                        f"GlobalRouter param group: lr={gr_lr:.2e} "
                        f"({router_scale}x of unet_lr={base_lr})"
                    )

        # ChimeraHydra FreqRouter mirrors the GlobalRouter param-group
        # treatment. Same router_lr_scale convention so a single knob tunes
        # both router families.
        if getattr(self, "freq_router", None) is not None:
            fr_params = list(self.freq_router.parameters())
            if len(fr_params) > 0:
                router_scale = float(self.cfg.router_lr_scale)
                freq_scale = float(self.cfg.freq_router_lr_scale)
                base_lr = unet_lr if unet_lr is not None else default_lr
                if base_lr is None or base_lr == 0:
                    logger.info("FreqRouter: no base LR, skipping param group")
                else:
                    fr_lr = float(base_lr) * router_scale * freq_scale
                    all_params.append({"params": fr_params, "lr": fr_lr})
                    lr_descriptions.append("chimera freq router")
                    logger.info(
                        f"ChimeraHydra FreqRouter param group: lr={fr_lr:.2e} "
                        f"({router_scale}x router_lr_scale × {freq_scale}x "
                        f"freq_router_lr_scale of unet_lr={base_lr})"
                    )

        # ChimeraHydra ContentRouter param group. Stack router_lr_scale and
        # content_router_lr_scale for symmetry with the freq side; the LN
        # is parameterless so the only params here are the two Linears.
        if getattr(self, "content_router", None) is not None:
            cr_params = list(self.content_router.parameters())
            if len(cr_params) > 0:
                router_scale = float(self.cfg.router_lr_scale)
                content_scale = float(self.cfg.content_router_lr_scale)
                base_lr = unet_lr if unet_lr is not None else default_lr
                if base_lr is None or base_lr == 0:
                    logger.info("ContentRouter: no base LR, skipping param group")
                else:
                    cr_lr = float(base_lr) * router_scale * content_scale
                    all_params.append({"params": cr_params, "lr": cr_lr})
                    lr_descriptions.append("chimera content router")
                    logger.info(
                        f"ChimeraHydra ContentRouter param group: lr={cr_lr:.2e} "
                        f"({router_scale}x router_lr_scale × {content_scale}x "
                        f"content_router_lr_scale of unet_lr={base_lr})"
                    )

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        pass  # not supported

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        spec: NetworkSpec = getattr(self, "_network_spec", NETWORK_REGISTRY["lora"])
        if metadata is None:
            metadata = {}
        if metadata:
            metadata["ss_network_spec"] = spec.name

        # Hard σ-band partition lives in non-persistent buffers (`_expert_band`)
        # and a Python attr (`_sigma_band_partition`); nothing of it survives
        # the state_dict write. Emit the two scalars needed to re-register the
        # partition at load time so inference (`make test`) and the ComfyUI
        # node can reconstruct the per-sample band mask. Only stamped when the
        # partition is on, so older non-band checkpoints stay byte-identical.
        if self.cfg.specialize_experts_by_sigma_buckets:
            metadata["ss_specialize_experts_by_sigma_buckets"] = "true"
            metadata["ss_num_sigma_buckets"] = str(int(self.cfg.num_sigma_buckets))
            if self.cfg.sigma_bucket_boundaries is not None:
                import json as _json

                metadata["ss_sigma_bucket_boundaries"] = _json.dumps(
                    list(self.cfg.sigma_bucket_boundaries)
                )

        # Three-axis routing config (plan2 §three-axis-config). Stamped on
        # every save so the loader can reconstruct the exact router layout
        # without key-sniffing — particularly important for distinguishing
        # ``stacked_experts_global_fei`` (independent-A) from ``hydra``
        # (shared-A) at a glance.
        if self.cfg.use_moe_style is not False:
            metadata["ss_use_moe_style"] = str(self.cfg.use_moe_style)
            metadata["ss_route_per_layer"] = (
                "true" if self.cfg.route_per_layer else "false"
            )
            metadata["ss_router_source"] = str(self.cfg.router_source)

        # OrthoHydra centered-gate: the distilled ``_moe`` ups are combined
        # with ``(g_e - 1/E)`` rather than the raw softmax. Stamp only when on
        # so unconfigured checkpoints stay byte-identical; the loader threads
        # it back into the runtime HydraLoRAModule combine for inference parity.
        if getattr(self.cfg, "ortho_centered_gate", False):
            metadata["ss_ortho_centered_gate"] = "true"

        # FEI router params (router-source-specific scalars the loader needs
        # to size the router input). Stamped for both per-Linear and global
        # FEI routers.
        if self.cfg.router_source == "fei" and self.cfg.fei_feature_dim > 0:
            metadata["ss_fei_feature_dim"] = str(int(self.cfg.fei_feature_dim))
            metadata["ss_fei_sigma_low_div"] = str(float(self.cfg.fei_sigma_low_div))

        # ChimeraHydra: the pool split is the only non-key info the loader
        # cannot reconstruct from state_dict (P_bases shape encodes E = K_c +
        # K_f but not the split point). FreqRouter weights survive as plain
        # ``freq_router.*`` keys without dedicated handling. FEI/σ feature
        # dims are also stamped so the loader can re-size the freq router
        # input — they live outside the standard ``router_source`` flow
        # (chimera uses BOTH simultaneously).
        if self.cfg.use_chimera_hydra:
            metadata["ss_use_chimera_hydra"] = "true"
            metadata["ss_num_experts_content"] = str(int(self.cfg.num_experts_content))
            metadata["ss_num_experts_freq"] = str(int(self.cfg.num_experts_freq))
            metadata["ss_chimera_fei_feature_dim"] = str(int(self.cfg.fei_feature_dim))
            metadata["ss_chimera_sigma_feature_dim"] = str(
                int(self.cfg.sigma_feature_dim)
            )
            metadata["ss_chimera_fei_sigma_low_div"] = str(
                float(self.cfg.fei_sigma_low_div)
            )
            # FreqRouter input LN flag. Parameterless LN leaves no tensor
            # footprint in the state_dict, so the loader can't sniff it from
            # weights — has to come from metadata. Default-off on rebuild
            # when absent preserves pre-LN checkpoint inference.
            metadata["ss_chimera_freq_router_layer_norm"] = (
                "true" if self.cfg.freq_router_layer_norm else "false"
            )
            # Freq routing mode + FEI-gate temperature. "fei" means the freq
            # pool was routed by the hardwired FEI simplex (no FreqRouter
            # weights in the state_dict) — the loader must NOT try to rebuild a
            # FreqRouter and must re-broadcast the simplex at inference. Absent
            # stamp ⇒ "learned" (every pre-2026-05-27 chimera checkpoint).
            metadata["ss_chimera_freq_router_mode"] = str(
                getattr(self, "freq_router_mode", "learned")
            )
            metadata["ss_chimera_freq_router_tau"] = str(
                float(getattr(self, "freq_router_tau", 1.0))
            )
            # Content routing is always the network-level ContentRouter fed
            # pooled ``crossattn_emb`` (the per-Linear router was removed), and
            # both pools are always centered-gate (ups combined with
            # ``(π - 1/K)``). Stamp both as constants so the ComfyUI node's
            # loader (`_parse_chimera_content_router`) rebuilds the ContentRouter
            # and the inference module applies the centered combine. The LN flag
            # is parameterless (no tensor footprint), so it must travel in
            # metadata.
            metadata["ss_chimera_content_router_source"] = "crossattn_emb"
            metadata["ss_chimera_content_router_layer_norm"] = (
                "true" if self.cfg.content_router_layer_norm else "false"
            )
            metadata["ss_chimera_centered_gate"] = "true"

        state_dict = self.state_dict()
        lora_save.save_network_weights(
            state_dict,
            file=file,
            dtype=dtype,
            metadata=metadata,
            save_variant=spec.save_variant,
        )

    def backup_weights(self):
        loras: List[LoRAModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                org_module._lora_org_weight = org_module.weight.detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        loras: List[LoRAModule] = self.text_encoder_loras + self.unet_loras
        with torch.no_grad():
            for lora in loras:
                org_module = lora.org_module_ref[0]
                if not org_module._lora_restored:
                    org_module.weight.data.copy_(org_module._lora_org_weight)
                    org_module._lora_restored = True

    def pre_calculation(self):
        loras: List[LoRAModule] = self.text_encoder_loras + self.unet_loras
        with torch.no_grad():
            for lora in loras:
                org_module = lora.org_module_ref[0]
                lora_weight = lora.get_weight().to(
                    org_module.weight.device, dtype=org_module.weight.dtype
                )
                org_module.weight.data.add_(lora_weight)

                org_module._lora_restored = False
                lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (
                    (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2))
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(
                    down.permute(1, 0, 2, 3), up
                ).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)
