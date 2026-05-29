# Anima LoRA training script (merged standalone)

import importlib
import argparse
import math
import os
import typing
from typing import Any, Union, Optional
import sys
import random
import time
from multiprocessing import Value

# Windows: suppress per-kernel ptxas.exe / cl.exe console flashes from
# torch.compile + Triton. Must run before any subprocess.Popen call (i.e.
# before torch import on Windows where inductor may prefetch toolchain).
from library.runtime.proc import install_no_window_default

install_no_window_default()

import torch
import torch.nn as nn
from library.runtime.device import clean_memory_on_device

from accelerate.utils import set_seed
from accelerate import Accelerator
from library import (
    train_util,
)
from library.anima import (
    models as anima_models,
    training as anima_train_utils,
    weights as anima_utils,
    strategy as strategy_anima,
    text_strategies,
)
from library.models import qwen_vae as qwen_image_autoencoder_kl
from library.models import sai_spec as sai_model_spec
from library.runtime import noise as noise_utils
from library.config import loader as config_util
from library.config.loader import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.training.method_adapter import (
    ForwardArtifacts,
    MethodAdapter,
    SetupCtx,
    StepCtx,
    resolve_adapters,
)
from library.config.io import (
    load_dataset_config_from_base,
    read_config_from_file,
)
from library.datasets import (
    DatasetGroup,
    MinimalDataset,
    collator_class,
    debug_dataset,
    load_arbitrary_dataset,
)
from library.datasets import base as _datasets_base
from library.runtime.accelerator import (
    patch_accelerator_for_fp16_training,
    prepare_accelerator,
    prepare_dtype,
    resume_from_local_or_hf_if_specified,
)
from library.training import (
    CheckpointSaver,
    LossContext,
    SAMPLER_REGISTRY,
    RuntimeState,
    SamplerContext,
    TrainCtx,
    add_custom_train_arguments,
    add_dataset_arguments,
    add_dataset_metadata,
    add_dit_training_arguments,
    add_masked_loss_arguments,
    add_model_hash_metadata,
    add_network_arguments,
    add_optimizer_arguments,
    add_sd_models_arguments,
    add_training_arguments,
    build_loss_composer,
    build_training_metadata,
    finalize_metadata,
    get_huber_threshold_if_needed,
    get_optimizer,
    get_optimizer_train_eval_fn,
    get_scheduler_fix,
    save_state_on_train_end,
    verify_command_line_training_args,
    verify_training_args,
)
from library.training.loop import build_loop_state, run_training_loop
from library.training.log_dispatch import dispatch_logs
from library.training.progress import ProgressSink, run_scope
from library.training.router_conditioning import apply_router_conditioning
from library.training.text_conds import prepare_text_conds
from library.training.forward_kwargs import build_forward_kwargs
from library.training.inversion_forward import compute_inversion_func_loss
from library.training.vr_forward import run_vr_reference_forward
from library.log import setup_logging, add_logging_arguments

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


class AnimaTrainer:
    def __init__(self):
        self.sample_prompts_te_outputs = None
        self._padding_mask_cache = {}
        # Per-method extensions (EasyControl, IP-Adapter, …). Resolved
        # from args+network in train() right after _create_and_apply_network.
        self._adapters: list[MethodAdapter] = []
        # Feature-specific per-run state — see ``RuntimeState``.
        self._state = RuntimeState()

    # region logging helpers

    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        mean_grad_norm=None,
        mean_combined_norm=None,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/max_key_norm"] = maximum_norm
        if mean_norm is not None:
            logs["norm/avg_key_norm"] = mean_norm
        if mean_grad_norm is not None:
            logs["norm/avg_grad_norm"] = mean_grad_norm
        if mean_combined_norm is not None:
            logs["norm/avg_combined_norm"] = mean_combined_norm

        if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0:
            lambda_ema = self._state.vr.get("lambda_ema")
            lambda_batch = self._state.vr.get("lambda_batch")
            if isinstance(lambda_ema, float):
                logs["vr/lambda_ema"] = lambda_ema
            if isinstance(lambda_batch, float):
                logs["vr/lambda_batch"] = lambda_batch

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if args.network_train_unet_only else -1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{idx}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            if (
                args.optimizer_type.lower().startswith("DAdapt".lower())
                or args.optimizer_type.lower() == "Prodigy".lower()
            ):
                # tracking d*lr value
                logs[f"lr/d*lr/{lr_desc}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"]
                    * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )
            if (
                args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower())
                and optimizer is not None
            ):  # tracking d*lr value of unet.
                logs["lr/d*lr"] = (
                    optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
                )
        else:
            idx = 0
            if not args.network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if (
                    args.optimizer_type.lower().startswith("DAdapt".lower())
                    or args.optimizer_type.lower() == "Prodigy".lower()
                ):
                    logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"]
                        * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )
                if (
                    args.optimizer_type.lower().endswith(
                        "ProdigyPlusScheduleFree".lower()
                    )
                    and optimizer is not None
                ):
                    logs[f"lr/d*lr/group{i}"] = (
                        optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]
                    )

        return logs

    def step_logging(
        self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int
    ):
        dispatch_logs(
            accelerator,
            logs,
            global_step,
            global_step,
            epoch,
            progress_sink=getattr(self, "progress_sink", None),
        )

    def epoch_logging(
        self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int
    ):
        dispatch_logs(
            accelerator,
            logs,
            epoch,
            global_step,
            epoch,
            progress_sink=getattr(self, "progress_sink", None),
        )

    def val_logging(
        self,
        accelerator: Accelerator,
        logs: dict,
        global_step: int,
        epoch: int,
        val_step: int,
    ):
        dispatch_logs(
            accelerator,
            logs,
            global_step + val_step,
            global_step,
            epoch,
            val_step,
            progress_sink=getattr(self, "progress_sink", None),
        )

    # endregion

    # region Anima-specific methods (from AnimaNetworkTrainer overrides)

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[DatasetGroup, MinimalDataset],
        val_dataset_group: Optional[DatasetGroup],
    ):
        # use_text_cache → cache_text_encoder_outputs{,_to_disk} is expanded in
        # verify_training_args (runs first); just read the derived flag here.
        if args.cache_text_encoder_outputs:
            assert train_dataset_group.is_text_encoder_output_cacheable(
                cache_supports_dropout=True
            ), (
                "when caching Text Encoder output, token_warmup_step or caption_tag_dropout_rate cannot be used"
            )
            if getattr(args, "cache_llm_adapter_outputs", False):
                # Adapter output caching is only valid when the adapter is frozen (no LoRA on adapter).
                if args.network_args is not None and any(
                    "train_llm_adapter" in a and "true" in a.lower()
                    for a in args.network_args
                ):
                    raise ValueError(
                        "--cache_llm_adapter_outputs is incompatible with --network_args train_llm_adapter=True"
                    )
        else:
            assert not getattr(args, "cache_llm_adapter_outputs", False), (
                "--cache_llm_adapter_outputs requires --cache_text_encoder_outputs"
            )

        assert args.network_train_unet_only or not args.cache_text_encoder_outputs, (
            "network for Text Encoder cannot be trained with caching Text Encoder outputs"
        )

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, (
            "blocks_to_swap is not supported with cpu_offload_checkpointing"
        )

        if args.unsloth_offload_checkpointing:
            if not args.gradient_checkpointing:
                logger.warning(
                    "unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled"
                )
                args.gradient_checkpointing = True
            assert not args.cpu_offload_checkpointing, (
                "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"
            )
            assert args.blocks_to_swap is None or args.blocks_to_swap == 0, (
                "blocks_to_swap is not supported with unsloth_offload_checkpointing"
            )

        # Propagate inversion_dir to datasets for functional-loss supervision (postfix-func).
        inversion_dir = getattr(args, "inversion_dir", None)
        if inversion_dir:
            num_runs = getattr(args, "functional_loss_num_runs", 3)
            for dataset in train_dataset_group.datasets:
                dataset.inversion_dir = inversion_dir
                dataset.inversion_num_runs = num_runs
            if val_dataset_group is not None:
                for dataset in val_dataset_group.datasets:
                    dataset.inversion_dir = inversion_dir
                    dataset.inversion_num_runs = num_runs

        # Propagate IP-Adapter feature-cache flag so datasets load
        # {stem}_anima_{encoder}.safetensors sidecars into batch["ip_features"].
        if getattr(args, "ip_features_cache_to_disk", False):
            ip_encoder = getattr(args, "ip_encoder", "pe")
            for dataset in train_dataset_group.datasets:
                dataset.ip_features_cache_to_disk = True
                dataset.ip_features_encoder = ip_encoder
            if val_dataset_group is not None:
                for dataset in val_dataset_group.datasets:
                    dataset.ip_features_cache_to_disk = True
                    dataset.ip_features_encoder = ip_encoder

        # IP-Adapter live PE encoding (PE-LoRA, or no cached features) needs
        # batch["images"] every step. With cache_latents=true the dataset
        # would normally skip image loading; this flag forces it to keep
        # decoding the source image alongside the cached latent so the live
        # PE forward has its input. VAE encoding still runs from cache.
        if getattr(args, "use_ip_adapter", False) and not getattr(
            args, "ip_features_cache_to_disk", False
        ):
            for dataset in train_dataset_group.datasets:
                dataset.force_load_images_for_ip = True
            if val_dataset_group is not None:
                for dataset in val_dataset_group.datasets:
                    dataset.force_load_images_for_ip = True

        # IP-Adapter distinct-pair (identity) training. When opted in
        # (ip_pair_mode != "self") each dataset draws the IP-path reference from
        # a *different* image of the target's identity instead of the target
        # itself, removing the self-pair copy shortcut. Requires cached PE
        # features (the pairing is a stem swap on disk). See
        # docs/proposal/ip-adapter-identity-pairs.md.
        ip_pair_mode = str(getattr(args, "ip_pair_mode", "self") or "self")
        if getattr(args, "use_ip_adapter", False) and ip_pair_mode != "self":
            if not getattr(args, "ip_features_cache_to_disk", False):
                raise ValueError(
                    "ip_pair_mode requires ip_features_cache_to_disk=true "
                    "(distinct-pair training swaps which stem's cached PE "
                    "features feed the IP path). PE-LoRA's live encoder is "
                    "incompatible — set pe_lora_enabled=false."
                )
            index_path = getattr(
                args,
                "ip_pair_index",
                "post_image_dataset/captions/caption_index.json",
            )
            if not os.path.exists(index_path):
                raise FileNotFoundError(
                    f"ip_pair_index not found: {index_path}. Run `make caption-index`."
                )
            pair_kwargs = dict(
                index_path=index_path,
                mode=ip_pair_mode,
                prob=float(getattr(args, "ip_pair_prob", 0.8)),
                min_level=str(getattr(args, "ip_pair_min_level", "artist")),
                caption_strip_p=float(getattr(args, "ip_pair_caption_strip_p", 0.0)),
            )
            for dataset in train_dataset_group.datasets:
                dataset.setup_identity_pairs(is_validation=False, **pair_kwargs)
            if val_dataset_group is not None:
                for dataset in val_dataset_group.datasets:
                    dataset.setup_identity_pairs(is_validation=True, **pair_kwargs)
            logger.info(
                f"IP-Adapter distinct pairs: mode={ip_pair_mode} "
                f"prob={pair_kwargs['prob']} min_level={pair_kwargs['min_level']} "
                f"caption_strip_p={pair_kwargs['caption_strip_p']} "
                f"index={index_path}"
            )

        # Soft-tokens contrastive negatives. The objective's knobs live in
        # ``network_args`` (see configs/methods/soft_tokens.toml); preview them
        # here to decide whether
        # the dataset should surface cached negative text embeddings. Off unless
        # contrastive_weight > 0. See docs/proposal/soft_tokens_contrastive.md.
        if str(getattr(args, "network_module", "") or "") == (
            "networks.methods.soft_tokens"
        ):
            net_arg_preview: dict[str, str] = {}
            for na in args.network_args or []:
                if "=" in na:
                    pk, pv = na.split("=", 1)
                    net_arg_preview[pk] = pv
            con_weight = float(net_arg_preview.get("contrastive_weight", 0.0) or 0.0)
            if con_weight > 0.0:
                con_k = int(net_arg_preview.get("contrastive_k", 1) or 1)
                con_mode = str(
                    net_arg_preview.get("contrastive_negative_mode", "shuffled")
                )
                # The negative grouping always comes from the shared caption
                # index `make caption-index` writes — not a user knob.
                con_index = "post_image_dataset/captions/caption_index.json"
                if not os.path.exists(con_index):
                    raise FileNotFoundError(
                        f"contrastive_index not found: {con_index}. "
                        f"Run `make caption-index`."
                    )
                if not getattr(args, "cache_llm_adapter_outputs", False):
                    raise ValueError(
                        "soft_tokens contrastive requires "
                        "cache_llm_adapter_outputs=true (negatives are cached "
                        "crossattn_emb swapped off disk)."
                    )
                # Negatives only feed the training-step contrastive forward; the
                # validation FM-MSE stays a clean baseline, so val datasets are
                # left untouched.
                for dataset in train_dataset_group.datasets:
                    dataset.setup_contrastive_negatives(
                        con_index, k=con_k, mode=con_mode, is_validation=False
                    )
                logger.info(
                    f"Soft-tokens contrastive: weight={con_weight} k={con_k} "
                    f"mode={con_mode} index={con_index}"
                )

    def load_target_model(
        self, args, weight_dtype, accelerator, load_qwen3=True, load_vae=True
    ):
        self.is_swapping_blocks = (
            args.blocks_to_swap is not None and args.blocks_to_swap > 0
        )

        # Load Qwen3 text encoder (tokenizers already loaded in get_tokenize_strategy).
        # Skipped when every text-encoder output is already cached and no live
        # encoding (sampling / TE training / cache disabled) needs it.
        if load_qwen3:
            logger.info("Loading Qwen3 text encoder...")
            qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(
                args.qwen3, dtype=weight_dtype, device="cpu"
            )
            qwen3_text_encoder.eval()
        else:
            logger.info(
                "Skipping Qwen3 text encoder load: all text-encoder outputs cached."
            )
            qwen3_text_encoder = None

        # Load VAE. Skipped when every latent is already cached and no sampling
        # (which decodes latents) is configured.
        if load_vae:
            logger.info("Loading Anima VAE...")
            vae = qwen_image_autoencoder_kl.load_vae(
                args.vae,
                device="cpu",
                disable_mmap=True,
                spatial_chunk_size=args.vae_chunk_size,
                disable_cache=args.vae_disable_cache,
            )
            vae.to(weight_dtype)
            vae.eval()
        else:
            logger.info("Skipping VAE load: all latents cached and no sampling.")
            vae = None

        # Return format: (model_type, text_encoders, vae, unet)
        return "anima", [qwen3_text_encoder], vae, None  # unet loaded lazily

    def load_unet_lazily(
        self, args, weight_dtype, accelerator, text_encoders
    ) -> tuple[nn.Module, list[nn.Module]]:
        loading_dtype = weight_dtype
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device

        attn_mode = "torch"
        if args.xformers:
            attn_mode = "xformers"
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        if attn_mode == "flash4":
            # Flash Attention 4 (flash-attention-sm120) is not supported yet.
            raise RuntimeError(
                "attn_mode='flash4' is not supported yet -- the flash-attention-sm120 "
                "kernel is disabled in this build. Use 'flash', 'torch', 'flex', "
                "'sageattn', or 'xformers' instead."
            )
        elif attn_mode == "flash":
            from networks.attention_dispatch import flash_attn, flash_attn_func

            if flash_attn_func is not None:
                logger.info(
                    f"Using Flash Attention 2 (flash_attn {flash_attn.__version__})"
                )
            else:
                raise RuntimeError(
                    "attn_mode='flash' requested but flash_attn is not available."
                )
        else:
            logger.info(f"Using attention mode: {attn_mode}")

        # Frozen LoRA: merged into DiT weights at load time (no runtime hooks).
        # Used by postfix runs that train on top of a fixed LoRA.
        lora_weights_list = None
        lora_multipliers = None
        if getattr(args, "lora_path", None):
            from safetensors.torch import load_file

            logger.info(
                f"merging frozen LoRA from {args.lora_path} into DiT weights "
                f"(multiplier={args.lora_multiplier})"
            )
            lora_sd = load_file(args.lora_path)
            lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
            lora_weights_list = [lora_sd]
            lora_multipliers = [args.lora_multiplier]

        # Load DiT
        attn_softmax_scale = getattr(args, "attn_softmax_scale", None)
        logger.info(
            f"Loading Anima DiT model with attn_softmax_scale: {attn_softmax_scale}..."
        )
        model = anima_utils.load_anima_model(
            accelerator.device,
            args.pretrained_model_name_or_path,
            attn_mode,
            loading_device,
            loading_dtype,
            lora_weights_list=lora_weights_list,
            lora_multipliers=lora_multipliers,
            attn_softmax_scale=attn_softmax_scale,
        )

        # Native-shape flattening + per-block torch.compile. compile_blocks turns
        # on the flatten (one block graph per token-count family: 4032/4200) and
        # raises the dynamo cache-size budget itself.
        if args.torch_compile:
            model.compile_blocks(
                args.dynamo_backend,
                mode=getattr(args, "compile_inductor_mode", None),
            )

        # Store unsloth preference so that when the base trainer calls
        # dit.enable_gradient_checkpointing(cpu_offload=...), we can override to use unsloth.
        self._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing

        # Block swap
        self.is_swapping_blocks = (
            args.blocks_to_swap is not None and args.blocks_to_swap > 0
        )
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        # Variance-reduced FM loss: the "frozen reference" is the trainable
        # DiT itself with ``network.set_multiplier(0)`` during the no-grad
        # forward — works because base weights are frozen and LoRA-family
        # adapters are additive. See ``get_noise_pred_and_target`` for the
        # bypass. Saves ~5 GB VRAM vs holding a second DiT copy.
        if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0:
            logger.info(
                f"VR loss enabled (vr_loss_weight={args.vr_loss_weight}); "
                f"using trainable DiT with multiplier=0 as the control variate"
            )

        return model, text_encoders

    def get_tokenize_strategy(self, args):
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.qwen3,
            t5_tokenizer_path=args.t5_tokenizer_path,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
        return tokenize_strategy

    def get_tokenizers(self, tokenize_strategy: strategy_anima.AnimaTokenizeStrategy):
        return [tokenize_strategy.qwen3_tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_anima.AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )

    def get_text_encoding_strategy(self, args):
        return strategy_anima.AnimaTextEncodingStrategy()

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
                False,
                cache_llm_adapter_outputs=getattr(
                    args, "cache_llm_adapter_outputs", False
                ),
                use_shuffled_caption_variants=getattr(
                    args, "use_shuffled_caption_variants", False
                ),
            )
        return None

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None  # no text encoders needed for encoding
        return text_encoders

    def _ensure_uncond_crossattn(
        self,
        args: argparse.Namespace,
        accelerator,
        weight_dtype: torch.dtype,
    ) -> None:
        """Lazily load the T5("") crossattn sidecar onto ``self._state.uncond_crossattn_1``.

        Primary producer is ``make preprocess-te`` (drops the file at
        ``post_image_dataset/_anima_uncond_te.safetensors``); this method is
        the fallback that stages on demand if a training run was kicked off
        without the preprocess step.
        """
        if self._state.uncond_crossattn_1 is not None:
            return
        from library.inference.uncond import (
            DEFAULT_UNCOND_DIR,
            default_uncond_path,
            load_uncond_crossattn,
            stage_uncond_sidecar,
        )

        sidecar = default_uncond_path()
        if not sidecar.exists():
            logger.info(
                f"T5('') uncond sidecar missing at {sidecar} — staging "
                f"on demand (would normally be produced by `make preprocess-te`)."
            )
            stage_uncond_sidecar(
                DEFAULT_UNCOND_DIR,
                qwen3_path=args.qwen3,
                dit_path=args.pretrained_model_name_or_path,
                t5_tokenizer_path=getattr(args, "t5_tokenizer_path", None),
                seq_len=512,
                overwrite=False,
            )
        self._state.uncond_crossattn_1 = load_uncond_crossattn(
            str(sidecar), device=accelerator.device, dtype=weight_dtype
        )
        logger.info(
            f"caption dropout uncond loaded: {sidecar} "
            f"shape={tuple(self._state.uncond_crossattn_1.shape)}"
        )

    def get_noise_scheduler(
        self, args: argparse.Namespace, device: torch.device
    ) -> Any:
        noise_scheduler = noise_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=args.discrete_flow_shift
        )
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage
        return vae.encode_pixels_to_latents(images)  # Keep 4D for input/output

    def shift_scale_latents(self, args, latents):
        # Latents already normalized by vae.encode with scale
        return latents

    def get_noise_pred_and_target(
        self,
        ctx: TrainCtx,
        latents,
        batch,
        text_encoder_conds,
        *,
        is_train=True,
    ):
        args = ctx.args
        accelerator = ctx.accelerator
        noise_scheduler = ctx.noise_scheduler
        unet = ctx.unet
        network = ctx.network
        weight_dtype = ctx.weight_dtype
        anima: anima_models.Anima = unet

        # Reset per-step adapter aux so stale tensors from a prior step can't
        # leak into the loss composer.
        self._state.extras_for_step = {}

        # Sample noise
        if latents.ndim == 5:  # Fallback for 5D latents (old cache)
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]

        # Method-adapter pre-forward priming. IP-Adapter encodes the reference
        # image and primes per-block K/V; EasyControl runs the cond pre-pass
        # and primes per-block (K_c, V_c). Both run on the 4D latent layout
        # the patched DiT forward expects. The patched cross-attn / self-attn
        # closures consume the primed tensors during attention.
        if self._adapters:
            step_ctx = StepCtx(
                args=args,
                accelerator=accelerator,
                network=network,
                weight_dtype=weight_dtype,
            )
            for adapter in self._adapters:
                adapter.prime_for_forward(step_ctx, batch, latents, is_train=is_train)
        noise = torch.randn_like(latents)

        # Draw noisy input + timesteps via the sampler registry (M1).
        sampler_fn = SAMPLER_REGISTRY[getattr(args, "sampler", "default") or "default"]
        sampler_out = sampler_fn(
            SamplerContext(
                args=args,
                noise_scheduler=noise_scheduler,
                latents=latents,
                noise=noise,
                device=accelerator.device,
                weight_dtype=weight_dtype,
            )
        )
        noisy_model_input = sampler_out.noisy_input
        timesteps = sampler_out.timesteps  # [0,1]-scaled, float32
        sigmas = sampler_out.sigmas

        # Per-step network conditioning: timestep masks, σ/FEI routers, balance-loss warmup.
        self._hydra_warmup_step = apply_router_conditioning(
            network=network,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            is_train=is_train,
            warmup_step=int(getattr(self, "_hydra_warmup_step", 0)),
            max_train_steps=int(getattr(args, "max_train_steps", 0) or 0),
            gradient_accumulation_steps=int(
                getattr(args, "gradient_accumulation_steps", 1) or 1
            ),
        )

        # Gradient checkpointing support
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            # Only require grads for text conditions when training the text encoder.
            # When using cached text encoder outputs (or training DiT-only), requiring grads here adds backward work.
            if self.is_train_text_encoder(args) and not args.cache_text_encoder_outputs:
                for t in text_encoder_conds:
                    if t is not None and t.dtype.is_floating_point:
                        t.requires_grad_(True)

        # Unpack text encoder conditions, H2D move, and on-device caption dropout.
        tc = prepare_text_conds(
            text_encoder_conds=text_encoder_conds,
            batch=batch,
            text_encoding_strategy=ctx.text_encoding_strategy,
            network=network,
            device=accelerator.device,
            weight_dtype=weight_dtype,
            uncond_crossattn_emb=self._state.uncond_crossattn_1,
        )
        crossattn_emb = tc.crossattn_emb
        prompt_embeds = tc.prompt_embeds
        attn_mask = tc.attn_mask
        t5_input_ids = tc.t5_input_ids
        t5_attn_mask = tc.t5_attn_mask

        # ChimeraHydra global content router (chimera with
        # ``content_router_source="crossattn"``): fire ONCE per step on the
        # pooled crossattn_emb. apply_router_conditioning above ran before
        # text conds were materialized, so the content router lives outside
        # that helper. No-op on non-chimera networks or per-Linear chimera.
        if (
            getattr(network, "use_content_router", False)
            and crossattn_emb is not None
            and hasattr(network, "set_content")
        ):
            network.set_content(crossattn_emb)

        # Network-level GlobalRouter routed on pooled text
        # (``router_source="crossattn_emb"``, route_per_layer=False). Same
        # timing rationale as the content router above — fires once per step
        # on the materialized cross-attn text features. No-op otherwise.
        if (
            getattr(network, "use_crossattn_router", False)
            and crossattn_emb is not None
            and hasattr(network, "set_crossattn_routing")
        ):
            network.set_crossattn_routing(crossattn_emb)

        # Create padding mask
        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask_key = (bs, h_latent, w_latent, weight_dtype, accelerator.device)
        padding_mask = self._padding_mask_cache.get(padding_mask_key)
        if padding_mask is None:
            padding_mask = torch.zeros(
                bs, 1, h_latent, w_latent, dtype=weight_dtype, device=accelerator.device
            )
            self._padding_mask_cache[padding_mask_key] = padding_mask

        # Call model
        noisy_model_input = noisy_model_input.unsqueeze(
            2
        )  # 4D to 5D, [B, C, H, W] -> [B, C, 1, H, W]

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            if crossattn_emb is None:
                model_pred = anima(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    padding_mask=padding_mask,
                    target_input_ids=t5_input_ids,
                    target_attention_mask=t5_attn_mask,
                    source_attention_mask=attn_mask,
                )
            else:
                # crossattn_emb is already in target (T5-compatible) space.
                # Postfix splice kwargs.
                fk = build_forward_kwargs(
                    network=network,
                    crossattn_emb=crossattn_emb,
                    t5_attn_mask=t5_attn_mask,
                    timesteps=timesteps,
                )
                crossattn_emb = fk.crossattn_emb
                kw = fk.kw
                has_postfix = fk.has_postfix
                model_pred = anima(
                    noisy_model_input,
                    timesteps,
                    crossattn_emb,
                    padding_mask=padding_mask,
                    **kw,
                )

                # Method-adapter extra forwards (soft-tokens, …).
                # Each adapter sees the primary forward's inputs + 5D output
                # and may run additional anima(...) calls inside this same
                # autocast / grad scope, returning aux loss tensors keyed for
                # the LossComposer.
                if self._adapters:
                    primary = ForwardArtifacts(
                        anima_call=anima,
                        noisy_model_input=noisy_model_input,
                        timesteps=timesteps,
                        crossattn_emb=crossattn_emb,
                        padding_mask=padding_mask,
                        forward_kwargs=kw,
                        model_pred=model_pred,
                        noise=noise,
                        latents=latents,
                        is_train=is_train,
                    )
                    step_ctx = StepCtx(
                        args=args,
                        accelerator=accelerator,
                        network=network,
                        weight_dtype=weight_dtype,
                    )
                    for adapter in self._adapters:
                        out = adapter.extra_forwards(step_ctx, primary)
                        if out:
                            self._state.extras_for_step.update(out)

                # Functional MSE loss against a sampled stochastic inversion run.
                # The captures dict is populated by trainer-owned forward hooks
                # on cross_attn.output_proj at ``self._func_blocks``.
                self._func_loss = None
                if is_train and getattr(self, "_func_blocks", None):
                    self._func_loss = compute_inversion_func_loss(
                        anima_call=anima,
                        captures=self._func_captures,
                        block_indices=self._func_blocks,
                        batch=batch,
                        noisy_model_input=noisy_model_input,
                        timesteps=timesteps,
                        padding_mask=padding_mask,
                        has_postfix=has_postfix,
                        kw=kw,
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )

                # Variance-reduced FM control variate (AsymFlow §5.2). Stash the
                # residual `z` so the loss composer can blend `(y + λ·z)²`.
                if (
                    is_train
                    and float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0
                ):
                    z_residual = run_vr_reference_forward(
                        anima_call=anima,
                        network=network,
                        latents=latents,
                        noise=noise,
                        sigmas=sigmas,
                        timesteps=timesteps,
                        crossattn_emb=crossattn_emb,
                        padding_mask=padding_mask,
                        forward_kwargs=kw,
                        weight_dtype=weight_dtype,
                        fei_sigma_low_div=float(args.vr_fei_sigma_low_div),
                    )
                    self._state.extras_for_step["vr"] = {
                        "z": z_residual.detach(),
                        "state": self._state.vr,
                    }
        model_pred = model_pred.squeeze(2)  # 5D to 4D, [B, C, 1, H, W] -> [B, C, H, W]

        # Note: do NOT clear timestep mask here -- gradient checkpointing recomputes the forward
        # pass during backward, so the mask must remain set. It gets overwritten on the next step.

        # Rectified flow target: noise - latents
        target = noise - latents

        # Loss weighting
        weighting = anima_train_utils.compute_loss_weighting_for_anima(
            weighting_scheme=args.weighting_scheme, sigmas=sigmas
        )

        return model_pred, target, timesteps, weighting

    def sample_images(
        self,
        accelerator,
        args,
        epoch,
        global_step,
        device,
        vae,
        tokenizer,
        text_encoder,
        unet,
    ):
        text_encoders = (
            text_encoder if isinstance(text_encoder, list) else [text_encoder]
        )  # compatibility
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        qwen3_te = te[0] if te is not None else None

        text_encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()
        tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            qwen3_te,
            tokenize_strategy,
            text_encoding_strategy,
            self.sample_prompts_te_outputs,
        )

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        # Re-apply with unsloth_offload if needed (after base has already enabled it).
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        if not self.is_swapping_blocks:
            return accelerator.prepare(unet)

        model = unet
        model = accelerator.prepare(
            model, device_placement=[not self.is_swapping_blocks]
        )
        accelerator.unwrap_model(model).move_to_device_except_swap_blocks(
            accelerator.device
        )
        accelerator.unwrap_model(model).prepare_block_swap_before_forward()

        return model

    def on_validation_step_end(self, ctx: TrainCtx, batch):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            ctx.accelerator.unwrap_model(ctx.unet).prepare_block_swap_before_forward()

    def process_batch(
        self,
        ctx: TrainCtx,
        batch,
        *,
        is_train=True,
    ) -> torch.Tensor:
        """Override base process_batch to surface caption_dropout_rates for on-device dropout."""

        # The cached text-encoder outputs list arrives as
        # [..., caption_dropout_rates] from the dataset (see strategy.py
        # cache layout). Split the trailing rates tensor off so the inner
        # path sees the canonical 4- or 5-element conds list, and stash the
        # rates on the batch -- get_noise_pred_and_target applies the dropout
        # in-place after the H2D transfer. Doing it here on CPU would clone
        # prompt_embeds / crossattn_emb on the critical path before the H2D
        # copy, blocking the main thread.
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            caption_dropout_rates = text_encoder_outputs_list[-1]
            encoder_outputs = text_encoder_outputs_list[:-1]
            # Shallow copy so the original list (with rates appended) stays
            # intact for validation's per-sigma loop that reuses the batch.
            batch = {
                **batch,
                "text_encoder_outputs_list": encoder_outputs,
                "caption_dropout_rates": caption_dropout_rates,
            }

        return self._process_batch_inner(ctx, batch, is_train=is_train)

    def _process_batch_inner(
        self,
        ctx: TrainCtx,
        batch,
        *,
        is_train=True,
    ) -> torch.Tensor:
        """
        Process a batch for the network (original NetworkTrainer.process_batch logic)
        """
        args = ctx.args
        accelerator = ctx.accelerator
        network = ctx.network
        vae = ctx.vae
        text_encoders = ctx.text_encoders
        text_encoding_strategy = ctx.text_encoding_strategy
        tokenize_strategy = ctx.tokenize_strategy
        noise_scheduler = ctx.noise_scheduler
        vae_dtype = ctx.vae_dtype
        weight_dtype = ctx.weight_dtype
        train_text_encoder = ctx.train_text_encoder
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(
                    torch.FloatTensor, batch["latents"].to(accelerator.device)
                )
            else:
                if (
                    args.vae_batch_size is None
                    or len(batch["images"]) <= args.vae_batch_size
                ):
                    latents = self.encode_images_to_latents(
                        args,
                        vae,
                        batch["images"].to(accelerator.device, dtype=vae_dtype),
                    )
                else:
                    chunks = [
                        batch["images"][i : i + args.vae_batch_size]
                        for i in range(0, len(batch["images"]), args.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        with torch.no_grad():
                            chunk = self.encode_images_to_latents(
                                args, vae, chunk.to(accelerator.device, dtype=vae_dtype)
                            )
                            list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(
                        torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents)
                    )

            latents = self.shift_scale_latents(args, latents)

        text_encoder_conds = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = (
                text_encoder_outputs_list  # List of text encoder outputs
            )

        if (
            len(text_encoder_conds) == 0
            or text_encoder_conds[0] is None
            or train_text_encoder
        ):
            with (
                torch.set_grad_enabled(is_train and train_text_encoder),
                accelerator.autocast(),
            ):
                if args.weighted_captions:
                    input_ids_list, weights_list = (
                        tokenize_strategy.tokenize_with_weights(batch["captions"])
                    )
                    encoded_text_encoder_conds = (
                        text_encoding_strategy.encode_tokens_with_weights(
                            tokenize_strategy,
                            self.get_models_for_text_encoding(
                                args, accelerator, text_encoders
                            ),
                            input_ids_list,
                            weights_list,
                        )
                    )
                else:
                    input_ids = [
                        ids.to(accelerator.device) for ids in batch["input_ids_list"]
                    ]
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                        tokenize_strategy,
                        self.get_models_for_text_encoding(
                            args, accelerator, text_encoders
                        ),
                        input_ids,
                    )
                if args.full_fp16:
                    encoded_text_encoder_conds = [
                        c.to(weight_dtype) for c in encoded_text_encoder_conds
                    ]

            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        # sample noise, call unet, get target
        noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(
            ctx,
            latents,
            batch,
            text_encoder_conds,
            is_train=is_train,
        )

        huber_c = get_huber_threshold_if_needed(args, timesteps, noise_scheduler)

        # Assemble aux dict for the composer: extra_forwards returns from each
        # method adapter plus the trainer-owned functional-loss capture.
        loss_aux: dict = dict(self._state.extras_for_step)

        func_loss = getattr(self, "_func_loss", None)
        if func_loss is not None:
            loss_aux["func_loss"] = func_loss

        composer = build_loss_composer(args, getattr(self, "_network", network))

        def _build_loss_ctx(aux: dict) -> LossContext:
            return LossContext(
                args=args,
                batch=batch,
                model_pred=noise_pred,
                target=target,
                timesteps=timesteps,
                weighting=weighting,
                huber_c=huber_c,
                loss_weights=batch["loss_weights"],
                network=getattr(self, "_network", network),
                aux=aux,
                is_train=is_train,
            )

        return composer.compose(_build_loss_ctx(loss_aux))

    # endregion

    # region Methods only in NetworkTrainer (not overridden by Anima)

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        self._network = (
            network  # composer reads _network for ortho / balance regularizers
        )
        self._func_loss = None
        self._func_hooks = []
        self._func_captures = {}
        self._func_blocks = []
        if getattr(args, "functional_loss_weight", 0.0) > 0.0 and getattr(
            args, "inversion_dir", None
        ):
            blocks_str = getattr(args, "functional_loss_blocks", "8,12,16,20")
            try:
                self._func_blocks = sorted(
                    int(b.strip()) for b in blocks_str.split(",") if b.strip()
                )
            except ValueError as e:
                raise ValueError(
                    f"functional_loss_blocks must be comma-separated integers, got {blocks_str!r}"
                ) from e

            def _make_hook(block_idx: int):
                def _hook(_module, _inputs, output):
                    # Save the cross_attn.output_proj output for this block.
                    # Hook fires twice per step (main forward + inversion forward);
                    # the main forward runs first, we snapshot before second forward overwrites.
                    self._func_captures[block_idx] = output

                return _hook

            blocks_list = unet.blocks  # nn.ModuleList of 28 Anima DiT blocks
            num_blocks = len(blocks_list)
            for bi in self._func_blocks:
                if not (0 <= bi < num_blocks):
                    raise ValueError(
                        f"functional_loss_blocks contains out-of-range index {bi} (model has {num_blocks} blocks)"
                    )
                module = blocks_list[bi].cross_attn.output_proj
                h = module.register_forward_hook(_make_hook(bi))
                self._func_hooks.append(h)
            logger.info(
                f"Functional loss enabled: hooks on cross_attn.output_proj at blocks {self._func_blocks}, "
                f"weight={args.functional_loss_weight}, num_runs={args.functional_loss_num_runs}"
            )

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec_dataclass(
            args, lora=True
        ).to_metadata_dict()

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # Set first parameter's requires_grad to True to workaround Accelerate gradient checkpointing bug
        first_param = next(text_encoder.parameters())
        first_param.requires_grad_(True)

    def get_text_encoders_train_flags(self, args, text_encoders):
        return (
            [True] * len(text_encoders)
            if self.is_train_text_encoder(args)
            else [False] * len(text_encoders)
        )

    def on_step_start(self, ctx: TrainCtx, batch, *, is_train: bool = True):
        if not self._adapters:
            return
        step_ctx = StepCtx(
            args=ctx.args,
            accelerator=ctx.accelerator,
            network=ctx.network,
            weight_dtype=ctx.weight_dtype,
        )
        for adapter in self._adapters:
            adapter.on_step_start(step_ctx, batch, is_train=is_train)

    def run_after_backward(self, ctx: TrainCtx):
        """Dispatch the post-backward hook to adapters (between
        ``accelerator.backward`` and gradient clipping)."""
        if not self._adapters:
            return
        step_ctx = StepCtx(
            args=ctx.args,
            accelerator=ctx.accelerator,
            network=ctx.network,
            weight_dtype=ctx.weight_dtype,
        )
        for adapter in self._adapters:
            adapter.after_backward(step_ctx)

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only

    def cast_text_encoder(self, args):
        return True

    def cast_vae(self, args):
        return True

    def cast_unet(self, args):
        return True

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
        **kwargs,
    ):
        noise_pred = unet(noisy_latents, timesteps, text_conds[0]).sample
        return noise_pred

    def cache_text_encoder_outputs_if_needed(
        self,
        args,
        accelerator: Accelerator,
        text_encoders,
        dataset: DatasetGroup,
    ):
        if not args.cache_text_encoder_outputs:
            # Live-encoding mode (e.g. IP-Adapter cache_text_encoder_outputs=false):
            # move the text encoder to device for per-step encoding.
            text_encoders[0].to(accelerator.device)
            return

        # With caching on, the on-disk cache is guaranteed complete (asserted in
        # train(), including the LLM adapter's crossattn_emb outputs, which
        # preprocess writes). The dataset thus never needs encoding here — run
        # the pass with no model purely to populate
        # ImageInfo.text_encoder_outputs_npz (forms no batches).
        dataset.new_cache_text_encoder_outputs([None], accelerator)

        # The text encoder is in memory only to encode sample prompts (TE
        # training is mutually exclusive with caching). It is None when no
        # sample prompts are configured — nothing left to do.
        if text_encoders[0] is not None and args.sample_prompts is not None:
            logger.info(
                f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}"
            )
            logger.info("move text encoder to gpu")
            text_encoders[0].to(accelerator.device)

            tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
            text_encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()

            prompts = train_util.load_prompts(args.sample_prompts)
            sample_prompts_te_outputs = {}
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [
                        prompt_dict.get("prompt", ""),
                        prompt_dict.get("negative_prompt", ""),
                    ]:
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"  cache TE outputs for: {p}")
                            tokens_and_masks = tokenize_strategy.tokenize(p)
                            sample_prompts_te_outputs[p] = (
                                text_encoding_strategy.encode_tokens(
                                    tokenize_strategy,
                                    text_encoders,
                                    tokens_and_masks,
                                )
                            )
            self.sample_prompts_te_outputs = sample_prompts_te_outputs

            logger.info("move text encoder back to cpu")
            text_encoders[0].to("cpu")
            clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # endregion

    # region Main training loop

    @staticmethod
    def _parse_profile_steps(args) -> tuple[int, int] | None:
        """Parse --profile_steps 'start-end' into (start, end) or None.

        When set, the loop calls ``torch.cuda.profiler.start()`` at ``start``
        and ``stop()`` after ``end``, so pair this with::

            nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
                accelerate launch ... train.py --profile_steps 3-5
        """
        raw = getattr(args, "profile_steps", None)
        if not raw:
            return None
        if "-" in raw:
            a, b = raw.split("-", 1)
            return int(a), int(b)
        n = int(raw)
        return n, n + 2

    @staticmethod
    def _switch_rng_state(
        seed: int,
    ) -> tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]:
        cpu_rng_state = torch.get_rng_state()
        gpu_rng_state = torch.cuda.get_rng_state()
        python_rng_state = random.getstate()

        torch.manual_seed(seed)
        random.seed(seed)

        return (cpu_rng_state, gpu_rng_state, python_rng_state)

    @staticmethod
    def _restore_rng_state(
        rng_states: tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple],
    ):
        cpu_rng_state, gpu_rng_state, python_rng_state = rng_states
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(gpu_rng_state)
        random.setstate(python_rng_state)

    def _prepare_dataset(self, args):
        """Build train/val dataset groups and the collator shared by both loaders."""
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(
                ConfigSanitizer(support_dropout=True)
            )
            if use_user_config:
                logger.info(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    logger.warning(
                        "ignoring the following options because config file is found: {0}".format(
                            ", ".join(ignored)
                        )
                    )
            else:
                base_ds = load_dataset_config_from_base(
                    overrides=vars(args),
                    method=getattr(args, "method", None),
                    methods_subdir=getattr(args, "methods_subdir", None) or "methods",
                )
                if base_ds is not None:
                    logger.info("Loading dataset config from configs/base.toml")
                    user_config = base_ds
                    use_user_config = True
                elif use_dreambooth_method:
                    logger.info("Using DreamBooth method.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                    args.train_data_dir, args.reg_data_dir
                                )
                            }
                        ]
                    }
                else:
                    logger.info("Training with captions.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": [
                                    {
                                        "image_dir": args.train_data_dir,
                                        "metadata_file": args.in_json,
                                    }
                                ]
                            }
                        ]
                    }

            # Global --sample_ratio override (used by the `[half]` preset).
            sample_ratio = getattr(args, "sample_ratio", None)
            if sample_ratio is not None:
                for ds in user_config.get("datasets", []):
                    for sub in ds.get("subsets", []):
                        sub["sample_ratio"] = sample_ratio
                logger.info(f"Applied --sample_ratio={sample_ratio} to all subsets")

            blueprint = blueprint_generator.generate(user_config, args)
            train_dataset_group, val_dataset_group = (
                config_util.generate_dataset_group_by_blueprint(
                    blueprint.dataset_group,
                    # Native constant-token bucketing is the only mode: the sampler
                    # buckets into CONSTANT_TOKEN_BUCKETS (the 4032/4200 families)
                    # so compile_blocks' flatten keys on token count, not resolution.
                    constant_token_buckets=True,
                )
            )

            rates = [
                subset.caption_dropout_rate
                for ds in train_dataset_group.datasets
                for subset in ds.subsets
            ]
            self._state.caption_dropout_enabled = bool(rates) and any(
                r > 0 for r in rates
            )
            if self._state.caption_dropout_enabled:
                logger.info(f"caption dropout ENABLED -- per-subset rates: {rates}")
            else:
                logger.info("caption dropout DISABLED (rate=0.0 on all subsets)")
        else:
            # use arbitrary dataset class
            train_dataset_group = load_arbitrary_dataset(args)
            val_dataset_group = (
                None  # placeholder until validation dataset supported for arbitrary
            )

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = (
            train_dataset_group if args.max_data_loader_n_workers == 0 else None
        )
        collator = collator_class(current_epoch, current_step, ds_for_collator)

        return (
            train_dataset_group,
            val_dataset_group,
            current_epoch,
            current_step,
            collator,
            use_user_config,
            use_dreambooth_method,
        )

    def _create_and_apply_network(
        self,
        args,
        accelerator,
        vae,
        text_encoder,
        unet,
        text_encoders,
        weight_dtype,
    ):
        """Import network module, merge base weights, build LoRA, apply to the model."""
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            for i, weight_path in enumerate(args.base_weights):
                if (
                    args.base_weights_multiplier is None
                    or len(args.base_weights_multiplier) <= i
                ):
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(
                    f"merging module: {weight_path} with multiplier {multiplier}"
                )

                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(
                    text_encoder,
                    unet,
                    weights_sd,
                    weight_dtype,
                    accelerator.device if args.lowram else "cpu",
                )

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=", 1)
                net_kwargs[key] = value

        # Forward known network-arg keys from top-level config (TOML) to net_kwargs.
        # CLI --network_args take precedence over top-level config keys.
        # Source of truth: `networks.all_network_kwargs()` (union of
        # `SHARED_KWARG_FLAGS` and each `NetworkSpec.kwarg_flags`), plus a
        # small tail of top-level training args the network modules still
        # want to read (e.g. postfix contrastive's step-boundary window).
        for key in NETWORK_KWARG_ALLOWLIST + _EXTRA_FORWARDED_TOP_LEVEL_ARGS:
            if (
                key not in net_kwargs
                and hasattr(args, key)
                and getattr(args, key) is not None
            ):
                net_kwargs[key] = str(getattr(args, key))

        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(
                1, args.network_weights, vae, text_encoder, unet, **net_kwargs
            )
        else:
            if "dropout" not in net_kwargs:
                net_kwargs["dropout"] = args.network_dropout

            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        if network is None:
            return None

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(
            network, "apply_max_norm_regularization"
        ):
            logger.warning(
                "warning: scale_weight_norms is specified but the network does not support it"
            )
            args.scale_weight_norms = False

        self.post_process_network(args, accelerator, network, text_encoders, unet)

        # apply network to unet and text_encoder
        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            info = network.load_weights(args.network_weights)
            accelerator.print(
                f"load network weights from {args.network_weights}: {info}"
            )

        if args.gradient_checkpointing:
            if args.cpu_offload_checkpointing:
                unet.enable_gradient_checkpointing(cpu_offload=True)
            else:
                unet.enable_gradient_checkpointing()

            for t_enc, flag in zip(
                text_encoders, self.get_text_encoders_train_flags(args, text_encoders)
            ):
                if flag:
                    if t_enc.supports_gradient_checkpointing:
                        t_enc.gradient_checkpointing_enable()
            network.enable_gradient_checkpointing()  # may have no effect

        return network, net_kwargs, train_unet, train_text_encoder

    def _setup_optimizer_and_dataloader(
        self,
        args,
        accelerator,
        network,
        train_dataset_group,
        val_dataset_group,
        collator,
    ):
        """Build optimizer, dataloaders, and LR scheduler; finalize max_train_steps."""
        accelerator.print("prepare optimizer, data loader etc.")

        # make backward compatibility for text_encoder_lr
        support_multiple_lrs = hasattr(
            network, "prepare_optimizer_params_with_multiple_te_lrs"
        )
        if support_multiple_lrs:
            text_encoder_lr = args.text_encoder_lr
        else:
            if (
                args.text_encoder_lr is None
                or isinstance(args.text_encoder_lr, float)
                or isinstance(args.text_encoder_lr, int)
            ):
                text_encoder_lr = args.text_encoder_lr
            else:
                text_encoder_lr = (
                    None if len(args.text_encoder_lr) == 0 else args.text_encoder_lr[0]
                )
        try:
            if support_multiple_lrs:
                results = network.prepare_optimizer_params_with_multiple_te_lrs(
                    text_encoder_lr, args.unet_lr, args.learning_rate
                )
            else:
                results = network.prepare_optimizer_params(
                    text_encoder_lr, args.unet_lr, args.learning_rate
                )
            if type(results) is tuple:
                trainable_params = results[0]
                lr_descriptions = results[1]
            else:
                trainable_params = results
                lr_descriptions = None
        except TypeError:
            trainable_params = network.prepare_optimizer_params(
                text_encoder_lr, args.unet_lr
            )
            lr_descriptions = None

        optimizer_name, optimizer_args, optimizer = get_optimizer(
            args, trainable_params
        )
        optimizer_train_fn, optimizer_eval_fn = get_optimizer_train_eval_fn(
            optimizer, args
        )

        # prepare dataloader
        train_dataset_group.set_current_strategies()
        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()

        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        persistent_workers = args.persistent_data_loader_workers and n_workers > 0

        dataloader_kwargs = {
            "batch_size": 1,
            "collate_fn": collator,
            "num_workers": n_workers,
            "persistent_workers": persistent_workers,
            "pin_memory": args.dataloader_pin_memory,
        }
        if n_workers > 0:
            dataloader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            shuffle=True,
            **dataloader_kwargs,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset_group if val_dataset_group is not None else [],
            shuffle=False,
            **dataloader_kwargs,
        )

        # Calculate training steps
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader)
                / accelerator.num_processes
                / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is"
            )

        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # lr scheduler
        lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

        return (
            optimizer,
            optimizer_name,
            optimizer_args,
            optimizer_train_fn,
            optimizer_eval_fn,
            text_encoder_lr,
            lr_descriptions,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        )

    def _prepare_with_accelerator(
        self,
        args,
        accelerator,
        network,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        unet,
        text_encoders,
        text_encoder,
        vae,
        vae_dtype,
        weight_dtype,
        train_unet,
        train_text_encoder,
        cache_latents,
    ):
        """Cast model dtypes, run accelerator.prepare, flip train/eval, optional torch.compile."""
        # full fp16/bf16 training
        if args.full_fp16:
            assert args.mixed_precision == "fp16", (
                "full_fp16 requires mixed precision='fp16'"
            )
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert args.mixed_precision == "bf16", (
                "full_bf16 requires mixed precision='bf16'"
            )
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet_weight_dtype = te_weight_dtype = weight_dtype

        unet.requires_grad_(False)
        if self.cast_unet(args):
            unet.to(dtype=unet_weight_dtype)
        for i, t_enc in enumerate(text_encoders):
            # None when the TE was never loaded (cache_text_encoder_outputs with
            # no sample prompts / val / TE-training -- qwen3_needed=False).
            if t_enc is None:
                continue
            t_enc.requires_grad_(False)

            # in case of cpu, dtype is already set to fp32 because cpu does not support fp16/bf16
            if t_enc.device.type != "cpu" and self.cast_text_encoder(args):
                t_enc.to(dtype=te_weight_dtype)

        # accelerator preparation (no deepspeed)
        if train_unet:
            unet = self.prepare_unet_with_accelerator(args, accelerator, unet)
        else:
            unet.to(
                accelerator.device,
                dtype=unet_weight_dtype if self.cast_unet(args) else None,
            )
        if train_text_encoder:
            text_encoders = [
                (accelerator.prepare(t_enc) if flag else t_enc)
                for t_enc, flag in zip(
                    text_encoders,
                    self.get_text_encoders_train_flags(args, text_encoders),
                )
            ]
            if len(text_encoders) > 1:
                text_encoder = text_encoders
            else:
                text_encoder = text_encoders[0]
        # else: text_encoder is unchanged; device and dtype are already set above

        network, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
            accelerator.prepare(
                network, optimizer, train_dataloader, val_dataloader, lr_scheduler
            )
        )
        training_model = network

        if args.gradient_checkpointing:
            # according to TI example in Diffusers, train is required
            unet.train()
            for i, (t_enc, frag) in enumerate(
                zip(
                    text_encoders,
                    self.get_text_encoders_train_flags(args, text_encoders),
                )
            ):
                if t_enc is None:
                    continue
                t_enc.train()

                # set top parameter requires_grad = True for gradient checkpointing works
                if frag:
                    self.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)

        else:
            unet.eval()
            for t_enc in text_encoders:
                if t_enc is None:
                    continue
                t_enc.eval()

        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        if not cache_latents:
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # patch for fp16 grad scale
        if args.full_fp16:
            patch_accelerator_for_fp16_training(accelerator)

        return (
            network,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
            training_model,
            unet,
            text_encoders,
            text_encoder,
            unet_weight_dtype,
        )

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        setup_logging(args, reset=True)

        cache_latents = args.cache_latents

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # Whether inductor will have CUDAGraphs active -- governs whether the
        # training loop needs to call torch.compiler.cudagraph_mark_step_begin()
        # each step (see the call site inside the accumulate block).
        self._cudagraph_mark_step = bool(
            getattr(args, "torch_compile", False)
            and getattr(args, "compile_inductor_mode", None)
            in ("reduce-overhead", "max-autotune")
        )

        tokenize_strategy = self.get_tokenize_strategy(args)
        text_strategies.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(
            tokenize_strategy
        )  # will be removed after sample_image is refactored

        # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        text_strategies.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        (
            train_dataset_group,
            val_dataset_group,
            current_epoch,
            current_step,
            collator,
            use_user_config,
            use_dreambooth_method,
        ) = self._prepare_dataset(args)

        if args.debug_dataset:
            train_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
            debug_dataset(train_dataset_group)

            if val_dataset_group is not None:
                val_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
                debug_dataset(val_dataset_group)
            return
        if len(train_dataset_group) == 0:
            logger.error(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images)"
            )
            return

        if cache_latents:
            assert train_dataset_group.is_latent_cacheable(), (
                "when caching latents, either color_aug or random_crop cannot be used"
            )
            if val_dataset_group is not None:
                assert val_dataset_group.is_latent_cacheable(), (
                    "when caching latents, either color_aug or random_crop cannot be used"
                )

        self.assert_extra_args(
            args, train_dataset_group, val_dataset_group
        )  # may change some args

        # Set the text-encoder-outputs caching strategy now (before the model
        # load) so the cache-completeness probe below can use it to decide
        # whether the Qwen3 text encoder needs loading at all.
        text_encoder_outputs_caching_strategy = (
            self.get_text_encoder_outputs_caching_strategy(args)
        )
        if text_encoder_outputs_caching_strategy is not None:
            text_strategies.TextEncoderOutputsCachingStrategy.set_strategy(
                text_encoder_outputs_caching_strategy
            )

        # Decide whether the heavy encoders are actually needed. When caching is
        # enabled the caches MUST already be complete on disk (run `make
        # preprocess` first) — train.py no longer encodes missing latents / TE
        # outputs on the fly. With complete caches and nothing else needing them
        # we skip loading the encoders entirely (saves the disk read, RAM, and
        # the GPU round-trip). `cache_latents = false` (e.g. IP-Adapter) is a
        # separate, explicit live-encoding mode, not a fallback.
        sampling_enabled = bool(
            args.sample_prompts
            and (
                args.sample_at_first
                or args.sample_every_n_steps
                or args.sample_every_n_epochs
            )
        )

        def _latents_complete(group):
            return group is None or group.is_latents_cache_complete()

        def _te_complete(group):
            return group is None or group.is_text_encoder_outputs_cache_complete()

        if cache_latents and not (
            _latents_complete(train_dataset_group)
            and _latents_complete(val_dataset_group)
        ):
            raise RuntimeError(
                "Latent cache is incomplete. train.py requires a completed "
                "preprocess pass — run `make preprocess` (or set "
                "use_vae_cache = false for live VAE encoding)."
            )

        if args.cache_text_encoder_outputs and not (
            _te_complete(train_dataset_group) and _te_complete(val_dataset_group)
        ):
            raise RuntimeError(
                "Text-encoder cache is incomplete. train.py requires a completed "
                "preprocess pass — run `make preprocess` (or set "
                "use_text_cache = false for live encoding)."
            )

        # CMMD validation generates samples and decodes them through the VAE
        # (see library/training/validation.py). It reads cached TE outputs, so
        # it needs the VAE but not the text encoder.
        cmmd_validation = val_dataset_group is not None and getattr(
            args, "use_cmmd", True
        )
        # VAE: needed only to live-encode (caching off), to decode training
        # samples, or to decode CMMD validation samples. With caching on the
        # cache is guaranteed complete above, so no encode pass is required.
        vae_needed = (not cache_latents) or sampling_enabled or cmmd_validation

        # Qwen3 TE: needed only to live-encode (caching off), to encode sample
        # prompts, or when the text encoder itself is being trained.
        qwen3_needed = (
            (not args.cache_text_encoder_outputs)
            or bool(args.sample_prompts)
            or self.is_train_text_encoder(args)
        )

        # Prepare accelerator
        logger.info("preparing accelerator")
        accelerator = prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        # mixed precision dtype
        weight_dtype, save_dtype = prepare_dtype(args)
        vae_dtype = (
            (torch.float32 if args.no_half_vae else weight_dtype)
            if self.cast_vae(args)
            else None
        )

        # load target models: unet may be None for lazy loading
        model_version, text_encoder, vae, unet = self.load_target_model(
            args,
            weight_dtype,
            accelerator,
            load_qwen3=qwen3_needed,
            load_vae=vae_needed,
        )
        if vae_dtype is None:
            vae_dtype = vae.dtype if vae is not None else weight_dtype
            logger.info(
                f"vae_dtype is set to {vae_dtype} by the model since cast_vae() is false"
            )

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = (
            text_encoder if isinstance(text_encoder, list) else [text_encoder]
        )

        # prepare dataset for latents caching if needed. When vae is None the
        # latents are already fully cached -- new_cache_latents still runs to
        # populate each ImageInfo.latents_npz path the dataloader reads, but
        # forms no encode batches so the (absent) VAE is never touched.
        if cache_latents:
            if vae is not None:
                vae.to(accelerator.device, dtype=vae_dtype)
                vae.requires_grad_(False)
                vae.eval()

            train_dataset_group.new_cache_latents(vae, accelerator)
            if val_dataset_group is not None:
                val_dataset_group.new_cache_latents(vae, accelerator)

            if vae is not None:
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

            accelerator.wait_for_everyone()

        # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
        text_encoding_strategy = self.get_text_encoding_strategy(args)
        text_strategies.TextEncodingStrategy.set_strategy(text_encoding_strategy)

        self.cache_text_encoder_outputs_if_needed(
            args,
            accelerator,
            text_encoders,
            train_dataset_group,
        )
        if val_dataset_group is not None:
            self.cache_text_encoder_outputs_if_needed(
                args,
                accelerator,
                text_encoders,
                val_dataset_group,
            )

        if unet is None:
            # lazy load unet if needed. text encoders may be freed or replaced with dummy models for saving memory
            unet, text_encoders = self.load_unet_lazily(
                args, weight_dtype, accelerator, text_encoders
            )

        # Stage the T5("") sidecar once if caption dropout is on — dropped
        # rows then get the same crossattn embedding Anima feeds at
        # CFG-uncond inference instead of all-zeros (which is out-of-dist).
        if self._state.caption_dropout_enabled:
            self._ensure_uncond_crossattn(args, accelerator, weight_dtype)

        network_result = self._create_and_apply_network(
            args, accelerator, vae, text_encoder, unet, text_encoders, weight_dtype
        )
        if network_result is None:
            return
        network, net_kwargs, train_unet, train_text_encoder = network_result

        # Resolve and run on_network_built for each method adapter (EasyControl,
        # IP-Adapter, …). Each adapter validates its runtime contract and
        # logs/sets up auxiliary state before optimizer / accelerator wiring.
        self._adapters = resolve_adapters(args, network)
        if self._adapters:
            setup_ctx = SetupCtx(
                args=args,
                accelerator=accelerator,
                network=network,
                unet=unet,
                text_encoders=text_encoders,
                weight_dtype=weight_dtype,
            )
            for adapter in self._adapters:
                adapter.on_network_built(setup_ctx)

        (
            optimizer,
            optimizer_name,
            optimizer_args,
            optimizer_train_fn,
            optimizer_eval_fn,
            text_encoder_lr,
            lr_descriptions,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        ) = self._setup_optimizer_and_dataloader(
            args,
            accelerator,
            network,
            train_dataset_group,
            val_dataset_group,
            collator,
        )

        (
            network,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
            training_model,
            unet,
            text_encoders,
            text_encoder,
            unet_weight_dtype,
        ) = self._prepare_with_accelerator(
            args,
            accelerator,
            network,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
            unet,
            text_encoders,
            text_encoder,
            vae,
            vae_dtype,
            weight_dtype,
            train_unet,
            train_text_encoder,
            cache_latents,
        )

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Structured progress sink (Phase 0): a JSONL event stream next to the
        # checkpoint that the GUI / daemon can tail instead of regex-parsing
        # tqdm. Main-process only; default on, gated by --progress_jsonl.
        self.progress_sink = None
        if is_main_process:
            progress_path = ProgressSink.resolve_path(args)
            if progress_path is not None:
                self.progress_sink = ProgressSink(
                    progress_path,
                    run=args.output_name or "run",
                    method=getattr(args, "method", None),
                    preset=getattr(args, "preset", None),
                    t0=training_started_at,
                )
                self.progress_sink.run_start(
                    total_steps=args.max_train_steps,
                    total_epochs=num_train_epochs,
                    pid=os.getpid(),
                )

        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = (
                math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1
            )

        total_batch_size = (
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )

        accelerator.print("running training")
        accelerator.print("  num train images * repeats")
        accelerator.print("  num validation images * repeats")
        accelerator.print("  num reg images")
        accelerator.print("  num batches per epoch")
        accelerator.print("  num epochs")
        accelerator.print("  batch size per device")
        accelerator.print("  gradient accumulation steps")
        accelerator.print("  total optimization steps")

        metadata = build_training_metadata(
            args,
            session_id=session_id,
            training_started_at=training_started_at,
            text_encoder_lr=text_encoder_lr,
            optimizer_name=optimizer_name,
            optimizer_args=optimizer_args,
            model_version=model_version,
            num_train_images=train_dataset_group.num_train_images,
            num_val_images=val_dataset_group.num_train_images
            if val_dataset_group is not None
            else 0,
            num_reg_images=train_dataset_group.num_reg_images,
            num_batches_per_epoch=len(train_dataloader),
            num_train_epochs=num_train_epochs,
        )
        self.update_metadata(metadata, args)  # architecture specific metadata
        add_dataset_metadata(
            metadata,
            train_dataset_group,
            args,
            use_user_config=use_user_config,
            use_dreambooth_method=use_dreambooth_method,
            total_batch_size=total_batch_size,
        )
        add_model_hash_metadata(metadata, args)
        metadata, minimum_metadata = finalize_metadata(
            metadata, net_kwargs=net_kwargs if args.network_args else None
        )

        # Saver owns every save / remove operation plus the accelerator
        # save/load pre-hooks that persist train_state.json. Hooks must be
        # registered before resume_from_local_or_hf_if_specified() so the
        # load hook fires and populates saver.steps_from_state.
        saver = CheckpointSaver(
            args=args,
            accelerator=accelerator,
            save_dtype=save_dtype,
            metadata=metadata,
            minimum_metadata=minimum_metadata,
            get_sai_model_spec_fn=self.get_sai_model_spec,
            current_epoch=current_epoch,
            current_step=current_step,
            progress_sink=self.progress_sink,
        )
        saver.register_hooks(network)

        # auto-resume from the resumable checkpoint if one exists
        saver.auto_resume()

        # resume
        resume_from_local_or_hf_if_specified(accelerator, args)
        steps_from_state = saver.steps_from_state

        # calculate steps to skip when resuming or starting from a specific step
        initial_step = 0
        if args.initial_epoch is not None or args.initial_step is not None:
            if steps_from_state is not None:
                logger.warning(
                    "steps from the state is ignored because initial_step is specified"
                )
            if args.initial_step is not None:
                initial_step = args.initial_step
            else:
                initial_step = (args.initial_epoch - 1) * math.ceil(
                    len(train_dataloader)
                    / accelerator.num_processes
                    / args.gradient_accumulation_steps
                )
        else:
            if steps_from_state is not None:
                initial_step = steps_from_state
                steps_from_state = None

        if initial_step > 0:
            assert args.max_train_steps > initial_step, (
                "max_train_steps should be greater than initial step"
            )

        epoch_to_start = 0
        if initial_step > 0:
            if args.skip_until_initial_step:
                if not args.resume:
                    logger.info(
                        "initial_step is specified but not resuming. lr scheduler will be started from the beginning"
                    )
                logger.info(f"skipping {initial_step} steps")
                initial_step *= args.gradient_accumulation_steps

                epoch_to_start = initial_step // math.ceil(
                    len(train_dataloader) / args.gradient_accumulation_steps
                )
            else:
                epoch_to_start = initial_step // math.ceil(
                    len(train_dataloader) / args.gradient_accumulation_steps
                )
                initial_step = 0  # do not skip

        # Drop the train dataset-group local before loop entry — the
        # dataloader already holds the data it needs. Keep val_dataset_group
        # alive: CMMD validation enumerates its image_data to pair held-out
        # references with generated samples.
        del train_dataset_group

        loop_state = build_loop_state(
            self,
            args=args,
            accelerator=accelerator,
            saver=saver,
            network=network,
            unet=unet,
            text_encoder=text_encoder,
            text_encoders=text_encoders,
            vae=vae,
            tokenizers=tokenizers,
            training_model=training_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            val_dataset_group=val_dataset_group,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_descriptions=lr_descriptions,
            optimizer_train_fn=optimizer_train_fn,
            optimizer_eval_fn=optimizer_eval_fn,
            weight_dtype=weight_dtype,
            unet_weight_dtype=unet_weight_dtype,
            vae_dtype=vae_dtype,
            text_encoding_strategy=text_encoding_strategy,
            tokenize_strategy=tokenize_strategy,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            current_epoch=current_epoch,
            current_step=current_step,
            num_train_epochs=num_train_epochs,
            epoch_to_start=epoch_to_start,
            initial_step=initial_step,
            metadata=metadata,
        )

        # run_scope emits the matching run_end (ok / stopped / error) on exit;
        # run_start already fired when the sink was constructed above.
        with run_scope(self.progress_sink, final_step=lambda: loop_state.global_step):
            run_training_loop(self, loop_state)

            accelerator.end_training()
            optimizer_eval_fn()

            if is_main_process and (args.save_state or args.save_state_on_train_end):
                save_state_on_train_end(args, accelerator)

            saver.cleanup_resumable()
            saver.save_final(network, loop_state.global_step, num_train_epochs)

    # endregion


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    add_dataset_arguments(parser, True, True, True)
    add_training_arguments(parser, True)
    add_masked_loss_arguments(parser)
    add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)

    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="[EXPERIMENTAL] enable offloading of tensors to CPU during checkpointing for U-Net or DiT, if supported"
        "",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save metadata in output model",
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors)",
    )

    parser.add_argument(
        "--unet_lr",
        type=float,
        default=None,
        help="learning rate for U-Net",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        nargs="*",
        help="learning rate for Text Encoder, can be multiple",
    )

    add_network_arguments(parser)
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16",
    )
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + "",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + "",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=None,
        help="Validation seed for shuffling validation dataset, training `--seed` used otherwise",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Split for validation images out of the training dataset",
    )
    parser.add_argument(
        "--validation_split_num",
        type=int,
        default=0,
        help=(
            "Count-based validation split (number of held-out images). When "
            "set (>0), wins over the fractional `--validation_split`. Also "
            "determines how many samples CMMD evaluation generates per pass."
        ),
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=None,
        help="Run validation on validation dataset every N steps. By default, validation will only occur every epoch if a validation dataset is available",
    )
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=None,
        help="Run validation dataset every N epochs. By default, validation will run every epoch if a validation dataset is available",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Max number of validation dataset items processed. By default, validation will run the entire validation dataset",
    )
    parser.add_argument(
        "--validation_sigmas",
        type=float,
        nargs="+",
        default=None,
        help="Sigma values for validation loss (0.0~1.0). Low values = fine detail. Default: 0.1 0.4 0.7. (Legacy FM-val path — unused under the CMMD val replacement.)",
    )
    parser.add_argument(
        "--validation_sample_steps",
        type=int,
        default=20,
        help="Denoising steps used by CMMD validation when sampling each held-out item. Default 20.",
    )
    parser.add_argument(
        "--validation_cfg_scale",
        type=float,
        default=1.0,
        help="CFG scale used by CMMD validation. Default 1.0 (no CFG, fastest). Bump to 4.0 to match production sampling but generation cost ~2×.",
    )
    parser.add_argument(
        "--use_cmmd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CMMD (PE-Core MMD²) as the validation signal. Set "
        "`use_cmmd = false` in the method TOML (or pass `--no-use_cmmd`) to "
        "skip CMMD and run only the legacy per-σ FM-MSE val pass — useful "
        "on tight VRAM where the PE encoder + sampling path doesn't fit.",
    )
    parser.add_argument(
        "--validation_baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each method adapter's validation baselines (e.g. IP-Adapter "
        "no_ip / shuffled_ref) as FM-MSE delta diagnostics during validation. "
        "Set `validation_baselines = false` in the method TOML (or pass "
        "`--no-validation_baselines`) to skip them — each baseline adds a full "
        "extra val forward per (batch, σ), so this roughly halves IP-Adapter "
        "validation time when you don't need the deltas.",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap.",
    )
    parser.add_argument(
        "--print-config",
        dest="print_config",
        action="store_true",
        help="Dump the fully merged config (base → preset → method → CLI) as TOML "
        "with provenance comments, then exit 0. Does not start training.",
    )
    parser.add_argument(
        "--config-snapshot",
        dest="config_snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write output/<output_name>.snapshot.toml next to the checkpoint on every real "
        "run (provenance + git SHA). Pass --no-config-snapshot to disable.",
    )
    parser.add_argument(
        "--config-strict",
        dest="config_strict",
        action="store_true",
        help="Treat config-schema warnings (unknown keys, off-list choices) as errors.",
    )
    return parser


from library.config import schema as _config_schema  # noqa: E402
from networks import all_network_kwargs as _all_network_kwargs  # noqa: E402


# Network-module-consumed flags (networks.lora_anima / networks.methods.*).
# These don't flow through argparse directly because `create_network` reads
# them from ``kwargs``. Derived from the registry in ``networks/__init__.py``
# (``SHARED_KWARG_FLAGS`` ∪ per-``NetworkSpec.kwarg_flags``) so adding a new
# kwarg to a variant spec automatically registers it here.
NETWORK_KWARG_ALLOWLIST: tuple[str, ...] = _all_network_kwargs()

# Top-level training args that aren't network kwargs but still flow through
# ``net_kwargs`` because a network module reads them. Kept explicit -- any
# growth here should be reviewed, since the right answer is usually to
# expose the value as a proper argparse flag the network module reads
# directly rather than tunneling it through kwargs.
_EXTRA_FORWARDED_TOP_LEVEL_ARGS: tuple[str, ...] = (
    # Postfix contrastive resets its intra-step reference set on step
    # boundary, so it needs the grad-accum window.
    "gradient_accumulation_steps",
)


def build_network_extras() -> dict[str, _config_schema.ConfigKey]:
    return {
        k: _config_schema.ConfigKey(name=k, type="str", source="network_module")
        for k in NETWORK_KWARG_ALLOWLIST
    }


def _install_crash_reporter(argv: list[str]) -> None:
    """Record a fatal startup/training exception into ``--progress_jsonl``.

    The daemon launches us windowless under ``pythonw.exe``; that interpreter
    drops the child's stdout/stderr (only the ``accelerate launch`` *parent*'s
    output reaches ``stdout.log``), so an uncaught traceback here is lost and the
    daemon falls back to a generic "process exited (code=1)" with nothing
    actionable. ``progress.jsonl`` is written by path, not via the dead std
    streams, so it survives — and it's what the daemon already reads to diagnose
    a job (``manager._finalize_from_exit`` → ``run_end.error``).

    ``run_scope`` already emits ``run_end(error=…)`` for failures inside the
    training loop, but only *after* ``ProgressSink.run_start`` has fired — late
    in ``train()``. Errors before that (latent/TE cache incomplete, config or
    dataset build, model load) escape it entirely. This excepthook is the
    catch-all: it appends a ``run_end`` error event for any uncaught exception,
    wherever it's raised, so the GUI's finish banner shows the real cause.
    """
    path = None
    for i, tok in enumerate(argv):
        if tok == "--progress_jsonl" and i + 1 < len(argv):
            path = argv[i + 1]
        elif tok.startswith("--progress_jsonl="):
            path = tok.split("=", 1)[1]
    if not path or path.strip().lower() in ("", "none", "off"):
        return

    import json as _json

    prev_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        # KeyboardInterrupt is a clean stop, handled by run_scope/the daemon's
        # stop_requested path — don't mislabel it an error.
        if not issubclass(exc_type, KeyboardInterrupt):
            try:
                # Dedupe: run_scope may already have written the terminal event
                # for an in-loop failure; don't append a second one.
                already_ended = False
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                last = line
                    try:
                        already_ended = _json.loads(last).get("ev") == "run_end"
                    except (NameError, ValueError):
                        already_ended = False
                if not already_ended:
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(
                            _json.dumps(
                                {
                                    "ev": "run_end",
                                    "status": "error",
                                    "final_step": -1,
                                    "error": f"{exc_type.__name__}: {exc}",
                                }
                            )
                            + "\n"
                        )
            except Exception:  # noqa: BLE001 — reporting must never mask the crash
                pass
        prev_hook(exc_type, exc, tb)

    sys.excepthook = _hook


if __name__ == "__main__":
    _install_crash_reporter(sys.argv)
    parser = setup_parser()
    _config_schema.populate_schema(parser, extras=build_network_extras())

    args = parser.parse_args()
    verify_command_line_training_args(args)
    args = read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    artist = getattr(args, "artist_filter", None)
    if artist:
        _datasets_base.set_artist_filter(artist)
        slug = artist.lstrip("@")
        args.output_dir = "output/ckpt-artist"
        args.output_name = f"{args.output_name}_{slug}"
        logger.info(
            f"artist_filter active: '{artist}' → output_dir={args.output_dir}, "
            f"output_name={args.output_name}"
        )

    trainer = AnimaTrainer()
    trainer.train(args)
