# Anima LoRA training script (merged standalone)

import gc
import _thread
import importlib
import argparse
import json
import math
import os
import typing
from typing import Any, Union, Optional
import sys
import random
import signal
import threading
import time
from multiprocessing import Value

# Windows: suppress per-kernel ptxas.exe / cl.exe console flashes from
# torch.compile + Triton. Must run before any subprocess.Popen call (i.e.
# before torch import on Windows where inductor may prefetch toolchain).
from library.runtime.proc import install_no_window_default

install_no_window_default()

# Allocator default must land before torch initializes the CUDA caching
# allocator: free-fit varies seq_len per step and fragments the reserved pool
# without expandable segments (issue #58). Opt out: ANIMA_EXPANDABLE_SEGMENTS=0.
from library.runtime.allocator import default_expandable_segments

if default_expandable_segments():
    print(
        "Anima: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "(default; set ANIMA_EXPANDABLE_SEGMENTS=0 to disable)"
    )

import torch
import torch.nn as nn
from library.runtime.device import clean_memory_on_device
from library.io.output_layout import layout_from_args, write_run_manifest
from library.training.state import restore_rng_state, signature

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
    ComputeLossCtx,
    ForwardArtifacts,
    MethodAdapter,
    SetupCtx,
    StepCtx,
    resolve_adapters,
)
from library.config.io import (
    load_dataset_config_from_base,
    read_config_from_file,
    refresh_config_snapshot,
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
    prepare_accelerator,
    prepare_dtype,
    resolve_run_log_dir,
    resume_from_local_or_hf_if_specified,
)
from library.runtime.block_swap_budget import check_block_swap_budget
from library.training import (
    AcceleratedBundle,
    CheckpointSaver,
    DatasetBundle,
    LivenessLedger,
    LossContext,
    NetworkBundle,
    OptimizerBundle,
    SAMPLER_REGISTRY,
    RuntimeState,
    SamplerContext,
    TrainCtx,
    add_custom_train_arguments,
    add_dataset_metadata,
    add_model_hash_metadata,
    build_loss_composer,
    build_training_metadata,
    finalize_metadata,
    get_huber_threshold_if_needed,
    get_optimizer,
    get_optimizer_train_eval_fn,
    get_scheduler_fix,
    save_state_on_train_end,
    StopController,
)
from library.config.cli_args import (
    add_dataset_arguments,
    add_dit_training_arguments,
    add_masked_loss_arguments,
    add_network_arguments,
    add_optimizer_arguments,
    add_sd_models_arguments,
    add_train_misc_arguments,
    add_training_arguments,
    add_validation_arguments,
    verify_command_line_training_args,
    verify_training_args,
)
from library.training.loop import (
    build_loop_state,
    release_text_encoder_handles,
    run_training_loop,
)
from library.training.resume import (
    resolve_persisted_resume_position,
    resolve_resume_position,
)
from library.preprocess.runs import PreprocessRun, PreprocessRunError, run_from_manifest
from library.training.stage_schedule import (
    prepare_stage_runtime,
    stage_epoch_upper_bound,
)
from library.training.convrot import maybe_apply_convrot_base
from library.training.staged_resolution import configure_staged_resolution
from library.training.sampling_config import normalize_sample_args
from library.training.log_dispatch import (
    dispatch_logs,
    generate_step_logs as _generate_step_logs,
)
from library.training.progress import ProgressSink, run_scope
from library.training.forward import (
    ForwardConditioning,
    apply_router_conditioning,
    build_forward_conditioning,
    compute_inversion_func_loss,
    prepare_text_conds,
    run_vr_reference_forward,
)
from library.log import setup_logging, add_logging_arguments

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


def _install_graceful_stop_handlers() -> None:
    """Turn daemon termination signals into cleanup-capable interrupts."""
    for name in ("SIGTERM", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            signal.signal(signum, signal.default_int_handler)
        except (OSError, ValueError):
            pass

    stop_file = os.environ.get("ANIMA_DAEMON_STOP_FILE")
    if os.name != "nt" or not stop_file:
        return

    def watch_stop_file() -> None:
        while not os.path.exists(stop_file):
            time.sleep(0.1)
        _thread.interrupt_main()

    threading.Thread(
        target=watch_stop_file,
        name="anima-daemon-stop-watcher",
        daemon=True,
    ).start()


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.lower() not in {"", "0", "false", "no", "off"}


_RESUME_SIGNATURE_EXCLUDES = {
    "resume",
    # The daemon materializes an identical config snapshot under each job's
    # directory.  Its absolute path is transport metadata, not training
    # semantics; including it would make a stopped job impossible to resume
    # from the next daemon submission.
    "config_file",
    "initial_step",
    "initial_epoch",
    "max_train_steps",
    "max_train_epochs",
    "progress_jsonl",
    "sample_dir",
    "logging_dir",
    "config_snapshot",
    "print_config",
    # These fields describe the hardware-dependent effective block-swap
    # policy.  They are recorded in the run manifest/metadata, but excluding
    # them keeps old state directories resumable when a newer trainer learns
    # to report the policy.
    "block_swap_reliable_mode",
    "block_swap_gpu_sm",
    "block_swap_requested_torch_compile",
    "block_swap_compile_disabled",
    "block_swap_effective_lora_fp32_compute",
    "block_swap_effective_use_custom_down_autograd",
    "block_swap_effective_network_spec",
}


def _resume_config_signature(args) -> str:
    """Hash resume-critical config while allowing a run-length extension."""

    payload = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_") and key not in _RESUME_SIGNATURE_EXCLUDES
    }
    return signature(payload)


def _dataset_group_signature(group) -> str | None:
    if group is None:
        return None
    datasets = []
    for dataset in getattr(group, "datasets", []):
        subsets = []
        for subset in getattr(dataset, "subsets", []):
            subsets.append(
                {
                    "image_dir": getattr(subset, "image_dir", None),
                    "cache_dir": getattr(subset, "cache_dir", None),
                    "text_cache_dir": getattr(subset, "text_cache_dir", None),
                    "cond_cache_dir": getattr(subset, "cond_cache_dir", None),
                    "repeats": getattr(subset, "num_repeats", None),
                    "pattern": getattr(subset, "path_pattern", None),
                }
            )
        datasets.append({"subsets": subsets, "length": len(dataset)})
    return signature(datasets)


def _apply_preprocess_run(args) -> PreprocessRun | None:
    """Pin dataset/cache path resolution to one completed preprocess run."""

    raw = getattr(args, "preprocess_run", None)
    if not raw:
        return None
    try:
        run = run_from_manifest(raw)
    except PreprocessRunError as exc:
        raise ValueError(f"invalid --preprocess_run manifest: {exc}") from exc

    # The base and GUI blueprints use these scalar placeholders.  Setting them
    # before ``_prepare_dataset`` routes every generated subset to this run;
    # legacy fixed directories remain untouched when the flag is absent.
    args.preprocess_run = str(run.manifest_path)
    args.resized_image_dir = str(run.resized_dir)
    args.lora_cache_dir = str(run.lora_dir)
    # Text-encoder caches normally share the run's ``lora`` directory, but a
    # subset may explicitly redirect them with ``text_cache_dir`` (for
    # example, colorization/easycontrol blueprints).  Seed the scalar fallback
    # as well so generated blueprints stay inside the selected run unless they
    # deliberately provide another path that is checked below.
    args.text_cache_dir = str(run.lora_dir)
    args.mask_dir = str(run.masks_dir)
    args.multires_image_dir = str(run.multires_dir)
    args.conditioning_data_dir = str(run.conditioning_data_dir)
    args.conditioning_resized_dir = str(run.conditioning_resized_dir)
    args.caption_index_path = str(run.caption_index_path)
    logger.info(
        "preprocess run pinned: manifest=%s source=%s resized=%s cache=%s",
        run.manifest_path,
        run.source_dir,
        run.resized_dir,
        run.lora_dir,
    )
    return run


def _validate_preprocess_dataset_paths(group, run: PreprocessRun | None) -> None:
    """Reject a dataset group that mixes caches from outside its selected run."""

    if run is None or group is None:
        return
    roots = {
        "image": (run.resized_dir, run.multires_dir, run.conditioning_resized_dir),
        "cache": (run.lora_dir,),
        "text": (run.lora_dir,),
        "mask": (run.masks_dir,),
        "conditioning": (run.conditioning_dir, run.conditioning_resized_dir),
    }

    def inside(value, allowed) -> bool:
        if not value:
            return True
        path = os.path.abspath(os.fspath(value))
        for root in allowed:
            try:
                root_path = os.path.abspath(os.fspath(root))
                if os.path.commonpath([path, root_path]) == root_path:
                    return True
            except ValueError:
                continue
        return False

    violations: list[str] = []
    for dataset in getattr(group, "datasets", []):
        for subset in getattr(dataset, "subsets", []):
            image_dir = getattr(subset, "image_dir", None)
            cache_dir = getattr(subset, "cache_dir", None)
            text_cache_dir = getattr(subset, "text_cache_dir", None)
            mask_dir = getattr(subset, "mask_dir", None)
            cond_dir = getattr(subset, "cond_cache_dir", None)
            if not inside(image_dir, roots["image"]):
                violations.append(f"image_dir={image_dir}")
            if not inside(cache_dir, roots["cache"]):
                violations.append(f"cache_dir={cache_dir}")
            if not inside(text_cache_dir, roots["text"]):
                violations.append(f"text_cache_dir={text_cache_dir}")
            if not inside(mask_dir, roots["mask"]):
                violations.append(f"mask_dir={mask_dir}")
            if not inside(cond_dir, roots["conditioning"]):
                violations.append(f"cond_cache_dir={cond_dir}")
    if violations:
        raise ValueError(
            "--preprocess_run cache mixing detected; all dataset paths must belong "
            f"to {run.manifest_path}: " + ", ".join(violations[:8])
        )


def _resolve_contrastive_index(args) -> str:
    """Resolve the authoritative caption index for contrastive training."""
    explicit = getattr(args, "caption_index_path", None)
    path = (
        os.fspath(explicit)
        if explicit
        else "post_image_dataset/captions/caption_index.json"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"contrastive_index not found: {path}. Run `make caption-index`."
        )
    return path


def _resolve_v100_flash_stability(args) -> str:
    value = getattr(args, "v100_flash_stability", None)
    if value is None:
        value = os.environ.get("ANIMA_V100_FLASH_STABILITY", "off")
    value = str(value).lower()
    if value not in {"off", "hybrid", "safe"}:
        logger.warning(
            "invalid ANIMA_V100_FLASH_STABILITY=%r; expected off|hybrid|safe, using off",
            value,
        )
        return "off"
    return value


def _is_v100_fp16_training(args, accelerator) -> bool:
    if getattr(args, "mixed_precision", None) != "fp16":
        return False
    if not torch.cuda.is_available():
        return False
    device = getattr(accelerator, "device", None)
    try:
        if device is not None:
            device_type = getattr(device, "type", None) or torch.device(device).type
            if device_type != "cuda":
                return False
            major, minor = torch.cuda.get_device_capability(device)
        else:
            major, minor = torch.cuda.get_device_capability()
    except Exception:
        logger.warning(
            "could not read GPU compute capability; V100 eager optimizations "
            "remain disabled."
        )
        return False
    return (major, minor) == (7, 0)


def _as_boolish(value: Any) -> bool:
    """Interpret config/CLI bools without treating non-empty ``"false"`` as true."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _blocks_to_swap_enabled(args) -> bool:
    """Return whether the run requests actual DiT block swapping."""

    try:
        return int(getattr(args, "blocks_to_swap", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _block_swap_rejection_reasons(args) -> list[str]:
    """List method features that cannot share the block-swap scheduler.

    Block swap owns the DiT block residency schedule.  The listed methods add
    extra forwards/conditioning streams or their own activation offload path;
    allowing them to opt in would make a successful-looking first step
    unreliable.  Keep this check independent of dataset construction so a bad
    combination fails before accelerator/model allocation.
    """

    if not _blocks_to_swap_enabled(args):
        return []

    module = str(getattr(args, "network_module", "") or "").strip().lower()
    method = str(getattr(args, "method", "") or "").strip().lower()
    reasons: list[str] = []

    if module.endswith(".soft_tokens") or module == "soft_tokens" or method in {
        "soft_tokens",
        "soft-tokens",
    }:
        reasons.append("Soft Tokens")
    if (
        module.endswith(".easycontrol")
        or module == "easycontrol"
        or method == "easycontrol"
        or _as_boolish(getattr(args, "use_easycontrol", False))
    ):
        reasons.append("EasyControl")
    if _as_boolish(getattr(args, "use_byg", False)) or method == "byg":
        reasons.append("BYG")
    if _as_boolish(getattr(args, "unsloth_offload_checkpointing", False)):
        reasons.append("unsloth_offload_checkpointing")

    # Preserve declaration order while avoiding duplicate labels when both a
    # method module and its top-level feature flag identify the same feature.
    return list(dict.fromkeys(reasons))


def _validate_block_swap_config(args) -> None:
    """Reject unsupported method/offload combinations before model startup."""

    reasons = _block_swap_rejection_reasons(args)
    if not reasons:
        return
    labels = ", ".join(reasons)
    raise ValueError(
        "blocks_to_swap>0 is unsupported with "
        f"{labels}; set blocks_to_swap=0 for this method/offload path. "
        "Block swap is currently limited to the standard LoRA/T-LoRA path."
    )


def _resolve_block_swap_reliable_mode(args) -> bool:
    """Apply the conservative Volta block-swap runtime policy.

    V100/sm_70 has no reliable high-peak compile path when blocks are moved
    between CPU and CUDA.  Resolve this *before* ``Accelerator`` construction,
    disable compile/CUDAGraph modes, and leave a machine-readable record on
    ``args`` for metadata and the run manifest.  Other GPUs retain the user's
    compile settings unchanged.
    """

    enabled = _blocks_to_swap_enabled(args)
    args.block_swap_reliable_mode = False
    args.block_swap_gpu_sm = None
    args.block_swap_requested_torch_compile = bool(
        getattr(args, "torch_compile", False)
    )
    args.block_swap_compile_disabled = False

    if not enabled:
        return False
    if not torch.cuda.is_available():
        logger.info(
            "block swap requested without a visible CUDA device; reliable mode "
            "cannot be enabled (the accelerator will report the actual device)."
        )
        return False

    try:
        major, minor = torch.cuda.get_device_capability()
    except (AssertionError, IndexError, NotImplementedError, RuntimeError, TypeError, ValueError):
        logger.warning(
            "block swap requested but GPU compute capability could not be read; "
            "leaving compile settings unchanged and disabling reliable-mode claim."
        )
        return False

    args.block_swap_gpu_sm = f"sm_{major}{minor}"
    if (major, minor) != (7, 0):
        logger.info(
            "block swap enabled on gpu_sm=%s; V100 reliable mode is not needed, "
            "keeping torch_compile=%s",
            args.block_swap_gpu_sm,
            bool(getattr(args, "torch_compile", False)),
        )
        return False

    args.block_swap_reliable_mode = True
    if getattr(args, "torch_compile", False):
        args.torch_compile = False
        args.block_swap_compile_disabled = True
    # ``compile_inductor_mode`` is inert once torch_compile is off, but clearing
    # it makes the effective configuration explicit and prevents a later
    # consumer from accidentally re-enabling a CUDAGraph/high-peak path.
    if getattr(args, "compile_inductor_mode", None) is not None:
        args.compile_inductor_mode = None
    if hasattr(args, "dynamo_backend"):
        args.dynamo_backend = "eager"

    logger.warning(
        "block swap reliable mode enabled for gpu_sm=sm_70: effective "
        "torch_compile=false, compile_inductor_mode=None, dynamo_backend=eager; "
        "standard LoRA/T-LoRA will use the eager/custom-autograd memory path."
    )
    return True


def _block_swap_manifest_fields(args) -> dict[str, Any]:
    """Return effective block-swap fields for run manifests."""

    raw_blocks = getattr(args, "blocks_to_swap", 0)
    try:
        blocks = int(raw_blocks or 0)
    except (TypeError, ValueError):
        blocks = 0
    fields = {
        "blocks_to_swap": blocks,
        "block_swap_reliable_mode": bool(
            getattr(args, "block_swap_reliable_mode", False)
        ),
        "block_swap_gpu_sm": getattr(args, "block_swap_gpu_sm", None),
        "block_swap_requested_torch_compile": bool(
            getattr(args, "block_swap_requested_torch_compile", False)
        ),
        "block_swap_compile_disabled": bool(
            getattr(args, "block_swap_compile_disabled", False)
        ),
        "block_swap_effective_torch_compile": bool(
            getattr(args, "torch_compile", False)
        ),
        "block_swap_effective_lora_fp32_compute": bool(
            getattr(args, "block_swap_effective_lora_fp32_compute", False)
        ),
        "block_swap_effective_use_custom_down_autograd": bool(
            getattr(args, "block_swap_effective_use_custom_down_autograd", False)
        ),
        "block_swap_effective_network_spec": getattr(
            args, "block_swap_effective_network_spec", None
        ),
    }
    # The budget is populated immediately before accelerator.prepare.  Keep it
    # optional so older callers/tests that only exercise the effective policy
    # retain their compact manifest shape.
    budget = getattr(args, "block_swap_budget", None)
    if isinstance(budget, dict):
        fields["block_swap_budget"] = dict(budget)
    return fields


def _training_budget_manifest_fields(args) -> dict[str, Any]:
    """Return the resolved budget and its explicit provenance for manifests."""

    return {
        "training_budget_source": getattr(args, "training_budget_source", None),
        "training_budget_origin": getattr(args, "training_budget_origin", None),
        "max_train_steps": int(getattr(args, "max_train_steps", 0) or 0),
        "max_train_epochs": getattr(args, "max_train_epochs", None),
    }


def _finalize_training_budget(args, *, dataloader_length: int, num_processes: int) -> int:
    """Resolve the optimizer-step budget without letting epochs overwrite it.

    ``argparse`` supplies a default step value even when the user only set
    epochs.  ``read_config_from_file`` records explicit intent in the private
    provenance flag; direct embedders that do not use that loader retain the
    historical interpretation that a non-None step value is explicit.
    """

    explicit_steps = getattr(args, "_max_train_steps_explicit", None)
    if explicit_steps is None:
        explicit_steps = getattr(args, "max_train_steps", None) is not None
    epochs = getattr(args, "max_train_epochs", None)
    if explicit_steps:
        effective = int(getattr(args, "max_train_steps", 0) or 0)
        source = "max_train_steps"
        if epochs is not None:
            logger.warning(
                "Both max_train_steps and max_train_epochs are set; using the "
                "explicit max_train_steps=%s (epochs=%s is ignored)",
                effective,
                epochs,
            )
    elif epochs is not None:
        effective = int(epochs) * math.ceil(
            dataloader_length / num_processes / args.gradient_accumulation_steps
        )
        source = "max_train_epochs"
    else:
        effective = int(getattr(args, "max_train_steps", 0) or 0)
        source = "max_train_steps_default"

    if effective <= 0:
        raise ValueError(f"resolved max_train_steps must be positive, got {effective}")
    args.max_train_steps = effective
    args.effective_max_train_steps = effective
    args.training_budget_source = source
    if source == "max_train_steps":
        args.training_budget_origin = getattr(args, "_max_train_steps_source", "unknown")
    elif source == "max_train_epochs":
        args.training_budget_origin = getattr(args, "_max_train_epochs_source", "unknown")
    else:
        args.training_budget_origin = "argparse default"
    refresh_config_snapshot(args)
    return effective


def _should_auto_enable_lora_fp32_compute(args, accelerator, net_kwargs: dict) -> bool:
    """Keep LoRA-family adapter projections in fp32 on V100/fp16.

    Explicit ``lora_fp32_compute=...`` from TOML / ``--network_args`` wins. The
    automatic fallback is intentionally narrow (Volta V100 sm_70 + fp16) so the
    long-tested bf16 path and other GPUs remain unchanged.
    """
    # On V100 block-swap reliable mode this setting is part of the safety
    # contract, so an explicit ``false`` is overridden after a warning at the
    # network-construction call site.  Outside that mode explicit config keeps
    # its historical precedence.
    if "lora_fp32_compute" in net_kwargs and not getattr(
        args, "block_swap_reliable_mode", False
    ):
        return False
    if getattr(args, "network_module", None) != "networks.lora_anima":
        return False
    return _is_v100_fp16_training(args, accelerator)


def _should_auto_enable_eager_lora_down_autograd(
    args, accelerator, net_kwargs: dict
) -> bool:
    """Enable the eager V100 adapter/operator-fusion memory path.

    ``use_custom_down_autograd`` is the compatibility name for saved-input
    LoRA rank projections, the rematerialized LoKr bypass, and bounded LoRA-up
    and MLP intermediates. RMSNorm and RoPE use their own V100-only guards.
    Explicit configuration wins. This path is useful only when compile is off
    (compiled graphs already use Dynamo fusion and AOTAutograd partitioning)
    and the adapter projection path resolved to fp32.
    """
    if "use_custom_down_autograd" in net_kwargs and not getattr(
        args, "block_swap_reliable_mode", False
    ):
        return False
    if getattr(args, "torch_compile", False):
        return False
    if not _is_v100_fp16_training(args, accelerator):
        return False
    if getattr(args, "network_module", None) != "networks.lora_anima":
        return False
    from networks import resolve_network_spec

    if resolve_network_spec(net_kwargs).name not in ("lora", "lokr"):
        return False
    return str(net_kwargs.get("lora_fp32_compute", "false")).strip().lower() in (
        "true",
        "1",
    )


def _apply_v100_adapter_runtime_policy(
    args, accelerator, net_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Apply and record the effective V100 LoRA eager-memory policy."""

    reliable_block_swap = bool(
        getattr(args, "block_swap_reliable_mode", False)
    )
    if _should_auto_enable_lora_fp32_compute(args, accelerator, net_kwargs):
        if reliable_block_swap and str(
            net_kwargs.get("lora_fp32_compute", "")
        ).strip().lower() in {"0", "false", "no", "off"}:
            logger.warning(
                "V100 block-swap reliable mode overrides explicit "
                "lora_fp32_compute=false; adapter projections stay in fp32."
            )
        net_kwargs["lora_fp32_compute"] = "true"
        if reliable_block_swap:
            logger.warning(
                "V100/sm_70 block-swap reliable mode: enforcing "
                "lora_fp32_compute=true so LoRA rank GEMMs stay fp32 while "
                "the frozen base remains fp16."
            )
        else:
            logger.warning(
                "V100/sm_70 fp16 training detected: auto-enabling "
                "lora_fp32_compute so LoRA rank GEMMs run in fp32 while the "
                "frozen base remains fp16. Set lora_fp32_compute=false to "
                "disable for A/B testing."
            )
    if _should_auto_enable_eager_lora_down_autograd(
        args, accelerator, net_kwargs
    ):
        if reliable_block_swap and str(
            net_kwargs.get("use_custom_down_autograd", "")
        ).strip().lower() in {"0", "false", "no", "off"}:
            logger.warning(
                "V100 block-swap reliable mode overrides explicit "
                "use_custom_down_autograd=false; eager rematerialization "
                "remains enabled for the supported LoRA family."
            )
        net_kwargs["use_custom_down_autograd"] = "true"
        if reliable_block_swap:
            logger.warning(
                "V100/sm_70 block-swap reliable mode: enforcing "
                "use_custom_down_autograd=true for bounded eager LoRA and "
                "MLP intermediates."
            )
        else:
            logger.warning(
                "eager V100 fp16/FP32-residual LoRA training detected: "
                "auto-enabling use_custom_down_autograd for bounded eager "
                "LoRA and MLP intermediates. Rank GEMMs remain fp32; frozen "
                "sublayer matmuls remain fp16. Set "
                "use_custom_down_autograd=false to disable for A/B testing."
            )

    # Record the *effective* adapter path rather than only the requested TOML
    # values.  Reliable mode may intentionally override explicit false values.
    args.block_swap_effective_lora_fp32_compute = _as_boolish(
        net_kwargs.get("lora_fp32_compute")
    )
    args.block_swap_effective_use_custom_down_autograd = _as_boolish(
        net_kwargs.get("use_custom_down_autograd")
    )
    try:
        from networks import resolve_network_spec

        args.block_swap_effective_network_spec = resolve_network_spec(net_kwargs).name
    except (NotImplementedError, RuntimeError, TypeError, ValueError):
        # The network factory will report the authoritative configuration error;
        # manifest generation should still remain best-effort.
        args.block_swap_effective_network_spec = None
    logger.info(
        "block swap effective config: enabled=%s reliable_mode=%s "
        "gpu_sm=%s torch_compile=%s lora_fp32_compute=%s "
        "use_custom_down_autograd=%s network_spec=%s",
        _blocks_to_swap_enabled(args),
        reliable_block_swap,
        getattr(args, "block_swap_gpu_sm", None),
        bool(getattr(args, "torch_compile", False)),
        args.block_swap_effective_lora_fp32_compute,
        args.block_swap_effective_use_custom_down_autograd,
        args.block_swap_effective_network_spec,
    )
    return net_kwargs


def _resolve_mixed_precision(args) -> None:
    """Back-write ``args.mixed_precision`` for pre-Ampere GPUs in place.

    Pre-Ampere GPUs (sm<8, e.g. V100 sm_70 / T4 sm_75) have no native bf16:
    under autocast, bf16 ops run via slower fp32 emulation. When the user is
    on the default bf16 and the GPU can't do bf16 natively, switch to fp16
    (which every sm≥7.0 GPU supports natively).

    This MUST run before ``prepare_accelerator``: ``Accelerator()`` bakes the
    autocast dtype in at construction time, and three other consumers
    (``enable_fp32_residual`` guard below, ``library/training/metadata.py``,
    and ``library/training/loop.py`` reading it off the accelerator object)
    must all see the same value. Back-writing args propagates to all of them.

    fp16 carries residual-overflow risk; the ``Anima.enable_fp32_residual()``
    guard in the loop-setup path (gated on ``args.mixed_precision=="fp16"``)
    handles it, so it stays consistent once we switch here.
    """
    if getattr(args, "mixed_precision", None) != "bf16":
        return
    if not torch.cuda.is_available():
        return
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        # CUDA reported available but capability probe failed (init failure,
        # multi-GPU index mismatch, …) — keep bf16, log so it's diagnosable.
        logger.warning(
            "could not read GPU compute capability; keeping --mixed_precision bf16.",
        )
        return
    if major < 8:
        args.mixed_precision = "fp16"
        logger.warning(
            "GPU sm_%d0 has no native bf16 (bf16 autocast runs the slower "
            "fp32 emulation) — auto-switching --mixed_precision from bf16 to "
            "fp16. Pass --mixed_precision bf16 explicitly to keep bf16.",
            major,
        )


def _resolve_vae_dtype(args, weight_dtype: torch.dtype) -> torch.dtype:
    """Derive the VAE dtype, forcing fp32 where fp16 decode is unsafe.

    The Qwen VAE decoder runs almost entirely in its parameter dtype (only
    ``QwenImageUpsample`` self-protects via ``forward(x.float()).type_as(x)``,
    see ``library/models/qwen_vae.py:399-400``). Under fp16 the decoder's
    conv accumulators and group-norm variance stats hit the ±65504 dynamic
    range and lose precision, producing artifacts in preview/sample images
    (花图/糊图). bf16 (Ampere+) is safe — its exponent matches fp32.

    So on pre-Ampere GPUs (sm<8, e.g. V100 sm_70 / T4 sm_75) under fp16 we
    force the VAE to fp32 unless the user explicitly opts into half with
    ``--half_vae``. ``--no_half_vae`` always wins (unconditional fp32).

    ``weight_dtype`` is passed in so this stays consistent with whatever
    ``prepare_dtype`` produced (and with ``_resolve_mixed_precision``, which
    has already back-written ``args.mixed_precision`` before this runs).
    """
    if getattr(args, "no_half_vae", False):
        return torch.float32
    if getattr(args, "half_vae", False):
        # Explicit user override — they accept the fp16 artifacts. Only
        # meaningful under fp16 on pre-Ampere; no-op elsewhere.
        return weight_dtype
    if getattr(args, "mixed_precision", None) != "fp16":
        return weight_dtype
    if not torch.cuda.is_available():
        return weight_dtype
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        # Probe failed (init failure, multi-GPU index mismatch, …) — keep
        # weight_dtype rather than forcing fp32 blindly. Logged so it's
        # diagnosable, mirroring _resolve_mixed_precision's safe fallback.
        logger.warning(
            "could not read GPU compute capability; keeping VAE dtype at "
            f"{weight_dtype} (fp16 decode artifacts possible on pre-Ampere).",
        )
        return weight_dtype
    if major < 8:
        logger.info(
            "pre-Ampere GPU (sm_%d%d) under fp16: forcing VAE to fp32 to avoid "
            "decode artifacts (花图/糊图). Pass --half_vae to allow half-precision "
            "VAE (not recommended).",
            major,
            minor,
        )
        return torch.float32
    return weight_dtype


def _flash_attn_v100_doc(flash_attn_module) -> tuple[str, bool]:
    doc = getattr(flash_attn_module, "__doc__", None) or ""
    is_v100_fork = "Tesla V100" in doc or "Flash Attention for Tesla V100" in doc
    return doc.strip().replace("\n", " "), is_v100_fork


class AnimaTrainer:
    def __init__(self):
        self.sample_prompts_te_outputs = None
        self._padding_mask_cache = {}
        # Per-method extensions (EasyControl, IP-Adapter, …). Resolved
        # from args+network in train() right after _create_and_apply_network.
        self._adapters: list[MethodAdapter] = []
        # Feature-specific per-run state — see ``RuntimeState``.
        self._state = RuntimeState()
        # Liveness ledger (issues.md P1.1): counts aux consumption per
        # skip-if-missing loss; the loop audits it (step-25 early check +
        # run end) and flags configured-but-dead features with `LIVENESS:`.
        self._liveness = LivenessLedger()

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
        # Thin wrapper (same shape as step_logging/epoch_logging below): the
        # loop calls this on the trainer; the assembly lives in log_dispatch,
        # with the trainer contributing only its VR λ state.
        return _generate_step_logs(
            args,
            current_loss,
            avr_loss,
            lr_scheduler,
            lr_descriptions,
            optimizer,
            keys_scaled,
            mean_norm,
            maximum_norm,
            mean_grad_norm,
            mean_combined_norm,
            vr_state=self._state.vr,
        )

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
        # Keep this guard here as well as at train() entry: a few embedders call
        # the trainer lifecycle directly, bypassing the command-line wrapper.
        _validate_block_swap_config(args)

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
        elif getattr(args, "cache_llm_adapter_outputs", False):
            # Adapter-output caching writes into the TE cache; with text caching
            # off there is nothing to write into (the caching strategy is None and
            # adapter outputs are computed live), so the flag is a harmless no-op.
            # Auto-disable it instead of crashing — this combination is easy to
            # hit from the GUI, where use_text_cache and cache_llm_adapter_outputs
            # are independent toggles while methods default the latter to true.
            logger.warning(
                "cache_llm_adapter_outputs=true has no effect without text-encoder "
                "caching (use_text_cache=false / live text encoding); disabling it."
            )
            args.cache_llm_adapter_outputs = False

        assert args.network_train_unet_only or not args.cache_text_encoder_outputs, (
            "network for Text Encoder cannot be trained with caching Text Encoder outputs"
        )

        if args.unsloth_offload_checkpointing:
            if not args.gradient_checkpointing:
                logger.warning(
                    "unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled"
                )
                args.gradient_checkpointing = True
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

        # Propagate BYG per-image edit-tuple cache dir so datasets load
        # {stem}_byg.safetensors into batch["byg_{role}_emb"]/["byg_{role}_mask"].
        if getattr(args, "use_byg", False):
            byg_text_dir = getattr(args, "byg_text_dir", None) or os.path.join(
                "post_image_dataset", "byg"
            )
            for dataset in train_dataset_group.datasets:
                dataset.byg_text_dir = byg_text_dir
                kept, dropped = dataset.restrict_to_byg_tuples()
                if dropped:
                    logger.info(
                        f"BYG: kept {kept} images with edit-tuple sidecars, "
                        f"dropped {dropped} without (no swappable tag in caption)."
                    )
            # restrict_to_byg_tuples re-buckets each member, shrinking its length;
            # refresh the ConcatDataset cumulative_sizes or global indices overflow.
            train_dataset_group.refresh_concat_state()
            if val_dataset_group is not None:
                for dataset in val_dataset_group.datasets:
                    dataset.byg_text_dir = byg_text_dir
                    dataset.restrict_to_byg_tuples()
                val_dataset_group.refresh_concat_state()

        # REPA v2: load cached PE-Spatial patch tokens into batches when
        # use_repa is set. The flag rides the network kwargs; read the resolved
        # merged view (--network_args + top-level TOML keys) rather than
        # re-scanning both intake paths.
        net_kwargs = resolve_network_kwargs(args)
        if net_kwargs.get("use_repa", "").lower() in ("true", "1", "yes"):
            repa_encoder = net_kwargs.get("repa_encoder") or "pe_spatial"
            for dataset in train_dataset_group.datasets:
                dataset.load_repa_pe = True
                dataset.repa_pe_encoder = repa_encoder
            # Probe PE sidecar coverage now. A missing PE cache makes the REPA
            # alignment term a silent no-op — the loss skips any batch without
            # repa_pe_features (library/training/repa.py) — so a run with
            # use_repa but no `make preprocess-pe` would train as if REPA were
            # off, with no error. Fail fast on a fully-absent cache; warn on a
            # partial one (the all-or-nothing collate tolerates per-batch gaps).
            present, total = train_dataset_group.count_repa_pe_sidecars()
            if total > 0 and present == 0:
                raise RuntimeError(
                    f"use_repa is enabled but none of the {total} training "
                    f"images have a {repa_encoder} PE feature cache "
                    f"(*_anima_{repa_encoder}.safetensors) — the REPA "
                    f"alignment loss would be a silent no-op. Run "
                    f"`make preprocess-pe ARGS='--encoder {repa_encoder}'` "
                    f"first, or disable use_repa."
                )
            if present < total:
                logger.warning(
                    f"REPA: only {present}/{total} training images have a "
                    f"{repa_encoder} PE sidecar; the alignment term is skipped "
                    f"for batches missing one. Run `make preprocess-pe "
                    f"ARGS='--encoder {repa_encoder}'` to cover the rest."
                )
            logger.info(
                f"REPA: PE feature loading enabled (encoder={repa_encoder}); "
                f"batches carry repa_pe_features ({present}/{total} cached)."
            )

        # Soft-tokens contrastive negatives. The objective's knobs live in
        # ``network_args`` (see configs/methods/soft_tokens.toml); preview them
        # via the resolved kwargs view to decide whether
        # the dataset should surface cached negative text embeddings. Off unless
        # contrastive_weight > 0. See docs/proposal/soft_tokens_contrastive.md.
        if str(getattr(args, "network_module", "") or "") == (
            "networks.methods.soft_tokens"
        ):
            con_weight = float(net_kwargs.get("contrastive_weight", 0.0) or 0.0)
            if con_weight > 0.0:
                con_k = int(net_kwargs.get("contrastive_k", 1) or 1)
                con_mode = str(net_kwargs.get("contrastive_negative_mode", "shuffled"))
                # The negative grouping comes from this preprocess run's
                # caption index. An explicit run-local path is authoritative:
                # falling back to a global index would silently mix datasets.
                con_index = _resolve_contrastive_index(args)
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
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        v100_flash_stability = _resolve_v100_flash_stability(args)
        debug_finite_checks = bool(
            getattr(args, "debug_finite_checks", False)
        ) or _env_flag("ANIMA_DEBUG_FINITE")

        if attn_mode == "flash4":
            # Flash Attention 4 (flash-attention-sm120) is not supported yet.
            raise RuntimeError(
                "attn_mode='flash4' is not supported yet -- the flash-attention-sm120 "
                "kernel is disabled in this build. Use 'flash', 'torch', 'flex', "
                "or 'sageattn' instead."
            )
        elif attn_mode == "flash":
            from networks.attention_dispatch import flash_attn, flash_attn_func

            if flash_attn_func is None:
                raise RuntimeError(
                    "attn_mode='flash' requested but flash_attn is not available."
                )
            flash_doc, is_v100_fork = _flash_attn_v100_doc(flash_attn)
            try:
                major, minor = torch.cuda.get_device_capability(accelerator.device)
            except Exception:
                major, minor = -1, -1
            logger.info(
                "Using Flash Attention 2 (flash_attn %s), gpu_sm=%s.%s, "
                "v100_fork=%s, v100_flash_stability=%s, debug_finite_checks=%s%s",
                getattr(flash_attn, "__version__", "unknown"),
                major,
                minor,
                is_v100_fork,
                v100_flash_stability,
                debug_finite_checks,
                f", doc={flash_doc}" if flash_doc else "",
            )
            if major == 7 and minor == 0 and is_v100_fork:
                logger.warning(
                    "Detected flash-attention-v100 on Volta/V100. This backend is "
                    "experimental for fp16 diffusion training; use "
                    "v100_flash_stability=hybrid or safe plus ANIMA_DEBUG_FINITE=1 "
                    "to diagnose NaN/Inf without disabling flash entirely."
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
            v100_flash_stability=v100_flash_stability,
            debug_finite_checks=debug_finite_checks,
        )

        # NOTE: torch.compile (compile_blocks) is intentionally NOT done here.
        # It must run AFTER the adapter's apply_to monkey-patches the targeted
        # Linears, or dynamo traces the un-adapted forward — see the compile
        # ordering in library/runtime/harness.py. compile is lazy, so the old
        # compile-here-apply-later ordering happened to work as long as no DiT
        # forward ran in the window; moved to _create_and_apply_network (after
        # apply_to + load_weights + grad-ckpt) so the invariant holds by
        # construction rather than by luck.

        # Store unsloth preference so that when the base trainer calls
        # dit.enable_gradient_checkpointing(), we can override to use unsloth.
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

    # Strategy construction + singleton installation lives in
    # library/anima/strategy.py (setup_training_strategies /
    # setup_text_encoder_outputs_caching_strategy) — the training-side
    # counterpart of library/inference/text.py::ensure_text_strategies.

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None  # no text encoders needed for encoding
        return text_encoders

    def get_noise_scheduler(
        self, args: argparse.Namespace, device: torch.device
    ) -> Any:
        noise_scheduler = noise_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=args.discrete_flow_shift
        )
        return noise_scheduler

    # ------------------------------------------------------------------
    # Per-step forward phases (issues.md P2.1)
    #
    # ``get_noise_pred_and_target`` is a flat sequence of named phases; the
    # conditional logic of each phase lives INSIDE it, never as lexical
    # nesting around it. "Always per step" is therefore structurally evident
    # at the call site — the silent-REPA dispatch bug was exactly an
    # always-phase written inside a sometimes-branch.
    # ------------------------------------------------------------------

    def _step_ctx(self, ctx: TrainCtx) -> StepCtx:
        return StepCtx(
            args=ctx.args,
            accelerator=ctx.accelerator,
            network=ctx.network,
            weight_dtype=ctx.weight_dtype,
        )

    def _prime_adapters(self, ctx: TrainCtx, batch, latents, *, is_train) -> None:
        """ALWAYS per step. Method-adapter pre-forward priming.

        IP-Adapter encodes the reference image and primes per-block K/V;
        EasyControl runs the cond pre-pass and primes per-block (K_c, V_c).
        Both run on the 4D latent layout the patched DiT forward expects. The
        patched cross-attn / self-attn closures consume the primed tensors
        during attention."""
        if not self._adapters:
            return
        step_ctx = self._step_ctx(ctx)
        for adapter in self._adapters:
            adapter.prime_for_forward(step_ctx, batch, latents, is_train=is_train)

    def _sample_noisy_input(self, ctx: TrainCtx, latents, noise, *, is_train):
        """ALWAYS per step. Draw (noisy input, timesteps, sigmas) via the
        sampler registry (M1) and run per-step network router conditioning
        (timestep masks, σ/FEI routers, balance-loss warmup)."""
        args = ctx.args
        sampler_fn = SAMPLER_REGISTRY[getattr(args, "sampler", "default") or "default"]
        sampler_out = sampler_fn(
            SamplerContext(
                args=args,
                noise_scheduler=ctx.noise_scheduler,
                latents=latents,
                noise=noise,
                device=ctx.accelerator.device,
                weight_dtype=ctx.weight_dtype,
            )
        )
        # timesteps are [0,1]-scaled, float32.
        self._hydra_warmup_step = apply_router_conditioning(
            network=ctx.network,
            noisy_model_input=sampler_out.noisy_input,
            timesteps=sampler_out.timesteps,
            is_train=is_train,
            warmup_step=int(getattr(self, "_hydra_warmup_step", 0)),
            max_train_steps=int(getattr(args, "max_train_steps", 0) or 0),
            gradient_accumulation_steps=int(
                getattr(args, "gradient_accumulation_steps", 1) or 1
            ),
        )
        return sampler_out.noisy_input, sampler_out.timesteps, sampler_out.sigmas

    def _prepare_conditioning(
        self, ctx: TrainCtx, batch, text_encoder_conds, noisy_model_input
    ):
        """ALWAYS per step. Returns the device-resident ``PreparedTextConds``
        (both text-conditioning modes; normalized to one uniform
        ``ForwardConditioning`` at the forward call site); fires the
        text-conditioned routers (each gated internally on cached crossattn);
        marks grad-checkpointing inputs."""
        args = ctx.args
        network = ctx.network

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
            device=ctx.accelerator.device,
            weight_dtype=ctx.weight_dtype,
            uncond_crossattn_emb=self._state.uncond_crossattn_1,
        )

        # ChimeraHydra global content router (chimera with
        # ``content_router_source="crossattn"``): fire ONCE per step on the
        # pooled crossattn_emb. apply_router_conditioning ran before text
        # conds were materialized, so the content router lives outside that
        # helper. No-op on non-chimera networks or per-Linear chimera.
        if (
            getattr(network, "use_content_router", False)
            and tc.crossattn_emb is not None
            and hasattr(network, "set_content")
        ):
            network.set_content(tc.crossattn_emb)

        # Network-level GlobalRouter routed on pooled text
        # (``router_source="crossattn_emb"``, route_per_layer=False). Same
        # timing rationale as the content router above — fires once per step
        # on the materialized cross-attn text features. No-op otherwise.
        if (
            getattr(network, "use_crossattn_router", False)
            and tc.crossattn_emb is not None
            and hasattr(network, "set_crossattn_routing")
        ):
            network.set_crossattn_routing(tc.crossattn_emb)

        return tc

    def _get_padding_mask(self, latents, *, weight_dtype, device):
        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask_key = (bs, h_latent, w_latent, weight_dtype, device)
        padding_mask = self._padding_mask_cache.get(padding_mask_key)
        if padding_mask is None:
            padding_mask = torch.zeros(
                bs, 1, h_latent, w_latent, dtype=weight_dtype, device=device
            )
            self._padding_mask_cache[padding_mask_key] = padding_mask
        return padding_mask

    def _run_primary_forward(
        self, ctx: TrainCtx, *, anima, noisy_model_input, timesteps, tc, padding_mask
    ):
        """ALWAYS per step. Single, branch-free forward call site (issues.md
        P2.3): both text-conditioning modes normalize to ONE uniform
        ``ForwardConditioning`` (cond, kw) bundle first — the mode split is
        data prep in ``build_forward_conditioning``, not control flow around
        the call. The normalization (postfix splice runs learned modules)
        must happen inside the primary forward's autocast / grad scope, which
        is why it lives here rather than in ``_prepare_conditioning``.
        Returns ``(model_pred, cond)``; ``cond`` is also consumed by the
        aux-loss and adapter-dispatch phases after the forward."""
        cond = build_forward_conditioning(
            network=ctx.network, tc=tc, timesteps=timesteps
        )
        model_pred = anima(
            noisy_model_input,
            timesteps,
            cond.cond,
            padding_mask=padding_mask,
            **cond.kw,
        )
        return model_pred, cond

    def _attach_aux_losses(
        self,
        ctx: TrainCtx,
        *,
        anima,
        batch,
        latents,
        noise,
        sigmas,
        timesteps,
        noisy_model_input,
        cond: ForwardConditioning,
        padding_mask,
        is_train,
    ) -> None:
        """Trainer-owned aux-loss producers riding the primary forward (func
        inversion loss, VR control variate). Every gate lives INSIDE this
        phase — including the cached-text requirement (``cond.crossattn_emb
        is not None``), which used to be implied by lexical position inside
        the else-branch. Must run inside the primary forward's autocast /
        grad scope (extra ``anima(...)`` calls)."""
        args = ctx.args

        # Functional MSE loss against a sampled stochastic inversion run.
        # The captures dict is populated by trainer-owned forward hooks
        # on cross_attn.output_proj at ``self._func_blocks``.
        self._func_loss = None
        if (
            is_train
            and getattr(self, "_func_blocks", None)
            and cond.crossattn_emb is not None
        ):
            self._func_loss = compute_inversion_func_loss(
                anima_call=anima,
                captures=self._func_captures,
                block_indices=self._func_blocks,
                batch=batch,
                noisy_model_input=noisy_model_input,
                timesteps=timesteps,
                padding_mask=padding_mask,
                has_postfix=cond.has_postfix,
                kw=cond.kw,
                device=ctx.accelerator.device,
                dtype=ctx.weight_dtype,
            )

        # Variance-reduced FM control variate (AsymFlow §5.2). Stash the
        # residual `z` so the loss composer can blend `(y + λ·z)²`.
        if (
            is_train
            and float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0
            and cond.crossattn_emb is not None
        ):
            z_residual = run_vr_reference_forward(
                anima_call=anima,
                network=ctx.network,
                latents=latents,
                noise=noise,
                sigmas=sigmas,
                timesteps=timesteps,
                crossattn_emb=cond.crossattn_emb,
                padding_mask=padding_mask,
                forward_kwargs=cond.kw,
                weight_dtype=ctx.weight_dtype,
                fei_sigma_low_div=float(args.vr_fei_sigma_low_div),
            )
            self._state.extras_for_step["vr"] = {
                "z": z_residual.detach(),
                "state": self._state.vr,
            }

    def _dispatch_adapter_extras(
        self, ctx: TrainCtx, primary: ForwardArtifacts
    ) -> None:
        """ALWAYS per step — both text-conditioning paths. Method-adapter
        extra forwards (soft-tokens, REPA, …).

        This dispatch used to live inside the cached-crossattn else-branch
        only, which silently skipped every adapter's aux loss on the in-model
        text path (crossattn_emb=None — EasyControl's default; REPA trained
        as baseline). Each adapter sees the primary forward's inputs + 5D
        output and may run additional anima(...) calls inside the same
        autocast / grad scope, returning aux loss tensors keyed for the
        LossComposer."""
        if not self._adapters:
            return
        step_ctx = self._step_ctx(ctx)
        for adapter in self._adapters:
            out = adapter.extra_forwards(step_ctx, primary)
            if out:
                self._state.extras_for_step.update(out)

    def get_noise_pred_and_target(
        self,
        ctx: TrainCtx,
        latents,
        batch,
        text_encoder_conds,
        *,
        is_train=True,
    ):
        accelerator = ctx.accelerator
        anima: anima_models.Anima = ctx.unet

        # Reset per-step adapter aux so stale tensors from a prior step can't
        # leak into the loss composer.
        self._state.extras_for_step = {}

        if latents.ndim == 5:  # Fallback for 5D latents (old cache)
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]

        self._prime_adapters(ctx, batch, latents, is_train=is_train)
        noise = torch.randn_like(latents)
        noisy_model_input, timesteps, sigmas = self._sample_noisy_input(
            ctx, latents, noise, is_train=is_train
        )
        tc = self._prepare_conditioning(
            ctx, batch, text_encoder_conds, noisy_model_input
        )
        padding_mask = self._get_padding_mask(
            latents, weight_dtype=ctx.weight_dtype, device=accelerator.device
        )
        noisy_model_input = noisy_model_input.unsqueeze(
            2
        )  # 4D to 5D, [B, C, H, W] -> [B, C, 1, H, W]

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred, cond = self._run_primary_forward(
                ctx,
                anima=anima,
                noisy_model_input=noisy_model_input,
                timesteps=timesteps,
                tc=tc,
                padding_mask=padding_mask,
            )
            self._attach_aux_losses(
                ctx,
                anima=anima,
                batch=batch,
                latents=latents,
                noise=noise,
                sigmas=sigmas,
                timesteps=timesteps,
                noisy_model_input=noisy_model_input,
                cond=cond,
                padding_mask=padding_mask,
                is_train=is_train,
            )
            self._dispatch_adapter_extras(
                ctx,
                ForwardArtifacts(
                    anima_call=anima,
                    noisy_model_input=noisy_model_input,
                    timesteps=timesteps,
                    crossattn_emb=cond.crossattn_emb,
                    padding_mask=padding_mask,
                    forward_kwargs=cond.kw,
                    model_pred=model_pred,
                    noise=noise,
                    latents=latents,
                    is_train=is_train,
                ),
            )
        model_pred = model_pred.squeeze(2)  # 5D to 4D, [B, C, 1, H, W] -> [B, C, H, W]

        # Note: do NOT clear timestep mask here -- gradient checkpointing recomputes the forward
        # pass during backward, so the mask must remain set. It gets overwritten on the next step.

        # Rectified flow target: noise - latents
        target = noise - latents

        # Loss weighting
        weighting = anima_train_utils.compute_loss_weighting_for_anima(
            weighting_scheme=ctx.args.weighting_scheme, sigmas=sigmas
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
        network=None,
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
            network=network,
            progress_sink=getattr(self, "progress_sink", None),
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
                    latents = vae.encode_pixels_to_latents(
                        batch["images"].to(accelerator.device, dtype=vae_dtype)
                    )
                else:
                    chunks = [
                        batch["images"][i : i + args.vae_batch_size]
                        for i in range(0, len(batch["images"]), args.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        with torch.no_grad():
                            chunk = vae.encode_pixels_to_latents(
                                chunk.to(accelerator.device, dtype=vae_dtype)
                            )
                            list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(
                        torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents)
                    )

        text_encoder_conds = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = (
                text_encoder_outputs_list  # List of text encoder outputs
            )

        if (
            len(text_encoder_conds) == 0
            or all(c is None for c in text_encoder_conds)
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

            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        # Step-owning adapter override: a method with no `target = noise -
        # latents` and its own multi-forward objective (BYG) computes the whole
        # scalar loss here, bypassing get_noise_pred_and_target + LossComposer.
        owners = [a for a in self._adapters if a.owns_training_step(args)]
        if owners:
            assert len(owners) == 1, (
                f"at most one adapter may own the training step; got {len(owners)}: "
                f"{[a.name for a in owners]}"
            )
            return owners[0].compute_loss(
                ComputeLossCtx(
                    args=args,
                    accelerator=accelerator,
                    network=getattr(self, "_network", network),
                    unet=ctx.unet,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                    batch=batch,
                    latents=latents,
                    text_encoder_conds=text_encoder_conds,
                    is_train=is_train,
                )
            )

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

        composer = build_loss_composer(
            args, getattr(self, "_network", network), ledger=self._liveness
        )

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
        # Aux-loss gating convention (library/training/losses.py docstring):
        # handlers read network._<name>_weight. functional's weight is a
        # top-level training arg, so the trainer stamps it here.
        network._functional_loss_weight = float(
            getattr(args, "functional_loss_weight", 0.0) or 0.0
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
        metadata["ss_sigmoid_bias"] = getattr(args, "sigmoid_bias", 0.0)
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift
        # Hardware-dependent block-swap policy is part of the reproducibility
        # record.  Keep it in safetensors metadata as well as run_manifest.json
        # so an exported adapter still explains why compile was disabled.
        metadata["ss_blocks_to_swap"] = getattr(args, "blocks_to_swap", 0) or 0
        metadata["ss_block_swap_reliable_mode"] = bool(
            getattr(args, "block_swap_reliable_mode", False)
        )
        metadata["ss_block_swap_gpu_sm"] = getattr(args, "block_swap_gpu_sm", None)
        metadata["ss_block_swap_effective_torch_compile"] = bool(
            getattr(args, "torch_compile", False)
        )
        metadata["ss_block_swap_effective_lora_fp32_compute"] = bool(
            getattr(args, "block_swap_effective_lora_fp32_compute", False)
        )
        metadata["ss_block_swap_effective_use_custom_down_autograd"] = bool(
            getattr(args, "block_swap_effective_use_custom_down_autograd", False)
        )

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
        step_ctx = self._step_ctx(ctx)
        for adapter in self._adapters:
            adapter.on_step_start(step_ctx, batch, is_train=is_train)

    def run_after_backward(self, ctx: TrainCtx):
        """Dispatch the post-backward hook to adapters (between
        ``accelerator.backward`` and gradient clipping)."""
        if not self._adapters:
            return
        step_ctx = self._step_ctx(ctx)
        for adapter in self._adapters:
            adapter.after_backward(step_ctx)

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only

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
                            encoded = text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                text_encoders,
                                tokens_and_masks,
                            )
                            # Sampling copies these back to the target device.
                            sample_prompts_te_outputs[p] = [
                                value.detach().cpu()
                                if torch.is_tensor(value)
                                else value
                                for value in encoded
                            ]
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

    def _prepare_dataset(self, args) -> DatasetBundle:
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
                    config_file=(
                        getattr(args, "config_file", None)
                        if getattr(args, "method", None) is None
                        else None
                    ),
                )
                if base_ds is not None:
                    if getattr(args, "method", None) is None and getattr(
                        args, "config_file", None
                    ):
                        logger.info("Loading dataset config from config_file")
                    else:
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
                    # Free-fit (the only resize mode): the predefined bucket set is
                    # the union of the on-disk resized sizes, so every cached latent
                    # exact-matches its own (W, H) and nothing AR-snaps. target_res
                    # is preprocess-only and inert here — the on-disk caches decide
                    # which tiers/shapes are present, not this list.
                    target_res=getattr(args, "target_res", None),
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
        for dataset_group in (train_dataset_group, val_dataset_group):
            if dataset_group is not None and hasattr(dataset_group, "set_shared_epoch"):
                dataset_group.set_shared_epoch(current_epoch)
        ds_for_collator = (
            train_dataset_group if args.max_data_loader_n_workers == 0 else None
        )
        collator = collator_class(current_epoch, current_step, ds_for_collator)

        return DatasetBundle(
            train_group=train_dataset_group,
            val_group=val_dataset_group,
            current_epoch=current_epoch,
            current_step=current_step,
            collator=collator,
            use_user_config=use_user_config,
            use_dreambooth_method=use_dreambooth_method,
        )

    def _derive_token_budget(self, args, train_group, val_group):
        """(n_token_families, seq_range) from the buckets the datasets populate.

        Reads each dataset's ``bucket_manager.resos`` (the buckets at least one
        selected image landed in) and reduces to the set of distinct token counts,
        unioned with the token counts the sample prompts will request (see
        ``_sample_prompt_token_counts``). This sizes ``compile_blocks``' dynamo
        cache to exactly the tiers on disk for this run — independent of
        ``args.target_res``. Returns ``(None, None)`` when no bucketed resos are
        available (e.g. a MinimalDataset), leaving compile_blocks on its own
        defaults.
        """
        from library.datasets.buckets import token_counts_for_resos

        resos: set = set()
        for group in (train_group, val_group):
            if group is None:
                continue
            for dataset in getattr(group, "datasets", []):
                bm = getattr(dataset, "bucket_manager", None)
                if bm is not None:
                    resos.update(bm.resos)
        if not resos:
            return None, None
        counts = token_counts_for_resos(resos) | self._sample_prompt_token_counts(args)
        return len(counts), (min(counts), max(counts))

    def _sample_prompt_token_counts(self, args) -> set:
        """Token counts the sample prompts will request; empty when sampling is off.

        Sample generation runs through the same compiled blocks as training, so a
        sample resolution outside the training buckets (e.g. ``--w 1024 --h 1536``
        over 1024-tier data) would land outside the dynamic-seq mark_dynamic range
        and crash the run mid-training with a ConstraintViolationError (#42).
        Folding the prompt resolutions into the budget compiles for them up front.
        Prompts are re-read from disk at every sample event, so resolutions added
        to the file mid-run are NOT covered here — those are skipped with a
        warning at sample time instead (``_sample_image_inference``).
        """
        from library.datasets.buckets import token_counts_for_sample_prompts

        if not getattr(args, "sample_prompts", None):
            return set()
        will_sample = (
            getattr(args, "sample_at_first", False)
            or getattr(args, "sample_every_n_steps", None)
            or getattr(args, "sample_every_n_epochs", None)
        )
        if not will_sample:
            return set()
        try:
            prompts = train_util.load_prompts(args.sample_prompts)
        except Exception as e:
            logger.warning(
                f"Could not parse sample prompts ({args.sample_prompts}) for the "
                f"compile token budget: {e}. Sample resolutions outside the "
                "training buckets may be skipped under torch_compile."
            )
            return set()
        return token_counts_for_sample_prompts(prompts)

    def _create_and_apply_network(
        self,
        args,
        accelerator,
        vae,
        text_encoder,
        unet,
        text_encoders,
        weight_dtype,
    ) -> Optional[NetworkBundle]:
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

        # prepare network — one resolved view of both config-intake paths
        # (--network_args + allowlisted top-level keys). Copied so the dropout
        # default below stays a factory-call detail, not part of the cached
        # ``args._network_kwargs`` view other consumers read.
        net_kwargs = _apply_v100_adapter_runtime_policy(
            args, accelerator, dict(resolve_network_kwargs(args))
        )

        factory_weights_sd = None
        if args.dim_from_weights:
            network, factory_weights_sd = network_module.create_network_from_weights(
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
            if factory_weights_sd is not None:
                info = network.load_state_dict(factory_weights_sd, strict=False)
            else:
                info = network.load_weights(args.network_weights)
            accelerator.print(
                f"load network weights from {args.network_weights}: {info}"
            )

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

            for t_enc, flag in zip(
                text_encoders, self.get_text_encoders_train_flags(args, text_encoders)
            ):
                if flag:
                    if t_enc.supports_gradient_checkpointing:
                        t_enc.gradient_checkpointing_enable()
            network.enable_gradient_checkpointing()  # may have no effect

        # Native-shape flattening + per-block torch.compile. COMPILE LAST —
        # after apply_to + load_weights (above) so dynamo traces the adapter's
        # monkey-patched Linear forwards, not the bare DiT. The full sequence
        # (partitioner activation-memory budget → per-signature cache
        # isolation → compile_blocks → EasyControl cond-stream compile) lives
        # in library/runtime/harness.py with the other compile entry points.
        # Matches the harness order: block-swap → grad-ckpt → compile.

        # fp16 overflow guard — MUST run before compile_blocks below: dynamo
        # specializes block._forward on the per-module ``fp32_residual`` bool,
        # so flipping it post-compile trips the guard and recompiles every
        # block graph on the first forward. No-op on bf16 (default) / fp32.
        # See Anima.enable_fp32_residual() for the full rationale
        # (docs/findings/selfflow.md).
        if args.mixed_precision == "fp16" and isinstance(unet, anima_models.Anima):
            unet.enable_fp32_residual()
            logger.info(
                "fp16 mixed precision: enabled fp32 residual accumulation "
                "(DiT residual stream exceeds fp16 range; prevents NaN). "
                "Sublayer matmuls still run fp16 under autocast; bf16/fp32 "
                "runs are unaffected."
            )

        maybe_apply_convrot_base(
            args,
            network,
            unet=unet,
            network_kwargs=net_kwargs,
        )

        if args.torch_compile:
            # Block swap coexists with compile: compile_blocks compiles only the
            # resident head blocks and leaves the tail swap blocks eager (their
            # .weight.data CPU↔GPU swaps would trip dynamo's device guard). So no
            # special-casing here — see Anima.compile_blocks.
            from library.runtime.harness import compile_blocks_for_training

            # Token-family budget derived from the buckets the dataset actually
            # populated (see _derive_token_budget) — not args.target_res, which is
            # a preprocess-only knob and inert at train time.
            n_token_families, seq_range = getattr(
                self, "_compile_token_budget", (None, None)
            )
            compile_blocks_for_training(
                unet,
                network,
                backend=args.dynamo_backend,
                mode=getattr(args, "compile_inductor_mode", None),
                n_token_families=n_token_families,
                seq_range=seq_range,
                dynamic_seq=bool(getattr(args, "compile_dynamic_seq", False)),
                activation_memory_budget=float(
                    getattr(args, "activation_memory_budget", 1.0) or 1.0
                ),
                partitioner_recompute_views=bool(
                    getattr(args, "partitioner_recompute_views", False)
                ),
                partitioner_aggressive_recomputation=bool(
                    getattr(args, "partitioner_aggressive_recomputation", False)
                ),
                grad_ckpt=bool(getattr(args, "gradient_checkpointing", False)),
                logger=logger,
            )

        return NetworkBundle(
            network=network,
            net_kwargs=net_kwargs,
            train_unet=train_unet,
            train_text_encoder=train_text_encoder,
        )

    def _setup_optimizer_and_dataloader(
        self,
        args,
        accelerator,
        network,
        model,
        train_dataset_group,
        val_dataset_group,
        collator,
    ) -> OptimizerBundle:
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

        # Do this after the real adapter optimizer exists, but before
        # ``accelerator.prepare`` can move/wrap it.  Block swapping only saves
        # resident frozen DiT blocks; adapter parameters, gradients, optimizer
        # state and the largest token-family workspace remain live.  Failing at
        # this point gives an actionable message before a misleading first-step
        # OOM or skipped optimizer update.
        if int(getattr(args, "blocks_to_swap", 0) or 0) > 0:
            prepare_swap = getattr(model, "prepare_block_swap_before_forward", None)
            if callable(prepare_swap):
                # ``enable_block_swap`` is installed during model loading, but
                # the tail weights are not parked until the first accelerator
                # preparation. Park them once here so the measured free memory
                # reflects the actual block-swap residency rather than the
                # full DiT, while keeping the check before accelerator.prepare.
                prepare_swap()
            check_block_swap_budget(
                args,
                model=model,
                network=network,
                optimizer=optimizer,
                token_budget=getattr(self, "_compile_token_budget", None),
                device=getattr(accelerator, "device", None),
            )
        optimizer_train_fn, optimizer_eval_fn = get_optimizer_train_eval_fn(
            optimizer, args
        )

        # prepare dataloader
        train_dataset_group.set_current_strategies()
        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()

        stage_plan = prepare_stage_runtime(args, train_dataset_group)
        schedule_on = stage_plan is not None
        if stage_plan is not None:
            if getattr(args, "max_train_epochs", None) is not None:
                logger.warning(
                    "stage_schedule with max_train_epochs derives max_train_steps "
                    "from the first stage length; set max_train_steps explicitly "
                    "for a stable percentage budget"
                )

        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        persistent_workers = args.persistent_data_loader_workers and n_workers > 0
        if schedule_on and persistent_workers:
            logger.info(
                "disabling persistent DataLoader workers for stage_schedule hot switches"
            )
            persistent_workers = False

        dataloader_kwargs = {
            "batch_size": 1,
            "collate_fn": collator,
            "num_workers": n_workers,
            "persistent_workers": persistent_workers,
            "pin_memory": args.dataloader_pin_memory,
        }
        train_dataloader_kwargs = dict(dataloader_kwargs)
        if stage_plan is not None:
            stage_plan.loader_generator = torch.Generator()
            stage_plan.loader_generator.manual_seed(int(args.seed or 0))
            train_dataloader_kwargs["generator"] = stage_plan.loader_generator
        if n_workers > 0:
            dataloader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
            train_dataloader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            shuffle=True,
            **train_dataloader_kwargs,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset_group if val_dataset_group is not None else [],
            shuffle=False,
            **dataloader_kwargs,
        )

        # Calculate training steps. Explicit max_train_steps is authoritative;
        # epochs are only a derivation source when no step budget was supplied.
        _finalize_training_budget(
            args,
            dataloader_length=len(train_dataloader),
            num_processes=accelerator.num_processes,
        )
        accelerator.print(
            "training budget: "
            f"source={args.training_budget_source}, "
            f"origin={args.training_budget_origin}, "
            f"max_train_steps={args.max_train_steps}"
        )

        train_dataset_group.set_max_train_steps(args.max_train_steps)

        if stage_plan is not None:
            stage_plan.dataloader_kwargs = dict(train_dataloader_kwargs)

        # lr scheduler
        lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

        return OptimizerBundle(
            optimizer=optimizer,
            optimizer_name=optimizer_name,
            optimizer_args=optimizer_args,
            optimizer_train_fn=optimizer_train_fn,
            optimizer_eval_fn=optimizer_eval_fn,
            text_encoder_lr=text_encoder_lr,
            lr_descriptions=lr_descriptions,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            lr_scheduler=lr_scheduler,
            stage_plan=stage_plan,
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
    ) -> AcceleratedBundle:
        """Cast model dtypes, run accelerator.prepare, flip train/eval, optional torch.compile."""
        unet_weight_dtype = te_weight_dtype = weight_dtype

        unet.requires_grad_(False)
        unet.to(dtype=unet_weight_dtype)

        for i, t_enc in enumerate(text_encoders):
            # None when the TE was never loaded (cache_text_encoder_outputs with
            # no sample prompts / val / TE-training -- qwen3_needed=False).
            if t_enc is None:
                continue
            t_enc.requires_grad_(False)

            # in case of cpu, dtype is already set to fp32 because cpu does not support fp16/bf16
            if t_enc.device.type != "cpu":
                t_enc.to(dtype=te_weight_dtype)

        # accelerator preparation (no deepspeed)
        if train_unet:
            unet = self.prepare_unet_with_accelerator(args, accelerator, unet)
        else:
            unet.to(
                accelerator.device,
                dtype=unet_weight_dtype,
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

        # Caches are complete and the frozen encoder has no remaining work.
        # Drop every alias so the large CPU model is not retained for the run.
        if args.cache_text_encoder_outputs and not train_text_encoder:
            logger.info("releasing cached-only text encoder from training state")
            text_encoder, text_encoders = release_text_encoder_handles(
                text_encoder, text_encoders
            )
            gc.collect()
            clean_memory_on_device(accelerator.device)

        if not cache_latents:
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        return AcceleratedBundle(
            network=network,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            lr_scheduler=lr_scheduler,
            training_model=training_model,
            unet=unet,
            text_encoders=text_encoders,
            text_encoder=text_encoder,
            unet_weight_dtype=unet_weight_dtype,
        )

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        selected_preprocess_run = _apply_preprocess_run(args)
        output_layout = layout_from_args(args)
        # Resolve block-swap compatibility and the hardware-dependent reliable
        # mode before computing the resume signature or constructing
        # Accelerator.  A V100 must never reach accelerator/model setup with a
        # stale high-peak compile request.
        _validate_block_swap_config(args)
        _resolve_block_swap_reliable_mode(args)
        args.config_signature = _resume_config_signature(args)
        # The daemon supplies ANIMA_DAEMON_STOP_FILE on Windows and sends a
        # process-group signal on Linux.  The controller only flips a flag;
        # loop.py consumes it after a complete optimizer step.
        self._stop_controller = StopController.from_environment()
        configure_staged_resolution(args)
        normalize_sample_args(args)
        verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        setup_logging(args, reset=True)

        # Free-fit is the only resize mode and it requires compile_dynamic_seq: a
        # free-fit pool populates many distinct (W, H) within one tier's token
        # band, which would explode the static N-graph compile cascade. dynamic_seq
        # marks only the seq axis dynamic over the band → a single graph per tier.
        # Auto-enable it whenever compile is on (no-op if torch_compile is off).
        if getattr(args, "torch_compile", False):
            if not getattr(args, "compile_dynamic_seq", False):
                logger.info(
                    "auto-enabling --compile_dynamic_seq "
                    "(free-fit shapes need the single-graph dynamic-seq path)"
                )
                args.compile_dynamic_seq = True

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

        # Build + install the strategy singletons (tokenize / latents-caching /
        # text-encoding). Must run before _prepare_dataset — dataset init reads
        # the tokenize + latents-caching strategies. The TE-OUTPUTS caching
        # strategy is installed separately below, after assert_extra_args has
        # had its chance to mutate cache_llm_adapter_outputs.
        strategies = strategy_anima.setup_training_strategies(args)
        tokenize_strategy = strategies.tokenize
        text_encoding_strategy = strategies.text_encoding
        tokenizers = [
            tokenize_strategy.qwen3_tokenizer
        ]  # will be removed after sample_image is refactored

        ds = self._prepare_dataset(args)
        train_dataset_group = ds.train_group
        val_dataset_group = ds.val_group
        _validate_preprocess_dataset_paths(
            train_dataset_group, selected_preprocess_run
        )
        _validate_preprocess_dataset_paths(val_dataset_group, selected_preprocess_run)
        args.dataset_signature = signature(
            {
                "train": _dataset_group_signature(train_dataset_group),
                "val": _dataset_group_signature(val_dataset_group),
            }
        )
        current_epoch = ds.current_epoch
        current_step = ds.current_step
        collator = ds.collator
        use_user_config = ds.use_user_config
        use_dreambooth_method = ds.use_dreambooth_method

        # Derive the torch.compile token-family budget from the buckets the
        # selected (path_pattern-filtered) images actually populate — NOT from
        # args.target_res. The on-disk caches are the source of truth for which
        # tiers are present, so this can't drift from preprocess, and a filtered
        # run sizes the dynamo cache to only the families it really uses. Sample
        # prompt resolutions are folded in (when sampling is enabled) so sample
        # generation outside the training buckets compiles instead of crashing.
        self._compile_token_budget = self._derive_token_budget(
            args, train_dataset_group, val_dataset_group
        )

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

        # Install the text-encoder-outputs caching strategy now: after
        # assert_extra_args (which may flip cache_llm_adapter_outputs, read by
        # the strategy ctor) and before the model load, so the
        # cache-completeness probe below can use it to decide whether the
        # Qwen3 text encoder needs loading at all.
        strategy_anima.setup_text_encoder_outputs_caching_strategy(args)

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

        # Resolve mixed precision BEFORE prepare_accelerator: Accelerator() bakes
        # the autocast dtype at construction, and the fp32-residual guard / metadata
        # read args.mixed_precision later. _resolve_mixed_precision back-writes args
        # so all consumers agree.
        _resolve_mixed_precision(args)

        # Prepare accelerator
        logger.info("preparing accelerator")
        accelerator = prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        # mixed precision dtype
        weight_dtype, save_dtype = prepare_dtype(args)
        vae_dtype = _resolve_vae_dtype(args, weight_dtype)

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

        # cache text encoder outputs if needed: Text Encoder is moved to cpu or
        # gpu (the encoding strategy was installed with the others up top).
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
            from library.preprocess.uncond import ensure_uncond_crossattn

            self._state.uncond_crossattn_1 = ensure_uncond_crossattn(
                qwen3_path=args.qwen3,
                dit_path=args.pretrained_model_name_or_path,
                t5_tokenizer_path=getattr(args, "t5_tokenizer_path", None),
                device=accelerator.device,
                dtype=weight_dtype,
                existing=self._state.uncond_crossattn_1,
            )

        net = self._create_and_apply_network(
            args, accelerator, vae, text_encoder, unet, text_encoders, weight_dtype
        )
        if net is None:
            return
        network = net.network
        net_kwargs = net.net_kwargs
        train_unet = net.train_unet
        train_text_encoder = net.train_text_encoder

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

        opt = self._setup_optimizer_and_dataloader(
            args,
            accelerator,
            network,
            unet,
            train_dataset_group,
            val_dataset_group,
            collator,
        )
        optimizer = opt.optimizer
        optimizer_name = opt.optimizer_name
        optimizer_args = opt.optimizer_args
        optimizer_train_fn = opt.optimizer_train_fn
        optimizer_eval_fn = opt.optimizer_eval_fn
        text_encoder_lr = opt.text_encoder_lr
        lr_descriptions = opt.lr_descriptions
        train_dataloader = opt.train_dataloader
        val_dataloader = opt.val_dataloader
        lr_scheduler = opt.lr_scheduler
        stage_plan = opt.stage_plan

        acc = self._prepare_with_accelerator(
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
        network = acc.network
        optimizer = acc.optimizer
        train_dataloader = acc.train_dataloader
        val_dataloader = acc.val_dataloader
        lr_scheduler = acc.lr_scheduler
        training_model = acc.training_model
        unet = acc.unet
        text_encoders = acc.text_encoders
        text_encoder = acc.text_encoder
        unet_weight_dtype = acc.unet_weight_dtype
        del acc

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if stage_plan is not None:
            stage_budget = stage_epoch_upper_bound(
                stage_plan.stages,
                args.max_train_steps,
                stage_plan.target_batch_counts,
                num_processes=accelerator.num_processes,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
            num_train_epochs = max(num_train_epochs, stage_budget)

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
                    log_dir=resolve_run_log_dir(args),
                    sampling_enabled=sampling_enabled,
                    config_signature=getattr(args, "config_signature", None),
                    dataset_signature=getattr(args, "dataset_signature", None),
                )
                # Mirror WARNING+ records into the stream so a reader debugging
                # the run gets them structured instead of buried in tqdm stdout.
                self.progress_sink.attach_log_mirror()

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
            num_train_images=(
                stage_plan.full_num_train_images
                if stage_plan is not None
                else train_dataset_group.num_train_images
            ),
            num_val_images=val_dataset_group.num_train_images
            if val_dataset_group is not None
            else 0,
            num_reg_images=(
                stage_plan.full_num_reg_images
                if stage_plan is not None
                else train_dataset_group.num_reg_images
            ),
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
            dataset_counts=(
                list(stage_plan.full_dataset_counts) if stage_plan is not None else None
            ),
        )
        if stage_plan is not None:
            metadata["ss_stage_schedule"] = json.dumps(stage_plan.as_dicts())
            metadata["ss_stage_index"] = str(stage_plan.initial_index)
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
        loaded_train_state = saver.loaded_train_state
        if loaded_train_state.get("rng_state"):
            restore_rng_state(loaded_train_state.get("rng_state"))

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
        initial_global_step = None
        stage_batch_cursor = 0
        stage_loader_generator_state = None
        if stage_plan is not None and initial_step > 0:
            initial_global_step = initial_step
            expected_stage_index = stage_plan.index_for_step(
                initial_step, args.max_train_steps
            )
            saved_stage_index = loaded_train_state.get("stage_index")
            if (
                saved_stage_index is not None
                and int(saved_stage_index) != expected_stage_index
            ):
                raise ValueError(
                    "resume stage does not match the active stage_schedule: "
                    f"state={saved_stage_index}, expected={expected_stage_index}"
                )
            if (
                args.skip_until_initial_step
                and "stage_batch_cursor" in loaded_train_state
            ):
                epoch_to_start = max(
                    0, int(loaded_train_state.get("stage_outer_epoch", 0))
                )
                stage_batch_cursor = max(
                    0, int(loaded_train_state.get("stage_batch_cursor", 0))
                )
                raw_generator_state = loaded_train_state.get(
                    "stage_loader_generator_state"
                )
                if raw_generator_state is not None:
                    stage_loader_generator_state = torch.tensor(
                        raw_generator_state, dtype=torch.uint8
                    )
                    stage_plan.loader_generator.set_state(stage_loader_generator_state)
                initial_step = stage_batch_cursor
            else:
                logger.warning(
                    "stage_schedule resume state has no stage-local cursor; "
                    "continuing from the start of the selected stage loader"
                )
                initial_step = 0
        elif initial_step > 0:
            if args.skip_until_initial_step:
                if not args.resume:
                    logger.info(
                        "initial_step is specified but not resuming. lr scheduler will be started from the beginning"
                    )
                initial_global_step = initial_step
                batches_per_epoch = len(train_dataloader)
                explicit_cursor = (
                    int(loaded_train_state.get("schema_version", 1) or 1) >= 2
                    and "current_epoch" in loaded_train_state
                    and "micro_batch_offset" in loaded_train_state
                )
                if explicit_cursor:
                    try:
                        epoch_to_start, initial_step = (
                            resolve_persisted_resume_position(
                                initial_global_step,
                                batches_per_epoch,
                                args.gradient_accumulation_steps,
                                current_epoch=loaded_train_state["current_epoch"],
                                micro_batch_offset=loaded_train_state[
                                    "micro_batch_offset"
                                ],
                            )
                        )
                    except ValueError as exc:
                        # A few early schema-v2 snapshots were written before
                        # the cursor was made absolute across a resumed epoch.
                        # Keep those states usable through the legacy step
                        # conversion, while refusing to silently trust a
                        # contradictory cursor.
                        logger.warning(
                            "ignoring inconsistent persisted resume cursor: %s",
                            exc,
                        )
                        epoch_to_start, initial_step = resolve_resume_position(
                            initial_global_step,
                            batches_per_epoch,
                            args.gradient_accumulation_steps,
                        )
                else:
                    epoch_to_start, initial_step = resolve_resume_position(
                        initial_global_step,
                        batches_per_epoch,
                        args.gradient_accumulation_steps,
                    )
                logger.info(
                    "resolved resume position: global_step=%s, "
                    "start_epoch=%s, batch_offset=%s",
                    initial_global_step,
                    epoch_to_start,
                    initial_step,
                )
            else:
                epoch_to_start = initial_step // math.ceil(
                    len(train_dataloader) / args.gradient_accumulation_steps
                )
                initial_step = 0  # do not skip

        # Stage schedules keep the mutable dataset group so the loop can rebuild
        # membership and its DataLoader. Ordinary runs drop the local as before.
        if stage_plan is None:
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
            stage_plan=stage_plan,
            initial_global_step=initial_global_step,
            stage_batch_cursor=stage_batch_cursor,
            stage_loader_generator_state=stage_loader_generator_state,
            stop_controller=getattr(self, "_stop_controller", None),
        )
        loop_state.stage_index = (
            stage_plan.initial_index if stage_plan is not None else -1
        )
        saver.set_runtime_state_provider(loop_state.checkpoint_runtime_state)

        # run_scope emits the matching run_end (ok / stopped / error) on exit;
        # run_start already fired when the sink was constructed above.
        with run_scope(
            self.progress_sink,
            final_step=lambda: loop_state.global_step,
            extra_fields=self._liveness.run_end_fields,
            stopped=lambda: loop_state.stop_requested,
        ):
            loop_completed = False
            try:
                run_training_loop(self, loop_state)
                loop_completed = True
            except BaseException:
                # Python exceptions can often still persist the most recently
                # committed optimizer state. SIGKILL, host OOM and power loss
                # cannot reach this block and fall back to the last complete
                # atomic state directory.
                if loop_state.at_optimizer_boundary:
                    try:
                        optimizer_eval_fn()
                        saver.save_interrupt_state(
                            network,
                            loop_state.global_step,
                            max(1, int(current_epoch.value)),
                        )
                    except Exception:
                        logger.exception("best-effort exception checkpoint failed")
                else:
                    # A failure in the middle of gradient accumulation cannot
                    # be represented by an Accelerate state (partial gradients
                    # are not a committed optimizer step). Keep the previous
                    # complete rolling/interrupted state instead of publishing
                    # a cursor that would silently skip the unfinished window.
                    logger.warning(
                        "skipping exception checkpoint inside an optimizer step; "
                        "resume will use the last complete state"
                    )
                raise
            finally:
                if loop_state.stop_requested:
                    # A cooperative stop has already committed the interrupt
                    # checkpoint at an optimizer boundary.  Accelerate's
                    # teardown can wait on DataLoader/CUDA/NCCL worker state
                    # after that point, defeating the daemon's stop deadline;
                    # the daemon path hard-exits after run_scope flushes the
                    # terminal progress event below.
                    logger.info(
                        "skipping accelerator cleanup after cooperative stop"
                    )
                elif not loop_completed:
                    try:
                        accelerator.end_training()
                    except Exception:
                        logger.exception("accelerator cleanup failed")

            if not loop_state.stop_requested:
                optimizer_eval_fn()

            if loop_state.stop_requested:
                logger.info(
                    "training stopped cooperatively at global_step=%s; "
                    "deferred sample latents remain in the job directory",
                    loop_state.global_step,
                )
                if is_main_process:
                    write_run_manifest(
                        output_layout,
                        {
                            "status": "stopped",
                            "global_step": loop_state.global_step,
                            "config_signature": getattr(args, "config_signature", None),
                            "dataset_signature": getattr(args, "dataset_signature", None),
                            "preprocess_run": getattr(args, "preprocess_run", None),
                            **_training_budget_manifest_fields(args),
                            **_block_swap_manifest_fields(args),
                        },
                    )

            else:
                # Catch-all sample decode for any latents not already decoded
                # inline. Explicit stop skips this path so preview work can never
                # consume the daemon's stop timeout.
                _decode_pending_samples_at_exit(
                    accelerator,
                    args,
                    vae,
                    loop_state.unet,
                    is_main_process=is_main_process,
                    progress_sink=getattr(self, "progress_sink", None),
                )

                # Every rank contributes its Accelerate RNG/sampler payload;
                # the checkpoint helper publishes the shared directory on rank 0.
                if args.save_state or args.save_state_on_train_end:
                    save_state_on_train_end(args, accelerator)

                saver.cleanup_resumable()
                saver.save_final(network, loop_state.global_step, num_train_epochs)
                if is_main_process:
                    write_run_manifest(
                        output_layout,
                        {
                            "status": "done",
                            "global_step": loop_state.global_step,
                            "config_signature": getattr(args, "config_signature", None),
                            "dataset_signature": getattr(args, "dataset_signature", None),
                            "preprocess_run": getattr(args, "preprocess_run", None),
                            **_training_budget_manifest_fields(args),
                            **_block_swap_manifest_fields(args),
                        },
                    )

                # ``end_training`` destroys the distributed process group. It
                # must run only after every rank has contributed its final state
                # and rank 0 has published the model/manifest.
                try:
                    accelerator.end_training()
                except Exception:
                    logger.exception("accelerator cleanup failed")

        # The daemon's cooperative-stop contract ends at the flushed
        # ``run_end(stopped)`` event.  Do not enter Python/CUDA interpreter
        # teardown afterward: DataLoader workers and accelerator atexit hooks
        # can block indefinitely on a V100.  ``os._exit`` mirrors the profiler
        # path in ``library/training/loop.py``; the OS releases all GPU/process
        # resources and the accelerate launcher observes a clean rc=0.
        if loop_state.stop_requested and os.environ.get("ANIMA_DAEMON_STOP_FILE"):
            getattr(self, "_stop_controller", None) and self._stop_controller.close()
            _hard_exit_after_cooperative_stop(True)

        # Remove the TensorBoard log dir for runs shorter than 2 steps — they
        # add noise to the runs list (e.g. aborted starts, dry-runs) and carry
        # no useful loss curves.
        if is_main_process and loop_state.global_step < 2:
            _cleanup_short_log_dir(args)
        getattr(self, "_stop_controller", None) and self._stop_controller.close()

    # endregion


def _decode_pending_samples_at_exit(
    accelerator,
    args,
    vae,
    unet,
    *,
    is_main_process: bool,
    progress_sink=None,
) -> None:
    """Best-effort normal/error preview drain; explicit stops bypass it."""
    if not is_main_process or not args.sample_prompts:
        return
    try:
        accelerator.unwrap_model(unet).to("cpu")
    except Exception as exc:
        logger.warning("could not park DiT before deferred sample decode: %s", exc)
    clean_memory_on_device(accelerator.device)
    try:
        anima_train_utils.decode_pending_samples(
            accelerator,
            args,
            vae,
            progress_sink=progress_sink,
        )
    except Exception as exc:
        logger.error("deferred sample decode during cleanup failed: %s", exc)


def _cleanup_short_log_dir(args) -> None:
    """Delete the TensorBoard log dir when a run completed fewer than 2 steps."""
    import shutil

    log_dir = resolve_run_log_dir(args)
    if log_dir is None:
        return
    try:
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
    except Exception as e:
        print(
            f"warn: could not remove short-run log dir {log_dir}: {e}", file=sys.stderr
        )


def _hard_exit_after_cooperative_stop(stop_requested: bool) -> None:
    """Exit a daemon-owned stopped worker after its terminal event is flushed.

    ``sys.exit`` runs Python/CUDA atexit handlers, which can wait forever for a
    DataLoader worker or an Accelerate process group that has already been
    interrupted.  The interrupted checkpoint and ``run_end(stopped)`` have
    been committed before this helper is called; the OS can reclaim the rest
    of the process tree safely.
    """
    if not stop_requested or not os.environ.get("ANIMA_DAEMON_STOP_FILE"):
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


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

    add_network_arguments(parser)
    add_validation_arguments(parser)
    add_train_misc_arguments(parser)
    return parser


from library.config import schema as _config_schema  # noqa: E402
from networks import all_network_kwargs as _all_network_kwargs  # noqa: E402


# Network-module-consumed flags (networks.lora_anima / networks.methods.*).
# These don't flow through argparse directly because `create_network` reads
# them from ``kwargs``. Sourced from the flat ``NETWORK_KWARGS`` allowlist in
# ``networks/__init__.py`` so adding a key there automatically registers it
# here.
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


def resolve_network_kwargs(args) -> dict[str, str]:
    """The single intake for network kwargs, merging both config paths.

    A network kwarg can arrive as ``--network_args k=v`` (CLI / method TOML
    ``network_args`` list) or as an allowlisted top-level config key landing
    on ``args``; CLI ``--network_args`` win on overlap. Consumers outside the
    network factory (e.g. the REPA dataset-sidecar enable in
    ``assert_extra_args``) must see the same merged view the factory gets, so
    the result is cached on ``args._network_kwargs`` — read a kwarg from here
    rather than scanning ``args.network_args`` with a ``getattr(args, …)``
    fallback. All values are strings, as ``create_network(**kwargs)`` expects.
    """
    cached = getattr(args, "_network_kwargs", None)
    if cached is not None:
        return cached

    net_kwargs: dict[str, str] = {}
    for net_arg in getattr(args, "network_args", None) or []:
        key, value = net_arg.split("=", 1)
        net_kwargs[key] = value

    # Forward known network-arg keys from top-level config (TOML). Source of
    # truth: `networks.all_network_kwargs()` (the flat `NETWORK_KWARGS`
    # allowlist), plus a small tail of top-level training args the network
    # modules still want to read (e.g. postfix contrastive's step-boundary
    # window).
    for key in NETWORK_KWARG_ALLOWLIST + _EXTRA_FORWARDED_TOP_LEVEL_ARGS:
        if (
            key not in net_kwargs
            and hasattr(args, key)
            and getattr(args, key) is not None
        ):
            net_kwargs[key] = str(getattr(args, key))

    args._network_kwargs = net_kwargs
    return net_kwargs


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
    _install_graceful_stop_handlers()
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

    # Tell the dataset layer whether a conditioning method is actually enabled.
    # The method-adapter resolver (library/training/method_adapter.py) gates on
    # these same flags; mirroring it here lets the loader ignore a stray
    # `cond_cache_dir` (e.g. the WebUI path-saver writing the `conditioning_data`
    # default into a plain LoRA variant's TOML) instead of crashing on the empty
    # dir. cond_diff_loss is included because it also consumes cond_latents.
    _datasets_base.set_conditioning_method_enabled(
        bool(
            getattr(args, "use_controlnet", False)
            or getattr(args, "use_easycontrol", False)
            or getattr(args, "use_byg", False)
            or getattr(args, "cond_diff_loss", False)
        )
    )

    trainer = AnimaTrainer()
    trainer.train(args)
