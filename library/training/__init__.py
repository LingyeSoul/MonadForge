# Training utilities for Anima LoRA training.
#
# This is a *curated* facade: it re-exports the public, stable surface of the
# training package, grouped by concern below. Two things are deliberately NOT
# re-exported here and must be imported by their submodule path:
#   - The argparse / CLI surface, which lives in `library.config.cli_args`
#     (it is config, not training).
#   - Loop-internal machinery only `train.py`/`loop.py` compose:
#       library.training.loop          (build_loop_state, run_training_loop)
#       library.training.validation    (run_validation)
#       library.training.log_dispatch  (dispatch_logs)
#       library.training.forward       (per-step forward helpers)
# Everything below is part of the supported import-from-the-package surface.
#
# The names resolve LAZILY (PEP 562): most of these submodules import torch and
# accelerate, and importing *any* submodule of this package executes this file —
# so an eager facade made every WebUI process load the whole training stack just
# to read `library.training.staged_resolution_plan`. Access a name and it is
# imported then, cached in the module globals afterwards; `from
# library.training import X` and `library.training.X` keep working unchanged.

from __future__ import annotations

from importlib import import_module
from typing import Any

# Public name → defining submodule.
_EXPORTS: dict[str, str] = {
    # library.training.contexts
    "AcceleratedBundle": "library.training.contexts",
    "DatasetBundle": "library.training.contexts",
    "NetworkBundle": "library.training.contexts",
    "OptimizerBundle": "library.training.contexts",
    "RuntimeState": "library.training.contexts",
    "TrainCtx": "library.training.contexts",
    "ValCtx": "library.training.contexts",
    # library.training.method_adapter
    "ComputeLossCtx": "library.training.method_adapter",
    "ForwardArtifacts": "library.training.method_adapter",
    "MethodAdapter": "library.training.method_adapter",
    "SetupCtx": "library.training.method_adapter",
    "StepCtx": "library.training.method_adapter",
    "ValidationBaseline": "library.training.method_adapter",
    "resolve_adapters": "library.training.method_adapter",
    # library.training.samplers
    "SamplerContext": "library.training.samplers",
    "SamplerOut": "library.training.samplers",
    "SAMPLER_REGISTRY": "library.training.samplers",
    # library.runtime.noise
    "fm_training_batch": "library.runtime.noise",
    # library.training.losses
    "LivenessLedger": "library.training.losses",
    "LossContext": "library.training.losses",
    "LossComposer": "library.training.losses",
    "LOSS_REGISTRY": "library.training.losses",
    "build_loss_composer": "library.training.losses",
    "add_custom_train_arguments": "library.training.losses",
    "apply_masked_loss": "library.training.losses",
    "conditional_loss": "library.training.losses",
    "get_huber_threshold_if_needed": "library.training.losses",
    # library.training.metrics
    "MetricContext": "library.training.metrics",
    "MetricProducer": "library.training.metrics",
    "collect_metrics": "library.training.metrics",
    # library.training.optimizers
    "get_optimizer": "library.training.optimizers",
    "get_optimizer_train_eval_fn": "library.training.optimizers",
    "is_schedulefree_optimizer": "library.training.optimizers",
    # library.training.schedulers
    "get_scheduler_fix": "library.training.schedulers",
    "get_dummy_scheduler": "library.training.schedulers",
    # library.training.metadata
    "SS_METADATA_KEY_V2": "library.training.metadata",
    "SS_METADATA_KEY_BASE_MODEL_VERSION": "library.training.metadata",
    "SS_METADATA_KEY_NETWORK_MODULE": "library.training.metadata",
    "SS_METADATA_KEY_NETWORK_DIM": "library.training.metadata",
    "SS_METADATA_KEY_NETWORK_ALPHA": "library.training.metadata",
    "SS_METADATA_KEY_NETWORK_ARGS": "library.training.metadata",
    "SS_METADATA_MINIMUM_KEYS": "library.training.metadata",
    "build_minimum_network_metadata": "library.training.metadata",
    "build_training_metadata": "library.training.metadata",
    "add_dataset_metadata": "library.training.metadata",
    "add_model_hash_metadata": "library.training.metadata",
    "finalize_metadata": "library.training.metadata",
    # library.training.hashing
    "model_hash": "library.training.hashing",
    "calculate_sha256": "library.training.hashing",
    "addnet_hash_legacy": "library.training.hashing",
    "addnet_hash_safetensors": "library.training.hashing",
    "precalculate_safetensors_hashes": "library.training.hashing",
    "get_git_revision_hash": "library.training.hashing",
    # library.training.checkpoints
    "EPOCH_STATE_NAME": "library.training.checkpoints",
    "EPOCH_FILE_NAME": "library.training.checkpoints",
    "EPOCH_DIFFUSERS_DIR_NAME": "library.training.checkpoints",
    "LAST_STATE_NAME": "library.training.checkpoints",
    "DEFAULT_EPOCH_NAME": "library.training.checkpoints",
    "DEFAULT_LAST_OUTPUT_NAME": "library.training.checkpoints",
    "DEFAULT_STEP_NAME": "library.training.checkpoints",
    "STEP_STATE_NAME": "library.training.checkpoints",
    "STEP_FILE_NAME": "library.training.checkpoints",
    "STEP_DIFFUSERS_DIR_NAME": "library.training.checkpoints",
    "CheckpointSaver": "library.training.checkpoints",
    "default_if_none": "library.training.checkpoints",
    "get_epoch_ckpt_name": "library.training.checkpoints",
    "get_step_ckpt_name": "library.training.checkpoints",
    "get_last_ckpt_name": "library.training.checkpoints",
    "save_sd_model_on_epoch_end_or_stepwise_common": "library.training.checkpoints",
    "save_state_on_epoch_end": "library.training.checkpoints",
    "save_state_stepwise": "library.training.checkpoints",
    "save_state_on_train_end": "library.training.checkpoints",
    "save_sd_model_on_train_end_common": "library.training.checkpoints",
    "get_checkpoint_state_dir": "library.training.checkpoints",
    "get_checkpoint_ckpt_name": "library.training.checkpoints",
    "save_checkpoint_state": "library.training.checkpoints",
    "get_interrupted_state_dir": "library.training.checkpoints",
    "INTERRUPTED_STATE_NAME": "library.training.checkpoints",
    # library.training.state
    "SCHEMA_VERSION": "library.training.state",
    "build_train_state": "library.training.state",
    "capture_rng_state": "library.training.state",
    "normalize_train_state": "library.training.state",
    "restore_rng_state": "library.training.state",
    "state_is_complete": "library.training.state",
    # library.training.stop
    "StopController": "library.training.stop",
    # library.training.progress
    "ProgressSink": "library.training.progress",
    "run_scope": "library.training.progress",
    # library.training.loss_recorder
    "LossRecorder": "library.training.loss_recorder",
}

__all__ = [
    "AcceleratedBundle",
    "DatasetBundle",
    "NetworkBundle",
    "OptimizerBundle",
    "RuntimeState",
    "TrainCtx",
    "ValCtx",
    "ComputeLossCtx",
    "ForwardArtifacts",
    "MethodAdapter",
    "SetupCtx",
    "StepCtx",
    "ValidationBaseline",
    "resolve_adapters",
    "SamplerContext",
    "SamplerOut",
    "SAMPLER_REGISTRY",
    "fm_training_batch",
    "LivenessLedger",
    "LossContext",
    "LossComposer",
    "LOSS_REGISTRY",
    "build_loss_composer",
    "add_custom_train_arguments",
    "apply_masked_loss",
    "conditional_loss",
    "get_huber_threshold_if_needed",
    "MetricContext",
    "MetricProducer",
    "collect_metrics",
    "get_optimizer",
    "get_optimizer_train_eval_fn",
    "is_schedulefree_optimizer",
    "get_scheduler_fix",
    "get_dummy_scheduler",
    "SS_METADATA_KEY_V2",
    "SS_METADATA_KEY_BASE_MODEL_VERSION",
    "SS_METADATA_KEY_NETWORK_MODULE",
    "SS_METADATA_KEY_NETWORK_DIM",
    "SS_METADATA_KEY_NETWORK_ALPHA",
    "SS_METADATA_KEY_NETWORK_ARGS",
    "SS_METADATA_MINIMUM_KEYS",
    "build_minimum_network_metadata",
    "build_training_metadata",
    "add_dataset_metadata",
    "add_model_hash_metadata",
    "finalize_metadata",
    "model_hash",
    "calculate_sha256",
    "addnet_hash_legacy",
    "addnet_hash_safetensors",
    "precalculate_safetensors_hashes",
    "get_git_revision_hash",
    "EPOCH_STATE_NAME",
    "EPOCH_FILE_NAME",
    "EPOCH_DIFFUSERS_DIR_NAME",
    "LAST_STATE_NAME",
    "DEFAULT_EPOCH_NAME",
    "DEFAULT_LAST_OUTPUT_NAME",
    "DEFAULT_STEP_NAME",
    "STEP_STATE_NAME",
    "STEP_FILE_NAME",
    "STEP_DIFFUSERS_DIR_NAME",
    "CheckpointSaver",
    "default_if_none",
    "get_epoch_ckpt_name",
    "get_step_ckpt_name",
    "get_last_ckpt_name",
    "save_sd_model_on_epoch_end_or_stepwise_common",
    "save_state_on_epoch_end",
    "save_state_stepwise",
    "save_state_on_train_end",
    "save_sd_model_on_train_end_common",
    "get_checkpoint_state_dir",
    "get_checkpoint_ckpt_name",
    "save_checkpoint_state",
    "get_interrupted_state_dir",
    "INTERRUPTED_STATE_NAME",
    "SCHEMA_VERSION",
    "build_train_state",
    "capture_rng_state",
    "normalize_train_state",
    "restore_rng_state",
    "state_is_complete",
    "StopController",
    "ProgressSink",
    "run_scope",
    "LossRecorder",
]


def __getattr__(name: str) -> Any:
    """Import (and cache) one facade name on first access."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
