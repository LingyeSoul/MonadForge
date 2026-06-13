# Training utilities for Anima LoRA training.
#
# This is a *curated* facade: it re-exports the public, stable surface of the
# training package, grouped by concern below. Two things are deliberately NOT
# re-exported here and must be imported by their submodule path:
#   - The argparse / CLI surface, which now lives in `library.config.cli_args`
#     (it is config, not training).
#   - Loop-internal machinery only `train.py`/`loop.py` compose:
#       library.training.loop          (build_loop_state, run_training_loop)
#       library.training.validation    (run_validation)
#       library.training.log_dispatch  (dispatch_logs)
#       library.training.forward       (per-step forward helpers)
# Everything below is part of the supported import-from-the-package surface.

# === contexts ===
from library.training.contexts import (
    AcceleratedBundle,
    DatasetBundle,
    NetworkBundle,
    OptimizerBundle,
    RuntimeState,
    TrainCtx,
    ValCtx,
)

# === per-method extension protocol (public — used by networks/methods/*) ===
from library.training.method_adapter import (
    ComputeLossCtx,
    ForwardArtifacts,
    MethodAdapter,
    SetupCtx,
    StepCtx,
    ValidationBaseline,
    resolve_adapters,
)

# === registries: samplers ===
from library.training.samplers import (
    SamplerContext,
    SamplerOut,
    SAMPLER_REGISTRY,
)

# === registries: losses ===
from library.training.losses import (
    LivenessLedger,
    LossContext,
    LossComposer,
    LOSS_REGISTRY,
    build_loss_composer,
    add_custom_train_arguments,
    apply_masked_loss,
    conditional_loss,
    get_huber_threshold_if_needed,
)

# === registries: metrics ===
from library.training.metrics import (
    MetricContext,
    MetricProducer,
    collect_metrics,
)

# === optimization ===
from library.training.optimizers import (
    get_optimizer,
    get_optimizer_train_eval_fn,
    is_schedulefree_optimizer,
)
from library.training.schedulers import (
    get_scheduler_fix,
    get_dummy_scheduler,
)

# === provenance: metadata ===
from library.training.metadata import (
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
    SS_METADATA_MINIMUM_KEYS,
    build_minimum_network_metadata,
    build_training_metadata,
    add_dataset_metadata,
    add_model_hash_metadata,
    finalize_metadata,
)

# === provenance: hashing ===
from library.training.hashing import (
    model_hash,
    calculate_sha256,
    addnet_hash_legacy,
    addnet_hash_safetensors,
    precalculate_safetensors_hashes,
    get_git_revision_hash,
)

# === provenance: checkpoints ===
from library.training.checkpoints import (
    EPOCH_STATE_NAME,
    EPOCH_FILE_NAME,
    EPOCH_DIFFUSERS_DIR_NAME,
    LAST_STATE_NAME,
    DEFAULT_EPOCH_NAME,
    DEFAULT_LAST_OUTPUT_NAME,
    DEFAULT_STEP_NAME,
    STEP_STATE_NAME,
    STEP_FILE_NAME,
    STEP_DIFFUSERS_DIR_NAME,
    CheckpointSaver,
    default_if_none,
    get_epoch_ckpt_name,
    get_step_ckpt_name,
    get_last_ckpt_name,
    get_remove_epoch_no,
    get_remove_step_no,
    save_sd_model_on_epoch_end_or_stepwise_common,
    save_and_remove_state_on_epoch_end,
    save_and_remove_state_stepwise,
    save_state_on_train_end,
    save_sd_model_on_train_end_common,
    get_checkpoint_state_dir,
    get_checkpoint_ckpt_name,
    save_checkpoint_state,
)

# === progress sink ===
from library.training.progress import (
    ProgressSink,
    run_scope,
)

# === loss recorder ===
from library.training.loss_recorder import LossRecorder

__all__ = [
    # contexts
    "AcceleratedBundle",
    "DatasetBundle",
    "NetworkBundle",
    "OptimizerBundle",
    "RuntimeState",
    "TrainCtx",
    "ValCtx",
    # extension protocol
    "ComputeLossCtx",
    "ForwardArtifacts",
    "MethodAdapter",
    "SetupCtx",
    "StepCtx",
    "ValidationBaseline",
    "resolve_adapters",
    # samplers
    "SamplerContext",
    "SamplerOut",
    "SAMPLER_REGISTRY",
    # losses
    "LivenessLedger",
    "LossContext",
    "LossComposer",
    "LOSS_REGISTRY",
    "build_loss_composer",
    "add_custom_train_arguments",
    "apply_masked_loss",
    "conditional_loss",
    "get_huber_threshold_if_needed",
    # metrics
    "MetricContext",
    "MetricProducer",
    "collect_metrics",
    # optimization
    "get_optimizer",
    "get_optimizer_train_eval_fn",
    "is_schedulefree_optimizer",
    "get_scheduler_fix",
    "get_dummy_scheduler",
    # metadata
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
    # hashing
    "model_hash",
    "calculate_sha256",
    "addnet_hash_legacy",
    "addnet_hash_safetensors",
    "precalculate_safetensors_hashes",
    "get_git_revision_hash",
    # checkpoints
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
    "get_remove_epoch_no",
    "get_remove_step_no",
    "save_sd_model_on_epoch_end_or_stepwise_common",
    "save_and_remove_state_on_epoch_end",
    "save_and_remove_state_stepwise",
    "save_state_on_train_end",
    "save_sd_model_on_train_end_common",
    "get_checkpoint_state_dir",
    "get_checkpoint_ckpt_name",
    "save_checkpoint_state",
    # progress
    "ProgressSink",
    "run_scope",
    # loss recorder
    "LossRecorder",
]
