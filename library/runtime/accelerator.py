"""HuggingFace Accelerate setup and FP16/BF16 plumbing.

Wraps the ``Accelerator`` construction so training scripts can stay out of the
logging-backend plumbing. Also hosts the state-resume helper (local dir or HF
repo) and the dtype resolver that maps ``--mixed_precision`` and
``--save_precision`` flags to torch dtypes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from typing import Optional

import torch
from accelerate import Accelerator
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def resume_from_local_or_hf_if_specified(accelerator, args):
    if not args.resume:
        return

    if not args.resume_from_huggingface:
        logger.info(f"resume training from local state: {args.resume}")
        accelerator.load_state(args.resume)
        return

    logger.info(f"resume training from huggingface state: {args.resume}")
    repo_id = args.resume.split("/")[0] + "/" + args.resume.split("/")[1]
    path_in_repo = "/".join(args.resume.split("/")[2:])
    revision = None
    repo_type = None
    if ":" in path_in_repo:
        divided = path_in_repo.split(":")
        if len(divided) == 2:
            path_in_repo, revision = divided
            repo_type = "model"
        else:
            path_in_repo, revision, repo_type = divided
    logger.info(
        f"Downloading state from huggingface: {repo_id}/{path_in_repo}@{revision}"
    )

    from huggingface_hub import list_repo_files

    list_files = list_repo_files(
        repo_id=repo_id,
        revision=revision,
        token=args.huggingface_token,
        repo_type=repo_type,
    )
    list_files = [f for f in list_files if f.startswith(path_in_repo)]

    async def download(filename) -> str:
        def task():
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type=repo_type,
                token=args.huggingface_token,
            )

        return await asyncio.get_event_loop().run_in_executor(None, task)

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        asyncio.gather(*[download(filename=filename) for filename in list_files])
    )
    if len(results) == 0:
        raise ValueError("No files found in the specified repo id")
    dirname = os.path.dirname(results[0])
    accelerator.load_state(dirname)


def resolve_run_log_dir(args: argparse.Namespace) -> Optional[str]:
    """Return the timestamped run-specific log directory for this invocation.

    Builds ``{args.logging_dir}/{log_prefix}{YYYYMMDD-HHMM}`` once and caches it
    on ``args._resolved_run_log_dir`` so the config-snapshot writer (which runs
    before ``prepare_accelerator``) and the accelerator setup land in the same
    dir even when they're a minute apart on the wall clock. Returns ``None``
    when ``--logging_dir`` is unset.
    """
    if args.logging_dir is None:
        return None
    cached = getattr(args, "_resolved_run_log_dir", None)
    if cached is not None:
        return cached
    if args.log_prefix is not None:
        log_prefix = args.log_prefix
    else:
        method = getattr(args, "method", None)
        preset = getattr(args, "preset", None)
        parts = [p for p in (method, preset) if p]
        log_prefix = ("_".join(parts) + "_") if parts else ""
    resolved = (
        args.logging_dir
        + "/"
        + log_prefix
        + time.strftime("%Y%m%d-%H%M", time.localtime())
    )
    args._resolved_run_log_dir = resolved
    return resolved


def prepare_accelerator(args: argparse.Namespace):
    logging_dir = resolve_run_log_dir(args)

    # WebUI-injected env overrides (set by ConfigEditor before launching
    # the training subprocess).  These take precedence over TOML / CLI args
    # so the user can toggle wandb from the UI without touching config files.
    wandb_env_enabled = os.environ.get("WANDB_ENABLED") == "1"
    if wandb_env_enabled and getattr(args, "log_with", None) not in ("wandb", "all"):
        args.log_with = "all"  # both tensorboard + wandb
    env_run_name = os.environ.get("WANDB_RUN_NAME")
    if env_run_name:
        args.wandb_run_name = env_run_name
    env_api_key = os.environ.get("WANDB_API_KEY")
    if env_api_key:
        args.wandb_api_key = env_api_key
    env_log_every = os.environ.get("WANDB_LOG_EVERY_N")
    if env_log_every:
        try:
            args.log_every_n_steps = int(env_log_every)
        except ValueError:
            pass
    env_project = os.environ.get("WANDB_PROJECT")
    if env_project:
        args._wandb_project = env_project
        os.environ.setdefault("WANDB_PROJECT", env_project)
    # Collector feature flags — stored on args for loop.py to read
    args._wandb_log_gradients = os.environ.get("WANDB_LOG_GRADIENTS") == "1"
    args._wandb_log_weights = os.environ.get("WANDB_LOG_WEIGHTS") == "1"
    args._wandb_log_artifact = os.environ.get("WANDB_LOG_ARTIFACT") == "1"

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError("logging_dir is required when log_with is tensorboard")
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    # Always "NO": torch.compile is applied per-block by DiT.compile_blocks (the
    # block loop is the shape-stable hot path; pre/post-block regions vary per
    # bucket). Letting Accelerate full-compile the model on top of that would
    # double-compile and trigger graph-break / recompile storms.
    dynamo_backend = "NO"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        dynamo_backend=dynamo_backend,
    )
    print("accelerator device:", accelerator.device)
    return accelerator


def prepare_dtype(args: argparse.Namespace):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    save_dtype: Optional[torch.dtype] = None
    if args.save_precision == "fp16":
        save_dtype = torch.float16
    elif args.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif args.save_precision == "float":
        save_dtype = torch.float32

    return weight_dtype, save_dtype


def patch_accelerator_for_fp16_training(accelerator):
    """Force ``allow_fp16=True`` inside ``GradScaler._unscale_grads_``.

    Needed for full-fp16 training (as opposed to mixed-precision) where the
    optimizer holds fp16 params that the default scaler refuses to touch.
    """
    from accelerate import DistributedType

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        return

    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer
