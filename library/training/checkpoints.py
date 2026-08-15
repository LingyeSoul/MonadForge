import argparse
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from library.anima.compat import validate_resume_model_signature
from library.io.output_layout import (
    atomic_replace_dir,
    atomic_write_json,
    layout_from_args,
    remove_path_with_retry,
)
from library.training.state import (
    COMPLETE_MARKER,
    build_train_state,
    normalize_train_state,
    read_train_state,
    state_is_complete,
    write_complete_marker,
)

logger = logging.getLogger(__name__)

# checkpoint filename templates
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

DEFAULT_STEP_NAME = "at"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"

CHECKPOINT_STATE_NAME = "{}-checkpoint-state"
CHECKPOINT_FILE_NAME = "{}-checkpoint"
INTERRUPTED_STATE_NAME = "{}-interrupted-state"
ROLLING_STATE_NAME = "{}-rolling-state"


def default_if_none(value, default):
    return default if value is None else value


def _ensure_parent_dir(path: str) -> None:
    """makedirs the directory that will contain ``path`` (the per-run subdir)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _canonical(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "_output_layout_canonical", False))


def _layout(args: argparse.Namespace):
    """Return the canonical layout, preserving legacy callers until train()
    explicitly opts into it."""

    return layout_from_args(args)


# Trajectory (step/epoch) checkpoints + their resumable -state dirs live in a
# per-run subdir ``<output_dir>/<model_name>/`` so they don't clutter
# output_dir; the final ``<output_name>.<ext>`` and the rolling
# ``<output_name>-checkpoint`` resumable family stay at the output_dir root
# (where inference / merge / auto_resume look). Returned names are relative to
# output_dir and carry the subdir prefix, so save and remove stay symmetric.
def get_epoch_ckpt_name(args: argparse.Namespace, ext: str, epoch_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
    if _canonical(args):
        return EPOCH_FILE_NAME.format(model_name, epoch_no) + ext
    return os.path.join(model_name, EPOCH_FILE_NAME.format(model_name, epoch_no) + ext)


def get_step_ckpt_name(args: argparse.Namespace, ext: str, step_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
    if _canonical(args):
        return STEP_FILE_NAME.format(model_name, step_no) + ext
    return os.path.join(model_name, STEP_FILE_NAME.format(model_name, step_no) + ext)


def get_last_ckpt_name(args: argparse.Namespace, ext: str):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return model_name + ext


def save_sd_model_on_epoch_end_or_stepwise_common(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    if on_epoch_end:
        epoch_no = epoch + 1
        saving = (
            epoch_no % args.save_every_n_epochs == 0 and epoch_no < num_train_epochs
        )
        if not saving:
            return

        model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
    else:
        model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
        epoch_no = epoch

    os.makedirs(args.output_dir, exist_ok=True)
    if save_stable_diffusion_format:
        ext = ".safetensors" if use_safetensors else ".ckpt"

        if on_epoch_end:
            ckpt_name = get_epoch_ckpt_name(args, ext, epoch_no)
        else:
            ckpt_name = get_step_ckpt_name(args, ext, global_step)

        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        _ensure_parent_dir(ckpt_file)  # create the per-run subdir
        logger.info("")
        logger.info(f"saving checkpoint: {ckpt_file}")
        sd_saver(ckpt_file, epoch_no, global_step)

    else:
        if on_epoch_end:
            out_dir = os.path.join(
                args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, epoch_no)
            )
        else:
            out_dir = os.path.join(
                args.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, global_step)
            )

        logger.info("")
        logger.info(f"saving model: {out_dir}")
        diffusers_saver(out_dir)

    if args.save_state:
        if on_epoch_end:
            save_state_on_epoch_end(args, accelerator, epoch_no)
        else:
            save_state_stepwise(args, accelerator, global_step)


def save_state_on_epoch_end(args: argparse.Namespace, accelerator, epoch_no):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")

    # Canonical runs already have ``args.output_dir=<ckpt>/<name>``; legacy
    # callers retain the historical extra model-name directory.
    state_dir = os.path.join(
        args.output_dir,
        EPOCH_STATE_NAME.format(model_name, epoch_no)
        if _canonical(args)
        else os.path.join(model_name, EPOCH_STATE_NAME.format(model_name, epoch_no)),
    )
    _atomic_accelerator_save_state(accelerator, state_dir)


def save_state_stepwise(args: argparse.Namespace, accelerator, step_no):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)

    logger.info("")
    logger.info(f"saving state at step {step_no}")

    state_dir = os.path.join(
        args.output_dir,
        STEP_STATE_NAME.format(model_name, step_no)
        if _canonical(args)
        else os.path.join(model_name, STEP_STATE_NAME.format(model_name, step_no)),
    )
    _atomic_accelerator_save_state(accelerator, state_dir)


def get_checkpoint_state_dir(args: argparse.Namespace):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    if _canonical(args):
        return os.path.join(args.output_dir, CHECKPOINT_STATE_NAME.format(model_name))
    return os.path.join(args.output_dir, CHECKPOINT_STATE_NAME.format(model_name))


def get_last_state_dir(args: argparse.Namespace):
    """Directory written by ``save_state_on_train_end`` (``<output_name>-state``).

    Distinct from ``get_checkpoint_state_dir`` (``<output_name>-checkpoint-state``)
    which is written mid-training by ``checkpointing_epochs``. Both carry a
    ``train_state.json`` and can be used to resume — ``auto_resume`` checks the
    checkpoint-state dir first, then falls back to this last-state dir so a run
    that finished with ``--save_state_on_train_end`` can be continued.
    """
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))


def get_interrupted_state_dir(args: argparse.Namespace) -> str:
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return os.path.join(args.output_dir, INTERRUPTED_STATE_NAME.format(model_name))


def get_rolling_state_dir(args: argparse.Namespace) -> str:
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return os.path.join(args.output_dir, ROLLING_STATE_NAME.format(model_name))


def get_checkpoint_ckpt_name(args: argparse.Namespace, ext: str):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return CHECKPOINT_FILE_NAME.format(model_name) + ext


def save_checkpoint_state(args: argparse.Namespace, accelerator):
    state_dir = get_checkpoint_state_dir(args)

    logger.info("")
    logger.info(f"saving checkpoint state to {state_dir} (overwriting)")
    os.makedirs(args.output_dir, exist_ok=True)

    _atomic_accelerator_save_state(accelerator, state_dir)


def save_rolling_state(args: argparse.Namespace, accelerator):
    """Publish the latest complete optimizer-boundary state atomically."""
    state_dir = get_rolling_state_dir(args)
    logger.info("saving rolling resume state to %s", state_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    _atomic_accelerator_save_state(accelerator, state_dir)


def save_state_on_train_end(args: argparse.Namespace, accelerator):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    _atomic_accelerator_save_state(accelerator, state_dir)


def _accelerator_process_count(accelerator) -> int:
    """Return the number of ranks participating in a save operation."""

    try:
        return max(1, int(getattr(accelerator, "num_processes", 1) or 1))
    except (TypeError, ValueError):
        return 1


def _accelerator_process_index(accelerator) -> int:
    """Return the stable rank index used by Accelerate's RNG filenames."""

    try:
        return max(0, int(getattr(accelerator, "process_index", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _write_save_status(path: Path, *, ok: bool, detail: str = "") -> None:
    """Publish a per-rank save status without exposing a partial file."""

    payload = "ok\n" if ok else f"error\n{detail}\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _read_save_error(path: Path) -> str | None:
    try:
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            return text or "checkpoint save failed"
    except OSError:
        return "checkpoint save failed (error marker unreadable)"
    return None


def _atomic_accelerator_save_state(accelerator, state_dir: str) -> None:
    """Atomically publish a complete Accelerate state across all ranks.

    ``Accelerator.save_state`` writes the shared model/optimizer payload from
    the main rank and a distinct RNG file from every rank.  All ranks therefore
    have to use one staging directory; publishing one PID-scoped directory per
    rank loses whichever files were written by the other ranks.  Rank status
    files make a failed rank visible to the publisher before the completion
    marker is written.
    """

    target = Path(state_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    process_count = _accelerator_process_count(accelerator)
    process_index = _accelerator_process_index(accelerator)
    is_main = bool(getattr(accelerator, "is_main_process", process_index == 0))
    distributed = process_count > 1

    # This name is intentionally shared by every rank.  Checkpoint calls are
    # serialized by the training loop; main removes a stale staging directory
    # before the first barrier so a prior crash cannot contribute old files.
    staging = target.with_name(f".{target.name}.tmp")
    error_marker = target.with_name(f".{target.name}.save-error")
    token_path = staging / ".save-token"
    expected_token: str | None = None

    preparation_error: BaseException | None = None
    if is_main:
        try:
            if staging.exists():
                remove_path_with_retry(staging)
            error_marker.unlink(missing_ok=True)
            staging.mkdir(parents=True, exist_ok=True)
            expected_token = f"{os.getpid()}-{time.time_ns()}"
            token_tmp = token_path.with_name(f".{token_path.name}.{os.getpid()}.tmp")
            token_tmp.write_text(expected_token, encoding="ascii")
            os.replace(token_tmp, token_path)
        except BaseException as exc:  # noqa: BLE001 - all ranks must see prep failure
            preparation_error = exc
            try:
                _write_save_status(
                    error_marker,
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            except BaseException as status_exc:  # noqa: BLE001
                logger.debug("could not write checkpoint prep failure marker: %s", status_exc)

    if distributed:
        accelerator.wait_for_everyone()

    remote_preparation_error = _read_save_error(error_marker)
    if preparation_error is not None or remote_preparation_error is not None or not staging.is_dir():
        if preparation_error is not None:
            raise preparation_error
        raise RuntimeError(
            remote_preparation_error or "checkpoint staging directory was not prepared"
        )

    if expected_token is None:
        try:
            expected_token = token_path.read_text(encoding="ascii")
        except OSError as exc:
            raise RuntimeError("checkpoint staging token is missing") from exc
    if not expected_token:
        raise RuntimeError("checkpoint staging token is empty")

    staging.mkdir(parents=True, exist_ok=True)

    rank_status = staging / f".rank-{process_index}.status"
    local_error: BaseException | None = None
    try:
        accelerator.save_state(str(staging))
    except BaseException as exc:  # noqa: BLE001 - synchronize all save failures
        local_error = exc
        try:
            _write_save_status(
                rank_status,
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )
        except BaseException as status_exc:  # noqa: BLE001
            # The original exception remains the actionable failure.  Missing
            # status is also rejected by the publisher after the barrier.
            logger.debug("could not write rank save failure marker: %s", status_exc)
    else:
        try:
            _write_save_status(rank_status, ok=True)
        except BaseException as exc:  # noqa: BLE001 - status failure is publish-fatal
            local_error = exc

    # Every rank must reach this barrier, including a rank whose save failed;
    # otherwise the publisher could observe a transiently incomplete directory
    # and incorrectly mark it complete.
    if distributed:
        accelerator.wait_for_everyone()

    publish_error: BaseException | None = None
    if is_main:
        failures: list[str] = []
        for rank in range(process_count):
            status_path = staging / f".rank-{rank}.status"
            try:
                status = status_path.read_text(encoding="utf-8").strip()
            except OSError:
                status = ""
            if status != "ok":
                failures.append(
                    f"rank {rank}: " + (status or "missing save status")
                )

        if failures:
            publish_error = RuntimeError(
                "Accelerate state save failed; refusing to publish incomplete state: "
                + "; ".join(failures)
            )
        else:
            try:
                write_complete_marker(staging)
                atomic_replace_dir(staging, target)
            except BaseException as exc:  # noqa: BLE001 - preserve publish failure
                publish_error = exc

        if publish_error is not None:
            try:
                _write_save_status(
                    error_marker,
                    ok=False,
                    detail=f"{type(publish_error).__name__}: {publish_error}",
                )
            except BaseException as status_exc:  # noqa: BLE001
                logger.debug("could not write checkpoint failure marker: %s", status_exc)

    # Let non-main ranks observe publication failures before returning.  The
    # marker is intentionally left on disk after a failure for diagnostics and
    # is removed at the start of the next save attempt.
    if distributed:
        accelerator.wait_for_everyone()

    remote_error = _read_save_error(error_marker)
    publication_token_error: RuntimeError | None = None
    try:
        published_token = (target / ".save-token").read_text(encoding="ascii")
    except OSError:
        published_token = None
    if (
        published_token != expected_token
        or not (target / COMPLETE_MARKER).is_file()
    ):
        publication_token_error = RuntimeError(
            "checkpoint publication was not observed by every rank"
        )
    if local_error is not None:
        raise local_error
    if publish_error is not None:
        raise publish_error
    if remote_error is not None:
        raise RuntimeError(remote_error)
    if publication_token_error is not None:
        raise publication_token_error

    # A failed publication intentionally leaves staging/status files alongside
    # the error marker for diagnostics and the next bounded cleanup attempt;
    # successful replaces have already moved staging into ``target``.


def save_sd_model_on_train_end_common(
    args: argparse.Namespace,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    if save_stable_diffusion_format:
        os.makedirs(args.output_dir, exist_ok=True)

        ckpt_name = model_name + (".safetensors" if use_safetensors else ".ckpt")
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        logger.info(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        sd_saver(ckpt_file, epoch, global_step)
    else:
        out_dir = (
            os.path.join(args.output_dir, "diffusers")
            if _canonical(args)
            else os.path.join(args.output_dir, model_name)
        )
        os.makedirs(out_dir, exist_ok=True)

        logger.info(f"save trained model as Diffusers to {out_dir}")
        diffusers_saver(out_dir)


class CheckpointSaver:
    """Owns every save / remove operation across a training run.

    Replaces the cluster of save_model / remove_model / save_model_hook /
    load_model_hook closures and the inline save-tick blocks scattered through
    train(). State that used to live in closures (metadata refs, save_dtype,
    sai-spec callable, mp.Value handles, ``steps_from_state``) becomes
    instance attributes.

    ``metadata`` is a shared mutable dict — the trainer also writes
    ``ss_epoch`` between saves; the saver only writes during a save.
    """

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        accelerator,
        save_dtype,
        metadata: dict,
        minimum_metadata: dict,
        get_sai_model_spec_fn: Callable[[argparse.Namespace], dict],
        current_epoch,
        current_step,
        progress_sink=None,
    ):
        self.args = args
        self.accelerator = accelerator
        self.save_dtype = save_dtype
        self.metadata = metadata
        self.minimum_metadata = minimum_metadata
        self.get_sai_model_spec_fn = get_sai_model_spec_fn
        self.current_epoch = current_epoch
        self.current_step = current_step
        # Optional structured-progress sink (Phase 0). When set, every
        # checkpoint write emits a ``ckpt`` event.
        self.progress_sink = progress_sink
        # Set by the load_state pre-hook when resuming. Read by train() to
        # decide initial_step.
        self.steps_from_state: int | None = None
        self.loaded_train_state: dict[str, Any] = {}
        self._runtime_state_provider: Callable[[], dict[str, Any]] | None = None
        self._saving_interrupted_state = False

    def set_runtime_state_provider(
        self, provider: Callable[[], dict[str, Any]] | None
    ) -> None:
        """Attach loop-owned fields that must ride in ``train_state.json``."""
        self._runtime_state_provider = provider

    def register_hooks(self, network: Any) -> None:
        """Install accelerator save/load pre-hooks that persist epoch/step
        state to ``train_state.json`` and strip non-network models from the
        save list (we only want the adapter weights, not the frozen DiT)."""
        accelerator = self.accelerator
        unwrap_type = type(accelerator.unwrap_model(network))

        def save_model_hook(models, weights, output_dir):
            # Accelerate invokes hooks on every rank, but ``train_state.json``
            # is a shared sidecar.  Restrict both model-list surgery and the
            # sidecar write to rank 0 so ranks cannot race on that JSON file.
            if not accelerator.is_main_process:
                return

            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, unwrap_type):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                if len(weights) > i:
                    weights.pop(i)

            train_state_file = os.path.join(output_dir, "train_state.json")
            runtime = self._runtime_state_provider() if self._runtime_state_provider else {}
            runtime = dict(runtime or {})
            global_step = int(
                runtime.get("global_step", self.current_step.value + 1)
            )
            # ``global_step`` is passed explicitly below; remove it from the
            # extension payload so old/custom runtime providers cannot cause a
            # duplicate keyword error when the hook serializes the state.
            runtime.pop("global_step", None)
            provider_interrupted = bool(runtime.pop("interrupted", False))
            runtime.pop("job_id", None)
            runtime.pop("root_job_id", None)
            stage_index = runtime.pop("stage_index", -1)
            if stage_index is None:
                stage_index = -1
            stage_outer_epoch = runtime.pop(
                "stage_outer_epoch", self.current_epoch.value
            )
            if stage_outer_epoch is None:
                stage_outer_epoch = self.current_epoch.value
            logger.info(
                f"save train state to {train_state_file} at epoch "
                f"{self.current_epoch.value} step {global_step}"
            )
            state = build_train_state(
                global_step=global_step,
                current_epoch=int(runtime.pop("current_epoch", self.current_epoch.value)),
                micro_batch_offset=int(runtime.pop("micro_batch_offset", 0) or 0),
                stage_index=int(stage_index),
                stage_batch_cursor=int(runtime.pop("stage_batch_cursor", 0) or 0),
                stage_outer_epoch=int(stage_outer_epoch),
                rng_state=runtime.pop("rng_state", None),
                config_signature=runtime.pop("config_signature", None),
                dataset_signature=runtime.pop("dataset_signature", None),
                job_id=os.environ.get("ANIMA_DAEMON_JOB_ID"),
                root_job_id=os.environ.get("ANIMA_DAEMON_ROOT_JOB_ID"),
                interrupted=bool(self._saving_interrupted_state or provider_interrupted),
                **runtime,
            )
            atomic_write_json(Path(train_state_file), state, indent=None)

        def load_model_hook(models, input_dir):
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, unwrap_type):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)

            train_state_file = os.path.join(input_dir, "train_state.json")
            if os.path.exists(train_state_file):
                if not state_is_complete(input_dir, require_marker=True):
                    raise ValueError(
                        "refusing to resume from an incomplete training state: "
                        f"{input_dir}"
                    )
                with open(train_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data = normalize_train_state(data)
                expected_root_job_id = os.environ.get("ANIMA_DAEMON_ROOT_JOB_ID")
                if expected_root_job_id:
                    actual_root_job_id = data.get("root_job_id")
                    if not actual_root_job_id:
                        raise ValueError(
                            "resume state is missing logical task owner: "
                            f"expected root_job_id={expected_root_job_id}"
                        )
                    if actual_root_job_id != expected_root_job_id:
                        raise ValueError(
                            "resume state logical task owner mismatch: "
                            f"state={actual_root_job_id}, expected={expected_root_job_id}"
                        )
                expected_config = getattr(self.args, "config_signature", None)
                actual_config = data.get("config_signature")
                if expected_config and actual_config not in (None, expected_config):
                    raise ValueError(
                        "resume state config signature mismatch: "
                        f"state={actual_config}, expected={expected_config}"
                    )
                expected_dataset = getattr(self.args, "dataset_signature", None)
                actual_dataset = data.get("dataset_signature")
                if expected_dataset and actual_dataset not in (None, expected_dataset):
                    raise ValueError(
                        "resume state dataset signature mismatch: "
                        f"state={actual_dataset}, expected={expected_dataset}"
                    )
                validate_resume_model_signature(
                    data,
                    expected_signature=getattr(
                        self.args, "anima_model_signature", None
                    ),
                    num_blocks=getattr(self.args, "anima_num_blocks", None),
                )
                self.loaded_train_state = data
                self.steps_from_state = data["global_step"]
                logger.info(f"load train state from {train_state_file}: {data}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    def auto_resume(self) -> None:
        """Point ``args.resume`` at a resumable state dir + force
        ``skip_until_initial_step`` when one exists below ``max_train_steps``.

        Checks two candidate dirs (in priority order):
          1. ``<output_name>-checkpoint-state`` — the mid-training resumable
             snapshot written by ``checkpointing_epochs`` (preferred; newest).
          2. ``<output_name>-state`` — the end-of-training snapshot written by
             ``--save_state_on_train_end``. Lets a finished run be continued.

        Both carry a ``train_state.json``. Candidate 1 is only consulted when
        ``checkpointing_epochs`` is set (its writer); candidate 2 whenever
        ``save_state_on_train_end`` is set. No-op when ``args.resume`` is
        already set or no eligible dir exists."""
        args = self.args
        if args.resume:
            return

        layout = _layout(args)
        roots: list[Path] = []
        for root in layout.legacy_candidates():
            root = Path(root)
            if root not in roots:
                roots.append(root)
        suffixes: list[str] = [
            INTERRUPTED_STATE_NAME.format(
                default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
            )
        ]
        rolling_suffix = ROLLING_STATE_NAME.format(
            default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
        )
        suffixes.append(rolling_suffix)
        if getattr(args, "checkpointing_epochs", None):
            suffixes.append(
                CHECKPOINT_STATE_NAME.format(
                    default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
                )
            )
        if getattr(args, "save_state_on_train_end", None):
            suffixes.append(
                LAST_STATE_NAME.format(
                    default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
                )
            )
        # A caller may explicitly enable auto-resume through the daemon even
        # when the corresponding save flag is absent.  The existence/completion
        # checks below keep this backwards-compatible and safe.
        if getattr(args, "auto_resume", False):
            for suffix in (
                CHECKPOINT_STATE_NAME.format(
                    default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
                ),
                LAST_STATE_NAME.format(
                    default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
                ),
            ):
                if suffix not in suffixes:
                    suffixes.append(suffix)

        candidates = [
            (root_priority, state_priority, root / suffix)
            for root_priority, root in enumerate(roots)
            for state_priority, suffix in enumerate(suffixes)
        ]
        seen: set[Path] = set()
        eligible: list[tuple[int, int, int, Path]] = []
        for root_priority, state_priority, state_dir in candidates:
            if state_dir in seen:
                continue
            seen.add(state_dir)
            if not state_is_complete(state_dir, require_marker=True):
                continue
            ckpt_data = read_train_state(state_dir) or {}
            ckpt_step = int(ckpt_data.get("global_step", 0) or 0)
            expected_root_job_id = os.environ.get("ANIMA_DAEMON_ROOT_JOB_ID")
            if expected_root_job_id:
                if (
                    int(ckpt_data.get("schema_version", 1) or 1) < 3
                    or not ckpt_data.get("job_id")
                    or ckpt_data.get("root_job_id") != expected_root_job_id
                ):
                    logger.warning(
                        "ignoring resume state owned by another logical task: %s",
                        state_dir,
                    )
                    continue
            expected_config = getattr(args, "config_signature", None)
            expected_dataset = getattr(args, "dataset_signature", None)
            if expected_config and ckpt_data.get("config_signature") not in (None, expected_config):
                logger.warning("ignoring resume state with mismatched config signature: %s", state_dir)
                continue
            if expected_dataset and ckpt_data.get("dataset_signature") not in (None, expected_dataset):
                logger.warning("ignoring resume state with mismatched dataset signature: %s", state_dir)
                continue
            try:
                validate_resume_model_signature(
                    ckpt_data,
                    expected_signature=getattr(args, "anima_model_signature", None),
                    num_blocks=getattr(args, "anima_num_blocks", None),
                )
            except ValueError as exc:
                logger.warning(
                    "ignoring incompatible Anima resume state %s: %s",
                    state_dir,
                    exc,
                )
                continue
            eligible.append((ckpt_step, state_priority, root_priority, state_dir))

        if not eligible:
            return

        # A completed continuation can leave its older interrupted state beside
        # a newer normal state.  Fixed suffix ordering would then replay already
        # committed optimizer steps on the next extension.  Step is therefore
        # authoritative; interrupted -> rolling -> normal remains the tie-break
        # policy when two complete states describe the same committed step.
        ckpt_step, _, _, state_dir = min(
            eligible,
            key=lambda item: (-item[0], item[1], item[2], str(item[3])),
        )
        if ckpt_step < args.max_train_steps:
            args.resume = state_dir
            args.skip_until_initial_step = True
            logger.info(f"auto-resuming from state at step {ckpt_step}: {state_dir}")
            return
        logger.info(
            f"state already reached max_train_steps "
            f"({ckpt_step} >= {args.max_train_steps}), starting fresh: {state_dir}"
        )

    def save(self, ckpt_name: str, network: Any, steps: int, epoch_no: int) -> None:
        """Write a network checkpoint with up-to-date training metadata."""
        args = self.args
        accelerator = self.accelerator
        unwrapped_nw = accelerator.unwrap_model(network)

        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        # ckpt_name carries a per-run subdir for trajectory (step/epoch) saves;
        # final + resumable names are bare and resolve to output_dir directly.
        _ensure_parent_dir(ckpt_file)

        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        self.metadata["ss_training_finished_at"] = str(time.time())
        self.metadata["ss_steps"] = str(steps)
        self.metadata["ss_epoch"] = str(epoch_no)

        metadata_to_save = self.minimum_metadata if args.no_metadata else self.metadata
        sai_metadata = self.get_sai_model_spec_fn(args)
        metadata_to_save.update(sai_metadata)

        unwrapped_nw.save_weights(ckpt_file, self.save_dtype, metadata_to_save)

        if self.progress_sink is not None:
            self.progress_sink.ckpt(global_step=steps, path=ckpt_file)

    def maybe_save_step(self, network: Any, global_step: int, epoch: int) -> None:
        """Step-cadence save. ``global_step`` must already be incremented."""
        args = self.args
        accelerator = self.accelerator
        if (
            args.save_every_n_steps is None
            or global_step % args.save_every_n_steps != 0
        ):
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            ckpt_name = get_step_ckpt_name(args, "." + args.save_model_as, global_step)
            self.save(ckpt_name, network, global_step, epoch)
        if args.save_state:
            # All ranks contribute their RNG/sampler state to the shared
            # staging directory; only rank 0 publishes it.
            save_state_stepwise(args, accelerator, global_step)

    def maybe_save_epoch(
        self, network: Any, global_step: int, epoch: int, num_train_epochs: int
    ) -> None:
        """Epoch-cadence save. ``epoch`` is 0-indexed; saver writes ``epoch+1``."""
        args = self.args
        accelerator = self.accelerator
        if args.save_every_n_epochs is None:
            return
        epoch_no = epoch + 1
        saving = (
            epoch_no % args.save_every_n_epochs == 0 and epoch_no < num_train_epochs
        )
        if not saving:
            return
        if accelerator.is_main_process:
            ckpt_name = get_epoch_ckpt_name(args, "." + args.save_model_as, epoch_no)
            self.save(ckpt_name, network, global_step, epoch_no)
        if args.save_state:
            # See ``maybe_save_step``: every rank must enter the state save.
            save_state_on_epoch_end(args, accelerator, epoch_no)

    def maybe_save_resumable(
        self, network: Any, global_step: int, epoch: int, num_train_epochs: int
    ) -> None:
        """``checkpointing_epochs``-cadence resumable save. Overwrites the
        same ``<output_name>-checkpoint`` file each time. ``epoch`` is 0-indexed."""
        args = self.args
        accelerator = self.accelerator
        if not (
            args.checkpointing_epochs is not None and args.checkpointing_epochs > 0
        ):
            return
        epoch_no = epoch + 1
        if not (
            epoch_no % args.checkpointing_epochs == 0 and epoch_no < num_train_epochs
        ):
            return
        if accelerator.is_main_process:
            ckpt_name = get_checkpoint_ckpt_name(args, "." + args.save_model_as)
            self.save(ckpt_name, network, global_step, epoch_no)
        save_checkpoint_state(args, accelerator)

    def maybe_save_rolling_state(self, global_step: int) -> None:
        """Save a rolling state after a committed optimizer/global step."""
        cadence = int(getattr(self.args, "resume_state_every_n_steps", 50) or 0)
        if cadence <= 0 or global_step <= 0 or global_step % cadence:
            return
        self.accelerator.wait_for_everyone()
        save_rolling_state(self.args, self.accelerator)

    def cleanup_resumable(self) -> None:
        """Retire crash-only state after a successful training completion."""
        args = self.args
        if not self.accelerator.is_main_process:
            return
        rolling_state_dir = get_rolling_state_dir(args)
        if os.path.exists(rolling_state_dir):
            logger.info(
                f"training complete, removing rolling state: {rolling_state_dir}"
            )
            remove_path_with_retry(Path(rolling_state_dir))
        if not getattr(args, "checkpointing_epochs", None):
            return
        checkpoint_state_dir = get_checkpoint_state_dir(args)
        if os.path.exists(checkpoint_state_dir):
            logger.info(
                f"training complete, removing checkpoint state: {checkpoint_state_dir}"
            )
            remove_path_with_retry(Path(checkpoint_state_dir))
        checkpoint_ckpt = os.path.join(
            args.output_dir,
            get_checkpoint_ckpt_name(args, "." + args.save_model_as),
        )
        if os.path.exists(checkpoint_ckpt):
            logger.info(f"removing checkpoint weights: {checkpoint_ckpt}")
            remove_path_with_retry(Path(checkpoint_ckpt))

    def save_final(self, network: Any, global_step: int, num_train_epochs: int) -> None:
        """Write the final ``<output_name>.<ext>`` checkpoint. Main-process only."""
        if not self.accelerator.is_main_process:
            return
        args = self.args
        ckpt_name = get_last_ckpt_name(args, "." + args.save_model_as)
        self.save(ckpt_name, network, global_step, num_train_epochs)
        logger.info("model saved.")

    def save_interrupt_state(
        self, network: Any, global_step: int, epoch: int, *, save_weights: bool = True
    ) -> str:
        """Persist the newest complete state for a cooperative stop.

        The interrupted state has its own directory and is never removed by
        normal-run cleanup.  This makes an explicit stop win over an older
        rolling checkpoint during automatic resume while leaving the final
        product untouched.
        """

        args = self.args
        if self.accelerator.is_main_process and save_weights:
            ext = "." + args.save_model_as
            interrupted_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME) + "-interrupted" + ext
            self.save(interrupted_name, network, global_step, epoch)
        self.accelerator.wait_for_everyone()
        state_dir = get_interrupted_state_dir(args)
        self._saving_interrupted_state = True
        try:
            _atomic_accelerator_save_state(self.accelerator, state_dir)
        finally:
            self._saving_interrupted_state = False
        logger.info("saved cooperative-stop state at step %s: %s", global_step, state_dir)
        return state_dir

    # Short alias used by stop/daemon integration and external embedders.
    save_interrupted = save_interrupt_state
