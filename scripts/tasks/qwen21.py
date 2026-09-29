"""Qwen-Image-2.1 sidecar commands; Anima's training/config paths are unchanged."""

from scripts.tasks._common import PY, queue_command, run


def _run_job(command: str, module: str, extra: list[str]) -> None:
    args = [arg for arg in extra if arg != "--queue"]
    argv = ["-m", module, *args]
    if "--queue" in extra:
        queue_command(command, argv, stall_timeout=900.0)
    else:
        run([PY, *argv])


def cmd_cache(extra: list[str]) -> None:
    """Cache --src into --out using --model_dir or explicit Comfy component paths.

    --queue submits through the daemon with a 900-second silence budget.
    Full flags: python -m scripts.qwen21.cache --help
    """
    _run_job("qwen21-cache", "scripts.qwen21.cache", extra)


def cmd_train(extra: list[str]) -> None:
    """Train --cache using --model_dir or --dit; write --output (no encoders needed).

    --model_dir retains the official config and weight-shard layout.
    --dit adds a standalone Comfy BF16 file; it does not replace directory support.
    Omit --blocks_to_swap and --activation_reserve_gb for upstream auto-sizing
    from free GPU VRAM and cache token counts; --blocks_to_swap 0 disables swap.
    Alpha defaults to rank; warmup steps follow epochs, cache size and ratio.
    Effective parameters are recorded in the JSON report beside the LoRA.
    --queue submits through the daemon with a 900-second silence budget.
    Full flags: python -m scripts.qwen21.train --help
    """
    _run_job("qwen21-train", "scripts.qwen21.train", extra)


def cmd_generate(extra: list[str]) -> None:
    """Render --prompt using --model_dir or explicit Comfy component paths.

    --queue submits through the daemon with a 900-second silence budget.
    Full flags: python -m scripts.qwen21.generate --help
    """
    _run_job("qwen21-generate", "scripts.qwen21.generate", extra)


def cmd_processor(extra: list[str]) -> None:
    """Fetch only processor/* into --out (default models/qwen_image_2.1)."""
    run([PY, "-m", "scripts.qwen21.processor", *extra])


def cmd_cache_train(extra: list[str]) -> None:
    """Run cache then train in one job; failure in caching prevents training.

    --cache-args and --train-args are JSON argv arrays; --queue uses the daemon.
    The WebUI constructs these from the canonical request forms.
    """
    _run_job("qwen21-cache-train", "scripts.qwen21.workflow", extra)
