"""Misc utility entry-points: merge, comfy-batch, distill-mod, test-unit, update,
export-logs, print-config."""

from __future__ import annotations

import os
import sys

from ._common import PY, ROOT, _preset, run


def cmd_merge(extra):
    """Bake latest LoRA in ADAPTER_DIR (env, default 'output/ckpt') into the base DiT."""
    adapter_dir = os.environ.get("ADAPTER_DIR", "output/ckpt")
    multiplier = os.environ.get("MULTIPLIER", "1.0")
    run(
        [
            PY,
            "scripts/merge_to_dit.py",
            "--adapter_dir",
            adapter_dir,
            "--multiplier",
            multiplier,
            *extra,
        ]
    )


def cmd_comfy_batch(extra):
    workflow = extra[0] if extra else "workflows/modhydra.json"
    remaining = extra[1:] if extra else []
    run([PY, "scripts/comfy_batch.py", workflow, *remaining])


def _distill_preset_flags(preset: str) -> list[str]:
    """Translate ``configs/presets.toml[<preset>]`` into distill_modulation.py CLI flags.

    Honored keys:
      - ``blocks_to_swap`` → ``--blocks_to_swap N``
      - ``gradient_checkpointing`` (bool) → ``--grad_ckpt`` / ``--no_grad_ckpt``
        (paired with the script-side hardcoded ``unsloth_offload=True`` when on,
        which matches the ``unsloth_offload_checkpointing=true`` flag every
        preset that enables ckpt also sets).
      - ``sample_ratio`` → ``--sample_ratio R`` (per-bucket subsample applied
        after the train/val split — makes ``PRESET=debug/half/quarter/tenth``
        actually run on a small slice for fast iteration).

    When the preset omits ``gradient_checkpointing`` we keep the historical
    distill-mod default of ``--no_grad_ckpt`` (the trainable footprint is tiny;
    ckpt is a perf loss when VRAM isn't tight). Other preset keys are silently
    dropped.
    """
    sys.path.insert(0, str(ROOT))
    try:
        from library.config.io import load_preset_section
    except Exception as e:  # noqa: BLE001
        print(f"warn: could not import preset loader: {e}", file=sys.stderr)
        return ["--no_grad_ckpt"]

    try:
        section = load_preset_section(preset)
    except (FileNotFoundError, KeyError) as e:
        print(f"warn: preset '{preset}' not found ({e}); using distill-mod defaults", file=sys.stderr)
        return ["--no_grad_ckpt"]

    flags: list[str] = []
    if "blocks_to_swap" in section:
        flags += ["--blocks_to_swap", str(int(section["blocks_to_swap"]))]
    if "gradient_checkpointing" in section:
        flags.append("--grad_ckpt" if section["gradient_checkpointing"] else "--no_grad_ckpt")
    else:
        flags.append("--no_grad_ckpt")
    if "sample_ratio" in section:
        flags += ["--sample_ratio", str(float(section["sample_ratio"]))]
    return flags


def cmd_distill_mod(extra):
    """Distill the pooled_text_proj MLP for modulation guidance.

    Honors ``PRESET`` (default ``default``) — translates ``blocks_to_swap`` and
    ``gradient_checkpointing`` from ``configs/presets.toml`` into CLI flags so
    ``make distill-mod PRESET=low_vram`` enables grad ckpt + unsloth offload.
    Trailing ``extra`` args are appended last, so user CLI overrides win.

    Saves to ``output/ckpt/pooled_text_proj.safetensors`` so ``test-mod`` picks it
    up automatically.
    """
    preset_flags = _distill_preset_flags(_preset())
    run(
        [
            PY,
            "scripts/distill_modulation.py",
            "--data_dir",
            "post_image_dataset/lora",
            "--dit_path",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--output_path",
            "output/ckpt/pooled_text_proj.safetensors",
            "--attn_mode",
            "flash",
            *preset_flags,
            *extra,
        ]
    )


def cmd_test_unit(extra):
    run([PY, "-m", "pytest", "-q", "tests/", *extra])


def cmd_update(extra):
    """Update anima_lora from a GitHub release (preserves datasets/output/models;
    prompts on configs/methods/ + configs/gui-methods/ conflicts; runs uv sync)."""
    run([PY, "scripts/update.py", *extra])


def cmd_export_logs(extra):
    """Dump TB scalar logs to JSON. RUN=<dir> (default output/logs), ALL=1, JSONL=1."""
    run_path = os.environ.get("RUN", "output/logs")
    cmd = [PY, "scripts/export_logs_json.py", run_path]
    if os.environ.get("ALL"):
        cmd.append("--all")
    if os.environ.get("JSONL"):
        cmd.append("--jsonl")
    run([*cmd, *extra])


def cmd_print_config(extra):
    method = os.environ.get("METHOD", "lora")
    preset = _preset()
    run(
        [
            PY,
            "train.py",
            "--method",
            method,
            "--preset",
            preset,
            "--print-config",
            "--no-config-snapshot",
            *extra,
        ]
    )
