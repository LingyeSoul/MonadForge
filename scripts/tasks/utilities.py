"""Misc utility entry-points: merge, comfy-batch, test-unit, update,
export-logs, print-config."""

from __future__ import annotations

import os

from ._common import bespoke_preset_flags, PY, _preset, run


def cmd_merge(extra):
    """Bake latest LoRA in ADAPTER_DIR (env, default 'output/ckpt') into the base DiT."""
    adapter_dir = os.environ.get("ADAPTER_DIR", "output/ckpt")
    multiplier = os.environ.get("MULTIPLIER", "1.0")
    run(
        [
            PY,
            "scripts/toolkits/merge_to_dit.py",
            "--adapter_dir",
            adapter_dir,
            "--multiplier",
            multiplier,
            *extra,
        ]
    )


def cmd_merge_loras(extra):
    """Merge N LoRA adapters into ONE LoRA via scripts/merge_loras.py.

    Concat-exact fusion (no SVD): rank grows to Σrᵢ. Adapter paths (positional,
    ≥2) plus ``--analyze`` / ``--out`` / ``--weights`` / ``--dtype`` /
    ``--normalize`` all come from the WebUI via ``extra``."""
    run([PY, "scripts/toolkits/merge_loras.py", *extra])


def cmd_comfy_batch(extra):
    """Run a ComfyUI workflow as a batch.

    Workflow via ``W=`` (bare names resolve under ``workflows/``) or positional
    ``ARGS``. For workflows with a ``LoadImage`` node, ``IMAGES=<dir>`` switches
    on per-image sequential mode (default ``../comfy/input/to_colorize``):

        make comfy-batch W=colorize.json
        make comfy-batch W=colorize.json IMAGES=/path/to/imgs

    ``PROMPTS=<file>`` (bare names resolve under ``workflows/``) iterates the
    prompt text as a third axis alongside artist.txt / chara.txt:

        make comfy-batch W=modhydra.json PROMPTS=preferred.txt

    ``RANDOMS=<file>`` (default ``workflows/randoms.yaml``, bare names resolve
    under ``workflows/``) is the grouped pool the ``__random__`` placeholders
    draw from — ``__random:<group>__`` picks one group, bare ``__random__`` the
    file's ``default:`` groups. Re-rolled per job, not an axis.
    """
    workflow = os.environ.get("W") or (
        extra[0] if extra else "workflows/modhydra-simple.json"
    )
    if os.sep not in workflow and "/" not in workflow:
        workflow = f"workflows/{workflow}"
    remaining = extra[1:] if (extra and not os.environ.get("W")) else list(extra)

    prompts = os.environ.get("PROMPTS")
    if prompts and "--prompts" not in remaining:
        if os.sep not in prompts and "/" not in prompts:
            prompts = f"workflows/{prompts}"
        remaining = ["--prompts", prompts, *remaining]

    randoms = os.environ.get("RANDOMS")
    if randoms and "--randoms" not in remaining:
        if os.sep not in randoms and "/" not in randoms:
            randoms = f"workflows/{randoms}"
        remaining = ["--randoms", randoms, *remaining]

    images_dir = os.environ.get("IMAGES", "../comfy/input/to_colorize")
    if images_dir and "--images_dir" not in remaining:
        remaining = ["--images_dir", images_dir, *remaining]

    run([PY, "scripts/toolkits/comfy_batch.py", workflow, *remaining])


def cmd_test_unit(extra):
    """Run smoke/unit tests, split for speed.

    The ``slow`` suites (daemon job queue) are sleep/subprocess-bound and run
    faster under xdist since their waits overlap; the fast unit tests run
    faster serially (xdist's per-worker startup dominates their short
    runtime). Falls back to a single serial run when xdist is missing
    (``pip install pytest-xdist`` / the ``dev`` extra) or the caller passes
    their own args (``ARGS=…``), which take full control.
    """
    import importlib.util

    if extra or importlib.util.find_spec("xdist") is None:
        run([PY, "-m", "pytest", "-q", "tests/", *extra])
        return
    run([PY, "-m", "pytest", "-q", "-m", "not slow and not qwen_training and not training_step", "tests/"])
    run([PY, "-m", "pytest", "-q", "-m", "slow and not qwen_training and not training_step", "-n", "2", "tests/"])


def cmd_update(extra):
    """Update anima_lora from a GitHub release (preserves datasets/output/models;
    prompts on configs/methods/ + configs/gui-methods/ conflicts; runs uv sync)."""
    run([PY, "scripts/update.py", *extra])


def cmd_vendor_sync(extra):
    """Refresh custom_nodes/*/_vendor/ trees from the live library.* sources.

    Run before bumping a custom-node version / publishing — the bundled
    vendor copies (tagger + directedit) are how the ComfyUI nodes import
    their inference subset when not running inside the anima_lora repo.
    """
    run([PY, "scripts/release/sync_vendor.py", *extra])


def cmd_export_logs(extra):
    """Dump TB scalar logs to JSON. RUN=<dir> (default output/logs), ALL=1, JSONL=1.

    ``SUMMARY=1`` skips the full series and prints max-step + last value per tag
    (the "where is this run at" digest — see also ``make run-status``).
    """
    run_path = os.environ.get("RUN", "output/logs")
    cmd = [PY, "scripts/toolkits/export_logs_json.py", run_path]
    if os.environ.get("ALL"):
        cmd.append("--all")
    if os.environ.get("JSONL"):
        cmd.append("--jsonl")
    if os.environ.get("SUMMARY"):
        cmd.append("--summary")
    run([*cmd, *extra])


def cmd_run_status(extra):
    """One-line status for a training run, read from its progress.jsonl.

    ``RUN=<output_name|path>`` picks the run (default: the most recently updated
    stream under ``output/logs``). Reports step/total, it/s, ETA, last losses and
    last checkpoint for train.py runs and the bespoke ``make turbo`` loop alike.
    ``ARGS="--list"`` reports every run; ``ARGS="--json"`` for the raw dict.
    """
    target = os.environ.get("RUN")
    run([PY, "scripts/run_status.py", *([target] if target else []), *extra])


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


def cmd_distill_mod(extra):
    """Distill the pooled_text_proj MLP for modulation guidance.

    Honors ``PRESET`` (default ``default``) — translates ``blocks_to_swap`` and
    ``gradient_checkpointing`` from ``configs/presets.toml`` into CLI flags so
    ``make distill-mod PRESET=low_vram`` enables grad ckpt + unsloth offload.
    Trailing ``extra`` args are appended last, so user CLI overrides win.

    Saves to ``output/ckpt/pooled_text_proj.safetensors`` so ``make test MOD=1``
    picks it up automatically.
    """
    preset_flags = bespoke_preset_flags(_preset())
    run(
        [
            PY,
            "-m",
            "scripts.distill_mod.distill",
            "--data_dir",
            "post_image_dataset/lora",
            "--dit_path",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            "--output_path",
            "output/ckpt/pooled_text_proj.safetensors",
            "--attn_mode",
            "flash",
            *preset_flags,
            *extra,
        ]
    )


def cmd_distill_prep(extra):
    """Pre-stage artifacts for ``make distill-mod``.

    Phase 1: emits ``post_image_dataset/_anima_uncond_te.safetensors``
    (T5("") cross-attn baseline) — consumed as the student's unconditional
    text input, replacing the zeroed-crossattn shortcut. ``make preprocess-te``
    already produces this for free; this Phase 1 block is the explicit
    re-stager (useful with ``--overwrite`` after a model swap).

    Phase 2: emits teacher-synthesized clean latents under
    ``post_image_dataset/distill_mod_synth/`` (same NPZ layout as
    ``cache_latents.py``). Train with
    ``make distill-mod ARGS='--synth_data_dir post_image_dataset/distill_mod_synth'``
    to fit on the teacher's manifold (paper-faithful; removes real-vs-teacher
    gap that floors val loss).

    Skip flags forwarded via ``extra``: ``--skip_uncond``, ``--skip_synth``,
    ``--max_samples N``, etc.
    """
    run([PY, "-m", "scripts.distill_mod.prep", *extra])
