"""Experimental training entry-points: turbo, spd, chimera, byg.

These are wired up under ``make exp-*`` / ``python tasks.py exp-*`` to keep
the unstable methods visually separate from the shipped ones (lora family,
modulation guidance, hydra, EasyControl). Each ``cmd_*`` is a thin shim that
translates env vars + extra argv into the right ``train.py`` (via
``accelerate launch``) or ``scripts/preprocess/*.py`` call.

(EasyControl graduated to the shipped ``make easycontrol*`` targets — see
``scripts/tasks/training.py``.)
"""

from __future__ import annotations

from scripts.tasks._common import (
    PY,
    _preset,
    bespoke_preset_flags,
    run,
    train,
)


def cmd_turbo(extra):
    """Turbo Anima — DP-DMD distillation (docs: docs/experimental/dpdmd.md).

    Bypasses train.py / accelerate (single-GPU bespoke loop, like distill-mod).
    Reads ``configs/methods/turbo.toml``; trailing args are forwarded so user
    CLI flags override TOML values, e.g.::

        make exp-turbo                                  # defaults: rank=64, 2-step
        make exp-turbo ARGS="--student_rank 64 --iterations 5000"
        make exp-turbo ARGS="--single_prompt_idx 0"     # Phase 0 single-prompt overfit

    Honors ``PRESET`` (default ``default``) — translates ``blocks_to_swap`` and
    ``gradient_checkpointing`` from ``configs/presets.toml`` into CLI flags so
    ``make exp-turbo PRESET=low_vram`` enables grad ckpt + unsloth offload, and
    ``PRESET=half/quarter/tenth`` shrinks the dataset via ``--sample_ratio``.
    ``extra`` is appended last, so user CLI overrides win.
    """
    extra = list(extra or [])
    preset_flags = bespoke_preset_flags(_preset())
    argv = ["-m", "scripts.distill_turbo.distill", *preset_flags, *extra]
    run([PY, *argv])


def cmd_spd(extra):
    """SPD fine-tuning LoRA — §4.3 trajectory adapter (proposal: docs/proposal/spd_finetune_lora.md).

    "Case B" of the SPD investigation. Bypasses train.py / accelerate (single-GPU
    bespoke loop, like distill-mod / turbo). Reads ``configs/methods/spd.toml``;
    trailing args are forwarded so user CLI flags override TOML values, e.g.::

        make exp-spd                                   # defaults: rank=32, single-late schedule
        make exp-spd ARGS="--iterations 2000 --single_prompt_idx 0"   # Phase 0 overfit
        make exp-spd ARGS="--stages 0.5 0.75 1.0 --transition_sigmas 0.6 0.4"
        make exp-spd ARGS="--torch_compile"            # per-stage static-shape compile

    ``--torch_compile`` pads each stage to its own constant token count so
    torch.compile traces only len(stages) fwd+bwd graphs (not one per
    aspect-bucket); forces attn_mode=flex. Keeps low-res stages cheap.

    Trains a plain LoRA to follow the SPD multi-resolution trajectory; output is
    a normal LoRA — infer with the SPD sampler (``make exp-test-spd``) at the
    *same* schedule (snapshotted into the safetensors metadata). Honors
    ``PRESET`` like ``exp-turbo`` (block swap / grad ckpt / sample_ratio).
    """
    preset_flags = bespoke_preset_flags(_preset())
    run([PY, "scripts/distill_spd.py", *preset_flags, *extra])


def cmd_soft_tokens(extra):
    train("soft_tokens", extra)


def cmd_chimera(extra):
    """ChimeraHydra (dual-pool additive routing — docs/proposal/chimera_hydra.md).

    Drives ``configs/methods/chimera.toml``: OrthoHydra split into a content
    pool (K_c=3, per-layer rank-R router on pooled lx) and a freq pool
    (K_f=3, network-level FreqRouter on concat(FEI, σ-features)). Pool
    outputs are added (no multiplicative gate, no σ-band overlap mask).
    Single-phase co-training; per-pool balance loss; T-LoRA mask on the
    content branch only.
    """
    train("chimera", extra)


def cmd_byg(extra):
    """BYG — Bootstrap Your Generator unpaired instruction editing.

    Plain rank-64 LoRA trained with a multi-forward unpaired objective (bootstrap
    rollout + DDS prior + cycle + identity), conditioned on a parameter-free
    token-concat source latent. Reads ``configs/methods/byg.toml``.

    Run ``exp-byg-data`` first to build the per-image edit-tuple sidecars under
    ``post_image_dataset/byg/``. The image VAE/TE caches are the standard
    ``preprocess`` ones (the source image IS the training image).
    """
    train("byg", extra)


def cmd_byg_data(extra):
    """Build BYG edit-tuple sidecars (tag-swap) into ``post_image_dataset/byg/``.

    One offline pass over the captioned corpus emitting
    ``<stem>_byg.safetensors`` (4 encoded role conditionings) per image. Pass
    ``--limit N`` for a quick smoke subset, ``--overwrite`` to rebuild.
    """
    run(
        [
            PY,
            "scripts/byg/build_edit_tuples.py",
            "--dir",
            "image_dataset",
            "--cache_dir",
            "post_image_dataset/byg",
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            *extra,
        ]
    )
