"""Experimental training entry-points: ip-adapter, easycontrol, turbo, chimera.

These are wired up under ``make exp-*`` / ``python tasks.py exp-*`` to keep
the unstable methods visually separate from the shipped ones (lora family,
modulation guidance, hydra). Each ``cmd_*`` is a thin shim that translates env
vars + extra argv into the right ``train.py`` (via ``accelerate launch``) or
``scripts/preprocess/*.py`` call.
"""

from __future__ import annotations

import os

from scripts.tasks import preprocess as _preprocess
from scripts.tasks._common import PY, _preset, bespoke_preset_flags, run, train

# EasyControl control-task projects under easycontrol_adapters/. Each maps to a
# configs/methods/<name>.toml that swaps the cond source / caption policy and an
# easycontrol_adapters/<name>/ project (mangafy/prep etc.). Selected at runtime
# via the EASYADAPTER env var (exported by the Makefile), e.g.
# ``make exp-easycontrol EASYADAPTER=colorize``.
_EASYADAPTERS = {"colorize"}


def _easyadapter() -> str:
    """Resolve the EASYADAPTER env var (validated). "" → default easycontrol."""
    adapter = (os.environ.get("EASYADAPTER") or "").strip()
    if adapter and adapter not in _EASYADAPTERS:
        raise SystemExit(
            f"Unknown EASYADAPTER={adapter!r}. Known: {sorted(_EASYADAPTERS)}."
        )
    return adapter


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
    preset_flags = bespoke_preset_flags(_preset())
    run([PY, "-m", "scripts.distill_turbo.distill", *preset_flags, *extra])


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


def cmd_ip_adapter(extra):
    train("ip_adapter", extra)


def cmd_ip_adapter_preprocess(extra):
    """Full IP-Adapter preprocess.

    IP-Adapter shares the LoRA pipeline's data layout — source images live in
    ``image_dataset/`` and caches in ``post_image_dataset/lora/``. This is just
    a convenience alias for ``make preprocess`` + ``make preprocess-pe`` so the
    GUI's IP-Adapter tab and ``make exp-ip-adapter-preprocess`` keep working.
    """
    _preprocess.cmd_preprocess(extra)
    _preprocess.cmd_preprocess_pe(extra)


def cmd_easycontrol(extra):
    """EasyControl. ``EASYADAPTER=<name>`` selects a control-task project under
    easycontrol_adapters/ (e.g. ``colorize``) → runs configs/methods/<name>.toml;
    unset → the default ref==target easycontrol.toml."""
    train(_easyadapter() or "easycontrol", extra)


def cmd_easycontrol_download(extra):
    """Download an EasyControl control-task project's extra weights.

    ``EASYADAPTER=colorize`` fetches the Sketch2Manga screening weights
    (``models/sketch2manga/``) used by the learned Phase B condition synthesizer
    (``easycontrol_adapters/colorization/prep.py --engine sd``). The default
    EasyControl (no adapter) needs no extra weights beyond the Anima base.
    """
    from scripts.tasks import downloads as _downloads

    adapter = _easyadapter()
    if adapter == "colorize":
        _downloads.cmd_download_sketch2manga(extra)
        return
    print(
        "Default EasyControl needs no extra weights (uses the Anima base from "
        "`make download-models`). Set EASYADAPTER=colorize for the Sketch2Manga "
        "screening weights."
    )


def cmd_easycontrol_preprocess(extra):
    """Full EasyControl preprocess: VAE latents + text-encoder outputs.

    Source: ``easycontrol-dataset/``  Caches: ``post_image_dataset/easycontrol/``.

    ``EASYADAPTER=colorize`` instead builds the colorization *condition* cache
    (mangafy the existing color images → VAE-encode into
    ``post_image_dataset/colorize_cond/``); the color target latents + TE are
    reused from the LoRA cache, so no target re-encode is needed. See
    ``easycontrol_adapters/colorization/prep.py``.
    """
    adapter = _easyadapter()
    if adapter == "colorize":
        run([PY, "easycontrol_adapters/colorization/prep.py", *extra])
        return

    src = "easycontrol-dataset"
    dst = "post_image_dataset/easycontrol"
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
        ]
    )
    run(
        [
            PY,
            "scripts/preprocess/cache_text_embeddings.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            "--caption_shuffle_variants",
            "4",
            "--caption_tag_dropout_rate",
            "0.1",
        ]
    )
