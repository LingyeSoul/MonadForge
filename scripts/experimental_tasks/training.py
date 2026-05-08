"""Experimental training entry-points: apex, postfix, ip-adapter, easycontrol.

These are wired up under ``make exp-*`` / ``python tasks.py exp-*`` to keep
the unstable methods visually separate from the shipped ones (lora family,
modulation guidance, hydra). Each ``cmd_*`` is a thin shim that translates env
vars + extra argv into the right ``train.py`` (via ``accelerate launch``) or
``preprocess/*.py`` call.
"""

from __future__ import annotations

from scripts.tasks import preprocess as _preprocess
from scripts.tasks._common import PY, run, train


def cmd_apex(extra):
    train("apex", extra)


def cmd_apex_combined(extra):
    """APEX combined-3F: shipped APEX + Forward 2 t-shift (Δt = -0.05).

    Selects ``configs/methods/apex_combined.toml``. The adapter dispatch in
    ``library/training/method_adapter.py`` matches both ``method == "apex"``
    and ``method.startswith("apex_")``, so ``ApexMethodAdapter`` is picked
    up automatically; ``apex_dt`` from the toml flows through to Forward 2
    via the standard merge chain.
    """
    train("apex_combined", extra)


def cmd_apex_temporal(extra):
    """APEX temporal-2F: t-shift only, no ConditionShift, no L_fake (2 fwd/step).

    Selects ``configs/methods/apex_temporal.toml``. Same adapter dispatch as
    ``cmd_apex_combined`` (``method.startswith("apex_")``); the temporal-2F
    branch is gated by ``apex_temporal_only=true`` in the toml.
    """
    train("apex_temporal", extra)


def cmd_postfix(extra):
    train("postfix", extra)


def cmd_soft_tokens(extra):
    train("soft_tokens", extra)


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
    train("easycontrol", extra)


def cmd_easycontrol_preprocess(extra):
    """Full EasyControl preprocess: VAE latents + text-encoder outputs.

    Source: ``easycontrol-dataset/``  Caches: ``post_image_dataset/easycontrol/``.
    """
    src = "easycontrol-dataset"
    dst = "post_image_dataset/easycontrol"
    run(
        [
            PY,
            "preprocess/cache_latents.py",
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
            "preprocess/cache_text_embeddings.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--caption_shuffle_variants",
            "4",
            "--caption_tag_dropout_rate",
            "0.1",
        ]
    )
