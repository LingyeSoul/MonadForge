"""Inference entry-points for shipped methods (test / test-* commands).

All variants share ``INFERENCE_BASE`` from ``_common`` and add method-specific
flags. Experimental inference commands (exp-test-postfix*, exp-test-prefix,
exp-test-ref) live in ``scripts/experimental_tasks/inference.py``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from ._common import (
    INFERENCE_BASE,
    ROOT,
    _random_ref_image,
    latest_hydra,
    latest_lora,
    latest_output,
    run,
)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _mod_flags() -> list[str]:
    """Resolve latest distilled pooled_text_proj for ``MOD=1``."""
    return ["--pooled_text_proj", str(latest_output("pooled_text_proj"))]


def _base_test_args(*, lora_default: bool = True) -> list[str]:
    """Build the shared ``inference.py`` argv prefix used by every ``test*`` command.

    Honors three env levers so they compose uniformly across ``test``,
    ``test-smc-cfg``, ``test-dcw``, ``test-dcw-v4``:

    - ``NOLORA=1`` skips ``--lora_weight`` (bare DiT). When unset, ``lora_default``
      decides whether the caller wants a LoRA by default — ``test-dcw-v4`` opts
      out (DCW v4 is meant to ride on the bare DiT unless the user adds one).
    - ``SPECTRUM=1`` appends Spectrum flags.
    - ``SPD=1`` appends SPD (Spectral Progressive Diffusion) flags. Mutually
      exclusive with ``SPECTRUM=1`` (both replace the denoise loop).
    - ``MOD=1`` appends ``--pooled_text_proj <latest>``.
    - ``DAVE=1`` appends the DAVE DC-attenuation flags (``--dave auto``); tune via
      ``DAVE_STRENGTH=``, ``DAVE_SIGMA='lo,hi'`` and ``DAVE_TAU=`` (early-step cutoff).
    """
    args = list(INFERENCE_BASE)
    nolora_env = os.environ.get("NOLORA")
    if nolora_env is None:
        include_lora = lora_default
    else:
        include_lora = not _env_truthy("NOLORA")
    if include_lora:
        args += ["--lora_weight", str(latest_lora())]
    if _env_truthy("SPECTRUM") and _env_truthy("SPD"):
        raise SystemExit(
            "SPECTRUM=1 and SPD=1 are mutually exclusive (both replace the denoise loop)."
        )
    if _env_truthy("SPECTRUM"):
        args += _spectrum_flags()
    if _env_truthy("SPD"):
        args += _spd_flags()
    if _env_truthy("MOD"):
        args += _mod_flags()
    if _env_truthy("DAVE"):
        args += _dave_flags()
    return args


def _dave_flags() -> list[str]:
    """DAVE DC-attenuation (training-free diversity). ``DAVE_STRENGTH``,
    ``DAVE_SIGMA='lo,hi'`` and ``DAVE_TAU`` tune the live knobs; all optional.
    ``DAVE_TAU`` (the paper's early-step cutoff, e.g. 0.15) overrides ``DAVE_SIGMA``."""
    flags = ["--dave", "auto"]
    if s := os.environ.get("DAVE_STRENGTH", "").strip():
        flags += ["--dave_strength", s]
    if win := os.environ.get("DAVE_SIGMA", "").strip():
        lo, hi = (x.strip() for x in win.split(","))
        flags += ["--dave_sigma_lo", lo, "--dave_sigma_hi", hi]
    if tau := os.environ.get("DAVE_TAU", "").strip():
        flags += ["--dave_tau", tau]
    if blk := os.environ.get("DAVE_BLOCKS", "").strip():
        lo, hi = (x.strip() for x in blk.split(","))
        flags += ["--dave_block_lo", lo, "--dave_block_hi", hi]
    return flags


def _spectrum_flags(stop_caching_step: int = 27) -> list[str]:
    return [
        "--spectrum",
        "--spectrum_window_size",
        "2.0",
        "--spectrum_flex_window",
        "0.25",
        "--spectrum_warmup",
        "7",
        "--spectrum_w",
        "0.3",
        "--spectrum_m",
        "3",
        "--spectrum_lam",
        "0.1",
        "--spectrum_stop_caching_step",
        str(stop_caching_step),
        "--spectrum_calibration",
        "0.0",
    ]


def _spd_flags() -> list[str]:
    """SPD single-late knee: one handoff 0.5 → 1.0 at σ0.7. Override on the CLI
    with --spd_stages / --spd_transition_sigmas (passed via ``extra``)."""
    return [
        "--spd",
        "--spd_stages",
        "0.5",
        "1.0",
        "--spd_transition_sigmas",
        "0.5",
    ]


def cmd_test(extra):
    """Inference with the latest LoRA. See ``_base_test_args`` for env levers."""
    run([*_base_test_args(), *extra])


def cmd_test_hydra(extra):
    # Uses the moe sibling (router-live); static-merge is auto-skipped in
    # library/inference_pipeline.py:_is_hydra_moe detection.
    run([*INFERENCE_BASE, "--lora_weight", str(latest_hydra()), *extra])


def cmd_test_merge(extra):
    """Inference with a baked (merged) DiT from MODEL_DIR (default 'output_temp').

    MODEL_DIR accepts either a directory (picks the latest
    ``*_merged.safetensors`` inside) or a direct ``.safetensors`` path. The
    merged file is a standalone DiT (LoRA folded in), so no ``--lora_weight``
    is passed. The trailing ``--dit`` overrides the base one in
    ``INFERENCE_BASE`` (argparse keeps the last value).
    """
    target = Path(os.environ.get("MODEL_DIR", "output_temp"))
    if not target.is_absolute():
        target = ROOT / target
    if target.is_file():
        chosen = target
    elif target.is_dir():
        candidates = sorted(
            target.glob("*_merged.safetensors"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"No '*_merged.safetensors' files found in {target}", file=sys.stderr)
            sys.exit(1)
        chosen = candidates[0]
    else:
        print(f"MODEL_DIR path not found: {target}", file=sys.stderr)
        sys.exit(1)
    run([*INFERENCE_BASE, "--dit", str(chosen), *extra])


def cmd_test_dcw(extra):
    """Inference with latest LoRA + DCW post-step correction.

    Defaults bake in λ=0.01 + one_minus_sigma schedule (see
    bench/dcw/findings.md). Override via --dcw_lambda / --dcw_schedule in extra.
    Honors SPECTRUM / MOD / NOLORA env levers (see ``_base_test_args``).
    """
    run([*_base_test_args(), "--dcw", "--dcw_lambda", "0.01", *extra])


def cmd_test_smc_cfg(extra):
    """Inference with latest LoRA + SMC-CFG (arXiv:2603.03281).

    Production defaults (λ=5, α=0.2). Override via --smc_cfg_lambda /
    --smc_cfg_alpha in extra. Honors SPECTRUM / MOD / NOLORA env levers
    (see ``_base_test_args``); composes with --dcw via extra.
    """
    run([*_base_test_args(), "--smc_cfg", *extra])


def _latest_fusion_head() -> str:
    """Resolve the most recent fusion_head.safetensors under any DCW root.

    Scans output/dcw/ (new `make dcw` output), post_image_dataset/dcw/
    (legacy), and bench/dcw/results/ (legacy). Newest mtime wins.
    """
    from pathlib import Path

    roots = [
        Path("output/dcw"),
        Path("post_image_dataset/dcw"),
        Path("bench/dcw/results"),
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("*/fusion_head.safetensors"))
    if not candidates:
        raise SystemExit(
            "no fusion_head.safetensors found under output/dcw/, "
            "post_image_dataset/dcw/, or bench/dcw/results/ — "
            "run `make dcw-train` first"
        )
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def cmd_test_dcw_v4(extra):
    """Inference with DCW learnable calibrator (no LoRA by default).

    Auto-resolves the most recent fusion_head.safetensors. Pass
    --dcw_calibrator <path> (or legacy --dcw_v4 <path>) in extra to override.
    Honors SPECTRUM / MOD / NOLORA env levers (see ``_base_test_args``).
    Defaults to NOLORA semantics; set ``NOLORA=0`` to attach the latest LoRA,
    or pass ``--lora_weight <path>`` in extra to attach a specific one.
    """
    extra_has_calib = any(a == "--dcw_calibrator" or a == "--dcw_v4" for a in extra)
    calib_args = [] if extra_has_calib else ["--dcw_calibrator", _latest_fusion_head()]
    run([*_base_test_args(lora_default=False), *calib_args, *extra])


def cmd_test_spectrum_dcw(extra):
    """Spectrum + DCW composed. Equivalent to ``make test SPECTRUM=1 --dcw``."""
    run(
        [
            *INFERENCE_BASE,
            "--lora_weight",
            str(latest_lora()),
            *_spectrum_flags(stop_caching_step=27),
            "--dcw",
            *extra,
        ]
    )


def cmd_test_dcw_v4_spectrum(extra):
    """Spectrum + DCW learnable calibrator composed.

    Spectrum knobs match ``cmd_test`` with stop_caching_step=27 to match
    DCW's 28-step contract, plus DCW calibrator (auto-resolves the most recent
    fusion_head.safetensors). Pass --dcw_calibrator <path> in extra to override.
    """
    extra_has_calib = any(a == "--dcw_calibrator" or a == "--dcw_v4" for a in extra)
    calib_args = [] if extra_has_calib else ["--dcw_calibrator", _latest_fusion_head()]
    run(
        [
            *INFERENCE_BASE,
            "--lora_weight",
            str(latest_lora()),
            *_spectrum_flags(stop_caching_step=27),
            *calib_args,
            "--infer_steps",
            "28",
            *extra,
        ]
    )


def cmd_test_easycontrol(extra):
    """Inference with latest EasyControl weight.

    Reference image is taken from REF_IMAGE env or the first positional arg.
    Falls back to a random image from ``easycontrol-dataset/`` (the EasyControl
    source layout) when neither is supplied.
    PROMPT, NEG, EC_SCALE env vars override defaults. Saves to
    output/tests/easycontrol/ and copies the ref image alongside the generated
    output as ``<name>_ref.png``.

    ``EASYADAPTER=colorize`` targets the colorization checkpoint
    (``anima_colorize``), saves to output/tests/colorize/, defaults the ref to a
    random image under ``post_image_dataset/resized/`` (feed a real B&W manga page
    via REF_IMAGE), and defaults to an EMPTY prompt (caption-free colorization).

    Examples:
      python tasks.py test-easycontrol ref.png --prompt "a girl in a coffee shop"
      REF_IMAGE=ref.png EC_SCALE=0.8 python tasks.py test-easycontrol
      python tasks.py test-easycontrol         # random ref from easycontrol-dataset/
      REF_IMAGE=manga.png EASYADAPTER=colorize python tasks.py test-easycontrol
    """
    adapter = (os.environ.get("EASYADAPTER") or "").strip()
    is_colorize = adapter == "colorize"
    weight_name = "anima_colorize" if is_colorize else "anima_easycontrol"
    out_sub = "colorize" if is_colorize else "easycontrol"
    ref_fallback_dir = (
        ROOT / "post_image_dataset" / "resized"
        if is_colorize
        else ROOT / "easycontrol-dataset"
    )

    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ref_fallback_dir) or ""
    if not ref_image:
        print(
            "Usage: python tasks.py test-easycontrol <ref_image> [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py test-easycontrol [extra...]\n"
            f"   (no ref given and {ref_fallback_dir.name}/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)

    save_dir = ROOT / "output" / "tests" / out_sub
    save_dir.mkdir(parents=True, exist_ok=True)

    args = [
        *INFERENCE_BASE,
        "--save_path",
        str(save_dir),
        "--easycontrol_weight",
        str(latest_output(weight_name)),
        "--easycontrol_image",
        ref_image,
        "--easycontrol_image_match_size",
    ]
    if scale := os.environ.get("EC_SCALE"):
        args += ["--easycontrol_scale", scale]
    if prompt := os.environ.get("PROMPT"):
        args += ["--prompt", prompt]
    elif is_colorize and not any(a == "--prompt" for a in extra):
        # caption-free default for colorization (empty prompt → uncond text path)
        args += ["--prompt", ""]
    if neg := os.environ.get("NEG"):
        args += ["--negative_prompt", neg]
    args += list(extra)
    run(args)

    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_ref.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        ref_dst = pngs[0].with_name(pngs[0].stem + "_ref.png")
        shutil.copy(ref_image, ref_dst)
        print(f"  > Ref pasted: {ref_dst}")
