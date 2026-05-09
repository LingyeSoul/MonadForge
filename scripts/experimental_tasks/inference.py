"""Experimental inference entry-points (exp-test-* commands).

Covers the unstable methods kept under ``make exp-*``: APEX distillation,
postfix / postfix_exp / postfix_func / prefix / reference-inversion prefix,
IP-Adapter, EasyControl. Reference-image variants (exp-test-ip /
exp-test-easycontrol) accept REF_IMAGE env or first positional arg, copy the
ref alongside the generated output.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path

from scripts.tasks._common import (
    INFERENCE_BASE,
    ROOT,
    latest_output,
    run,
)

_REF_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _random_ref_image(directory: Path) -> str | None:
    if not directory.is_dir():
        return None
    pool = [p for p in directory.iterdir() if p.suffix.lower() in _REF_IMAGE_EXTS]
    if not pool:
        return None
    pick = random.choice(pool)
    print(f"  > Random ref: {pick}")
    return str(pick)


def cmd_test_apex(extra):
    # APEX silently bakes the warm-start LoRA into the DiT base at training
    # time (see networks/methods/apex.py::promote_warmstart_to_merge), so the
    # saved anima_apex.safetensors is a delta on top of that merged base. To
    # reproduce the same base at inference, stack the warm-start (read from
    # the apex run's .snapshot.toml) ahead of the apex delta.
    import tomllib

    apex_ckpt = latest_output("anima_apex")
    snapshot = apex_ckpt.with_suffix(".snapshot.toml")
    warmstart: Path | None = None
    if snapshot.is_file():
        with open(snapshot, "rb") as f:
            snap = tomllib.load(f)
        nw = snap.get("network_weights")
        if isinstance(nw, str) and nw:
            cand = Path(nw)
            if not cand.is_absolute():
                cand = ROOT / cand
            if cand.is_file():
                warmstart = cand
            else:
                print(
                    f"  ! APEX warm-start from {snapshot.name} not found at "
                    f"{cand}; skipping stack — output will likely be garbage.",
                    file=sys.stderr,
                )
    else:
        print(
            f"  ! No {snapshot.name} alongside {apex_ckpt.name}; can't recover "
            f"the warm-start path. Output will likely be garbage if the apex "
            f"run was warm-started.",
            file=sys.stderr,
        )

    lora_args = ["--lora_weight"]
    if warmstart is not None:
        lora_args += [str(warmstart), str(apex_ckpt)]
    else:
        lora_args += [str(apex_ckpt)]

    # 4 euler steps + guidance_scale=1.0 (no CFG, conditional branch only) per
    # apex.toml and docs/experimental/apex.md. guidance_scale=0.0 here previously
    # silently collapsed to uncond-only (do_cfg=True, weight=0) so the model
    # was queried with an empty prompt and produced a featureless blur.
    run(
        [
            *INFERENCE_BASE,
            *lora_args,
            "--infer_steps",
            "8",
            "--guidance_scale",
            "1.0",
            "--sampler",
            "euler",
            *extra,
        ]
    )


def cmd_test_prefix(extra):
    run(
        [*INFERENCE_BASE, "--prefix_weight", str(latest_output("anima_prefix")), *extra]
    )


def cmd_test_ref(extra):
    # Reference-inversion prefixes ride the same loader as prefix-mode tuning;
    # the prefix loader at inference hard-prepends the K slots to crossattn_emb
    # (matches exactly how invert_reference.py assembled them at training time).
    run([*INFERENCE_BASE, "--prefix_weight", str(latest_output("anima_ref")), *extra])


def cmd_test_postfix(extra):
    # exclude both _exp and _func so the vanilla postfix target doesn't grab them
    outputs = sorted(
        (
            f
            for f in (ROOT / "output" / "ckpt").glob("anima_postfix*.safetensors")
            if "_exp" not in f.name and "_func" not in f.name
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        print(
            "No 'anima_postfix*.safetensors' files found in output/ckpt/",
            file=sys.stderr,
        )
        sys.exit(1)
    run(
        [
            *INFERENCE_BASE,
            "--postfix_weight",
            str(outputs[0]),
            *extra,
        ]
    )


def cmd_test_postfix_exp(extra):
    run(
        [
            *INFERENCE_BASE,
            "--postfix_weight",
            str(latest_output("anima_postfix_exp")),
            *extra,
        ]
    )


def cmd_test_postfix_func(extra):
    run(
        [
            *INFERENCE_BASE,
            "--postfix_weight",
            str(latest_output("anima_postfix_func")),
            *extra,
        ]
    )


def cmd_test_ip(extra):
    """Inference with latest IP-Adapter weight.

    Reference image is taken from REF_IMAGE env or the first positional arg.
    Falls back to a random image from ``post_image_dataset/resized/`` (the
    IP-Adapter source layout) when neither is supplied.
    PROMPT, NEG, IP_SCALE env vars override defaults. Saves to output/tests/ip/
    and copies the ref image alongside the generated output as ``<name>_ref.png``.

    Examples:
      python tasks.py exp-test-ip ref.png --prompt "a girl in a coffee shop"
      REF_IMAGE=ref.png IP_SCALE=0.8 python tasks.py exp-test-ip
      python tasks.py exp-test-ip                 # random ref from post_image_dataset/resized/
    """
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ROOT / "post_image_dataset" / "resized") or ""
    if not ref_image:
        print(
            "Usage: python tasks.py exp-test-ip <ref_image> [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-test-ip [extra...]\n"
            "   (no ref given and post_image_dataset/resized/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)

    save_dir = ROOT / "output" / "tests" / "ip"
    save_dir.mkdir(parents=True, exist_ok=True)

    args = [
        *INFERENCE_BASE,
        "--save_path",
        str(save_dir),
        "--ip_adapter_weight",
        str(latest_output("anima_ip_adapter")),
        "--ip_image",
        ref_image,
        "--ip_image_match_size",
    ]
    if scale := os.environ.get("IP_SCALE"):
        args += ["--ip_scale", scale]
    args += ["--prompt", os.environ.get("PROMPT") or "double peace, v v,"]
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


def cmd_test_directedit(extra):
    """DirectEdit on a random source image, seeded by wd-swinv2-tagger-v3.

    Pipeline:
      1. Pick source image (REF_IMAGE env, first positional arg, or random
         from ``post_image_dataset/resized/``).
      2. Run wd-swinv2-tagger-v3 on the source -> ``src_tags`` caption
         (downloaded on first use to ``models/captioners/wd-swinv2-tagger-v3/``).
      3. Build edit prompts:
            prompt_src = src_tags
            prompt_tar = src_tags + ", " + PROMPT
         (PROMPT env or ``--prompt`` extra arg supplies the edit instruction.
         Defaults to ``"double peace"``.)
      4. Call ``scripts/edit.py`` (DirectEdit invert + edit) using the same
         DiT/VAE/TE trio as the other inference targets.
      5. Save under ``output/tests/directedit/`` and copy the source image
         alongside as ``<name>_src.png``.

    Examples:
      make exp-test-directedit PROMPT='double peace'
      REF_IMAGE=foo.png make exp-test-directedit PROMPT='glasses'
      python tasks.py exp-test-directedit foo.png --prompt 'smile'
    """
    # 1. Resolve source image — same logic as cmd_test_ip / cmd_test_easycontrol.
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ROOT / "post_image_dataset" / "resized") or ""
    if not ref_image:
        print(
            "Usage: python tasks.py exp-test-directedit [<ref_image>] [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-test-directedit\n"
            "   (no ref given and post_image_dataset/resized/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Pull the user-supplied edit instruction. PROMPT env wins; fall back
    #    to a ``--prompt`` flag in extra; final default = "double peace".
    edit_prompt = os.environ.get("PROMPT", "").strip()
    cleaned_extra: list[str] = []
    skip_next = False
    for j, tok in enumerate(extra):
        if skip_next:
            skip_next = False
            continue
        if tok == "--prompt" and j + 1 < len(extra):
            if not edit_prompt:
                edit_prompt = extra[j + 1]
            skip_next = True
            continue
        cleaned_extra.append(tok)
    extra = cleaned_extra
    if not edit_prompt:
        edit_prompt = "double peace, v v. She is showing double peace"

    # 3. Run a tagger on the source. ``TAGGER`` env picks the backend
    #    (``anima`` / ``wd``); default is auto — Anima if the checkpoint
    #    is present, else wd-tagger as a portable fallback.
    sys.path.insert(0, str(ROOT))
    from PIL import Image  # noqa: PLC0415

    tagger_choice = os.environ.get("TAGGER", "").strip().lower() or "auto"
    anima_ckpt = ROOT / "models" / "captioners" / "anima-tagger-v1" / "model.safetensors"
    if tagger_choice == "auto":
        tagger_choice = "anima" if anima_ckpt.exists() else "wd"

    print(f"  > tagging source: {ref_image} (tagger={tagger_choice})")
    if tagger_choice == "anima":
        if not anima_ckpt.exists():
            print(
                f"  ! TAGGER=anima requested but {anima_ckpt} is missing — "
                "train via `python -m scripts.anima_tagger.cli`. "
                "Falling back to wd-tagger.",
                file=sys.stderr,
            )
            tagger_choice = "wd"
    if tagger_choice == "anima":
        from library.captioning.anima_tagger import AnimaTagger  # noqa: PLC0415

        tagger = AnimaTagger(ckpt_dir=anima_ckpt.parent)
    elif tagger_choice == "wd":
        from library.captioning.wd_tagger import WDTagger  # noqa: PLC0415

        tagger = WDTagger()
    else:
        raise SystemExit(
            f"unknown TAGGER={tagger_choice!r} — expected 'anima', 'wd', or 'auto'"
        )

    src_caption = tagger.predict_caption(Image.open(ref_image))
    if not src_caption:
        print(
            "  ! tagger produced no tags above threshold; using empty source "
            "prompt — DirectEdit reconstruction will be weaker than usual.",
            file=sys.stderr,
        )
    print(f"  > src caption: {src_caption[:120]}{'...' if len(src_caption) > 120 else ''}")

    tar_caption = (
        f"{src_caption}, {edit_prompt}" if src_caption else edit_prompt
    )

    # 4. Save dir + edit.py invocation. Reuse INFERENCE_BASE for the model
    #    path trio (--dit / --text_encoder / --vae) so this stays in sync with
    #    the other test commands automatically.
    save_dir = ROOT / "output" / "tests" / "directedit"
    save_dir.mkdir(parents=True, exist_ok=True)

    base_iter = iter(INFERENCE_BASE)
    py = next(base_iter)
    next(base_iter)  # drop "inference.py"
    leftover_base = list(base_iter)
    args = [py, "scripts/edit.py", *_filter_inference_base_for_edit(leftover_base)]
    args += [
        "--image", str(ref_image),
        "--prompt_src", src_caption,
        "--prompt_tar", tar_caption,
        "--save_path", str(save_dir),
    ]
    args += list(extra)
    run(args)

    # 5. Copy the source alongside the edited output for side-by-side review.
    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_src.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        src_dst = pngs[0].with_name(pngs[0].stem + "_src.png")
        shutil.copy(ref_image, src_dst)
        print(f"  > Source pasted: {src_dst}")


def cmd_test_directedit_dry(extra):
    """DirectEdit functional sanity check using preprocessed cross-emb variants.

    Bypasses wd-tagger and the text encoder. Auto-resolves the source image's
    `_anima_te.safetensors` cache (the file `cache_text_embeddings.py` writes
    — same format the trainer consumes) and runs one invert + edit pass per
    stored variant with ψ_tar == ψ_src. With `--caption_shuffle_variants N`
    caches, this sweeps v0 (pristine) + v1..v{N-1} (tag-shuffled). Each pass
    should reconstruct the source; divergence flags numeric drift in
    invert/edit_forward against that variant's cross-emb representation.

    Examples:
      make exp-test-directedit-dry
      REF_IMAGE=foo.png make exp-test-directedit-dry
      python tasks.py exp-test-directedit-dry foo.png --seed 7
    """
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ROOT / "post_image_dataset" / "resized") or ""
    if not ref_image:
        print(
            "Usage: python tasks.py exp-test-directedit-dry [<ref_image>] [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-test-directedit-dry\n"
            "   (no ref given and post_image_dataset/resized/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Auto-resolve the matching TE cache file. Try the standard cache_dir
    # location first (post_image_dataset/lora/ — what configs/base.toml's
    # subset cache_dir points at), then the legacy sidecar location next to
    # the source image.
    stem = Path(ref_image).stem
    suffix = "_anima_te.safetensors"
    candidates = [
        ROOT / "post_image_dataset" / "lora" / f"{stem}{suffix}",
        Path(ref_image).parent / f"{stem}{suffix}",
    ]
    cache_path = next((p for p in candidates if p.is_file()), None)
    if cache_path is None:
        print(
            f"  ! No TE cache found for {ref_image}.\n"
            f"    Looked in: {candidates[0]}\n"
            f"           and: {candidates[1]}\n"
            "    Run `make preprocess-te` first (with --caption_shuffle_variants N "
            "to get a multi-variant cache).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  > TE cache: {cache_path}")

    save_dir = ROOT / "output" / "tests" / "directedit_dry"
    save_dir.mkdir(parents=True, exist_ok=True)

    base_iter = iter(INFERENCE_BASE)
    py = next(base_iter)
    next(base_iter)  # drop "inference.py"
    leftover_base = list(base_iter)
    args = [py, "scripts/edit.py", *_filter_inference_base_for_edit(leftover_base)]
    args += [
        "--image", str(ref_image),
        "--cached_embed", str(cache_path),
        "--save_path", str(save_dir),
    ]
    args += list(extra)
    run(args)

    # Copy the source alongside the reconstruction for side-by-side review.
    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_src.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        src_dst = pngs[0].with_name(pngs[0].stem + "_src.png")
        shutil.copy(ref_image, src_dst)
        print(f"  > Source pasted: {src_dst}")


def _filter_inference_base_for_edit(args: list[str]) -> list[str]:
    """Drop INFERENCE_BASE flags that ``scripts/edit.py`` doesn't accept.

    INFERENCE_BASE bundles plenty of generation-only flags (--prompt, --seed,
    --image_size, --infer_steps, --sampler, etc.) that overlap with or
    conflict with edit.py's own. Keep only the model/path flags we actually
    want to forward; let edit.py supply its own defaults for the rest.
    """
    keep_flags = {
        "--dit",
        "--text_encoder",
        "--vae",
        "--vae_chunk_size",
        "--attn_mode",
    }
    boolean_flags = {"--vae_disable_cache"}
    out: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in keep_flags and i + 1 < len(args):
            out.extend([tok, args[i + 1]])
            i += 2
        elif tok in boolean_flags:
            out.append(tok)
            i += 1
        else:
            i += 1
    return out


def cmd_test_easycontrol(extra):
    """Inference with latest EasyControl weight.

    Reference image is taken from REF_IMAGE env or the first positional arg.
    Falls back to a random image from ``easycontrol-dataset/`` (the EasyControl
    source layout) when neither is supplied.
    PROMPT, NEG, EC_SCALE env vars override defaults. Saves to
    output/tests/easycontrol/ and copies the ref image alongside the generated
    output as ``<name>_ref.png``.

    Examples:
      python tasks.py exp-test-easycontrol ref.png --prompt "a girl in a coffee shop"
      REF_IMAGE=ref.png EC_SCALE=0.8 python tasks.py exp-test-easycontrol
      python tasks.py exp-test-easycontrol         # random ref from easycontrol-dataset/
    """
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ROOT / "easycontrol-dataset") or ""
    if not ref_image:
        print(
            "Usage: python tasks.py exp-test-easycontrol <ref_image> [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-test-easycontrol [extra...]\n"
            "   (no ref given and easycontrol-dataset/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)

    save_dir = ROOT / "output" / "tests" / "easycontrol"
    save_dir.mkdir(parents=True, exist_ok=True)

    args = [
        *INFERENCE_BASE,
        "--save_path",
        str(save_dir),
        "--easycontrol_weight",
        str(latest_output("anima_easycontrol")),
        "--easycontrol_image",
        ref_image,
        "--easycontrol_image_match_size",
    ]
    if scale := os.environ.get("EC_SCALE"):
        args += ["--easycontrol_scale", scale]
    if prompt := os.environ.get("PROMPT"):
        args += ["--prompt", prompt]
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
