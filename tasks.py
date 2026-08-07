#!/usr/bin/env python3
"""Cross-platform task runner -- replaces Makefile for Windows compatibility.

Usage:
    python tasks.py <command> [extra args...]

Examples:
    python tasks.py lora
    python tasks.py lora --network_dim 32 --max_train_epochs 64
    python tasks.py test
    python tasks.py test                     # add SPECTRUM=1 to enable Spectrum
    python tasks.py test                     # add MOD=1 to enable modulation guidance
    python tasks.py test                     # add NOLORA=1 to run against the bare DiT
    python tasks.py download-models
    python tasks.py turbo                    # DP-DMD 4-step distillation
    python tasks.py exp-chimera              # experimental method
    python tasks.py exp-test-spd             # experimental inference

Command implementations live under ``scripts/tasks/`` (shipped methods) and
``scripts/experimental_tasks/`` (unstable methods exposed under ``exp-*``).
This file is just a name → callable dispatch table.
"""

import importlib
import sys


class _LazyCmd:
    """A command callable that imports its module only when first invoked.

    tasks.py is a pure dispatch table: a single ``python tasks.py <cmd>`` needs
    exactly one command module, but importing all of them up front to build
    ``COMMANDS`` cost ~100ms (the daemon client's urllib/http chain dominates) —
    wasted for the common case (``make <target>`` immediately spawns a child for
    the real work). Wrapping each entry defers the import to dispatch time, so
    every target stops paying for modules it won't run.
    """

    def __init__(self, modpath: str, name: str):
        self._modpath = modpath
        self._name = name

    def resolve(self):
        return getattr(importlib.import_module(self._modpath), self._name)

    def __call__(self, extra):
        return self.resolve()(extra)


class _LazyModule:
    """``<alias>.cmd_x`` in the COMMANDS table → a _LazyCmd, with no import yet."""

    def __init__(self, modpath: str):
        self._modpath = modpath

    def __getattr__(self, name: str) -> _LazyCmd:
        return _LazyCmd(self._modpath, name)


curate = _LazyModule("scripts.tasks.curate")
daemon = _LazyModule("scripts.tasks.daemon")
dcw = _LazyModule("scripts.tasks.dcw")
downloads = _LazyModule("scripts.tasks.downloads")
inference = _LazyModule("scripts.tasks.inference")
masking = _LazyModule("scripts.tasks.masking")
preprocess = _LazyModule("scripts.tasks.preprocess")
sr = _LazyModule("scripts.tasks.sr")
staged_resolution = _LazyModule("scripts.tasks.staged_resolution")
tagger = _LazyModule("scripts.tasks.tagger")
training = _LazyModule("scripts.tasks.training")
utilities = _LazyModule("scripts.tasks.utilities")
v100_flash = _LazyModule("scripts.tasks.v100_flash")
exp_inference = _LazyModule("scripts.experimental_tasks.inference")
exp_training = _LazyModule("scripts.experimental_tasks.training")

COMMANDS = {
    "v100-flash-install": (
        v100_flash.cmd_install,
        "Build and install the pinned flash-attention-v100 cp313 wheel into .venv.",
    ),
    "v100-flash-validate": (
        v100_flash.cmd_validate,
        "Run the strict V100 FlashAttention capture, matrix, compile, and Anima gates.",
    ),
    # ── Training ──────────────────────────────────────────────────────
    "lora": (
        training.cmd_lora,
        "LoRA family (lora|tlora|hydralora via configs/methods/lora.toml)",
    ),
    "lora-gui": (
        training.cmd_lora_gui,
        "Train from a self-contained configs/gui-methods/<variant>.toml "
        "(variant from GUI_PRESETS env or 1st positional; e.g. tlora, hydralora).",
    ),
    "staged-train": (
        staged_resolution.cmd_staged_train,
        "Train a saved WebUI staged-resolution profile with one continuous "
        "model/optimizer/scheduler state.",
    ),
    "turbo": (
        training.cmd_turbo,
        "Turbo (DP-DMD) distillation — bakes CFG=4 / 28-step Anima into a 4-step "
        "LoRA student (configs/methods/turbo.toml). Single-GPU bespoke loop "
        "(bypasses train.py/accelerate, like distill-mod). Output is a normal LoRA "
        "(https://huggingface.co/sorryhyun/anima-turbo-4step).",
    ),
    "easycontrol": (
        training.cmd_easycontrol,
        "EasyControl training (extended self-attn KV with VAE-encoded reference). "
        "EASYADAPTER=<task> (e.g. colorize) selects a control-task project.",
    ),
    "easycontrol-staging": (
        training.cmd_easycontrol_staging,
        "Generate an EasyControl adapter's staging dataset (no VAE/TE caching). "
        "EASYADAPTER=near_twin → mine the in-artist near-twin pair tree.",
    ),
    "easycontrol-preprocess": (
        training.cmd_easycontrol_preprocess,
        "Full EasyControl preprocess: latents + text emb. "
        "Source: easycontrol-dataset/  Cache: post_image_dataset/easycontrol/.",
    ),
    # ── ResShift SR/RSD sidecar (optional `sr` dependency group) ────────
    "sr-setup": (
        sr.cmd_sr_setup,
        "Install the optional SR dependencies and verify the vendored ResShift tree.",
    ),
    "sr-prep": (
        sr.cmd_sr_prep,
        "Build the frozen synthetic-LR evaluation set from image_dataset/.",
    ),
    "sr-phase0": (
        sr.cmd_sr_phase0,
        "Run released ResShift x4 on the evaluation set and produce metrics/montage.",
    ),
    "sr-test": (
        sr.cmd_sr_test,
        "Run tiled SR on IN=<image|dir> (VERSION=v3|v2|x2|x4ft|x4s4).",
    ),
    "sr-build-hr-pool": (
        sr.cmd_sr_build_hr_pool,
        "Build a sharp HR pool for SR/RSD training.",
    ),
    "sr-detect-text": (
        sr.cmd_sr_detect_text,
        "Detect CTD text regions for SR crop sampling (not OCR transcription).",
    ),
    "sr-train": (
        sr.cmd_sr_train,
        "Finetune a ResShift teacher on the HR pool (VERSION=x2|x4|x4s4).",
    ),
    "sr-rsd-train": (
        sr.cmd_sr_rsd_train,
        "Distill a ResShift teacher to a one-step RSD student.",
    ),
    "sr-rsd-dryrun": (
        sr.cmd_sr_rsd_dryrun,
        "Run the RSD VRAM feasibility dry-run without training.",
    ),
    "sr-rsd-infer": (
        sr.cmd_sr_rsd_infer,
        "Run one-step RSD inference and MUSIQ scoring.",
    ),
    # ── Inference ─────────────────────────────────────────────────────
    "test": (
        inference.cmd_test,
        "Inference with latest LoRA. SPECTRUM=1 enables Spectrum acceleration; "
        "MOD=1 adds the latest distilled pooled_text_proj (modulation guidance); "
        "NOLORA=1 runs against the bare DiT (skips --lora_weight).",
    ),
    "test-hydra": (
        inference.cmd_test_hydra,
        "Inference with latest HydraLoRA moe (router-live)",
    ),
    "test-merge": (
        inference.cmd_test_merge,
        "Inference with latest *_merged.safetensors (MODEL_DIR=..., default 'output_temp')",
    ),
    "test-dcw": (
        inference.cmd_test_dcw,
        "Inference with latest LoRA + DCW post-step bias correction. "
        "Honors SPECTRUM=1 / MOD=1 / NOLORA=1.",
    ),
    "test-smc-cfg": (
        inference.cmd_test_smc_cfg,
        "Inference with latest LoRA + SMC-CFG (sliding-mode control CFG, arXiv:2603.03281). "
        "Honors SPECTRUM=1 / MOD=1 / NOLORA=1.",
    ),
    "test-dcw-v4": (
        inference.cmd_test_dcw_v4,
        "Inference with DCW v4 learnable calibrator (auto-resolves fusion_head.safetensors). "
        "No LoRA by default; SPECTRUM=1 / MOD=1 / NOLORA=0 (attach latest LoRA) all compose.",
    ),
    "dcw": (
        dcw.cmd_dcw,
        "Calibrate DCW v4: sample 5 aspect buckets (default 130×1 seeds, "
        "shuffle_seed=0) + train fusion head",
    ),
    "dcw-train": (
        dcw.cmd_dcw_train,
        "Train-only on existing bench/dcw/results/ pool (~30s, no sampling)",
    ),
    "test-spectrum-dcw": (
        inference.cmd_test_spectrum_dcw,
        "Spectrum-accelerated inference + DCW post-step bias correction",
    ),
    "test-dcw-v4-spectrum": (
        inference.cmd_test_dcw_v4_spectrum,
        "Spectrum-accelerated inference + DCW v4 learnable calibrator (auto-resolves fusion_head.safetensors)",
    ),
    "test-easycontrol": (
        inference.cmd_test_easycontrol,
        "Inference with latest EasyControl weight. Usage: test-easycontrol <ref_image> [--prompt ... --easycontrol_scale ...]",
    ),
    "test-turbo": (
        inference.cmd_test_turbo,
        "Inference with latest Turbo student LoRA at 4 steps, cfg=1.0 "
        "(CFG is baked into the student).",
    ),
    # ── Preprocess ────────────────────────────────────────────────────
    "preprocess": (
        preprocess.cmd_preprocess,
        "Full preprocessing (resize + VAE + text embeddings + caption index "
        "when the tagger vocab is present)",
    ),
    "preprocess-config": (
        preprocess.cmd_preprocess_config,
        "Preprocess the dirs named in a --dataset_config TOML (resize --src "
        "→ image_dir, then VAE + TE caches → cache_dir). Used by the trainer node.",
    ),
    "staged-preprocess": (
        staged_resolution.cmd_staged_preprocess,
        "Preprocess the source images once for every tier in a saved WebUI "
        "staged-resolution profile.",
    ),
    "preprocess-resize": (
        preprocess.cmd_preprocess_resize,
        "Resize images to bucket resolutions",
    ),
    "preprocess-reconcile": (
        preprocess.cmd_preprocess_reconcile,
        "Remove resized/latent/PE/mask caches stale for the configured "
        'target_res (dry-run; ARGS="--delete" to act). Run after changing tiers.',
    ),
    "preprocess-vae": (preprocess.cmd_preprocess_vae, "Cache VAE latents"),
    "preprocess-te": (preprocess.cmd_preprocess_te, "Cache text encoder embeddings"),
    "preprocess-captions": (
        preprocess.cmd_preprocess_captions,
        "Write corrected caption sidecars next to resized preprocessing images",
    ),
    "preprocess-pe": (
        preprocess.cmd_preprocess_pe,
        "Cache PE-Core vision-encoder features into the LoRA cache dir. "
        "Consumed by CMMD validation and the DCW v4 fusion head.",
    ),
    "preprocess-pe-spatial": (
        preprocess.cmd_preprocess_pe_spatial,
        "Cache PE-Spatial (dense, B16-512) patch tokens into the LoRA cache "
        "dir as {stem}_anima_pe_spatial.safetensors. Consumed by REPA v2.",
    ),
    "preprocess-cond-resize": (
        preprocess.cmd_preprocess_cond_resize,
        "Resize conditioning images to bucket resolutions",
    ),
    "preprocess-cond-vae": (
        preprocess.cmd_preprocess_cond_vae,
        "Cache conditioning VAE latents",
    ),
    "caption-index": (
        preprocess.cmd_caption_index,
        "Build the typed-tag caption index (character/copyright/artist groups) "
        "at post_image_dataset/captions/caption_index.json. Pure data, no GPU.",
    ),
    # ── Curation ──────────────────────────────────────────────────────
    "curate-group": (
        curate.cmd_curate_group,
        "Group dataset images by PE-Spatial visual similarity (per-artist "
        "connected-components) → post_image_dataset/groups/groups.json. The GUI "
        'Dataset tab reads it to filter by group. ARGS="--threshold 0.95".',
    ),
    # ── Anima Tagger ──────────────────────────────────────────────────
    "preprocess-tagger": (
        tagger.cmd_preprocess_tagger,
        "Build the Anima Tagger vocab/manifest + cache PE-Core & PE-Spatial "
        "features (build_vocab + build_features). Needs CAPTION_CORPUS_DIR "
        "in .env.",
    ),
    "tagger": (
        tagger.cmd_tagger,
        "Train the dual-encoder hard-routed tagger head on cached PE-Core + "
        "PE-Spatial features. Requires `make preprocess-tagger` first.",
    ),
    "test-tagger": (
        tagger.cmd_test_tagger,
        "Predict tags for a single image (--image <path>) or sample a random "
        "val-split stem. Pass --show_scores for rating + top-K kept tags.",
    ),
    "autotag": (
        tagger.cmd_autotag,
        "Autotag one image (--image <path>) with the Anima Tagger "
        "(auto-downloaded on first use); prints the predicted caption. CLI "
        "one-shot — the GUI Dataset tab uses a resident worker instead.",
    ),
    # ── Downloads ─────────────────────────────────────────────────────
    "download-models": (downloads.cmd_download_models, "Download all models"),
    "download-anima": (downloads.cmd_download_anima, "Download Anima model"),
    "download-sam3": (downloads.cmd_download_sam3, "Download SAM3 model"),
    "download-mit": (downloads.cmd_download_mit, "Download MIT model"),
    "download-pe": (
        downloads.cmd_download_pe,
        "Download PE-Core-L14-336 (img2emb encoder)",
    ),
    "download-pe-spatial": (
        downloads.cmd_download_pe_spatial,
        "Download PE-Spatial-B16-512 (Anima Tagger aux encoder)",
    ),
    "download-tagger": (
        downloads.cmd_download_tagger,
        "Download Anima Tagger v2 vocab.json (caption-index dependency; not the full model)",
    ),
    "download-danbooru-tags": (
        downloads.cmd_download_danbooru_tags,
        "Download danbooru tag tables (KR base + EN sibling) for caption correction",
    ),
    # ── Masking ───────────────────────────────────────────────────────
    "mask": (
        masking.cmd_mask,
        "Run SAM + MIT (via tempdir) and write merged masks under post_image_dataset/masks/",
    ),
    "mask-clean": (
        masking.cmd_mask_clean,
        "Remove post_image_dataset/masks/",
    ),
    # ── Daemon (serial job queue — the WebUI/CLI/MCP training runner) ──
    "daemon": (
        daemon.cmd_daemon,
        "Start the training daemon (idempotent, detached). The daemon owns a "
        "serial job queue + GPU guard; the WebUI submits training/preprocess "
        "jobs here. Jobs survive a WebUI restart.",
    ),
    "daemon-status": (
        daemon.cmd_daemon_status,
        "One JSON status object: daemon health + base URL + job summaries "
        "(--full for raw records). Passive, exit 1 when down.",
    ),
    "daemon-attach": (
        daemon.cmd_daemon_attach,
        "Non-owning viewer. JOB=<id> follows that job's stdout; otherwise "
        "tails daemon lifecycle events. Ctrl-C detaches only.",
    ),
    "daemon-kill": (
        daemon.cmd_daemon_kill,
        "Abort the running (or JOB=<id>) job and free the GPU; daemon stays up.",
    ),
    "daemon-terminate": (
        daemon.cmd_daemon_terminate,
        "Shut the whole daemon down (active job dies, queue discarded).",
    ),
    # ── Utilities ─────────────────────────────────────────────────────
    "merge": (
        utilities.cmd_merge,
        "Bake latest LoRA (ADAPTER_DIR=..., default 'output/ckpt') into base DiT",
    ),
    "merge-loras": (
        utilities.cmd_merge_loras,
        "Merge N LoRA adapters into one (concat-exact, with interference analysis)",
    ),
    "comfy-batch": (utilities.cmd_comfy_batch, "Run ComfyUI batch workflow"),
    "distill-prep": (
        utilities.cmd_distill_prep,
        'Pre-stage artifacts for distill-mod: T5("") uncond sidecar + '
        "teacher-synthetic clean latents pool (--skip_synth / --skip_uncond to "
        "stage only one).",
    ),
    "distill-mod": (
        utilities.cmd_distill_mod,
        "Distill pooled_text_proj MLP for modulation guidance",
    ),
    "test-unit": (utilities.cmd_test_unit, "Run smoke/unit tests (pytest tests/)"),
    "export-logs": (
        utilities.cmd_export_logs,
        "Dump TB scalar logs to JSON (RUN=<dir>, ALL=1 for every subrun, JSONL=1 for line-delimited)",
    ),
    "print-config": (
        utilities.cmd_print_config,
        "Dump merged config (METHOD=<name> PRESET=<name>) with provenance",
    ),
    "update": (
        utilities.cmd_update,
        "Update from GitHub release (preserves datasets/output/models, prompts on "
        "config conflicts, runs uv sync). Pass --dry-run / --version v1.0 / --no-sync.",
    ),
    "vendor-sync": (
        utilities.cmd_vendor_sync,
        "Refresh custom_nodes/*/_vendor/ from live library/* (run before publishing nodes)",
    ),
    # ── Experimental ──────────────────────────────────────────────────
    # Unstable methods kept under exp-* so they don't pollute the main command
    # surface. May produce broken output, change without notice, or be removed.
    "exp-spd": (
        exp_training.cmd_spd,
        "[experimental] SPD fine-tuning LoRA — §4.3 trajectory adapter that teaches a "
        "plain LoRA to follow the SPD multi-resolution trajectory (configs/methods/spd.toml). "
        "Single-GPU bespoke loop (bypasses train.py/accelerate, like distill-mod). "
        "Output is a normal LoRA — infer with the SPD sampler at the trained schedule.",
    ),
    "exp-soft-tokens": (
        exp_training.cmd_soft_tokens,
        "[experimental] SoftREPA-style per-layer × per-t soft tokens (training-only v1)",
    ),
    "exp-chimera": (
        exp_training.cmd_chimera,
        "[experimental] ChimeraHydra dual-pool additive routing "
        "(content + freq pools on OrthoHydra; configs/methods/chimera.toml)",
    ),
    "exp-byg": (
        exp_training.cmd_byg,
        "[experimental] BYG unpaired instruction-editing training (plain LoRA, "
        "bootstrap + DDS prior + cycle + identity; configs/methods/byg.toml). "
        "Run exp-byg-data first.",
    ),
    "exp-byg-data": (
        exp_training.cmd_byg_data,
        "[experimental] Build BYG edit-tuple sidecars (tag-swap) into "
        "post_image_dataset/byg/. Usage: exp-byg-data [--limit N --overwrite].",
    ),
    "exp-test-soft": (
        exp_inference.cmd_test_soft,
        "[experimental] Inference with latest soft_tokens weight "
        "(SoftREPA-style per-layer × per-t bank, spliced into cross-attn via "
        "monkey-patched Block.forward). Composes freely with --spectrum.",
    ),
    "exp-test-spd": (
        exp_inference.cmd_test_spd,
        "[experimental] Inference with latest SPD fine-tune LoRA on the SPD sampler "
        "at its trained schedule (read from safetensors metadata). cfg=4.0, Euler.",
    ),
    "exp-test-byg": (
        exp_inference.cmd_test_byg,
        "[experimental] Inference with latest BYG editing LoRA. Usage: exp-test-byg <ref_image> --prompt 'change background to a forest'",
    ),
    "exp-test-directedit": (
        exp_inference.cmd_test_directedit,
        "[experimental] DirectEdit on a random source image. PROMPT='...' supplies the edit "
        "instruction (appended to the Anima Tagger source caption). REF_IMAGE=path overrides the "
        "random pick. Usage: exp-test-directedit [ref_image] [extra...]",
    ),
    "exp-test-directedit-dry": (
        exp_inference.cmd_test_directedit_dry,
        "[experimental] DirectEdit functional check: random source image + random crossattn "
        "embed (no TE, no captioner); ψ_tar == ψ_src so output should reconstruct the source. "
        "REF_IMAGE=path overrides the random pick. Usage: exp-test-directedit-dry [ref_image] [extra...]",
    ),
    "exp-invert-directedit": (
        exp_inference.cmd_invert_directedit,
        "[experimental] Probe: invert the K-slot orthogonal postfix tail per image, then run "
        "DirectEdit dry mode twice (baseline T5(tags) vs T5(tags)+tail). Outputs side-by-side under "
        "output/tests/invert_directedit/<stem>/. Env: N_IMAGES (default 1), REF_IMAGE, K (default 48), "
        "INVERT_STEPS, LAMBDA_ZERO, SIGMA_MIN, BASIS, SEED.",
    ),
}


def _force_utf8_stdio():
    """Make stdout/stderr UTF-8 so non-UTF-8 consoles don't crash on glyphs.

    Several commands print Unicode status glyphs (``✓``/``✗``). On a Windows
    console whose code page isn't UTF-8 (e.g. cp949 on a Korean install)
    ``print`` raises ``UnicodeEncodeError`` and aborts the whole task. Re-encode
    stdio as UTF-8 with ``errors="replace"`` so output is never fatal — UTF-8
    when the terminal can show it, a replacement char at worst when it can't.
    Best-effort: some wrapped streams (pytest capture, certain pipes) lack
    ``reconfigure``; skip them silently.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main():
    _force_utf8_stdio()
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python tasks.py <command> [extra args...]\n")
        print("Commands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:30s} {desc}")
        print("\nExtra arguments are forwarded to the underlying command.")
        print("Example: python tasks.py lora --network_dim 32 --max_train_epochs 64")
        sys.exit(0)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run 'python tasks.py --help' for available commands", file=sys.stderr)
        sys.exit(1)

    extra = sys.argv[2:]
    fn, desc = COMMANDS[command]
    if extra and extra[0] in ("-h", "--help"):
        print(f"python tasks.py {command} -- {desc}\n")
        doc = fn.resolve().__doc__ if isinstance(fn, _LazyCmd) else fn.__doc__
        if doc:
            print(doc.strip())
        else:
            print("(no detailed help available)")
        print(
            "\nUnrecognised flags are forwarded verbatim to the underlying script. "
            "Run the underlying script with --help for its full flag set."
        )
        sys.exit(0)
    fn(extra)


if __name__ == "__main__":
    main()
