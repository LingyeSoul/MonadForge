#!/usr/bin/env python3
"""Cross-platform task runner; the Makefile forwards every target here.

Usage:
    python tasks.py <command> [extra args...]

Examples:
    python tasks.py lora
    python tasks.py lora --network_dim 32 --max_train_epochs 64
    python tasks.py test                     # env: SPECTRUM=1 or SPD=1, MOD=1, NOLORA=1
    python tasks.py download-models
    python tasks.py turbo                    # DP-DMD 4-step distillation
    python tasks.py exp-soft-tokens          # experimental method

Command implementations live under ``scripts/tasks/`` (shipped methods) and
``scripts/experimental_tasks/`` (unstable methods exposed under ``exp-*``).
This file is just a name → callable dispatch table.
"""

import importlib
import sys


class _LazyCmd:
    """A command callable that imports its module only when first invoked, so a
    single ``python tasks.py <cmd>`` imports only the one module it runs."""

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
downloads = _LazyModule("scripts.tasks.downloads")
inference = _LazyModule("scripts.tasks.inference")
masking = _LazyModule("scripts.tasks.masking")
preprocess = _LazyModule("scripts.tasks.preprocess")
tagger = _LazyModule("scripts.tasks.tagger")
training = _LazyModule("scripts.tasks.training")
utilities = _LazyModule("scripts.tasks.utilities")
exp_inference = _LazyModule("scripts.experimental_tasks.inference")
exp_training = _LazyModule("scripts.experimental_tasks.training")

dcw = _LazyModule("scripts.tasks.dcw")
sr = _LazyModule("scripts.tasks.sr")
staged_resolution = _LazyModule("scripts.tasks.staged_resolution")
qwen21 = _LazyModule("scripts.tasks.qwen21")
moss = _LazyModule("scripts.tasks.moss")
curation = _LazyModule("scripts.tasks.curation")
v100_flash = _LazyModule("scripts.tasks.v100_flash")

COMMANDS = {
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
    "soup": (
        training.cmd_soup,
        "Uncond-init soup (TARGET=<artist> required): uncond inter-train on the "
        "target's artist shard (reused if it exists) → 3 seeded fine-tunes → ΔW "
        "soup SVD-truncated to network_dim (bench/memorization/report.md).",
    ),
    "turbo": (
        training.cmd_turbo,
        "Turbo (DP-DMD) distillation — bakes CFG=4 / 28-step Anima into a 4-step "
        "LoRA student (configs/methods/turbo.toml). Single-GPU bespoke loop "
        "(bypasses train.py/accelerate). Output is a normal LoRA "
        "(https://huggingface.co/sorryhyun/anima-turbo-4step).",
    ),
    "easycontrol": (
        training.cmd_easycontrol,
        "EasyControl training (extended self-attn KV with VAE-encoded reference). "
        "EASYADAPTER=<task> (e.g. colorize) selects a control-task project.",
    ),
    "i2i-stage": (
        training.cmd_i2i_stage,
        "Generate an i2i task's staging tree (i2i_tasks/<task>, no VAE/TE "
        "caching) under post_image_dataset/i2i/. TASK=<task> (colorize, inpaint, "
        "region, near_twins, …); read by easycontrol-preprocess and the MOSS i2i cache.",
    ),
    "i2i-preprocess": (
        training.cmd_i2i_preprocess,
        "Cache a staged i2i task for one model: TASK=<task> MODEL=anima (EasyControl "
        "encode, = easycontrol-preprocess) | moss (configs/moss/i2i/<task>/cache.toml "
        "→ target latents + TE + reference latents, daemon job).",
    ),
    "easycontrol-staging": (
        training.cmd_easycontrol_staging,
        "Alias of i2i-stage with EASYADAPTER=<task>.",
    ),
    "easycontrol-preprocess": (
        training.cmd_easycontrol_preprocess,
        "Full EasyControl preprocess: latents + text emb. "
        "Source: easycontrol-dataset/ (or TASK staging)  Cache: post_image_dataset/i2i/.",
    ),
    # ── Training daemon ───────────────────────────────────────────────
    "daemon": (
        daemon.cmd_daemon,
        "Start the local training-job daemon (idempotent; detached, waits for /health).",
    ),
    "daemon-jobs": (
        daemon.cmd_daemon_jobs,
        "Job history as lines, OLDEST first — the newest row is last, so "
        "'| tail -5' means the five most recent. Filters: "
        "--limit N/--all/--running/--failed/--done/--state s; reads the "
        "on-disk records when the daemon is down. One job's full record + its "
        "result envelope: python -m anima_daemon status <id>.",
    ),
    "daemon-status": (
        daemon.cmd_daemon_status,
        "Daemon health in one line, then the running / paused / queued jobs "
        "(exit 1 = daemon down). JOB=<id>: that job's full record + result "
        "envelope as JSON.",
    ),
    "daemon-log": (
        daemon.cmd_daemon_log,
        "Dump a job's captured stdout from disk: JOB=<id> (default: the most "
        'recent job), ARGS="-n 200" to bound the tail (-n 0 = whole file). '
        "Works on finished jobs and with the daemon down — daemon-attach only "
        "follows a live stream.",
    ),
    "daemon-run": (
        daemon.cmd_daemon_run,
        'Run an arbitrary command on the GPU queue: ARGS="<script.py> [flags]". '
        "Attach-by-default (ctrl-C detaches, exits with the job's code); --queue "
        "detaches, --inline bypasses the daemon; --stall-timeout S (0=off) for a "
        "legitimately quiet loop.",
    ),
    "daemon-wait": (
        daemon.cmd_daemon_wait,
        "Block until JOB=<id> (or the active job) is terminal, print its record "
        "+ bench result envelope as JSON, and exit with the job's own exit code. "
        'ARGS="--timeout S" bounds the wait (exit 124).',
    ),
    "daemon-attach": (
        daemon.cmd_daemon_attach,
        "Follow the daemon (read-only): JOB=<id>'s stdout, else the active "
        "job's, else the event stream (ARGS=--events forces it); ctrl-C "
        "detaches only — training keeps running.",
    ),
    "daemon-pause": (
        daemon.cmd_daemon_pause,
        "Freeze the running job (or JOB=<id>) in place — SIGSTOP the process "
        "tree; VRAM stays allocated, SM util drops to zero, resume is instant. "
        "The queue does not advance past it. RELEASE=1: the trainer saves a "
        "resumable state at its next step and exits instead — GPU freed, queue "
        "advances, daemon-resume relaunches it (train jobs only).",
    ),
    "daemon-resume": (
        daemon.cmd_daemon_resume,
        "Thaw a paused job (or JOB=<id>) — SIGCONT the process tree back to "
        "running; a RELEASE-paused job is re-enqueued at the front with --resume.",
    ),
    "daemon-kill": (
        daemon.cmd_daemon_kill,
        "Abort the running job (or JOB=<id>) and free the GPU; daemon stays up "
        "and starts the next queued job.",
    ),
    "daemon-terminate": (
        daemon.cmd_daemon_terminate,
        "Stop the daemon entirely (active job killed, GPU freed, queue discarded).",
    ),
    "daemon-prune": (
        daemon.cmd_daemon_prune,
        "Delete old terminal job dirs (the daemon also sweeps at boot). Dry-run "
        'unless ARGS="--apply"; --days N / --keep N override the retention knobs.',
    ),
    "runpod": (
        daemon.cmd_runpod,
        "RunPod pods (RUNPOD_API_KEY from the env or .env): "
        'ARGS="gpus" (price/stock), "ls", "create --gpu ID [--gpu ID …] '
        '[--dry-run] [--wait]" (sorryhyun96/anima_lora, 200 GB disk, CUDA ≥ 13.2; '
        'bills from create), "wait|ssh|terminate <pod_id>". '
        "Same as python -m anima_daemon.runpod; anima_daemon/README.md § RunPod pods.",
    ),
    # ── Inference ─────────────────────────────────────────────────────
    "test": (
        inference.cmd_test,
        "Inference with latest LoRA. SPECTRUM=1 enables Spectrum acceleration; "
        "MOD=1 adds the latest distilled pooled_text_proj (modulation guidance); "
        "NOLORA=1 runs against the bare DiT (skips --lora_weight).",
    ),
    "gen": (
        inference.cmd_gen,
        "Batch generation via the daemon (attach-by-default): queues behind a "
        "live train run instead of OOM-colliding, survives the terminal, and "
        "lands a result manifest in the job record. Same argv/env levers as "
        "'test' (NOLORA/SPECTRUM/MOD/DAVE/FSG). --queue detaches; --inline "
        "bypasses the daemon. Target adapters/prompts/seeds via ARGS.",
    ),
    "test-hydra": (
        inference.cmd_test_hydra,
        "Inference with latest HydraLoRA moe (router-live)",
    ),
    "test-merge": (
        inference.cmd_test_merge,
        "Inference with latest *_merged.safetensors (MODEL_DIR=..., default 'output_temp')",
    ),
    "test-smc-cfg": (
        inference.cmd_test_smc_cfg,
        "Inference with latest LoRA + SMC-CFG (sliding-mode control CFG, arXiv:2603.03281). "
        "Honors SPECTRUM=1 / MOD=1 / NOLORA=1.",
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
    "preprocess-demote": (
        preprocess.cmd_preprocess_demote,
        "Emit σ-demote sibling latents (sigma_lowres: demoted_{H}x{W} keys "
        "inside the native npz) for --sigma_lowres training — one pass per "
        "route in preprocess.toml's sigma_demote (default 1024:896)",
    ),
    "preprocess-te": (preprocess.cmd_preprocess_te, "Cache text encoder embeddings"),
    "preprocess-captions": (
        preprocess.cmd_preprocess_captions,
        "Write corrected caption sidecars next to resized preprocessing images",
    ),
    "preprocess-pe": (
        preprocess.cmd_preprocess_pe,
        "Cache PE-Core vision-encoder features into the LoRA cache dir. "
        "Consumed by CMMD validation.",
    ),
    "preprocess-pe-spatial": (
        preprocess.cmd_preprocess_pe_spatial,
        "Cache PE-Spatial (dense, B16-512) patch tokens into the LoRA cache "
        "dir as {stem}_anima_pe_spatial.safetensors. Consumed by REPA v2.",
    ),
    "caption-index": (
        preprocess.cmd_caption_index,
        "Build the typed-tag caption index (character/copyright/artist groups) "
        "at post_image_dataset/captions/caption_index.json. Pure data, no GPU.",
    ),
    "caption-autotag": (
        preprocess.cmd_caption_autotag,
        "Auto-tag the dataset with the Anima Tagger and write .txt caption "
        'sidecars. ARGS="--mode missing|merge|overwrite" (default missing = '
        'only uncaptioned images). Dry-run by default; ARGS="--apply" writes, '
        "then `make preprocess-te` is REQUIRED.",
    ),
    "caption-position": (
        preprocess.cmd_caption_position,
        "Append position-aware clauses ('On the left, <tags>. ...') to "
        "multi-subject captions via SAM3 + Anima Tagger. Writes the resized "
        "captions (post_image_dataset/), never the image_dataset/ master. "
        'Dry-run by default; ARGS="--apply" writes, then `make preprocess-te` '
        "is REQUIRED.",
    ),
    "caption-full": (
        preprocess.cmd_caption_full,
        "The whole derived-caption chain in the one order that composes: "
        "position clauses -> OCR read -> OCR lines attached to the caption as "
        "text clauses ('Japanese text reads as \"…\"'). Writes the resized "
        "captions (post_image_dataset/), never the image_dataset/ master — so "
        'it WRITES by default (ARGS="--dry_run" to plan instead), and `make '
        'preprocess-te` after it is REQUIRED. ARGS="--skip_position --skip_ocr '
        '--ocr_min_det 0.6" re-combines from the sidecars already read.',
    ),
    # ── Curation ──────────────────────────────────────────────────────
    "curate-group": (
        curate.cmd_curate_group,
        "Group dataset images by PE-Spatial visual similarity (per-artist "
        "connected-components) → workspace/groups/groups.json. The GUI "
        'Dataset tab reads it to filter by group. ARGS="--threshold 0.95".',
    ),
    # ── Anima Tagger ──────────────────────────────────────────────────
    "tagger": (
        tagger.cmd_tagger,
        "Build the Anima Tagger vocab/manifest (build_vocab; needs "
        "CAPTION_CORPUS_DIR in .env), or any other CLI mode via "
        'ARGS="--mode predict|scan_role_markers|derive_groups". PE-head '
        "training was archived 2026-08-27 — sidecar training runs via "
        "daemon-run -m anime_tools.tagger.cli.train_sidecar.",
    ),
    "test-tagger": (
        tagger.cmd_test_tagger,
        "Predict tags for a single image (--image <path>) or sample a random "
        "val-split stem. Pass --show_scores for rating + top-K kept tags.",
    ),
    "tagger-dbv4": (
        tagger.cmd_tagger_dbv4,
        "Build the dbv4-backed (external caformer_b36) tagger checkpoint dir "
        "from the archived anima-tagger-v5's vocab; sidecar training runs via "
        "daemon-run -m anime_tools.tagger.cli.train_sidecar.",
    ),
    "autotag": (
        tagger.cmd_autotag,
        "Autotag one image (--image <path>) with the Anima Tagger "
        "(auto-downloaded on first use); prints the predicted caption. CLI "
        "one-shot — the GUI Dataset tab uses a resident worker instead.",
    ),
    # ── Downloads ─────────────────────────────────────────────────────
    "download-models": (
        downloads.cmd_download_models,
        "Download the first-run model set (Anima base + PE + CJK vocab pack + "
        "tagger + tag KB)",
    ),
    "download-list": (
        downloads.cmd_download_list,
        "List every model catalog row: installed / MISSING, repo, destination",
    ),
    "download-model": (
        downloads.cmd_download_model,
        "Download models by pack, legacy alias or catalog id; ARGS='ocr sam3 pe', "
        "no args lists them",
    ),
    "download-anima": (downloads.cmd_download_anima, "Download Anima model"),
    "download-anima-variant": (
        downloads.cmd_download_anima_variant,
        "Download an alternate Anima base DiT (aesthetic / turbo / preview / "
        "2.9B); ARGS=<name>, no args lists them",
    ),
    "download-sam3": (downloads.cmd_download_sam3, "Download SAM3 model"),
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
        "Download the Anima Tagger checkpoint (vocab/rules/thresholds/sidecar; "
        "not the gated backbone — see download-tagger-model)",
    ),
    "download-tagger-model": (
        downloads.cmd_download_tagger_model,
        "Download the full Anima Tagger (our head + the gated GPL-3.0 caformer_b36 "
        "dbv4 backbone); needs `hf auth login` and accepting the upstream repo terms",
    ),
    "download-danbooru-tags": (
        downloads.cmd_download_danbooru_tags,
        "Download danbooru tag tables (KR base + EN sibling) for caption correction",
    ),
    "download-vocab-pack": (
        downloads.cmd_download_vocab_pack,
        "Re-fetch the CJK vocab pack (JA/KO/ZH prompt + caption rows) to "
        "models/vocab_packs/; part of download-models, on by default via "
        "`vocab_pack` in configs/base.toml",
    ),
    # ── Masking ───────────────────────────────────────────────────────
    "mask": (
        masking.cmd_mask,
        "Run the SAM3 mask rules (via tempdir) and write merged masks under "
        "mask_dir (configs/preprocess.toml; default post_image_dataset/masks/). "
        "Training reads them only with masked_loss = true",
    ),
    "mask-clean": (
        masking.cmd_mask_clean,
        "Remove mask_dir/ (configs/preprocess.toml; default post_image_dataset/masks/)",
    ),
    # ── Utilities ─────────────────────────────────────────────────────
    "merge": (
        utilities.cmd_merge,
        "Bake latest LoRA (ADAPTER_DIR=..., default 'output/ckpt') into base DiT",
    ),
    "comfy-batch": (utilities.cmd_comfy_batch, "Run ComfyUI batch workflow"),
    "test-unit": (utilities.cmd_test_unit, "Run smoke/unit tests (pytest tests/)"),
    "export-logs": (
        utilities.cmd_export_logs,
        "Dump TB scalar logs to JSON (RUN=<dir>, ALL=1 for every subrun, JSONL=1 for "
        "line-delimited, SUMMARY=1 for max-step + last value per tag)",
    ),
    "run-status": (
        utilities.cmd_run_status,
        "Step/ETA/last-ckpt one-liner for a run from its progress.jsonl "
        "(RUN=<output_name|path>, default newest; ARGS='--list|--json')",
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
    # May produce broken output, change without notice, or be removed.
    "exp-soft-tokens": (
        exp_training.cmd_soft_tokens,
        "[experimental] SoftREPA-style per-layer × per-t soft tokens (training-only v1)",
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

    # MonadForge compatibility commands.
    "v100-flash-install": (
        v100_flash.cmd_install,
        "Build and install the pinned flash-attention-v100 cp313 wheel into .venv.",
    ),
    "v100-flash-validate": (
        v100_flash.cmd_validate,
        "Run the strict V100 FlashAttention capture, matrix, compile, and Anima gates.",
    ),
    "staged-train": (
        staged_resolution.cmd_staged_train,
        "Train a saved WebUI staged-resolution profile with one continuous "
        "model/optimizer/scheduler state.",
    ),
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
    "test-dcw": (
        inference.cmd_test_dcw,
        "Inference with latest LoRA + DCW post-step bias correction. "
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
    "staged-preprocess": (
        staged_resolution.cmd_staged_preprocess,
        "Preprocess the source images once for every tier in a saved WebUI "
        "staged-resolution profile.",
    ),
    "preprocess-cond-resize": (
        preprocess.cmd_preprocess_cond_resize,
        "Resize conditioning images to bucket resolutions",
    ),
    "preprocess-cond-vae": (
        preprocess.cmd_preprocess_cond_vae,
        "Cache conditioning VAE latents",
    ),
    "preprocess-tagger": (
        tagger.cmd_preprocess_tagger,
        "Build the Anima Tagger vocab/manifest + cache PE-Core & PE-Spatial "
        "features (build_vocab + build_features). Needs CAPTION_CORPUS_DIR "
        "in .env.",
    ),
    "download-mit": (downloads.cmd_download_mit, "Download MIT model"),
    "download-ms": (
        downloads.cmd_download_ms,
        "Download any ModelScope (魔搭) repo: download-ms <org/name> "
        "[--local-dir DIR] [--include GLOB]... [--force]. The download-* "
        "commands also accept --source modelscope to fetch from verified "
        "ModelScope mirrors instead of HuggingFace",
    ),
    "daemon-status": (
        daemon.cmd_daemon_status,
        "One JSON status object: daemon health + base URL + job summaries "
        "(--full for raw records). Passive, exit 1 when down.",
    ),
    "merge-loras": (
        utilities.cmd_merge_loras,
        "Merge N LoRA adapters into one (concat-exact, with interference analysis)",
    ),
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
    "curation-stage": (curation.cmd_curation_stage, "Run an immutable curation stage snapshot"),
    "moss-cache": (moss.cmd_cache, "MOSS: cache images and text"),
    "moss-train": (moss.cmd_train, "MOSS: train an adapter"),
    "moss-cache-train": (moss.cmd_cache_train, "MOSS: cache then train in one queued job"),
    "moss-generate": (moss.cmd_generate, "MOSS: generate images"),
    "moss-tune": (moss.cmd_tune, "MOSS: tune block swapping (runs training steps)"),
    "qwen21-processor": (
        qwen21.cmd_processor,
        "Download only Qwen-Image-2.1 tokenizer/processor assets",
    ),
    "qwen21-cache": (
        qwen21.cmd_cache,
        "Cache Qwen-Image-2.1 text/latents from official or ComfyUI weights",
    ),
    "qwen21-train": (
        qwen21.cmd_train,
        "Train Qwen-Image-2.1 LoRA from official shards or a ComfyUI DiT file",
    ),
    "qwen21-cache-train": (
        qwen21.cmd_cache_train,
        "Cache then train Qwen-Image-2.1 in one sequential job",
    ),
    "qwen21-generate": (
        qwen21.cmd_generate,
        "Generate Qwen-Image-2.1 LoRA/base A/B images",
    ),
    "exp-spd": (
        exp_training.cmd_spd,
        "[experimental] SPD fine-tuning LoRA — §4.3 trajectory adapter that teaches a "
        "plain LoRA to follow the SPD multi-resolution trajectory (configs/methods/spd.toml). "
        "Single-GPU bespoke loop (bypasses train.py/accelerate, like distill-mod). "
        "Output is a normal LoRA — infer with the SPD sampler at the trained schedule.",
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
    "exp-test-spd": (
        exp_inference.cmd_test_spd,
        "[experimental] Inference with latest SPD fine-tune LoRA on the SPD sampler "
        "at its trained schedule (read from safetensors metadata). cfg=4.0, Euler.",
    ),
    "exp-test-byg": (
        exp_inference.cmd_test_byg,
        "[experimental] Inference with latest BYG editing LoRA. Usage: exp-test-byg <ref_image> --prompt 'change background to a forest'",
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
    """Reconfigure stdout/stderr as UTF-8 with ``errors="replace"``.

    Status glyphs (``✓``/``✗``) otherwise raise ``UnicodeEncodeError`` on a
    non-UTF-8 Windows console (e.g. cp949). Streams without ``reconfigure``
    (pytest capture, some pipes) are skipped.
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
