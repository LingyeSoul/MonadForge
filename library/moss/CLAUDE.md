# CLAUDE.md — `library/moss/`

**This is not Anima.** MOSS-Image (OpenMOSS, Apache-2.0) is a 5B single-stream DiT on a
frozen Qwen3.5-9B text encoder and the frozen FLUX.2 VAE. The root `CLAUDE.md`
invariants describe the Anima DiT; most of them are wrong here. Shared with Anima:
`library.runtime.offloading` (+ `library.runtime.blockswap`, also used by `library/qwen21/`), `library.runtime.dynamo`, `library.env`, the daemon,
the ER-SDE sampler + CNS core (`library.inference.sampling`, `library.inference.corrections.cns_core`),
and the relational-REPA math (`library.training.repa`). `spectrum.py` imports nothing from
`networks/`: it re-implements Anima's growing-window rule and the
`networks/spectrum_forecast.py` estimator (`tests/test_moss_spectrum.py` checks it against the core).
Shared with `library/qwen21/`: `library.lines` (`requests.py` request base, `adapters.py`
LoRA / LoKr / LoHa network) and `library.runtime.blockswap.place`.
Nothing Anima-side imports `library.moss`.

| Module | Role |
|---|---|
| `model.py` | vendored `MOSSImageTransformer` / `MOSSImageScheduler` (Apache-2.0, attribution kept; state-dict names unchanged). Reference per-sample `forward`, `forward_batched` (masked text, per-sample `skip_layers`), `head` for Spectrum, `layer_hook`, `rotate` (RoPE), `patchify` / `unpatchify`, `pack_latents` / `unpack_latents` / `encode` / `decode`. i2i probe: `ref=` clean reference latents on RoPE's `r = 1` axis (grid stretched over the image's), `ref_timestep` (their own block modulation) and `ref_isolate` (reference queries see reference keys only) as attributes, one `ref_span` through the blocks |
| `requests.py` | **torch-free** `CacheRequest` / `GenerateRequest` / `TrainRequest` / `TuneSwapRequest`, `to_argv()` / `from_argv()`, model-dir resolution, the reference sampler's `DEFAULT_STEPS` / `DEFAULT_CFG` / `DEFAULT_UNCOND_SKIP` / `DEFAULT_SHIFT` |
| `loader.py` | phase split: TE (whole decoder layers paged onto the card: `offload_by_layer`, `te_reserve_gb`) → encode → free → DiT; VAE; `compile_blocks` |
| `fp8.py` | fp8 TE: `quantize_te` exports the 248 decoder linears as `float8_e4m3fn` + per-tensor `weight_scale` + `comfy_quant` (ComfyUI's mixed-precision format; `scripts/moss/quantize_te_fp8.py`, published as catalog row `moss_te_fp8`); `load_fp8_text_encoder` reads it back with `FP8Linear` (dequantised per forward). Selected by `text_encoder=fp8` on `CacheRequest` / `GenerateRequest`; TE caches stamp `encoder_id(precision)` and re-encode on a switch. Bench `bench/moss_te_fp8` |
| `cache.py` | TE cache (natural length, never padded) + latent cache (packed, BN-normalised). Caption variants `embeds_v{k}` (stored after a shared `variant_prefix`): `variant_entries` / `add_variants` write them, `read_te` rebuilds them, a re-encode keeps them while the caption is unchanged |
| `generate.py` | `run_generate(req)`: Euler or ER-SDE (`sampler`, Anima's `ERSDESampler`), CFG + uncond skip-22, Spectrum hook-in. `denoise`, `cached_embeds` (on-disk TE cache by text) and `build_spectrum(req)` are what the benches and probes reuse |
| `cns.py` | CNS noise recolor for ER-SDE on the un-patchified 32 ch grid (`model.unpatchify`) + γ estimator; bench-only so far (`project/moss_lora/sampler.md`) |
| `spectrum.py` | per-branch feature forecaster (cond and uncond forecast separately); `MossSpectrum()` defaults are `GenerateRequest`'s (`flex_window` 0.5) |
| `dave.py` | DAVE on the image tokens only, run through `model.layer_hook` (a post-block edit that sees only the rows that ran the block, so skip-22 is handled); τ is a step count; the massive-activation channels (`calibration/massive_channels.npz`) are left out of the mean. `auto` = `calibration/dave_alpha.npz`, blocks 19–30 from a DAVE-on pool sweep; the ComfyUI node `~/ComfyUI-Anima-DAVE` bundles the same pool + channel list; user doc `docs/moss/dave.md`, research `project/moss_lora/dave.md` |
| `lora.py` | `library.lines.adapters` bound to `layers` (held **outside** the blocks: the swapper moves every `.weight` under a block). `channel_scaling_alpha` reads `calibration/channel_stats.safetensors` and `dc_isolate` (on by default) reads `calibration/channel_mean.safetensors` (both from `bench/moss_channel_stats/probe_lora_inputs.py`); `docs/moss/channel_scaling.md`. `load_for_inference` also sets the transformer's `ref_timestep` / `ref_isolate` from the LoRA's metadata (`ref_settings`; `ref_isolate` stamped `"1"`, `"True"` on the first isolated LoRAs) |
| `train.py` | `run_train(req)`. Per step `choose_embeds` picks the caption: plain Anima caption (`plain_caption_rate`, layout-JSON caches), else a caption variant (`caption_variant_rate`), then the empty prompt (`--caption_dropout`, the shipped `asset/empty_prompt.safetensors`, one for every cache); one RNG each. Reference latents (`*.moss_ref` files) train with `ref_timestep` / `ref_isolate`, stamped in the LoRA. Relational REPA (`repa_weight`, parked). `--fit_report`: step 1 is the largest sample at its longest caption, no dropout, peak vs free VRAM. Above `PRELOAD_LIMIT_GB` samples stream from disk |
| `repa.py` | relational REPA (Anima's DoG-Gram term, `library.training.repa`) on one block's image tokens via `layer_hook`, massive-activation channels dropped and tokens centered; targets are the Anima PE-Spatial sidecars of un-resampled latents. Parked: no gain over channel scaling (2026-10-20), still wired (`repa_weight` 0 = off) |
| `tune.py` | `run_tune(req)`: one process, one arm per swap count at the largest budget that fits, timed on the largest sample → recommended `--blocks_to_swap` / `--activation_memory_budget` |
| `scan.py` | **torch-free** cache file layout (suffixes, `latent_files`, stem prefixing) + source/cache counts for the GUI; `cache.py` imports its constants. The GUI's stale count compares `source_sha256` to the `.txt`, `cache.py` the encoded caption: they can disagree (see `_te_current`) |

Sidecars: `scripts/moss/{cache,generate,train,tune_swap}.py`, thin argparse over the requests,
submitted through the daemon; `make gui-moss` (`gui/moss/`) submits the same jobs. User doc:
`docs/moss/README.md` (findings under `docs/moss/findings/`). Weights: catalog packs `moss-base` / `moss-sft`
(`library/downloads.py`) under `models/moss_image/MOSS-Image-{Base,SFT}`; SFT ships its
DiT alone, and TE / VAE / tokenizer always load from Base.

Research line, measured numbers, open questions: `project/moss_lora/`.

## What does NOT carry over

| Root `CLAUDE.md` says | Here |
|---|---|
| Latents are 5D `(B,C,T=1,H,W)`, singleton at **dim 2** | **4D packed `(B, 128, H/16, W/16)`**: FLUX.2's 32 ch at /8, 2×2-patchified into 128 ch. No frame axis. Normalisation is the VAE's BatchNorm buffers (`model.pack_latents` / `unpack_latents`) |
| TE outputs **must** be max-padded; never mask padding | **Never pad.** Each prompt is encoded alone at its true length (Qwen3.5's GDN layers misbehave on padding) and the DiT sees only valid tokens; `forward_batched` masks padded keys. RoPE puts the image block at frame position = text length, so padding shifts the image |
| Free-fit bucketing via `EDGE_TOKEN_BANDS`, edges 512…1536 | Native aspect, both sides **multiples of 16** |
| DiT depth probed from the checkpoint (28 vs 40) | Fixed **42** blocks, `num_layers` in `transformer/config.json` |
| Standard CFG | **CFG 3** and the **uncond branch skips block 22** (still 2 forwards/step); optional SLG third forward. Timestep input is `1000·σ`. Defaults from `pipeline_moss_image.py`, not from Anima habit |
| Anima's sampler shift | Euler, **shift 5 snapped to the nearest training-grid σ** (`model.snapped_sigmas`) |
| Adapters live in `networks/`, selected by `network_module` | `networks/` is bound to Anima module names; `lora.py` targets the six bias-free linears per block (`proj.{q,k,v,o,ffn_up_gate,ffn_down}`) |
| `make lora`, config chain `base → preset → method → CLI`, `masked_loss` | None of it. Flags defined once in `requests.py`; `--config configs/moss/lora.toml` (flat field-name TOML, `library.lines.requests`) layers field default → TOML → CLI |

## Standing rules

- **Read the pipeline before setting a sampler knob.** `pipeline_moss_image.py` (in the
  model folder) is the reference; `model.py` must stay op-for-op with it on the
  unbatched path.
- RoPE frequencies stay fp32 (`_RoPE` rebuilds them on every `.to()`).
- Decode with the DiT off the card: the fp32 VAE decode at ~2 MP does not fit beside it.
- GPU work goes through the daemon, as everywhere in this repo.
