# CLAUDE.md — `library/moss/`

**This is not Anima.** MOSS-Image (OpenMOSS, Apache-2.0) is a 5B single-stream DiT on a
frozen Qwen3.5-9B text encoder and the frozen FLUX.2 VAE. The root `CLAUDE.md`
invariants describe the Anima DiT; most of them are wrong here. Shared with Anima:
`library.runtime.offloading` (+ `library.runtime.blockswap`, also used by `library/qwen21/`), `library.runtime.dynamo`, `library.env`, the daemon, and
the pure Spectrum core (`networks/spectrum_forecast.py`, `networks/spectrum_sea.py`), and
the ER-SDE sampler + CNS core (`library.inference.sampling`, `library.inference.corrections.cns_core`).
Shared with `library/qwen21/`: `library.lines` (`requests.py` request base, `adapters.py`
LoRA / LoKr / LoHa network) and `library.runtime.blockswap.place`.
Nothing Anima-side imports `library.moss`.

| Module | Role |
|---|---|
| `model.py` | vendored `MOSSImageTransformer` / `MOSSImageScheduler` (Apache-2.0, attribution kept; state-dict names unchanged). Reference per-sample `forward`, `forward_batched` (masked text, per-sample `skip_layers`), `head` for Spectrum, `pack_latents` / `unpack_latents` / `encode` / `decode` |
| `requests.py` | **torch-free** `CacheRequest` / `GenerateRequest` / `TrainRequest` / `TuneSwapRequest`, `to_argv()` / `from_argv()`, model-dir resolution |
| `loader.py` | phase split: TE (whole decoder layers paged onto the card) → encode → free → DiT; VAE |
| `cache.py` | TE cache (natural length, never padded) + latent cache (packed, BN-normalised) |
| `generate.py` | `run_generate(req)`: Euler or ER-SDE (`sampler`, Anima's `ERSDESampler`), CFG + uncond skip-22, Spectrum hook-in |
| `cns.py` | CNS noise recolor for ER-SDE on the un-patchified 32 ch grid + γ estimator; bench-only so far (`project/moss_lora/sampler.md`) |
| `spectrum.py` | per-branch feature forecaster (cond and uncond forecast separately) |
| `lora.py` | `library.lines.adapters` bound to `layers` (held **outside** the blocks: the swapper moves every `.weight` under a block) |
| `train.py` | `run_train(req)` |
| `tune.py` | `run_tune(req)`: one process, one arm per swap count at the largest budget that fits, timed on the largest sample → recommended `--blocks_to_swap` / `--activation_memory_budget` |
| `scan.py` | **torch-free** cache file layout (suffixes, stem prefixing) + source/cache counts for the GUI; `cache.py` imports its constants |

Sidecars: `scripts/moss/{cache,generate,train,tune_swap}.py`, thin argparse over the requests,
submitted through the daemon; `make gui-moss` (`gui/moss/`) submits the same jobs. User doc:
`docs/experimental/moss_lora.md`. Weights: catalog packs `moss-base` / `moss-sft`
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
| `make lora`, config chain `base → preset → method → CLI`, `masked_loss` | None of it. Flags only, defined once in `requests.py` |

## Standing rules

- **Read the pipeline before setting a sampler knob.** `pipeline_moss_image.py` (in the
  model folder) is the reference; `model.py` must stay op-for-op with it on the
  unbatched path.
- RoPE frequencies stay fp32 (`_RoPE` rebuilds them on every `.to()`).
- Decode with the DiT off the card: the fp32 VAE decode at ~2 MP does not fit beside it.
- GPU work goes through the daemon, as everywhere in this repo.
