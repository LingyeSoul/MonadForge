# examples/

Runnable scripts showing the Anima programmatic API for library embedders —
the Python you write when you `import anima_lora` into your own code instead of
going through `make` targets. Each script is self-contained and runs from the
repo root (`anima_lora/`).

After `uv sync` (which installs this repo editable), the front-door package is
importable from anywhere — the curated entry points live on `anima_lora`:

```python
import anima_lora
settings = anima_lora.get_generation_settings(args)
latent = anima_lora.generate(args, settings)
image = anima_lora.decode_to_pil(vae, latent, device)
```

`anima_lora` is a thin lazy re-export of `library.inference` /
`library.config.io` / `library.anima.weights` / `library.models.qwen_vae` (see
`anima_lora/__init__.py` for the full map). Repo-relative model/config paths
resolve against the repo home, not the CWD — so `import anima_lora` works from
any directory; set `ANIMA_HOME` to point at a relocated checkout. The high-level flows
(`01`–`04`) import the curated entry points from `anima_lora`; the building-block
scripts (`05`/`06`) reach into the `library.*` homes directly, since their point
is to show the raw primitives. Either way each script keeps a `sys.path` shim so
`python examples/<script>.py` runs straight from the repo without an install.

**High-level flows** — the supported entry points:

| Script | Shows | Needs |
|---|---|---|
| [`01_generate.py`](01_generate.py) | Text-to-image: `get_generation_settings` → `generate` → `save_output` | DiT + VAE + text encoder |
| [`02_generate_with_lora.py`](02_generate_with_lora.py) | Same, with one or more LoRA adapters attached at DiT load | + adapter `.safetensors` |
| [`03_config_and_network.py`](03_config_and_network.py) | `load_method_preset` merge chain + `create_network` (three-axis routing) | config part: nothing; `--build-network`: DiT |
| [`04_train_lora.py`](04_train_lora.py) | In-process training via `AnimaTrainer().train(args)` | preprocessed dataset cache |

**Building blocks** — the raw primitives for writing your own `scripts/` tool:

| Script | Shows | Needs |
|---|---|---|
| [`05_load_models.py`](05_load_models.py) | Load DiT / VAE / text encoder directly; encode a prompt to the DiT-ready cross-attn embedding | DiT + VAE + text encoder |
| [`06_vae_and_dataset.py`](06_vae_and_dataset.py) | VAE pixel↔latent round-trip; iterate the on-disk training cache (`CachedDataset`) | VAE (+ cache for part B) |
| [`07_frozen_dit_training_build.py`](07_frozen_dit_training_build.py) | Frozen DiT + fresh adapter build for *training* via the `harness` helpers (`place_dit_for_training` / `compile_dit_blocks` / `enable_training_grad_ckpt`) — the `scripts/distill_*` model-build sequence | DiT |

## Setup

```bash
uv sync
hf auth login
make download-models      # DiT, text encoder, VAE, …
# 04 also needs the training cache:
make preprocess
```

Model paths default to the `configs/base.toml` locations. Override per-run with
`ANIMA_DIT` / `ANIMA_VAE` / `ANIMA_TEXT_ENCODER` env vars.

## Quick start

```bash
python examples/01_generate.py --prompt "a red fox in a snowy forest"
python examples/02_generate_with_lora.py --lora_weight output/ckpt/my_lora.safetensors --prompt "…"
python examples/03_config_and_network.py --method lora --preset default
python examples/04_train_lora.py --max_train_epochs 8
python examples/05_load_models.py --prompt "a lighthouse at dusk"
python examples/06_vae_and_dataset.py                       # iterate the cache
python examples/06_vae_and_dataset.py --image some/photo.png  # VAE round-trip
python examples/07_frozen_dit_training_build.py             # build a trainable adapter
```

## Notes for embedders

- **`anima_lora` is the stable API; `library.*` / `networks.* `/ `scripts.*` are internal.**
  The curated `anima_lora` façade is the surface we keep stable across releases.
  The underlying trees are installed and importable for advanced use (`05`/`06`
  reach into `library.*` on purpose), but they may move or change signature
  without a deprecation cycle — pin a tag (`ANIMA_VERSION`) if you depend on them.
- **Inference is request-driven.** `01`/`02` build a typed
  `anima_lora.GenerationRequest` and call `.to_args()` — which feeds the request
  through `inference.parse_args` under the hood, so every optional knob the
  generation code reads via `getattr()` still gets a value. The long tail of
  method knobs (spectrum/dcw/ip-adapter) rides through the request's `extra_argv`,
  or you can build the `argparse.Namespace` straight from `inference.parse_args(argv)`.
- **Adapter family is in the checkpoint, not the call.** `02` passes any LoRA /
  OrthoLoRA / T-LoRA / Hydra / FeRA `.safetensors`; the DiT loader reads the
  metadata and merges-or-keeps-live accordingly.
- **Prompt encoding uses two process-global strategy singletons.** `generate()` /
  `prepare_text_inputs()` lazily install them from `args.text_encoder` (via
  `anima_lora.ensure_text_strategies`), so the high-level flows just work; `05`
  shows the explicit one-liner. Encoding also needs the DiT — the encoder hidden
  states are projected by `Anima._preprocess_text_embeds`.
- **Multi-GPU training** must go through `accelerate launch train.py …`
  (`make lora`). `04` is the single-process equivalent.
- The text-encoder padding and constant-token bucketing invariants in
  `../CLAUDE.md` apply — they're handled inside the called functions, but worth
  reading before you deviate from these flows.
