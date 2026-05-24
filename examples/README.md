# examples/

Runnable scripts showing the Anima programmatic API for library embedders —
the Python you write when you import `anima_lora` into your own code instead of
going through `make` targets. Each script is self-contained and runs from the
repo root (`anima_lora/`).

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
```

## Notes for embedders

- **Inference is args-driven.** The scripts build an `argparse.Namespace` via
  `inference.parse_args(argv)` rather than hand-rolling one — that guarantees
  every optional knob the generation code reads via `getattr()` has a value.
- **Adapter family is in the checkpoint, not the call.** `02` passes any LoRA /
  OrthoLoRA / T-LoRA / Hydra / FeRA `.safetensors`; the DiT loader reads the
  metadata and merges-or-keeps-live accordingly.
- **Multi-GPU training** must go through `accelerate launch train.py …`
  (`make lora`). `04` is the single-process equivalent.
- The text-encoder padding and constant-token bucketing invariants in
  `../CLAUDE.md` apply — they're handled inside the called functions, but worth
  reading before you deviate from these flows.
