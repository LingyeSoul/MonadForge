# Minimal real Anima adapter training

This CPU-only smoke benchmark creates a random, 32-channel Anima DiT and
synthetic latents/text. It executes the real adapter factory, DiT forward,
backward, AdamW update, production safetensors save/load factory, and an output
comparison against the same frozen base. No pretrained weights, GPU, real
dataset, training daemon, or user configuration writes are involved.

Run all built-in GUI variants with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python bench/anima_minimal_training/run_bench.py
```

To reproduce the focused regressions and direct artifacts outside the repo:

```bash
uv run python bench/anima_minimal_training/run_bench.py vera controlnet hydralora-8gb --output-root /tmp/anima-minimal-training
uv run pytest -o addopts= -q tests/test_anima_minimal_training.py tests/test_vera_module.py
```

Defaults are batch 1, rank 1 (rank 2 for timestep masks/DyLoRA/Chimera),
one transformer block, 4x4 VAE latents and 3 optimizer steps. EasyControl uses
two blocks so its condition stream actually propagates. Hydra uses 6 steps
to visit each of its 3 sigma buckets twice with 2 experts per bucket.
Chimera keeps 2 experts in each pool. Soft Tokens keeps both banks and 2 real
negative forwards per step. ConvRot keeps its real quantized path with group
size 64. External calibration/REPA teachers, compilation, offload and sampling
are disabled. Built-in variants still use the normal read-only custom overlay
merge chain; this is a numerical smoke test, not a complete training CLI run.

Each run writes the standard `result.json` envelope, `cases.json`, adapter
safetensors and exact optimizer/training state under
`bench/anima_minimal_training/results/<timestamp>/`, or `--output-root`.
Loss, gradient norms, changed parameters, module classes, and checkpoint output
error are recorded. A passing case needs finite nonzero gradients, an actual
parameter update, and checkpoint output agreement within `atol=rtol=1e-4`.

Observed before/after on the 2026-10-18 audit:

| Case | Before | After |
| --- | --- | --- |
| VeRA | `torch.mm` rejects real 3D/5D DiT activations | Training and reload pass with the same 2D matrix operation generalized over leading dimensions |
| ControlNet | Only output bias learns; changed conditioning produces exactly zero output difference | Input/injection weights learn; changed conditioning produces max absolute output difference 0.0305100 after 3 steps |
| Hydra 8GB | Unsupported OrthoInit + shared-A combination raises during construction | Same supported Hydra architecture as the normal variant; 6 sigma-bucket training steps and reload pass |

ControlNet's fresh initialization changes intentionally: its input projection
is now nonzero, while zero per-block outputs retain exact initial base-model
equivalence. Loading an existing checkpoint preserves its tensors unchanged;
an old checkpoint with a dead all-zero condition path is not silently migrated.
These tiny synthetic runs establish execution and learning-path liveness, not
full-model convergence, memory requirements, or image quality.
