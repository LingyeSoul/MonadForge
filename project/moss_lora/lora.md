# Phase 2 — light LoRA with activation checkpointing (2026-10-16)

Code: `library/moss/lora.py`, `library/moss/train.py`, `scripts/moss/train.py`, flags in
`TrainRequest` (`library/moss/requests.py`), tests `tests/test_moss_lora.py` (CPU, tiny
DiT). Smoke / eval scripts: `project/moss_lora/src/{fit_smoke,lora_ab}.py`.

## What the trainer does

- LoRA on the six bias-free linears per block, `layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`
  (42 × 6 = 252; rank 16 = 37.2 M params). The adapters are held outside `layers`, and
  each target's `forward` is patched (the qwen21 construction). Masters are fp32, rank
  GEMMs run in bf16, the file is saved in bf16 with `rank / alpha / targets / ss_num_blocks`
  metadata. `load_for_inference(transformer, path, multiplier)` hooks it on; multiplier 0
  is the base model exactly.
- Objective per report App. A.2: `z_σ = (1−σ)z0 + σε`, target `ε − z0`, timestep `1000σ`,
  plain MSE over image tokens. σ uses the final Stage-3 interval exactly: `S = 1 − sigmoid(Z)`,
  `Z ~ N(−1.09, 0.95²)`, truncated at 0.95 and rescaled onto the 1/1000 grid, with a 1 %
  uniform mix. This is noise-heavy (median σ ≈ 0.79). No resolution shift: that interval
  trained at area ≤ 1024², and the cache is ~1 MP.
- Checkpointing wraps each block's `forward`, not `__call__`, so the swap hooks never
  re-fire during recompute. Block swap is available (`--blocks_to_swap N|-1`) through
  `library.runtime.blockswap`, but the default is 0 and it was not needed.
- `--compile` (default **on**): `torch.compile(block.forward, dynamic=True)` after
  `apply_to`. One graph covers the varying joint length, with one recompile at the
  third distinct shape and none after that (open question 5, answered for training).
- `--caption_dropout p` trains a step on the empty-prompt embedding (`""` through the
  chat template, the pipeline's negative), encoded once into
  `<cache>/_empty.moss_te.safetensors`. The real run used 0.

## Fit (synthetic, `fit_smoke.py`, Base, rank 16, swap 0, checkpointing on)

| sizes | compile | peak alloc / reserved | steady s/step |
|---|---|---|---|
| 4473 img + 400 txt (largest), 4096 + 300 | off | 12.65 / 13.22 GiB | 3.1 |
| same + 4140 + 180 | on | 11.93 / 12.67 GiB | 2.0 (≈30 s compile + 1 recompile up front) |

The gate wants a peak below 15 GiB: **met with 2+ GiB to spare, no swap**.

## Real run — `channel_(caststation)` style set, Base

30 images, Anima-format tag captions (artist tag `@channel (caststation)`), cache
by fork C: image 4032–4200 tokens, text 127–406.
`--epochs 20 --rank 16 --lr 1e-4 --save_every_epochs 10` (AdamW, warmup 5 %, clip 1.0,
compile on). The user cut the plan from 40 epochs to 20.

- **600 steps in 21.0 min**, 2.1 s/step mean (63 s/epoch). Peak was 11.78 GiB allocated
  (11.50 at step 1, with the 4096 + 406 sample), 2.07 GiB actually free at step 1.
- The loss is dominated by σ: epoch means are 0.22–0.35 with no clean trend. Binned by σ
  over 150-step quarters, low σ (< 0.5) fell 0.265 → 0.199, the 0.5–0.85 bins stayed flat
  (≈ 0.17 / 0.22), and the > 0.85 bin moved 0.36 → 0.35. That is a weak signal, as usual for
  a style LoRA. The images decide.
- Output: `output/moss/lora/caststation_r16.safetensors` (= `_e20`), `_e10`, run report
  `caststation_r16.json` (per-step loss + σ).

## A/B (`lora_ab.py`, 30 steps, CFG 3, skip-22, 1024², seeds 1–2, same noise per pair)

Prompts: (0) a Miku tag bag in the training format, (1) Akita Neru, whom the set never
shows, (2) a natural-language sentence with the artist tag.
`output/moss/lora/ab_caststation_r16_30st/` (Base + SFT, ×1.0 / ×0.0) and
`ab_caststation_r16_e10_30st/` (Base, e10 ×1.0); grids are `grid_<variant>_p<i>_s<seed>.png`.

By eye:
- **Base ×1.0 vs ×0.0**: the shift points the right way and is consistent. Line art gets
  thinner and cleaner, fills flatter, the palette moves from saturated glossy cyan to the
  set's pale aqua, and backgrounds go to white. Faces get the set's round dark eyes and
  blunt bangs on prompt 2. Pose and composition per seed mostly survive (same noise),
  and character identity holds, including Neru, who is not in the set. The result is
  moderate, not overpowering: recognisably "flatter / cleaner, this palette", but I would
  not call it a sure artist ID from these 1024² headshots.
- **e10 vs e20**: e10 is a bit looser and more varied (darker clothing, one yellow
  background). e20 is cleaner and more uniform: frontal, centred upper bodies, white shirts.
  That is a mild drift toward a house composition. Neither is broken.
- **Base LoRA on SFT: it transfers.** SFT ×0.0 has its glossy gacha look (heavy
  saturation, big specular eyes, rendered hair). SFT ×1.0 loses most of it and lands
  close to Base ×1.0, while keeping a little more SFT background detail (the prompt-2
  school yard). Open question 3: no SFT-specific LoRA is needed for a style at this
  strength.
- The tag-bag captions trained and prompted fine, although the report says MOSS's captions
  are natural language (open question 1 still stands for character sets and long
  captions).

## Open

- Recognisability is a by-eye "moderate"; a stronger setting (lr 2e-4 or more steps)
  and a side-by-side with training crops at matched prompts would settle it.
- Swap + checkpoint + compile together is untested. Swap + budget + compile without
  checkpointing is the default now (§ Swap + budget).
- The trainer keeps `MOSSImageTransformer.gradient_checkpointing` off and checkpoints
  `block.forward` itself (hooks once, compile-compatible). The flag in `model.py` is
  unused by training.

## Swap + budget instead of checkpointing (2026-10-17)

Question: does swapping blocks to CPU free enough VRAM to turn checkpointing off, and is
that faster? Turning checkpointing off saves up to a third of a step (frozen base
weights: backward computes dX only, so forward ≈ backward, and ckpt adds one forward).

Saved-for-backward per block (one real-width block + LoRA r16 on CPU, saved-tensor
hooks, linear in L from 512 to 1024 tokens, no L² term): **235 KiB/token eager,
128 KiB/token compiled** (budget 1.0). × 42 blocks at 4.4k joint tokens that is
**22.6 GiB compiled**, against ~12 GiB free even at swap 40, so swap alone never fits
1024². `--activation_memory_budget` (`torch._functorch.config`, set before
`compile_blocks`) caps the rest.

`fit_smoke.py --sizes 1024x1024:300 --steps 12`, Base, rank 16, compile on, synthetic;
s/step excludes step 1; free = `mem_get_info` at step 1:

| arm | ckpt | swap | budget | s/step | vs A | peak alloc / reserved | free |
|---|---|---|---|---|---|---|---|
| A | on | 0 | — | 2.09 | — | 11.52 / 12.48 | 1.60 |
| B | off | 40 | 0.5 | 2.29 | −10 % | 12.72 / 13.94 | — |
| C | off | 40 | 0.35 | 2.43 | −16 % | 9.53 / 10.97 | 2.95 |
| E | off | 12 | 0.15 | 1.77 | +15 % | 12.15 / 13.35 | 0.73 |
| **D** | off | 24 | 0.2 | **1.78** | **+15 %** | 10.63 / 11.75 | **2.26** |
| G | off | 24 | 0.25 | 1.73 | +17 % | 11.66 / 13.09 | 1.06 |
| F | off | 0 | 0.1 | OOM | | | |

- **PCIe is the limit, not recompute.** 40 swapped blocks (~17 GB across the bus per
  step) do not hide behind compute, even with the budget raised to 0.5. Lower budgets at
  moderate swap win.
- **D is the default** (`--no-grad_checkpointing` default, `--blocks_to_swap 24`,
  `--activation_memory_budget 0.2`). G is 0.05 s faster with only 1.06 GiB free, and real
  sets run up to ~4.9k joint tokens, so D keeps the margin.
- Step memory, measured as free VRAM before training minus free at step 1: 6.51 / 7.38 /
  8.78 GiB at budget 0.15 / 0.2 / 0.25 (4.4k tokens). `--blocks_to_swap -1` sizes from
  `3.5 + budget · 6.0 MB · joint tokens`, ~1 GiB above those measurements.
- Not yet checked: a real-cache run with varying lengths (dynamic compile under the
  budget), and loss parity against checkpointing. Recompute is numerically the same
  graph, but the synthetic |g| spread is too wide to show that.
- The optimum swap depends on the machine (PCIe generation and lanes, GPU speed), so a
  16 GB card on a narrower link may want less swap and a lower budget.

## Swap tuner (`scripts/moss/tune_swap.py`, 2026-10-17)

`library/moss/tune.py`: the DiT loads once. Each arm places it with N blocks swapped,
takes the largest budget on a 0.05 grid that `step_reserve_gb` fits into the free VRAM
(`train.budget_for_free`), resets dynamo, recompiles, and times 5 steps after 2 warm-up
steps on the largest sample. An OOM retries one budget step lower. The sweep stops 2 arms
past the fastest. The recommendation is the least swap within 2 % of the fastest arm
that leaves ≥ 1 GiB free. Report: JSON with GPU, PCIe link and every arm.

First run, synthetic 1024²:300 on the 5070 Ti, 4.1 min
(`output/moss/tune/tune_swap_synth1024.json`):

| arm | budget | s/step | peak reserved | free |
|---|---|---|---|---|
| ckpt, swap 0 | — | 2.104 | 12.29 | 2.00 |
| swap 8 | 0.1 | 1.821 | 13.28 | 1.00 |
| swap 12 | 0.1 | 1.844 | 12.43 | 1.85 |
| swap 16 | 0.15 | 1.803 | 12.61 | 1.70 |
| swap 20 | 0.2 | 1.794 | 12.70 | 1.50 |
| **swap 24** | 0.2 | **1.770** | 11.86 | 2.47 |
| swap 28 | 0.25 | 1.756 | 12.42 | 1.90 |
| swap 32 | 0.3 | 1.750 | 11.62 | 2.70 |
| swap 36 | 0.3 | 1.772 | 10.78 | 3.56 |
| swap 40 | 0.35 | 2.341 | 11.01 | 3.33 |

- **The curve is flat from swap 8 to 36 (1.75–1.84) and falls off a cliff at 40.** Budget
  0.1–0.3 barely moves the step, so the speed-up is almost all from dropping the
  checkpoint recompute. Below the knee, swap count is a VRAM choice, not a speed choice.
- Strict fastest was swap 32 (1.750), within timing noise of 24 (1.770). With the 2 %
  tie rule the tuner recommends **swap 24 / budget 0.2**, the shipped default.
- The budget predictor is conservative: arms finished with 1.0–3.6 GiB free.
