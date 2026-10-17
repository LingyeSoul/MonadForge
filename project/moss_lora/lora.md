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

## Layout-JSON captions vs plain tag captions (2026-10-17)

Same 30 images, latents (symlinked), schedule and seed as the real run above; only the
TE input differs: `--caption_format layout_json` (`anime_tools.captions.layout_json`,
0.7.13), text 141–455 tokens vs 127–406. Trained with `--blocks_to_swap 12
--activation_memory_budget 0.1` (no checkpointing): 600 steps in 18.5 min, 1.85 s/step,
peak 11.51 GiB. Output `output/moss/lora/caststation_layout_r16.safetensors`; A/B
`ab_caststation_layout_r16_30st/` (Base, the same three prompts converted with
`--prompt_format layout_json`, 30-step Euler, seeds 1–2) against `ab_caststation_r16_30st/`.

By eye, 3 prompts × 2 seeds:
- **Style transfer is about the same strength**: thin clean line art, flat fills, pale
  aqua palette, white backgrounds, blunt bangs on the NL prompt.
- **Less attribute leakage from the set.** On Neru (not in the set), the tag-caption LoRA
  turns her yellow necktie green/teal in both seeds (the set's Miku `aqua necktie`); the
  JSON LoRA keeps it yellow in both. On Miku seed 2 the tag LoRA drifts to a white shirt,
  the JSON LoRA keeps her grey outfit.
- Base without a LoRA renders the JSON prompts as well as the tag prompts.
- The NL prompt (p2) passes through the converter as one `summary.caption` string and
  behaves like its plain-text twin.

ER-SDE + Spectrum (the new `lora_ab` default: 50 steps, flex_window 0.5) on the JSON LoRA,
`ab_caststation_layout_r16_50st/`: the job took 6.1 min against 9.4 min for 30-step
Euler. Compositions differ from Euler (another sampler, injected noise), so compare
base vs LoRA within one sampler. The LoRA's shift holds: paler palette, thinner lines;
Neru keeps a yellow (yellow/black striped) necktie in both seeds. No Spectrum artifacts
at grid scale. One oddity: Neru seed 1 grew a flat yellow panel behind her.

## weight_svd vs kaiming down init (2026-10-17)

`--down_init weight_svd` (`LoRAAdapter.init_down_weight_svd`, Anima's Gram-routed
`_top_right_singular_vectors`, `V_rᵀ/√3`). The Kaiming draw still runs first, so a
weight_svd run at seed 0 sees the kaiming run's σ / noise / order step for step (σ 0.710 /
0.679 at steps 10 / 20 in both; loss 0.1803 vs 0.1804 at step 10).

Arms on the layout-JSON caststation cache, 20 epochs, rank 16, lr 1e-4, swap 12:
kaiming seed 0 (the earlier `caststation_layout_r16`, budget 0.1), weight_svd seed 0
(`caststation_layout_r16_svd`, budget 0.1, 18.2 min), kaiming seed 1 as the seed-twin
floor (`caststation_layout_r16_s1`, budget 0.2). A/B: `lora_ab.py --variants base
--seeds 1,2,3,4`, ER-SDE + Spectrum 50 steps, CFG 3, the same noise across arms;
`output/moss/lora/ab_initcmp/<run>/`.

Mean |Δ| at 256 px over 3 prompts × 4 seeds:

| pair | mean \|Δ\| |
|---|---|
| kaiming s0 vs weight_svd s0 | 0.121 |
| kaiming s0 vs kaiming s1 | 0.121 |
| weight_svd s0 vs kaiming s1 | 0.155 |
| LoRA vs base: kaiming s0 / s1 / weight_svd | 0.145 / 0.148 / 0.166 |

- The init picks a different sample, as far from kaiming as a seed change. All three
  shift the same way (thin clean lines, flat pale fills, white backgrounds); Neru keeps a
  yellow tie in all three.
- weight_svd sits ~13 % further from base by pixel L1, a coarse, composition-dominated
  measure on 12 images. The user's read: weight_svd keeps the caststation style better.
  **`down_init` defaults to `weight_svd`** on that read.

## Compiled A/B, alpha 90 at lr 5e-5 (2026-10-17)

`lora_ab.py` compiles the blocks after the LoRA is attached by default now
(`--no-compile` for eager). Compiled vs eager on the same LoRA: mean |Δ| 0.03 at ×1.0,
0.037 at ×0.0, a quarter of the seed-twin floor, but not zero, so compiled grids compare
only with compiled grids. The floor arms were regenerated compiled under
`output/moss/lora/ab_compiled/<run>/` (`ab_initcmp/` stays eager). One A/B job (3 prompts × 4
seeds × 2 arms = 24 denoises) is 8.8 min compiled against 11.8 min eager: 20.6 s against
28.5 s per denoise (1.38×), no recompiles after the first cell of each arm, and the
first latent lands 34 s after start in either mode.

Arm: `caststation_layout_r16a90_svd`, rank 16 / alpha 90 (scale 5.6), lr 5e-5,
weight_svd, everything else as the weight_svd run (swap 12, budget 0.1, seed 0,
19.7 min). Under Adam, scale s at lr η equals scale 1 with up at s·η and down at η, so
this arm is up 2.8e-4 / down 5e-5 against the α16 arm's 1e-4 / 1e-4.

Mean |Δ| at 256 px (bilinear, which reproduces the eager table above to the
third decimal), 3 prompts × 4 seeds, compiled:

| pair | mean \|Δ\| |
|---|---|
| LoRA vs base: svd α16 / svd α90 / kaiming s0 / kaiming s1 | 0.171 / 0.162 / 0.152 / 0.156 |
| svd α16 vs svd α90 | 0.126 |
| kaiming s0 vs kaiming s1 (seed-twin floor) | 0.123 |
| svd α90 vs kaiming s0 | 0.104 |

- α90 moves less than α16 and sits a seed change away from it. By eye it shifts the same
  way (thin lines, flat pale fills) but softer: Miku keeps more of base's shading, Neru
  gets two pale yellow panels and one sticker outline (tie stays yellow / black), the NL
  prompt keeps darker uniforms and fuller backgrounds. svd α16 at lr 1e-4 stays the
  flattest and closest to the set. The default (alpha = rank) stands.

## LyCORIS LoKr / LoHa (2026-10-17)

`--algo lokr|loha` (`library/moss/lora.py` `LoKrAdapter` / `LoHaAdapter`), init and ΔW
math from `lycoris-lora` 4.0.0 (`lycoris.functional`, `backend="torch"`, traced fullgraph
under the block compile). LoKr runs as a bypass (grouped GEMMs on the tokens); LoHa
rebuilds ΔW and runs one GEMM against `W + ΔW` (the delta rounds to W's bf16 grid in the
forward, as LyCORIS's own LoHa). Upstream bug worked around: `lokr.weight_gen` with its
default `tucker=True` inits a `None` t2 for a Linear.

Fit, `fit_smoke.py --sizes 1024x1024:300 --steps 12`, both 74.3 M params (LoRA r16 is
37.2 M): LoKr `--rank 100000 --factor 8` (full w2, scale 1), LoHa `--rank 16`.

| arm | swap | budget | s/step (mean of steps 2–12) | peak alloc / reserved | free |
|---|---|---|---|---|---|
| LoKr f8 | 12 | 0.15 | OOM at step 1 (13.8 GB allocated) | | |
| LoHa r16 | 12 | 0.15 | OOM at step 1 | | |
| LoKr f8 | 24 | 0.1 | 2.26 | 10.48 / 11.08 | 3.63 |
| LoHa r16 | 24 | 0.1 | 2.49 | 11.54 / 13.04 | 1.67 |

LoRA at swap 24 / 0.2 is 1.77. The LoKr OOM allocation is T × 16384 bf16: the bypass
saves a full-width intermediate on `ffn_up_gate`, where a LoRA saves T × r.
`step_reserve_gb` is LoRA-calibrated, so `--blocks_to_swap -1` under-reserves for both.
LoHa's |g| runs 11–66 against LoKr's 1–31 on these 12 steps (`hada_w2` starts at N(0, 0.1) × N(0, 1));
clip 1.0 bounds the step.

## Round 2: LoKr / LoHa, alpha 90 at lr 1e-4, Anima-style σ (2026-10-17)

All on the layout-JSON caststation cache, 20 epochs, seed 0, compiled A/B
(`ab_compiled/<run>/`, base, seeds 1–4). Arms:

| run | adapter | lr | σ sampling | swap / budget | train |
|---|---|---|---|---|---|
| `caststation_layout_r16_svd` (floor) | LoRA r16 α16 weight_svd | 1e-4 | report final interval | 12 / 0.1 | 18.2 min |
| `caststation_layout_r16a90_svd` | LoRA r16 α90 weight_svd | 5e-5 | same | 12 / 0.1 | 19.7 min |
| `caststation_layout_r16a90_lr1e4_svd` | LoRA r16 α90 weight_svd | 1e-4 | same | 12 / 0.1 | 19.2 min |
| `caststation_layout_r16_svd_sig0` | LoRA r16 α16 weight_svd | 1e-4 | sigmoid(N(0,1)), no cutoff / uniform (Anima base.toml) | 12 / 0.1 | 18.6 min |
| `caststation_layout_lokr_f8` | LoKr factor 8, full w2 (scale 1), 74 M | 1e-4 | report final interval | 24 / 0.1 | 22.7 min, 2.23 s/step |
| `caststation_layout_loha_r16` | LoHa r16 α16, 74 M | 1e-4 | same | 24 / 0.1 | 23.5 min, 2.30 s/step |

`--sigma_mu 0 --sigma_s 1 --sigma_cutoff 1.0 --sigma_uniform 0` is Anima's default
sampler in MOSS's sign convention (σ = 1 − sigmoid(Z) with μ 0 equals sigmoid(Z)):
median σ 0.50 against the final interval's ~0.79. A shift s is the same as μ − ln s in
this sampler (logit σ = −Z), so the final interval (μ −1.09, shift 1) equals μ 0 with
shift ≈ 3; there is no separate shift to turn off.

Mean |Δ| at 256 px (bilinear), 3 prompts × 4 seeds:

| vs base | mean \|Δ\| |
|---|---|
| floor svd α16 | 0.171 |
| α90 lr 5e-5 / lr 1e-4 | 0.162 / 0.169 |
| σ sigmoid(N(0,1)) | **0.124** |
| LoKr f8 | 0.172 |
| LoHa r16 | **0.191** |
| kaiming s0 / s1 | 0.152 / 0.156 |

| vs floor (svd α16) | mean \|Δ\| |
|---|---|
| α90 lr 1e-4 | 0.109 |
| LoKr f8 | 0.116 |
| kaiming s0 | 0.119 |
| α90 lr 5e-5 | 0.126 |
| σ sigmoid(N(0,1)) | 0.127 |
| LoHa r16 | 0.149 |
| (seed-twin floor, kaiming s0 vs s1) | 0.123 |

- The low-σ arm moves the image least (0.124 from base): with half the steps below
  σ 0.5, composition and palette stay closer to base.
- LoHa moves it most (0.191) and sits furthest from every other arm (0.147–0.181).
- α90 at lr 1e-4 and LoKr f8 land within a seed change of the floor.

Eyeball (a general-purpose Opus agent, columns lettered and shuffled, judged per prompt ×
seed against 8–12 of the training images; the column → folder map it opened carried
descriptive folder names, so it was not fully blind). Its ranking for "looks like the
artist without damage":

1. **floor svd α16 lr 1e-4**: the most artist cues (red eyeliner stroke, sweat drops,
   flat pale fills, airy white backgrounds), no leakage seen. Medium-high confidence.
2. **LoKr f8**: as flat and clean as the floor, a little less character in the faces,
   no damage. Medium.
3. **α90 lr 1e-4**: style indistinguishable from LoKr at 4 seeds, but p2 s1 carries a
   **"caststanel" watermark plus a handle** (checked at full size) and p2 s4 crops the
   frame. Low confidence on LoKr vs α90-lr1e-4.
4. **α90 lr 5e-5**: partial transfer, base gloss on Miku, a signature scribble (p1 s2),
   an aqua collar stripe on Neru (p1 s4), a tangled peace-sign hand (p0 s3).
5. **σ sigmoid(N(0,1))**: weakest transfer, nearly base on every prompt (saturated,
   glossy, orange blonde), undamaged. Reads as under-trained.
6. **LoHa r16**: strong but wrong: a grey-teal desaturation that is not the artist's
   palette, Neru's tie replaced by the set's red striped bow + black vest in 4/4 seeds,
   amber eyes, Miku's tie turned into a bow in 2/4. Over-fit with costume leakage.

- Defaults stand: LoRA r16, alpha = rank, lr 1e-4, the report's final σ interval.
- The low-σ sampler lost: a style LoRA on MOSS wants the noise-heavy pretraining σ, the
  opposite of the guess that style lives at low σ.
- A higher effective lr (α90 at 1e-4, up 5.6e-4) starts to copy the set's signature.
- LoKr f8 is a usable alternative to the LoRA at 2× the params and +26 % s/step. LoHa at
  lr 1e-4 overshoots; a lower lr would be its next arm.

## Round 3: rank 32 / alpha 128 at lr 1e-4 vs 2e-5, complex prompts (2026-10-17)

Same cache and recipe as round 2 (layout-JSON caststation cache, weight_svd, 20 epochs,
seed 0, swap 12 / budget 0.1), rank 32 / alpha 128 (scale 4, 74 M params):

| run | lr | effective up / down | train | s/step | free at fit |
|---|---|---|---|---|---|
| `caststation_layout_r32a128_svd` | 1e-4 | 4e-4 / 1e-4 | 18.5 min | 1.85 | 1.71 GB |
| `caststation_layout_r32a128_lr2e5_svd` | 2e-5 | 8e-5 / 2e-5 | 19.3 min | 1.86 | 1.71 GB |

r16 at the same swap / budget: 1.82 s/step, 1.98 GB free.

Test: the two Space gallery prompts (`project/moss_lora/space/{004,017}`, each at its own
size / seed), with the artist injected as a top-level `"artists":["channel (caststation)"]`
and a `gen_brief` prefix (`output/moss/test/space_prompts_caststation/`), er_sde +
Spectrum + compile, LoRA ×1.0 against base (`output/moss/test/space_cs_*`). One seed per
prompt; read by eye.

- **lr 1e-4**: the strongest move toward the set: 004's courtyard washes out to the
  set's airy white, fills go flat and pale, sweat drops appear; all ten characters and the
  arch composition hold. Prompt details slip: Koharu holds the schoolbag Azusa should
  hold, the Peroro plush is gone, the slapstick reads calmer. 017 holds every prop.
- **lr 2e-5**: between base and 1e-4. Background (stairs, building) kept and lightened,
  cleaner line work, fills still fuller than the set's. Prompt details hold: Azusa holds
  the open bag under the plush, Peroro is there.
- The user judged 2e-5 much better. **Defaults moved to rank 32, alpha 128 (lora; lokr /
  loha keep alpha = rank), lr 2e-5, budget 0.1.** Not yet run against the round-2 tag
  prompts; the 1e-4 run's `_e10` and an lr 5e-5 arm are the open comparisons.
