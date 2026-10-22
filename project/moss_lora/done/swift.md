# SWIFT target extrapolation on a channel LoRA

Status: settled 2026-10-22 — two arms (α 0.4, α 0.8) with an epoch-5 guide, both
indistinguishable from the floor. Paper: arXiv 2610.25844 (SWIFT). Code:
`src/swift_train.py`.

## Idea

Train on `y = v + α (v − v_weak)` instead of `v`, with `v_weak` a frozen early
checkpoint of the same recipe (one extra no-grad forward a step); the guide is dropped
after training. The paper trains from scratch on ImageNet / CIFAR (default α 0.4,
U-shaped in α); it has no fine-tuning, LoRA or text-to-image run.

## First arm

- caststation (`channel_(caststation)_layout`, 30 images), the shipped recipe: Base,
  r32 / α128, `weight_svd`, lr 5e-5, channel scaling 0.5, `dc_isolate`, 20 epochs = 600
  steps, seed 0, `--plain_caption_rate 0`.
- Guide: the step-150 (epoch 5) state of a rerun of the floor
  `caststation_layout_at_r32a128_lr5e5` with `--save_every_epochs 5` (a standalone
  150-step run would end on a different LR schedule).
- SWIFT run: the same recipe from scratch, α 0.4, that guide frozen. One randn per step,
  so σ / noise / caption order match the α 0 run.

Result: loss against `v` within 0.003 of the paired rerun every epoch; renders on the
two Space prompts (`space/`) inside run-to-run noise.

## Second arm: α 0.8

- Floor `caststation_lay08p02_at`: the config defaults (`plain_caption_rate` 0.2),
  seed 0, 20 epochs, `--save_every_epochs 5`. Guide: its `_e5` checkpoint.
- SWIFT `caststation_lay08p02_at_swift_a08`: same recipe and seed, α 0.8. 82 s / epoch
  vs 57 s (guide forward), peak 12.19 vs 12.06 GB.
- Loss against `v` (`.swift.json` `loss_v`, per-epoch mean): 0.002–0.008 above the
  floor every epoch, widest at e15–20 (+0.005–0.007); final 0.2343 vs 0.2279. The
  SWIFT loss itself sits at ≈ (1 + α)² × `loss_v` (0.74–1.11), and `gap`
  (‖v − v_weak‖²) ≈ `loss_v` all run.
- Renders (Base + SFT, the three `lora_ab.py` prompts × 2 seeds, ER-SDE 50 steps;
  sheets `output/tests/ab_caststation_lay08p02/cols/`): no breakage; SWIFT draws
  slightly heavier, darker line work and greyer flat fills. Seed 2 cells keep the
  floor's composition, seed 1 cells diverge. One prompt miss: Base Miku seed 1 points
  instead of the `peace sign`. No consistent win either way.

## Reading

With the student still ≈ the guide, `f − y = (f − v) − α (v − v_weak) ≈ (1 + α)(f − v)`:
the SWIFT gradient is the plain one scaled by `1 + α`, which Adam normalises away. The
extrapolation only acts through `f − v_weak`, which stays small when the guide is
step 150 of the same 600-step recipe. Hence α 0.4 and α 0.8 both land in run-to-run
noise; raising α further would not change that. (Interpretation, not measured
directly.)

## Open

A weaker guide (an earlier checkpoint, e.g. e2, or a lower-rank guide) is the arm that
could move the result; not queued.

## Risk to watch

On 30 images the posterior mean at low σ is close to the nearest training image, so
`v − v_weak` is largely per-image detail; extrapolating it may speed up memorisation
rather than style. Check for copied compositions, not only style strength.

## Cost

One extra no-grad DiT forward a step with the guide LoRA swapped in (the paper's
quarter-size guide costs 9 %; a same-size guide under block swap costs more).
