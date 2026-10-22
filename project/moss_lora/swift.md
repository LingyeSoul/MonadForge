# SWIFT target extrapolation on a channel LoRA

Status: first arm run 2026-10-22, inside run-to-run noise. Paper: arXiv 2610.25844
(SWIFT). Code: `src/swift_train.py`.

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

## Risk to watch

On 30 images the posterior mean at low σ is close to the nearest training image, so
`v − v_weak` is largely per-image detail; extrapolating it may speed up memorisation
rather than style. Check for copied compositions, not only style strength.

## Cost

One extra no-grad DiT forward a step with the guide LoRA swapped in (the paper's
quarter-size guide costs 9 %; a same-size guide under block swap costs more).
