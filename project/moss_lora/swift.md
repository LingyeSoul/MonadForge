# SWIFT target extrapolation on a channel LoRA

Status: planned 2026-10-22. Paper: arXiv 2610.25844 (SWIFT).

## Idea

Train on `y = v + α (v − v_weak)` instead of `v`, with `v_weak` a frozen early
checkpoint of the same recipe (no grad, one extra forward a step). In expectation the
target is autoguidance applied to the posterior mean; the guide is dropped after
training. The paper trains from scratch on ImageNet / CIFAR (default α 0.4, U-shaped in
α, guide "weak but competent"); it has no fine-tuning, LoRA or text-to-image run.

## Arm

- caststation (`channel_(caststation)_layout`, 30 images), the shipped recipe: Base,
  r32 / α128, `weight_svd`, lr 5e-5, channel scaling 0.5, `dc_isolate`, 20 epochs = 600
  steps, seed 0, `--plain_caption_rate 0`.
- Guide: that recipe's step-150 state (epoch 5). The floor
  `caststation_layout_at_r32a128_lr5e5` kept no epoch saves, so rerun it with
  `--save_every_epochs 5`; a standalone 150-step run would end on a different LR schedule.
- SWIFT run: the same recipe from scratch, α 0.4, the step-150 LoRA as the frozen guide.
- Read against the floor (α 0) on the two Space prompts (`space/`), as `lora_ab.py`.

## Risk to watch

On 30 images the posterior mean at low σ is close to the nearest training image, so
`v − v_weak` is largely per-image detail the guide has not memorised yet. Extrapolating
it may speed up memorisation rather than style. Check for copied compositions, not only
style strength.

## Cost

One extra no-grad DiT forward a step with the guide LoRA swapped in (the paper's
quarter-size guide costs 9 %; a same-size guide under block swap costs more).
