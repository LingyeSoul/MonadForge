# Correction training on MOSS-Image (dual LoRA)

Port of *Correction Training for Diffusion Models* (Ahn et al., arXiv 2610.14136): a
denoiser LoRA D and a corrector LoRA C chained after it, plus a per-channel gate γ.
Question: can correction replace MOSS's CFG 3 (the community reads CFG 3–4 at 50 steps
as blurry)?

**Verdict (2026-10-21, one seed, three prompts):** only joint D + C in x-space works. At
CFG 1 it repairs structure step by step up to K 8 without drifting, but does not reach
CFG 3. **CFG 1.5 + K 2 matches CFG 3** at twice the forwards, and CFG 2 alone is most of
the way there: correction replaces guidance strength, not speed (that needs distillation
or a lower NFE).

## Method

Code: `src/correction.py` (corrector network, `correct_once`), `src/correction_train.py`
(wraps `library/moss/train.py`: swaps `LoRANetwork`, `training_loss`, `place`),
`src/correction_ab.py` (cfg × K grid). Nothing under `library/` changed.

C is the DiT with D's LoRA plus its own (starts as a copy of D), reads D's clean estimate
`x̂ = z_t − σ·v₀` and the same caption.

- **v-space** (paper eq. 8, `--corr_sigma 0`): `v₁ = v₀ + γ(C − v₀)`, C at timestep t.
- **x-space** (paper App. B.2, `--corr_sigma σ_c`): C cleans `x̂` at
  `σ_c = min(0.15, σ)`, `x̂₁ = x̂ − γ·σ_c·C`, `v₁ = (z_t − x̂₁)/σ`.

Loss: MSE on v₁ against `ε − z₀`, the trainer's weighting. `γ = gate·40` (γ's Adam step
is 40× the LoRA lr). Sampling corrects the cond branch K times per step before CFG; the
uncond branch is not corrected (C never trained on `""`).

## Runs

caststation (`channel_(caststation)_layout`, 30 images), the shipped recipe per LoRA:
Base, r32 / α128, `weight_svd`, lr 5e-5, channel scaling 0.5, `dc_isolate`, 20 epochs =
600 steps, seed 0, `--plain_caption_rate 0`. Frozen D = the floor
`caststation_layout_at_r32a128_lr5e5`.

| Run | D | Form | γ init | Steps | γ end | L_corr vs L_denoise | s/step |
|---|---|---|---|---|---|---|---|
| 1 | frozen floor | v-space | 0 | killed at 200 | 0.012 | −0.10 % | 2.8 |
| 2 | frozen floor | v-space | 0.5 | killed at 100 | 0.33, falling | +55 % (+95 % at 50) | 2.8 |
| 3 | frozen floor | x-space 0.15 | 0 | 600 | 0.177 | −1.0 % | 2.8 |
| 4 | **joint**, from Base | x-space 0.15 | 0 | 600 | 0.126 | −0.37 % | 3.9 |

Run 4's D ends at L_denoise 0.2272 vs the floor's 0.2261 on the same σ sequence.
Checkpoints `output/ckpt_moss/caststation_corr_*` (C + `.gate.safetensors` +
`.corr.json`; run 4's D is `…_joint_r32_lr5e5_g40_D.safetensors`). Renders:
`lora_ab.py`'s three prompts, seed 1, Base, 1024², ER-SDE 50 steps, uncond skip 22, no
Spectrum; arms `cfg:K`, forwards per step `1 + K` (+1 with CFG); grids under
`output/tests/corr_caststation_corr_x015_{frozenD,joint}_r32_lr5e5_g40/`.

## Reads

- **v-space stalls.** C ≈ D reads `x̂` at timestep t as a noisy input, so `C − v₀`
  points far off: from γ 0 the gate never opens, from γ 0.5 the loss doubles and the gate
  closes. (The paper uses a full-model copy, batch 256–1024, 50–100K steps.)
- **x-space opens**: γ grows 4.7× faster than run 1 from the first window.
- **Frozen D (run 3) amplifies the dataset.** K 1–2 at CFG 1: slightly more colour and
  contrast, no sharper lines. K 4 drifts (phone → pink scarf, a "CHICANEL caststation
  2022" signature, p2's skin darkens, eyes go red): the paper's frozen-D failure (Fig.
  17–18).
- **Joint (run 4) holds.** At CFG 1, K keeps the K 0 composition and repairs it (stray
  soda can gone, phone gets three lenses, hands close around it, p2's cardigan → the
  prompted sailor uniform). No drift or signature at K 4 or K 8; K 8 adds a white disc
  over p2's eye. p0's tangled peace sign is the weakest repair.

| Arm | fwd/step | Read (seed 1) |
|---|---|---|
| `3:0` | 2 | reference; writes a "channel … caststation" name tag on p0 |
| `1:0` | 1 | composition errors (soda can, cardigan, tangled hands), pastel / low contrast |
| `1:4` | 5 | structure fixed, still lower contrast and looser adherence than `3:0` |
| `1:8` | 9 | p2 full-body on the bench like `3:0`; eye-disc artifact |
| `2:0` | 2 | close to `3:0`; adds unprompted glasses and a sign with garbled text (p2) |
| `1.5:2` | 4 | **on par with `3:0`**: p2 on-prompt, no glasses, no sign; p0 / p1 clean |
| `2:2` | 4 | close to `3:0`; glasses come back on p2 |

No arm looks blurry at 1024² on this LoRA, CFG 3 included, so this grid cannot test the
blur report.

## Open

1. Seed 2+ and a second LoRA; the `2:0` vs `1.5:2` call is two cells.
2. No-LoRA control: the blur complaint is about Base; train a corrector on a wider set
   (or score Base + corrector).
3. Cost: distill `1.5:2` (or `1:8`) into one LoRA (paper §3.3: 3× fewer NFE at equal
   FID); correct only on a σ window (paper's SiT-REPA runs: 0.7 → 0.5); fewer steps at
   matched NFE; Spectrum on the D / uncond forwards with corrections on full steps.
4. Training length: γ climbed through steps 350–550 in both 600-step runs (run 3
   flattening at 0.177, run 4 dipping to 0.126 at 600); longer training and a larger
   `gate_scale` are untested.
