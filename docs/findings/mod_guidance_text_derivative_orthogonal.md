# Mod-guidance: the distilled head matches the teacher pointwise but its text-derivative is orthogonal

> **STATUS (resolved 2026-06-05).** The GAD experiment this finding motivated was
> run, and the *negative* branch held. A JVP/finite-difference GAD term (and a
> σ-FiLM head) did **not** lift `cos` off zero. A DC/AC decomposition explains why:
> the teacher's text response is ~99% AC, but AdaLN modulation can only write DC,
> so the reachable `cos` ceiling for *any* pooled-AdaLN head is just 0.05–0.17 —
> an **architectural ceiling**, not a fit gap — and the head already sits at it.
> Mod-guidance via AdaLN is a global-tone/contrast lever, not a content lever:
> GAD-for-mod-guidance ships at `gad_weight=0` (dead) and σ-FiLM is inert even when
> opted in. The measurement below stands — read it for the *mechanism*, not as open
> work. Full verdict + reproduce: `_archive/gad/gad.md` → "GAD for
> mod-guidance" (doc archived 2026-06-12; the mod-guidance code ships at
> `gad_weight=0` and the turbo instantiation was removed 2026-06-10, so GAD
> has no live home). See [[project_mod_guidance_sigma_film]].

> **TL;DR.** The `pooled_text_proj` head reaches ~2.5% held-out relative error
> (the "low val loss" it was trained to) yet its *response to a text change* is
> statistically uncorrelated with the teacher's (`cos(ΔS, ΔT) ≈ 0` at every σ).
> This is not a fit failure or a distribution artifact — it is intrinsic to
> pointwise MSE distillation when the text-conditioning signal is small relative
> to the achievable residual. It is the textbook precondition for a
> geometry-aware (first-order / JVP) distillation term — which was then wired and
> run (see the STATUS note above). Mechanism home: `_archive/gad/gad.md`.

## What was measured

`bench/mod_guidance/text_jacobian.py` (generation-free; reuses the distill
teacher/student forwards). For held-out `(latent, σ, noise)` it perturbs the
text from sample A toward sample B and compares the two pathways' output deltas:

- teacher: `ΔT = v(crossattn_B) − v(crossattn_A)`           (text via cross-attn, `skip_pooled_text_proj=True`)
- student: `ΔS = v(pooled_B)   − v(pooled_A)`                (text via the modulation MLP, crossattn pinned at uncond)

Run on `pooled_text_proj-0602.safetensors`, **matched to its training
distribution** (`--synth_data_dir post_image_dataset/distill_mod_synth`), n=96
pairs/σ. Baseline run dir: `bench/mod_guidance/results/20260604-1848-0602-synth/`.

| σ | cos(ΔS,ΔT) | ratio ‖ΔS‖/‖ΔT‖ | ‖ΔT‖ | err_a=‖s_a−t_a‖ | rel_err=err_a/‖t_a‖ | ΔSNR=‖ΔT‖/err_a |
|---|---|---|---|---|---|---|
| 0.10 |  0.002 | 0.826 | 11.2 | 13.9 | **0.025** | 0.80 |
| 0.40 |  0.003 | 0.262 | 18.7 | 14.1 | **0.024** | 1.32 |
| 0.70 | −0.002 | 0.103 | 41.0 | 28.6 | **0.047** | 1.43 |
| 0.90 | −0.005 | 0.053 | 90.4 | 64.0 | **0.108** | 1.41 |

`cos` standard error ≈ std/√96 ≈ 0.002, so every `cos` is within ~1 SE of zero —
**orthogonal, not merely degraded.**

## Why this happens (mechanism)

1. **Good pointwise fidelity.** `rel_err` 2.5–11% confirms the head reproduces
   the teacher's prediction well on its own (synth) distribution — consistent
   with the low val MSE it was trained to.
2. **The text signal is tiny relative to the prediction.** Swapping to a fully
   unrelated prompt moves the teacher's velocity by only `ΔT/‖t_a‖ ≈ 2%` at low
   σ (growing to ~15% at σ=0.9). At low σ that signal is **smaller than the
   distillation residual itself** (ΔSNR<1).
3. **MSE never constrains the text *direction*.** Pointwise MSE minimizes
   `‖s−t‖` by nailing the latent-dominated bulk; the ~2% text contribution sits
   below the error floor and is free to point anywhere. So the head learns
   *that* the output should match, never *which way* it should move as text
   changes → orthogonal derivative.

## Two robust, separable readings

- **Direction (cos ≈ 0):** confounded by an *architectural ceiling* — a global
  AdaLN shift cannot spatially reproduce cross-attn's text response, so cos=1 is
  not achievable. (Consistent with `bench/mod_guidance/channel_attribution.py`'s
  finding that the pooled channel is a weak/orthogonal carrier vs cross-attn.)
  The clean test is **differential**: does a derivative-aware objective lift cos
  *off zero*?
- **Magnitude (ratio 0.83 → 0.05):** *not* confounded by direction. The head
  transmits 83% of the text magnitude at σ=0.1 but only 5% at σ=0.9 — it
  increasingly ignores text exactly where the teacher leans on it most. At σ=0.9
  the signal exceeds the residual (ΔSNR=1.41), so there is supervisable signal
  there; this is a clean target a first-order term can bite on.

## Implication

This is precisely the regime GAD (geometry-aware distillation) targets: outputs
match, the derivative is unconstrained/collapsed. That motivated adding a
JVP/finite-difference term to `distill_mod` to supervise `ΔS → ΔT` — **done, and
it did not move `cos`.** The decisive test was "does cos lift off zero"; it did
not, and a DC/AC decomposition (`text_jacobian.py`, extended) showed why: the
teacher's text response is ~99% AC while AdaLN modulation can only push DC, so the
reachable `cos` ceiling is 0.05–0.17 and the head already sits at it. The "clean
negative result" branch held — mod-guidance cannot carry a teacher-aligned text
*direction*; it is a global-tone lever. Full verdict + the σ-FiLM no-op:
`_archive/gad/gad.md` → "GAD for mod-guidance".

## Caveats / how to reproduce

- A first run on **real** latents (`post_image_dataset/lora`, no `--synth_data_dir`)
  gave nearly identical cos≈0 but inflated `err_a` — a distribution mismatch
  (0602 was synth-trained). Always probe with the head's training distribution.
- Cross-pathway by construction: teacher text rides cross-attn, student text
  rides modulation. Read `cos` for direction and `ratio` for magnitude
  separately; raw L2 of `ΔS−ΔT` mixes the two.

```bash
uv run python -m bench.mod_guidance.text_jacobian \
    --pooled_text_proj output/ckpt/pooled_text_proj-0602.safetensors \
    --synth_data_dir post_image_dataset/distill_mod_synth \
    --n_pairs 96 --sigmas 0.1 0.4 0.7 0.9 --h 1.0 --label 0602-synth
```
