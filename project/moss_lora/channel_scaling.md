# Channel scaling on MOSS-Image

Port of Anima's channel scaling to the MOSS LoRA. Usage and mechanism:
`docs/moss/channel_scaling.md`. This file is the log and the open items.

Question that started it: the DAVE probe found massive channels holding ~99 % of the
residual's DC (`dave.md`). Is that the same thing Anima's channel scaling rebalances, and
does it reach the LoRA inputs?

## Log

**2026-10-18 — weights only (CPU).** `norm1.weight[1356]` ≈ 0.000–0.03 on blocks 5–20
(median 0.9): the attention inputs mostly drop it. `norm2.weight[1356]` ≈ 0.6. The shared
adaLN (`shared_block_modulation`, one for all 42 blocks) gives the FFN input `1+scale` 0.22
and shift −6.8 on 1356, `1+scale` 0.24 and shift −4.3 on 42 (median |shift| 0.03), flat
over σ 0.1–0.95. Whether the shift cancels the normalized offset or adds to it is not
readable from weights.

**2026-10-18 — LoRA-input probe** (`bench/moss_channel_stats/results/20261018-1120-base`,
Base, `_calib_mixed` 16 × σ {0.1 … 0.9}, 1.6 min). The shift does not cancel:

- `ffn_up_gate`: ch 42 is the top channel on blocks 11–41, 11→26× median; 1356 reaches
  9–13× on blocks 27–28 and 35–41. Top-channel peak-to-mean 2.7 (DC bias).
- `q`: 1356 sits below median on blocks 1–17, rises to 6.7× late; 42 is 4× on blocks 0–5.
- Leverage α=0 → 0.5: q 4.0–18.7 → 2.0–4.4, o 2.6–6.2 → 1.6–2.7, ffn_up_gate 9.3–22.8 →
  3.1–4.9, ffn_down 4.5–34.8 → 2.2–6.2 (block 41 worst, peak-to-mean 9.5). Anima's
  matching groups run about 2× higher (mlp.layer1 11–80, qkv 7.6–37.9).

Reading: the massive channels arrive ~1900× → ≤ 26× after the norm; the one that matters
for a LoRA is 42 into the FFN. The part DAVE leaves out is channel scaling's territory; the
prompt-specific DC DAVE uses is not.

**2026-10-18 — implementation.** `library/lines/adapters.py` (shared with qwen21; inert at
α = 0), scale on the down weight rather than the activation. Calibration shipped as
`library/moss/calibration/channel_stats.safetensors` from the probe above. Default α = 0.

**2026-10-18 — A/B at lr 2e-5.** `caststation_layout_at_r32a128_lr2e5_svd` vs `…_cs05`
(same cache `channel_(caststation)_layout`, 20 epochs, r32 / α128, `weight_svd`, seed 0,
α 0 vs 0.5). Grids `output/tests/ab_caststation_layout_at_r32a128_lr2e5_svd{,_cs05}_50st`
(Base + SFT × 3 prompts × seeds 1, 2, ER-SDE + Spectrum, layout_json).

- Training: loss per epoch equal to the third decimal through epoch 20; peak 12.04 GB,
  1.93 s/step both.
- Weights, `ffn_up_gate` blocks 15 / 25 / 35: `down` drift from the SVD init 6–9 % in both;
  ch 42's column drift / mean column 0.99, 0.61, 0.92 (α 0) vs 0.25, 0.15, 0.17 (α 0.5).
  `q` block 30, ch 922: 1.88 vs 0.58. ΔW norm ratio 0.98–1.01; output leverage on the loudest
  channel 45.8 vs 44.0 (ffn_up_gate median). Per-block ΔW cosine between the runs 0.07–0.92.
- Images: style strength the same by eye. Distance to the no-LoRA image (same prompt and
  seed): cs05 closer in 9 / 12 cells, mean pixel distance 49.4 → 43.7, 32-px 39.2 → 33.6.

Reading: since `up` carries most of the learning, ΔW's channel structure is set by
`up @ down_init` in both runs; what channel scaling removes is the global drift that an
update on the constant ch-42 input causes (a bias on the FFN for every prompt). The
user's read of the grids: it touches less of what the style does not need.

**2026-10-18 — A/B at lr 1e-4.** `caststation_layout_at_r32a128_lr1e4_svd` vs `…_cs05`,
same setup otherwise, on the current cache (the older `caststation_layout_r32a128_svd`
predates the `@artists` captions, so it is not a twin). Grids
`output/tests/ab_caststation_layout_at_r32a128_lr1e4_svd{,_cs05}_50st`, LoRA arm only; the
no-LoRA reference is the 2e-5 grid's ×0.0 (same prompts, seeds, sampler).

- Training: losses within ±0.005 per epoch; 1.87 vs 1.84 s/step.
- Weights: `down` drift 0.27 / 0.28 / 0.23 (α 0) vs 0.30 / 0.29 / 0.25 (α 0.5) on
  ffn_up_gate blocks 15 / 25 / 35; ch-42 column / mean 0.45, 0.71, 0.80 vs 0.19, 0.13,
  0.15. `q` block 30: 1.08 vs 0.42. ΔW norm ratio 0.98–1.03, ΔW cosine 0.02 / 0.17 / 0.48
  / 0.30.
- Distance to no-LoRA: cs05 closer in 5 / 12, mean 50.7 vs 51.2, 32-px 41.4 vs 42.3 —
  the same amount of change.
- Eyeball (user verdict: cs05 much better): both strongly stylised, but α 0 adds
  off-prompt content — glasses + garbled sign (Base p2 s1), garbled sign (SFT p2 s1),
  street and car instead of the park (SFT p2 s2), broken hand pose (Base p0 s1). cs05
  stays on-prompt at the same style strength.

Reading: the mechanism holds (column suppressed, `down` now moves 23–30 %), and at the
lr where plain LoRA loses prompt details, channel scaling keeps them. The distance metric
cannot see this: it measures how much changed, not whether it is right. A first draft of
this entry read the flat distance as "no benefit"; that was the metric, not the images.
The 2e-5 "touches less" and the 1e-4 "stays on-prompt" may be one effect.

**2026-10-18 — defaults flipped** on the lr 1e-4 eyeball: `channel_scaling_alpha` 0 → 0.5,
`lr` 2e-5 → 1e-4 (rank 32 / alpha 128 unchanged). The trainer passes α only for
`algo = lora`.

**2026-10-18 — up-gradient decomposition** (`bench/moss_channel_stats/probe_up_gradient.py`,
`results/20261018-1355-upgrad-fftref`, Base, `_calib_mixed` 16 × σ {0.1 … 0.9}, rank 32
`weight_svd`, the trainer's loss, one backward per step; hook vs autograd `up.grad` 0.4 %).
GraLoRA's complaint is `∂L/∂up = Σ_c (∂L/∂W)[:, c] ⊗ down[:, c]`, which channel scaling
leaves untouched (`h = down·x` is unchanged by the reparam), so this measures it directly.
Medians over steps:

| | q | o | ffn_up_gate | ffn_down |
|---|---|---|---|---|
| rank-1 fraction (σ₁²/Σσ²) — `∂L/∂up` vs `∂L/∂W` | 0.86 vs 0.46 | 0.49 vs 0.39 | 0.82 vs 0.58 | 0.61 vs 0.34 |
| DC share, `(Σ_t g_t) ⊗ (down·x̄)` in `∂L/∂up` vs `(Σ_t g_t) ⊗ x̄` in `∂L/∂W` | 0.89 vs 0.39 | 0.38 vs 0.29 | 0.79 vs 0.53 | 0.51 vs 0.24 |
| effective channels `1/Σ share²` — LoRA vs FFT | 484 vs 1268 | 123 vs 1084 | 51 vs 320 | 133 vs 1393 |
| ch 42 share, blocks 30–41 — LoRA vs FFT | ≈0 | — | 0.29–0.62 vs 0.04–0.11 | — |

- The per-step `up` gradient is close to one direction, and that direction is the bias
  learner: the token-mean output gradient times the latent mean. The token-varying part
  the adapter needs for structure is the remaining 10–20 % (ffn_up_gate, q).
- ch 42 is the top channel of `ffn_up_gate`'s up gradient on 28 / 42 blocks (80 / 80
  steps on blocks 20–40), 3–6× its FFT share: the paper's mechanism, as stated.
- In `q` no single channel carries it; the DC comes from the whole mean vector. The
  entanglement is the low-rank projection itself: `down` (top-r right singular vectors of
  W0) keeps the DC, which lives in that subspace, and drops most of the token-varying
  signal, which is spread over all 3072 dims.
- σ-dependence is mild (DC share 0.87 at σ 0.1 → 0.74 at σ 0.9 on ffn_up_gate).
- **FFT reference**: the full gradient is itself 0.4–0.5 bias on `q` / `ffn_up_gate` (the
  flow-matching error has a large spatial-mean part, more so at low σ: 0.69 → 0.48 over
  σ 0.1 → 0.9), so about half of the DC is the loss. The rank-32 projection concentrates
  the rest: ×2.3 on `q` (0.39 → 0.89), ×1.4 on `ffn_up_gate`, ×1.8 on `ffn_down`, ~×1 on
  `o`. Put the other way, the non-DC share the adapter learns structure from goes
  0.61 → 0.11 (`q`) and 0.47 → 0.21 (`ffn_up_gate`): a 2–5× dilution against FFT.

Reading: the lr 1e-4 benefit of α 0.5 cannot be this term — the reparam does not reach
it. Whatever `α` does happens on the `down` side alone. The up side is where the
paper's effect lives on MOSS, and it is large.

**2026-10-18 — DC isolation, gradient check** (`--dc_isolate`, `library/lines/adapters.py`;
calibration `library/moss/calibration/channel_mean.safetensors` = the signed token mean
`x̄` per target from `probe_lora_inputs.py`, Base; probe `results/20261018-1407-upgrad-dc`,
same setup as above). Row 0 of `down` := `x̄/‖x̄‖`, rows 1–31 re-orthonormalised ⊥ it,
re-projected after every optimizer step; composes with α 0.5 (the constraint is applied in
the stored `down·s` coordinates). `x̄` at the FFN input is dominated by ch 42 with constant
sign, but ch 42 is only 14–21 % of `‖x̄‖²`, so the whole mean direction is isolated, not one
channel. Medians over steps and blocks, plain `weight_svd` → isolated:

| | q | o | ffn_up_gate | ffn_down |
|---|---|---|---|---|
| DC share of `∂L/∂up[:, 1:]` | 0.89 → **0.05** | 0.38 → **0.10** | 0.79 → **0.12** | 0.51 → **0.09** |
| rank-1 fraction of `∂L/∂up[:, 1:]` | 0.86 → 0.43 | 0.49 → 0.32 | 0.82 → 0.37 | 0.61 → 0.27 |
| `‖∂L/∂up[:, 0]‖² / ‖∂L/∂up‖²` | 0.90 | 0.96 | 0.96 | 0.97 |
| effective channels | 484 → 510 | 122 → 582 | 51 → 98 | 133 → 783 |
| ch 42 share, ffn_up_gate blocks 30–41 | — | — | 0.32 → 0.14 | — |

- Works as designed: the bias learner moves into `up[:, 0]` (90–97 % of the gradient
  energy), and the other 31 columns' gradient is centered to within 5–12 % (p90 0.24–0.38:
  the per-step token mean is not the calibration mean, so the isolation is partial per step,
  a 7–8× reduction rather than zero). Their rank-1 fraction (0.3–0.4) is now *below* the
  full-fine-tune gradient's (0.46–0.58).
- Adam normalises per element, so columns 1.. take full-size steps on the cleaner gradient
  even though column 0 holds almost all the energy. That is the intended effect.
- Image arm queued: `caststation_layout_at_r32a128_lr1e4_svd_cs05_dc` (α 0.5 + dc_isolate,
  otherwise the lr 1e-4 twin), grid `output/tests/ab_…_cs05_dc_50st`.

**2026-10-18 — DC isolation, image A/B** (`caststation_layout_at_r32a128_lr1e4_svd_cs05_dc`:
the lr 1e-4 α 0.5 twin + `--dc_isolate`, seed 0; grid `output/tests/ab_…_cs05_dc_50st`, same
prompts / seeds / sampler as the other arms; 20.6 min, 1.99 vs 1.84 s/step for the
re-projection).

- Training: loss per epoch within ±0.002 of the α 0.5 twin (0.2227 vs 0.2242 at epoch 20).
- Weights: `up[:, 0]` (the bias column, 96 % of the gradient energy) ends at 0.67× an
  average `up` column on `ffn_up_gate` — Adam's per-element normalisation means the energy
  share does not turn into a size share.
- Distance to no-LoRA: 50.0 vs 50.0 (α 0.5), closer in 8 / 12 cells; same amount of change.
- Eyeball (Claude's read, sheets in the session scratchpad; user verdict pending): style
  strength and prompt adherence on par with α 0.5 on 9 / 12 cells, but **text artifacts in
  3 / 12** — the artist tag written literally in the sky ("caststation", Base p2 s1), and
  garbled text on hair accessories (Base p1 s1, SFT p1 s1). The α 0.5 arm has none in
  this grid; the α 0 arm had the garbled-sign kind in 2. The set has `english text` in 5 /
  60 captions, so one reading is that the centered gradient learns more of the data,
  signatures included; one seed, so it may also be noise.

Reading: the up-side mechanism is real and the init fixes it as measured, but on this
seed it does not improve the images over α 0.5 and brings back the off-prompt text that
decided the earlier A/B. `dc_isolate` stays opt-in (default off); defaults unchanged.

**2026-10-18 — DC isolation at lr 5e-5** (`caststation_layout_at_r32a128_lr5e5_svd_cs05_dc`,
the dc arm with lr halved and nothing else changed; grid `output/tests/ab_…_lr5e5_svd_cs05_dc_50st`).
Hypothesis: the centered up gradient learns faster, so lr 1e-4 × 20 epochs over-fits with
it (the text artifacts).

- Loss at epoch 20: 0.2207 (lr 1e-4 dc 0.2227, lr 1e-4 α 0.5 0.2242).
- Distance to no-LoRA 43.7 (lr 1e-4 arms 50.0, lr 2e-5 α 0.5 42.3).
- Eyeball (Claude's read, user verdict pending): **no text artifact in 12 / 12 cells** — the
  three cells that failed at lr 1e-4 (sky "caststation", hair-clip text ×2) are clean; style
  strength by eye close to the lr 1e-4 arms, stronger than lr 2e-5 α 0.5; on-prompt
  throughout (two solid-colour backgrounds on p1, which asks for a simple background).

Reading: consistent with the hypothesis, but not a controlled comparison — the α 0.5-only
arm at lr 5e-5 does not exist yet, so "dc at half the lr" is not separable from "half the
lr". That control is the next run.


1. **Second seed** — train seed 1 for both arms at lr 1e-4 and check the eyeball verdict
   holds; a metric for it would need prompt adherence (tagger / VLM on the off-prompt
   items above), not distance to no-LoRA.
2. **lr 1e-4 + cs05 vs lr 2e-5 plain** on the two complex gallery prompts (ten-character
   ensemble, 4:1 banner) that set the earlier 2e-5 default — the new default has not been
   read on them.
3. **α sweep** — 0.25 / 0.5 / 1.0.
4. **Calibration** — sample saturation (re-probe on `sincos_layout`, compare `s`) and SFT
   (`--model sft`); both 2-min jobs, low priority.
5. If (1) or (2) goes against cs05, revert the defaults to lr 2e-5 + α 0.
6. **Up-side arm: DC isolation** — built and measured (above). At lr 1e-4 it is not better
   than α 0.5 (text artifacts); at lr 5e-5 it is clean with lr 1e-4-like style strength. Next:
   **α 0.5 alone at lr 5e-5** (the missing control), then a second seed; α 0 + dc_isolate
   separates the two sides. Original note: not centering. The FFT reference says the bias is half
   the loss's own, so closing the bias path (centering `x`) would fight the target;
   isolating it keeps the path and frees the other rank directions. Output-neutral at init
   (`up` is zero): project `x̄` (or the massive channels' columns) into one row of `down`,
   zero that direction from the other r−1 rows, and keep the row from re-mixing (gradient
   mask on the column block, or the existing column damping). Measure the same probe on
   the new init (DC share should collapse on r−1 of the `up` columns), then the lr 1e-4
   image A/B against α 0.5 on the same seeds. Centering would also need a bias key or a
   bake through ch 42, which is only approximately constant (peak-to-mean 2.7).
