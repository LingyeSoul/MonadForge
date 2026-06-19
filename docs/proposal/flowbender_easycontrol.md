# flowbender_easycontrol — closed-loop feedback-aware training for EasyControl colorize

Adapt **FlowBender** (arXiv:2606.20404, *Feedback-Aware Training for Self-Correcting
Conditional Flows*) onto the EasyControl colorize adapter: train the model to consult
its own condition-alignment error and bend the trajectory toward the manga condition,
instead of running open-loop and hoping the colorization respects the lineart/tone it
was handed.

## TL;DR

The deployed colorize model reproduces structure/pose/edges well but **disrespects the
luminance/tone the manga screentone encodes** — it'll color a mid-tone dress near-black.
FlowBender closes that gap by feeding the model, at each sampling step, the deviation
between (a) the manga re-extracted from its *own* current clean estimate and (b) the
manga condition it was given, as an extra conditioning input. The model learns a
non-linear correction policy over that error.

We take the **zero-order** variant (feed the re-extracted measurement, no gradients)
because our forward operator `H = mangafy` (XDoG + halftone) is non-differentiable — this
is exactly the regime the paper's zero-order branch targets (JPEG, black-box ops). The
feedback rides as a **second cond-token stream** through EasyControl's existing
LSE-extended attention — no patch-embed surgery, frozen DiT stays frozen.

## Phase 0 — PASSED (done)

`bench/flowbender/probe_alignment_drift.py`, 2026-06-19. For the deployed
`anima_colorize_comic` checkpoint we captured the per-step clean estimate
`x̂₁ = x_t − σ·v` (forward hook on the DiT, CFG=1, euler), decoded it, re-applied the real
operator `H = mangafy`, and measured `‖mangafy(decode(x̂₁)) − mangafy(target)‖²` against
the VAE-roundtrip floor.

| Metric | Value |
|---|---|
| Median `err_final / floor` (12 samples) | **17.8×** (range 7–45×) |
| Blurred ratio | 64× (drift *survives* blur → low-freq structural, not screentone dot-phase) |
| Median luma MSE | 0.004 (brightness preserved) |

The reference `mangafy(GT)` matches the input condition (operator validated). The drift is
a genuine tone/value-mapping disagreement, separable from the already-fine luma channel —
precisely the condition-infidelity FlowBender corrects. **There is real, large headroom.**

This is the opposite of the luma-preserving-colorize null we worried about: because the
real operator (mangafy) is lossy, the model has room to be inconsistent with it.

## Why EasyControl is a clean fit

- **Conditioning is already token-based.** EasyControl's target stream attends over
  `[target_k ; cond_k]` with a per-block scalar gate `b_cond`
  (`networks/methods/easycontrol_attention.py::_extended_target_attention`). Adding the
  feedback signal is just **a third key/value block** `[target_k ; cond_k ; feedback_k]`
  with its own gate — architecturally identical to a second reference. No channel-concat
  into PatchEmbed (the paper's ControlNet route), which would need a trainable patch-embed
  extension and fight the frozen DiT.
- **The operator already exists, runnable.** `mangafy_array_gpu`
  (`easycontrol_adapters/colorization/mangafy_gpu.py`) is the same op that synthesized the
  training conditions. `H` is free.
- **The cond pre-pass already runs every step in Phase-1 colorize** (no KV cache yet), so a
  per-step feedback stream is consistent with the existing cost structure.
- **colorize is `blocks_to_swap=0`** (`configs/easycontrol/colorize.toml`), which sidesteps
  the load-bearing gotcha below.

## Architecture

```
Pass 1 (look-ahead, no_grad, feedback gate OFF):
    v_LA = v_θ(x_t, t, cond)              # unguided velocity
    x̂₁   = x_t − σ·v                      # Euler clean estimate  (5D boundary, see below)
    m̂    = mangafy(decode(x̂₁))            # H(x̂₁): re-extracted manga  (stop-grad)
    feedback_latent = encode(m̂)           # manga-domain, same space as cond

Pass 2 (refine, with grad):
    set_feedback(feedback_latent)         # → per-block (K_f, V_f), gate b_feedback
    v_ref = v_θ(x_t, t, cond, feedback)   # attends [target_k; cond_k; feedback_k]
    L_FA  = ‖v_ref − u_t‖²                 # standard flow target; sg[feedback]
```

The feedback stream is the **re-extracted measurement** `H(x̂₁)` (manga domain). The
original condition `y` is already present as `cond_k`, so the model sees "what you were
asked for" (`cond`) and "what you currently produce, re-measured" (`feedback`) side by side
and learns to drive them together — equivalent to feeding the residual, but it keeps both
streams in the same VAE-encodable manga domain.

### The 5D-latent boundary (load-bearing, do not skip)

`x̂₁` is computed from the loop latent; decode wants 4D. **Squeeze dim 2 explicitly**
(`x1.squeeze(2)` only if `dim()==5`), never `squeeze()`. `decode_to_pixels` returns 4D for
4D input; if ever fed 5D it returns `(B,3,1,H,W)` — squeeze dim 2 before mangafy. See the
project invariant on the dim-2 singleton.

### Two-pass = a 2nd DiT forward per step → block-swap constraint

Pass 1 is an extra (no_grad) DiT forward. The block-swap offloader **desyncs on any 2nd
DiT forward per step** ([[project_blockswap_extra_forwards_gradcache]] — soft-tokens hit
this). colorize is already `blocks_to_swap=0`, so we're safe; **the FlowBender path must
refuse `blocks_to_swap>0`** (assert + clear error) rather than silently corrupt grads.
Only pass 2 backprops, so there is one backward per step — no grad-cache gymnastics needed.

### Step-0 / init equivalence (mirror b_cond)

`b_feedback` initialises to a large negative logit (like `b_cond=-10`) so the feedback rows
contribute ≈0 mass at init → **step-0 forward is bit-equivalent to plain EasyControl**.
This is the invariant test (below).

## Training

- Modify the EasyControl path in `train.py` to the two-pass loop above, gated behind
  `use_flowbender` (off by default). batch_size=1 (colorize) keeps the per-step single
  `mangafy` call cheap; use the 2D-fold VAE for the decode/encode.
- **Null-feedback dropout** `flowbender_p_un` (paper's `p_un`, best at 0.1): randomly run
  pass 2 with the feedback gate off, so the unguided look-ahead stays reliable. Distinct
  from `easycontrol_drop_p` (image-CFG), which colorize sets to 0.
- **Composes with REPA** (colorize already runs `use_repa`/`repa_layer=8`/`repa_target_dog`,
  [[project_easycontrol_repa_validated]]): conditioning-consistency ⟂ representation
  alignment. Keep REPA on; report the interaction.
- **channel_scaling**: the feedback stream is another trainable-down cond-LoRA, so
  channel_scaling is LIVE on it and may want its own calib like the cond stream did
  ([[project_channel_scaling_cond_stream_needs_own_calib]]) — start with the shared
  cond calib, profile transfer before adding a third.

Per-step cost ≈ 2 DiT forwards + 1 VAE decode + 1 mangafy + 1 VAE encode ≈ ~3× a plain
step. colorize is 4 epochs — acceptable.

## Metrics, A/B, and kill criteria

Baseline = **plain colorize FT** (FLAIR, the obvious training-free comparator, was reverted
2026-06-14 — [[project_flair_reverted]] — and is not available).

- **Fidelity** = the Phase-0 mangafy-consistency ratio on a **held-out** set. Success =
  FlowBender drives `err_final` materially toward the floor vs FT.
- **Plausibility** = CMMD (paired PE-Core MMD², [[project_cmmd_val_signal]]); FM val loss is
  uninformative here ([[project_fm_val_loss_uninformative]]).
- **Saturation guard** (the one-to-many risk made concrete): track the output saturation
  distribution. Part of the Phase-0 "drift" is the model making a *legitimate* palette
  choice that differs from GT; feedback that over-enforces the flat-lineart tone could
  **desaturate/flatten** — which is already colorize's failure mode
  ([[project_easycontrol_comfy_washout_is_cfg]]).

**Kill** if mangafy-err drops but CMMD or saturation regress (it flattened the image to
match the lineart). That outcome means the operator is over-constraining tone — reduce
feedback gate / `p_un`, or restrict `H` to the XDoG edge channel only (drop the screentone
band) so feedback enforces *structure* not *value*.

## Out of scope (v1)

- **First-order feedback** — needs a differentiable `H`; halftone quantization isn't.
- **Inference prior-step shortcut** (`t_thresh`, the N+1-forward trick) — a Phase-2
  optimization once Phase-1 shows a win; v1 inference is honest two-pass (2N forwards, same
  user step count — *not* "use 2× steps").
- **KV-cache for the feedback stream** — it's recomputed per step by construction
  (depends on the evolving `x̂₁`), unlike the static cond.
- Sanitize / near-twins / 3D texturing — no clean forward operator; not this proposal.

## Files

- `networks/methods/easycontrol.py` — `set_feedback()`, feedback cond-LoRA pool +
  `b_feedback` gate, wire into the extended-attention call.
- `networks/methods/easycontrol_attention.py` — `[target_k; cond_k; feedback_k]` (extend
  the LSE combine to three blocks).
- `train.py` — two-pass FlowBender loop on the EasyControl path; `use_flowbender`,
  `flowbender_p_un`; `blocks_to_swap>0` refusal.
- `library/inference/.../forward_operators.py` (new) — thin `mangafy` operator wrapper
  shared by training + the Phase-0 probe.
- `configs/easycontrol/colorize.toml` (or a `colorize_flowbender` descriptor) — the flags.
- `bench/flowbender/` — add the training A/B harness alongside `probe_alignment_drift.py`.
- `tests/test_flowbender_easycontrol.py` — **invariant**: step-0 forward with `b_feedback`
  at init is bit-equivalent to plain EasyControl; look-ahead pass (gate off) ≡ plain
  forward (Tier-2 new-method requirement).

## Prior art in tree

- Phase-0 probe + result: `bench/flowbender/results/20260619-2045-phase0/`.
- EasyControl two-stream mechanics: `networks/CLAUDE.md`, `docs/experimental/easycontrol.md`.
- REPA on EasyControl (composes): [[project_easycontrol_repa_validated]].
