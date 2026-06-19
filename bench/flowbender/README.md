# bench/flowbender — closed-loop feedback-aware colorize (arXiv:2606.20404)

FlowBender on the EasyControl colorize adapter. Proposal:
`docs/proposal/flowbender_easycontrol.md`.

## Phase 0 — PASSED (done)

`probe_alignment_drift.py` — does open-loop colorize leave an uncorrected
forward-operator (`mangafy`) alignment error? Median `err_final/floor` **17.8×**
(range 7–45×), drift survives blur (low-freq structural, not screentone phase),
luma preserved. Real, large headroom → proceed. Result in
`results/20260619-2045-phase0/`.

```
uv run python bench/flowbender/probe_alignment_drift.py --num_samples 12
```

## What shipped (Phase 1 — training)

The trainable method is wired end-to-end:

- `library/inference/forward_operators.py` — `MangafyOperator` (`H` = decode →
  mangafy → encode), the zero-order operator shared by training + the probe.
- `networks/methods/easycontrol_attention.py` — `_ExtendedSelfAttnLSEFunc3`:
  target attends `[target_k; cond_k; feedback_k]` with independent `b_cond` /
  `b_feedback` gates. Forward + backward (incl. the analytic bias grads)
  validated against an SDPA reference in `tests/test_flowbender_easycontrol.py`.
- `networks/methods/easycontrol.py` — feedback cond-LoRA pool + `b_feedback`
  gate, `set_feedback`/`clear_feedback`, the three-stream block forward. Built
  only under `use_flowbender` (off → bit-identical to plain EasyControl).
- `train.py` — two-pass loop: pass 1 (look-ahead, no_grad, feedback off) →
  `x̂₁ = x_t − σ·v` → `set_feedback(H(x̂₁))` → pass 2 (grad). `p_un`
  null-feedback dropout; `blocks_to_swap>0` refused. `torch_compile` is
  supported — the three-stream inner is compiled with its own dynamic_seq mark
  prologue (`compile_cond_stream` / `_make_patched_block_forward`).
- `configs/easycontrol/colorize_flowbender.toml` — the A/B descriptor (reuses
  colorize's staged caches).

Train (after the colorize staging/preprocess in `colorize.toml` has run once):

```
make easycontrol EASYADAPTER=colorize_flowbender
```

## A/B and kill criteria

Baseline = **plain colorize FT** (`configs/easycontrol/colorize.toml`;
`output/ckpt/anima_colorize_comic`). FlowBender = `anima_colorize_flowbender`.

- **Fidelity** — the Phase-0 `mangafy`-consistency ratio on a **held-out** set.
  Success = FlowBender drives `err_final` materially toward the floor vs FT. Run
  `probe_alignment_drift.py --ckpt <each>` and compare `ratio_final_over_floor`.
  *Requires inference-time feedback* (see "Remaining" below) — without it the
  checkpoint runs off-distribution (the cond stream was co-trained with feedback).
- **Plausibility** — CMMD (paired PE-Core MMD²); FM val loss is uninformative.
- **Saturation guard** — track output saturation distribution. **Kill** if
  `mangafy`-err drops but CMMD or saturation regress (feedback over-enforced the
  flat-lineart tone → desaturated/flattened, colorize's known failure mode). If
  so: reduce `b_feedback` headroom / `p_un`, or restrict `H` to the XDoG edge
  channel (drop the screentone band) so feedback enforces structure not value.

## Remaining — inference two-pass (Phase 2 wiring)

Training is complete; **inference** still runs open-loop. To realize the gain
and run the fidelity A/B, the sampler loop
(`library/inference/generation.py`) needs the honest two-pass per step (2N
forwards, same user step count):

1. pass 1 with feedback off → `x̂₁ = latents − σ·noise_pred` (already computed as
   `denoised`); 2. `network.set_feedback(MangafyOperator(vae)(x̂₁))`;
3. pass 2 with feedback on → use that `noise_pred` for the step (and the CFG
   uncond branch). Do **not** call `precompute_cond_kv` (feedback recomputes per
   step; the network already disables the cond KV cache when feedback is active).
The `t_thresh` prior-step shortcut + feedback KV-cache are explicitly out of
scope (proposal §"Out of scope v1").
