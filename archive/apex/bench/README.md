# bench/apex/ — APEX feasibility probes

## Why this directory exists

We've been trying to use APEX (arXiv:2604.12322 — *Self-Adversarial One
Step Generation via Condition Shifting*) to distill an existing Anima
T-LoRA into a 1–4 NFE student. So far it isn't working, and the cost of
"keep tuning hyperparameters" is several hours per failed run.

A deep-research review of the method on Anima's architecture concluded
that the most likely failure mode is **structural, not numerical**:

> Anima cross-attn does ``kv_proj(c) → split → RMSNorm(k) → softmax``.
> Softmax kills any token-independent additive offset to logits, and
> RMSNorm kills global rescaling. So scalar ``c_fake = a·c + b`` is
> approximately invisible to the model — APEX degenerates into anchored
> self-distillation because ``F_θ(x_t, t, c) ≈ F_θ(x_t, t, c_fake)``
> and the "adversarial" branch has nothing to disagree about.

Our ``configs/methods/apex.toml`` already records the consequence: the
shipped scalar mode collapsed at Anima scale, ``(a, b)`` drifted <3%
in 7k steps, and we switched to ``diag``. But we never directly verified
that diag (or any later state) actually reaches the cross-attention key
subspace — we were inferring liveness from a downstream metric
(``v_fake_divergence``, the MSE between ``v_fake_sg`` and ``F_real``
through the entire DiT) which is too coarse to localize the problem.

This directory holds the cheap-to-run forward-pass probes that answer
the structural question directly, **before** committing to a training
run. They don't train; they run a few cached text embeddings through
the cross-attn modules and measure whether ``ConditionShift(c)`` makes
the keys and values move in a way attention can see.

## When to (re-)run

- Before kicking off any new APEX training run (``--mode init`` probe).
- After the first ramp-up window of an APEX run, against the partial
  checkpoint (``--apex-ckpt`` probe), to confirm ``(a, b)`` drift is
  producing visible K/V deltas — not just non-zero-but-absorbed bias.
- After changing ``apex_condition_shift_mode`` or ``apex_shift_lr_scale``.
- As a sanity check that scalar mode IS dead (expected FAIL — a
  baseline that confirms the metric is calibrated).

## Scripts

### ``probe_temporal_shift.py``

A characterization probe (no PASS/FAIL gate) for an alternative
adversarial-signal lever inspired by EMF (arXiv:2602.02571): instead of
shifting the *condition* (``c_fake = A·c + b``), shift the *timestep*
(``t_fake = t ± Δt``) and use the second forward as the L_mix target.
Compares MSE / cosine / SNR of ``v_fake`` vs ``v_real`` across:

  - ``identity`` (sanity floor, ~0)
  - ``cond_diag`` (shipped APEX baseline, diag init at the live (a,b))
  - ``cond_signflip`` (scalar a=-0.5, b=0)
  - ``dt_pos_small`` / ``dt_neg_small`` (Δt = ±0.05)
  - ``dt_pos_big`` / ``dt_neg_big`` (Δt = ±0.20)

Decision rule (informal):

  - ``dt_*`` median MSE ≈ ``cond_diag`` median MSE  → temporal shift is
    a comparable supervision lever; worth a training-side trial.
  - ``dt_*`` ≪ ``cond_diag`` → too weak, kill the idea.
  - ``dt_*`` ≫ ``cond_diag`` → target may be too noisy; investigate
    before training.

```bash
python bench/apex/probe_temporal_shift.py \
    --warmstart output/ckpt/anima-tlora-0507-12.safetensors
```

### ``probe_attention_visibility.py``

Per cross-attn block, runs ``c`` and ``c_fake = ConditionShift(c)``
through ``kv_proj + k_norm + v_norm`` and reports:

| Metric                       | What it tells you |
|------------------------------|-------------------|
| ``attn_sym_kl`` (gate 1)     | Symmetric KL between attention maps under ``c`` vs ``c_fake`` with synthetic q's — direct measure of whether the maps differ at all. |
| ``k_pre_token_indep_frac`` (gate 2) | Fraction of ``ΔK_pre`` that is the same vector at every token — softmax kills this. High = invisible. |
| ``k_post_rel`` (sanity floor) | ``\|\|ΔK_post\|\| / \|\|K_post\|\|`` after k_norm. Catches bit-identical ``c_fake`` bugs but isn't load-bearing under uniform ``(a,b)`` init — see calibration notes below. |
| ``v_post_rel``               | ``\|\|ΔV\|\| / \|\|V\|\|`` (v_norm = Identity), reported for symmetry. |
| ``k_pre_rel``                | Pre-norm K perturbation, reported for symmetry. |

**Decision rule.** PASS requires all three gates (median over blocks):

1. ``attn_sym_kl ≥ --gate-kl`` (default **0.5 nats**). Adversarial signal
   needs attention maps to actually differ. For reference, two unrelated
   prompts give KL on the order of nats per row; 0.06 nats (what
   shipped scalar/diag init produces on Anima) is ~1% of "different
   prompt" — enough to be technically nonzero but too weak to drive
   useful gradient.
2. ``k_pre_token_indep_frac ≤ --gate-indep`` (default **0.5**). At most
   half the ΔK_pre energy may live in the softmax-invisible
   token-independent subspace. Shipped uniform inits sit at ~0.97.
3. ``k_post_rel ≥ --gate-k-post`` (default **0.05**, sanity floor).
   Matches the 2D toy ``mean_rel_shift`` bound in
   ``archive/bench/apex_phase0.py``. Only catches ``c_fake = c`` wiring
   bugs; does not by itself imply attention sees the perturbation.

PASS = "perturbation is alive in K-space AND softmax-visible — APEX
adversary has signal." FAIL = "perturbation will degenerate into anchored
self-distillation under any hyperparameter sweep."

**Why two real gates instead of one.** An earlier version of this script
gated only on ``k_post_rel ≥ 0.05`` and gave false PASS on every
shipped setting: under ``c_fake = -0.5·c + 1.0`` (uniform diag init),
``k_post_rel ≈ 1.91`` (38× the threshold) but
``k_pre_token_indep_frac ≈ 0.97`` and ``attn_sym_kl ≈ 0.06`` — i.e.
the perturbation was huge in raw magnitude but almost entirely in
softmax's null space. The two-gate rule directly measures the property
APEX needs (attention maps differ) instead of a proxy that uniform
parameterizations can fake.

**Recommended invocations.**

```bash
# 1. Probe shipped diag init under the actual training base
#    (warm-start LoRA merged in, matching promote_warmstart_to_merge)
python bench/apex/probe_attention_visibility.py \
    --warmstart output/ckpt/anima_lora.safetensors

# 2. Confirm scalar mode IS dead (expected FAIL — baseline)
python bench/apex/probe_attention_visibility.py \
    --warmstart output/ckpt/anima_lora.safetensors --mode scalar

# 3. After a partial APEX run, probe the trained shift
python bench/apex/probe_attention_visibility.py \
    --warmstart output/ckpt/anima_lora.safetensors \
    --apex-ckpt output/ckpt/anima_apex.safetensors
```

**Outputs** → ``bench/apex/results/<YYYYMMDD-HHMM>[-<label>]/``:

- ``result.json`` — standard envelope (git SHA, env, args, summary,
  pass/fail).
- ``per_layer.csv`` — raw per-block numbers for plotting / inspection.

## What FAIL implies

If diag-init fails the probe under the actual warm-start base, the
options in increasing-cost order are:

1. **Try ``--mode full`` with non-trivial off-diagonal init** (e.g.
   ``A = -0.5·I + ε·randn`` orthogonalized). Off-diagonal mixing is the
   only way the existing ``c_fake = A·c + b`` parametrization can put
   energy into per-token-different directions at init, which is what
   softmax sees. Cheapest experiment; doesn't require code changes.
2. **Initialize diag with per-channel jitter** (e.g.
   ``init_a ~ N(-0.5, 0.1)``, ``init_b ~ N(1.0, 0.1)`` per dim) so the
   shift isn't uniform-across-channels at step 0. Diag at uniform
   ``(a, b)`` is mathematically identical to scalar — only training
   drift breaks the equivalence, but warmup may end before drift takes
   hold. A jittered init gives the optimizer a non-degenerate starting
   gradient.
3. **Replace ``ConditionShift`` with something that lives in the K/V
   subspace** — low-rank rotation on ``kv_proj`` output, per-token
   learned offset rather than broadcast bias, or a perturbation trained
   to directly maximize ``attn_sym_kl``. This stops being "APEX" and
   starts being a fork; cite the deep-research report
   (``deep-research-report (1).md``) for rationale.
4. **Pivot to a teacher-based few-step distillation method** (LCM-LoRA,
   Hyper-SD) targeting 2–4 NFE rather than 1 NFE. The merged
   ``warm-start ⊕ DiT`` already provides a usable teacher.
5. **Accept that the existing T-LoRA at 20 euler steps is the right
   operating point** for this base.

## What PASS implies

PASS doesn't guarantee APEX works — it only rules out the structural
"perturbation is invisible" failure. Other failure modes (warm-start
incoherence, ``L_fake`` contaminating the real branch, ``T_mix``
collapsing to a self-confirming fixed point at high λ) still apply and
need their own diagnostics during training. PASS just means the
hyperparameter sweep is *worth running* — under FAIL it isn't.
