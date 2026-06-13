# Turbo prompt-following — Phase 0 measured (degradation confirmed), Phase 1 soft-rank auxiliary

Status: **Phase 0 COMPLETE 2026-06-11 — gate FIRED → Phase 1 unlocked, not started.**

- Bench (now the standing regression probe): `bench/dpdmd/caption_ranking_probe.py`
- Phase-0 run: `bench/dpdmd/results/20260611-2020-phase0-turboN1250/` (checkpoint
  `anima_turbo_N_1250.safetensors` — the current 4-step DP-DMD student, 1250 iters)

Premise sources: `docs/findings/agsm_reward_premise_holds.md` (the validated ranking
reward), `docs/findings/turbo_fei_band_deficit_falsified.md` (the "measure at the
distribution the loss sees" lesson the probe obeys), `docs/experimental/dpdmd.md` +
`docs/experimental/soft_tokens.md` (the two implementations being crossed).

## The question — now answered

**Does DP-DMD distillation preserve the teacher's caption discriminability?** No —
it degrades it, concentrated exactly at the student's step-0 state (σ → 1), the
state the proposed soft-rank auxiliary trains. Nothing else measures turbo
prompt-following (CMMD is distribution match, not text alignment), so this probe
is also the first quantitative confirmation of the "pose/prompt effects only show
in grids" evaluation gap.

## Why the ranking reward is trustworthy here

`agsm_reward_premise_holds.md` established on the frozen base that relative
FM-ranking at a shared `(x_t, ε, t)` is a valid compass (everything that pollutes
absolute FM-MSE cancels across candidate captions), and that discriminability is
perfect (hard rank@1 = 1.000) for σ ≥ 0.45. The **4-step** student operates at
σ ∈ {1.0, 0.9, 0.75, 0.5} (`student_sigmas`, `scripts/distill_turbo/distill.py:703`,
flow_shift=3) — its entire band sits inside the validated regime, with 0.5 at the
boundary (thin margins, so the probe gates it relative-to-base rather than on the
absolute threshold).

> The original draft targeted the 2-step student (σ ∈ {1.0, 0.75});
> `student_steps` went 2→4 in `5ef128d`. Everything below is the 4-step version:
> probe σ grid {0.5, 0.75, 0.9, 0.97, 1.0}, inference/A-B at `--infer_steps 4`.

## Phase 0 — what was run and what it showed

`bench/dpdmd/caption_ranking_probe.py` (defaults = the run): 24 anchors, k=2
negatives/pool, 2 noise draws, one DiT with the turbo LoRA toggled via
`set_multiplier` (base arm = multiplier 0, the teacher's backbone). Negative
sourcing (shuffled + same-artist/diff-character hard pools) is imported from
`bench/soft_tokens_contrastive/reward_premise_probe.py`, same RNG streams, so the
frozen reference numbers stay comparable. Three arms:

- **0a renoised-real ranking** — `x_t = (1−σ)x0 + σε` from cached latents, reward
  `−‖v(x_t,c_j) − (ε − x0)‖²`, both arms on identical states.
- **0b anchor ranking (on-trajectory)** — the `distill.py:826-838` step-0
  construction verbatim: teacher CFG=4 rolled `k_anchor=6` of 12 grid steps →
  `v_target = (ε − z_tk)/(1 − t_k)`; rank `v(ε, t=1, c_j)` against it.
- **0c caption-contrast transfer** — `‖v_S(c_pos) − v_S(c_neg)‖ /
  ‖v_T_cfg(c_pos) − v_T_cfg(c_neg)‖` + cosine, at 0a's x_t (σ=0.75/0.90) and at
  the student's own z1. CFG is affine, so the teacher contrast is exactly
  `α·(v_T(c_pos) − v_T(c_neg))` — no uncond forwards needed outside the 0b rollout.

### Results (run 20260611-2020-phase0-turboN1250)

0a shuffled rank@1 (chance 0.333; pre-registered absolute gate 0.93 at σ ≥ 0.75):

| σ | base | student | gate |
|---|---|---|---|
| 0.50 | 1.000 | 0.917 | rel −0.083 |
| 0.75 | 1.000 | 0.958 | — |
| 0.90 | 1.000 | 0.958 | — |
| 0.97 | 1.000 | **0.875** | abs + rel |
| 1.00 | 0.958 | **0.750** | abs + rel −0.208 |

Hard pool: student 0.882–1.000, below base by 0.059–0.118 at four σ's (rel gate).
0b: base 1.000 / student 0.958, student margins ~2× base (the trained quantity is
nearly intact). 0c: ratio 1.109 / 0.794 / 0.768 (x_t@0.75 / x_t@0.90 / z1@0.90) —
no magnitude collapse, that gate arm correctly silent — but cos(ΔS, ΔT) ≈
0.32–0.41 everywhere.

### Reading

1. **The degradation is real and lives at the step-0 state.** Monotone fall-off
   toward σ=1, worst exactly at t=1 where the student must rank from text alone
   (a quarter of anchors lose to a *random* caption). Per the pre-registered
   site rule ("σ=1.0/0.97 → step-0 site; σ≤0.75 → re-scope"), **the Phase-1
   siting is settled: step 0, as proposed.**
2. **Wider margins, worse ranking.** Student mean margins are 2–3× base at every
   σ while rank@1 is lower — an overcommitted, noisier text response, not a
   uniformly attenuated one. (This is why a margin-style objective would be the
   wrong fix and a *rank* objective is the right one.)
3. **0b vs 0a is the diagnosis.** On the trained quantity (teacher-anchor
   explanation) the student is nearly perfect; on renoised real data it isn't.
   `div_loss` taught "explain the anchor", not "discriminate captions" — which is
   precisely the gap the soft-rank term closes (make the *matched* caption explain
   the anchor better than mismatched ones).
4. **0c caveat.** The low cosine (≈0.38) says the student's caption-contrast
   direction rotated away from the teacher's CFG contrast, but the probe has no
   cosine noise floor (e.g. teacher-vs-teacher contrast across noise draws), so
   don't quote it as an absolute rotation yet — see open controls.

### Open controls (cheap, before/alongside Phase 1)

- **Over-distilled arm**: rerun the probe on an over-baked checkpoint
  ([[project_turbo_alpha4_overdistill]] predicts text response dies first there);
  degradation deepening with over-baking strengthens the causal story. One
  command: `uv run python bench/dpdmd/caption_ranking_probe.py --adapter <ckpt>
  --label overdistill`.
- **0c noise floor**: add a teacher-vs-teacher (different ε) contrast cosine to
  calibrate what cos ≈ 0.38 means.

## Phase 1 — soft-rank auxiliary in `distill.py`

### Why soft-rank (and not InfoNCE or AGSM)

The soft-tokens program already burned down this decision tree: AGSM was removed
2026-05-30 (its `w_matched` pinned at chance all run); InfoNCE's negative branch is
unbounded (the SoftREPA degrade-while-loss-drops failure). Soft-rank won the live
A/B and the offline gradient probe (`cos(∂L/∂V, ∂margin/∂V) ≈ 0.86` vs AGSM's
0.25, [[project_softrank_agsm_gradient_probe]]). It is bounded by construction
(`L → 0` as the matched caption's rank → 1, self-annealing at the win) — and
Phase 0's "wider margins, worse ranking" result (reading #2) independently argues
for optimizing *rank*, not margin. Reuse
`bench/soft_tokens_contrastive/_softrank.py` / the `softtorch` dependency.

### Wiring (the real seams)

Site it at **step 0, the diversity site** — `distill.py:846-865` (empirically
confirmed as where the damage is). Everything needed is already in scope there:

- `v_target` (the matched-caption teacher anchor, `:838`) — **free**.
- `v_first = v_student(ε, t=1, c_pos)` (`:855`) — **free**.
- New per firing step: `k` extra student forwards `v_j = v_student(ε, t=1, c_neg_j)`
  under `no_grad`, same `(ε, t)`, only `crossattn_emb` swapped — the
  `extra_forwards` contract soft-tokens validated (and the same swap the Phase-0
  probe just exercised 3k+ times without surprises).

```python
r = stack([-mse(v_first, v_target), *(-mse(v_j, v_target) for j in negs)])
L_rank = soft_rank(r, method, softness)[0] - 1          # 0 when matched wins
loss_div_total = cfg.div_weight * div_loss_t + cfg.softrank_weight * L_rank
```

It rides the existing step-0 backward (the `split_bwd` branch at `:860-865`
backwards the diversity term before the detach — the rank term joins that
backward, so the DMD graph separation is untouched). v0 detaches the negatives:
gradient flows through the live `v_first` only ("make the matched caption explain
the anchor better"), the bounded-objective property holds, and there is no second
retained graph. The soft-tokens `after_backward` grad-cache replay (push on the
negatives too) is a documented fidelity upgrade, not v0.

**Negative sourcing.** `CachedDataset` already yields per-sample `crossattn_emb`;
v0 negatives = TE tensors of other samples in the pool (`shuffled` — the pool
where Phase 0 measured the worst damage, σ=1.0 rank@1 0.750). `hard` (via
`IdentityPairSampler` + `caption_index.json`) is a later knob; Phase 0 saw the
hard-pool deficit too (−0.059…−0.118) but its strict pool covers only ~29% of
stems (7/24 anchors came up short in the run).

### Config (`configs/methods/turbo.toml`, new `[softrank]` section)

| key | default | note |
|---|---|---|
| `weight` | `0.0` | **off → byte-identical training** (no negatives loaded, no extra forwards) |
| `k` | `2` | matches the probe (chance 0.333); `softtorch` needs k ≥ 2 |
| `every_n` | `4` | fire on every Nth student step; effective strength ≈ weight/N (manual, not auto-scaled — soft-tokens precedent) |
| `softness` / `method` | `0.1` / `neuralsort` | straight from the soft-tokens defaults |

### Cost & caveats

- `k=2, every_n=4` ≈ +0.5 no-grad student forwards per step amortized — small next
  to the existing `2·k_anchor` teacher + N student + 4 fake forwards.
- **Block swap:** extra DiT forwards are the offloader's audited-risk area
  ([[project_blockswap_extra_forwards_gradcache]]). The turbo default is
  `blocks_to_swap=0` and the loop already calls
  `prepare_block_swap_before_forward` per forward, but do not enable this together
  with swap without auditing.
- **σ=1.0 only — now evidence-backed.** Phase 0 located the damage at
  σ=0.97/1.0 with the σ=0.75/0.90 operating states near-intact, so the step-0
  site is the treatment site; the "move the site to z1 against a teacher
  referee" contingency is dead unless a later probe run says otherwise.

### Decision gate (Phase 1)

A/B at fixed seed/data/iterations, `--infer_steps 4 --cfg 1.0`:
`softrank.weight = 0` vs `0.05` (sweep one order of magnitude if the first point
is inert). Ship only on all three:

1. **Probe recovery** — rerun `caption_ranking_probe.py` on the trained
   checkpoint; 0a shuffled rank@1 at σ=0.97/1.0 recovers toward base (clearing
   the original 0.93 absolute gate at all σ ≥ 0.75 = full recovery; the baseline
   to beat is 0.875/0.750). Don't trust the training-time rank scalar alone.
2. **CMMD non-regressing** ([[project_cmmd_val_signal]]).
3. **Grids show no diversity collapse** (reuse `diversity.py`).

## Sequencing

```
Phase 0 probe (0a + 0b + 0c)                              ✅ DONE 2026-06-11
   └─ DEGRADATION (σ=1.0 shuffled 0.750 vs base 0.958)    → Phase 1 unlocked
        ├─ optional controls: over-distilled arm, 0c noise floor
        └─ Phase 1 soft-rank auxiliary (off-by-default flag)
              └─ Gate: probe recovery + CMMD + grids ──► ship / kill
```

## Contributing tier

Phase 1 is numerics-changing → **Tier 1.5**: bench script (the Phase-0 probe
doubles as it — already in tree) + invariant test (`softrank.weight=0`
byte-identical to current `distill.py`).

## References

- `bench/dpdmd/caption_ranking_probe.py` + `results/20260611-2020-phase0-turboN1250/`
  — the probe and the Phase-0 numbers above.
- `docs/findings/agsm_reward_premise_holds.md` — the reward premise + σ-trend.
- `bench/soft_tokens_contrastive/reward_premise_probe.py` — negative
  sourcing/anchor machinery the probe imports; `_softrank.py` — the rank loss;
  `negative_audit.py` — hard-pool coverage.
- `scripts/distill_turbo/distill.py:703` (grids), `:826-838` (ε / anchor),
  `:846-865` (step-0 diversity site the rank term joins), `:860-865` (split
  backward).
- `docs/experimental/soft_tokens.md` §"Soft-rank objective" — objective history
  (AGSM removal, boundedness, knob semantics).
- `docs/findings/turbo_fei_band_deficit_falsified.md` — why 0b measures
  on-trajectory; the probe/loss distribution-match rule.
- [[project_fm_val_loss_uninformative]] — why the reward must stay *relative*.
