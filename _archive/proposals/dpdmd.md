# DP-DMD — diversity-preserved distillation (turbo successor proposal)

> Proposal to replace the CA-decoupled DMD2 turbo objective with **DP-DMD**
> (Wu, Li, Zhang, Ma — *Diversity-Preserved Distribution Matching Distillation*,
> arXiv 2602.03139). Reference implementation cloned at `dpdmd/`
> (`train_sd35_dpdmd.py`, SD3.5-M, flow-matching — an architectural sibling of
> Anima). For the incumbent see `docs/experimental/dmd2-decoupled.md` (ops),
> `docs/structure/dmd2-decoupled.md` (math), and the live decision log
> `docs/proposal/dmd2_decoupled_improvements.md`.

Status: **Phase 1 WIRED — awaiting the bench gate.** Phase 0 was GO on the
pose/composition diversity axis (first-step anchoring visibly de-collapses the
4-step turbo student on the real caption distribution; see §2.2 for the metric
caveat that nearly masked this). The training port now exists in
`scripts/distill_turbo/` as a **selectable objective** (`objective = "dpdmd"`,
alongside the incumbent `"dmd2"`) — see §3 for the code surface and §8 for the
plan to retire the incumbent once the gate passes. Inference-only probe:
`bench/dpdmd/probe_first_step_anchor.py`, results under `bench/dpdmd/results/`.
The build target is **vanilla first-step DP-DMD**; the multi-step "depth-m"
generalization in §5 is a documented fallback, not the plan.

---

## 0. Why pivot (read this first)

The shipped turbo objective is **structurally over-distilling and diversity-
collapsing**, and our own decision log already diagnosed why. From
`dmd2_decoupled_improvements.md §0`, the student gradient is two branches:

```
grad_signal = grad_dm                               # DM branch — has a fixed point, converges
            + tau_ca·(alpha_eff − 1)·delta_cfg      # CA branch — constant CFG bias, NO fixed point
```

The **CA branch bakes the classifier-free-guidance direction into a single-step
student and never settles** — "every step with α>1 adds another dose of CFG; it
never settles, it just keeps pushing the student off the real-data manifold
toward oversaturation." We have spent the whole turbo program managing that
dose (α-ramp area, grab-early checkpoints, `dm_x0_norm` to claw back seed
diversity) and the CA-side levers keep coming back inert or harmful:

- `ca_band` — inert + redundant (`project_turbo_alpha4_overdistill`).
- FEI-band-deficit CA — Phase-0 GO but **falsified live** (`project_turbo_fei_gap_phase0`).
- turbo_H @ 4k — latest scale-up, still does not work.

The common thread: **CA is the problem, not the missing lever.** DP-DMD removes
the CA branch entirely. It does not bake CFG into one step; it keeps a genuine
multi-step student and reallocates *which step* sees *which loss*.

## 1. The DP-DMD mechanism (mapped to our notation)

DP-DMD is **role-separated** over the student's `N` denoising steps (we ship
`N=4`). Linear flow path `z_t = (1−t)x + t·ε`, velocity `v = ε − x` — identical
to Anima's schedule.

**Step 1 — diversity supervision (teacher-anchored, NOT DMD).** From initial
noise ε, run the *teacher* (CFG-guided) for `K` steps to an intermediate latent
`z_tk` at continuous time `t_k`. The teacher-derived target velocity for the
student's first step is

```
v_target = (ε − z_tk) / (1 − t_k)
loss_div = ‖ v_student(ε, t=1) − v_target ‖²        # supervise step 1 only
```

This forces different ε to land on the teacher's *diverse* manifold point after
step 1, instead of the mode-seeking DMD collapse. `K` is the **diversity anchor**
(their Table 3: too-early K = weak semantics, too-late K = teacher detail the
student can't reproduce; moderate K best, default `K=5` of 30 teacher steps).

**Detach after step 1 (load-bearing).** `z_1 = stopgrad(student_step_1(ε))`. The
DMD gradient from the later steps must NOT flow back into step 1, or the mode-
seeking reverse-KL overrides the diversity mapping. Their Fig 5 ablation shows
the conflict appears early (preference rises while diversity falls) without the
detach.

**Steps 2..N — DMD quality supervision.** Roll the student the remaining `N−1`
steps from `z_1` to `x_θ`, then the standard DMD loss against teacher + fake
(auxiliary) scores refines perceptual quality:

```
x_θ   = student_rollout(z_1, N−1)
loss_dmd = DMD(x_θ; teacher, fake)
loss  = loss_dmd + λ · loss_div                      # λ = 5e-2 default
```

No perceptual loss, no GAN, no extra modules, no teacher-generated reference
images. Output is still a plain LoRA run at `--infer_steps 2 --cfg 1.0`
(matched to the `student_steps=2` rollout).

Reference: `dpdmd/train_sd35_dpdmd.py:435-527` (anchor rollout → `v_target` →
first-step `div_loss` → detach → `N−1` student rollout → `compute_dmd_loss`).

## 2. Grounding on Anima

1. **Phase 0 — two inference-only runs** (`bench/dpdmd/probe_first_step_anchor.py`,
   R=4, 1024², `anima_turbo_H_4k`). Three arms: teacher (ceiling, 28-step CFG),
   4-step student (floor), and *anchor-injection* (teacher rolled to the student's
   step-b boundary σ, student finishes b..N — `anchor@b` is the faithful proxy
   for "DP-DMD whose step 1 lands teacher-like"). Eq. 9 diversity, PE-Core pooled:

   | arm | σ | generic prompts | real captions |
   |---|---|---|---|
   | teacher | — | 0.138 | 0.043 |
   | anchor@1 | 0.90 | 16.5% | 13.4% |
   | anchor@2 | 0.75 | 51% | 62.5% |
   | anchor@3 | 0.50 | 94% | 67.4% |
   | student | — | 0.040 | 0.027 |

   (anchor rows = % of the teacher−student gap recovered.)

2. **The pooled-cosine metric measures the wrong axis — read the grids, not the
   number.** PE-Core *pooled* features are CLIP-aligned → they score **semantic
   content** diversity. On the real caption distribution the content is pinned by
   ~40 booru tags (same character, clothes, setting), so semantic diversity is
   structurally ≈0 regardless of the sampler — hence the tiny 0.043 teacher
   ceiling and the harsh anchor@1 = 13.4%. The axis that actually varies, and the
   one we want, is **pose/composition**, which pooled cosine barely sees. The
   saved grids show `anchor@1` clearly de-collapsing pose on several real prompts
   (p02: student = 4 near-identical downward gazes → anchor@1 varies head tilt /
   gaze-up / crop; p00 modest; p01 ~unchanged). **The GO is on the pose axis,
   visually, prompt-dependent.** Phase-1 must replace the eval metric with a
   structure-sensitive one (`pe_spatial` spatial tokens, in-repo, or DINOv3 as the
   paper uses) — `project_fm_val_loss_uninformative` already warned our scalar
   metrics don't track what matters.

3. **Where diversity is actually determined (the b-sweep).** On the *cosine*
   (semantic) axis, diversity is set in the student's **middle** steps (σ 0.90→0.50,
   anchor@2/@3), not step 1 — consistent with `project_sigma_signal_resolves_by_045`
   (x0 resolves by σ≈0.45). This is why §5's depth-m generalization exists as a
   fallback: if a trained first-step supervision proves too weak, the role
   boundary moves to σ≈0.5–0.75. But the *pose* recovery visible at anchor@1 says
   the cheap first-step method is worth trying first.

4. **Infra reuse — we already own every component.** Teacher rollout, fake/
   auxiliary critic, `dmd_loss` via renoise, flow-matching `ε−x` target, the
   plain-LoRA save path. DP-DMD reuses all of it; only the student *loop* changes.

## 3. What changes in our code

The incumbent student is **single-step** (`scripts/distill_turbo/distill.py:435-504`:
sample one `t`, one student forward → `x_pred`, then CA/DM renoise branches). DP-
DMD needs a **genuine N-step scheduler rollout student**. That is the one real
structural change; everything else is deletion or reuse.

| Piece | Incumbent | DP-DMD | Action |
|---|---|---|---|
| Student forward | single-step `x_pred = student(x_t)` | N-step rollout, step 1 detached | **rewrite** |
| Diversity signal | `dm_x0_norm` band-aid on seed variance | first-step `loss_div` vs teacher K-anchor | **add** |
| CA branch | `tau_ca·(α−1)·delta_cfg` (CFG bake) | — | **delete** |
| α-ramp / `alpha_warmup` | dose management for CA | — | **delete** |
| DM branch | renoise-DMD vs fake, single-step | standard DMD on `x_θ` (steps 2..N) | **keep / re-site** |
| Fake critic update | `fake_steps_per_student_step` inner loop | unchanged | **keep** |
| Teacher | no-grad teacher forwards | + K-step CFG anchor rollout | **extend** |
| Output | plain LoRA, 4-step cfg=1 | plain LoRA, 2-step cfg=1 (matched to N) | **keep / retarget** |

### 3.1 New config knobs (`scripts/distill_turbo/config.py`) — AS BUILT

The objective is selected by a top-level `objective` key (CLI `--objective`);
DP-DMD adds a `[dpdmd]` block plus `flow_shift` under `[sampling]`:

```
objective: str           = "dmd2"  # "dmd2" (incumbent) | "dpdmd" (this method)
# [dpdmd]
k_anchor: int            = 5       # teacher steps to the diversity anchor (their K)
teacher_anchor_steps:int = 28      # teacher grid the K is counted against
div_weight: float        = 5e-2    # λ (their default)
detach_after_first: bool = True    # the load-bearing stopgrad (keep True; A/B only)
# [sampling]
flow_shift: float        = 3.0     # σ-schedule shift for the student/teacher Euler grids
```

Run it with `--objective dpdmd` (or set the TOML key); `student_steps` (N) and
`teacher_cfg` (the anchor's CFG scale) are reused from `[dmd]`. Validation
enforces `student_steps ≥ 2` and `1 ≤ k_anchor < teacher_anchor_steps`.

**Kept-but-inert under `dpdmd` (NOT yet retired):** `tau_ca_strategy`,
`tau_ca_min_gap`, `tau_ca_skip_above_t`, `alpha`, `alpha_warmup_steps` — these
drive only the CA branch, which `dpdmd` never enters. They survive so the
incumbent `dmd2` path stays runnable for A/B; their removal is staged in §8 and
gated on the Phase-1 decision. `dm_x0_norm` IS honoured by the DP-DMD DM branch
(default on via the TOML); expect it **unnecessary** here — its reason for
existing (seed-diversity recovery) is now handled at the source by `loss_div`,
so the first A/B should test turning it off.

The student loop branch lives in `scripts/distill_turbo/distill.py` (the
`if cfg.objective == "dpdmd":` block): teacher K-step CFG anchor → `v_target`,
N-step student rollout with first-step detach + `loss_div`, then the existing
renoise-DMD machinery re-sited onto `x_θ`. The fake-critic update, warmup,
mean-variance shield, metrics (`train/div_loss`), and plain-LoRA save path are
shared with the incumbent. Block-swap stays at `blocks_to_swap=0` for Phase 1
(the multi-forward offloader audit is deferred to Phase 3, per §3.3).

### 3.2 Loss assembly (replaces distill.py:435-535)

```python
# --- teacher K-step anchor (no grad, CFG) ---
z_tk, t_k = teacher_anchor_rollout(eps, k_anchor, teacher_cfg)   # extend existing teacher fwd
v_target  = (eps - z_tk) / (1.0 - t_k)

# --- student N-step rollout; step 1 div-supervised + detached ---
x_stu, v_first = eps, None
for i, t_i in enumerate(student_sigmas[:-1]):
    v = student(x_stu, t_i)
    x_stu = euler_step(x_stu, v, student_sigmas, i)
    if i == 0:
        v_first = v
        x_stu = x_stu.detach()            # detach_after_first
loss_div = mse(v_first, v_target.detach())
x_theta  = x_stu

# --- DMD on x_theta (steps 2..N), against teacher + fake ---
loss_dmd = dmd_loss(x_theta, teacher_CFG, fake_cond)   # real score is CFG-GUIDED
loss = loss_dmd + div_weight * loss_div
```

The fake-critic update branch (`distill.py:539+`) is unchanged — it still
regresses the fake to the student's `x_θ` distribution at resampled τ.

> **⚠ CFG trap — the real score in `dmd_loss` MUST be CFG-guided
> (`v_u + α·(v_c − v_u)`), not cond-only.** "Reuse current renoise-DMD
> machinery" above is misleading: the incumbent's DM branch is *deliberately
> cond-only* because under the CA-decoupled scheme guidance lived in the
> separate CA branch (`delta_cfg`). DP-DMD deletes the CA branch, so guidance
> has to ride the single DMD real score — exactly like the reference
> `compute_dmd_loss` (`dpdmd/train_sd35_dpdmd.py:118-129`). The original Phase-1
> port (`anima_turbo_I`, 2026-05-30) copied the cond-only teacher verbatim and
> dropped CFG entirely → `v_real ≈ v_fake` (`dm_cos ≈ 0.9999` in TB), the
> quality gradient went to noise, and the student looked worse than `g_agg_250`.
> Fixed at `distill.py:517` by routing the DM real score through
> `_teacher_cfg_velocity`. The fake stays cond-only (matches the reference).
> If the incumbent CA path is ever decommissioned (§8), do **not** revert the
> DM teacher to cond-only — un-decoupling the DMD term is now load-bearing.

### 3.3 Constraints that survive contact with Anima

- **Block-swap + extra forwards.** The student now runs `N` forwards per step and
  the teacher runs `K` anchor forwards — multiple DiT forwards per optimizer
  step. `project_blockswap_extra_forwards_gradcache` warns the offloader desyncs
  on a 2nd DiT forward. The N-step rollout must call
  `model.prepare_block_swap_before_forward(free_cache=False)` before each forward
  (the soft-tokens fix pattern), or run with `blocks_to_swap=0`. **Audit this
  first** — it bit soft-tokens and the incumbent single-step turbo never exercised
  it.
- **Compile.** `compile_blocks()` keys on token count; the N-step rollout reuses
  the same bucket each step, so no new graphs. Toggle is multiplier-free here (no
  teacher/student adapter flip mid-graph — teacher is the frozen base, student is
  the live adapter), so compile is clean.
- **Memory / no full-BPTT.** Batch=1 on 16 GB. The N-step student rollout with
  grad only through steps that aren't detached: step 1 (detached → no grad
  retained past it) + steps 2..N feed `loss_dmd`. Grad checkpointing per block
  as today. The detach bounds the graph to `N−1` steps; verify peak VRAM at
  1024² native buckets.
- **Output bake.** Plain LoRA, no change to merge/inference.

## 4. Phasing

- **Phase 0 — GO on the pose axis** (done): inference-only anchor@1 visibly de-
  collapses pose on the real caption distribution; the pooled-cosine metric
  understates it (§2.2). `bench/dpdmd/probe_first_step_anchor.py`.
- **Phase 1 — minimal training port (CODE WIRED).** N-step rollout student +
  `loss_div` + detach + DMD on `x_θ`, landed as a selectable `objective="dpdmd"`
  (§3.1) — the incumbent CA path is bypassed at runtime, not yet deleted (§8).
  Single-GPU, `blocks_to_swap=0` to sidestep the offloader audit initially.
  **Remaining before the gate: fix the eval metric** — pose/structure-sensitive
  diversity (`pe_spatial` spatial tokens or DINOv3), not pooled cosine; pooled is
  blind to the axis we're optimizing. Gate: a checkpoint whose CMMD/quality
  matches the best turbo (`g_agg_250`) while measurably beating it on the
  *structure-sensitive* diversity metric at fixed seeds, with grids that confirm
  pose variety. Bench: `bench/dpdmd/` with the standard `result.json` envelope
  (`bench/_common.py`).
- **Phase 2 — anchor/λ sweep.** Reproduce their Table 3 (`K`) / Table 4 (`λ`) on
  Anima. The b-sweep says deep-K (step-1 velocity aimed at a σ≈0.5 anchor) commits
  to diverse branches harder — sweep `K` upward from their default. Confirm
  `dm_x0_norm` is now redundant (turn it off).
- **Phase 3 — detach ablation + scale.** Verify the detach is load-bearing on
  Anima (their Fig 5). Then block-swap audit for multi-forward correctness and a
  longer run. **If first-step supervision underperforms when trained**, escalate
  to the §5 depth-m role split.

## 5. Fallback — depth-m role separation (only if Phase-1 first-step is too weak)

The b-sweep (§2.3) shows that on the *semantic* axis diversity is locked in the
student's middle steps (σ 0.90→0.50), which vanilla DP-DMD detaches to DMD. If a
trained first-step `loss_div` fails to hold pose diversity through those steps,
generalize the **single-step** role split to a **depth-m** split:

> supervise the first **m** steps toward teacher-anchored (per-ε, diverse)
> targets → **detach after step m** → DMD on the remaining `N−m` steps.

`m` replaces `K` as a **diversity↔quality dial**, set so the role boundary sits at
Anima's σ≈0.5–0.75 lock point (m=2 buys ~half the gap, m=3 ~all). This is the
authors' own anticipated extension (their Conclusion: *"adaptive role
separation… a trajectory-aware objective"*). **Cost, stated plainly:** m
supervised steps = trajectory distillation for those steps — the blur/mean-
regression mode DMD was built to avoid (cf. shelved `project_l2p_pixel_transfer`,
SPD-analytic-target findings). Two mitigations: targets are diverse *per-ε* (pin
each noise to its own branch, not one path), and the `(ε−z_tk)/(1−t_k)` shortcut
target points at the endpoint rather than tracing the path. Whether that avoids
blur is a training question, not readable from the inference probe — which is why
depth-m is a fallback, not the plan.

## 6. Risks / open questions

1. **The detach may interact with our renoise-DMD differently than their closed-
   form DMD.** Their `loss_dmd` is the SD3.5 DMD; ours is the renoise/fake-critic
   variant. The role separation (step-1 detach) is objective-agnostic in
   principle, but verify the fake critic still tracks `x_θ` once step 1 is
   detached (the fake regresses to `x_θ`, which is downstream of the detach — fine).
2. **K-anchor cost.** `K` extra teacher forwards per step inflate step time.
   `K=5` teacher + `N=4` student + fake inner loop ≈ 10+ DiT forwards/step at
   batch=1. Acceptable at our scale but the block-swap audit is mandatory.
3. **`v_target` at the anchor σ.** The teacher anchor lands at `t_k` on the
   teacher grid, not the student grid; their code reads `t_k_cont` from the
   teacher scheduler (`train_sd35_dpdmd.py:462-466`). Mirror that exactly — a σ
   mismatch in `(1 − t_k)` silently mis-scales `v_target`.
4. **CMMD vs diversity tension.** `project_fm_val_loss_uninformative` /
   `project_cmmd_val_signal`: our live val is CMMD. DP-DMD trades a little
   quality for diversity by design; pick the checkpoint on the *joint* signal,
   not CMMD alone. Add the Eq. 9 diversity metric to the turbo eval.

## 7. Decision gate for Phase 1 — PASSED

Gate (revised from the pooled-cosine version, which was the wrong axis): build
Phase 1 if `anchor@1` — the faithful proxy for trained first-step DP-DMD —
visibly de-collapses **pose** on the real caption distribution without degrading
images. **Met:** the `20260530-1250-real_prompts` grids show clear per-prompt
pose variety at `anchor@1` (p02 strongly, p00 modestly; p01 unchanged — the win
is real but prompt-dependent). The cosine number (13.4%) is discounted because
PE-Core pooled scores tag-pinned *semantic* content, not pose (§2.2).

Phase-1 kill condition: once trained, if the first-step student cannot hold the
pose diversity that `anchor@1` previewed (measured with the structure-sensitive
metric, not pooled cosine), escalate to §5 depth-m before shelving; only shelve
if depth-m also fails to beat `g_agg_250` on the joint quality+diversity signal —
then write the finding to `docs/findings/`.

## 8. Decommissioning the incumbent CA-decoupled DMD2 path

DP-DMD shipped as a **selectable** objective so the incumbent stays runnable for
A/B (§3.1). This section is the plan to delete the CA-decoupled path once DP-DMD
proves out — it is **not** to be executed until the trigger below is met.

**Trigger (all required):**
1. Phase-1 gate (§7) passes — a DP-DMD checkpoint matches `g_agg_250` quality and
   beats it on the structure-sensitive diversity metric, grids confirmed.
2. Phase 2/3 settle which variant ships (vanilla first-step or §5 depth-m). Both
   reuse the same rollout/detach/DMD code — neither resurrects the CA branch — so
   either outcome clears removal.
3. No open A/B that still needs the incumbent (`dm_x0_norm` off-test in Phase 2
   runs under `dpdmd`, so it does **not** block this).

If the trigger is NOT met (DP-DMD shelved), do the inverse: delete the `dpdmd`
branch + knobs and write the finding to `docs/findings/`. Removal is one
direction or the other — the dual-path state is temporary by design.

**Removal checklist (once triggered):**

- `scripts/distill_turbo/distill.py` — drop the `else:` (dmd2) branch in the
  student-update region: the CA branch (teacher cond/uncond × renoise →
  `delta_cfg`), the `alpha_warmup`/`alpha_eff` ramp, and the CA augmentation of
  `grad_signal`. The `if cfg.objective == "dpdmd":` guard then becomes the
  unconditional body; delete the `div_loss_t = 0` fallback and the `do_ca` /
  generator-`t` sampling block. Drop the `sample_t_above` import.
- `scripts/distill_turbo/primitives.py` — `sample_t_above` becomes dead; remove
  it. `renoise` / `sample_t` / `make_scheduler` / `PadCache` all stay (DP-DMD and
  the fake critic use them).
- `scripts/distill_turbo/config.py` — remove the `objective` selector (DP-DMD is
  the only path), the `--alpha` / `--alpha_warmup_steps` flags, the
  `tau_ca_strategy` / `tau_ca_min_gap` / `tau_ca_skip_above_t` keys + their
  validation, and the now-unused dataclass fields. **Keep** `teacher_cfg` (anchor
  CFG), `t_distribution` / `sigmoid_scale` (still drive the fake-critic `sample_t`
  and warmup), `dm_x0_norm` / `norm_floor`, and the `[mean_var]` block.
- `scripts/distill_turbo/metrics.py` — remove the CA-only scalars (`cfg`,
  `dm_to_ca`, `ca_steps`, `alpha`), `accumulate_dm_to_ca`, `add_alpha`, and their
  TB/postfix keys. Keep `div`, `dm`, `grad`, `xpred`, `v_student`, the
  fake-tracking ratios, and `mv`.
- `configs/methods/turbo.toml` — drop the `objective` key, the CA keys in `[dmd]`
  (`tau_ca_*`), `alpha_warmup_steps` in `[optim]`; promote the `[dpdmd]` knobs to
  the canonical schedule. The header comment stops referencing the CA-decoupled
  doc.
- Docs — move `docs/experimental/dmd2-decoupled.md`, `docs/structure/dmd2-decoupled.md`,
  and `docs/proposal/dmd2_decoupled_improvements.md` under `_archive/` (retired-
  material convention); replace this proposal with a shipped method deep-dive at
  `docs/experimental/dpdmd.md` and update the `CLAUDE.md` Methods table (the
  **Turbo** row) to describe DP-DMD. Refresh the `ss_turbo_*` metadata note if the
  CA-era keys are dropped.
- Memory — the `project_turbo_*` notes (sign-fix, x0-norm, FEI-gap, α4-overbake,
  curation) become historical once the CA path is gone; leave them as the record
  of why CA was abandoned, but add a one-line "superseded by DP-DMD" pointer to
  `project_dpdmd_pivot_phase0`.

**Invariants that must survive removal:** the output is still a plain LoRA run at
`--infer_steps 2 --cfg 1.0` (matched to `student_steps=2`); `merge` / inference are untouched; `make exp-turbo` /
`exp-test-turbo` targets keep working (they just stop accepting the CA flags).
Land the deletion behind `make test-unit` (config-resolution + registry tests)
and one short `dpdmd` smoke run.
