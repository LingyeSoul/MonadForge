# Turbo FEI band-deficit revival — CLOSED on-trajectory (the line is dead twice)

Status: **CLOSED 2026-06-20.** This retires the FEI band-deficit lever on turbo
for good. Do not re-propose FEI / band-split levers on the DP-DMD loop.

## What was reopened, and why it was legitimate to reopen

`docs/findings/turbo_fei_band_deficit_falsified.md` killed item 2 (band-deficit
reweighting of the CFG-uplift) in the CA-era loop, but on a *diagnosed*
precondition: it measured the gap at renoised **real** data, where the only
deficit is the trivial "one-shot x0 is noisier than the teacher's," while the
over-blur the lever targeted lives on the student's **own rollout states** — which
DMD2 single-call training never visited. Its stated revival condition:

> The right way to revive it is to measure the band deficit *on the student's
> rollout states*, which only exist if training visits them.

The 2026-05-30 DP-DMD migration (commit `9410a3a`) met that condition as a side
effect: the student now rolls its genuine N-step trajectory in-loop, and the DMD
point is a renoise of the **student's own output** (`distill.py:1006`), not real
data. `docs/proposal/turbo_fei_band_on_trajectory.md` reopened the line to
re-measure at that distribution — measurement only.

## How it was measured (this time, at the right distribution)

`bench/turbo/probe_fei_band.py` on the trained checkpoint
`anima_turbo_P_5k.safetensors` (student_steps=4, anchor 6/12, flow_shift=3,
teacher_cfg=4), n=30 images × 2 ε draws = 60 paired samples. FEI = 2-band,
σ_low = min(H,W)/16 (the original Phase-0 SNR sweep's divisor). Teacher = LoRA
multiplier 0 (CFG-guided), student = multiplier 1 (cond-only).

A trained checkpoint — **not** a fresh short run — is the right instrument: the
zero-init student ≈ teacher for the first thousands of steps, so a 100-step run's
gap is the same trivial one the falsification warned about.

* **Site A** — paired trajectory LF gap at MATCHED σ. At student 4 / anchor 6/12
  the student's post-step-0 state z1 (σ=0.9) coincides *exactly* with the teacher
  anchor's step-3 state (σ=0.9), both integrated from the same ε → **|Δσ|=0**,
  the clean paired comparison the 2026-05 wiring lacked (a σ-mismatch there
  injected a ~0.08 LP shift and inverted the TB sign).
* **Site B** — x0-scale band deficit at the DMD point, swept over a fixed τ grid
  (the falsified headline was a *sign structure across the schedule*, so a
  τ-pooled mean would hide a flip).

## The result — the falsified inverse persists, and the trajectory is clean

**Site A (trajectory, σ-matched):** `gap_low = −0.00140 ± 0.00116`,
`frac_over_low = 0.03`. e_low student 0.00671 < teacher 0.00811 → the student
carries *less* LF than the teacher at σ=0.9; it does **not** over-blur on its own
rollout. 97% of samples show no over-low. The direct test the falsification never
had is negative.

**Site B (DMD point), per τ (n=60):**

| τ | dh_pos (HF deficit → over-blur / w_high) | dl_pos (LF deficit → w_low) | frac dh | frac dl |
|---|---|---|---|---|
| 0.10 | 0.00000 | 0.00381 | 0.00 | 1.00 |
| 0.30 | 0.00000 | 0.01401 | 0.00 | 1.00 |
| 0.50 | 0.00000 | 0.02816 | 0.00 | 1.00 |
| 0.70 | 0.00113 | 0.03346 | 0.07 | 0.93 |
| 0.90 | 0.01922 | 0.03273 | 0.45 | 0.55 |

The over-blur arm `dh` is **dead in the lever's mid/low-τ region** (exactly 0 for
τ≤0.5) and only fires at τ=0.9 — the noisy-renoise end where the teacher's 1-step
x0 estimate is itself noisy (the trivial artifact, not the lever). The LF arm
`dl` — the exact one that killed item 2 — fires at 100% frac across low/mid τ.

## Verdict and why

**CLOSE.** Both the 2026-05 falsification (wrong distribution) and this null (right
distribution, σ-matched) point the same way: there is no on-trajectory over-blur
deficit for a `w_high` band lever to act on. The most likely mechanism is the one
the proposal pre-registered as the meaningful null:

> the diversity anchor may already have fixed the over-blur (it was introduced to
> de-collapse exactly the mode-seeking behavior the old band lever chased).

Site A's no-over-low directly supports that — the DP-DMD anchor (+ `dm_x0_norm`,
`docs/methods/turbo.md`) absorbed the over-blur the lever was born from
([[project_turbo_alpha4_overdistill]], [[project_turbo_dmd_x0_norm_wins]]).

## What survives

- `bench/turbo/probe_fei_band.py` — the reusable Phase-0 instrument (Site A
  σ-matched by construction; result envelope under
  `bench/turbo_fei_band/results/`). Re-run on a future checkpoint if the loop's
  diversity mechanism changes materially.
- The mechanism (DoG LP/HP split + bounded per-sample deficit) was always "sound,
  just mis-sited" — it is now also *un-needed*. Do not resurrect `ca_band.py`.

## Pointers

- `docs/findings/turbo_fei_band_deficit_falsified.md` — the original
  falsification + root cause (distribution + quantity mismatch).
- `_archive/proposals/turbo_fei_band_on_trajectory.md` — the revival design
  (archived 2026-06-20, closed by this finding).
- `bench/turbo/probe_fei_band.py`, run `20260620-1327-phase0_n30`.
