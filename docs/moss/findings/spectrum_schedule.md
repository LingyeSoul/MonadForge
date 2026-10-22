# MOSS Spectrum's full-step schedule: the growing window stays

Measured 2026-10-21 on MOSS-Image Base. Probe, runs and grids: `bench/moss_spectrum/`
(`probe_skip_cost.py`, `run_bench.py` with `full=` / `--example`; runs
`20261021-2300-skipcost-base5`, `20261021-2321-sched19`, `20261021-2329-space_ba`).
Follows `sigma_staircase.md`: given the layout knee at σ 0.87–0.80 (steps 22–28 of 50),
are Spectrum's full forwards spent in the right places?

## Setting

The shipped schedule (ws 2, fw 0.5, warmup 6, tail 3) runs 19 full forwards in 50
steps: steps 0–5, then 7, 9, 12, 15, 19, 23, 28, 33, 39, 45, then 47–49. Six full
forwards land at σ 1.0–0.978, where the pooled `x0_pred` is still a blur, and four land
in the layout band σ 0.95–0.80.

## Observations

**Open loop points the wrong way.** `probe_skip_cost.py` replays 250 schedules on
recorded p50 trajectories (5 prompts) and scores `Σ |Δσ_i|·‖v̂_i − v_i‖`.

- The per-step error concentrates below σ 0.5, because the shift-5 grid's Δσ is large
  there.
- Spacing the full forwards evenly in log-SNR halves the score at NFE 19 (0.093 against
  the shipped 0.121).
- NFE 16 matches the shipped NFE 19 on this score.
- Within the window family the ranking agrees with closed loop (fw 0.25 < 0.5 < 1.0).
  Across families it does not.

**Closed loop: the shipped window wins.** Five schedules were run as Euler-50 arms.

| arm | NFE | 5 prompts, 1024² rel-L2 | Space 004 / 017 rel-L2 | grids |
|---|---|---|---|---|
| shipped | 19 | **0.401** | **0.619 / 0.441** | closest to p50 everywhere |
| warmup 4, two forwards moved to steps 11/13 | 19 | — | 0.627 / 0.446 | same images as shipped |
| log-SNR spacing | 19 | 0.455 | 0.654 / 0.450 | close; manga bubble and a hand change |
| step-index spacing | 19 | 0.587 | 0.748 / 0.582 | poses and compositions change (fridge, sunset, 017 right character) |
| log-SNR spacing | 16 | 0.578 | 0.789 / 0.530 | fridge layout, 004 bottom centre and 017 Hina move |

The 5-prompt set is `run_bench.PROMPTS` at seed 42 and cfg 3. The Space set is
`project/moss_lora/space/` 004 and 017 at their own size, cfg and seed. The numbers are
mean latent rel-L2 against p50, which is a drift number. The grids are the verdict.

## Reading

- An error made above σ ≈ 0.87 grows into a composition change. A larger error made below
  σ 0.5 is absorbed as detail.
- The open-loop score assumes displacement errors carry through unchanged, so it
  weights the late steps and misses this.
- The shipped window is dense early and sparse late, which matches the staircase.
- Moving forwards out of warmup into the layout band changes nothing. The early
  forwards cost about the same wherever they fall before σ 0.92.
- Lowering the NFE to 16 costs layout. The speed-up stays at 2.6×.

## Not tested

- Reweighting the open-loop score by a measured per-σ propagation gain, from perturb
  and rollout runs.
- A capped forecaster history (`history`/K). At 2 MP the history costs about 1.7 GiB
  over p50 (12.5 GiB against 10.8 GiB), which fits 16 GB as it is.
