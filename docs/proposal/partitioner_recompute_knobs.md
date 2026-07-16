# Partitioner recompute knobs — freefit VRAM recovery without the budget knapsack

Status: **flags landed 2026-07-02 (opt-in), Phase 0 bench done; Phases 1–3
proposed.** Motivated by issue #58 (freefit raised 1536² no-grad-ckpt training
from ~23 GB to ~29–30 GB on a 4090; `activation_memory_budget=0.55` restored
the footprint but cost +35% wall time). Mechanism background:
`docs/findings/custom_autograd_removal_partitioner_oom.md`.

## The problem this fixes

The 2026-06-10 custom-autograd removal deleted the last human-chosen
save-for-backward boundary from the compiled block graph, and the AOT min-cut
partitioner's default partition saves more intermediates than the old
`LoRADownProjectFn` did. The sanctioned mitigation, `activation_memory_budget`,
is a knapsack *on top of* that default partition — and its cheap-recompute pool
(casts, scale-folds, pointwise) exhausts somewhere around 0.85. Below that it
starts recomputing GEMMs/attention, which is exactly the +35% the issue-58
reporter measured at 0.55. So the budget is the wrong lever for large-token
tiers: the deeper you need it, the more it costs.

Two levers that act *before* the knapsack instead:

1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (now the default on
   Linux, `library/runtime/allocator.py`, opt-out `ANIMA_EXPANDABLE_SEGMENTS=0`).
   Freefit hands the allocator a different seq_len every step — the old
   constant-token pool allocated identical sizes forever, freefit fragments the
   reserved pool. Allocator-only; numerics-inert by construction.
2. **Partitioner heuristic flags** (opt-in CLI/TOML knobs, applied in
   `library/runtime/harness.py::_apply_partitioner_tuning` before
   `compile_blocks`, auto-skipped under grad-ckpt like the budget):
   - `partitioner_recompute_views` → `torch._functorch.config.recompute_views`.
     Views are metadata (recompute ≈ free) but *saving* one pins its base
     tensor alive. Theoretically free VRAM.
   - `partitioner_aggressive_recomputation` →
     `torch._functorch.config.aggressive_recomputation`. Drops the min-cut
     ban list on whole op classes. **Not** theoretically free — the bans exist
     because some recomputes are slow.

Both are plain module attrs (no ContextVar revert — same class as the budget,
see [[project_dynamo_limit_contextvar]] for the trap they avoid).

## Phase 0 — done (bench/freefit_vram, 2026-07-02)

Default `make lora` job, 1024 tier (4032–4200 tok), batch 1, no grad-ckpt,
cold compile every arm (`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` so a cached
partition can never replay across arms — [[project_compile_cache_guard_poisoning]]).
Runs: `results/20260702-1447-issue58-levers` (control) +
`results/20260702-1451-issue58-levers-rest` (16 GB 5070 Ti; peaks are absolute
nvidia-smi — ignore `peak_delta_mib` in the second run, the dying previous arm
polluted the idle baseline).

| arm | budget | levers | outcome | peak | s/it |
|---|---|---|---|---|---|
| b100_base | 1.0 | — | **OOM step 0** | >15.8 GB | — |
| b100_exp | 1.0 | expandable | **OOM step 0** | >15.8 GB | — |
| b100_flags | 1.0 | both flags | ok | **11.6 GB** | 0.647 |
| b100_exp_flags | 1.0 | both | ok | **11.5 GB** | 0.649 |
| ship_base | 0.99 | — | ok | 13.7 GB | **0.585** |
| ship_exp_flags | 0.99 | both | ok | **11.2 GB** | 0.669 |

What Phase 0 settles:

- **The flags are the real lever.** At budget=1.0 they turn a hard OOM into an
  11.6 GB run — ≥4.2 GB recovered, more than the knapsack's whole free region.
  With flags on, budget 0.99-vs-1.0 barely matters (11.2 ≈ 11.5 GB): the
  saved-set reduction happens before the knapsack ever bites.
- **expandable_segments does not rescue step-0 pressure** (OOM is
  saved-activation volume, not fragmentation). Its target is reserved-pool
  growth over long runs; 150 steps was too short to isolate that term — the
  per-second VRAM CSVs are in the run dirs for a longer look. Keep it as
  default (it cannot hurt numerics and costs nothing measurable) but don't
  claim the fragmentation win until a multi-hour run shows it.
- **The bundled flags cost ~10–14% step time** (0.585 → 0.65–0.67 s/it at
  matched budget). Still far better than the budget-0.55 alternative (+35%)
  per GB recovered, but not free — and Phase 0 cannot say which flag carries
  the cost vs. which carries the saving, because it only ran them bundled.

## Phase 1 — flag separation: DONE 2026-07-02, the saving is ALL aggressive

Run `results/20260702-1510-issue58-views-split` (drain-wait between arms, so
peaks/idles are clean):

| arm | budget | outcome | peak | s/it |
|---|---|---|---|---|
| b100_views | 1.0 | **OOM step 0** | >15.8 GB | — |
| ship_views | 0.99 | ok | 13.66 GB | 0.581 |
| ship_aggr | 0.99 | ok | **11.41 GB** | 0.659 |

- **`recompute_views` is exactly inert on this workload** — identical peak
  (13.66 vs 13.66 GB) and identical speed (0.581 vs 0.585) to ship_base, and
  it does not rescue budget=1.0. The block graph evidently saves few
  standalone views. Harmless, but pointless → **not promoted to default**;
  the flag stays for other torch versions/workloads where it may bite.
- **`aggressive_recomputation` carries the whole effect**: −2.25 GB at +12.6%
  step time, reproducing the bundled Phase-0 arm by itself (11.4 ≈ 11.2 GB,
  0.659 ≈ 0.669 s/it). This is a genuine memory↔time tradeoff point, not a
  free lunch — but a far better exchange rate than the deep budget knapsack
  (issue-58's budget=0.55 cost +35% for a similar-scale recovery).

Per the pre-registered decision rule: **nothing becomes a training default**;
`--partitioner_aggressive_recomputation` is the documented issue-58 remedy
(the views flag drops out of the recommendation), and Phases 2–3 proceed with
aggressive only.

## Phase 2 — EasyControl: raise the budget from 0.3, pay with flags

`configs/easycontrol/easycontrol.toml` ships `activation_memory_budget = 0.3`
(no recorded rationale) — deep inside the knapsack's GEMM-recompute region, so
EasyControl training is very likely paying a large hidden recompute tax. The
flags reach the cond stream for free: `_apply_partitioner_tuning` sets
process-global functorch config before *both* `compile_blocks` and
`EasyControlNetwork.compile_cond_stream()` in `compile_blocks_for_training`
([[project_easycontrol_cond_path_eager_compile]]). Wiring is also free: the
knobs are argparse dests, so the method TOML can set them exactly like
`chimera.toml` sets `activation_memory_budget`.

Arms (easycontrol job, standard sanitize/colorize dataset, no grad-ckpt):

1. `budget=0.3` bare — shipping baseline (peak + s/it).
2. `budget=0.99` + `partitioner_aggressive_recomputation` (Phase-1 winner).
3. `budget=0.99` bare — control, expected OOM or near-OOM; quantifies how much
   the 0.3 was actually buying.

Adopt (2) into `easycontrol.toml` iff peak ≤ baseline + 0.5 GB **and** s/it ≤
baseline. Expected outcome: same-or-less VRAM *and faster*, since arm 1 is
recomputing expensive ops today. Watch REPA: it's on by default for
cond≠target tasks ([[project_easycontrol_repa_validated]]) and adds its own
activation traffic — keep it on in all arms so the comparison is the shipped
recipe.

## Phase 3 — 1536² / issue-58 validation

We cannot run 8640+ tokens locally (16 GB). Two routes, either suffices:

- **Analytic**: compile a *single* block at seq=8836 with
  `torch._functorch.config.debug_partitioner`, read saved-bytes with and
  without the flags, ×28 blocks. One block fits easily in 16 GB.
- **Community**: reply on issue #58 offering the flag
  (`--partitioner_aggressive_recomputation` at `activation_memory_budget=1.0`)
  against the reporter's budget-0.55 baseline (23 GB / 8h45). Saved-set
  reductions scale ~linearly with tokens, so the Phase-0/1 ≥4.2 GB at 4200 tok
  projects to ~8 GB at 8836 tok → ~22 GB peak, fitting the 4090 at ~+13% time
  instead of +35%.

## Kill criteria

- Views-only saves <1 GB *and* aggressive costs >15% at 1536² scale → the
  flags stay documented-but-opt-in and the issue-58 answer becomes "grad-ckpt
  or budget, pick your tax" (plus a possible custom-autograd resurrection as a
  pure memory boundary — the finding doc's road-not-taken).
- expandable_segments shows any reproducible step-time cost → flip the default
  off (`ANIMA_EXPANDABLE_SEGMENTS=0` semantics inverted); keep it documented
  for long runs.

## Landed surface (2026-07-02)

- `library/runtime/allocator.py` — expandable-segments default (train.py
  pre-torch prologue; Linux-only, respects user conf, opt-out env).
- `--partitioner_recompute_views` / `--partitioner_aggressive_recomputation`
  (`library/config/cli_args.py`), applied via
  `harness.py::_apply_partitioner_tuning`, plumbed in `train.py`.
- `bench/freefit_vram/run_bench.py` — 6-arm A/B harness (per-arm log +
  1 s VRAM timeseries CSV), `tests/test_partitioner_tuning.py` (gating).
- Bespoke loops (turbo/spd/mod) do **not** get the flags yet — mirror via the
  usual pattern ([[project_daemon_wiring_pattern]]) if Phase 1 promotes one.
