# EasyControl × REPA — from wired to validated operating point

Status: **IMPLEMENTED & VALIDATED — relational REPA on EasyControl is
effective; archived on closure 2026-06-13.** Both blockers were fixed: the
`train.py` `extra_forwards` dispatch was moved out of the `crossattn_emb is
not None` branch (so the aux loss now dispatches on EasyControl's in-model
text path; see the docstring in `train.py::_run_extra_forwards`), and the
colorize PE-sidecar trap was repaired with a fallback chain in
`library/datasets/base.py::_repa_pe_sidecar_candidates` (try the TE-cache dir
first, then the subset latent-cache dir — repairs every `text_cache_dir` +
REPA combination, not just colorize). The first real A/B confirmed the
relational arm lands the structural-consistency signal as intended; REPA is
now the shipped EasyControl default for cond ≠ target tasks (sanitize /
near-twins, colorize). Live reference: `docs/experimental/easycontrol.md`
§"REPA auxiliary loss" and `docs/experimental/repa.md`. The plan below is the
historical operating-point roadmap, kept for the rationale.

## Why this is not just LoRA-REPA with a different config

The DiT is **frozen**. The block-8 hook captures the *target stream* (the
patched `Block.forward` returns `target_x_out` alone; `cond_x` rides side
channels), so the alignment gradient reaches the trainable cond LoRA **only
through the extended self-attention in blocks ≤ 8**. The term is therefore a
*conditioning-utilization pressure*: the only way to satisfy it is to pull
clean spatial structure out of the reference image. Three consequences:

1. **The LoRA-side operating numbers don't transfer automatically.** Weight
   0.05 ≈ 4% of loss, plateau by ~18%, ~uniform grad heatmap — all measured
   on a trainable-DiT path with direct gradients. The indirect path here may
   want a different weight, plateau elsewhere (runs are ~576 steps, not
   2292), and the lever-3 heatmap closure does **not** carry over.
2. **Task semantics decide the sign.** For cond ≠ target tasks (sanitize /
   near-twins, colorize) the pressure lands exactly where we want it —
   structural consistency with the reference. For ref == target subject
   control it rewards layout copying (already flagged in easycontrol.md) —
   out of scope here.
3. **Cond-dropped steps have no trainable path** (frozen target stream alone)
   — REPA runs should keep `drop_p` at or near zero.

## Known facts going in

- Sanitize has full sidecar coverage (96/96 PE-Spatial colocated with the TE
  caches) — PE loading works there (`repa/active = 1.0` end to end).
- The existing "pair" (`anima_easycontrol_sanitize` vs
  `anima_easycontrol_sanitize_repa{,_normed}`) is **baseline vs baseline**
  (dispatch bug above) — do not read it as a REPA A/B. Silver lining: the
  two "REPA" checkpoints are honest *seed/noise-variance controls* — any
  grid delta between them and the baseline bounds the run-to-run noise floor
  the real A/B has to clear.
- Runs are cheap: ~14 min for the 576-step sanitize recipe.
- LoRA-side reference points (for calibration, not assumption): weighted
  share ~4% at w=0.05; align plateau at 17–18% of the run; spatial_norm
  eyeball-validated ON; anneal candidate 0.25 untested.

## TRAP — silent no-op, and colorize is currently in it

Two layers:

1. **Generic**: missing sidecars ⇒ the adapter skips the term *silently*
   (per-step `return None`). The `repa/active` metric is the guard — **every
   EasyControl REPA run must show `repa/active = 1.0` from the first logged
   steps**; treat `active = 0` as a failed launch, not a soft degradation.
2. **Colorize-specific (blocking)**: `_try_load_repa_pe`
   (`library/datasets/base.py:1393`) resolves the sidecar **next to the TE
   cache**. Colorize redirects its TE cache via `text_cache_dir` (the
   color-only-caption mechanism), so the loader looks in
   `post_image_dataset/easycontrol/colorize/…` — which has **0** PE sidecars
   — even though **all 3058 colorize targets already have PE-Spatial
   sidecars** in the shared LoRA cache (`post_image_dataset/lora/`), because
   the color targets are reused from the main dataset. The staged
   colorize config (`use_repa=true` in `configs/easycontrol/colorize.toml`)
   would therefore train as a **silent baseline**.

   Phase-1 wiring item (small, test-covered): teach `_try_load_repa_pe` a
   fallback — try the TE-cache dir first (current behavior, preserves
   sanitize), then the latent-cache dir (`subsets[].cache_dir`) — OR write
   the PE pass into the colorize text cache. The fallback is the better fix:
   it repairs every future `text_cache_dir` + REPA combination, not just
   colorize, and costs no duplicate cache.

## Phase 0 — the first real sanitize A/B (one ~14-min run)

1. Rerun the REPA arm from the fixed tree (same config as
   `anima_easycontrol_sanitize_repa_normed`). **Launch sanity, in order**:
   `repa/align_loss` present in the progress jsonl from the first logged
   step (this is the metric whose absence caught the dispatch bug) and
   `repa/active = 1.0`.
2. `scripts/repa_progress_report.py` on its log — align curve decreasing
   then plateauing (flat-from-step-0 would mean the cond path can't move
   the target-stream Gram at all), weighted-share decile table for the
   Phase-2 weight calibration.
3. **Grids** vs the baseline checkpoint on the sanitize task, fixed
   seeds/prompts: edit-region structural consistency, identity preservation
   outside the edit region, and a conditioning-bypass check (does the output
   still track the reference, or did REPA teach it to hallucinate plausible
   structure?). Use the two inert "REPA" checkpoints as the seed-noise
   yardstick — a REPA win has to exceed their spread vs baseline.

Gate (pre-registered): REPA arm **equal-or-better structural consistency
with no identity/style regression, beyond the seed-noise spread** → proceed.
Visible regression → stop; first retry is weight *down* (0.02), not up — the
indirect path multiplies poorly understood factors and the LoRA experience
says the early-run share is double the steady-state share.

## Phase 1 — colorize (gated on Phase 0 + the sidecar fix)

Mechanism fit: the target is the **color** image; aligning target-stream
features to its PE-Spatial tokens presses the model to commit to globally
coherent structure pulled from the manga cond — the plausible wins are
region-coherent color (less bleed across line boundaries) and fewer
structure drifts on busy panels.

A/B: next colorize run ± the staged REPA flags (everything else fixed).
Gates: (1) `repa/active = 1.0` sanity, (2) grids — color fidelity to the
caption + region coherence + structure preservation vs the manga cond, (3)
CMMD-vs-PE-Core tripwire, (4) align curve sane (decreasing then plateau, not
flat-from-step-0 which would mean the cond path can't move it).

## Phase 2 — operating point (only on a task with Phase-0/1 signal)

In priority order, one lever per run (the LoRA Phase-1 discipline):

1. **Weight** {0.05 → 0.1} if the effect is directionally right but weak —
   the indirect gradient path plausibly needs more dose than the direct one;
   {→ 0.02} if anything drifts.
2. **Anneal** — run `scripts/repa_progress_report.py` on the Phase-0/1 logs
   first; if the align curve plateaus early (LoRA: ~18%), test the same
   fraction here. Note 576-step runs make the cutoff cheap to test.
3. **spatial_norm parity** — currently NOT in the EasyControl network_args
   (LoRA default is on). One A/B.
4. **Optional probe**: `repa_grad_heatmap=1` on any Phase-2 run — the
   gradient reaches the tokens through extended attention, so the
   concentration profile may differ from the LoRA family's ~uniform; if it
   concentrates (≳3×), loss-side token-subset becomes relevant *here* even
   though it closed on LoRA.

## Non-goals

- **ref == target subject control** — rewards layout copying; revisit only
  with a mechanism change (e.g. masked alignment), not a weight tweak.
- **Absolute arm** — `EasyControlNetwork` carries no `repa_head` by design;
  the factory raises on `repa_mode="absolute"`. Keep it that way.
- **High `drop_p` + REPA** — cond-dropped steps contribute nothing and
  dilute the term.

## References

- `docs/experimental/easycontrol.md` §"REPA auxiliary loss" — mechanism +
  the original wiring notes.
- `docs/experimental/repa.md` (§"Annealing plan" for live lever status + the
  metrics/report tooling); historical plan archived at
  `_archive/proposals/repa_phase1_operating_point.md`.
- `networks/methods/easycontrol.py:311–342` — factory stamping;
  `library/datasets/base.py:1393` — sidecar resolution (the colorize trap).
- `tests/test_repa.py::test_easycontrol_create_network_stamps_repa` and
  siblings — the wiring contract.
- Checkpoints: `output/ckpt/anima_easycontrol_sanitize{,_repa}.safetensors`
  (+ `.snapshot.toml`) — the existing Phase-0 pair.
