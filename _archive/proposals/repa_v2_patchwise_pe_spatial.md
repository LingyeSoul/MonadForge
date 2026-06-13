# REPA v2 — patchwise / relational alignment against PE-Spatial

Status: **IMPLEMENTED + Phase 0 CLOSED 2026-06-12 — archived.** Both arms
shipped (`library/training/repa.py`); the three-arm A/B landed on the GO
outcome: REPA decisively better than baseline, relational preferred ~6:4–7:3
over absolute on eyeballed grids. Live doc: `docs/experimental/repa.md`.
Phase-1 successor: `docs/proposal/repa_phase1_operating_point.md`.

Original text below, unedited.

Phase 0 is a three-arm short-run A/B with
pre-registered gates, designed so that two of the three outcomes **close this
line permanently** — that outcome is as valuable as the lever.

Premise sources: `_archive/repa/repa.py` (the v1 implementation this revives),
`docs/experimental/soft_tokens.md:246` (the autopsy line), commits `8581e3e`
(2026-04-30, integration) → `23a0125` (2026-05-19, removal).

## Autopsy of v1 — what actually failed, and what was never tried

v1 was REPA (Yu et al., arXiv:2410.06940) with three deviations from the paper,
bundled into one run and never separated:

1. **Globally pooled matching, not per-token.** Both sides mean-pooled to a
   single `[B, D]` vector before the cosine (`_archive/repa/repa.py:163-172`).
   The paper aligns per-token; v1 pooled because the grids don't match (DiT
   ~64×64 at 1024px vs PE-Core 24×24 at 336px) — and its own docstring marks
   per-token as the planned upgrade "if v0 shows traction" (`repa.py:13-18`).
2. **`repa_weight = 0.5` — the paper's *pretraining* default — applied to a
   LoRA fine-tune.** The config comment ("lower if it overpowers FM loss")
   shows the imbalance was suspected but never swept.
3. **PE-Core as the target encoder** — contrastive/global-tuned, the weakest
   of our PE variants at localized structure (the tagger hard-routes localized
   tags to PE-Spatial for exactly this reason).

Documented outcome (`docs/experimental/soft_tokens.md:246`):

> REPA helped anatomy on Anima but broke anime style (vision-encoder
> photo-prior leak).

So the signal was real (anatomy improved) but arrived bundled with a domain
pull that broke the product. The follow-up (Soft Tokens) removed the external
visual prior entirely — and with it, the anatomy gain.

## Hypothesis

The anatomy gain and the style break live in **different components of the
alignment target**, and v1's pooled-global form maximized exactly the wrong
one:

- A single global cosine target constrains the image *gestalt* — which is
  where "photo vs anime" lives in encoder space. Maximum style pressure.
- Per-patch targets supervise *local* content: what is at this location, part
  identity, boundaries. That is where the anatomy signal lives.
- **Relational (Gram) targets go one further**: matching the patch-affinity
  structure `G = F̂F̂ᵀ` instead of absolute features cancels any global domain
  direction out of the pairwise similarities by construction. What transfers
  is only *which patches are alike* — layout, part structure, anatomy — with
  the encoder's photo-prior geometry removed from the target.

Encoder switch PE-Core → PE-Spatial is the secondary lever: dense-prediction-
tuned patch tokens (B16-512 → 32×32 grid, d=768), with in-house evidence that
they resolve localized anime detail (tagger routing,
`library/captioning/anima_tagger.py:24-29`).

Expectation management: REPA's published wins are pretraining-from-scratch; on
a trained backbone this is a regularizer. v1 showed it adds *some* signal. The
bar is "anatomy gain at zero visible style cost", not a headline metric.

## Mechanics (both arms)

- **Capture**: v1's forward post-hook on `unet.blocks[repa_layer]` (default 8
  of 28), unchanged — captured from the **primary** forward, no second DiT
  forward (block-swap offloader desync is a second-forward problem; this
  doesn't create one).
- **DiT side**: `[B, T=1, H, W, D]` → `adaptive_avg_pool2d` over `(H, W)` to
  the PE grid `(h, w)` (64×64 → 32×32 is a clean 2×2; aspect buckets land on
  non-integer ratios, which adaptive pooling absorbs).
- **PE side**: cached PE-Spatial tokens `[T, d]` (CLS at 0) → drop CLS →
  unflatten to `(h, w)`. The cache writer must gain `grid_h`/`grid_w` metadata
  (today it stores only `encoder`/`d_enc`/`patch`,
  `library/preprocess/pe.py:121-125`) so the consumer doesn't re-derive the
  encoder bucket.
- **Arm A — absolute patchwise**: reuse `REPAHead` (`repa.py:57-78`) applied
  tokenwise, cosine per token in fp32, mean. Paper-faithful REPA at the right
  granularity. Keeps the head (+param group, +resume wart: head stripped from
  saved adapters, re-inits on warm start — v1 behavior, acceptable).
- **Arm B — relational (Gram)**: L2-normalize tokens on each side
  *independently*, `G = F̂F̂ᵀ ∈ [B, N, N]` (N=1024 → 1M entries, cheap), loss
  = `MSE(G_dit, G_pe)` in fp32. **No head at all** — similarities are computed
  within each space, so dimensions never need to match. No extra params, no
  resume wart, and the domain-offset cancellation above.
- **Weight**: 0.05 default (vs v1's 0.5) with a hard cutoff
  (`repa_anneal_steps`, default ~50% of run) — alignment matters early; the
  late-run gradient is where style drift compounds.
- **σ-conditioning**: paper-faithful — noisy-input block features aligned to
  clean-image encoder targets at every sampled σ.
- Loss attach: stage-2 scalar-broadcast in the registry
  (`library/training/losses.py:526-549`), same slot family as `fera_fecl`;
  v1's `MethodAdapter` skeleton (`extra_forwards` → `loss_aux`) carries over.

## Phase 0 — plumbing + three-arm short run

Plumbing (small, mostly exists):

- **PE-Spatial caching is already one command**: `--encoder pe_spatial` exists
  (`scripts/preprocess/cache_pe_encoder.py:91`), the writer is
  encoder-parameterized → sidecars land as
  `{stem}_anima_pe_spatial.safetensors`, disjoint from the PE-Core caches CMMD
  reads. Add the grid metadata (above) while touching it.
- Batch channel: v1 rode `batch['ip_features']` /
  `ip_features_cache_to_disk`, hardcoded to encoder `"pe"` — parameterize the
  sidecar suffix by encoder name.
- Resurrect `_archive/repa/repa.py` with the two loss forms; new network
  kwargs need `*_KWARG_FLAGS` registration in `networks/__init__.py` or they
  are silently inert and fail the config test.
- One-time check: confirm the block hook fires under `compile_blocks()`
  (hooks run in `nn.Module.__call__` outside the compiled graph; v1 predates
  some compile changes — verify once, don't assume).

Run: three arms at fixed seed/data/steps (~600 steps, standard dataset):
**baseline / Arm A w=0.05 / Arm B w=0.05**, block 8, cutoff off (short run).

### Gate (pre-registered — decide the read BEFORE looking)

Readouts: (1) **sample grids first** — style intact? anatomy (hands, limbs,
eye structure)? scalars are guides, images decide; (2) **CMMD against the
training set's own PE-Core features** (the live val signal) as the
domain-drift tripwire — v1's failure mode is exactly what CMMD-vs-own-data
measures; (3) **never** FM val loss (uninformative on Anima), **never**
fraction-of-Δ readouts (they reward no-ops).

- **Arm B breaks style too** → **CLOSE the line permanently.** If the
  domain-offset-cancelled form still leaks photo prior, an external visual
  prior cannot be made domain-safe at any granularity. Write the
  `docs/findings/` entry, stop re-proposing encoder-alignment aux losses;
  Soft Tokens remain the no-prior alternative.
- **Style holds in both, anatomy gain in neither** → **CLOSE.** v1's anatomy
  gain was inseparable from the global pull (it *was* the gestalt pressure) —
  the decomposition hypothesis is wrong. Finding entry, line retired.
- **Style holds + anatomy gain in either arm** → **Phase 1.** Prefer Arm B on
  a tie (no head, no resume wart, stronger invariance story).

## Phase 1 — operating-point sweep + full-length run (only on a GO)

- Sweep on the winning arm: weight {0.02, 0.05, 0.1} × cutoff {25%, 50%,
  none}; optionally layer {6, 8, 10} if the weight sweep is inconclusive.
- One full-length training run at the winner vs baseline, judged on grids +
  CMMD non-regression.
- Ship as off-by-default toggle in `configs/methods/lora.toml` + a
  `configs/gui-methods/` variant if it earns it.

## Hard guardrails (from the v1 burn + memory)

- **Do not re-run v1's operating point** (pooled + w=0.5). Phase 0 exists to
  separate the confounds v1 bundled.
- **No second DiT forward.** Capture from the primary forward only — the
  offloader desyncs on extra forwards under block swap.
- **Style gate is non-negotiable.** Any arm that visibly moves style fails
  regardless of its anatomy delta — that is the exact failure that killed v1.
- Compute Gram/cosine in fp32 (`repa.py:171`'s low-norm sensitivity note);
  caches are bf16.
- REPA gradient only reaches LoRA modules in blocks ≤ `repa_layer` (by
  design) — remember this when reading per-block deltas.

## Sequencing

```
Phase 0 plumbing (pe_spatial cache + grid metadata + resurrect adapter)
   └─ 3-arm short run (baseline / absolute-patchwise / relational)
        ├─ Arm B breaks style ────────► CLOSE: prior unsafe at any granularity
        ├─ no anatomy gain anywhere ──► CLOSE: gain was the gestalt pull itself
        └─ style holds + gain ────────► Phase 1 sweep → full run → ship toggle
```

## Contributing tier

Numerics-changing training feature → **Tier 1.5**: bench script
(`bench/repa_v2/`, `bench/_common.py` envelope, three-arm grids + CMMD in the
`result.json`) + invariant test (flag off ⇒ loss byte-identical to baseline).

## References

- `_archive/repa/repa.py` — v1 implementation; per-token upgrade pre-planned
  at lines 13-18; head + hook + adapter skeleton to resurrect.
- `docs/experimental/soft_tokens.md:246` — the autopsy line (anatomy ↑, anime
  style ✗, photo-prior leak).
- Commits `8581e3e` (integration 2026-04-30) / `23a0125` (removal 2026-05-19).
- `scripts/preprocess/cache_pe_encoder.py:91` (`--encoder` already exists),
  `library/preprocess/pe.py:121-125` (metadata to extend).
- `library/vision/encoders.py:219` (pe_spatial registry entry),
  `library/vision/buckets.py:67` (`PE_SPATIAL_B16_512_SPEC`).
- `library/captioning/anima_tagger.py:24-29` — in-house evidence PE-Spatial
  patch tokens resolve localized anime detail.
- `library/training/losses.py:526-564` (registry + stage rules),
  `library/training/cmmd.py` (the drift gate).
- REPA: Yu et al., arXiv:2410.06940. Relational form cf. relational knowledge
  distillation (Park et al., 2019) — match structure, not coordinates.
