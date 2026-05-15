# ChimeraHydra — Dual-pool additive routing for timestep-aware MoE

> **2026-05-15 erratum.** ChimeraHydra moved from one-A-per-Linear (this
> proposal's original framing) to **two A's per Linear** — one A per pool,
> shared within that pool's experts. The chimera now is literally two
> HydraLoRAs (Tian et al., NeurIPS'24, [arXiv:2404.19245](https://arxiv.org/abs/2404.19245))
> glued at the residual: content half routed by per-Linear lx-router,
> freq half routed by network-level FEI router (FeRA shape,
> [arXiv:2511.17979](https://arxiv.org/abs/2511.17979)), with TimeStep Master
> ([arXiv:2503.07416](https://arxiv.org/abs/2503.07416)) asymmetric per-pool
> timestep treatment (freq always full-rank, content T-LoRA-masked). The
> SVD partition now applies on **both** sides — pool A's project from
> disjoint right-singular subspaces, pool B's project to disjoint
> left-singular subspaces — strictly stronger orthogonality than the
> single-A version. See `docs/experimental/chimera-hydra.md` for the
> live design. The text below preserves the original single-A motivation
> for archival reasons.

A single-phase MoE-LoRA recipe for Anima's ortho-hydra setup. Two pools
of B-heads share one A per adapted Linear — a **content** pool and a
**frequency** pool (the two heads of the chimera). Each pool is routed by
a small router whose **input** structurally owns one axis: content
(pooled `lx` / text) or frequency-stage (FEI + logSNR + t_emb). Pool
outputs are added. No staging, no σ-band overlap mask, no multiplicative
gate.

> Supersedes the earlier **staged 2D** design (multiplicative gate +
> Phase 1/2/3 curriculum), preserved in git history at the prior path
> `docs/proposal/staged_2d_experts.md`. Staging existed to break
> gradient confounding in the multiplicative gate `g_c ⊙ g_t`. Additive
> composition removes the confounding directly, so the curriculum is
> solving a problem ChimeraHydra doesn't have.

## TL;DR

```
shared A per adapted Linear

content pool:   B_content[0 .. K_c-1]    routed by π_c
freq pool:      B_freq   [0 .. K_f-1]    routed by π_f

π_c = content_router(pooled lx, rank-R)              # existing per-Linear
π_f = freq_router  (FEI(z_t), logSNR, t_emb)         # new, one per network

Δy = Σ_c π_c[c] · B_content[c] (A x)
   + Σ_f π_f[f] · B_freq   [f] (A x)

E_total = K_c + K_f       # default K_c = K_f = 3 (parity with today's E=6)
```

OrthoHydra's `P_basis_k` slicing of A's SVD is split by name: the first
K_c slices go to content B-heads, the last K_f to freq B-heads. Each pool
gets its own subspace of A; A itself stays a generic shared basis.

## Why additive over multiplicative

|  | Multiplicative `g_c ⊙ g_t` | **Additive (this proposal)** |
|---|---|---|
| Gradient ownership | Both routers shape every gate slot; staging needed to avoid co-training confounding | Each B-head owned by exactly one router; gradients decouple at the source |
| Specialization driver | Learned partition of router input | **Enforced by router-input separation** — content router can't see σ; freq router can't see text features |
| Veto risk | `π_c[k]·π_f[k]` can starve an expert the content router preferred | None — pools are disjoint |
| Phases needed | 2–3 (cold-start → calibration → optional joint) | **1** (single co-train) |
| Cold-start surface | Symmetry-breaking on shared E experts | Two routers risk one-pool collapse (mitigated below) |

The "two routers fighting for the same gradient" worry that motivated
staging is a multiplicative-composition pathology. With additive output
composition, π_c only routes B_content; π_f only routes B_freq. They
never touch the same slot.

## Why HydraLoRA's auto-specialization argument gets stronger

In single-router Hydra, B-heads differentiate because one router *learns*
a partition of its input space — that's the cold-start surface
`expert_warmup_ratio` / per-expert init tries to break
([[project_hydra_init_std_dead_code]]).

Here, specialization is **enforced by input separation**, not router
learning:

- The content router cannot see σ or FEI; it physically *cannot* become
  the time router.
- The freq router cannot see pooled text features; it physically *cannot*
  become the content router.

Each pool's B-heads necessarily specialize along its router's available
axis. The auto-specialization claim is structural, not just empirical.

## What's in the repo, what's new

| Need | Status |
|---|---|
| Shared-A + per-expert orthogonal slicing of A's SVD | **Exists.** `OrthoHydraLoRAExpModule` (`networks/lora_modules/ortho.py`). |
| Per-Linear rank-R content router on `lx` | **Exists.** `_compute_gate` in `networks/lora_modules/hydra.py`. |
| σ broadcast to every adapted module | **Exists.** `set_sigma` path. Aliasing bug fixed ([[project_set_sigma_aliasing_bug]]). |
| FEI(z_t) features | **Exists.** `router_source="fei"` in `configs/methods/lora.toml`. |
| Switch / balance loss | **Exists.** Currently one term; extend to two named pools. |
| **Two named B-head pools per Linear with OrthoHydra slice allocation** | **New, medium.** Split `lora_ups` / `P_bases` into `content` and `freq` groups in `networks/lora_anima/factory.py`. ~80 lines. |
| **`FreqRouter` (one per network): `Linear(FEI ⊕ logSNR ⊕ t_emb → H) → SiLU → Linear(H → K_f) → softmax`** | **New, small.** `networks/lora_modules/freq_router.py`, ~50 lines. |
| **Two-branch composition in `_apply_expert_mixture`** | **New, small.** `Δy = Σ π_c B_c (Ax) + Σ π_f B_f (Ax)`. ~20 lines. |
| **Per-pool balance loss** | **New, small.** Two named coefficients in the training loop. ~15 lines. |
| **Save/load `freq_router.*` + named pool keys** | **New, small.** `loading.py` + `save_weights`. ~30 lines. |
| **ComfyUI mirror** | **New, small.** `custom_nodes/comfyui-hydralora/` already plumbs σ; add FEI + freq router branch. ~60 lines. |

Total new code: **~250 lines** plus tests. Tier 1.5 (efficiency/numerics
revision + bench) per `CONTRIBUTING.md`.

## Architecture

### Per-Linear adapter

For each adapted Linear with shared A ∈ ℝ^{r × d_in}:

```
h = A x ∈ ℝ^r                                     # one shared projection

# content branch  (T-LoRA mask optional, see below)
h_c[c]   = P_basis_content[c] · (mask_c(σ) ⊙ h)   # OrthoHydra slice, content half
π_c      = content_router(pool(h))                # existing rank-R, (B, K_c)
Δy_c     = Σ_c π_c[c] · B_content[c] · h_c[c]

# freq branch  (unmasked)
h_f[f]   = P_basis_freq[f] · h                    # OrthoHydra slice, freq half
π_f      = freq_router(FEI(z_t), logSNR, t_emb)   # (B, K_f), broadcast across Linears
Δy_f     = Σ_f π_f[f] · B_freq[f] · h_f[f]

Δy = Δy_c + Δy_f
```

### Routers

**Content router** — unchanged. Per-Linear rank-R router on pooled `lx`.
Already validated (2026-04-20 rewiring).

**Freq router** — new, **one per network**:

```
input  = concat([FEI(z_t), logSNR(σ), t_emb(σ), φ(σ)])   # ~F = 32
hidden = SiLU(Linear(F → H))                              # H = 32
logits = Linear(H → K_f)
π_f    = softmax(logits / τ_f)                            # (B, K_f)
```

~F·H + H·K_f ≈ 2–4k params at H=32, K_f=3–6. Broadcast across every
adapted Linear — σ/FEI/t_emb carry no per-Linear signal, so per-Linear
freq routers would E× the param count for nothing.

**Init:** bias = 0, weights ~ N(0, 0.1). Output is near-uniform but **not
at uniform** at step 0 — the freq router immediately differentiates as
FEI/σ vary across the batch. Zero-weight init would be a fixed point and
the freq pool would never start learning.

### OrthoHydra slice allocation

A's SVD column space is split by name into two disjoint subspaces:

```
P_basis_content[c] = V[:, slice_content_c]       # c = 0..K_c-1
P_basis_freq   [f] = V[:, slice_freq_f]          # f = 0..K_f-1
```

Each pool gets its own subspace of A. A itself never has to compromise
between content-style and frequency-band specialization — both pools
project through A but downstream live in disjoint slices.

Constraint: K_c + K_f orthogonal slices must fit inside `network_dim`. At
default `network_dim = 32` and E=6, each slice is width 5; at E=12 it's
width 2 (likely too narrow). Practical ceiling: **K_c + K_f ≤
network_dim / 4**.

### Balance loss

Two **independent** Switch losses, one per pool:

```
L_balance = w_c · switch_loss(π_c) + w_f · switch_loss(π_f)
```

A single combined balance term would let the optimizer satisfy the
constraint by collapsing one pool to uniform and concentrating the other —
meeting the balance objective while killing the dual-pool design.
Per-pool balance forces each pool to spread independently.

`w_c` keeps the current ortho-hydra value (~2e-5,
[[project_hydra_balance_weight_ceiling]]); `w_f` starts at the same
value, tunable.

### T-LoRA integration (free composition)

T-LoRA's `use_timestep_mask` already exists. Apply it to the **content
branch only**:

```
h_c[c] = P_basis_content[c] · (mask_c(σ) ⊙ h)   # rank-mask at high σ
h_f[f] = P_basis_freq[f]    · h                  # freq branch full-rank
```

Rationale: T-LoRA's whole argument is that high-σ steps are where LoRA
memorizes layout/identity. The content branch is exactly that risk
surface. The freq branch **wants** full rank at high σ to learn
coarse-stage denoising features. Per-branch masking falls out cleanly
because the pools are physically separate — no per-expert mask
bookkeeping.

## Cold-start risk and mitigations

Two routers init random ⇒ risk one pool dominates the denoise loss while
the other settles at uniform (a local minimum).

1. **Per-pool balance loss** (above).
2. **Non-zero freq router init** — output near-uniform but differentiates
   immediately as input varies.
3. **Live diagnostic.** First 1k steps, log per-pool
   `Σ ||π[k] − 1/K||²`. If freq pool stays < 1e-3 (flat-uniform) while
   content pool diverges, raise `w_f` or add a FEI warm-start residual.

Persistent freq-pool flatness after warmup ⇒ the freq router has no
signal the content router didn't already capture via `lx`-σ correlation.
That's risk §1 below, not a training bug.

## Training — single phase

```bash
# new GUI variant, one-line to add:
make lora-gui GUI_PRESETS=dual_pool_2d
# or
python train.py --method lora --preset default \
                --dual_pool --K_c 3 --K_f 3
```

No freezing, no LR schedule changes beyond what ortho-hydra already
does. Both pools, both routers, and shared A trained jointly under the
denoise loss + per-pool balance loss.

**Optional Stage B** (reactive, only if the joint run shows persistent
gate instability): freeze B-heads + A; train only routers at LR × 3 for
~0.5 epoch. Same idea as the staged proposal's Phase 2, but **reactive,
not prophylactic** — most runs shouldn't need it.

## Bench plan — `bench/dual_pool_2d/`

Standard envelope via `bench/_common.py`.

| Run | What | Wall-clock |
|---|---|---|
| A | Stock ortho-hydra (`hydralora_experimental`), 12 epochs | 1× |
| B | Ortho-hydra + `router_source="sigma"`, 12 epochs | 1× |
| **C** | **Dual-pool additive**, K_c=K_f=3, 12 epochs | 1.05× |
| C+T | C with T-LoRA mask on content pool | 1.05× |
| C-split | Dual-pool with K_c=4, K_f=2 (more content capacity) | 1.05× |
| C-fei | C but content router *also* fed FEI (falsification: does freq pool add unique signal?) | 1.05× |

A vs. C = value of dual-pool over single-router. C vs. B = dual-pool vs.
σ-only routing. **C-fei is the key falsification:** if it matches C, the
freq pool is redundant and the design should be archived.

**Metrics:**

- CMMD on validation ([[project_cmmd_val_signal]]) — primary live signal.
- FM val loss — necessary baseline, not sufficient
  ([[project_fm_val_loss_uninformative]]).
- Sample quality on 16-prompt × 3-seed grid (DCW eval harness).
- **Pool-divergence diagnostics:**
  - Per-pool gate entropy (median across Linears).
  - `||π − 1/K||²` per pool — collapse detector.
  - **Freq-gate variance across σ buckets** at inference. Floor > 0.01:
    freq router is using σ. Below floor ⇒ freq pool is dead weight.
  - Per-expert usage histogram (mean-gate, per
    [[project_fera_expert_usage_mean_gates]]).
- **Branch contribution norms:** `||Δy_c||` vs. `||Δy_f||` per Linear,
  averaged over the dataset. Healthy ratio ~0.3–3.0; out-of-range = one
  pool dominating.

## Decision tree

```
C > A on CMMD + sample quality?
├── yes, C > B too        → ship C as new default; T-LoRA composes if C+T > C
├── yes, C ≈ B            → orthogonal to σ-router; ship as alternative
└── no:
    ├── freq pool flat (||π_f − 1/K|| → 0)
    │   ├── C-fei ≈ A     → freq signal absorbed by content branch; archive
    │   └── C-fei > A     → freq signal exists but freq router can't access it;
    │                        redesign freq router input
    └── freq pool active but C < A
        ├── ||Δy_c||/||Δy_f|| extreme   → one pool dominating; raise the other w
        └── ratio healthy               → additive can't reach the multiplicative
                                          regime; revisit staged 2D as fallback
```

## Risks

1. **Freq router degeneracy.** The content router already gets
   σ-implicit signal via `lx` (which depends on `x_t` which depends on
   σ). If that's sufficient, the freq router learns nothing the content
   router can't. **C-fei is the falsification check.** Mitigation: feed
   the freq router signals genuinely *unavailable* to `lx` — explicit
   logSNR scalar, t_emb, FEI bands.
2. **One-pool collapse.** Per-pool balance + non-zero freq init are the
   structural mitigations. Live diagnostic at first 1k steps detects it.
3. **Shared-A two-masters (mitigated, monitor).** OrthoHydra slice
   allocation gives each pool its own subspace, but A is still optimized
   once and its gradients come from both pools. If they fight over A's
   update direction, both could degrade. Monitor
   `||grad_A_from_content||` vs. `||grad_A_from_freq||` early in
   training; if one dominates ≥ 10× the dominant pool is effectively
   training A alone.
4. **Balance loss tuning doubled.** Two coefficients instead of one.
   Start with both at the proven content value; only sweep if collapse
   appears.
5. **FEI signal weakness on Anima.** [[project_fera_probe_2band_decision]]
   found FEI probes collapsed to 2 bands at the σ_mid cell — FEI doesn't
   carry strong mid-band signal at the Anima latent distribution. K_f=3
   may effectively collapse to K_f=2. Start K_f=3; let the mid expert
   collapse via usage if it's genuinely empty.
6. **OrthoHydra slice exhaustion.** K_c + K_f ≤ network_dim / 4 in
   practice. At E=6 and `network_dim = 32`, slices are width 5 — fine.
7. **Inference compute ~5–10% higher.** Two B-head passes per Linear vs.
   one. Negligible.

## What this proposal is not

- **Not a new adapter family.** Tier 1.5 — efficiency/numerics revision
  on the existing OrthoHydra path.
- **Not a replacement for σ-router or σ-band partition.** Those remain
  as single-router alternatives. C vs. B decides which is recommended by
  default.
- **Not staged.** Single phase. (Staged 2D — multiplicative gate, Phase
  1/2/3 — lives in git history at this path.)
- **Not coupled to multiplicative gate composition.** Additive output
  composition is the whole point — it's what makes single-phase
  co-training safe.
- **Not a curriculum / slicing approach.** No dataset subsetting, no
  per-cluster expert fostering. HydraLoRA's auto-specialization plus
  input-separated routers does the work.

## Open questions

1. **K_c / K_f split.** Default K_c=K_f=3 (parity with today's E=6).
   K_c=4, K_f=2 (more content) or 2,4 (more freq) sweep in C-split.
2. **Content router input.** Keep current pooled-`lx` rank-R router, or
   switch to pooled text features? `lx` is task-adaptive but σ-contaminated
   (which is fine — content router *should* be allowed to use σ
   indirectly; the dual-pool design only requires the freq router add
   signal `lx` doesn't already give). Stay with `lx`; ablate text as
   C-text if needed.
3. **Freq router scope.** One per network (proposed) vs. per-block.
   Per-block adds `n_blocks × ~3k` params and may capture
   block-depth × σ specialization (early blocks structure, late blocks
   detail). Cheap to ablate.
4. **T-LoRA mask schedule.** T-LoRA's schedule was tuned for single-image
   personalization; under Anima's broad training distribution the optimal
   high-σ rank may be higher. Sweep inside C+T.
5. **Composition with DCW v4.** Sampler-boundary correction conditioned
   on (aspect, prompt, g_obs). Dual-pool is a per-step expert
   reweighting inside the DiT. Likely orthogonal; bench check at C
   whether optimal DCW λ changes.
6. **REPA / VR-loss interaction.** Both modify what shared A learns.
   Co-train compat unknown; sanity-check at bench.
