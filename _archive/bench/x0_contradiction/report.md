# x0-contradiction — measuring "in-step contradiction" in Anima t2i

> **ARCHIVED 2026-07-12** — line closed: hypothesis reversed (wander = scene
> complexity, base-owned, ⊥ breakage; F1–F3 closed the wander→breakage caveat
> with the SAM3 finger detector). Map entry: `_archive/shelved_benches.md`;
> memory: `project_x0_contradiction_bench`. The scripts' `parents[2]` bootstrap
> assumed `bench/` depth — rerun from repo root with `PYTHONPATH=.`.

**Question (user).** Broken regions in t2i (hands, weird objects) seem to track
*under-specified captions*: the model "knows" more detail is needed than the
caption gives (tag dropout), tries to draw it, and **different denoising steps
disagree** — step 10 draws a thing, step 11 erases it. Is this "in-step
contradiction" real and measurable? Is it the LoRA's fault? Which way does
`caption_dropout_rate` push it?

**Model under test.** `output/ckpt/anima_sincos.safetensors` — rank-32 LoRA,
`use_ortho_init`, REPA-DoG, `caption_dropout_rate=0.1`, trained on one artist
(`sincos/*`, 668 images). Base = `anima-base-v1.0`.

---

## TL;DR

1. **The contradiction is mostly base-owned; the LoRA adds a ~10% margin in
   early/mid σ.** Tail-only the LoRA looks inert (×1.005), but **full-trajectory
   it amplifies wander ×1.08–1.12** (more on its own-artist captions) — the base
   still owns ~90%. Most likely benign: the LoRA adds stylistic *detail* → busier
   scene → more wander (Finding 5: wander = complexity), not "destabilizes."
2. **We were looking at the wrong σ region.** The σ<0.45 "resolve tail" is only
   **4–5% of all x̂₀ motion** — calm refinement, a non-issue. The real wander is
   **early/mid σ (the mode-selection phase): ~95% of the motion.**
3. **Wander tracks scene *complexity/richness*, not under-determination —
   and this *flips* the original hypothesis.** Cross-prompt: floor ≈1.8 (cube,
   *simple* 1girl ≈1.7 — so it's **not** "people are hard"), dense real captions
   4.9–6.2 (2.6–3.4× floor). Within one image intent, stripping a caption
   *lowers* wander monotonically (6.1 → 5.0) and **empty crashes to 1.7 (floor)**.
   So *removing* detail *reduces* contradiction — the opposite of "missing detail
   → steps disagree." Wander measures how busy a scene is, not how unspecified.
   (Per-patch path/net, so >1 is genuine non-monotonicity, not spatial heterogeneity.)
4. **The one visible artifact we dissected (awkward eyes) was a *committed bad
   mode*, not a contradiction** — the x̂₀ filmstrip shows the eyes snap to their
   final (awkward) shape early and hold, no wobble. That's the **overfit/LoRA
   axis**, which `caption_dropout` / rank / data / REPA *do* move.

**Two distinct failure axes — don't conflate them:**

| Failure | Signature | Owned by | Lever |
|---|---|---|---|
| **Contradiction** (x̂₀ backtracks) | wander 4.9–6.2 (early σ), per-patch path≫net | **~90% base** field geometry; LoRA adds ~10% (early/mid) | base-level: full-finetune, or inference-time (CFG / sampler / Restart / DAVE). LoRA levers only reach its ~10% margin. |
| **Overfit / committed-bad-mode** | awkward but *stable* feature (eyes), cross-seed inconsistency | the **LoRA** | LoRA-level: REPA, caption dropout, rank, more data |

> **REPA correction.** REPA is a *LoRA-training representation regularizer* — its
> gradients flow only into the rank-r delta, so it lives on the **overfit/destination**
> axis (better committed modes), **not** the contradiction/path axis. It cannot
> reshape the frozen base's field curvature. (An earlier draft mis-filed REPA as a
> base-level contradiction lever — wrong.)

---

## Follow-up (2026-06-21): closing the wander→breakage caveat with a real detector

The original report ends on a caveat: *"We did not establish that early wander
causes specific visible breakages — the one artifact dissected (eyes) was a
different mechanism."* Three follow-up probes close it. Net: **the LoRA's extra
wander is not real degradation, and wander is not the breakage axis** — the n=1
eyes dissection now holds on n=30 with a validated pixel-space detector.

### F1. The "benign detail" story holds for *motion*, not *contradiction*

`wander_is_it_real.py` decomposes the LoRA's +10% (TL;DR #1) by spatial
coincidence at fixed caption (Δ = lora−base, per patch, seed-avg):

| signal | corr with Δ early-path | corr with Δ backtrack |
|---|---|---|
| Δ detail (Sobel HF energy of final) | **+0.19** (consistent, all 4 prompts) | −0.09 (≈0) |
| lora cross-seed var (multimodal/breakage-prone) | — | −0.02 (≈0) |

So TL;DR #1's "busier scene buys the wander" is earned by the **monotone motion**
(extra path ↔ added detail, +0.19), **not** by the **backtracking** the report
actually calls *contradiction* (≈0 with both detail and seed-lottery regions —
it's diffuse, structureless, low-amplitude). Correction to TL;DR #1: *the +10%
splits — path tracks detail; the contradiction component tracks nothing
detectable.*

### F2. SAM3 finger-count detector — no degradation, wander ⊥ breakage

OpenPose is useless on anime (0 poses — photo-trained). **SAM3 is anime-robust**
and text-promptable: `"hand"`/`"finger"` → instance masks+scores; finger-count is
THE canonical breakage (6-finger hand). `breakage = Σ_hand max(0, n_fingers−5)`
(excess-only — a fist <5 is occlusion, not breakage). **Detector validated by
eye**: 0-excess images are clean 5-finger hands; the 8-excess outlier is a
visibly broken multi-finger grip on a mug. Paired base-vs-LoRA, same prompt+seed
(`wander_vs_breakage.py`, 6 hand prompts × 5 seeds = 30 pairs):

| | base | lora |
|---|--:|--:|
| **A. excess fingers / img** | 0.90 | 0.70 |
| total fingers / img | 6.50 | 6.23 |
| pairs LoRA *worse* / *better* | — | **8 / 7** (symmetric coin-flip; median excess 0 both) |
| **B. hand-region wander enrichment** | 1.21× | 1.20× (identical) |
| B. corr(hand-region wander, excess) | — | **−0.06** (null) |

**A: the LoRA does not break more hands than base.** Breakage is real (~30% of
hand prompts carry some excess) but **base-owned**. **B: wander is not the
mechanism** — hands carry mild wander enrichment but *identically* for base and
LoRA, and hand-region wander doesn't predict excess fingers. This is exactly the
two-axis table on real statistics: breakage = overfit axis, not wander/path axis.

### F3. A protective hint — robust but observational, and *not* causally testable here

The paired link `corr(Δ hand-region wander, Δ excess)` is **−0.35** (Pearson,
p=0.06), strengthening to **−0.47** (p=0.01) dropping the one excess-8 outlier,
Spearman −0.38 (p=0.04) — i.e. where the LoRA adds hand-region wander it adds
*fewer* broken fingers. Mechanistically consistent with the report's filmstrip
(committed bad modes *snap early and hold* = low wander; more wander = more
exploration, less early lock-in). **But it is observational** (the level-corr is
null −0.06; the signal lives only in the base→LoRA paired delta) and the
**causal test is not doable with available knobs**: er_sde `s_noise`
(`wander_intervention.py`) is degenerate outside ≈1.0 — `s_noise=0` collapses to
flat color (multistep diverges without noise), `s_noise≥2` is RGB static. **Trap
noted:** SAM3 finds no hands in static → scores it 0%-broken → would have
manufactured a spurious "more noise → less breakage → wander protective." *Always
eyeball the pixels before trusting the detector.* A clean test needs Restart
sampling (re-noise mid-trajectory, deterministic re-converge) — unimplemented;
low priority since the LoRA isn't degrading hands anyway. The protective trend
stays a **hint, not a result**.

---

## Method

- **Capture seam.** Monkeypatch `library.inference.sampling.step` (the euler ODE
  step), recomputing the *guided* `x̂₀ = x_t − σ_i·v` exactly as
  `generation.py:799`. Forces euler (no er_sde/spectrum) so the seam fires.
  Reuses the real `generate()` path (correct text padding / CFG / schedule) via
  `build_inference_bundle` — base bundle (no adapter) and base+LoRA bundle, each
  loaded once and reused across prompts/seeds. Latent is 5D `(B,C,1,H,W)`, B=1.
- **Metrics (per-patch, channel-vector L2):**
  - `tv_tail` — path length of x̂₀ across steps with σ<0.45.
  - **`wander = Σ path / Σ net`** — =1 for a straight monotone resolve, >1 when
    x̂₀ backtracks. *The* contradiction metric. (Triangle inequality ⇒ >1 requires
    genuine per-patch direction change, not spatial heterogeneity.)
  - `backtrack` — Σ max(0, −cos(d_i, d_{i+1})), energy of direction reversals.
  - **cross-seed variance** of x̂₀ at a fixed early σ — regime classifier:
    high = genuinely multimodal region (seed lottery) vs diffuse score error.
- **x̂₀ filmstrip** (`x0_strip.py`) — decode x̂₀ every step, crop to the face,
  tile base (top) vs LoRA (bottom). Makes "wobble vs committed" directly visible.

---

## Findings

### 1. LoRA effect on contradiction: inert in the tail, +~10% full-trajectory

**Tail wander** (`run_bench.py`, 6+6 real captions, 3 seeds) — LoRA ≈ inert:

| group | tail wander base→lora | backtrack ratio | tail-TV ratio | regime r |
|---|---|---|---|---|
| sincos (in-dist) | 2.40 → 2.41 (×1.005) | 1.00 | 1.015 | 0.23 |
| others (OOD)     | 2.08 → 2.09 (×1.006) | 1.00 | 1.031 | 0.11 |

**Full-trajectory wander** (`base_vs_lora_wander.py`, 3+3 captions, 2 seeds,
compiled) — LoRA amplifies ~10%:

| full wander | sincos | others |
|---|---|---|
| base | 5.46 | 4.98 |
| lora | 6.09 | 5.41 |
| **lora/base** | **×1.116** | **×1.084** |

Reconciliation: the LoRA adds wander in **early/mid σ** (mode selection, 95% of
motion) but nothing in the tail (already resolved). The "inert" impression was
tail-biased. The amplification is larger on in-distribution (sincos) captions and
is consistent with the LoRA adding stylistic detail (busier scene) rather than
destabilizing — but it is a real ~10% margin on top of the base, not zero. Base
still owns ~90%. (Generic out-of-style prompts showed ×1.05–1.14 *tail* — an OOD
artifact.)

### 2. The tail is a non-issue; wander lives early/mid

Absolute path breakdown (base, seed 0):

| | full wander | early (σ>0.45) | tail (σ<0.45) | **tail % of all motion** |
|---|---|---|---|---|
| sincos | 6.21 | 6.04 | 2.12 | **5%** |
| others | 4.86 | 4.69 | 2.35 | **4%** |

The tail's 2.1–2.4 is a high ratio over microscopic motion (x̂₀ is ~resolved by
σ≈0.45). The mode-selection phase (σ>0.45) carries 95% of the motion and all the
wander.

### 3. Wander ladder — *scene complexity* drives it (not under-determination)

Base model, seed 0, sorted:

| prompt | full wander | × floor |
|---|---|---|
| red cube on white bg | 1.85 | 1.0× |
| **1girl, standing, plain bg, simple** | **1.73** | **1.0×** (control) |
| red apple on white bg | 2.22 | 1.2× |
| ceramic mug on table | 4.29 | 2.3× |
| others (real captions) | 4.86 | 2.6× |
| sincos (real captions) | 6.21 | 3.4× |

Floor ≈1.8, so sincos's 6.2 is genuinely elevated (not intrinsic high-σ jitter).
The **simple-character control (1.73)** proves it's *not* "drawing people" — a
sparse character prompt is as straight as a cube. The ladder tracks **how much
content the caption asks for.** (Finding 5 disambiguates complexity from
under-determination and shows it's the former.)

### 4. The awkward eyes = committed mode, not contradiction

`final_p0_s0`: same seed, base eyes clean, LoRA eyes awkward (so not seed luck —
the LoRA did it). But s1/s2 LoRA eyes are fine (so not systematic). The x̂₀
filmstrip (`x0strip_seed0_tail.png`) shows both base and LoRA eyes **snap to
their final shape by ~σ0.50 and hold** — the LoRA's is just an awkward-but-stable
basin for that seed. Destination error (overfit axis), not path backtracking.

### 5. Caption-sparsity sweep — the hypothesis flips

`sparsity_sweep.py`: 3 sincos captions, nested tag-strip, 2 seeds (LoRA model).

| level | ~tags | wander | early |
|---|---|---|---|
| full | 50.3 | 6.10 | 5.94 |
| ¾ | 39.3 | 6.07 | 5.91 |
| ½ | 28.3 | 5.93 | 5.77 |
| ¼ | 17.3 | 5.49 | 5.32 |
| scaffold | 6.3 | 5.03 | 4.85 |
| **empty** | 0.0 | **1.69** | 1.57 |

**Predicted:** stripping detail → more under-determination → *higher* wander.
**Observed:** stripping detail → *lower* wander, monotonically; empty crashes to
the floor (1.69). This **falsifies the "missing detail → in-step contradiction"
mechanism.** Wander is driven by **how much content the caption asks for**, not by
gaps in it — more tags = a busier scene = a more winding assembly path. The big
step is scaffold→empty (5.03 → 1.69), i.e. removing the *specified subject*; the
descriptive tags only add a gentle gradient (5.03 → 6.10). Plot:
`wander_vs_sparsity.png`.

Corollary: caption quality probably *does* help hands/anatomy — but via
**destination/conditioning** (steering to a well-represented mode), **not** by
reducing trajectory contradiction. The two are different channels, and wander
only sees the latter.

---

## Interpretation & levers

- **x̂₀ non-monotonicity ("contradiction") is real and measurable**, but it
  reflects **scene complexity**, not caption under-determination — Finding 5
  shows *removing* detail *reduces* it. So the original mechanism ("missing
  detail makes steps disagree → breakage") is **not supported by the data.**
- **It is base-owned** (LoRA wander ratio ≈1.00): a frozen-base LoRA, REPA
  included, can't reshape it. So `caption_dropout` / rank won't touch it — but
  there's no evidence you'd *want* to, since it isn't linked to breakage.
- **What actually broke in the image we dissected (eyes) was the overfit axis** —
  a committed bad mode (filmstrip: snaps early, holds). *That* is what LoRA levers
  (REPA, dropout, rank, more data) move. So your `caption_dropout` instinct is
  aimed at the right axis (overfit/destination), just not at "contradiction."
- **Caption quality almost certainly still helps hands/anatomy** — but via the
  **destination/conditioning** channel (steer to a well-represented mode), which
  this bench didn't measure, *not* by reducing wander. To probe that, you'd
  measure mode *correctness* (e.g. a hand/anatomy detector or cross-seed mode
  consistency), not path geometry.
- **Net:** the in-step-contradiction framing was a productive hypothesis that the
  data **redirected** — the lever for your visible artifacts is the
  overfit/destination axis (LoRA-side: REPA/data/rank/dropout; caption-side:
  conditioning), not trajectory contradiction.

## Reproduce

```bash
# base-vs-LoRA stability on real captions (tail + regime maps)
python bench/x0_contradiction/run_bench.py \
    --prompts-file bench/x0_contradiction/prompts/sincos.txt --label real-sincos
# full-trajectory base-vs-LoRA wander, 4-cell table (compiled, fixed res)
python bench/x0_contradiction/base_vs_lora_wander.py --seeds 0,1 --compile
# x̂₀ filmstrip for one image (contradiction vs committed mode)
python bench/x0_contradiction/x0_strip.py --prompt "<prompt>" --seed 0
# caption-sparsity wander sweep -> wander_vs_sparsity.png
python bench/x0_contradiction/sparsity_sweep.py --seeds 0,1

# --- follow-up (2026-06-21): detail-vs-contradiction split (F1) ---
python bench/x0_contradiction/wander_is_it_real.py \
    --prompts-file bench/x0_contradiction/prompts/sincos.txt --seeds 0,1,2 --compile
# SAM3 finger-count breakage: does the LoRA degrade? is wander the cause? (F2/F3)
python bench/x0_contradiction/wander_vs_breakage.py \
    --prompts-file bench/x0_contradiction/prompts/hands.txt --seeds 0,1,2,3,4 --compile
# er_sde s_noise intervention (F3) — DEAD instrument: degenerate outside ≈1.0
python bench/x0_contradiction/wander_intervention.py --seeds 0,1,2,3,4 --compile
```

All scripts take `--compile` (compile_blocks; safe at fixed square res).
`wander_vs_breakage` / `wander_intervention` need SAM3 (`make download-models`).

## Caveats

- wander is a latent-space, per-patch channel-vector metric; high values mean
  per-patch non-monotonicity but a latent reshuffle need not equal a pixel-space
  artifact. The original "we did not establish wander→breakage" caveat is now
  **resolved by the F2 SAM3 detector** (see Follow-up): on n=30 hand prompts the
  LoRA does not break more hands than base, and wander does not predict/locate
  the breakage. Breakage = overfit axis, not the wander/contradiction axis.
- F2/F3 caveats: n=30, one artist, one LoRA; SAM3 finger-count is noisy
  per-image (validated at the extremes, fuzzy in the middle); excess-only ignores
  fused/<5-finger breakage to dodge the occlusion confound. Direction (no
  degradation, null mechanism) is consistent across all F2 metrics + the
  symmetric paired test, so it's robust as a direction even if magnitudes are soft.
- Single LoRA, single artist, euler/cfg=4. The base-owned conclusion is robust
  across the in-dist/OOD split; the magnitude numbers are model-specific.
- Tail wander (×1.005) and full-trajectory wander (×1.08–1.12) disagree because
  the LoRA's effect concentrates in early/mid σ; report both, not just the tail.
- `compile_blocks` (`--compile`) used for `base_vs_lora_wander`; bit-exact to
  eager per the DiT docs, ~20% faster (2.4 vs 2.0 it/s) at fixed 1024².
