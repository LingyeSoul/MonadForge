# Mod guidance: the pooled-text "quality axis" — DEMOTED (geometry-only); replaced by image-space channel attribution

> **STATUS (2026-05-30): the "quality axis" framing is demoted. Read this first.**
>
> Two load-bearing claims in this doc were never tested the way they were
> stated, and one is contradicted by the doc's own data:
>
> 1. **It is not a "quality axis" — it is a content-magnitude axis.** The doc's
>    own Finding 1 shows an *arbitrary artist tag drives the axis 3–4× harder
>    than `score_9`* (`@sincos +0.31` vs `score_9 +0.07`). There is no sense in
>    which `@sincos` is "more quality" than `score_9`; what the projection
>    actually measures is the **magnitude of a tag's pooled shift along whatever
>    direction your steering pair happens to point**, and that direction is
>    dominated by high-variance content channels. The doc half-admits this
>    ("the steering axis isn't pure 'quality'; it correlates with strong,
>    specific content"). So: the robust core is the trivial *"strong content
>    tags make large pooled marginals"*; the *quality* interpretation is
>    falsified. Do **not** cite this as a quality lever.
>
> 2. **"Double-drive degrades quality / DC-blowout" was never imaged.** Every
>    number below lives in `pooled_text_proj` **geometry** (cosines between
>    projected pooled vectors). Not a single image was sampled. The
>    over-saturation / pink-DC-blowout claim is *inferred from cosines*, never
>    observed — exactly the kind of geometry-space claim that, on Anima, has a
>    track record of not surviving contact with pixels (cf.
>    [[project_fm_val_loss_uninformative]]).
>
> 3. **Position artifact.** The geometry scan measured tags **appended at the
>    tail** (general band). But Anima's caption `SLOT_ORDER` (rating → count →
>    character → copyright → artist → general,
>    `library/captioning/anima_tagger.py:90`) puts booru quality/meta tags right
>    after the rating literal, and artist `@`-tags in their own pre-general slot.
>    So the comparison pitted correctly-placed artist tags against an
>    **off-distribution tail `score_9`** — and cross-attn is strongly
>    position-sensitive (see below). The "artists beat `score_9` 3–4×" gap is
>    therefore *partly a placement mismatch*, not only content-magnitude.
>
> **What replaces it — image-space channel attribution**
> (`bench/mod_guidance/channel_attribution.py`). A tag edit enters the DiT
> through two separable inputs — the pooled/mod channel (`max → pooled_text_proj
> → AdaLN`) and the cross-attn channel (full `crossattn_emb` sequence) — split via
> `pooled_text_override`. Full run (2026-05-31, 6 dense real captions × 2 seeds,
> tags spliced at their *correct* slot; `results/20260531-0005-full/`):
>
> | tag | pool_share (latent / PE) | cross_share | cos(cross,pool) |
> |---|---|---|---|
> | `score_9` | 0.50 / 0.77 | 0.97 | +0.27 |
> | `masterpiece` | 0.50 / 0.71 | 1.04 | +0.27 |
> | `holding a sword` (content) | 0.40 / 0.44 | 0.98 | +0.23 |
>
> Reads (n=12/tag, pool_share σ≈0.2–0.3 — directional):
>
> - **The mod channel is a real, *quality-selective* lever — not inert.** Quality
>   tags route ~0.5 (latent) / ~0.7+ (PE) of their image movement through pooled,
>   vs ~0.4 / 0.44 for the content tag. Cross-attn is the larger single channel
>   (~1.0), but pooled is a substantial secondary contributor, *more so for
>   quality than content*. The original *intuition* (quality acts via pooled
>   modulation) survives; the "axis"/geometry framing and the tail-append
>   measurement were the wrong parts. (Tail-append gave pool_share 0.22 —
>   placement roughly doubled it, corroborating point 3.)
> - **"Double-drive" is mild reinforcement, not over-saturation:** cos ≈ +0.26,
>   but additivity residual ≈ 0.63 — strongly non-linear interaction, not a clean
>   superposition.
> - **DC-blowout from a hard push is NOT reproduced** under the shipped schedule
>   (blocks 8–26, tonal-DC blocks 0–7 protected): sweeping steering `w`→8 gives
>   only +3% pixel-std drop / ~1% tone shift, and the effect *saturates* past
>   w≈3 (pushing harder stops changing the image rather than degrading it). The
>   doc's pink/DC-blowout was likely an unprotected-schedule / duplication
>   artifact, not an intrinsic property of pooled steering.
> - **Cross-attn is strongly order-sensitive:** reordering the tags (pooled pinned)
>   moves the image 64% (latent) / 84% (PE) of a *full seed change* — which is why
>   tag placement (point 3) is load-bearing.
>
> **Qualitative read of the `swap` grids (eyeball, valence — what the scalar
> magnitudes can't show; n=12/tag, treat as a hypothesis):** the pooled/mod
> channel behaves like a **global grade/polish operator, not a content editor** —
> consistent with AdaLN being a per-feature-channel shift applied uniformly across
> all spatial positions (it can shift tone/contrast/sharpness but cannot move
> content). Observed pattern: when **BB is already good, BT (pooled-only) reads
> slightly *better* than BB** (a small finish on already-right content); when **BB
> is poor, TB (cross-attn-only) beats BT** (only the content channel can repair
> structure — a global grade can't rescue a bad base). This also explains why
> `pool_share` is higher in PE (~0.7, mean-pooled → tuned to global tone) than in
> latent (~0.5): the metric most attuned to grading is the one that most rewards
> the pooled channel. **Practical framing: mod-guidance/quality-pooled is a
> finishing knob conditional on a good base, not a quality rescue.** To harden:
> stratify `swap` pairs by base quality and check the sign of `BT−BB` (good
> stratum) vs `TB−BB` (poor stratum) — not yet done.
>
> **Base-vs-distill origin of the routing (2026-05-31, `--experiment origin`).**
> The pooled→AdaLN path is **100% distillation-induced**: the base DiT ships a
> zero-init `pooled_text_proj` with `enable_pooled_text_modulation=False`
> (`models.py:1283-1338`, `weights.py:235`), so base AdaLN is purely
> timestep-conditioned and text reaches the base model *only* through
> cross-attention. There is no "native pooled path" to swap in (an identity proj is
> trivially `pool_share=0`). The split that *does* exist: the pooled **vector**
> `max(crossattn_emb)` is 100% base encoder (Qwen3 + LLM-adapter ship with the DiT;
> distill trains only the proj). `origin` decomposes a tag's pooled response into
> upstream `rel_dpool` (base, no proj) vs `proj_gain` (distilled). On n=12 dense
> real captions:
>
> | | quality tags | content tags |
> |---|---|---|
> | `rel_dpool` (base encoder, no proj) | 0.059 | 0.031 |
> | `proj_gain` (distilled proj) | 2.94 | 3.04 |
>
> **The quality-selectivity is entirely upstream.** The base encoder already moves
> the pooled vector ~1.9× more for quality than content; the distilled proj is a
> **tag-agnostic ~3× amplifier** (gain identical for quality/content — content
> marginally higher), having learned no quality preference, consistent with the
> distill data containing **no quality tags at all**. So "quality-selective lever"
> is right at the image level but must NOT be read as *"the proj prefers quality"*:
> the proj amplifies everything uniformly; quality just arrives with a bigger input.
> **Consequence:** you cannot make mod-guidance *more* quality-selective by
> conditioning the distill teacher on quality tags — the selectivity ceiling is set
> by the base encoder, not the proj. The productive levers exploit the existing
> upstream signal (adaptive steering `w` on base quality; the base-quality
> stratification above), not retraining the proj. (Sparse-prompt smoke n=2 showed a
> spurious proj_gain gap 2.54 vs 2.03 — a sparse-base-collapse artifact, gone on
> dense captions. Render-free: `--experiment origin --dataset_samples N`.)
>
> The geometry analysis below is preserved as the **original record** — a valid
> map of the pooled-projection space, useful for reasoning about the *mechanism*,
> but its "quality" labels, its tail-append placement, and its "degrades quality"
> conclusion are the demoted parts.

---

## Schedule axis (σ + layer): both falsified — the shipped `8–26` full-dose is validated

> **STATUS (2026-05-31): closed.** A separate proposal
> (`docs/proposal/mod_guidance_layer_sigma_schedule.md`, now removed) asked whether
> the mod-guidance steering *schedule* — which blocks carry it (`build_mod_schedule`'s
> hand-set `8–26`) and whether to gate it by σ — could be set better than the shipped
> default. Two phases on `channel_attribution.py` killed **both** axes. The shipped
> per-block `8–26` schedule applied at **full dose, every step** is the right call;
> there is no layer-placement or σ lever, so a "learnable `w` scheduler" degenerates
> to a constant already set correctly. Recorded here so it isn't re-proposed.

**Readout.** Each arm is rendered vs an unguided `off`: `delta_norm` (how much the
steering moved the latent) + pixel **SSIM**-to-`off` (structure preserved). The σ-band
gate and per-block range are both buffer writes on the live `_mod_guidance_schedule`
(no recompile). `J = HF − λ·LF` is kept only as a guide — it is an HF-noise trap, never
the verdict. **Read the grids.**

### Phase 0 — σ axis (C1): DEAD (dose-controlled)

`--experiment sigma_window`, n=12 (6 dense real captions × 2 seeds), 20 steps, shipped
`step_i8_skip27` pair, w=3 (`results/20260531-1155-phase0b/`):

| arm | SSIM-to-`off` | delta_norm |
|---|---:|---:|
| uniform (σ-blind, shipped) | 0.885 | 79.5 |
| high045 (σ≥0.45) | 0.885 | 79.1 |
| low045 (σ<0.45 tail, equal-w) | 0.995 | 7.8 |
| low045d (σ<0.45 tail, w=15, dose-matched ×5) | 0.980 | 18.6 |

`uniform ≈ high045` exactly — the whole effect is the σ≥0.45 structure-forming steps.
Dose-matching the σ<0.45 tail 5× bought only ~2.3× effect (7.8→18.6), still ~4× short
of uniform — per-step **saturation** (effect plateaus past w≈3). You physically cannot
buy the grade in the 4 tail steps. C1 ("restrict the schedule to the tail to preserve
the grade") is falsified, dose confound and all.

### Phase 0b — layer axis (C2): FALSIFIED (it's dose, not placement)

The paper's actual axis (arXiv:2602.09268: per-layer strategies; Table 7 aspect-dependent
optimum). Anima ships only a hand-set `8–26` derived from geometry, never image-validated.

- **Per-block map** (`--layerwin_mode single`, blocks `8–26`, n=12;
  `results/20260531-1259-phase0b/`): between-block SSIM std **0.0079** sits *below* the
  n=12 estimation noise floor (SE of a block mean ≈ 0.0121) — not statistically resolved.
  A faint ordered gradient exists (SSIM vs block-index r=+0.70, delta r=−0.65: early
  `8–16` steer ~15% harder + drift a hair more, late `18–26` milder + safer) but both
  ends are "structure preserved" (0.929 vs 0.941). No single block is a drift block
  (all SSIM ≥ 0.92). Critically, `full08-26` (SSIM 0.877, delta 85) is **lower-SSIM and
  ~1.5× more effect than the hardest single block** → the shipped path's structure
  movement is **emergent from stacking 19 blocks, not localizable to any block**.
- **Band sweep** (`--layerwin_mode band` = early/mid/late thirds, on 4 *lengthy* dense
  captions via `--prompt_min_chars 550`, × 2 seeds; `results/20260531-1330-phase0b_band/`):
  band SSIM ordering is **at chance** — the safest band varies prompt-to-prompt (`b08-13`
  safest in only 3/8), and `full` is not reliably worse than the worst band. No
  contiguous band owns the drift → dose- and content-dependent, not placement.
- **Qualitative grid read (the verdict).** `full [8–27]` **wins in every case**; when
  `off` is already anatomically/expressively correct, all arms converge; when `full`
  corrects something (face expression, anatomy, hands), the partial arms look like an
  **interpolation between `off` and `full`** — weaker versions of the *same* correction,
  not different corrections. **Interpolation ⇒ pure scalar dose, not placement.** If
  block placement mattered, partials would fix *different* things; they fix *less*.

**Valence resolved (the Phase-0 open question).** `full` winning means its large
movement is **correction, not drift** — so the one surviving thread from Phase 0
("cap/taper `w` out of high-σ to stop the drift") is *also* unmotivated: tapering just
reduces the correction → worse.

**Methodological correction.** The structural readout assumes movement-from-`off` =
drift (low SSIM = bad). On this channel the human read flips the sign: `full`'s low SSIM
/ high delta is **more correction**, not more damage. SSIM/`delta_norm` here measure
*amount of correction*, not harm — only the grid read disambiguates. (This is the
"read the grids, J/SSIM are guides" lesson firing in the unexpected direction.)

### Consequence

The hand-set `8–26` at full dose is validated. No layer lever, no σ lever, no taper.
A learnable `w` scheduler has nothing per-layer to allocate and degenerates to a scalar
already at its saturated-optimal value — so the SAM-hand-brokenness reward we scoped for
a per-block headroom allocator is unmotivated; don't build it. Caveat: the band read is
n=3 usable prompts × 2 seeds, qualitative — but the *interpolation* signature is a
structural claim that doesn't need large n.

**Reproduce:**

```bash
# Phase 0 (σ axis)
uv run python bench/mod_guidance/channel_attribution.py --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
    --experiment sigma_window --dataset_samples 6 --seeds 0,1 --sigwin_dose both --compile --label phase0
# Phase 0b (layer axis): per-block map, then band sweep on lengthy prompts
uv run python bench/mod_guidance/channel_attribution.py --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
    --experiment layer_window --dataset_samples 6 --seeds 0,1 --layerwin_mode single --compile --label phase0b
uv run python bench/mod_guidance/channel_attribution.py --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
    --experiment layer_window --dataset_samples 4 --prompt_min_chars 550 --seeds 0,1 --layerwin_mode band --grid_thumb 1024 --compile --label phase0b_band
```

---

## Original geometry-only analysis (pooled-projection space; quality labels demoted)

This recorded why putting a tag in **both** the positive prompt and the
mod-guidance steering prompt (or even just in the positive prompt while
steering is on) was *believed* to degrade quality, plus a map of how booru
"quality" tags sit in the `pooled_text_proj` modulation space. The short
version (read with the STATUS box above):

1. The base modulation and the steering delta **both** read the same
   `crossattn_emb.max(dim=1)` → `pooled_text_proj`. A tag that pushes the
   projected modulation along the steering axis gets that axis driven **twice**
   in *projection geometry* (`docs/methods/mod-guidance.md`). **Whether this
   double-drive actually degrades the image was never measured — see STATUS.**
2. The hazard is **directional overlap**, not string duplication. It must be
   measured *in context* (max-pool is non-additive), and an **artist tag drives
   the axis ~3–4× harder than `score_9` itself** — which is the evidence the
   axis is content-magnitude, not quality (STATUS point 1).
3. The `score_X` ladder *looks like* **a rotation, not a magnitude scale**:
   measured against a near-empty base, `score_9`/`absurdres` and
   `masterpiece`/`best quality`/`score_6` read as two near-opposite poles with
   `score_7` an orthogonal hinge. **But this two-pole geometry is an artifact of
   the sparse base** — it weakens or inverts on dense real prompts (see
   *Robustness*), so treat it as low-context structure, not a property of the
   modulation space.
4. Recency meta-tags (`year 2025/2024/2023`, `old`, `recent`) **collapse onto
   the masterpiece pole and oppose `score_9` steering** — and do *not* separate
   by year. **This is the least base-stable finding**: at the neutral base it's
   clean, but on dense real prompts recency tags go near-inert (push ≈ ±0.01–0.04)
   and lose any stable sign — so it's a near-empty-prompt phenomenon, not a
   real-prompt hazard (see *Robustness*).

**What is robust vs base-dependent.** The *class-level* claim in point 2 (named
entities drive the axis harder than `score_X`) holds across every base tested;
`absurdres` stays a positive driver everywhere. The *absolute* push magnitudes,
the 0.05 flag line, and the point-3 pole geometry are **specific to a sparse
base** and shrink ~5–25× / reorient on dense prompts — max-pool saturation. See
*Robustness: base-prompt sensitivity* before quoting any single number.

Mechanism reference: `library/anima/models.py:1726-1736` (base pooled inject),
`library/inference/corrections/mod_guidance.py:108-122` (steering delta).
Reproduce (geometry, original — scripts were never committed): the numbers below
were produced by an uncommitted `bench/mod_guidance/{run_bench,plot_quality_axis,
base_sensitivity}.py`; the live, committed bench is
`bench/mod_guidance/channel_attribution.py` (image-space channel attribution).

## Mechanism

Each block's AdaLN modulation receives, additively:

```
t_emb = t_embedder(t) + proj(pool(prompt))                  # base — from the FULL positive prompt
                      + w · [proj(pool(p₊)) − proj(pool(p₋))]   # steering delta, scaled by w (=3)
```

where `pool(·) = crossattn_emb.max(dim=1).values` and `proj = pooled_text_proj`.
If a tag in the positive prompt moves `proj(pool(prompt))` along the steering
direction `d = proj(pool(p₊)) − proj(pool(p₋))`, that axis is pre-loaded by the
base term and then driven again by the `w`-scaled delta. The injection is
`after_norm` (~10× sensitive); over-driving collapses early-block channels to
uniform tone (the documented pink/DC-blowout failure).

## Measure: marginal, not isolated

Because pooling is per-dimension **max**, a tag's contribution is non-additive
and context-dependent — its projection *in isolation* can have the opposite
sign of its effect *appended to a real prompt* (e.g. `@sweetonedollar`:
direct cos −0.47, marginal cos +0.67). So we score the **marginal** effect
against a neutral content base `B = "1girl, solo, looking at viewer, outdoors"`:

```
marginal(t) = proj(pool(B + ", " + t)) − proj(pool(B))
push(t)     = ⟨marginal(t), d̂⟩            # signed, model-channel units
push_ratio  = push(t) / ‖w·d‖             # fraction of the full steering drive
```

`push_ratio ≈ +0.07` ⇒ the tag adds one `score_9`'s worth of same-direction
drive on top of steering. (Shipped steering pair, `w=3`: `‖d‖=0.032`,
`‖w·d‖=0.096`, `‖base_proj‖=0.008`.)

## Finding 1 — named-entity tags double-drive harder than quality tags

Reference anchors against the shipped `absurdres, score_9` steering axis are
*modest* (`absurdres +0.073`, `score_9 +0.070`, `score_8 +0.009`,
`score_7 −0.011`, `masterpiece −0.038`, `best quality −0.054`). Yet artist and
character tags blow past them:

| tag | axis | push_ratio |
|---|---|---:|
| sonoda chiyoko | character | +0.320 |
| @sincos | artist | +0.310 |
| nahida (genshin impact) | character | +0.309 |
| @ie (raarami) | artist | +0.240 |
| @kukiyuusha | artist | +0.232 |
| @yamamoto souichirou | artist | +0.222 |

71/303 typed tags exceed `push_ratio ≥ 0.05`. A single strong artist tag adds
~30% of the full `w=3` steering drive — and since `‖base_proj‖` is only 0.008,
these named-entity tags *dominate* the base modulation's quality-axis component.
The steering axis isn't pure "quality"; it correlates with **strong, specific
content** (named entities inject a large, well-formed pooled shift,
`marg_cos ≈ 0.8–0.97`), which is why they dominate it.

## Finding 2 — the score ladder is a rotation with two poles (sparse-base only)

> **Robustness caveat:** everything in this section is measured against the
> neutral 4-tag base. The *Robustness* section below shows the two-pole geometry
> is an artifact of that sparse context — it does **not** survive on dense real
> prompts. Read this as "how the anchors relate when little else is in the
> prompt", not as a fixed property of the modulation space.

![quality-tag direction structure](assets/mod_guidance_quality_axis.png)

Marginal-direction cosine between quality tags:

| | score_9 | score_8 | score_7 | score_6 | masterpiece | best quality | absurdres |
|---|---:|---:|---:|---:|---:|---:|---:|
| **score_9** | 1.00 | 0.53 | −0.04 | −0.79 | −0.94 | −0.91 | 0.74 |
| **score_8** | 0.53 | 1.00 | 0.60 | −0.04 | −0.39 | −0.33 | 0.46 |
| **score_7** | −0.04 | 0.60 | 1.00 | 0.62 | 0.23 | 0.26 | 0.19 |
| **score_6** | −0.79 | −0.04 | 0.62 | 1.00 | 0.88 | 0.91 | −0.49 |
| **masterpiece** | −0.94 | −0.39 | 0.23 | 0.88 | 1.00 | 0.95 | −0.59 |
| **best quality** | −0.91 | −0.33 | 0.26 | 0.91 | 0.95 | 1.00 | −0.67 |
| **absurdres** | 0.74 | 0.46 | 0.19 | −0.49 | −0.59 | −0.67 | 1.00 |

The ladder rotates: adjacent rungs correlate (`score_9→8→7→6` ≈ 0.53/0.60/0.62)
but the endpoints are near-opposite (`score_9 ↔ score_6 = −0.79`). It maps onto
the two booru conventions as **two poles of one axis**:

```
  score_9 / absurdres  ←— score_8 —— [score_7 ⊥ hinge] —— score_6 —→  masterpiece / best quality
```

`score_6 ≈ masterpiece (0.88) ≈ best quality (0.91)`; `score_7` is essentially
orthogonal to the steering axis. **Artists sort onto the same axis** into two
near-disjoint families:

- **score_9/absurdres pole** (positive push): `@ie +0.240`, `@kukiyuusha
  +0.232`, `@yamamoto souichirou +0.222`, `@ebifurya +0.219`, `@yaegashi nan
  +0.212`, `@belko +0.208`.
- **masterpiece pole** (negative push — these *oppose* the shipped steering):
  `@ame +0.231` (|·|), `@deyui −0.222`, `@hayate (leaf98k) −0.209`,
  `@coro fae −0.209`, `@mikozin −0.106`.

The poles are crisp (artists hit them at cos ~0.9); the mid-ladder rungs
(`score_7/8`) are diffuse "nobody's-home" directions (best artist alignment only
~0.5–0.68), so they're weak both as steering tags and as duplication hazards.

## Finding 3 — recency meta-tags collapse onto the masterpiece pole (sparse-base only)

> **Robustness caveat — strongest here:** this is the *least* base-stable
> finding. The "collapse onto masterpiece / oppose `score_9`" picture is entirely
> a neutral-base effect; on dense real prompts recency tags go near-inert and
> their direction has no stable sign (see the addendum after the table). Don't
> port these numbers to real prompts.

The community-controversial recency tags all land on the masterpiece pole and
**oppose** the shipped `score_9` steering — and they do **not** separate by year
(neutral base):

| recency tag | push_ratio | cos→score_9 | cos→masterpiece |
|---|---:|---:|---:|
| year 2025 | −0.075 | −0.926 | +0.950 |
| year 2024 | −0.116 | −0.951 | +0.930 |
| year 2023 | −0.104 | −0.942 | +0.909 |
| old | −0.064 | −0.877 | +0.907 |
| recent | −0.028 | −0.851 | +0.813 |
| newest | −0.033 | −0.504 | +0.545 |

`year 2025 ≈ year 2024 ≈ year 2023 ≈ old` all point the same way (cos→score_9
≈ −0.9): in the pooled projection there is **no clean recency axis** — "year
XXXX" collapses onto the masterpiece/anime-style convention regardless of the
year. With `score_9` steering on, every recency tag is a mild *anti*-driver
(`year 2024` strongest at −0.116), which is a concrete reason year tags "fight"
score tags under mod guidance.

**Base-sensitivity addendum (`base_sensitivity.py`, `recency_by_base`).** Re-scored
against the three dense real captions, this finding does not hold — it inverts or
vanishes:

| | neutral | solo_night | duo_beach | group_beach |
|---|---:|---:|---:|---:|
| `year 2025` push_ratio | −0.075 | **+0.042** | **+0.041** | +0.006 |
| `year 2024` push_ratio | −0.116 | +0.019 | +0.007 | −0.016 |
| `year 2025` cos→score_9 | −0.926 | **+0.771** | +0.109 | −0.657 |
| `year 2025` cos→masterpiece | +0.950 | **−0.948** | −0.247 | −0.641 |
| `year 2024` cos→masterpiece | +0.930 | −0.974 | −0.452 | **+0.974** |

On dense prompts the push collapses to ≈ ±0.01–0.04 (recency tags are nearly
*inert* on the steering axis), and the direction to `score_9`/`masterpiece` has
no stable sign — e.g. `year 2025` flips from cos→masterpiece +0.95 (neutral) to
−0.95 (solo_night). Two reasons compound: (1) max-pool saturation shrinks the
marginal as before, and (2) at those tiny magnitudes the marginal direction is
ill-conditioned, so the cosine is essentially noise. The honest read: **once a
prompt has real content, recency tags barely touch the modulation axis** — the
"fights score tags" effect is a near-empty-prompt phenomenon, not a real-prompt
hazard.

## Robustness: base-prompt sensitivity

All numbers above use one neutral base (`1girl, solo, looking at viewer,
outdoors`). Because the pool is per-dimension `max`, a dense base already maxes
most channels, so an appended tag's marginal shrinks — the geometry can change.
We re-ran the full vocab against three dense real captions (rating word +
`@artist` stripped, all content kept), spanning solo-night / duo-day-beach /
group-beach (`bench/mod_guidance/base_sensitivity.py`):

| | neutral | solo_night | duo_day_beach | group_beach |
|---|---:|---:|---:|---:|
| `‖base_proj‖` | 0.008 | 0.015 | 0.020 | 0.015 |
| Spearman ρ of push_ratio vs neutral | — | 0.66 | 0.62 | 0.64 |
| flagged-set Jaccard vs neutral | — | 0.43 | 0.45 | 0.36 |
| `score_9` push_ratio | +0.070 | +0.013 | +0.007 | +0.003 |
| `absurdres` push_ratio | +0.073 | +0.076 | +0.041 | +0.010 |
| `score_9 ↔ masterpiece` cos | −0.94 | −0.68 | **+0.35** | **+0.35** |
| `score_9 ↔ score_6` cos | −0.79 | **+0.77** | +0.05 | +0.30 |

Three things change, one holds:

- **Magnitudes collapse (~5–25×).** `score_9` falls from +0.070 to +0.003–0.013;
  the "one `score_9`'s worth ≈ 0.07" unit and the 0.05 flag line are
  **sparse-base quantities**. On a realistic dense prompt the same duplication is
  a much smaller double-drive — the hazard ranking still points the right way,
  but the absolute risk numbers are upper bounds, not what you'd see in practice.
- **The two-pole geometry does not survive.** `score_9 ↔ masterpiece` goes from
  a clean −0.94 anti-correlation to **+0.35** on both beach bases, and
  `score_9 ↔ score_6` flips from −0.79 to **+0.77** (solo). Finding 2's
  rotation/poles picture is a property of the near-empty base, not the space.
- **The ranking only moderately agrees** (ρ ≈ 0.62–0.66; flagged Jaccard
  0.36–0.45) — fewer than half the flagged tags are stable, and the flagged
  count itself swings 54→105. Don't treat a specific tag's push_ratio as portable
  across prompts.
- **What holds:** the *class-level* Finding 1. On every base the top push tags
  are artist/character tags well above any `score_X` (`@sincos` lands in the
  top-8 of all four bases), and `absurdres` is the one quality tag that stays a
  positive driver everywhere. "Named entities double-drive the steering axis
  harder than quality words" is the robust takeaway.

## Practical rules

> These rules describe **projection geometry**, and each says "degrade" / "cancel"
> / "fights itself" as if confirmed in the image — they were not (STATUS point 2).
> They predict *where the pooled marginals point*, not that the picture gets
> worse. Use them as hypotheses to check against the channel-attribution bench,
> not as established image-space behaviour.

- **Risk is matched to your steering convention, not to "good artist".** A tag
  is hazardous to put in the positive prompt exactly when it shares the quality
  *pole* of the tag in your steering `p₊`. Steering with `score_9` → keep
  score_9-pole artists (`@ie`, `@kukiyuusha`, …) out of the positive prompt.
  Flip steering to `masterpiece` → the risk list inverts to `@deyui`, `@hayate`.
- **Opposite-pole tags cancel, they don't degrade.** masterpiece-pole tags
  (and all recency tags) *subtract* from `score_9` steering rather than
  double-drive it — mixing conventions weakens steering instead of saturating.
- **Don't mix `score_X` with `masterpiece`/`year` under steering** — they're
  near-opposite directions; the modulation fights itself.
- Read it off for your own steering prompt via the `cos_<anchor>` columns:
  `run_bench.py --anchor_tags "score_9,masterpiece,…"`.

## Reproduce

**Live, committed bench (image-space channel attribution — the current tool):**

```bash
uv run python bench/mod_guidance/channel_attribution.py \
    --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
    --experiment all --dataset_samples 6 \
    --tags "score_9,masterpiece,holding a sword" --seeds 0,1 --compile --label full
```

This decomposes a tag's *image* movement into a cross-attn delta and a pooled/mod
delta (the `swap` experiment), isolates pure cross-attn order-sensitivity by
pinning the pooled vector (`order`), and sweeps the steering weight to test the
DC-blowout claim in pixels (`intensity`). Use `--dataset_samples` for the dense
real-prompt regime where the geometry numbers below collapse. Read the saved
grids, not just the scalars.

**Geometry scan (original — scripts were never committed):** the
`pooled_text_proj`-space numbers in this doc came from an uncommitted
`bench/mod_guidance/{run_bench,plot_quality_axis,base_sensitivity}.py` run against
`anima-base-v1.0.safetensors` + `output/ckpt/pooled_text_proj*.safetensors`,
`attn_mode=torch`, bf16, with artist/character vocab from
`post_image_dataset/captions/caption_index.json`. They are not re-runnable as
written; treat the tables as a frozen geometry record (and see STATUS).
