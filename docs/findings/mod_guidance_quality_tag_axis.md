# Mod guidance: why duplicated/aligned tags degrade, and the quality-tag direction structure

This records why putting a tag in **both** the positive prompt and the
mod-guidance steering prompt (or even just in the positive prompt while
steering is on) degrades quality, plus a map of how booru "quality" tags sit in
the `pooled_text_proj` modulation space. The short version:

1. The base modulation and the steering delta **both** read the same
   `crossattn_emb.max(dim=1)` → `pooled_text_proj`. A tag that pushes the
   projected modulation along the steering axis gets that axis driven **twice**
   → over-saturation / DC blowout (`docs/methods/mod-guidance.md`).
2. The hazard is **directional overlap**, not string duplication. It must be
   measured *in context* (max-pool is non-additive), and an **artist tag can
   drive the quality axis ~3–4× harder than `score_9` itself**.
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
Reproduce: `bench/mod_guidance/run_bench.py` + `plot_quality_axis.py` +
`base_sensitivity.py`.

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

```bash
# full vocab scan + per-anchor alignment + anchor cosine matrix
uv run python bench/mod_guidance/run_bench.py --label vocab-scan \
    --anchor_tags "score_9,score_8,score_7,score_6,masterpiece,best quality,absurdres"
# the figure (reloads the DiT; recency overlay included)
uv run python bench/mod_guidance/plot_quality_axis.py
# base-prompt sensitivity: full vocab re-scored against 3 dense real captions
uv run python bench/mod_guidance/base_sensitivity.py
```

Both use `anima-base-v1.0.safetensors` + newest
`output/ckpt/pooled_text_proj*.safetensors`, `attn_mode=torch`, bf16. Artist/
character/copyright vocab from `post_image_dataset/captions/caption_index.json`.
Note bf16 precision makes the smallest entries (~0.001) noise; the flagged tier
(`|push_ratio| ≥ 0.05`) and the pole structure are robust.
