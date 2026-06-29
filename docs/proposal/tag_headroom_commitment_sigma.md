# Tag-level headroom & commitment-σ map

**One-liner:** Build a per-tag map of *when in the σ schedule each prompt feature
commits* and *whether it renders correctly*, then disentangle the four cells that
matter — genuinely-hard vs simple-but-failing, memorized vs genuinely-capable — so
we can tell real adapter/conditioning headroom from base-owned dead ends.

Status: **Phase-0 PASS — narrow.** Bench: `bench/cross_attn_drive/tag_influence.py`
(0a + 0b); reading guide `bench/cross_attn_drive/how_to_observe.md`. The σ-gated
tag-drop hook is `generate_body(context_alt=…, tag_drop_sigma=…)` in
`library/inference/generation.py` — the tag-level generalization of the
`ANIMA_TEXT_KNOCKOUT_SIGMA` null-swap.

**First full run (2026-06-25, base DiT, 20 suspect tags × 6 captions × 4 seeds,
1024px/28-step/CFG4, 924 gens, ~3 h).** 0a PASS, 0b PASS — but the headroom cell is
*thin*: exactly **one common tag clears all three gates — `speech bubble`**
(influence 0.98, concentration 4.5, instability 2.32× = high excess wobble ⇒ not
memorized, train_freq 192). It commits late (`commitment_σ=0.8`; rel curve
0.95→0.46, 0.8→0.16, 0.6→0.075) — but at the band's *upper edge* (σ≈0.8–0.85),
overlapping bulk-content lock, so a localized cross-attn lever there is most exposed
to risk #2 (global sharpening). The broader used∧failing family (`english text`,
`signature`, `claw pose`, `interlocked fingers`) is all **low-frequency** → data-
limited, demoted by the frequency guard. Hands (`holding`/`fingernails`) +
accessories (`choker`/`nail polish`) cluster at low influence (redundant with
co-occurring tags); counts/big-logos (`2girls`/`1boy`) are stable (solved/diffuse).

**Phase-1 call:** the success criterion is met (non-empty named target), so the line
stays open — but pursue the localized mid-σ cross-attn lever **only if rendered-
text/bubble quality is a priority**. It's a real but narrow lever (one common tag) at
an awkward σ; not a sweeping win.

## Motivation

The cross-attn-drive probes ([[project_crossattn_drive_frontloaded]]) established:
bulk content locks by σ≈0.8, but a thin tail of **localized prompt-specified
features** (faces/expressions, logos, accessories, hands, rendered text) keeps
committing through the σ∈[0.6,0.8] band and follows the text. Global metrics
(RMSE, cos) are bulk-dominated and blind to these.

That raises the operational question this proposal answers: **which tags are
those, when do they commit, and which of them is the model getting wrong?** A tag
that commits late *and* renders wrong *and* is common-and-simple is real headroom
(a localized mid-σ cross-attn lever, or a training emphasis, could move it). A tag
that fails because it's intrinsically hard, rare, or because the "success" we see
is memorization — is not. We need to separate these before building anything.

## What we measure — detector-free, by intervention not detection

**No local tagger.** A small danbooru tagger is too noisy to be the success oracle,
and any classifier we'd swap in (CLIP/SigLIP) is weak on this anime+NSFW domain.
So we don't *detect* whether a tag rendered — we *intervene* on each tag and read
the model's own causal response. Three signals, all detector-free, all from
generations we already know how to run:

1. **influence(T)** — generate with the full caption C vs C∖T (tag removed), same
   seed; the localized image distance is how much the model *uses* T. A
   high-frequency tag with near-zero influence is being ignored — a candidate
   failure — with no classifier asked. (This is the per-tag version of the
   knockout's diff map; `knockout_diffmap.py` already localizes the change.)
2. **commitment-σ(T)** — present T only above a cutoff (swap C→C∖T below σ=x); the
   x at which the result rejoins the full-caption baseline is when T locks. The
   tag-level analog of the global knockout.
3. **instability(T)** — seed-variance of T's affected region (the region influence
   localized), across a few seeds with the full caption. **High variance = the
   model is trying but rendering T inconsistently = a difficulty/failure proxy;
   near-zero variance = either solved or memorized.** This is the key move that
   replaces the tagger: "does it render *correctly*" is unmeasurable without a
   detector, but "does it render *consistently*" is fully detector-free.

The one genuinely subjective question — *is the rendered feature actually good* —
stays **human, on a short ranked list the automation surfaces**, never a flaky
classifier over thousands of images. Automation ranks; the eye adjudicates a
shortlist.

## The confound grid (the hard part), now tagger-free

| axis | detector-free signal | asset |
|---|---|---|
| **commits early ↔ late** | commitment-σ (σ-gated tag-drop) | knockout hook (generalized) |
| **used ↔ ignored** | influence(T) = dist(C, C∖T), localized | tag-drop + `knockout_diffmap.py` |
| **consistent ↔ struggling** | instability(T) = seed-variance of T's region | multi-seed gen |
| **hard ↔ simple** | training frequency + tag category | `caption_index.json`, `danbooru_tags_classified.csv` |
| **memorized ↔ genuine** | low instability **and** high nearest-training PE-Spatial similarity | `bench/memorization/probe.py` |

The seed-variance axis does double duty: high instability ⇒ struggling; near-zero
instability + complex + train-similar ⇒ memorized. So "memorized vs genuine" and
"struggling vs solved" both fall out of one detector-free measurement plus the
existing image-based memorization probe — no tagger anywhere.

Interpretation (the cell we hunt in **bold**):

- **high influence + late commit + high instability + high-freq** → prime headroom:
  the model cares about T, commits it late, renders it inconsistently, and it's
  common ⇒ plausibly adapter/conditioning-addressable.
- low influence + high-freq → ignored tag (a different kind of failure: not even
  attempted) — worth a look but maybe redundant with other tags.
- high influence + low instability + complex + train-similar → **memorized**; flag,
  don't celebrate.
- late commit + low instability → text-dependent fine feature that works; no action.
- any failure at low frequency → data/capability-limited; not an architecture lever.

## Cost control

Captions carry 20-50 tags; dropping every tag is 20-50× generation per image.
Don't. Target an **a-priori suspect set** from the known-hard categories in
`danbooru_tags_classified.csv` — rendered text, hands/`fewer digits`, counts
(`Ngirls`), small accessories, logos — plus drop the rest *by typed group* (all
"expression" tags together, etc.) for a coarse first pass. Tag-drop itself is pure
prompt manipulation (remove a comma token) over the existing generate; only the
σ-gated variant needs the embed-swap hook.

## Phase-0 (cheap, falsifiable gate) — no tagger

**0a — influence + instability on the suspect set.** ~15 a-priori-suspect tags ×
~10 captions each × 4 seeds: baseline vs tag-drop, measure influence + seed-variance
(no σ sweep yet). *Kill criterion:* if the suspect tags all show either high
influence + low instability (solved) or near-zero influence (redundant), there is no
"trying-but-failing" population and the line stops. *Pass:* a population of high-
influence + high-instability + high-frequency tags exists.

**0b — commitment-σ on the flagged tags only.** Add the σ-gated tag-drop for the
0a-flagged tags. *Pass:* the flagged tags commit late (need the [0.6,0.8] band).
That co-location (used ∧ late ∧ unstable ∧ common) is the headroom signature and the
green light for Phase-1.

Reuses `run_knockout_sweep.sh` + new `tag_influence.py` (prompt manipulation +
diff/variance aggregation); the σ-gated drop generalizes the existing knockout hook
from "→ null below cutoff" to "→ alternate-caption embedding below cutoff".

## Phase-1 (only if Phase-0 passes) — sketch

Two candidate levers for the identified headroom cell, both gated on a *named*
failing-tag set, not a global metric:

1. **Inference (training-free first):** localized mid-σ cross-attn upweight — boost
   `gate_cross` in σ∈[0.6,0.8] *only where the cross-attn map concentrates* (not
   globally; a global boost amplifies the locked collinear bulk ≈ harmful CFG
   sharpening). Measure tag-recall delta on the failing set vs an artifact check on
   the bulk.
2. **Training:** emphasize the failing tags' loss in the mid-σ band (a σ×tag loss
   weight), validated by recall recovery on a held-out set.

## Risks / priors (be honest)

- **Base-owned ceiling.** x̂₀-wander/complexity is ~90% base-owned and caption-level
  fixes don't move it ([[project_x0_contradiction_bench]]); the failing tags may be
  base failures an adapter can't reach. Phase-0b's frequency split is the guard —
  only high-freq failures are plausibly adapter-addressable.
- **Per-σ reweighting graveyard.** Global per-σ guidance reweights are a NO-WIN line
  here ([[project_sigma_reshape_no_win]]); Phase-1 lever #1 must be *localized*, or
  it inherits that result.
- **No oracle for "correct."** The detector-free design measures *used /
  consistent / committed-when*, not *correct* — instability is a proxy that
  confounds genuine variation (a "smile" legitimately varies) with failure. Guard:
  normalize instability against the tag's variance on *training* images (how
  variable is this feature for real?), and keep human adjudication on the final
  shortlist. Don't ship a "failing-tag" verdict from variance alone.
- **Memorization is a feature here, not only a bug** — a memorized-but-correct early
  commit still "succeeds"; the axis exists to stop us mistaking memorization for the
  capability we'd be trying to build.

## Success criterion

A ranked, deduplicated list of high-frequency tags that (1) the base renders wrong,
(2) commit late (need the [0.6,0.8] band), and (3) are not memorization artifacts —
i.e. a concrete, falsifiable headroom target list. Empty list = honest negative,
and the cross-attn-lever line closes for good.
