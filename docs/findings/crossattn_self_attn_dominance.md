# Cross-attn vs self-attn drive across σ — text writes the low-σ plan early, self-attn + MLP render it; the dominant pathway is self/MLP at *every* σ

> **STATUS (2026-06-28).** Measurement-only finding. Two probes added under
> `bench/cross_attn_drive/`: `attn_evolution.py` (cross-attn *map* re-routing —
> "where patches look") and `attn_contribution.py` (gated residual contribution
> per pathway — "what each pathway writes"). Extends the velocity-level result
> [[project_crossattn_drive_frontloaded]] and [[project_crossattn_map_evolution]].
> No code path changed.

## The question

[[project_crossattn_drive_frontloaded]] showed the **velocity-level** text drive is
front-loaded: `‖v_cond − v_uncond‖/‖v_cond‖` peaks at σ=1 and `cos(v_cond,v_uncond)`
→0.997 below σ≈0.85, so below mid-σ text only *rescales* the base velocity. Open
question: is the cross-attention **pattern** itself frozen early, and is the late-σ
detail formation driven by self-attn rather than cross-attn? If so, why, and what
does it imply for adapters?

Two complementary probes, both baseline-free (single generation, no full−drop diff,
so the tag-dropped-baseline quality confound that pollutes `tag_influence` cannot
touch them):

1. **`attn_evolution.py` — WHERE.** Per block per step, the eager softmax cross-attn
   map (image-patch × text-token, head-avg), reduced to **drift** (1−cos vs prev
   step = re-routing) and **mass** (attention on a tag's key columns). Tag→column
   mapping is exact via qwen3 token-id subsequence match against the padded
   `input_ids`.
2. **`attn_contribution.py` — WHAT.** Mirrors `Block._forward` (bit-identical;
   verified `max|Δlatent| = 0.0`) to log the L2 norm of each *gated* term added to
   the residual: `gate_self·self_attn`, `gate_cross·cross_attn`, `gate_mlp·mlp`.
   The gates are adaln-modulated by the timestep, so this captures the σ-scheduled
   down-weighting that the WHERE probe is blind to.

## Result 1 — the cross-attn map never locks, but its re-routing rate is front-loaded

Whole-map drift is **meaningless** — 512 text positions are mostly padding sinks
(pad-id 151643), which dominate the cosine and dilute drift to ~0.006→0.002. Scope
to the tag's token columns. On real captions (18 cap × 2 seed × 28 step), per-tag
**Δσ-normalized** re-routing rate (`drift/Δσ`):

| band (σ) | rate (speech bubble) |
|---|---|
| hi (>0.9) | ~1.65 |
| mid (0.45–0.7) | ~0.29 |
| low (<0.45) | ~0.23 |

The rate collapses ~8× from the high-σ burst to a **flat non-zero floor** (~0.2–0.3)
that persists to σ=0.1 — the map keeps churning but never re-intensifies (a raw
per-step "tail rebound" is a wide-Δσ artifact; it vanishes on normalization). Tag
attention **mass declines monotonically** and sits *below* per-token uniform
(2/512≈0.004) for the generic text tags — they are under-attended and the late
churn rides a shrinking budget. **Reframe:** the low-σ drift floor is not text
re-engaging — mass is falling — it is the *image-derived queries* drifting as
self-attn sharpens the patches, dragging the cross-attn map along mechanically.

## Result 2 — self-attn + MLP dominate the residual at *every* σ; cross-attn is a small, fading term

Gated contribution L2 norm (base model, 12 cap × 2 seed × 28 step):

| band (σ) | self | cross | mlp | cross_frac = cross/(self+cross) |
|---|---|---|---|---|
| hi (>0.9) | 248 653 | 28 498 | 254 358 | **0.151** |
| upper_mid | 281 332 | 25 264 | 274 210 | 0.101 |
| mid | 301 985 | 22 269 | 291 838 | 0.076 |
| low (<0.45) | 292 827 | 19 393 | 286 331 | **0.064** |

- **self-attn dominates cross-attn at the highest σ already** (`self_dominant_below_sigma
  = 1.0`): ~9× at σ=1, growing to ~15× at low σ. There is no σ where cross-attn
  "takes over"; the user's "self-attn dominant from σ<0.9" is an *understatement* —
  it dominates throughout.
- **cross-attn's share is front-loaded**: 15% of the attention update at high σ →
  6% at low σ. Including MLP, text is ~5%→3.5% of the *total* residual update.
- **self and MLP GROW into mid/low σ** (detail-formation phase) while **cross
  shrinks** — direct confirmation that low-σ detail is rendered by self-attn + MLP,
  not by cross-attn re-reading the prompt.

## Result 3 — the sincos style LoRA rides the self/MLP stream, not cross-attn

Same probe, base vs `anima_sincos2` LoRA (identical captions/seeds):

| band | Δself | Δcross | Δmlp | cross_frac base→lora |
|---|---|---|---|---|
| hi | +4.5% | +3.4% | (up) | 0.151 → 0.148 |
| mid | +4.6% | +5.1% | (up) | 0.076 → 0.075 |
| low | +5.4% | +2.9% | (up) | 0.064 → 0.060 |

The LoRA amplifies **all three pathways ~uniformly (+3–5%)** and leaves the
self/cross/MLP **balance unchanged** (cross_frac flat). Because self+MLP carry
85–94% of the update at every σ, the overwhelming majority of the LoRA's added
signal flows through self+MLP, not cross-attn. The companion WHERE probe agrees on
its own trigger: for `@sincos`, the base model already attends *strongly and stably*
(mass ≈0.0206 ≈ 2.6× uniform, roughly flat across σ — unlike the under-attended text
tags), and the LoRA changes that allocation by ~0 (slightly **lowers** it,
Δmass −3e-4…−6e-4). So the adapter does not steer by making patches look harder at
its trigger; the trigger is a switch, and the style is delivered downstream.

This cross-checks [[project_lora_crossattn_learns_labeled_only]] (text tags are a
data/capability limit, not a cross-attn-mass deficit the adapter can fix) and
explains why late cross-attn levers are inert ([[project_tag_boost_late_sigma_kill]]):
there is almost no cross-attn budget to lever below mid-σ.

## Interpretation — coarse-to-fine division of labor

Flow-matching denoising is coarse→fine in frequency: high σ sets low-frequency
structure (layout, identity, color blocks), low σ fills high-frequency detail
(texture, edges, small parts). The pathways split along this axis:

- **Cross-attn writes the low-frequency plan early.** Text influence (velocity
  delta, re-routing rate, and contribution share) all peak at high σ and fade. By
  mid-σ the cond/uncond velocities are near-parallel — text only rescales.
- **Self-attn + MLP elaborate the plan into high-frequency detail late.** Their
  contribution grows as detail forms; a late-emerging detail's *cause* was committed
  early by text, but its low-σ *velocity* is computed by self/MLP. Emergence-time ≠
  causation-time, and the late cross-attn map churn is a downstream consequence of
  self-attn sharpening the (image-derived) queries, not text re-asserting itself.

**Adapter consequence:** a style/identity LoRA cannot work by re-routing or
up-weighting cross-attn — there is no budget there. It must (and the sincos LoRA
does) ride the dominant self/MLP stream, triggered by an early, already-well-attended
token. Capabilities that genuinely need *new* text→pixel routing (readable glyphs,
speech-bubble text) can't be bought by amplifying late cross-attn; the deficit is
that the early low-freq plan never encodes them and self-attn has no prior to
elaborate them.

## Caveats

- The contribution probe measures **magnitude, not direction**. "cross_norm +3%"
  means the cross term's *size* is ~unchanged; the LoRA could rotate the cross
  contribution's *direction* (inject style via the cross value path) while keeping
  its norm flat. The where/what attribution rests on the magnitude argument
  (self+MLP adds dwarf cross adds) + cross_frac invariance, not on a claim that
  cross *content* is identical. A direction-resolved follow-up (cosine of the
  per-pathway LoRA delta-contribution) would close this fully.
- Norms are over the full flattened latent (4096 patches × 2048 dim); only ratios
  are meaningful, not absolute values.
- Probes run with `compile_blocks=False` (the eager hooks/mirror must run every
  step). Generation is otherwise unchanged.

## Reproduce

```
uv run python bench/cross_attn_drive/attn_evolution.py   --captions 12 --seeds 2 --tags "speech bubble,japanese text,english text" [--lora_weight …]
uv run python bench/cross_attn_drive/attn_contribution.py --captions 12 --seeds 2 [--tags "@sincos"] [--lora_weight …]
```

Results: `bench/cross_attn_drive/results/*-contrib_{base,sincos}/` (+ `attn_contribution.png`),
`*-tag_{base,sincos2}/`, `*-sincostag_{base,lora}/`.
