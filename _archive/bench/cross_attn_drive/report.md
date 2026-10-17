# Cross-attn-drive — tag-headroom exploration report

Consolidated record of the cross-attn / tag-rendering exploration (2026-06-25 → 06-26).
Companion to `how_to_observe.md` (how to read the montages). All verdicts are
detector-free + eye-adjudicated (no quality reward exists for Anima — montage is the gate).

## The question

Do tags that "need more timesteps" render finer if we keep cross-attn driving later
in σ, retain a tag pattern late, or otherwise lever the cross-attn path? And does a
fine-tuned (artist) LoRA change the cross-attn pattern vs base?

## Levers tested on the BASE model — all KILL

| Lever | Probe | Verdict |
|---|---|---|
| **Boost late cross-attn** | Phase-0c `tag_boost_scale` (extrapolate `embed_alt + s·(embed−embed_alt)` below σ-cut) | KILL — inert when safe (s≤1.25), destroys image when strong (s=3.0). Rescale trap. |
| **Retain tag pattern late** | commitment-σ data | KILL — speech bubble 84% committed by σ0.8; retaining just rescales a decided feature. |
| **Regularize to a "good" seed-mode** | eye on 0a instability strips | KILL *on base* — flagged tags are uniformly-mediocre, not bimodal (no good mode to lock onto). **Reopened by sincos2 — see below.** |
| **Freeze cross-attn late ("keep pushing same direction")** | `ANIMA_TEXT_KNOCKOUT_SIGMA` (proxy) + faithful `ANIMA_FREEZE_GUIDANCE_SIGMA` (keep magnitude, lock direction) | KILL — instability barely moves (−2.4/−2.7/+1.0% @0.85; sign-FLIPS @0.9 = noise). influence rose but wangle deaf to it. |

**Unifying result: the tag-region "wangle" is BASE-OWNED stochasticity**, confirmed on
three independent probes (x̂₀-contradiction, text-knockout, faithful direction-freeze)
× two cutoffs. You cannot boost / retain / freeze / regularize it via the cross-attn
path — it does not live there. Text-class tags (glyphs, QR) are a **capability/data
wall**: the model renders them as consistent gibberish across seeds (uniform-mediocre),
clean when the tag is dropped (drop@0 sidesteps the failed sub-task).

## Base vs fine-tuned LoRA (anima_sincos2) — the LABEL law

Paired `tag_influence` 0a, base vs `anima_sincos2` (rank-32, trained on all `sincos/*`),
identical captions/seeds. Cross-attn influence change tracks **training-label frequency**:

| tag | labeled in sincos training | influence Δ | instability Δ | read |
|---|---|---|---|---|
| speech bubble | 19/334 (6%) | **+13.4%** | +5.9% | text MODESTLY more legible (ceiling↑), no broken seed = **good variance**; cross-attn-mediated gain |
| japanese text | 2/334 (1%) | −1.6% (flat) | +2.8% | no change — JP text in sincos art is mostly UNTAGGED → baked into style, not cross-attn |
| english text | 0/334 (0%) | −9.9% | −13.4% | BROKE (QR/hands) = **bad variance / OOD degradation** |

**Law:** cross-attn learns the **LABELED** feature; untagged-but-present content is
baked into style/unconditional, invisible to the cross-attn token. "Does the LoRA
render X differently?" ≠ "did cross-attn for the X token change?" — only if X was labeled.

## Metric caveat (a counterexample to the bench premise)

`instability_rel` (region seed-variance) is the bench's failure proxy — but it
**conflates GOOD variance (ceiling raised, no broken seed) with BAD variance (breakage)**.
sincos2 proves it: speech bubble (+5.9%, no broken seed, better-on-some) and english
text (broke) are the *same* instability axis with *opposite* meaning. → read this axis
as **ceiling (best-of-N) / modality (did a tight good mode form)**, not raw variance.
The "regularize-to-good-mode = KILL" verdict was conditional on base being
uniformly-mediocre; a LoRA can CREATE a good mode that fires intermittently (raises
ceiling AND variance together), in which case the lever is consistency/selection.

## Open (not run — paused here)

- **label-retrain test of "increased numbers → better image":** label japanese-text
  (+ speech bubble) in sincos and retrain → does the tag show **influence↑ AND
  modality↑ AND ceiling↑** together? Confirms the LABEL law + turns baked JP text into
  a promptable cross-attn feature + cleanly tests the user's "more drive → better, just
  not every seed" thesis. Needs a captioning pass (Anima Tagger) + one rank-32 retrain.
- **Phase-0e modality probe:** detector-free test of whether a tight good mode formed
  (bimodal vs uniform region-distribution), to replace raw-variance reading.
- **Literal cross-attn attention maps:** no ready tool — would need a DAAM-style hook on
  `attention_with_lse` (`return_attn_probs` exists, used only for LSE).

## Artifacts

- **Probe code:** `tag_influence.py` (`--phase 0c` boost; `--lora-weight` base-vs-LoRA;
  `--montage-px 0` native-res sheets). Hooks in `library/inference/generation.py`:
  `tag_boost_scale` param, `ANIMA_TEXT_KNOCKOUT_SIGMA` + `ANIMA_FREEZE_GUIDANCE_SIGMA` env.
- **Result dirs** (`bench/cross_attn_drive/results/`): `*-phase0_run` (0a/0b),
  `*-boost_phase0c` / `*-boost_mild` (boost), `*-ko_control` / `*-ko_knockout`
  (knockout), `*-freeze_dir` / `*-freeze_dir_p9` (freeze), `*-sincos2_base` /
  `*-sincos2_lora` (LoRA compare).
- **Memory:** `project_tag_boost_late_sigma_kill`, `project_lora_crossattn_learns_labeled_only`.
