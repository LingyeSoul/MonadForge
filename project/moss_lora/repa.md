# REPA on MOSS-Image

**Verdict: parked.** Between-training-seed variance swamps any effect (blind 3-seed
judge, REPA 8 / control 11 / tie 5; [below](#three-training-seeds-blind-judge-2026-10-20)).

Anima's relational REPA (`docs/methods/repa.md`) ported to the MOSS trainer:
`library/moss/repa.py`, `--repa_weight` / `--repa_layer` / `--repa_pe_root` on
`TrainRequest` (off by default). Targets are the Anima PE-Spatial sidecars of the
resized PNG each latent was encoded from (MOSS latents are not resampled, so the crop
matches). (2026-10-20)

## Layer probe (`bench/moss_repa/probe_layer_sigma_cka.py`)

Anima's zero-training probe (`_archive/bench/repa/results.md`): centered CKA of each
block's image tokens, pooled to the PE grid, vs the image's PE-Spatial tokens; minus a
mismatched same-grid image; minus the σ = 1 floor (`useful_gap`). Base, 96 images from
six MOSS caches over 7 PE grids, 11 σ, 15 min. `results/20261020-1441-base`.

- **No dead zone.** Anima is bimodal (blocks 7–11, nothing at 12–20, a deep peak).
  MOSS's band-mean `useful_gap` (raw target) is already 0.30 at block 0, reaches
  0.40–0.45 by blocks 7–12 and stays there to block 24, then falls to 0.33–0.37.
  The DoG target rises from 0.11 to ~0.18 by block 9 and to 0.20 at 23–24. Block 0's
  0.30 says the FLUX.2 latent itself carries much of the PE structure.
- **σ**: flat to σ 0.8, decaying from 0.85, ~0 by 0.95 at every layer (Anima's shape).
- **Massive channels break the uncentered Gram.** Mean pairwise cosine of the image
  tokens is 0.999 on blocks 0–12 (still 0.68–0.78 at 38–41), so the DoG-Gram loss reads
  1.03: the Gram is all ones. Centered over tokens with the 18 massive channels
  (`calibration/massive_channels.npz`) dropped it is 0.077–0.093 on blocks 0–30
  (`results/20261020-1444-dit_side`, 24 images). The training term uses that form;
  uncentered it would mainly ask the LoRA to cancel the massive-activation DC.
  Anima's base loss at its block 8 was 0.15, so MOSS has less headroom.

Layer 11 (26 % depth): the top of the early plateau (raw 0.45, DoG 0.19), before the
uncond-skip block 22 and the deep half that was net-negative on Anima.

## A/B against the channel-scaling run

caststation, `caststation_soup_lr5e5_s0` (control: α 0.5, `dc_isolate`, lr 5e-5,
20 epochs, seed 0, `--plain_caption_rate 0`) vs `caststation_soup_lr5e5_s0_repa11`
(+ `--repa_weight 0.2 --repa_layer 11`). Weight 0.2 puts REPA at ~7.5 % of the loss at
step 1 (Anima's 0.05: ~10 %).

- **Cost**: none. 1.85 s/step and 12.09 GB peak, the same as the control.
- **Loss**: both runs draw the same σ and ε every step, so steps pair. FM loss ratio of
  means −2.0 / −0.8 / −0.2 / +2.7 % over steps 1–150 / –300 / –450 / –600, medians
  −2.6 / +0.2 / +0.4 / 0.0 %, REPA lower on 47–55 % of steps: no difference.
- **REPA term**: 0.072 → 0.051 (−30 %; Anima's block-8 run −60 %), still drifting down.
- **Images** (Space 004 / 017 × 4 seeds, ER-SDE + Spectrum;
  `output/tests/ab_caststation_soup/s0_repa11`, sheets `pairs_repa11_*.jpg`, crops
  `crop_repa11_*.jpg`; Claude's read, user verdict pending): the style holds (flat pale
  fills, line weight). REPA reads a little more saturated with slightly heavier lines,
  and on 004 frames the characters a little closer. Compositions diverge seed for
  seed. No anatomy gain visible in 8 cells. One seed of one set.
- **User read (017 s2)**: w0.2 is the most physically plausible of the three arms.
  Full-res crops (`crop_017s2_half{0,1}.jpg`, `crop_017s{0,1,3}_3arm.jpg`) agree on
  that cell: the drinks stand on a table in consistent perspective, Shiroko and Yuuka sit
  *in* chairs whose frames show, the sandcastle and crab are on the sand. In the control,
  Shiroko floats in front of an empty lounger and the sandcastle sits on a wooden table.
  w0.6 has a transparent ghost object by Shiroko's knee and drinks cut off at the bottom
  edge. Claude's first read called the folded leg in the w0.2 cell a defect. Seen at full
  res it reads as Hoshino's leg, so that call is withdrawn. On seeds 0 / 1 / 3 the arms
  are closer. The one pattern that repeats is seating: both REPA arms put the characters
  in chairs with visible structure, where the control often has them on the sand in front
  of a lounger. That is support/contact, the kind of layout REPA is meant to carry.

## Weight 0.6 (`caststation_soup_lr5e5_s0_repa11_w06`)

Same run at 3× the weight (user's question: does a stronger term help?).
Sheets `pairs_repa11_w06_*.jpg` (ctrl / w0.2 / w0.6), crop `crop_w06_space004s3.jpg`.

- **REPA term**: 0.059 / 0.049 / 0.048 / 0.045 over the four 150-step windows, vs
  0.068 / 0.057 / 0.055 / 0.051 at w0.2. Three times the weight buys ~12 % more
  alignment: the term saturates, consistent with the small headroom above.
- **FM loss**: ratio of means vs control −1.9 / −0.4 / +0.2 / +3.4 %, medians −2.8 /
  +1.1 / +1.0 / +0.9 %, lower on 46–54 % of steps. Still noise, drifting slightly the
  wrong way late.
- **Images** (Claude's read): style kept. On 004 the ensemble packs tighter and closer
  than at w0.2 (which was already closer than the control), with more overlap and
  clutter, and costumes gain ornament the prompt does not ask for (Hina's coat, chains
  and red tie on s3). 017 is close to w0.2. No anatomy fix to set against it.

Reading: more weight moves the image more without buying alignment, i.e. it touches
what the style does not need. Not a direction to push.

## Three training seeds, blind judge (2026-10-20)

`…_s{1001,1002}_repa11` (w0.2) against the soup controls `caststation_soup_lr5e5_s{1001,1002}`
(same recipe), plus the seed-0 pair: 3 training seeds × 8 cells = 24 pairs. An Opus
general-purpose agent judged them blind (sides shuffled per pair, key held outside its
folder) with criteria in this order: physical plausibility (support, contact, placement),
anatomy, prompt adherence, style.

- **Overall**: REPA 8, control 11, tie 5. Two-sided sign test p = 0.65.
- **By training seed** (decided pairs): seed 0 REPA 6–0; seed 1001 control 5–2; seed
  1002 control 6–0. Within a seed, one LoRA wins across all four sampling seeds; across
  seeds the direction flips. The 1002 REPA LoRA blew up Mika's and Koharu's heads on all
  four space004 cells, and the seed-0 control leaked forbidden ID badges. Those are
  traits of a single trained LoRA, and they decided the pairs.
- The judge found **no consistent signature** to sort the 48 images into two families
  (saturation, edges, blur, palette all flip direction between pairs). Both arms share
  the same failure modes: oversized heads, airborne figures, sitting in front of empty
  chairs, duplicated props. The seed-0 017 s2 cell that prompted the user's read was
  judged a tie.

Reading: between-training-seed variance swamps any REPA effect. The seed-0 advantage
(including the grounding read above) does not replicate. Effective n is 3 LoRAs, and 1
favours REPA, 2 the control. REPA stays implemented and off by default; the line is
parked.

## Open

- Parked. If revisited: a much larger seed count, or a prompt set with simpler layouts
  where one bad seed does not decide a pair. The σ taper is the untested lever (the probe says the term has no
  target above σ ≈ 0.85; MOSS training samples there on about a third of the steps).
- Layer 8 vs 11 is a small lever (both on the plateau).
