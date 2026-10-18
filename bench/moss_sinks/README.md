# bench/moss_sinks

Attention sinks / registers on MOSS-Image: does a small set of keys take most of the
attention, and if so, where is it? Tier 1 §1 of
`docs/proposal/moss_anima_negatives_revisit.md`; started 2026-10-18.

Summary of what the runs found: `docs/moss-findings/attention_sink_chat_template.md`.
This file is the run log.

```bash
make daemon-run ARGS="bench/moss_sinks/probe_attention.py --label base"
```

## Motivation

- **Anima**: `_archive/null_key/report.md` (pad-position registers, pad-tail trim). It
  closed arXiv 2607.19139 ("Text Template Tokens Are Implicit Semantic Registers in
  DiTs", Qwen-Image) because Anima has no template span. The zero-pad tail already is
  the null-attention sink ("the pad tail is analytically pure null attention"), and the
  DiT, trained only on zero pads, collapses when pads carry content.
- **Why MOSS differs**: there are no pads, and MOSS has the template span Anima lacked.
  `loader.prompt_ids` wraps every prompt in the Qwen chat template
  (`add_generation_prompt`, `enable_thinking`): `<|im_start|>user\n` before the caption
  (3 tokens), `<|im_end|>\n<|im_start|>assistant\n<think>\n` after it (7). The uncond
  prompt is `''`, so the uncond text segment is the 10 template tokens alone. The
  per-query OFF-gate the pad tail gave Anima has no other carrier in a MOSS block:
  `gate_a` is per channel and time-only. The sink, if there is one, sits in the
  template, the first text token, or image tokens.
- **Not evidence either way**: the DAVE massive channels (`project/moss_lora/dave.md`).
  The DC holds 0.98–0.99 of the image-token energy on blocks 0–18, so channels 1356 / 42
  are a near-constant offset on *every* image token (built in block 0; the
  `video_embedder` bias is only 7× / 15× median there), not the few-token outliers that
  accompany LLM sinks. A shared DC also gives uniform attention over image keys a
  null-like output, which is a sink-free way to write nothing.

## Probe

`probe_attention.py` (docstring has the details). The shipped sampler (Euler, CFG 3,
uncond skip 22, reference per-sample forward), Base, 2 prompts (`anime_sunset`,
`ja_manga`) × 4 seeds × 30 steps at 1024², read on 8 of the 30 steps. On those steps
every joint block's attention is recomputed with an explicit softmax (the block output
still comes from SDPA). Keys are split into image / `tok0` (`<|im_start|>`) / `prefix` /
`caption` / `suffix`; cond and uncond are kept apart. Per block and head:

- the mass each query class (image / text) puts on each key class;
- the top-1 % key share: the head's mean attention per key over the image (or text)
  queries, summed over its top `ceil(0.01·N)` keys (uniform = 0.01);
- a per-token norm census on the block output with the 18 channels of
  `library/moss/calibration/massive_channels.npz` left out (with them in, DC 0.99 caps
  image max/median near 1 on blocks 0–18 by construction); raw norms are kept too.

**Gate**: on most blocks, ≥ 25 % of heads put ≥ 30 % of their image-query mass on
≤ 1 % of the keys; or massive-channel-excluded image norm max/median ≥ 10 on most
blocks. If neither holds, close.

Output: `results/<stamp>-<label>/result.json` (gate, per-block head fraction, top-1 %
share, mass by key class, norms; cond and uncond), `per_block.npz` (per seed × step ×
block × head), `top_keys.json` (per block, the 8 keys receiving the most image-query
mass, with their class) and `probe.png`.

## Results

**2026-10-18 — smoke** (`results/20261018-1553-smoke`: 512², 4 steps, 1 prompt × 2
seeds). Plumbing check only: at 512² the 10 template tokens alone are ~1 % of the keys,
so the top-1 % share passes easily. The template took 25–55 % of image-query mass on
blocks 8–30; excluded-channel image norm max/median 1.1–1.7 on every block.

**2026-10-18 — base** (`results/20261018-1601-base`, job `20261018-155337-1cf9d5`;
anime_sunset + ja_manga × 4 seeds, 1024², steps 0 3 7 11 15 19 23 29). **Gate passes on
attention (35 / 42 blocks), fails on norm (0 / 42). The sink is the 10 chat-template
tokens.**

- **Where**: across heads, the top-1 % key share tracks the template's mass at
  r = 0.98. Per key, image queries give `<|im_start|>` 0.06–0.07 of their mass, each
  prefix token 0.044–0.048 and each suffix token 0.016–0.017, against 0.0002 for an
  image token (~300× / ~220× / ~80×) and 0.0005–0.0006 for a caption token (~2.5×). The
  head-mean top-8 keys are template tokens on nearly every block (`tok0` and both prefix
  tokens in all 42); image position 0 (top-left corner) reaches the top 8 on a few early
  blocks only.
- **How much**: the template takes 0.24–0.34 of the image-query mass (head and block
  mean), U-shaped over the run (step 0 0.31–0.32, mid 0.24–0.25, step 29 0.31–0.34).
  Uncond is the same picture, a little higher at step 0 (0.38). Text queries put ~0.35 of
  their mass on the template as well.
- **By block**: strongest on 19–21 and 25–29 (~10 of 24 heads put ≥ 0.5 of their mass on
  the template), weakest on 0, 4, 7 and 35–41 (≤ 3 heads). 36–41 keep 0.8–0.88 of the
  mass on image keys.
- **Norms**: image tokens have no outliers (massive channels out, max/median 1.1–1.8 on
  every block; raw 1.06–1.6). The text side is where the high norms are: over the image
  median, prefix tokens are 21× (max 39× over blocks), suffix 10×, `<|im_start|>` 7×,
  caption 5.5×. Caveat: only the 18 image-derived massive channels are left out; the
  text stream's own massive channels have not been ranked.

Reading: the role Anima's zero-pad tail had is played on MOSS by a fixed, trained-in
register span, the chat template. Its prefix (`<|im_start|>user\n`) has identical TE
states for every prompt (Qwen is causal), and it is the only text the uncond branch
sees. This is the 2607.19139 substrate. Not yet known: whether the sink *writes* anything
(a null-valued OFF-gate like Anima's pads, or a semantic register as the paper claims).

**2026-10-18 — base-write** (`results/20261018-1616-base-write`, same run shape, with the
write stats). Values are centred on the image keys' mean v̄_img first: the values carry
the massive-channel DC, and on raw values every write is near-constant (image write
coherence 0.92–0.98 in the smoke), so raw shares cannot separate the two. Since Σp = 1,
`out − v̄_img = Σ p·(v − v̄_img)`, and a key whose value is v̄_img writes nothing.
**The template is not a null sink: it writes a per-head bias.**

- **Write share**: the template writes 0.15–0.51 of the centred image-query output
  (blocks 19–21 and 25–29 ≈ 0.46–0.51), 1.13–2.1× its mass share. A null sink would
  sit well below 1×.
- **One vector for every query**: coherence of the template write across image queries
  0.89–0.98 on blocks 0–40 (0.34 on 41), against 0.18–0.65 for the image keys' write.
  Per head, it adds a fixed vector to every image token, scaled by the mass that query
  parks on the template.
- **Value norms** (centred, over image): `<|im_start|>` 0.43–0.9 on blocks 0–37 (the
  nearest to a null value), prefix 0.5–1.5 past block 7, suffix 1.0–2.4. On blocks 1–7
  caption and suffix values are 3.5–9.4×.
- **Cond vs uncond**: template write cosine 0.93–0.96 on blocks 0–21, 0.76–0.94 after
  block 22 (the uncond row skips 22). Uncond's template takes a little more mass and
  writes a little more (0.16–0.62 against 0.15–0.51). Under CFG the template write
  mostly cancels in cond − uncond.
- **Across prompts** (cond, anime_sunset vs ja_manga): cosine 0.70–0.88, lower than
  cond vs uncond. The two prompts also run different trajectories, so this mixes
  template content with the image state that sets each query's weights over the 10
  tokens. It is not yet a read on caption semantics in the template.

Reading: on MOSS the template does two jobs. It is the OFF-gate (it takes 0.11–0.45 of
the mass), and it is a learned per-head bias channel into every image token, mostly
prompt-independent. Anima's pads did only the first.

**2026-10-18 — LoRA arms** (`results/20261018-1632-lora-lr1e4`,
`results/20261018-1645-lora-lr5e5dc`; same run shape, `--lora` at ×1.0). Two caststation
layout LoRAs (r32 α128, SVD init): `lr1e4_svd` (no channel scaling, no `dc_isolate`) and
`lr5e5_svd_cs05_dc` (the current defaults). Step 0 is the clean read: the same x_T and
σ, so only the LoRA differs. Base's seed-to-seed CV of step-0 template mass is 0.5 %, so
paired-by-seed shifts are resolved on 35 / 42 and 33 / 42 blocks (|t| > 4). **The LoRAs
move the sink by a small, systematic amount, and its write keeps its direction.**

| | lr1e4_svd | lr5e5_svd_cs05_dc |
|---|---|---|
| step-0 template mass, mean \|Δ\| / base | 5.4 % | 3.0 % |
| largest block shifts | −20 % (6), −19 % (5, 17), +18 % (23), −12 % (1) | −11 % (8), −9 % (40), +8 % (7) |
| pooled mass / write, mean \|Δ\| / base | 7.1 % / 4.8 % | 4.2 % / 2.5 % |
| template write vs base, cosine (mean, min) | 0.905, 0.825 (41) | 0.948, 0.852 (41) |
| template write norm ratio | 1.06 (0.78–1.33) | 1.04 (0.92–1.14) |

For scale: within one Base run the template mass swings ±15 % over the steps (0.24 →
0.34), and cond vs uncond at the same latent sits at write cosine 0.93–0.96. The pooled
cosines also include the LoRA's change of trajectory (different images from step 1 on),
so they are a lower bound on how well the write direction is kept. The two arms differ
in lr, channel scaling and `dc_isolate` at once, so the 2× gap between them is not
attributed. Nor is the shift itself: it can come from the template rows' K / V or from
the image rows' Q, since the LoRA edits both.

Reading: a style LoRA does not lean on the template channel much. The per-head bias
stays where it was (cos ≥ 0.9, norm ±5 %), and the gate opens or closes by a few percent
on most blocks, up to ~20 % on a handful with the stronger LoRA.

## Next

1. ~~**Null or register**~~: done (base-write above). A bias channel, not null.
2. **Does the prompt move it**: caption length and format (short tag bag, long layout
   JSON) against the template mass and the per-block head counts above.
3. **Only then** anything that edits the template: a shortened or altered template is off
   the training distribution, as real pads were on Anima (they collapsed the image).
   Null-TTA (arXiv 2511.20889, test-time optimisation of the CFG null embedding) lands
   here: on MOSS the null embedding *is* the template span, so optimising it moves the
   sink, not a semantic direction. Read (1) before considering it.
4. **LoRA and the template rows**: `library/moss/lora.py` wraps whole linears, so the 10
   template rows get the LoRA delta in every block's q / k / v / o. Given (1), an edit to
   their values is a global per-head shift of every image token, and an edit to their
   keys changes how far each head shuts off. That is a cheap global lever, the token-axis
   counterpart of what `dc_isolate` removes on the channel axis. First read done (LoRA
   arms above): the shift is small (3–5 % mean, ≤ 20 % on a few blocks, write cosine
   ≥ 0.9). Masked renders (`render_template_mask.py`, 2026-10-18): style stays, content
   details change on 4 / 8 seeds. Open: is masking better or worse, read on
   layout-JSON prompts and more seeds.
5. Test-time registers: dropped. The model already ships fixed registers.
