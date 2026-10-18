# bench/moss_sinks

Attention sinks / registers on MOSS-Image: does a small set of keys take most of the
attention, and if so, where is it? Tier 1 §1 of
`docs/proposal/moss_anima_negatives_revisit.md`; started 2026-10-18.

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

## Next

1. **Null or register**: per block and head, the template's share of the attention
   *output* (`‖Σ_template p·v‖` against `‖Σ p·v‖`) and the template value norms against
   image values. A small output share at a large mass share means an OFF-gate. Cheap: an
   extra stat in `probe_attention.py`, same run shape.
2. **Does the prompt move it**: caption length and format (short tag bag, long layout
   JSON) against the template mass and the per-block head counts above.
3. **Only then** anything that edits the template: a shortened or altered template is off
   the training distribution, as real pads were on Anima (they collapsed the image).
   Null-TTA (arXiv 2511.20889, test-time optimisation of the CFG null embedding) lands
   here: on MOSS the null embedding *is* the template span, so optimising it moves the
   sink, not a semantic direction. Read (1) before considering it.
4. Test-time registers: dropped. The model already ships fixed registers.
