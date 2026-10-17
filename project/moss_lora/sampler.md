# ER-SDE and CNS on MOSS-Image (2026-10-17)

`denoise(sampler="er_sde")` runs Anima's `ERSDESampler` on MOSS's snapped shift-5 grid,
fed `x̂₀ = x_t − σ·v`. Shipped as `GenerateRequest.sampler` (default `er_sde` since 2026-10-17, was `euler`). CNS
(`library/moss/cns.py`) recolors its noise. It stays a bench knob (`sde=1,cns=1`).

All runs: Base, 1024², CFG 3, uncond skip 22, 50 steps, compiled, seed 42. The image dir is
`/media/sorryhyun/data/moss_image/bench/cmp_compile/`; results are under
`bench/moss_spectrum/results/20261017-00{48,105}-cmp_compile/`.

## ER-SDE vs Euler

Same cost (52.5 vs 52.2 s). It produces a different sample at the same seed (latent
rel-L2 1.22), so this is a lean, not a quality ranking. Over 3 prompts, er_sde gave
sharper line work and denser detail: manga hatching and speed lines, the cherry-tree
background. It also drifts from some prompt details: the sunset light is mostly gone,
blonde turns orange, and a yellow tie appears that the prompt did not ask for. 待ってくれ！
stays correct. With Spectrum (s50 fw 0.5) it keeps the p50-er_sde layout. The forecast is
about as close to its own p50 as under Euler: rel-L2 0.42 / 0.61 / 0.58 vs Euler
0.35 / 0.36 / 0.66.

## CNS

γ is calibrated from 10 Euler trajectories (5 bench prompts × seeds 0, 1), with the FFT
on the un-patchified 32 ch /8 grid (`bench/moss_spectrum/cns_calibrate.py` →
`output/moss/cns_gamma.npz`). **The precondition holds**: σ at γ = 0.5 falls
monotonically from 0.94 for the lowest band to 0.41 for the highest. Spread +0.53,
aggregate σ50 0.61 (Anima: +0.41…+0.52, ≈0.45–0.49).

A/B against er_sde white noise with the same white draw (5 prompts, p50 and s50 fw 0.5):
composition is kept and details move (rel-L2 0.43). By eye, at grid scale plus full-res
crops of the zh note and the ja_manga face:

- zh_fridge note: all three lines correct in all four arms, and the handwriting looks
  the same.
- ja_manga: p50 + CNS **broke the eyes** (mismatched, a ring around one). s50 + CNS
  removed a stray black tie fragment that s50 white had.
- anime_sunset s50: CNS brought back p50's red neckerchief, where s50 white had a
  white ribbon.
- zh_fridge p50: the white arm has a green smear at the fridge edge, and CNS doesn't.
- danbooru_tags, ko_neon: a wash. Background signage is pseudo-glyphs in every arm.

Net: a near-tie with one defect and two or three cleanups at one seed. Not enough to
ship. The calibration prompts overlap the eval prompts (different seeds).

## Open

- CNS over more seeds and prompts held out from calibration, before it gets a
  request field.
- er_sde on SFT (already the glossier model). With a LoRA: see below.

## With the caststation LoRA (Base, `caststation_r16` ×1.0, 30 steps, eager)

`project/moss_lora/src/lora_ab.py --arms euler:0.0,euler:1.0,er_sde:1.0,cns:1.0`, 3 prompts ×
seeds 1–2. The two Euler columns are the existing floor. Grids:
`output/moss/lora/ab_caststation_r16_30st/grid_base_p*_s*_euler0.0_euler1.0_er_sde1.0_cns1.0.png`.

- **The LoRA style survives er_sde**: pale aqua palette, thin clean lines, flat fills,
  mostly white backgrounds. On top of that, er_sde is more finished. Hands are the clearest
  gain: the phone hands on p1 s1/s2 are clean, where Euler ×1.0 has a gloved claw.
- It costs some of the set's identity and some prompt detail. p2 s1 trades the set's
  dark round eyes for generic blue-grey ones. p2 s2 loses "sitting on a bench" (frontal
  bust in front of the bench). p1 s1 adds a green backing panel with a white sticker
  outline. Compositions lean frontal and centred, the same house-composition drift e20
  showed.
- CNS vs er_sde white is again a wash. It removed a stray signature scribble (p1 s2) and
  garbled the "01" arm tattoo into "0C" (p0 s1). Everything else is near-identical.
