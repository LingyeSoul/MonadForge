# bench/moss_spectrum

Wall-clock and fidelity of Spectrum (`library/moss/spectrum.py`) on MOSS-Image against
the plain 50-step reference sampler, plus plain fewer-step arms. Every arm runs
`library.moss.generate.denoise`, the path `scripts/moss/generate.py` ships.

```bash
make daemon-run ARGS="bench/moss_spectrum/run_bench.py --label base3 \
    --models base --prompts anime_sunset,danbooru_tags,ja_manga"
make daemon-run ARGS="bench/moss_spectrum/run_bench.py --label sft5 --models sft \
    --prompts anime_sunset,danbooru_tags,zh_fridge,ja_manga,ko_neon --arms p50,s50,s30"
```

Arms: `p<N>` plain N steps, `s<N>[:k=v,…]` Spectrum (`ws fw warm stop w m lam K cal`,
see the script docstring); `b=1` on any arm uses the batched cond+uncond forward. The
first arm is the reference for speed-up and latent rel-L2.

Output: `results/<stamp>-<label>/result.json` (`metrics.summary` per `model/arm`:
mean seconds, speed-up, full forwards, peak GiB, mean final-latent rel-L2 vs the
reference; `metrics.rows` per image) and `grid_<model>_<prompt>.png` (every arm side by
side, 512 px tiles). Full-resolution PNGs go to
`/media/sorryhyun/data/moss_image/bench/<label>/` (`metrics.image_dir`); TE
embeds are cached in `/media/sorryhyun/data/moss_image/bench/_te_cache/`.

`--compile` compiles every block before the warm-up (whole run; compare two runs for
eager vs compiled); the warm-up time lands in `metrics.warmup_seconds`.

`--decode_only <image dir>` decodes saved latents and writes the envelope with no DiT.

Rel-L2 is a drift number, not a quality score: a plain 30-step arm drifts from the
50-step one without being worse. Judge by the grids. Numbers and verdicts:
`project/moss_lora/spectrum.md`.
