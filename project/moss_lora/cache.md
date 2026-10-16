# moss_lora — Phase 2 caches (2026-10-16)

Code: `library/moss/cache.py` (`run_cache`, `list_items`, `load_item`), sidecar
`scripts/moss/cache.py`, flags `library/moss/requests.py::CacheRequest`, test
`tests/test_moss_cache.py` (14 CPU tests: pack/unpack inverse + FLUX.2 patchify order,
sizing invariants, list/load, stem collisions).

## Layout

Flat folder, two files per image:

- `{stem}.moss_te.safetensors`: `embeds (S, 4096)` bf16 at the caption's true length
  (never padded). Metadata includes `caption`, `caption_sha256`, `n_text`,
  `max_text_tokens` and `encoder`. It is re-encoded only when the caption's sha or the
  truncation length changes.
- `{stem}_{W}x{H}.moss_latent.safetensors`: `latent (128, H/16, W/16)` fp32, already
  packed and BN-normalised, so the DiT takes it as-is. It is re-encoded when the source
  image is newer. A latent at a different size for the same stem is deleted.
- `cache_report.json`: timings and token statistics.

Sizing: native aspect, both sides multiples of 16, area at most `resolution²` (default
1024), **never upscaled**. An image already on the 16-px grid within 10 % of the cap is
encoded as-is: the Anima resized tree is ~1.05 MP on that grid, so it isn't resampled a
second time. Anything else is LANCZOS cover-resized and then centre-cropped by less than
16 px. Alpha is composited onto white.

## First run: `post_image_dataset/resized/channel_(caststation)`

This is the same folder qwen21 used, so the two lines can be compared. The 30 images use
their existing Anima tag-bag captions (open question 1 is still open).
Output is `output/moss/cache/channel_(caststation)/`; `output/moss` → `/media/sorryhyun/new/anima_output/moss`.

| | value |
|---|---|
| images | 30, 0 resampled, 21 distinct shapes |
| text tokens (chat template incl.) | min 127 / median 216.5 / max 406 |
| image tokens | min 4032 / median 4096 / max 4200 |
| joint max | 4502 |
| TE pass | load 2.1 s + encode 286.6 s (**9.6 s / caption**), peak 2.43 GiB |
| VAE pass (fp32) | 7.6 s for 30, peak 2.85 GiB |
| on disk | 117 MB (~3.9 MB / image) |

**The TE pass is PCIe-bound.** `loader._offload_by_layer` pages every decoder layer
(~16 GB) across the bus for each caption, and at peak only 2.4 GiB of the card is in use.
At this rate a 1000-image set takes ~2.7 h. The fix is in `loader.py`, not here: keep as
many decoder layers resident as free VRAM allows (roughly 10 GiB of the ~16 GiB) and page
only the rest. That should cut the per-caption time by about 3×. A layer-major encode
(each layer runs every caption before the next layer) would cut it further, but the HF
forward has no hook for that.

## VAE checks (`output/moss/cache_probe/`, two images)

| image | size | PSNR encode→decode fp32 | PSNR bf16-encode | bf16 vs fp32 latent max / mean \|Δ\| |
|---|---|---|---|---|
| 10391952 | 896×1152 | 36.67 dB | 36.62 dB | 0.148 / 0.0057 |
| 12248674 | 1008×1040 | 36.27 dB | 36.19 dB | 0.154 / 0.0063 |

The packed latent has std ≈ 0.84 and mean ≈ 0.06, which is what a correct BN
normalisation should give. Side-by-side sheets (original | fp32 | bf16-encode) show no
visible difference; line art and the small kana in the speech bubble survive the round
trip. A bf16 encode costs < 0.1 dB, but the whole fp32 pass takes only 7.6 s, so fp32
stays the default (`--vae_dtype bf16` exists).
