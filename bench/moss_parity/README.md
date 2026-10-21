# bench/moss_parity

Phase 0 gate of `project/moss_lora/`: `library.moss` (TE encode, vendored DiT, `denoise` unbatched and batched) against the reference `MOSSImagePipeline` at the same seed and embeds. Passed at `75d9d01a`: unbatched is bit-exact (TE embeds and final latents); batched is ~25 % slower and drifts to a different sample, so it stays opt-in.
Run: `make daemon-run ARGS="bench/moss_parity/parity.py"`; results land in `results/<ts>[-label]/`.
