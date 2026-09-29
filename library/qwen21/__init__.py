"""Qwen-Image-2.1 LoRA sidecar, ported from upstream qwen21 at 50bf09d5.

This is not Anima: RGBA VAE (/16, 64 latent channels), unpadded Qwen3-VL text,
32 single-stream DiT blocks and Qwen's flow schedule stay isolated here.
Weights retain official Diffusers directories/shards and add ComfyUI single
files; see requests.py for CLI paths and defaults.
The WebUI config/preprocess workspaces submit daemon jobs; the CLI adds --queue to the same
cache/train/generate commands. Hardware auto-sizing stays in the worker, not
in the browser. Cache-then-train uses separate child processes in one job.
Run tasks.py qwen21-cache, qwen21-train or qwen21-generate for their help.
"""
