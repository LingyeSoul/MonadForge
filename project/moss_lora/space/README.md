# space/ — the A/B eval prompts

Two MOSS-Image HF Space gallery entries (`004` Blue Archive ensemble, 1776×1184, cfg 5; `017` 2880×720 beach panorama, cfg 3), the prompt set every LoRA A/B in this line renders (caption dropout, soup, REPA); `scripts/moss/generate.py --example project/moss_lora/space/004/parameters.json,project/moss_lora/space/017/parameters.json` replays each with its own prompt, negative, size, cfg, seed and uncond skip (`GenerateRequest.example`).
