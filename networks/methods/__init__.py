"""Per-method network bolt-ons (ip_adapter / easycontrol / soft_tokens).

These attach to a frozen-DiT or LoRA-adapted DiT depending on the method:
- ``ip_adapter`` — image cross-attention via Perceiver resampler + ip_kv heads.
- ``easycontrol`` — extended self-attention image conditioning + per-block cond LoRA.
- ``soft_tokens`` — SoftREPA per-layer × per-t soft text tokens.

The classic LoRA / OrthoLoRA / T-LoRA / HydraLoRA / ReFT family lives in
``networks.lora_anima`` because of its size and internal structure. (The
``postfix`` method was archived — see ``_archive/postfix/``.)
"""
