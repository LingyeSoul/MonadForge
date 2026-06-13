"""Per-method network bolt-ons (easycontrol / soft_tokens).

These attach to a frozen-DiT or LoRA-adapted DiT depending on the method:
- ``easycontrol`` — extended self-attention image conditioning + per-block cond LoRA.
- ``soft_tokens`` — SoftREPA per-layer × per-t soft text tokens.

The classic LoRA / OrthoLoRA / T-LoRA / HydraLoRA family lives in
``networks.lora_anima`` because of its size and internal structure. (The
``postfix`` method was archived — see ``_archive/postfix/``; ``ip_adapter`` was
downgraded to a bench probe — see ``bench/ip_adapter/``.)
"""
