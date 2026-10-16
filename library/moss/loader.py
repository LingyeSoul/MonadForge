"""MOSS-Image loading under a 16 GB budget: TE → encode → free → DiT → (DiT off) → VAE.

    text_encoder  Qwen3.5-9B (+ unused vision tower)  16.8 GB  <- larger than the card
    transformer   5B, 42 single-stream blocks         10.2 GB  (9.7 GiB resident, bf16)
    vae           FLUX.2 AutoencoderKLFlux2            0.25 GB (decoded in fp32)

The TE runs with whole decoder layers paged onto the card one at a time. Leaf-level
offload hooks break on Qwen3.5's gated-DeltaNet layers, which read
``conv1d.weight`` functionally instead of calling the module.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from library.moss.model import MOSSImageTransformer
from library.moss.requests import variant_dir

MAX_SEQUENCE_LENGTH = 4096


def empty_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def free_vram_gb(device: int = 0) -> float:
    free, _total = torch.cuda.mem_get_info(device)
    return free / 1024**3


def _offload_by_layer(te, device, reserve_gb):
    """Non-layer modules and as many leading decoder layers as fit in free VRAM minus
    ``reserve_gb`` stay resident; the rest page in for their own forward. The vision
    tower stays on the CPU (unused for T2I)."""
    lm = te.language_model
    for name, child in lm.named_children():
        if name != "layers":
            child.to(device)
    budget = free_vram_gb(device.index or 0) - reserve_gb
    paged = 0
    for layer in lm.layers:
        size = sum(p.numel() * p.element_size() for p in layer.parameters()) / 1024**3
        if size <= budget:
            layer.to(device)
            budget -= size
            continue
        paged += 1
        layer.register_forward_pre_hook(lambda m, a: (m.to(device), None)[1])
        layer.register_forward_hook(lambda m, a, o: (m.to("cpu"), None)[1])
    return paged


def load_text_encoder(model_dir: str | Path | None, device="cuda", reserve_gb=2.5, log=print):
    """``(tokenizer, text_encoder)`` on ``device``, layers that do not fit paged in
    per forward (``reserve_gb`` is left free for activations at 4096 tokens).
    TE and tokenizer are byte-identical between Base and SFT; Base's are used."""
    from transformers import AutoModel, AutoTokenizer

    root = variant_dir(model_dir, "base")
    tok = AutoTokenizer.from_pretrained(root / "tokenizer")
    te = AutoModel.from_pretrained(root / "text_encoder", dtype=torch.bfloat16).eval()
    paged = _offload_by_layer(te, torch.device(device), reserve_gb)
    if log:
        log(f"[te] {len(te.language_model.layers) - paged} decoder layers resident, {paged} paged")
    return tok, te


@torch.no_grad()
def encode_prompt(tok, te, text: str, device="cuda", max_length=MAX_SEQUENCE_LENGTH):
    """One prompt → ``last_hidden_state`` ``(S, 4096)`` bf16 at its true length.
    Exactly ``MOSSImagePipeline.encode_prompt``: chat template with
    ``add_generation_prompt=True, enable_thinking=True``, one prompt per forward
    (no padding — the GDN layers misbehave on it)."""
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tok(formatted, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    return te(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]


def encode_prompts(model_dir, texts, device="cuda", max_length=MAX_SEQUENCE_LENGTH, log=print):
    """Load the TE, encode each distinct text once, free it → ``{text: (S,4096) cpu}``."""
    tok, te = load_text_encoder(model_dir, device, log=log)
    out = {}
    for text in dict.fromkeys(texts):
        out[text] = encode_prompt(tok, te, text, device, max_length).cpu()
        if log:
            log(f"[te] {out[text].shape[0]:5d} tokens | {text[:60]!r}")
    del tok, te
    empty_cache()
    return out


def pad_embeds(rows, device=None):
    """List of ``(S_i, D)`` → ``embeds (B, S_max, D)``, bool ``mask (B, S_max)``."""
    length = max(r.shape[0] for r in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for i, r in enumerate(rows):
        embeds[i, : r.shape[0]] = r
        mask[i, : r.shape[0]] = True
    if device is not None:
        embeds, mask = embeds.to(device), mask.to(device)
    return embeds, mask


def load_transformer(model_dir, variant="base", dtype=torch.bfloat16, device=None):
    tr = MOSSImageTransformer.from_pretrained(
        variant_dir(model_dir, variant) / "transformer", torch_dtype=dtype
    ).eval()
    return tr.to(device) if device is not None else tr


def compile_blocks(blocks) -> None:
    """``torch.compile(dynamic=True)`` on each block's ``forward`` (instance
    attribute, so swap hooks stay outside the graph). One graph for every joint
    length; must run after ``LoRANetwork.apply_to`` / ``load_for_inference``."""
    limit = max(torch._dynamo.config.recompile_limit, 16)
    torch._dynamo.config.recompile_limit = limit
    for block in blocks:
        block.forward = torch.compile(block.forward, dynamic=True)


def load_vae(model_dir, dtype=torch.float32, device=None):
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(
        variant_dir(model_dir, "base") / "vae", torch_dtype=dtype
    ).eval()
    return vae.to(device) if device is not None else vae
