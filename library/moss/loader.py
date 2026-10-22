"""MOSS-Image loading under a 16 GB budget: TE → encode → free → DiT → (DiT off) → VAE.

    text_encoder  Qwen3.5-9B (+ unused vision tower)  16.8 GB  <- larger than the card
    transformer   5B, 42 single-stream blocks         10.2 GB  (9.7 GiB resident, bf16)
    vae           FLUX.2 AutoencoderKLFlux2            0.25 GB (decoded in fp32)

The TE's 248k-row token embedding (1.9 GB) stays on the CPU — a lookup needs no
GPU — and the activation reserve is sized to the longest prompt, so at caption
lengths (< ~1k tokens) all 32 decoder layers (12.9 GB) fit on the card. Layers
that still do not fit are paged in whole, one at a time, from pinned memory; the
encode is PCIe-bound while any are (7 paged ran at ~20 % GPU, 2.3 s per caption).
Leaf-level offload hooks break on Qwen3.5's gated-DeltaNet layers, which read
``conv1d.weight`` functionally instead of calling the module. The quantized TEs
(``precision="int8"``, the default request setting, W8A8 ``library/moss/int8.py``;
``precision="fp8"``, ``library/moss/fp8.py``) put the decoder layers at 6.5 GiB,
so none page at any prompt length.
"""

from __future__ import annotations

from pathlib import Path

import torch

from library.moss.model import MOSSImageTransformer
from library.moss.requests import variant_dir
from library.runtime.device import clean_memory as empty_cache
from library.runtime.device import free_vram_gb

MAX_SEQUENCE_LENGTH = 4096
TE_PRECISIONS = ("bf16", "fp8", "int8")


def te_reserve_gb(max_tokens: int) -> float:
    """VRAM left free for the TE's activations: 2.5 GB at 4096 tokens, linear in
    length above a fixed floor."""
    return 0.75 + 1.75 * max_tokens / MAX_SEQUENCE_LENGTH


def _embed_on_cpu(embed: torch.nn.Module, device) -> None:
    """Look tokens up on the CPU and hand the rows to ``device``."""
    embed.register_forward_pre_hook(lambda m, a: (a[0].cpu(),))
    embed.register_forward_hook(lambda m, a, o: o.to(device))


def _page_layer(layer: torch.nn.Module, device) -> None:
    """Copy ``layer`` onto ``device`` for its own forward from a pinned CPU copy,
    and drop the device copy after it (no copy back: the weights are unchanged)."""
    tensors = [*layer.parameters(), *layer.buffers()]
    host = [t.data.pin_memory() for t in tensors]
    for t, h in zip(tensors, host):
        t.data = h

    def load(_m, _a):
        for t, h in zip(tensors, host):
            t.data = h.to(device, non_blocking=True)

    def drop(_m, _a, _o):
        for t, h in zip(tensors, host):
            t.data = h

    layer.register_forward_pre_hook(load)
    layer.register_forward_hook(drop)


def offload_by_layer(te, device, reserve_gb):
    """Non-layer modules but the token embedding, and as many leading decoder layers
    as fit in free VRAM minus ``reserve_gb``, stay resident; the rest page in for
    their own forward. The vision tower stays on the CPU (unused for T2I)."""
    lm = te.language_model
    for name, child in lm.named_children():
        if name == "embed_tokens":
            _embed_on_cpu(child, device)
        elif name != "layers":
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
        _page_layer(layer, device)
    return paged


def load_tokenizer(model_dir: str | Path | None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(variant_dir(model_dir, "base") / "tokenizer")


def load_text_encoder(
    model_dir: str | Path | None,
    device="cuda",
    max_tokens=MAX_SEQUENCE_LENGTH,
    log=print,
    tok=None,
    precision="bf16",
):
    """``(tokenizer, text_encoder)`` on ``device``, layers that do not fit paged in
    per forward. ``max_tokens`` is the longest prompt the caller will encode (it
    sizes the activation reserve). TE and tokenizer are byte-identical between
    Base and SFT; Base's are used. ``precision="int8"`` / ``"fp8"`` loads the
    catalog's quantized file (``library/moss/fp8.py``, fetched on first use)
    instead of the bf16 shards; the tokenizer still comes from Base."""
    from transformers import AutoModel

    if precision not in TE_PRECISIONS:
        raise ValueError(
            f"text encoder precision {precision!r}, not one of {TE_PRECISIONS}"
        )
    tok = tok or load_tokenizer(model_dir)
    if precision != "bf16":
        from library.moss.fp8 import load_fp8_text_encoder

        te = load_fp8_text_encoder(log=log, precision=precision)
    else:
        root = variant_dir(model_dir, "base")
        te = AutoModel.from_pretrained(
            root / "text_encoder", dtype=torch.bfloat16
        ).eval()
    reserve = te_reserve_gb(max_tokens)
    paged = offload_by_layer(te, torch.device(device), reserve)
    if log:
        log(
            f"[te] {precision}: {len(te.language_model.layers) - paged} decoder layers resident, "
            f"{paged} paged (reserve {reserve:.2f} GB for {max_tokens} tokens)"
        )
    return tok, te


def prompt_ids(tok, text: str, max_length=MAX_SEQUENCE_LENGTH):
    """The pipeline's tokenisation of one prompt (chat template, no padding)."""
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return tok(formatted, return_tensors="pt", truncation=True, max_length=max_length)


@torch.no_grad()
def encode_prompt(tok, te, text: str, device="cuda", max_length=MAX_SEQUENCE_LENGTH):
    """One prompt → ``last_hidden_state`` ``(S, 4096)`` bf16 at its true length.
    Exactly ``MOSSImagePipeline.encode_prompt``: chat template with
    ``add_generation_prompt=True, enable_thinking=True``, one prompt per forward
    (no padding — the GDN layers misbehave on it)."""
    inputs = prompt_ids(tok, text, max_length).to(device)
    return te(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]


def encode_prompts(
    model_dir,
    texts,
    device="cuda",
    max_length=MAX_SEQUENCE_LENGTH,
    log=print,
    precision="bf16",
):
    """Load the TE, encode each distinct text once, free it → ``{text: (S,4096) cpu}``."""
    texts = list(dict.fromkeys(texts))
    tok = load_tokenizer(model_dir)
    longest = max(prompt_ids(tok, t, max_length).input_ids.shape[1] for t in texts)
    tok, te = load_text_encoder(
        model_dir, device, longest, log=log, tok=tok, precision=precision
    )
    out = {}
    for text in texts:
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
    # dynamic=True otherwise lifts float attributes (the LoRA's scale ·
    # multiplier) into 0-dim CPU float64 inputs, and Inductor emits a C++ CPU
    # kernel for them — which needs cl.exe on Windows (InvalidCxxCompiler).
    torch._dynamo.config.specialize_float = True
    for block in blocks:
        block.forward = torch.compile(block.forward, dynamic=True)


def load_vae(model_dir, dtype=torch.float32, device=None):
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(
        variant_dir(model_dir, "base") / "vae", torch_dtype=dtype
    ).eval()
    return vae.to(device) if device is not None else vae
