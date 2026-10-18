"""Qwen3.5-9B with its vision tower: the MOSS TE call with an image in the user turn.

The pipeline's encode (chat template, add_generation_prompt, enable_thinking,
last_hidden_state, one prompt per forward, no padding) with an image item ahead of
the text, preprocessed with Qwen3.5-9B's own processor settings (patch 16, merge 2,
mean/std 0.5; MOSS ships no processor config). A text-only call keeps the
pipeline's string content, byte-identical to its prompts.
"""

import torch

from library.moss.loader import _offload_by_layer, _te_reserve_gb
from library.moss.requests import variant_dir


def processor(tok, max_pixels):
    from transformers import (
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    ip = Qwen2VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        size={"shortest_edge": 65536, "longest_edge": max_pixels},
    )
    return Qwen3VLProcessor(
        image_processor=ip,
        tokenizer=tok,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=tok.chat_template,
    )


def load_te(tok, device, model_dir=""):
    """``load_text_encoder`` with the vision tower on the card too (placed before the
    layer budget is taken, so the paging plan accounts for it)."""
    from transformers import AutoModel

    te = AutoModel.from_pretrained(
        variant_dir(model_dir, "base") / "text_encoder", dtype=torch.bfloat16
    ).eval()
    te.visual.to(device)
    paged = _offload_by_layer(te, device, _te_reserve_gb(4096))
    print(f"[te] vision resident, {paged} decoder layers paged", flush=True)
    return te


@torch.no_grad()
def encode(tok, te, image, text, max_pixels, device, drop_rows=False):
    """``(last_hidden_state (S, 4096), image token count)``; ``image=None`` is the
    pipeline's text-only encode. ``drop_rows`` removes the image-token rows."""
    content = text
    if image is not None:
        content = [{"type": "image"}] + (
            [{"type": "text", "text": text}] if text else []
        )
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if image is None:
        inputs = tok(formatted, return_tensors="pt").to(device)
        return te(**inputs, use_cache=False, return_dict=True).last_hidden_state[0], 0
    inputs = processor(tok, max_pixels)(
        text=[formatted], images=[image], return_tensors="pt"
    ).to(device)
    hidden = te(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]
    is_image = inputs["mm_token_type_ids"][0] == 1
    if drop_rows:
        hidden = hidden[~is_image]
    return hidden, int(is_image.sum())
