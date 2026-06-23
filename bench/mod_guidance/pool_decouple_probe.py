"""Offline geometry probe for mod-guidance: does decoupling the mod-neg from the
content/CFG negative matter, and does the pooling choice (max vs mean) change the
steering axis?

No image generation. We only compute the mod steering delta

    delta = proj(pool(mod_pos)) - proj(pool(mod_neg))

under different mod_neg choices and different poolings, and measure cosine
similarity between the resulting axes. Cosine ~1 => same direction => the choice
doesn't matter. Low cosine => it changes where you steer.

Usage:
    uv run python -m bench.mod_guidance.pool_decouple_probe
"""

import logging
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import anima_lora
from anima_lora import ROOT
from library.anima import weights as anima_utils
from library.inference.models import load_text_encoder
from library.inference.text import ensure_text_strategies
from library.inference.corrections.mod_guidance import _encode_prompt_for_mod

logging.basicConfig(level=logging.WARNING)
device = torch.device("cuda")
dtype = torch.bfloat16

ck = anima_lora.default_checkpoints()
dit_path = str(ROOT / ck.dit)
te_path = str(ROOT / ck.text_encoder)
head_path = str(ROOT / "output/ckpt/pooled_text_proj-0611.safetensors")

print("loading DiT + head + text encoder ...")
anima = anima_utils.load_anima_model(
    device, dit_path, attn_mode="torch", loading_device=device, dit_weight_dtype=dtype
)
anima.to(device)  # ensure llm_adapter rotary buffers land on GPU
anima.pooled_text_proj.to_empty(device="cpu")
anima.pooled_text_proj.load_state_dict(load_file(head_path))
anima.pooled_text_proj.to(device=device, dtype=torch.float32)

ensure_text_strategies(te_path)
text_encoder = load_text_encoder(text_encoder=te_path, dtype=dtype, device=device)
text_encoder.eval()


def proj(pooled):
    return anima.pooled_text_proj(pooled.to(anima.pooled_text_proj[0].weight.dtype))


def pools(prompt):
    """Return (max_pad, max_real, mean_real) pooled vectors for a prompt."""
    emb = _encode_prompt_for_mod(prompt, anima, text_encoder, device)  # (1, 512, C)
    real = emb.abs().sum(-1) > 0  # (1, 512) non-pad rows (pads were zeroed)
    max_pad = emb.max(dim=1).values  # node-faithful (max over padded zeros)
    masked = emb.masked_fill(~real.unsqueeze(-1), float("-inf"))
    max_real = masked.max(dim=1).values
    mean_real = (emb * real.unsqueeze(-1)).sum(1) / real.sum(1, keepdim=True).clamp(min=1)
    return max_pad, max_real, mean_real


def delta(pos, neg, which):
    pp, nn = pools(pos)[which], pools(neg)[which]
    return (proj(pp) - proj(nn)).float().flatten()


def cos(a, b):
    return F.cosine_similarity(a, b, dim=0).item()


MOD_POS = "masterpiece, score_9"
NEG_CLEAN = "score_1, old"
NEG_CLEAN2 = "worst quality, score_1"
NEG_KITCHEN = (
    "worst quality, score_1, score_2, artist name, lowres, old, simple background, "
    "multiple views, sepia, blurry, bad anatomy, glitch, jpeg artifacts, bad hands, "
    "missing fingers, extra fingers, english text, japanese text, speech bubble, "
    "white border, black border"
)

print("\n" + "=" * 70)
print("Q1: DOES DECOUPLING THE MOD-NEG MATTER?  (max_pad = node-faithful)")
print("=" * 70)
d_clean = delta(MOD_POS, NEG_CLEAN, 0)
d_clean2 = delta(MOD_POS, NEG_CLEAN2, 0)
d_kitchen = delta(MOD_POS, NEG_KITCHEN, 0)
print(f"  cos(clean 'score_1,old'      , kitchen-sink) = {cos(d_clean, d_kitchen):+.3f}")
print(f"  cos(clean2 'worst q,score_1' , kitchen-sink) = {cos(d_clean2, d_kitchen):+.3f}")
print(f"  cos(clean                    , clean2      ) = {cos(d_clean, d_clean2):+.3f}")
print(f"  ||delta clean||={d_clean.norm():.3f}  ||delta kitchen||={d_kitchen.norm():.3f}")
print("  -> cos~1 across mod-neg = decoupling pointless. low cos = it matters.")

print("\n" + "=" * 70)
print("Q2: DOES THE POOLING CHOICE CHANGE THE AXIS?")
print("=" * 70)
for label, neg in [("clean neg", NEG_CLEAN), ("kitchen neg", NEG_KITCHEN)]:
    dmp = delta(MOD_POS, neg, 0)  # max_pad (node)
    dmr = delta(MOD_POS, neg, 1)  # max_real
    dme = delta(MOD_POS, neg, 2)  # mean_real
    print(f"  [{label}]")
    print(f"    cos(max_pad , max_real ) = {cos(dmp, dmr):+.3f}  (pad-zero artifact)")
    print(f"    cos(max_pad , mean_real) = {cos(dmp, dme):+.3f}  (max vs mean axis)")
    print(f"    cos(max_real, mean_real) = {cos(dmr, dme):+.3f}")

print("\n" + "=" * 70)
print("Q3: DOES MAX SATURATE AS YOU ADD TAGS?  (axis drift + norm growth)")
print("=" * 70)
FILLER = [
    "best quality", "highres", "absurdres", "detailed", "intricate details",
    "cinematic lighting", "vibrant colors", "sharp focus", "8k", "ultra detailed",
]
base_pos = "masterpiece, score_9"
d_base = delta(base_pos, NEG_CLEAN, 0)
print(f"  {'#tags':>5}  {'||pool_max||':>12}  {'cos(axis vs 2-tag base)':>24}")
for k in [0, 2, 4, 6, 10]:
    pos = base_pos + ("" if k == 0 else ", " + ", ".join(FILLER[:k]))
    pm = pools(pos)[0]
    dk = delta(pos, NEG_CLEAN, 0)
    print(f"  {2 + k:>5}  {pm.float().norm().item():>12.3f}  {cos(dk, d_base):>+24.3f}")
print("  -> norm climbing + cos falling = max saturates/drifts as tags pile up.")
