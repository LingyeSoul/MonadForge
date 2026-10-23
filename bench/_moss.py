"""What the MOSS-Image probes share: the prompt set and the held-out cached captions."""

import glob
import random

import torch
from safetensors import safe_open

PROMPTS = {
    "anime_sunset": "An anime illustration of a girl with long silver hair and red eyes, wearing a black "
    "school uniform, sitting on a classroom windowsill at sunset, cherry blossom petals drifting "
    "outside the window, detailed cel shading, soft rim light.",
    "danbooru_tags": "1girl, solo, akita neru, yellow eyes, blonde hair, side ponytail, gray shirt, "
    "holding phone, looking at viewer, upper body, simple background, white background",
    "zh_fridge": "A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky "
    'note on the refrigerator door is handwritten in Chinese:\n"记得买牛奶和鸡蛋！\n牛奶两盒，鸡蛋一打。\n顺便带一袋面包。"',
    "ja_manga": "A black-and-white manga panel of a boy with messy hair shouting toward the reader, speed "
    'lines in the background, with a large speech bubble that reads "待ってくれ！" in Japanese.',
    "ko_neon": "A cozy café storefront on a rainy night, with a pink neon sign above the door that reads "
    '"오늘도 수고했어" in Korean, reflections on the wet street.',
}


def sample_cached(per_dir: int, seed: int):
    """``[(text, ref_embeds, source)]`` from TE caches whose stored caption hashes
    to its ``caption_sha256``."""
    from library.moss.scan import caption_sha

    rng = random.Random(seed)
    out = []
    for d in sorted(glob.glob("post_image_dataset/moss/*/")):
        files = sorted(glob.glob(f"{d}*.moss_te.safetensors"))
        rng.shuffle(files)
        got = 0
        for f in files:
            with safe_open(f, "pt") as s:
                meta = s.metadata() or {}
                if meta.get("max_text_tokens") != "4096":
                    continue
                cap = meta.get("caption")
                if cap is None or caption_sha(cap) != meta.get("caption_sha256"):
                    continue
                out.append((cap, s.get_tensor("embeds").to(torch.bfloat16), f))
            got += 1
            if got >= per_dir:
                break
    return out
