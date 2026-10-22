#!/usr/bin/env python3
"""Build the calibration + evaluation manifest for the TE's calibrated fp8 rounding
(``te_gptq.py``): MOSS TE inputs of every kind the pipeline feeds it.

v1 (cached captions, the empty prompt, 12 colorize pages) cut held-out caption
error 54 % but natural-language error only 29 %, so v2 draws nine kinds of input,
each balanced by TE tokens rather than entries:

  tags           Anima tag bags (revised captions with no clauses, re-composed from
                 their flat bag)
  position       revised captions with trailing position clauses, no text clause
  ocr            revised captions with text clauses (``Japanese text reads as …``;
                 the export stage's ``--combine_ocr`` already wrote them in place)
  layout         layout-JSON captions of the ``post_image_dataset/moss/*_layout`` caches
  natural        English natural-language prompts (handwritten: the data has none)
  cjk            Japanese / Korean / Chinese, per language: the cjk_distill corpus
                 (translated tag bags, CJK names in English bags, artist commentary,
                 quoted sign text, tag definitions), and handwritten prompts on a
                 quarter of the language's budget
  page_colorize  B&W page + OCR caption, ``moss_colorize_3k_ocr`` training pairs,
                 half manga
  page_only      a page with no text: Manga109-s pages with lettering
                 (``ANIMA_MANGA109S_ROOT``) and the colorize cache's manga pages
  image_caption  a colour illustration + its revised caption

The calib / eval split is per source image (per work for manga, per book for
Manga109-s), a seeded hash shared by every category, so no image is on both sides
in any form. Kept out of both: the TE-bench eval captions
(``bench/_moss.py`` ``sample_cached(4, 42)``, by source image), its PROMPTS, the empty prompt, and the colorize holdout (its works, for manga).

``tokens`` is the TE input length, chat template included; an image entry's is
counted by the processor at ``width × height`` (``cache.target_size`` of the
source at 1024, the size the page is fitted to), checked against the colorize
cache's own ``n_text``. CPU only: the tokenizer and the processor, no weights.

    .venv/bin/python project/moss_lora/src/te_calib.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from anime_tools.captions.position_clauses import compose_caption, parse_caption
from PIL import Image
from safetensors import safe_open

from library.env import anima_home, resolve_under_home
from library.moss import cache as moss_cache
from library.moss.loader import load_tokenizer
from library.moss.scan import TE_SUFFIX, caption_sha
from vision_te import te_inputs

OUT = "post_image_dataset/moss/_te_calib_v2/manifest.json"
MOSS = "post_image_dataset/moss"
RESIZED = "post_image_dataset/resized"
COLORIZE = "post_image_dataset/moss/moss_colorize_3k_ocr"
CJK = "post_image_dataset/cjk_distill"
CATEGORIES = (
    "tags",
    "position",
    "ocr",
    "layout",
    "natural",
    "cjk",
    "page_colorize",
    "page_only",
    "image_caption",
)
IMAGE_CATEGORIES = {"page_colorize", "page_only", "image_caption"}
# (file, register) per language; tag_alt registers are deliberate noise, left out.
CJK_REGISTERS = {
    "ja": (
        ("pairs.jsonl", "tags"),
        ("pairs.jsonl", "names"),
        ("pairs.jsonl", "commentary"),
        ("pairs.jsonl", "quote_preserved"),
    ),
    "ko": (
        ("pairs_ko.jsonl", "tags_ko"),
        ("pairs_ko.jsonl", "names_ko"),
        ("pairs_desc_ko.jsonl", "desc_ko"),
    ),
    "zh": (
        ("pairs_zh.jsonl", "tags_zh"),
        ("pairs_zh.jsonl", "tags_zh_hant"),
        ("pairs_zh.jsonl", "names_zh"),
    ),
}
M109_MIN_TEXT = 6  # lettering boxes on a Manga109-s page

NATURAL = (
    "A watercolor illustration of a fox spirit resting under a red torii gate in light rain.",
    "Two schoolgirls sharing an umbrella at a bus stop on a summer evening, pastel colors.",
    "Chibi knight with an oversized helmet holding a wooden sword, white background.",
    "A lone mecha kneeling in a flooded city at dawn, ivy growing over its armor, volumetric "
    "light falling through broken skyscrapers, muted teal and orange palette.",
    "Portrait of an elderly witch with braided white hair and a crow on her shoulder, thick "
    "oil-paint texture, dark green background.",
    "A cozy attic bedroom at night: a girl in oversized pajamas reads a book by lantern light, "
    "a cat asleep on the blanket, rain running down the skylight.",
    "Pixel art of a tiny shrine on a floating island, 16-bit style, limited palette.",
    "A 1990s anime screenshot: a detective in a long coat lights a cigarette in a rainy alley, "
    "film grain, faded VHS colors.",
    "Full-body character sheet of a desert nomad girl, front and back views, layered scarves, "
    "bronze jewelry and sandals, flat colors on a white background.",
    "Close-up of hands holding a cup of hot cocoa, steam rising, knitted sweater sleeves.",
    "An idol singer on stage as confetti falls, pink and blue spotlights, a crowd of "
    "silhouettes waving glow sticks, dynamic low angle.",
    "A quiet mountain train station in winter with nobody around, snow piling on the platform "
    "bench, one lit vending machine, blue hour.",
    "Ukiyo-e inspired illustration of a samurai cat fighting a giant koi in crashing waves.",
    "A boy and his robot companion fishing off a pier, long shadows, gentle slice-of-life mood.",
    "Three adventurers around a campfire in a forest. A dwarf stirs a pot of stew, an elf tunes "
    "a lute and a young human girl studies a worn map. Fireflies drift between the trees, the "
    "night is deep blue and the firelight paints their faces orange.",
    "Simple pencil sketch of a girl stretching after waking up, unfinished lines.",
    "A vampire noblewoman in a red velvet gown on a balcony under a full moon, gothic "
    "architecture, bats in the sky, dramatic contrast.",
    "Food illustration of a bowl of ramen with chashu, a soft-boiled egg, nori and scallions, "
    "glossy broth, top-down view.",
    "An android girl with exposed circuitry on one cheek studies her reflection in a shop "
    "window.",
    "Magical girl transformation: ribbons of light spiral around her, sparkles everywhere, "
    "star-shaped hair ornaments, high energy.",
    "A middle-aged office worker asleep on a late-night train, loosened tie, city lights "
    "streaking past the window.",
    "Fantasy map of an archipelago with a hand-drawn compass rose and sea serpents in the "
    "margins, parchment texture.",
    "A tall woman with short black hair and a scar across her nose, wearing a military "
    "greatcoat, stands in a snowstorm holding a rifle. Stern expression, cinematic composition.",
    "Kids splashing in a summer river, sunflowers on the bank, bright saturated colors.",
    "Monochrome ink drawing of a dragon coiled around a pagoda.",
    "A shy librarian peeking from behind a tall bookshelf, round glasses, cardigan, dust motes "
    "in the afternoon light.",
    "Poster for a fictional rock band: a girl with a bass guitar mid-jump, bold title text "
    'that reads "STATIC BLOOM", halftone dots, two-color print.',
    "An underwater palace where mermaids swim among coral pillars, shafts of light from above, "
    "schools of silver fish.",
    "Inside a convenience store on a rainy afternoon, a girl in a school uniform chooses an "
    "onigiri, fluorescent lighting, reflections on the floor.",
    "Battle scene: a swordsman parries a giant's axe, sparks fly, motion blur, low camera "
    "angle, debris in the air.",
    "A cute slime monster wearing a tiny wizard hat.",
    "Two old friends on a park bench feeding pigeons in autumn, falling maple leaves, "
    "nostalgic soft focus.",
    "Isometric illustration of a small bakery with an open storefront, bread on wooden "
    "shelves, a baker waving, morning light.",
    "Horror illustration of a long hallway in an abandoned school; a figure with long hair "
    "stands at the far end under a flickering light, desaturated green tint.",
    "A girl in a yukata holding a goldfish bag at a summer festival, paper lanterns overhead, "
    "fireworks reflected in her eyes.",
    "Space opera: a captain on the bridge of a starship, a nebula filling the viewscreen, "
    "crew members at glowing consoles, wide shot.",
    "Sticker-style shiba inu eating dango, thick white outline.",
    "A giant forest spirit with moss-covered antlers lets a small girl hold its finger, misty "
    "woods, painterly background.",
    "Four-panel comic: a girl tries to bake a cake, the oven starts smoking, she panics, and "
    "her roommate laughs at the burnt cake. Expressive faces, clean lines.",
    "Fashion illustration of a streetwear outfit, cropped bomber jacket, cargo pants and "
    "chunky sneakers, full body, gray background, marker rendering.",
    "A knight kneels before a queen in a throne room while stained glass windows cast colored "
    "light across the marble floor. Solemn atmosphere, highly detailed armor.",
    "A boy with headphones sitting on the edge of a rooftop at night, city skyline, a "
    "shooting star.",
    "A sprawling market street in a fantasy port town at midday. Merchants sell spices, "
    "fabrics and glowing crystals under striped awnings; a cat-eared courier weaves through "
    "the crowd with a stack of parcels; sailors unload crates from a ship docked at the end "
    "of the street while gulls circle overhead. Warm sunlight, crisp shadows, a lively "
    "crowded composition with a detailed background, anime style.",
    "Character portrait of a young alchemist at a cluttered workbench covered in glass "
    "flasks, brass instruments and open notebooks. She has messy auburn hair held up with a "
    "pencil, soot on one cheek and a satisfied grin, and holds up a vial of glowing blue "
    "liquid. Late-night lamp light, warm and cool contrast, painterly anime rendering.",
    "Sumi-e landscape of misty mountains and a small boat on a lake, lots of negative space.",
    "A wide establishing shot of a floating academy city above the clouds at sunrise. Spiral "
    "towers of white stone are linked by rope bridges and glass walkways, airships with "
    "patched canvas balloons drift between them, and students in navy capes ride winged "
    "bicycles along a tram line that loops around the central clock tower. Far below, a "
    "patchwork of farmland shows through gaps in the cloud sea. The light is soft gold and "
    "pink, the air hazy with distance, and the whole scene is drawn in a clean anime "
    "background style with crisp linework and gentle gradients.",
    "An anime illustration of a tired nurse taking a break on the hospital rooftop at dawn. "
    "She sits on an upturned crate with her shoes kicked off, a paper cup of coffee in both "
    "hands and her hair coming loose from its clip. Laundry flaps on a line behind her, and "
    "the city is still blue and quiet below, with only a few windows lit. Her expression is "
    "exhausted but peaceful. Muted colors, soft morning light, realistic proportions, a "
    "slice-of-life mood.",
    "Dynamic action illustration of a girl on a motorcycle racing along a coastal highway at "
    "night, pursued by three drones with red searchlights. Her jacket and long scarf whip "
    "behind her, one hand on the throttle and the other aiming a flare pistol backwards. The "
    "guardrail streaks past in motion blur, sparks scatter from the asphalt where a shot "
    "just missed, and the ocean on the right reflects a crescent moon. Strong perspective, "
    "high contrast, saturated reds against deep blue.",
    "A detailed interior of a witch's apothecary shop: shelves crowded with jars of dried "
    "herbs, glowing mushrooms and labelled potions, bundles of lavender hanging from the "
    "beams, a black cat asleep on the counter beside an old brass scale. The shopkeeper, a "
    "young woman with freckles, a pointed hat and rolled-up sleeves, grinds something in a "
    "mortar while a customer, a small goblin in a raincoat, waits on tiptoe. Warm candle "
    "light, cozy cluttered composition, storybook illustration style.",
    "Character design sheet for an original fantasy heroine. She is a lancer from a northern "
    "mountain clan, about twenty, with ash-blond hair in a long braid, pale grey eyes and a "
    "thin scar through her left eyebrow. She wears layered fur and leather armor with "
    "engraved silver plates on the shoulders, a heavy wool cloak fastened with a wolf-head "
    "clasp, and carries a long spear with a crescent blade. Show a full-body front view, a "
    "back view, three facial expressions (calm, angry, laughing) and close-ups of the clasp "
    "and the spearhead, on a plain light grey background.",
    "A melancholic scene in an empty amusement park after closing time. A carousel is still "
    "lit, turning slowly with no riders, and a girl in a yellow raincoat stands alone in "
    "front of it holding a deflated balloon. Puddles reflect the colored bulbs, the Ferris "
    "wheel behind is dark against a cloudy purple sky, and scattered popcorn lies on the wet "
    "pavement. Painterly anime style, cinematic framing, quiet and lonely atmosphere.",
    "Illustration for a light novel cover: a boy in a worn school blazer and a demon girl "
    "with curled horns and a tail stand back to back in the middle of a ruined library, both "
    "ready to fight. Torn pages swirl around them in a magical wind, a summoning circle glows "
    "violet on the floor, and shadowy hands reach out of the bookshelves. The girl smirks "
    "while the boy looks nervous. Vivid colors, dramatic lighting from below, polished "
    "commercial anime rendering, space at the top for a title.",
    "A peaceful countryside morning in early summer. A narrow path runs between rice paddies "
    "that mirror the sky, a small wooden bus shelter with a faded tin sign stands at the "
    "crossroads, and a girl in a white sundress and straw hat waits there with a suitcase. "
    "Dragonflies hover over the water, cumulus clouds pile up over green hills, and power "
    "lines stretch toward the horizon. Bright, airy colors, detailed background art in the "
    "style of a nostalgic anime film.",
    "Close-up portrait of a sniper lying prone on a snowy ridge, her face half hidden by a "
    "white balaclava and the scope of her rifle. Frost clings to her eyelashes, her breath "
    "makes a faint cloud, and the scope's lens reflects a distant mountain fortress. Cold "
    "desaturated palette with a single warm accent from the sunrise on the horizon, sharp "
    "focus on the eye, shallow depth of field, semi-realistic anime style.",
    "A whimsical scene of tiny people living inside a teapot house on a kitchen windowsill. "
    "A spiral staircase winds up the spout, laundry hangs between two sugar cubes, a little "
    "girl waters thimble-sized flower pots, and her grandfather reads a newspaper made from a "
    "postage stamp. Outside the window, a giant sparrow peers in curiously. Soft watercolor "
    "textures, pastel colors, picture-book charm.",
    "An intense chess match in a smoky 1920s café. Two players sit across a small marble "
    "table: a sharp-featured young woman in a tailored suit with her chin on her hand, and an "
    "older gentleman with a monocle frozen in mid-move. Onlookers crowd behind them in hats "
    "and coats, a waiter balances a tray of espresso cups, and a gramophone sits on the bar. "
    "Sepia and amber tones, art deco details, light from a hanging lamp, anime illustration "
    "with a vintage feel.",
    "A group-photo illustration of a high school archery club after the summer tournament: "
    "twelve students in white and black kyudo uniforms gathered on the steps of the dojo, "
    "some holding bows, one holding a small trophy, the captain in the center giving a peace "
    "sign, a first-year crying with joy, and the advisor standing slightly apart with crossed "
    "arms and a hidden smile. Late afternoon light, clean character designs, cheerful mood.",
    "Dark fantasy illustration of a knight in blackened plate armor walking through a field "
    "of withered sunflowers toward a cathedral that has collapsed into a sinkhole. Crows "
    "circle overhead, the sky is the color of ash, and a faint red glow seeps from the pit. "
    "The knight drags a broken greatsword behind him, leaving a furrow in the dry earth. "
    "Heavy texture, grim desaturated palette, painterly brushwork, epic scale.",
    "A cheerful cooking scene: a chubby bear chef and a small rabbit apprentice make "
    "dumplings together in a steamy kitchen. Flour dusts the table and the rabbit's nose, a "
    "bamboo steamer stack puffs on the stove, and finished dumplings are lined up neatly on a "
    "tray. Round shapes, warm colors, thick soft outlines, children's anime style.",
    "An astronaut girl sits on the edge of a lunar crater with her visor up and a pair of "
    "earbuds in, looking at a huge blue Earth that fills half the sky. Her suit is covered in "
    "hand-drawn doodles and patches. The stars are sharp and dense, the ground is pale grey "
    "with long black shadows, and the mood is dreamy and calm. Clean cel shading with a soft "
    "glow on the planet.",
    "A busy newsroom at night before a deadline. Desks overflow with paper, monitors show "
    "headlines, a young reporter with a pencil behind her ear argues on the phone, an editor "
    "slams a printout on a desk, and someone asleep under a jacket uses a keyboard as a "
    "pillow. Through the windows, city lights and a passing train. Cool fluorescent light, "
    "lots of small details, energetic composition.",
)
CJK_HANDWRITTEN = {
    "ko": (
        "벚꽃이 흩날리는 강변 산책로를 걷는 교복 차림의 소녀, 부드러운 파스텔 톤 애니메이션 일러스트",
        "눈 내리는 밤 버스 정류장에서 목도리를 두른 채 기다리는 소년, 따뜻한 가로등 불빛",
        "갑옷을 입은 여기사가 불타는 성을 배경으로 검을 들고 서 있다. 극적인 구도, 디테일한 배경.",
        "귀여운 SD 캐릭터, 큰 리본, 흰 배경, 스티커 스타일",
        "1girl, 금발, 트윈테일, 메이드복, 카메라 응시, 미소, 상반신",
        "해 질 녘 옥상에서 기타를 치는 밴드부 소녀들, 청춘 만화 느낌",
        "한복을 입은 여우 요괴가 달빛 아래 연못가에 앉아 있는 수채화",
        "의수를 단 용병이 폐공장 안에서 총을 점검하고 있다, 차가운 청록색 조명, 사이버펑크",
        "따뜻한 카페 안에서 태블릿으로 그림을 그리는 안경 쓴 여자, 창밖에는 단풍이 물든 거리",
        "판타지 RPG 파티: 마법사, 궁수, 전사가 던전 입구 앞에 모여 작전을 짜고 있다",
        "masterpiece, best quality, 1boy, 검은 머리, 교복, 벚꽃, 바람에 날리는 머리카락",
        "흑백 펜화로 그린 고양이와 할머니의 평범한 하루",
    ),
    "ja": (
        "夕暮れの商店街を自転車で走る女子高生、ノスタルジックな雰囲気のアニメ風イラスト",
        "雪の積もった神社で巫女さんが箒で掃除している、静かな冬の朝",
        "1girl, 黒髪ロング, セーラー服, 窓際, 微笑み, 柔らかい光",
        "巨大なロボットと少年が草原に並んで座っている。入道雲が広がる夏の午後。",
        "ゴシックロリータの服を着た吸血鬼の少女、薔薇と月、ダークファンタジー",
        "猫耳メイドがオムライスにケチャップでハートを描いている、デフォルメ、かわいい",
        "魔法学校の図書館で、宙に浮かぶ本に囲まれて勉強する少女。幻想的な光。",
        "雨のバス停で傘を差し出す少年と、驚いた顔の少女。少女漫画風。",
        "水彩画風の紫陽花とかたつむり、梅雨の庭",
        "宇宙船のブリッジで指揮を執る女性艦長、SF、ワイドショット",
        "温泉宿の縁側で浴衣姿の二人がスイカを食べている夏の夕方",
        "best quality, 2girls, 手を繋ぐ, 花火大会, 浴衣, 夜空",
    ),
    "zh": (
        "一个穿着汉服的少女站在开满桃花的山间小路上，水彩风格的动漫插画",
        "下雪的古镇夜晚，红灯笼照亮石板路，一只白猫坐在屋檐上",
        "1girl, 银色短发, 机械义肢, 废墟, 黄昏, 风",
        "可爱的Q版熊猫在竹林里吃包子，贴纸风格，白色背景",
        "仙侠风格：白衣剑客御剑飞行于云海之上，远处是金色的宫殿。",
        "放学后的社团活动室里，三个女生围着桌子玩桌游，热闹的氛围",
        "一位老爷爷在茶馆里下象棋，周围的人在围观，温暖的午后光线",
        "魔法少女变身的瞬间，彩色丝带环绕全身，闪闪发光",
        "水墨画风格的山水，一叶扁舟，大量留白",
        "穿着旗袍的女子撑着油纸伞走在江南的雨巷中",
        "masterpiece, 1boy, 黑色长发, 古风, 持剑, 竹林, 月光",
        "星空下露营，女孩和她的狗依偎在帐篷前，篝火噼啪作响",
    ),
}


def is_eval(key: str, seed: int, frac: float) -> bool:
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < frac


def interleave(groups: dict, rng: random.Random, shuffle_items: bool = True) -> list:
    """Round-robin over ``groups`` in a shuffled order, so one artist / work / register
    never fills a category before the others get a turn."""
    names = sorted(groups)
    rng.shuffle(names)
    lists = []
    for n in names:
        items = list(groups[n])
        if shuffle_items:
            rng.shuffle(items)
        lists.append(items)
    out = []
    for i in range(max((len(x) for x in lists), default=0)):
        out += [x[i] for x in lists if i < len(x)]
    return out


def colorize_key(stem: str) -> str:
    head, _, rest = stem.partition("__")
    return f"manga:{rest.split('_')[0]}" if head == "manga" else rest


def excluded() -> tuple[set[str], set[str]]:
    """``(keys, texts)`` no entry may use: the TE-bench eval captions (by source
    image), PROMPTS, the empty prompt, the colorize holdout."""
    spec = importlib.util.spec_from_file_location(
        "bench_moss", resolve_under_home("bench/_moss.py")
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    keys, texts = set(), {""} | set(bench.PROMPTS.values())
    for cap, _, f in bench.sample_cached(4, 42):
        with safe_open(f, "pt") as s:
            keys.add(Path((s.metadata() or {}).get("source", f)).stem)
        texts.add(cap)
    holdout = json.loads((resolve_under_home(COLORIZE) / "holdout.json").read_text())
    keys |= {colorize_key(r["stem"]) for r in holdout}
    return keys, texts


def resized_captions() -> dict[str, list]:
    """Revised captions by kind: ``{kind: [(artist, txt, caption, parsed)]}``."""
    out = {"plain": [], "position": [], "text": []}
    for txt in sorted(resolve_under_home(RESIZED).rglob("*.txt")):
        if txt.name.endswith(".variants.txt"):
            continue
        cap = txt.read_text(encoding="utf-8").strip()
        parsed = parse_caption(cap)
        kind = (
            "text"
            if parsed.text_clauses
            else "position"
            if parsed.position_clauses
            else "plain"
        )
        out[kind].append((txt.parent.name, txt, cap, parsed))
    return out


def layout_captions() -> dict[str, list]:
    out = {}
    for d in sorted(resolve_under_home(MOSS).glob("*_layout")):
        for f in sorted(d.glob(f"*{TE_SUFFIX}")):
            with safe_open(str(f), "pt") as s:
                meta = s.metadata() or {}
            cap = meta.get("caption")
            if (
                meta.get("caption_format") != "layout_json"
                or meta.get("max_text_tokens") != "4096"
                or cap is None
                or caption_sha(cap) != meta.get("caption_sha256")
            ):
                continue
            out.setdefault(d.name, []).append((f, cap, Path(meta["source"]).stem))
    return out


def cjk_rows(lang: str) -> dict[str, list]:
    """``{register: [(id, key, text, file)]}``; the key is the source image id where
    the row has one (D1 / D2), else the row id without its register."""
    out = {}
    root = resolve_under_home(CJK)
    for fn, reg in CJK_REGISTERS[lang]:
        tag = f'"register": "{reg}"'
        with open(root / fn, encoding="utf-8") as f:
            for line in f:
                if tag not in line:
                    continue
                r = json.loads(line)
                if r["register"] != reg or not r.get("ja"):
                    continue
                parts = r["id"].split("/")
                key = parts[-2] if parts[0] in ("D1", "D2") else "/".join(parts[:-1])
                out.setdefault(reg, []).append(
                    (r["id"], key, r["ja"].strip(), root / fn)
                )
    return out


def colorize_pairs() -> list[dict]:
    """``moss_colorize_3k_ocr`` training TE files (``holdout/`` is a subdirectory)."""
    out = []
    for f in sorted(resolve_under_home(COLORIZE).glob(f"*{TE_SUFFIX}")):
        with safe_open(str(f), "pt") as s:
            meta = s.metadata()
        stem = f.name[: -len(TE_SUFFIX)]
        out.append({"stem": stem, "te": f, **meta})
    return out


def manga109_pages(root: Path) -> dict[str, list[Path]]:
    """Per book, the pages with at least ``M109_MIN_TEXT`` lettering boxes (covers
    and the first two pages left out)."""
    out = {}
    for book in sorted((root / "images").iterdir()):
        tree = ET.parse(root / "annotations" / f"{book.name}.xml")
        for page in tree.iter("page"):
            i = int(page.get("index"))
            if i >= 3 and len(page.findall("text")) >= M109_MIN_TEXT:
                out.setdefault(book.name, []).append(book / f"{i:03d}.jpg")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--text_tokens", type=int, default=4000, help="calib, per category")
    ap.add_argument("--cjk_tokens", type=int, default=1500, help="calib, per language")
    ap.add_argument(
        "--page_tokens", type=int, default=12000, help="calib, per category"
    )
    ap.add_argument("--eval_ratio", type=float, default=0.5, help="eval / calib tokens")
    ap.add_argument("--eval_frac", type=float, default=1 / 3, help="keys on eval side")
    ap.add_argument("--max_entry", type=int, default=1000, help="text entry token cap")
    ap.add_argument(
        "--manga109_root", default=os.environ.get("ANIMA_MANGA109S_ROOT", "")
    )
    args = ap.parse_args()
    torch.set_num_threads(2)
    os.chdir(anima_home())  # sample_cached globs relative paths

    tok = load_tokenizer("")
    bad_keys, bad_texts = excluded()
    entries, used = [], set()
    mismatches = []

    def n_tokens(text, size=None):
        if size is None:
            return int(te_inputs(tok, None, text, 0, "cpu")["input_ids"].shape[1])
        # The count depends on the size alone; the consumer fits the real page to it.
        page = Image.new("RGB", size)
        return int(
            te_inputs(tok, page, text, size[0] * size[1], "cpu")["input_ids"].shape[1]
        )

    def fill(category, cands, budget, label=""):
        """Take ``cands`` in order into whichever split their key falls on until each
        split's token budget is met. ``cands``: dicts with id / key / text / image /
        size_from / source (+ ``check``: the expected token count)."""
        want = {"calib": budget, "eval": int(budget * args.eval_ratio)}
        got = {"calib": 0, "eval": 0}
        for c in cands:
            split = "eval" if is_eval(c["key"], args.seed, args.eval_frac) else "calib"
            if got[split] >= want[split]:
                if all(got[s] >= want[s] for s in got):
                    break
                continue
            if c["key"] in bad_keys or (
                c["text"].strip() in bad_texts and not c.get("image")
            ):
                continue
            ident = (c.get("image"), c["text"])
            if ident in used:
                continue
            size = None
            if c.get("image"):
                with Image.open(c["size_from"]) as im:
                    size = moss_cache.target_size(*im.size, 1024)
            tokens = n_tokens(c["text"], size)
            if size is None and tokens > args.max_entry:
                continue
            if c.get("check") is not None and tokens != c["check"]:
                mismatches.append((c["id"], tokens, c["check"]))
            used.add(ident)
            got[split] += tokens
            entries.append(
                {
                    "id": f"{category}:{c['id']}",
                    "category": category,
                    "split": split,
                    "text": c["text"],
                    "image": str(c["image"]) if c.get("image") else None,
                    "width": size[0] if size else None,
                    "height": size[1] if size else None,
                    "source": str(c["source"]),
                    "tokens": tokens,
                }
            )
        short = [s for s in got if got[s] < want[s]]
        if short:
            print(
                f"[{category}{label}] short of budget: "
                + ", ".join(f"{s} {got[s]}/{want[s]}" for s in short),
                flush=True,
            )

    def rng(name):
        return random.Random(f"{args.seed}:{name}")

    caps = resized_captions()

    def by_artist(rows, text_of, image=False):
        groups = {}
        for artist, txt, cap, parsed in rows:
            png = txt.with_suffix(".png")
            if image and not png.exists():
                continue
            groups.setdefault(artist, []).append(
                {
                    "id": f"{artist}/{txt.stem}",
                    "key": txt.stem,
                    "text": text_of(cap, parsed),
                    "image": png if image else None,
                    "size_from": png,
                    "source": txt,
                }
            )
        return groups

    fill(
        "tags",
        interleave(
            by_artist(caps["plain"], lambda c, p: compose_caption(p.flat_tags)),
            rng("tags"),
        ),
        args.text_tokens,
    )
    fill(
        "position",
        interleave(by_artist(caps["position"], lambda c, p: c), rng("position")),
        args.text_tokens,
    )
    fill(
        "ocr",
        interleave(by_artist(caps["text"], lambda c, p: c), rng("ocr")),
        args.text_tokens,
    )

    layout = {
        d: [
            {"id": f"{d}/{f.name}", "key": key, "text": cap, "source": f}
            for f, cap, key in rows
        ]
        for d, rows in layout_captions().items()
    }
    fill("layout", interleave(layout, rng("layout")), args.text_tokens)

    natural = [
        {"id": f"natural/{i:02d}", "key": f"natural/{i:02d}", "text": t}
        for i, t in enumerate(NATURAL)
    ]
    rng("natural").shuffle(natural)
    fill(
        "natural",
        [{**c, "source": "handwritten"} for c in natural],
        args.text_tokens,
    )

    for lang in ("ja", "ko", "zh"):
        groups = {
            reg: [
                {
                    "id": f"cjk/{lang}/{rid}",
                    "key": key,
                    "text": text,
                    "source": f"{fn}#{rid}",
                }
                for rid, key, text, fn in rows
            ]
            for reg, rows in cjk_rows(lang).items()
        }
        hand = [
            {
                "id": f"cjk/{lang}/handwritten/{i:02d}",
                "key": f"cjk/{lang}/handwritten/{i:02d}",
                "text": t,
                "source": "handwritten",
            }
            for i, t in enumerate(CJK_HANDWRITTEN[lang])
        ]
        rng(f"cjk/{lang}/handwritten").shuffle(hand)
        # Real rows are long tag bags; the short prompts get their own quarter.
        real = args.cjk_tokens * 3 // 4
        fill("cjk", interleave(groups, rng(f"cjk/{lang}")), real, f"/{lang}")
        fill("cjk", hand, args.cjk_tokens - real, f"/{lang}/handwritten")

    pairs = colorize_pairs()

    def colorize_cands(rows, with_caption):
        groups = {}
        for p in rows:
            key = colorize_key(p["stem"])
            group = key if key.startswith("manga:") else p["stem"].split("__")[0]
            groups.setdefault(group, []).append(
                {
                    "id": f"colorize/{p['stem']}",
                    "key": key,
                    "text": p["caption"] if with_caption else "",
                    "image": p["condition"],
                    "size_from": p["source"],
                    "source": p["te"],
                    "check": int(p["n_text" if with_caption else "n_text_plain"]),
                }
            )
        return groups

    manga = [p for p in pairs if p["stem"].startswith("manga__")]
    illus = [p for p in pairs if not p["stem"].startswith("manga__")]
    r = rng("page_colorize")
    fill(
        "page_colorize",
        interleave(
            {
                "manga": interleave(colorize_cands(manga, True), r),
                "illus": interleave(colorize_cands(illus, True), r),
            },
            r,
            shuffle_items=False,
        ),
        args.page_tokens,
    )

    r = rng("page_only")
    streams = {"colorize_manga": interleave(colorize_cands(manga, False), r)}
    if args.manga109_root:
        m109 = {
            book: [
                {
                    "id": f"manga109/{book}/{p.stem}",
                    "key": f"m109:{book}",
                    "text": "",
                    "image": p,
                    "size_from": p,
                    "source": p,
                }
                for p in pages
            ]
            for book, pages in manga109_pages(Path(args.manga109_root)).items()
        }
        streams["manga109"] = interleave(m109, r)
    else:
        print("[page_only] ANIMA_MANGA109S_ROOT unset: colorize manga only", flush=True)
    fill("page_only", interleave(streams, r, shuffle_items=False), args.page_tokens)

    illus_caps = caps["plain"] + caps["position"] + caps["text"]
    fill(
        "image_caption",
        interleave(
            by_artist(illus_caps, lambda c, p: c, image=True), rng("image_caption")
        ),
        args.page_tokens,
    )

    if mismatches:
        print(f"[tokens] {len(mismatches)} differ from the cache's n_text:", flush=True)
        for m in mismatches[:10]:
            print(f"  {m[0]}: counted {m[1]}, cached {m[2]}", flush=True)
    out = resolve_under_home(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"seed": args.seed, "args": vars(args), "entries": entries},
            ensure_ascii=False,
            indent=1,
        )
    )

    print(f"{'category':<14} {'calib n':>8} {'tokens':>7} {'eval n':>7} {'tokens':>7}")
    tot = {"calib": [0, 0], "eval": [0, 0]}
    for cat in CATEGORIES:
        row = {"calib": [0, 0], "eval": [0, 0]}
        for e in entries:
            if e["category"] == cat:
                row[e["split"]][0] += 1
                row[e["split"]][1] += e["tokens"]
        for s in tot:
            tot[s] = [a + b for a, b in zip(tot[s], row[s])]
        print(
            f"{cat:<14} {row['calib'][0]:>8} {row['calib'][1]:>7} "
            f"{row['eval'][0]:>7} {row['eval'][1]:>7}"
        )
    print(
        f"{'total':<14} {tot['calib'][0]:>8} {tot['calib'][1]:>7} "
        f"{tot['eval'][0]:>7} {tot['eval'][1]:>7}"
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
