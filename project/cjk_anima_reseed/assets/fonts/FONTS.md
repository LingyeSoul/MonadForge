# Manga lettering fonts for the S-line renders

Picked from https://oekaki-zukan.com/articles/516 (user, 2026-09-15) — the
comic-lettering set the base model's training captions were drawn in. Binaries
are gitignored (48 MB); this file and `licenses/` are tracked. `find_fonts()`
(`src/common/render/flat.py`) lists Noto CJK plus every `*.ttf` / `*.otf` here, and
`pick_font()` draws only among fonts whose cmap covers the string.

| file | font | role (per the article) | licence | source |
|---|---|---|---|---|
| `GenEiAntiqueNv6-M.ttf` | 源暎アンチック v6.0a | basic dialogue (antique gothic — kana mincho-ish, kanji gothic) | SIL OFL 1.1 (`licenses/GenEiAntique_OFL.txt`) | https://okoneya.jp/font/download.html |
| `GenJyuuGothic-{Medium,Bold}.ttf` | 源柔ゴシック 2015-06-07 | monologue / narration (rounded gothic) | SIL OFL 1.1 (`licenses/GenJyuuGothic_*`) | http://jikasei.me/font/genjyuu/ (OSDN mirror) |
| `GenShinGothic-{Medium,Bold}.ttf` | 源真ゴシック 2015-06-07 | emphasis (angular gothic) | SIL OFL 1.1 (`licenses/GenShinGothic_*`) | http://jikasei.me/font/genshin/ (OSDN mirror) |
| `Corporate-Logo-Bold-ver3.otf` | コーポレート・ロゴ ver3 Bold | playful / childish dialogue | SIL OFL 1.1 (`licenses/CorporateLogo_OFL.txt`) | https://logotype.jp/corporate-logo-font-dl.html |
| `TanukiMagic.ttf` | たぬき油性マジック 1.22 | comedy emphasis, hand-lettered marker (JIS L2 kanji) | free incl. commercial; no resale of the file (`licenses/TanukiMagic_readme.txt`) | https://tanukifont.com/tanuki-permanent-marker/ |
| ~~`CHI-hasenG.ttf`~~ | 破線G (Chiba Design; a dashed-stroke 源ノ角ゴシック derivative, ≈ 8 200 glyphs incl. JIS L2) | **out** (user, 2026-09-30: renders as a white-fill dashed outline; file parked in `_dl/`, which `find_fonts()` does not glob) | SIL OFL 1.1 (per the BOOTH page; no licence file ships in the zip) | https://booth.pm/en/items/6454019 (BOOTH, pixiv login; user-downloaded 2026-09-15) |
| `DelaGothicOne-Regular.ttf` | Dela Gothic One v1.005 (デラゴシック; 9 030 glyphs, 7 654 kanji, `vert` alternates; no ゕ ゖ) | heavy poster / title gothic — the free stand-in for Fontworks' ラグランパンチ (user, 2026-10-03) | SIL OFL 1.1 (`licenses/DelaGothicOne_OFL.txt`) | https://github.com/google/fonts/tree/main/ofl/delagothicone |
| `koyomiyuru.otf` | こよみゆる v1.000 (iki-font; FontDrawer handwriting, JIS L1 kanji — `pick_font` skips it for L2 strings) | scribbly handwritten dialogue | SIL OFL 1.1 (`licenses/Koyomiyuru_2026_0410_Readme.txt`) | https://booth.pm/en/items/8185365 (BOOTH, pixiv login; user-downloaded 2026-09-15) |
| (system) `NotoSerifCJK-*.ttc` | 源ノ明朝 = Noto Serif CJK | declarations (serif) | SIL OFL 1.1 | installed |
| ~~`NotoSansCJK-*.ttc`~~ | Noto Sans CJK — the pre-2026-09-15 only font | **out** (user, 2026-09-15: reads ambiguous beside the manga faces) | | |

Not fetched: やさしさアンチック (fontna.com page 404 / no link found), 851チカラヨワク
(distribution page empty), しねきゃぷしょん (Vector form download), the horror /
brush / handwriting fonts (off the dialogue-and-SFX distribution we train).

Re-fetch: `curl -L -o x.zip <source>` and unzip the listed face here.

## `kozh/` — Korean / Chinese faces (2026-10-09)

For the reseed line's `lang` runs (`../cjk_anima_reseed/configs/kozh16.toml`): `reseed.pools.lang_fonts` adds
every face here to `find_fonts()`'s list, which never globs `kozh/` — several ZH faces cover kana and would
otherwise join every JA run. A face whose cmap maps a run's row to an empty outline leaves that run's draw
(`TanukiMagic.ttf` maps 你 to one; `render_grid` divides by its zero width). Coverage checked with
fontTools: KS X 1001 = 2 350 Hangul syllables, all = 11 172 (`../cjk_ko/probes/jamo_sets.py` re-checks the
Hangul column, 10-09), GB2312 = 6 763 hanzi. Noto Serif CJK (every weight, in `find_fonts()`) covers
all 11 172 too, so a lang run draws Hangul in it beside the faces here. All SIL OFL 1.1 (`licenses/kozh/`).

| file | face | role | coverage | source |
|---|---|---|---|---|
| `NanumGothic-Regular.ttf` | Nanum Gothic | KO dialogue | all 11 172 | google/fonts `ofl/nanumgothic` |
| `DoHyeon-Regular.ttf` | Do Hyeon | KO emphasis | KS X 1001 (+87) | `ofl/dohyeon` |
| `Jua-Regular.ttf` | Jua | KO rounded / playful | KS X 1001 (+17) | `ofl/jua` |
| `BlackHanSans-Regular.ttf` | Black Han Sans | KO poster | KS X 1001 (+231) | `ofl/blackhansans` |
| `NanumPenScript-Regular.ttf` | Nanum Pen Script | KO marker | all 11 172 | `ofl/nanumpenscript` |
| `NanumMyeongjo-Regular.ttf` | Nanum Myeongjo | KO serif | all 11 172 | `ofl/nanummyeongjo` |
| `NotoSansSC-Medium.otf` | Noto Sans SC Medium | ZH dialogue (+ kana) | GB2312 | notofonts/noto-cjk `Sans/SubsetOTF/SC` |
| `SmileySans-Oblique.ttf` | Smiley Sans 得意黑 v2.0.1 | ZH emphasis (+ kana) | GB2312 | atelier-anchor/smiley-sans (release zip) |
| `ZCOOLKuaiLe-Regular.ttf` | ZCOOL KuaiLe 站酷快乐体 | ZH playful | GB2312 | `ofl/zcoolkuaile` |
| `DouyinSansBold.ttf` | Douyin Sans 抖音美好体 | ZH poster (+ kana) | GB2312 | bytedance/fonts `DouyinSans/` |
| `LXGWMarkerGothic-Regular.ttf` | LXGW Marker Gothic 霞鹜漫黑 v1.003 | ZH marker (+ kana) | GB2312 | lxgw/LxgwMarkerGothic (release zip, `fonts/ttf/`) |
| `LXGWWenKai-Regular.ttf` | LXGW WenKai 霞鹜文楷 v1.522 | handwriting; KO + ZH + JA in one face | all 11 172 Hangul, GB2312, kana | lxgw/LxgwWenKai (release asset) |

Re-fetch: google/fonts faces from `https://github.com/google/fonts/raw/main/ofl/<dir>/<file>`, the rest from
the release / repo paths above, into `kozh/`.

## `ko_extra/` — more Korean faces (2026-10-11)

User's download (`~/Downloads/Pretendard-1.3.9`, a bundle of four families). Not globbed by `lang_fonts`
on its own: a run names it in `faces.dirs` (`../cjk_ko/configs/jamo_j128_ko.toml`), so the lang runs of record draw
as they did. Every face covers all 11 172 Hangul syllables. File names are ASCII copies of the bundle's.

| file | face | role | licence | source |
|---|---|---|---|---|
| `Pretendard-{Regular,Bold,Black}.ttf` | Pretendard 1.3.9 (3 of its 9 weights; `public/static/alternative/`) | KO dialogue / emphasis gothic | SIL OFL 1.1 (`licenses/ko_extra/Pretendard_OFL.txt`) | orioncactus/pretendard |
| `KMU80TTFHaeongSans.ttf`, `KMU80TTFSungkokSerif.ttf` | KMU80 해옹 Sans / 성곡 Serif (`TTF/`) | heavy sans / serif | no licence file in the bundle, none in the name table | user download |
| `KyoboHandwriting2025lyb.ttf` | 교보손글씨 2025 | handwriting | no licence file in the bundle (name table: kyobobook.co.kr) | user download |
| `GangwonEdu{All-Bold,Saeeum,Teunteun,HyeonokSaem}.ttf` | 강원교육 모두 Bold / 새음 / 튼튼 / 현옥샘 (`강원교육 서체(5종)_TTF`; 모두 Light left out) | gothic / handwriting / poster / handwriting | no licence file in the bundle (name table: gwe.go.kr) | user download |

At 14 px the handwriting faces (새음, 현옥샘, 교보) draw small in the em box and blur, 튼튼 and
Pretendard Black fill their counters: `jamo_j128_ko` floors them in `faces.px_min`.
