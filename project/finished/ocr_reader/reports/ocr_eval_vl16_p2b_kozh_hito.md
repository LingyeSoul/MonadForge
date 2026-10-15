# OCR eval — `vl16_p2b_kozh_hito` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 122 s for 5117 crops (41.9 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 2210 | 86.4 | 0.941 | 90.0 % | 16 |
| speech | 2559 | 2274 | 88.9 | 0.987 | 98.2 % | 9 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 78.4 | 0.910 |
| square | 486 | 87.9 | 0.930 |
| vertical | 1448 | 89.4 | 0.958 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 78.4 | 0.784 |
| 2 | 1017 | 91.7 | 0.949 |
| 3 | 730 | 87.8 | 0.956 |
| 4 | 315 | 81.3 | 0.943 |
| 5 | 167 | 79.0 | 0.927 |
| 6 | 135 | 88.9 | 0.965 |
| 7 | 48 | 70.8 | 0.909 |
| 8+ | 58 | 43.1 | 0.854 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| LoveHina_vol14 003 10006d70 | んにゃー | ヒザー | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ツ | 0.00 |
| SaladDays_vol18 074 1000b7e1+1000b7e2 | ザワ | ガガ | 0.00 |
| SaladDays_vol18 042 1000b738 | ワ | ゴ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| SaladDays_vol18 059 1000b796 | カン | ヤフ | 0.00 |
| SaladDays_vol18 024 1000b70a | ォ | オ | 0.00 |
| SaladDays_vol18 003 1000b68a | ケ | ん | 0.00 |
| ParaisoRoad 103 1000a369 | チン | ぞこ | 0.00 |
| SaladDays_vol18 005 1000b694 | ッ | リ | 0.00 |
| SaladDays_vol18 009 1000b6aa | ギャー | ぎゃー | 0.00 |
| SaladDays_vol18 003 1000b685 | ケ | イ | 0.00 |
| SaladDays_vol18 008 1000b69b | ゴ | コソ | 0.00 |
| SaladDays_vol18 027 1000b718 | ゴ | ア | 0.00 |
| SaladDays_vol18 003 1000b689 | ケケ | ムム | 0.00 |
| ParaisoRoad 103 1000a350 | わー | あー | 0.00 |
| ParaisoRoad 103 1000a358 | わー | キィー | 0.00 |
| ParaisoRoad 018 1000a198 | ィィ | イイ | 0.00 |
| ParaisoRoad 016 1000a183 | じ~ | ぐ~ | 0.00 |
| ParaisoRoad 084 1000a2eb | ふっ | ひ | 0.00 |
| MukoukizuNoChonbo 047 100090b6 | ラ~ | う~ | 0.00 |
| ParaisoRoad 078 1000a2d1 | パッ | ポゥ | 0.00 |
| LoveHina_vol14 019 10006ddf | ぽー | ぼー | 0.00 |
