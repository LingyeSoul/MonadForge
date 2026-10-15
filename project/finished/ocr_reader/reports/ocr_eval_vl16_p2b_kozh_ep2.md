# OCR eval — `vl16_p2b_kozh_ep2` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 125 s for 5117 crops (40.8 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 2223 | 86.9 | 0.942 | 90.3 % | 19 |
| speech | 2559 | 2273 | 88.8 | 0.988 | 98.2 % | 9 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 80.4 | 0.913 |
| square | 486 | 88.3 | 0.933 |
| vertical | 1448 | 89.2 | 0.957 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 78.4 | 0.803 |
| 2 | 1017 | 92.3 | 0.952 |
| 3 | 730 | 87.4 | 0.953 |
| 4 | 315 | 82.5 | 0.946 |
| 5 | 167 | 79.0 | 0.931 |
| 6 | 135 | 89.6 | 0.960 |
| 7 | 48 | 75.0 | 0.914 |
| 8+ | 58 | 48.3 | 0.818 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 085 1000b824 | イ | ィ | 0.00 |
| SaladDays_vol18 085 1000b81d | イ | ィ | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| MAD_STONE 077 10007094 | グオ | ギバ | 0.00 |
| SaladDays_vol18 074 1000b7e1+1000b7e2 | ザワ | ドガッ | 0.00 |
| SaladDays_vol18 082 1000b806 | ダッ | ゴ | 0.00 |
| SaladDays_vol18 081 1000b802 | ぱぁ | ぽあ | 0.00 |
| MAD_STONE 060 1000706a | バギャ | ボカ | 0.00 |
| ParaisoRoad 018 1000a19b | ィ | イ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| ParaisoRoad 016 1000a183 | じ~ | ぐ~ | 0.00 |
| MukoukizuNoChonbo 041 1000907a | リー | ソー | 0.00 |
| SaladDays_vol18 024 1000b70a | ォ | オ | 0.00 |
| ParaisoRoad 036 1000a20d | ドゴ | ぐぐ | 0.00 |
| ParaisoRoad 059 1000a277 | あ | 持 | 0.00 |
| SaladDays_vol18 003 1000b68a | ケ | ム | 0.00 |
| SaladDays_vol18 001 1000b677 | ガヤ | がや | 0.00 |
| SaladDays_vol18 001 1000b67f | ガヤ | がや | 0.00 |
| ParaisoRoad 103 1000a369 | チン | ぇこ | 0.00 |
| ParaisoRoad 103 1000a358 | わー | キィー | 0.00 |
| SaladDays_vol18 003 1000b689 | ケケ | ムム | 0.00 |
| SaladDays_vol18 008 1000b69b | ゴ | コソ | 0.00 |
| SaladDays_vol18 003 1000b685 | ケ | イ | 0.00 |
| ParaisoRoad 018 1000a198 | ィィ | イイ | 0.00 |
