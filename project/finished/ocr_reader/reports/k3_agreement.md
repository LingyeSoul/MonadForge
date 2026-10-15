# K3 — teacher × voter for the KO / ZH relabel

Precision = kept rows whose label (the teacher's string) is exact-right; `spaced` = right *with* spacing (the KO 띄어쓰기 label). ±1σ Wilson.

## Readers alone

| lang | n | stock | v4 | hayai25 |
|---|---|---|---|---|
| ko | 48 | 21 | 26 | 28 |
| zh | 48 | 31 | 30 | 16 |

## Agreement filters

| lang | teacher | voter | kept / n | precision exact | precision spaced | right rows lost |
|---|---|---|---|---|---|---|
| ko | stock | hayai25 | 20 / 48 | 19/20 (0.88–0.98) | 13/20 | 2 |
| ko | v4 | hayai25 | 26 / 48 | 24/26 (0.85–0.96) | 21/26 | 2 |
| ko | v4 | stock | 22 / 48 | 20/22 (0.83–0.95) | 17/22 | 6 |
| zh | stock | hayai25 | 13 / 48 | 12/13 (0.82–0.97) | 9/13 | 19 |
| zh | v4 | hayai25 | 14 / 48 | 12/14 (0.74–0.93) | 11/14 | 18 |
| zh | v4 | stock | 29 / 48 | 26/29 (0.83–0.94) | 22/29 | 4 |

## v4 alone, gated on mean token logprob

| lang | logprob ≥ | kept / n | precision exact | precision spaced |
|---|---|---|---|---|
| ko | -1.015 (q0.00) | 46 / 48 | 26/46 (0.49–0.64) | 22/46 |
| ko | -0.146 (q0.25) | 34 / 48 | 25/34 (0.65–0.80) | 22/34 |
| ko | -0.077 (q0.50) | 23 / 48 | 20/23 (0.78–0.92) | 19/23 |
| ko | -0.012 (q0.75) | 12 / 48 | 12/12 (0.92–1.00) | 12/12 |
| zh | -0.837 (q0.00) | 47 / 48 | 29/47 (0.54–0.68) | 25/47 |
| zh | -0.215 (q0.25) | 35 / 48 | 26/35 (0.66–0.81) | 23/35 |
| zh | -0.065 (q0.50) | 24 / 48 | 20/24 (0.74–0.90) | 18/24 |
| zh | -0.016 (q0.75) | 12 / 48 | 11/12 (0.80–0.97) | 10/12 |
