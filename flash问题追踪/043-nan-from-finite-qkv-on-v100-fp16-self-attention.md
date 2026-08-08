---
repository: ai-bond/flash-attention-v100
issue: 43
state: open
labels: []
created_at: 2026-07-29T07:59:51Z
updated_at: 2026-07-29T21:36:28Z
source: https://github.com/ai-bond/flash-attention-v100/issues/43
loaded_at: 2026-07-30
---

# NaN from finite Q/K/V on V100 FP16 self-attention (head_dim=128, seq_len=2925)

## Report

The public `flash_attn_func` API returns non-finite dense self-attention output
from finite FP16 Q/K/V on a Tesla V100. The original report used:

- GPU: Tesla V100-SXM2-16GB, SM 7.0
- Python: 3.12
- PyTorch: `2.10.0+cu129`
- CUDA runtime/toolkit: 12.9 / 12.9.1
- Wheel: `flash_attn_v100-26.6-cp312-cp312-linux_x86_64.whl`
- Wheel SHA-256: `74b4cdbd3a225745be82ec44db4104a89bb77094119feec1cc1bf99c7bc1a6a2`
- Package facade/backend: FlashAttention 2.8.3 / V100 backend v26.06
- Initial shape: `(1, 2925, 16, 128)` in FP16
- Call: `flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None)`

Finite checks locate the first non-finite value at the returned attention
tensor, before output projection, residual addition, loss, or backward. Torch
SDPA remains finite on the same inputs. A separate Anima FP16 residual overflow
was fixed independently and does not explain this failure.

## Follow-up Evidence

The reporter published captures and replay results at:

https://github.com/buxinzi2233/MonadForge/releases/tag/diagnostic-v100-20260730

The primary capture contains finite FP16 tensors with shape
`(1, 4130, 16, 128)` and normal input ranges:

| Tensor | Finite range | Non-finite count |
| --- | --- | ---: |
| Q | `[-4.19922, 4.21875]` | 0 |
| K | `[-4.09375, 4.24609]` | 0 |
| V | `[-64.6875, 64.8125]` | 0 |
| Flash output | finite values `[-914, 840]` | 1,412,224 NaN, 562 +Inf, 590 -Inf |

Raw Flash eager, the MonadForge compatibility path in eager mode, and the
compiled compatibility path produced identical output masks and counts. Torch
SDPA was fully finite.

The decisive prefix sweep used the same captured Q/K/V:

- Lengths 4112 and 4128 were finite.
- Every length from 4113 through 4127 was non-finite in V100 FlashAttention.
- Torch SDPA was finite for all 17 lengths.

An independent full Anima forward reproduced the same class of failure at
sequence length 986 (`986 % 16 == 10`) with finite Q/K/V.

## Upstream Responses

The maintainer pointed to head-dimension padding in
`flash_attn_interface.py`. That code pads `head_dim` to a multiple of 8, but it
does not address the observed sequence-length boundary; `head_dim=128` is
already aligned.

No upstream comment refuted the tensor captures, the raw API replay, or the
4112..4128 sweep. The last maintainer response asked for investigation of the
MMA tail path.

## Artifact Timeline Discovered During Analysis

- Release tag `26.06` points to commit
  `d89800edf608d85744f3ab6188be5fd0736acf39`.
- The tested wheel was published on 2026-06-05 and its GitHub asset digest
  exactly matches the issue's SHA-256.
- Commit `8a862131975a5ae213e5733b7a81a104e2c07834`, titled
  `Softmax: Fix compile/runtime tail process in dense kernel`, landed on
  2026-06-25, after the wheel was published.
- Current `main` at analysis time is
  `c91cad40c0539805754819e6ea96c75184d816a6` (2026-06-30).
- No wheel newer than `26.06` is published.

This means the issue demonstrates a defect in the published wheel. It does not
yet demonstrate that current `main` still fails, because the relevant tail fix
has not been rebuilt and replayed on the V100 capture.

## Related Local Material

- `bench/v100_flash/README.md`
- `bench/v100_flash/run_probe.py`
- `tests/test_v100_flash_stability.py`
- MonadForge diagnostic commit `45221697987778b75f54e194f68d4568ea4f9d71`

