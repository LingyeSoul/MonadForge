# `--no_static_pad` — native-shape buckets without the static-pad leak

Design notes for the `--no_static_pad` training flag, an alternative to the
static-shape zero-padding that [`for_compile.md`](for_compile.md) §2.5–2.8
documents. Same goal — keep `torch.compile` from recompiling across aspect
buckets — but it pays for it with a small bounded set of compiled graphs instead
of a single padded shape, which removes a correctness leak the padded path has
under flash attention.

Read [`for_compile.md`](for_compile.md) first; everything here is what changes
when you turn the padding off.

---

## 1. The problem the padding had

The static-shape contract pads every bucket's patch sequence up to
`static_token_count` (4096) so all forwards share one shape and one compiled
block graph. The standing assumption was that the zero-padded patch tokens act
as harmless "attention sinks" — the same contract that holds for the text
encoder's max-length padding.

That assumption is **false for the visual padding**. AdaLN applies a
σ-dependent shift and the Q/K/V projections carry bias, so a zero-input padded
token still emits non-trivial K/V into self-attention. Under `attn_mode="flash"`
(the training default) there is no mask excluding those positions — the flex
`BlockMask` is only built for `attn_mode="flex"` — so they leak into the real
tokens' output.

`bench/static_padding/probe_pad_leak.py` measures it against the native
no-padding forward (flash is deterministic, so that's exact ground truth):

| gap (pad tokens) | bucket | flash rel-L2 vs native |
|---|---|---|
| 0 | 1024×1024 | bit-exact (control) |
| 16 | 960×1088 | ~0.6–0.75% |
| 40–64 | 832×1248 … 896×1152 | ~1%, **spikes to 4–6.5%** at σ 0.5–0.8 |
| 1024–3072 (synthetic) | — | 5–27%, grows with gap and σ |

flex masks it (flat ~0.6%, a kernel difference) but is ~4.5× slower than flash
(`bench/static_padding/bench_attn_speed.py`). So flex is a diagnostic baseline,
not a production answer — the fix is to remove the padding, not mask it.

**This is training-only.** Inference (`library/inference/models.py`) never calls
`set_static_token_count`, so it always ran native/unpadded. Padding to 4096 was
therefore also a pre-existing train/infer mismatch; `--no_static_pad` closes it.

---

## 2. What the flag does

`set_static_token_count(count, pad=False)` keeps constant-token-bucket mode
(block-compile, dataset uses `CONSTANT_TOKEN_BUCKETS`) but runs each bucket at
its native token count.

The load-bearing detail: **it still flattens** the patch grid to the fake-5D
`(B, 1, L, 1, D)` shape — just with `target = seq_len` so the pad amount is
zero. This is not cosmetic. If you skip the flatten and hand blocks the native
`(B, T, H, W, D)`, dynamo guards on `H` and `W` *separately* and recompiles once
per **resolution** (17 buckets). Flattening makes the block stack key on token
count alone, so the 17 resolutions collapse to the **5 distinct token-counts**
they actually have:

```
{4032, 4050, 4056, 4080, 4096}
```

Because there is no padding, the self-attention pad-mask and `compile_core`'s
single-shape assumption no longer apply:

- the flex self-attn `BlockMask` is gated off (`pad_to_static`),
- `compile_core` / `--compile_mode full` is **rejected** — `_run_blocks` is only
  shape-invariant when padded. Use the default `--compile_mode blocks`.

Implementation: `library/anima/models.py` (`pad_to_static` attr +
`forward_mini_train_dit`), `library/anima/training.py` (`--no_static_pad`),
`train.py` (passes the flag, bumps `cache_size_limit`, rejects `full`).

---

## 3. Cost: graphs and memory

`bench/static_padding/bench_compile_mem.py` (forward-only, inductor default, no
CUDAGraphs):

| mode | shapes run | unique dynamo graphs | peak alloc |
|---|---|---|---|
| pad (`pad=True`) | 1 | 2 | 4.12 GiB |
| no-pad (`pad=False`) | **17 resolutions** | **10** (= 5 token-counts × fwd/bwd) | 7.77 GiB |

The "10 graphs from 17 resolutions" result is the proof the collapse works. The
recompile budget is bounded and one-time: `train.py` sets
`torch._dynamo.config.cache_size_limit = 2·5 + 8 = 18`, comfortably above 10, so
no silent fallback to eager mid-warmup.

Memory overhead is **~+3.7 GiB** forward-only from holding 5 graphs instead of 1.
A real `make lora --no_static_pad` run lands at ~13.4 GiB peak (fwd+bwd), inside
a 16 GB card. That overhead is the reason the flag is **opt-in, not default**:
on low_vram / 8 GB presets it does not fit. `--compile_mode full` aside,
`reduce-overhead` (per-shape CUDAGraph memory pools) is expected to multiply the
overhead further — bench before pairing.

---

## 4. When to use it

| | static pad (default) | `--no_static_pad` |
|---|---|---|
| flash self-attn correctness | leaks (up to 6.5% at the 4032 buckets) | **bit-exact** to native |
| compiled graphs | 1 | 5 (block-compile only) |
| extra VRAM | — | ~+3.7 GiB |
| `--compile_mode full` | supported | rejected |
| good for | 8 GB / low_vram, max-throughput single shape | 16 GB+ where the leak matters |

Turn it on when the static-pad leak is a concern (default-stack flash training
on the leaky 4032-token aspect buckets) and you have the VRAM headroom. Leave it
off on memory-constrained presets or when using `compile_mode=full`.

---

## 5. Status / open items

- Verified correct (probe: bit-exact) and graph-collapsed (mem bench: 10 graphs).
- **Not yet** validated end-to-end: a full run with loss/CMMD parity vs the
  padded path is still outstanding.
- Existing shipped adapters were trained *under* the padded (leaky) regime; the
  base DiT is frozen so the change should be neutral-to-better, but re-bench one
  checkpoint native before flipping the project default.

See also: [`for_compile.md`](for_compile.md) (the static-shape foundation),
[`full_model_cudagraph.md`](full_model_cudagraph.md) (the incompatible `full`
mode), and `bench/static_padding/README.md` (the probes).
