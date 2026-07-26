# V100 compile OOMs and the bounded eager fallback for T-LoRA and LoKr

This report records two related memory investigations on a 16 GiB V100 while
training Anima at the 1024px free-fit tier (up to approximately 4200 latent
tokens):

1. why `torch.compile` can be the difference between fitting and OOM for
   ordinary T-LoRA, and why full-factor LoKr can still exceed the same compiled
   memory envelope; and
2. how the non-compiled path was changed so both T-LoRA and full-factor LoKr can
   train without gradient checkpointing or block swapping.

The important distinction is that these are two different memory-management
systems. The compiled path relies on Dynamo/Inductor fusion plus AOTAutograd's
global saved-tensor partition. The eager fallback uses explicit row chunking
and rematerializing custom autograd Functions. Neither mechanism should be
described as a general replacement for the other.

## Target configuration

The bounded eager work targets the following V100 setup:

```toml
torch_compile = false
gradient_checkpointing = false
blocks_to_swap = 0
mixed_precision = "fp16"
gradient_accumulation_steps = 4
attn_mode = "mem_efficient"
```

The precision policy remains mixed precision:

- frozen DiT sublayer GEMMs run in FP16;
- LoRA-family adapter projections run in FP32 on V100 for stability;
- the DiT residual stream and gated residual products accumulate in FP32; and
- adapter outputs are cast back only when they merge into the frozen sublayer
  result.

This is not full-FP32 training. Keeping the frozen model matmuls in FP16 is
required for practical V100 throughput and memory use.

## Why compile normally saves memory here

For the ordinary T-LoRA workload, `torch_compile=true` gains memory from three
coupled effects:

1. Inductor fuses pointwise normalization, rotary, gating, and residual work.
2. The generated graph has explicit tensor lifetime and buffer-reuse planning.
3. AOTAutograd's min-cut partitioner decides which values to retain for
   backward and which values to recompute.

MonadForge exposes the third effect through:

```toml
activation_memory_budget = 0.99
```

In the original T-LoRA investigation, both paths entered the DiT forward at
approximately 4.09 GiB, but reached different peaks:

| T-LoRA path | Peak allocated | Peak reserved |
|---|---:|---:|
| Bounded eager path | 14.217 GiB | 14.496 GiB |
| Compile + budget 0.99 | 12.152 GiB | 12.453 GiB |

The approximately 2 GiB gap therefore came from forward/backward activation
handling, not model weights or the CUDA caching allocator.

The control was `torch_compile=true` with `activation_memory_budget=1.0`. A
cold first forward reached 15.29 GiB allocated and failed an 18 MiB request.
Budget 0.99 was therefore load-bearing for this 16 GiB compiled workload; it
reduced the selected saved set by at least approximately 3.14 GiB relative to
the failing default partition.

This behavior also explains why removing or adding a custom
`autograd.Function` can change compiled memory without changing numerical
results. Such a Function is a partition boundary with an explicit
`ctx.save_for_backward` contract. Replacing it with equivalent traceable ops
gives the AOT partitioner a different graph and can produce a different saved
set. See `docs/findings/custom_autograd_removal_partitioner_oom.md` for the
earlier isolated reproduction.

## The full-factor LoKr compile failure

Full-factor LoKr maps to LyCORIS `full_matrix`: both Kronecker factors are
complete matrices and LyCORIS forces the effective scale to 1. Its parameter
count and checkpoint size are not large enough to explain the observed OOM by
themselves. The failure comes from activation and workspace width.

The reproduced compiled job used:

```toml
torch_compile = true
compile_dynamic_seq = true
activation_memory_budget = 0.99
gradient_checkpointing = false
blocks_to_swap = 0
mixed_precision = "fp16"
attn_mode = "mem_efficient"

use_lokr = true
lokr_factor = 8
lokr_full_factor = true
```

It failed on the first training step with:

```text
allocated by PyTorch: 14.93 GiB
reserved but unallocated: 86.80 MiB
device free: 41.44 MiB
requested: 66.00 MiB
```

The generated Inductor source identified the failed allocation as the frozen
MLP first projection:

```text
shape = (s27, 8192)
dtype = float16
```

At the maximum dynamic sequence length, `s27 = 4200`:

```text
4200 * 8192 * 2 bytes = 65.625 MiB
```

That exactly matches the reported 66 MiB allocation after allocator rounding.
The same graph also contains the full-factor FP32 grouped projections. For
example, one `(8 * s27, 1024)` FP32 intermediate is 131.25 MiB at 4200 tokens,
with additional transpose/clone/output buffers around it.

`activation_memory_budget` can reduce values retained from preceding compiled
blocks, but it cannot remove the live workspace required to execute the
current forward operation. In this run another process occupied 346 MiB of the
GPU, which made the immediate failure easier to trigger, but the compiled
full-factor graph was already operating without a production-safe margin.
`expandable_segments` was enabled and only 86.80 MiB was reserved but unused,
so allocator fragmentation was not the root cause.

The eager memory Functions intentionally return `None` or use the ordinary
traceable path while `torch.compiler.is_compiling()` is true. This preserves
AOTAutograd's authority over compiled graphs. Consequently, enabling compile
does not combine the eager row-chunked MLP with the compiled block today.

## Why ordinary eager autograd OOMed

Disabling compile removes Inductor fusion, buffer lifetime planning, and the
global AOT saved-set decision. Several eager-only peaks then become visible.

For T-LoRA, the major retained or simultaneously live values were:

- FP32 copies of full-width Linear inputs retained for LoRA down-weight
  gradients;
- wide FP32 LoRA-up outputs and scaled residuals;
- explicit FP32 RMSNorm and rotary intermediates; and
- both full `d_ff` pre-activation and GELU activation tensors in each MLP.

For LoKr, there were two additional problems:

1. LyCORIS' eager bypass retained the converted FP32 input and a grouped
   projection close to the base layer's full output width.
2. Anima's bounded eager GELU MLP originally recognized plain LoRA only. LoKr
   silently fell back to ordinary `layer1 -> GELU -> layer2`, retaining the
   full `d_ff` activations.

LyCORIS' bypass does not materialize `torch.kron(w1, w2)` in this training
path. Materializing a full Kronecker delta was therefore not the OOM source.

## Bounded eager implementation

The eager path is enabled only for V100 FP16 training with FP32 adapter
compute, either automatically or through:

```toml
use_custom_down_autograd = true
```

The implementation has four bounded-memory pieces.

### Saved-input LoRA down projection

Plain eager autograd saves the converted FP32 activation used by the rank
projection. `EagerLoRADownProjectFn` and its channel-scaled variant instead
save the original input and weight storage. FP32 casts and scaling are
reconstructed during backward.

This preserves FP32 rank arithmetic without retaining another full-width FP32
copy for every adapted projection.

### Chunked LoRA, normalization, rotary, and MLP work

The Anima eager helpers bound the remaining wide intermediates:

- LoRA-up/residual work is processed in row chunks.
- native RMSNorm avoids explicit full-width FP32 temporary chains.
- rotary q/k work reuses fresh outputs and reconstructs gradients.
- the two-Linear GELU MLP saves the original `d_model` input and rematerializes
  layer1, GELU, and layer2 work by row chunk during backward.

The production T-LoRA chunk sizes were increased after V100 benchmarking to
reduce Python/autograd graph reconstruction and small-kernel launch overhead
while retaining enough memory margin.

### Chunked LoKr bypass

`EagerLoKrResidualFn` evaluates the official grouped LoKr bypass formula by row
chunk. It saves the original activation and factor storage, then reconstructs
the FP32 grouped projections in backward. Factor and scalar gradients are
accumulated in FP32 before their final cast to parameter dtype.

It supports all four LyCORIS layouts used by the wrapper:

- full `w1`, full `w2`;
- full `w1`, decomposed `w2`;
- decomposed `w1`, full `w2`; and
- decomposed `w1`, decomposed `w2`.

The default LoKr chunk is 1024 rows because a full-factor LoKr projection is
substantially wider than a rank-32 LoRA projection.

### LoKr-aware rematerialized GELU MLP

`EagerFusedLoKrMLPFn` extends the bounded two-Linear GELU path to LoKr. Both
adapted linears are recomputed by row chunk in backward, including frozen base
matmuls, GELU, LoKr bypass math, and factor/scalar gradients. The full
`rows x d_ff` activation is never saved.

## V100 results

### T-LoRA

Two independent six-optimizer-step runs completed without OOM or non-finite
values:

| T-LoRA eager revision | Time per optimizer step | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| Initial bounded chunks | 7.09-7.11 s | approximately 14.39 GiB | approximately 14.54 GiB |
| Tuned chunks | 6.56-6.57 s | approximately 14.59 GiB | approximately 14.70 GiB |

The tuned chunks improved complete-training throughput by approximately 7.7%
at the cost of approximately 200 MiB more peak allocated memory.

### Full-factor LoKr

Before the LoKr-aware MLP was added, the unchanged eager configuration failed
during the first frozen MLP layer1 projection:

```text
allocated: 15.30 GiB
device free: 31.44 MiB
requested: 64 MiB
```

After adding both LoKr rematerialization paths, the same six-step configuration
completed twice:

```text
peak allocated: 14.2324 GiB
peak reserved:  14.3164 GiB
time:           approximately 10.93 s per optimizer step
saved modules:  280 native LoKr modules
```

The resulting BF16 checkpoint contained 27,543,320 parameters and occupied
52.63 MiB, confirming that checkpoint parameter volume was not the dominant
source of the original failure.

The isolated wide-projection V100 benchmark used
`4200 x 3072 -> 9216` full-factor LoKr:

| Path | Time | Saved tensors | Reserved workspace |
|---|---:|---:|---:|
| Official eager bypass | 14.32 ms | 198.6 MiB | 814 MiB |
| Chunked, 1024 rows | 19.78 ms | 25.5 MiB | 382 MiB |
| Chunked, 2048 rows | 18.87 ms | 25.5 MiB | 610 MiB |
| Chunked, 3072 rows | 18.74 ms | 25.5 MiB | 728 MiB |

The 1024-row default intentionally accepts projection-level overhead to keep
the complete training run inside 16 GiB.

## Configuration and semantic guards

The known-good eager recipes are:

```toml
# T-LoRA
torch_compile = false
mixed_precision = "fp16"
gradient_checkpointing = false
blocks_to_swap = 0
gradient_accumulation_steps = 4
attn_mode = "mem_efficient"
use_custom_down_autograd = true
```

```toml
# Full-factor LoKr
torch_compile = false
mixed_precision = "fp16"
gradient_checkpointing = false
blocks_to_swap = 0
gradient_accumulation_steps = 4
attn_mode = "mem_efficient"

use_lokr = true
lokr_factor = 8
lokr_full_factor = true
use_custom_down_autograd = true
use_timestep_mask = false
```

LoKr cannot use T-LoRA's timestep mask. T-LoRA masks a shared
`network_dim` bottleneck; LoKr has independent Kronecker factors, and
full-matrix mode has no shared rank axis at all. Configuration parsing and the
WebUI now reject `use_lokr=true` together with `use_timestep_mask=true` instead
of silently training an unmasked adapter.

Channel scaling is also disabled for LoKr because a Kronecker factorization
cannot generally absorb an arbitrary per-input-channel scale while preserving
the same factor layout.

## Validation

The focused implementation, configuration, and benchmark suite completed:

```text
114 passed, 14 warnings
```

Coverage includes:

- LoRA output and gradient parity with and without channel scaling;
- original-storage saved-tensor assertions;
- LoKr output and gradient parity for all four factor layouts;
- LoKr scalar and factor gradient accumulation;
- assertions that full `d_ff` activations are not saved;
- compile guards that keep eager Functions out of Dynamo traces;
- V100 automatic configuration resolution; and
- CLI/WebUI rejection of LoKr plus timestep masking.

`ruff` passed on all changed implementation, test, and benchmark files.
`git diff --check` also passed.

## Scope and remaining compile work

This change provides a bounded eager fallback. It does not claim that the
current full-block compiled full-factor LoKr graph fits every 16 GiB V100.

Configuration-only compile mitigations remain available:

- lower `activation_memory_budget` to retain fewer preceding activations;
- enable `partitioner_aggressive_recomputation` for a larger memory reduction
  with a measured throughput cost on other workloads;
- swap a small number of tail blocks; or
- use gradient checkpointing as the most conservative, highest-recompute
  option.

A future hybrid path could keep attention and residual regions compiled while
placing the wide LoKr MLP behind an eager graph boundary. That requires a
separate performance and correctness investigation: an eager graph break can
change AOT partitioning for the surrounding block, and the current LoKr custom
backward reconstructs local graphs with `torch.autograd.grad`, so it should not
simply be forced into a compiled trace.

The practical result of this work is narrower and verified: when compile is
unavailable, undesirable, or over the V100 memory envelope, 1024px T-LoRA and
full-factor LoKr now have an explicit bounded-memory training path instead of
requiring gradient checkpointing, block swapping, or a lower resolution.
