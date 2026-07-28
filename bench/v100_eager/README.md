# bench/v100_eager

Tier 1.5 before/after benchmark for the bounded eager operators used by V100
FP16 training. It covers the cross-cutting paths outside the LoKr-only
projection benchmark:

- LoRA down projection;
- LoRA up plus residual;
- two-linear GELU MLP with FP32 LoRA rank math;
- two-linear GELU MLP with official full-factor LoKr bypass math;
- rotary q/k; and
- explicit FP32 RMSNorm versus the native kernel selected on V100.

Each case reports saved-tensor bytes and forward-plus-backward time. CUDA runs
also report peak allocated and reserved workspace above the allocator baseline.
The standard `result.json` is written below `bench/v100_eager/results/`.

## Run

The defaults are CPU-friendly and prove every comparison is runnable:

```bash
python -m bench.v100_eager.run_bench
```

Use production-scale Anima dimensions on a V100 for representative numbers:

```bash
python -m bench.v100_eager.run_bench \
  --device cuda --rows 4200 --model-dim 2048 --ffn-dim 8192 \
  --rank 32 --heads 16 --chunk 3072 --lokr-chunk 1024 --label v100
```

`--cases` accepts any subset of `lora_down`, `lora_up`, `mlp`, `lokr_mlp`,
`rope`, and `rms_norm`. CPU timing and the CPU RMSNorm backend are smoke data;
only a V100 run measures the production CUDA kernels and allocator behavior.

`bench/lokr/run_eager_memory_bench.py` remains the official-LyCORIS versus
rematerialized LoKr factor benchmark.
