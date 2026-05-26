# ComfyUI Anima Block Compile

A single ComfyUI node — **Anima Block Compile** (`AnimaBlockCompile`) — that
applies `torch.compile` to the Anima DiT **one transformer block at a time**
instead of compiling the whole `diffusion_model` in one frame.

## Why

Anima's `diffusion_model.blocks` is a plain `nn.ModuleList` of identical
transformer blocks. Compiling each block on its own:

- **compiles much faster** — dynamo builds one small graph and reuses it across
  all N blocks, instead of tracing one giant graph;
- **breaks less** — each block is a small, regular subgraph, so you're far less
  likely to hit a graph break or a recompile that silently falls back to eager.

This is the same strategy the `anima_lora` training/inference pipeline uses
(`DiT.compile_blocks`).

Under the hood this is just ComfyUI core's
`set_torch_compile_wrapper(model, keys=[...])` pointed at
`diffusion_model.blocks.{i}`. The bundled `TorchCompileModel` node only does
whole-model compile; KJNodes' `TorchCompileModelAdvanced` can do per-block but
behind a generic name and a layer-name heuristic. This node is the
one-purpose, Anima-named equivalent.

## Install

Clone or copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.
(Inside the `anima_lora` repo it's already under `custom_nodes/`, which is
symlinked into the sibling ComfyUI install.)

No extra dependencies — it only uses `torch` and ComfyUI's own compile API.

## Usage

Wire it between your model loader (e.g. `UNETLoader`) and the sampler:

```
UNETLoader ──> Anima Block Compile ──> KSampler / Spectrum KSampler ──> ...
```

The returned MODEL is a clone with a compile wrapper installed. **Compilation
is lazy** — it happens on the first sample, so the first generation after
wiring this in is slow and every subsequent one is fast.

The only input is `model`. There are no knobs: it always uses the `inductor`
backend with default settings, applied per transformer block — the safe, fast
configuration. (Per-block compile makes the aggressive `mode` / `fullgraph`
options unnecessary, which is why they aren't exposed.)

## Notes

- Composes with the Anima adapter / Spectrum KSampler nodes — compile is
  installed as a sample-time wrapper and restored after each `apply_model`.
- Experimental, like all torch.compile nodes.
