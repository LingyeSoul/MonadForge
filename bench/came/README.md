# CAME multi-resolution compatibility bench

This bench verifies the CAME optimizer integration used by MonadForge. It is a
small CPU regression probe, not a comparison between CAME and other optimizers.

## What it measures

`probe_multires.py` trains one convolution against a fixed teacher while
alternating among `8x8`, `12x16`, and `16x12` inputs. It checks that:

- Adam-style two-beta configuration is accepted by the CAME integration.
- every loss and update remains finite;
- optimizer state shapes depend on parameter shapes, not input resolution;
- the aggregate teacher-fitting loss decreases over the run.

The focused unit tests in `tests/test_optimizer_branches.py` separately lock
the boundary between the built-in `pytorch_optimizer.CAME` compatibility shim
and arbitrary fully-qualified third-party optimizers named `CAME`.

## Run

From the repository root:

```bash
uv run python bench/came/probe_multires.py --steps 48 --label verify
```

On Windows without `uv` on `PATH`, use the project environment directly:

```powershell
.\.venv\Scripts\python.exe bench\came\probe_multires.py --steps 48 --label verify
```

## Pass criteria

The command exits successfully only when both of these conditions hold:

- `came_descends` is true (`final_aggregate_mse < initial_aggregate_mse`);
- `state_shapes_invariant` is true across all three input resolutions.

`finite` must also remain true; it is included in `came_descends`.

## Output

Each run writes a standard benchmark envelope under
`bench/came/results/<timestamp>-probe_multires-<label>/`:

- `result.json`: environment, arguments, pass signals, and headline metrics;
- `loss_curve.csv`: per-step resolution and MSE.

The `results/` tree is ignored by default. Keep the exact command and headline
metrics in the change or pull-request description when using this as evidence.

## Observed baseline

Observed on 2026-07-13 with Python 3.13.9, PyTorch 2.12.0, CPU, seed `0`,
48 steps, and learning rate `2e-3`:

| Metric | Result |
|---|---:|
| Initial aggregate MSE | 28.258284 |
| Final aggregate MSE | 3.138827 |
| Loss reduction | 9.0028x |
| Finite | true |
| State shapes invariant | true |

The expected result is directional rather than bit-exact across PyTorch and
platform versions: the run must remain finite, preserve state shapes, and end
below its initial aggregate MSE.
