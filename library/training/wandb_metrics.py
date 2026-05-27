"""WandB-specific metric collectors for professional training dashboards.

Three collectors that produce richer metrics than the basic scalar loss/lr
pipeline, gated behind env-var flags so they never run unless the user opts in.

Collectors follow the same ``MetricProducer`` protocol used by the network
and adapter metric producers, but they also return wandb-native objects
(``wandb.Histogram``) that ``dispatch_wandb_extras`` routes to wandb only.
"""

from __future__ import annotations


import torch


# ---------------------------------------------------------------------------
# System metrics — GPU util/mem/temp + CPU/RAM
# ---------------------------------------------------------------------------

class SystemMetricsCollector:
    """Collect GPU and CPU/RAM metrics at every log step.

    Uses ``torch.cuda`` for GPU stats (accurate in the training process) and
    ``psutil`` for host stats.  Returns plain floats that merge into the
    normal scalar log dict.
    """

    def __init__(self) -> None:
        self._has_cuda = torch.cuda.is_available()
        self._psutil = None
        try:
            import psutil as _ps  # noqa: F811
            self._psutil = _ps
        except ImportError:
            pass
        self._peak_mem: float = 0.0

    def collect(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self._has_cuda:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / (1024 ** 3)
            used_gb = torch.cuda.memory_allocated(idx) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(idx) / (1024 ** 3)
            peak_gb = torch.cuda.max_memory_allocated(idx) / (1024 ** 3)
            self._peak_mem = max(self._peak_mem, peak_gb)

            out["sys/gpu_mem_used_gb"] = round(used_gb, 2)
            out["sys/gpu_mem_reserved_gb"] = round(reserved_gb, 2)
            out["sys/gpu_mem_total_gb"] = round(total_gb, 2)
            out["sys/gpu_mem_peak_gb"] = round(self._peak_mem, 2)
            out["sys/gpu_mem_%"] = round(used_gb / total_gb * 100, 1) if total_gb > 0 else 0

            # Temperature via nvidia-smi is expensive per-step; use NVML if
            # available, otherwise skip (torch.cuda doesn't expose temp).
            try:
                torch.cuda.temperature  # Python 3.13+ / torch 2.6+
            except (AttributeError, TypeError):
                pass  # not available — skip silently

        if self._psutil is not None:
            out["sys/cpu_%"] = self._psutil.cpu_percent(interval=0)
            vm = self._psutil.virtual_memory()
            out["sys/ram_%"] = vm.percent
            out["sys/ram_used_gb"] = round(vm.used / (1024 ** 3), 1)

        return out


# ---------------------------------------------------------------------------
# Gradient histograms — per-layer grad-norm distribution
# ---------------------------------------------------------------------------

class GradientHistogramCollector:
    """Log ``wandb.Histogram`` of per-layer gradient norms.

    Only produces values every *freq* steps and only when wandb is available.
    Returns a mixed dict: plain floats for summary stats + ``wandb.Histogram``
    objects that ``dispatch_wandb_extras`` forwards to wandb only.
    """

    def __init__(self, freq: int = 50) -> None:
        self.freq = max(1, freq)
        self._step_counter = 0
        self._has_wandb = False
        self._wb = None
        try:
            import wandb as _wb  # noqa: F811
            self._wb = _wb
            self._has_wandb = True
        except ImportError:
            pass

    def should_collect(self, global_step: int) -> bool:
        return self._has_wandb and global_step > 0 and global_step % self.freq == 0

    def collect(self, network) -> dict:
        if not self._has_wandb:
            return {}
        norms_by_group: dict[str, list[float]] = {}
        with torch.no_grad():
            for name, param in network.named_parameters():
                if param.grad is None:
                    continue
                # Extract a short group name: e.g. "lora_up.weight" → "lora_up"
                parts = name.split(".")
                group = parts[0] if len(parts) > 0 else name
                norm = float(param.grad.data.norm(2))
                norms_by_group.setdefault(group, []).append(norm)

        out: dict = {}
        all_norms: list[float] = []
        for group, norms in norms_by_group.items():
            all_norms.extend(norms)
            out[f"grad_hist/{group}"] = self._wb.Histogram(norms)

        if all_norms:
            out["grad/global_norm_mean"] = sum(all_norms) / len(all_norms)
            out["grad/global_norm_max"] = max(all_norms)

        return out


# ---------------------------------------------------------------------------
# Weight snapshots — per-layer weight-norm distribution
# ---------------------------------------------------------------------------

class WeightSnapshotCollector:
    """Log ``wandb.Histogram`` of per-layer weight value distributions.

    Fires at epoch boundaries (or every *freq* steps when configured).
    Returns a mix of plain summary floats and ``wandb.Histogram`` objects.
    """

    def __init__(self, freq: int = 0) -> None:
        # freq=0 means "epoch only" (caller decides when to fire)
        self.freq = freq
        self._step_counter = 0
        self._has_wandb = False
        self._wb = None
        try:
            import wandb as _wb  # noqa: F811
            self._wb = _wb
            self._has_wandb = True
        except ImportError:
            pass

    def should_collect(self, global_step: int = 0, epoch_boundary: bool = False) -> bool:
        if not self._has_wandb:
            return False
        if epoch_boundary:
            return True
        if self.freq > 0 and global_step > 0 and global_step % self.freq == 0:
            return True
        return False

    def collect(self, network) -> dict:
        if not self._has_wandb:
            return {}
        layers: dict[str, list[float]] = {}
        with torch.no_grad():
            for name, param in network.named_parameters():
                parts = name.split(".")
                group = parts[0] if len(parts) > 0 else name
                flat = param.data.detach().cpu().float().flatten().tolist()
                # Sample at most 4096 values per layer to avoid huge uploads
                if len(flat) > 4096:
                    import random
                    flat = random.sample(flat, 4096)
                layers.setdefault(group, []).extend(flat)

        out: dict = {}
        all_norms: list[float] = []
        for group, values in layers.items():
            out[f"weight_hist/{group}"] = self._wb.Histogram(values)
            norms = [abs(v) for v in values]
            all_norms.extend(norms)

        if all_norms:
            out["weight/mean_abs"] = sum(all_norms) / len(all_norms)
            out["weight/max_abs"] = max(all_norms)

        return out
