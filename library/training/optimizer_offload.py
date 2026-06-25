import torch
from torch.optim import Optimizer
from typing import Any, Dict


class OffloadedOptimizer(Optimizer):
    def __init__(self, optimizer: Optimizer, pin_memory: bool = True):
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self._step_counter = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def _move_state_to_cpu(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p not in self.optimizer.state:
                    continue
                state = self.optimizer.state[p]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        if self.pin_memory:
                            state[key] = value.cpu().pin_memory()
                        else:
                            state[key] = value.cpu()

    def _move_state_to_gpu(self):
        device = None
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if device is None:
                    device = p.device
                break
            if device is not None:
                break

        if device is None:
            return

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p not in self.optimizer.state:
                    continue
                state = self.optimizer.state[p]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device, non_blocking=True)

    def step(self, closure=None):
        self._move_state_to_gpu()
        result = self.optimizer.step(closure)
        self._move_state_to_cpu()
        return result

    def state_dict(self) -> Dict[str, Any]:
        self._move_state_to_gpu()
        sd = self.optimizer.state_dict()
        self._move_state_to_cpu()
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.optimizer.load_state_dict(state_dict)
        self._move_state_to_cpu()

    def add_param_group(self, param_group: Dict[str, Any]):
        self.optimizer.add_param_group(param_group)

    def __repr__(self):
        return f"OffloadedOptimizer({self.optimizer!r})"
