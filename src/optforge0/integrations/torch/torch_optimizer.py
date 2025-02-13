from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from functools import partial
from typing import Any, Optional, TYPE_CHECKING

import torch

from ..._utils import _ensure_float
from ...optim.optimizer import Config, Optimizer
from ...paramdict import ParamDict
from ...study import Study
from ...trial.trial import Trial

__all__ = [
    "TorchOptimizer",
    "Optforge2Torch"
]
class TorchOptimizer(Optimizer):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode=False,
        
    )
    wrapped_optimizer: "torch.optim.Optimizer" # type:ignore
    def __init__(self, optimizer_constructor:Optional[Callable] = None, **extra_kwargs):
        super().__init__()
        self.wrapped_constructor = optimizer_constructor
        self.wrapped_kwargs = extra_kwargs

    def make_wrap_args(self) -> tuple[Sequence, Mapping]:
        if len(self.params) == 0:
            self.torch_params = {}
            return (([torch.empty(1, requires_grad=True)], ), {})
        else:
            self.torch_params = {k: torch.from_numpy(v.data).requires_grad_(True) for k, v in self.params.items()}
            return ((self.torch_params.values(), ), {})

    def _closure(self):
        self.wrapped_optimizer.zero_grad()
        for k, v in self.params.items():
            if k in self.torch_params: v.data = self.torch_params[k].detach().numpy()
        return self.closure()

    def step(self, study):
        self.closure = study.evaluate_return_scalar
        for name, p in self.params.items():
            if name not in self.torch_params:
                self.torch_params[name] = torch.from_numpy(p.data).requires_grad_(True)
                self.wrapped_optimizer.add_param_group({'params': [self.torch_params[name]]})
        loss = self.wrapped_optimizer.step(self._closure) # type:ignore

class Optforge2Torch(torch.optim.Optimizer): # type:ignore
    def __init__(self, parameters, optimizer:Optimizer, low=None, high=None, scale = None, enable_grad = False):
        super().__init__(parameters, {})
        self.low = low; self.high = high; self.scale = scale
        self.enable_grad = enable_grad

        self.study = Study()

        optimizer.set_params(self._param_init())
        self.optimizer = optimizer

    def _param_init(self):
        trial = Trial(ParamDict(), self.study, asked=False)
        for gi, g in enumerate(self.param_groups):
            for pi, param in enumerate(g['params']):
                if param.requires_grad:
                    trial.suggest_array(
                        f"param_{gi}_{pi}",
                        shape=param.shape,
                        low=self.low,
                        high=self.high,
                        scale=self.scale,
                        init=param.detach().cpu().numpy(),
                    )
        return trial.params

    @torch.no_grad
    def _closure(self, trial:Trial):
        for gi, g in enumerate(self.param_groups):
            for pi, param in enumerate(g['params']):
                if param.requires_grad:
                    p = trial.suggest_array(
                        f"param_{gi}_{pi}",
                        shape=param.shape,
                        low=self.low,
                        high=self.high,
                        scale=self.scale,
                    )
                    param.set_(torch.as_tensor(p, dtype=param.dtype, device = param.device))
        self.last_loss = _ensure_float(self.closure())
        return self.last_loss

    @torch.no_grad
    def step(self, closure): # type:ignore #pylint:disable=W0222
        self.closure = closure
        with torch.enable_grad() if self.enable_grad else nullcontext():
            self.study.step(self._closure, self.optimizer) # type:ignore
        return self.last_loss