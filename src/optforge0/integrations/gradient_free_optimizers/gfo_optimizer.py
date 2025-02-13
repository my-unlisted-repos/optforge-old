#pylint:disable = W0707
from typing import TYPE_CHECKING, Optional, Any

import numpy as np

from ...optim.optimizer import Config, Optimizer
from ...paramdict import ParamDict
from ...python_tools import reduce_dim

if TYPE_CHECKING:
    from gradient_free_optimizers.optimizers.core_optimizer.core_optimizer import CoreOptimizer
    from gradient_free_optimizers.search import Search
    from ..._types import Numeric

__all__ = [
    "GFOWrapper",
    "get_all_gfo_optimizers",
]

def _gfo_names(cls) -> list[str]:
        names: list[str] = [cls.__name__]
        if 'Optimizer' in cls.__name__: names.append(cls.__name__.replace('Optimizer', ''))
        return names

class GFOWrapper(Optimizer):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode=False,
        # I couldn't find if it supports multi-objective anywhere, but maybe it does, need to look at source code
    )
    wrapped_optimizer: "Search"
    def __init__(
        self,
        optimizer_cls: "type[CoreOptimizer]",
        opt_kwargs: Optional[dict[str, Any]] = None,
        continuous_space_steps = 1000,
        budget = None,
        seed = None,
    ):
        super().__init__(seed = seed)
        self.optimizer_cls = optimizer_cls
        self.wrapped_kwargs = opt_kwargs or {}
        self.continuous_space_steps = continuous_space_steps
        self.budget = budget
        self.cur_step = 0

        if not hasattr(self, 'names'): self.names = _gfo_names(optimizer_cls)

    def set_params(self, params:ParamDict):
        super().set_params(params)
        search_space = {}
        for name, param in self.params.items():
            #if param.low is None or param.high is None: raise ValueError("PyOp7 requires low and high bounds for all parameters")
            if param.low is None: low = param.fallback_low
            else: low = param.low
            if param.high is None: high = param.fallback_high
            else: high = param.high

            if param.step is None: step = (high - low) / self.continuous_space_steps
            else: step = param.step

            for idx in range(param.data.size):
                search_space[f'{name}_{idx}'] = np.arange(low, high, step)

        self.wrapped_optimizer = self.optimizer_cls(search_space, random_state=self.rng.seed,) # type:ignore
        return self


    def _closure(self, para: dict[str, float]) -> "Numeric":
        for name, param in self.params.items():
            for idx in range(param.data.size):
                param.data.flat[idx] = para[f'{name}_{idx}']

        self.last_value = self.closure()
        return - self.last_value

    def step(self, study):
        self.closure = study.evaluate_return_scalar
        # if self.cur_step == 0: self.wrapped_optimizer.init_search(
        #     objective_function = self._closure,
        #     n_iter = self.budget,
        #     max_time=None,
        #     max_score=None,
        #     early_stopping=None,
        #     memory=True,
        #     memory_warm_start=None,
        #     verbosity=[],
        #     )
        #self.cur_step += 1
        #self.res = self.wrapped_optimizer.search_step(self._closure)
        self.res = self.wrapped_optimizer.search(self._closure, n_iter=self.budget, verbosity=[])
        if self.last_value is None: raise ValueError(f"{self.wrapped_optimizer.__class__.__name__} didn't do anything")

def get_all_gfo_optimizers() -> "list[type[CoreOptimizer]]":
    import gradient_free_optimizers
    from gradient_free_optimizers.search import Search
    from ...python_tools import subclasses_recursive
    return list(sorted(list(subclasses_recursive(Search)), key=lambda x: x.__name__))