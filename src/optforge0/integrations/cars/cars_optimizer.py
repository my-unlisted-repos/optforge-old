#pylint:disable = W0707, W0621
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional
from functools import partial
import numpy as np

from ..._types import Numeric
from ...optim.optimizer import Config, Optimizer
from ...paramdict import ParamDict
from ...python_tools import reduce_dim

if TYPE_CHECKING:
  from cars.optimizers.base_optimizers import BaseOptimizer


__all__ = [
    "CARSOptimizer",
]

class CARSOptimizer(Optimizer):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode=False,
    )
    wrapped_optimizer: "BaseOptimizer | None"
    lib = 'cars'

    def __init__(
        self,
        optimizer: "Callable[..., BaseOptimizer]",
        call_back: Optional[Callable] = None,
    ):
        super().__init__()
        from cars.optimizers.opt_helper_classes import ReachedMaxEvals
        self.ReachedMaxEvals = ReachedMaxEvals
        self.opt_cls: "Callable[..., BaseOptimizer]" = optimizer
        self.config = {'f_name': self.opt_cls.__name__, "f_module": "optforge", "budget": 10000}
        self.call_back = call_back

        if not hasattr(self, 'names'): self.names = [self.opt_cls.__name__, ]

    def set_params(self, params):
        super().set_params(params)
        self.x0, self.slices = self.params.params_to_vec(only_used=False)
        return self

    def step(self, study):

        # create optimizer if it is None
        if self.wrapped_optimizer is None:
            self.wrapped_optimizer = self.opt_cls(
                config=self.config,
                x0=self.x0,
                f = partial(study.evaluate_vec, slices=self.slices, return_scalar = True),
                call_back=self.call_back,
            )

        try: self.wrapped_optimizer.step()
        except self.ReachedMaxEvals:
            self.x0, self.slices = self.params.params_to_vec(only_used=False)
            self.x0 = self.wrapped_optimizer.sol
            self.wrapped_optimizer = None
