#pylint:disable = W0707, W0621
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..._types import Numeric
from ...optim.optimizer import Config, Optimizer
from ...paramdict import ParamDict
from ...python_tools import reduce_dim

if TYPE_CHECKING:
  from pysors.utils import SecondOrderRandomSearchOptimizer

  from pysors import ALL_METHODS

__all__ = [
    "METHODS_LIST",
    "PySORSOptimizer",
]
METHODS_LIST ="stp","bds", 'ahds','rs','rspifd', 'rspispsa'
class PySORSOptimizer(Optimizer):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode=False,
    )
    def __init__(self, optimizer: "SecondOrderRandomSearchOptimizer | Literal['stp','bds', 'ahds','rs','rspifd', 'rspispsa']"):
        super().__init__()

        from pysors import ALL_METHODS
        if isinstance(optimizer, str): optimizer = ALL_METHODS[optimizer]() # type:ignore
        self.wrapped_optimizer: "SecondOrderRandomSearchOptimizer" = optimizer
        self._best_value = float('inf')

        if not hasattr(self, 'names'): self.names = [self.wrapped_optimizer.__class__.__name__, ]

    def set_params(self, params:ParamDict):
        super().set_params(params)
        self.x, self.slices = self.params.params_to_vec(only_used=False)
        return self

    def step(self, study):
        self.x = self.wrapped_optimizer.step(partial(study.evaluate_vec, slices = self.slices, return_scalar = True), self.x) # type:ignore

class RHO:
    def __init__(self, p=2., c=0.0):
        self.p = p
        self.c = c
    def __call__(self, z):
        return self.c*z**self.p