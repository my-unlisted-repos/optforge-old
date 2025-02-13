from bisect import insort
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

import numpy as np

from .optimizer import Config, Optimizer

if TYPE_CHECKING:
    from .._types import Numeric
    from ..paramdict import ParamDict

__all__ = [
    "BaselineMean",
]


class BaselineNoop(Optimizer):
    CONFIG = Config(
        supports_ask = True,
        supports_multiple_asks = True,
        requires_batch_mode = False,
    )
    names = "baseline no-op", 'no-op'
    def __init__(self):
        super().__init__()

    def ask(self, study):
        yield self.params
BaselineNoop.register()

class BaselineMean(Optimizer):
    CONFIG = Config(
        supports_ask = True,
        supports_multiple_asks = True,
        requires_batch_mode = False,
    )
    names = "baseline mean", 'mean'
    def __init__(self):
        super().__init__()

    def ask(self, study):
        new_params = self.params.copy()
        for param, store in self.yield_params_stores(new_params):
            if param.low is not None and param.high is not None: param.data = np.full(param.data.shape, (param.low + param.high) / 2, )
            else: param.data = np.full(param.data.shape, 0)
        yield new_params
BaselineMean.register()
