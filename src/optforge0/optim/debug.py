from bisect import insort
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

import numpy as np

from .optimizer import Config, Optimizer

if TYPE_CHECKING:
    from .._types import Numeric
    from ..paramdict import ParamDict

__all__ = [
    "PrintParams",
]


class PrintParams(Optimizer):
    CONFIG = Config(
        supports_ask = True,
        supports_multiple_asks = True,
        requires_batch_mode = False,
    )
    def __init__(self):
        super().__init__()

    def step(self, study):
        print('_____ first eval _____')
        print(f'{self.params.used_params}')
        for p, store in self.yield_params_stores():
            print(p, store)
        print()
        # evaluate loss
        loss = study()

        print('_____ second eval _____')
        print(f'{self.params.used_params}')
        for p, store in self.yield_params_stores():
            print(p, store)
        print()

        loss = study()


class MultithreadingDebug(Optimizer):
    CONFIG = Config(
        supports_ask = True,
        supports_multiple_asks = True,
        requires_batch_mode = False,
    )
    def __init__(self, n = 4):
        super().__init__()
        self.n = n
        self.current_step = 0

    def ask(self, study):
        self.current_step += 1
        for i in range(self.n):
            new_params = self.params.copy()
            for p, store in self.yield_params_stores(new_params):
                store['i'] = i
                store['step'] = self.current_step
            print(f'yielding {i}th params')
            yield new_params
        print()


    def tell(self, trials, study):
        print(f'received {len(trials)} evals')
        for ei, e in enumerate(trials):
            for p, store in self.yield_params_stores(e.params):
                print(f'{ei = }, {store["i"] = }, {store["step"] = }')
        print()


class _DefaultOptimizer(Optimizer):
    def ask(self, study): raise AttributeError('Study has no optimizer!')
    def tell(self, trials, study): raise AttributeError('Study has no optimizer!')
    def step(self, study): raise AttributeError('Study has no optimizer!')