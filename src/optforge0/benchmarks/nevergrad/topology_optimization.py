from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..._types import Numeric, NumericArrayLike
from ..benchmark import Benchmark

if TYPE_CHECKING:
    from ...trial import Trial


class NevergradTO(Benchmark):
    def __init__(self, n = 50, complex_tsp = False, log_params=True,note=None,):
        super().__init__(log_params = log_params, note = note)
        from nevergrad.functions.topology_optimization import TO
        self.bench = TO(n=n)
        self.x0 = self.bench.parametrization.value

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        return self.bench(
            trial.suggest_array('X', init = self.x0, low = -1, high = 1)
        )
