from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..._types import Numeric, NumericArrayLike
from ..benchmark import Benchmark

if TYPE_CHECKING:
    from ...trial import Trial


class NevergradRocket(Benchmark):
    def __init__(self,log_params=True,note=None,):
        super().__init__(log_params = log_params, note = note)
        from nevergrad.functions.rocket import Rocket
        self.bench = Rocket()
        self.x0 = self.bench.parametrization.value

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        return self.bench(
            trial.suggest_array('X', init = self.x0, fallback_low=-10, fallback_high=10)
        )
