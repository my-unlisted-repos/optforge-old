from typing import TYPE_CHECKING

from ..._types import NumericArrayLike, Numeric
from ..benchmark import Benchmark

if TYPE_CHECKING:
    from ...trial import Trial


class NevergradOptimizeFish(Benchmark):
    def __init__(self, time = 365, log_params = True, note = None):
        super().__init__(log_params = log_params, note = note)
        from nevergrad.functions.fishing import OptimizeFish
        self.bench = OptimizeFish(time)
        self.x0 = self.bench.parametrization.value

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        X = trial.suggest_array('X', init = self.x0, low = 0, high = 1)
        return self.bench(X)