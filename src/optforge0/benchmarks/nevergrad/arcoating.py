from typing import TYPE_CHECKING

from ..._types import NumericArrayLike, Numeric
from ..benchmark import Benchmark

if TYPE_CHECKING:
    from ...trial import Trial


class NevergradARCoating(Benchmark):
    def __init__(self, nbslab: int = 10,d_ar: int = 400, log_params = True, note = None):
        """Anti-reflective coating benchmark from nevergrad. A continuous, bounded and single-objective problem.
        Gradient based methods work great on it (BFGS in particular).

         Notes from nevergrad documentation:

        - This is the minimization of reflexion, i.e. this is an anti-reflexive coating problem in normale incidence.
        - Typical parameters (nbslab, d_ar) = (10, 400) or (35, 700) for instance
        - d_ar / nbslab must be at least 10

        :param nbslab: _description_, defaults to 10
        :param d_ar: _description_, defaults to 400
        :param log_params: _description_, defaults to True
        :param note: _description_, defaults to None
        """
        super().__init__(log_params = log_params, note = note)
        from nevergrad.functions.arcoating import ARCoating
        self.bench = ARCoating(nbslab, d_ar)
        self.x0 = self.bench.parametrization.value

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        X = trial.suggest_array('X', init = self.x0, low = self.bench.epmin, high = self.bench.epf)
        return self.bench(X)