"""Sarma, M.S. (1990). "On the convergence of the Baba and Dorea random optimization methods". Journal of Optimization Theory and Applications. 66 (2): 337–343. doi:10.1007/bf00939542."""
from typing import TYPE_CHECKING

import numpy as np

from .._types import NumericArrayLike, Numeric
from .benchmark import Benchmark

if TYPE_CHECKING:
    from ..optim.optimizer import Optimizer
    from ..trial import Trial

class TransformerDesignSarma(Benchmark):
    """This is a different formulation of the same problem from `ballard_problems`."""
    def __init__(self, constr_weight = 1e4, log_params = True, note = None):
        super().__init__(log_params=log_params, note=note)
        self.constr_weight = constr_weight
    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        x1 = trial.suggest_float('x1', low = 1e-8, fallback_high=100)
        x2 = trial.suggest_float('x2', low = 1e-8, fallback_high=100)
        x3 = trial.suggest_float('x3', low = 1e-8, fallback_high=100)
        x4 = trial.suggest_float('x4', low = 1e-8, fallback_high=100)
        x5 = trial.suggest_float('x5', low = 1e-8, fallback_high=100)
        prod5 = x1 * x2 * x3 * x4 * x5
        x6 = trial.suggest_constant('x6', 2070 / prod5) # so product = 2070

        K = 0.16225
        D = K * x1 * x2 * x3 - (x1 + x2 + x3)
        x2 = trial.param_constr_bounds('x2', low = 1 / (K * x1))
        x3 = trial.param_constr_bounds('x3', low = (x1 + x2) / (K * x1 * (x2 - 1)))
        x4 = trial.param_constr_bounds('x4', low = (x1 + x2 + x3) + (x1 + 1.57 * x2) / D) # division by zero why?

        B = 0.00058 * (x1 + 0.57 * x2 + x4) * 2.07 ** 2 * (10 ** 6 / (x1 ** 2 * x2 * x3 * x4 ** 2))
        B = trial.constr_bounds(B, low = 1e-8, constr_name='B', weight=self.constr_weight)
        A = 0.00062 * x1 * x4 * (x1 + x2 + x3)
        A = trial.constr_bounds(A, low = 1e-8, constr_name='A', weight=self.constr_weight)
        AB = A * B
        AB = trial.constr_bounds(AB, high = 1/4, constr_name='AB', weight=self.constr_weight)
        A = AB / B

        oned2A = 1 / (2 * A)
        c = oned2A * ((1 - 4 * AB) ** (1/2))
        trial.constr_bounds(x5**2, low = oned2A-c, high=oned2A+c, weight=self.constr_weight)

        L1 = 0.0204 * x1 * x4
        L2 = 0.0607 * x1 * x4 * x5 ** 2
        L3 = 0.0187 * x2 * x3
        L4 = 0.0437 * x2 * x3 * x6 ** 2
        return (L1 + L3) * (x1 + x2 + x3) + (L2 + L4) * (x1 + 1.57 * x2 + x4)