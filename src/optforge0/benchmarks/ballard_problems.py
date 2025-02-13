"""
Problems from
```
D. H. Ballard, C. O. Jelinek, R. Schinzinger, An algorithm for the solution of constrained generalised polynomial programming problems, The Computer Journal, Volume 17, Issue 3, 1974, Pages 261–266, https://doi.org/10.1093/comjnl/17.3.261
```
"""

from typing import TYPE_CHECKING

import numpy as np

from .._types import NumericArrayLike, Numeric
from .benchmark import Benchmark

if TYPE_CHECKING:
    from ..optim.optimizer import Optimizer
    from ..trial import Trial

__all__ = [
    "WasteTreatmentPlantDesign",
    "ChemicalEquilibriumProblem",
    "TransformerDesign",
    #"_CataloguePlanning",
    #"_ChemicalProcessControlProblem",
]

class WasteTreatmentPlantDesign(Benchmark):
    """Problem 1. Minima reported in paper is 71.765e3, value with minima params provided in paper is 71763.7674943996.
    I was able to reach 71758.45384677197 with L-BFGS-B (presumaly in paper lower float precision was used).
    This is generally a very easy problem for global, local and gradient approximation methods.
    L-BFGS-B converges to a very good solution in 200 iterations.
    """
    def __init__(self, constr_weight = 1, log_params = True, note = None):
        super().__init__(log_params=log_params, note=note)
        self.constr_weight = constr_weight

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        x1 = trial.suggest_float('x1', low = 1e-8) # fraction of feed chemical oxygen demand not met (dimensionless), minima at 0.6169
        x2 = trial.suggest_float('x2', low = 1e-8, scale = 1e-6, fallback_high=1e7) # influent volitile solids concentration (lb/million) gal, minima at 5.814e5
        x3 = trial.suggest_float('x3', low = 1e-8, scale = 1e-6, fallback_high=1e7) # effluent volitile solids concentration (lb/million) gal, minima at 2.999e5

        try:
            y0 = 2.1 * 10**-11 * x2**2.55 + 6.29 * 10**7 * x2**5 / x3**6 + 8.5 * 10**10 / (x2 * x3**0.2 * x1**2)\
                + 1.6 * 10**5 * x1**2.5 * x3 / x2 # ?
        except OverflowError: return float('inf')
        y1 = (1/3) * 10**-5 * x3 # ?

        trial.constr_bounds(y1, high = 1, weight = self.constr_weight, constr_name='y1')
        return y0

    def get_minima(self):
        #return 71.765e3
        return 71758.45384677197
    def get_minima_params(self):
        # params suggested in paper return dict(x1 = 0.6169, x2 = 5.814e5, x3 = 2.999e5)
        return {'x1': 0.6168611530822911, 'x2': 581517.9097822552, 'x3': 299999.99999927485}

class ChemicalEquilibriumProblem(Benchmark):
    """Problem 2. Minima reported in paper is 5.525e20, with params provided in paper it is 5.5258e20.
    Random annealing with 1m evals was able to reach 4.978323454476186e+19.
    This is a harder problem, with many very small local minima.
    It doesn't seem to favor local or global methods and works with gradient approximation.
    About 70% of the landscape is not viable,
    """
    def __init__(self, constr_weight = 1, log_params = True, note = None):
        super().__init__(log_params=log_params, note=note)
        self.constr_weight = constr_weight

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        x1 = trial.suggest_float('x1', low = 1e-12, scale = 1e6, fallback_high=1e-3) # composition variables, minima at 5.628e-5
        x2 = trial.suggest_float('x2', low = 1e-12, scale = 1e6, fallback_high=1e-3) # composition variables, minima at 2.450e-7
        x3 = trial.suggest_float('x3', low = 1e-12, scale = 1e6, fallback_high=1e-3) # composition variables, minima at 2.332e-6

        y0 = 1 / (x1**2 * x2 * x3)

        y1 = 440.98 * x1 + 2.846 * 10**7 * x1 ** 2 + 6.1584 * 10**14 * x1**2 * x2\
            + 370.18 * x3 * 5.4474 * 10**10 * x3**2 + 3.2236 * 10**6 * x1 * x3\
            + 3.7964 * 10**11 * x2**2 + 4.2876 * 10**9 * x1 * x2 # equilibrium mole fraction balance

        trial.constr_bounds(y1, high = 1, weight = self.constr_weight, constr_name='y1')
        return y0

    def get_minima(self):
        #return 5.525e20
        return 4.978323454476186e19
    def get_minima_params(self):
        #return dict(x1 = 5.628e-5, x2 = 2.450e-7, x3 = 2.332e-6)
        return {'x1': 5.062658697773953e-05,'x2': 3.4474586099608844e-07,'x3': 2.2733197685463987e-05}


class TransformerDesign(Benchmark):
    """Problem 3. Minima reported in paper is 135.1023.
    However while I was optimizing hyperparameters for Approximate Hessian Direct Search I got a 134.9351. (Need to redo and get params)
    This one is the hardest of the problems in the paper. Many local minima, and a tiny fraction of the landscape is viable.
    """
    def __init__(self, constr_weight = 1, log_params = True, note = None):
        super().__init__(log_params=log_params, note=note)
        self.constr_weight = constr_weight

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        x1 = trial.suggest_float('x1', low = 1e-8, scale = 1e-1, fallback_high=100) # physical dimension, minima at 5.3336
        x2 = trial.suggest_float('x2', low = 1e-8, scale = 1e-1, fallback_high=100) # physical dimension, minima at 4.6585
        x3 = trial.suggest_float('x3', low = 1e-8, scale = 1e-1, fallback_high=100) # physical dimension, minima at 10.4365
        x4 = trial.suggest_float('x4', low = 1e-8, scale = 1e-1, fallback_high=100) # physical dimension, minima at 12.0840
        x5 = trial.suggest_float('x5', low = 1e-8, fallback_high=10) # magn. flux density, minima at 0.7525
        x6 = trial.suggest_float('x6', low = 1e-8, fallback_high=10) # current density, minima at 0.8781

        y0 = 0.0204 * (x1**2 * x4 + x1*x2*x4 + x1*x3*x4) + 0.0187 *\
            (x1*x2*x3 + 1.57*x2**2*x3 + x2*x3*x4) + 0.0607 *\
            (x1**2*x4*x5**2 + x1*x2*x4*x5**2) + 0.0437 *\
            (x1*x2*x3*x6**2 + 1.57*x2**2*x3*x6**2 + x2*x3*x4*x6**2)\
            + 0.0607*x1*x3*x4*x5**2 # worth of the transformer

        y1 = 2070 / (x1*x2*x3*x4*x5*x6) # rating

        y2 = 0.00062 * (x1**2*x4*x5**2 + x1*x2*x4*x5**2 + x1*x3*x4*x5**2)\
            + 0.00058 * (x1*x2*x3*x6**2 + x2*x3*x4*x6**2\
            + 1.57*x2**2*x3*x6**2) # loss constraint

        trial.constr_bounds(y1, high = 1, weight = self.constr_weight, constr_name='y1')
        trial.constr_bounds(y2, high = 1, weight = self.constr_weight, constr_name='y2')
        return y0

    def get_minima(self): return 135.1023
    def get_minima_params(self): return dict(x1 = 5.3336, x2 = 4.6585, x3 = 10.4365, x4 = 12.0840, x5 = 0.7525, x6 = 0.8781)


class _CataloguePlanning(Benchmark):
    """DOESN'T WORK. Problem 4. Minima is supposed to be 2420.284, but it just goes negative."""
    def __init__(self, constr_weight = 1, log_params = True, note = None):
        super().__init__(log_params=log_params, note=note)
        self.constr_weight = constr_weight

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        x1 = trial.suggest_float('x1', low = 1e-8, scale = 1e-3, fallback_high=10000) # distribution quantity, minima at 1000
        x2 = trial.suggest_float('x2', low = 1e-8, scale = 1e-3, fallback_high=10000) # no. of pages devoted to line 1, minima at 99.962
        x3 = trial.suggest_float('x3', low = 1e-8, scale = 1e-3, fallback_high=10000) # no.of items in line 1, minima at 4.607
        x4 = trial.suggest_float('x4', low = 1e-8, scale = 1e-3, fallback_high=10000) # no. of pages devoted to line 2, minima at 52.336
        x5 = trial.suggest_float('x5', low = 1e-8, scale = 1e-3, fallback_high=10000) # no.of items in line 2, minima at 276.299
        x6 = trial.suggest_float('x6', low = 1e-8, scale = 1e-3, fallback_high=10000) # no. of pages devoted to line 3, minima at 21.453
        x7 = trial.suggest_float('x7', low = 1e-8, scale = 1e-3, fallback_high=10000) # no.of items in line 3, minima at 202.248

        y0 = - 1.1*x1**0.51*x2**0.47*x3**0.24 - 0.9*x1**0.51*x4**0.53*x5**0.19\
            - 1.4*x1**0.51*x6**0.5*x7**0.21 # negative of demand

        y1 = (50*x2 + 120*x4 + 85*x6) / 10_000 # constraint on development and printing costs
        y2 = x1 / 1000 # constraint on distribution
        y3 = (x3 + x5 + x7) / 500 # constraint on total number of items


        trial.constr_bounds(y1, high = 1, weight = self.constr_weight, constr_name='y1')
        trial.constr_bounds(y2, high = 1, weight = self.constr_weight, constr_name='y2')
        trial.constr_bounds(y3, high = 1, weight = self.constr_weight, constr_name='y3')
        return y0

    def get_minima(self): return 2420.284
    def get_minima_params(self): return dict(x1 = 1000, x2 = 99.962, x3 = 4.607, x4 = 52.336, x5 = 276.299, x6 = 21.453, x7 = 202.248)


class _ChemicalProcessControlProblem(Benchmark):
    """DOESN'T WORK. Minima is supposed to be 0.5, but it just goes negative."""
    def __init__(self, constr_weight = 1, log_params = True, note = None):
        super().__init__(log_params=log_params, note=note)
        self.constr_weight = constr_weight

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        x1 = trial.suggest_float('x1', low = 1e-8, fallback_high=10) # , minima at 1.054
        x2 = trial.suggest_float('x2', low = 1e-8, fallback_high=10) # , minima at 0.122
        x2 = trial.param_constr_bounds('x2', high = 0.12321 - x1)

        minq2 = 1/2 * (0.1211/x2 + (1.11*10**-6)/x1*x2)

        y1 = 8.1162243 * (x1 + x2)
        trial.constr_bounds(y1, high = 1, weight = self.constr_weight, constr_name='y1')
        return minq2

    def get_minima(self): return 0.5
    def get_minima_params(self): return dict(x1 = 1.054, x2 = 0.122)