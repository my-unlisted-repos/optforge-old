#pylint:disable = W0707, W0621

from typing import TYPE_CHECKING, Literal

import numpy as np

from ..._types import Numeric
from ...optim.optimizer import Config, Optimizer
from ...paramdict import ParamDict
from ...python_tools import reduce_dim
from ...study import EndStudy

if TYPE_CHECKING:
    import lmfit

__all__ = [
    'LMFitOptimizer',
    'ALL_METHODS'
]


ALL_METHODS = [
    'leastsq', #: Levenberg-Marquardt (default)
    'least_squares', #: Least-Squares minimization, using Trust Region Reflective method
    'differential_evolution', #: differential evolution
    'brute', #: brute force method
    'basinhopping', #: basinhopping
    'ampgo', #: Adaptive Memory Programming for Global Optimization
    'nelder', #: Nelder-Mead
    'lbfgsb', #: L-BFGS-B
    'powell', #: Powell
    'cg', #: Conjugate-Gradient
    'newton', #: Newton-CG
    'cobyla', #: Cobyla
    'bfgs', #: BFGS
    'tnc', #: Truncated Newton
    'trust-ncg', #: Newton-CG trust-region
    'trust-exact', #: nearly exact trust-region
    'trust-krylov', #: Newton GLTR trust-region
    'trust-constr', #: trust-region for constrained optimization
    'dogleg', #: Dog-leg trust-region
    'slsqp', #: Sequential Linear Squares Programming
    'emcee', #: Maximum likelihood via Monte-Carlo Markov Chain
    'shgo', #: Simplicial Homology Global Optimization
    'dual_annealing', #: Dual Annealing optimization
]



class LMFitOptimizer(Optimizer):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode=False,
    )

    def __init__(
        self,
        method: Literal[
            "leastsq",  #: Levenberg-Marquardt (default)
            "least_squares",  #: Least-Squares minimization, using Trust Region Reflective method
            "differential_evolution",  #: differential evolution
            "brute",  #: brute force method
            "basinhopping",  #: basinhopping
            "ampgo",  #: Adaptive Memory Programming for Global Optimization
            "nelder",  #: Nelder-Mead
            "lbfgsb",  #: L-BFGS-B
            "powell",  #: Powell
            "cg",  #: Conjugate-Gradient
            "newton",  #: Newton-CG
            "cobyla",  #: Cobyla
            "bfgs",  #: BFGS
            "tnc",  #: Truncated Newton
            "trust-ncg",  #: Newton-CG trust-region
            "trust-exact",  #: nearly exact trust-region
            "trust-krylov",  #: Newton GLTR trust-region
            "trust-constr",  #: trust-region for constrained optimization
            "dogleg",  #: Dog-leg trust-region
            "slsqp",  #: Sequential Linear Squares Programming
            "emcee",  #: Maximum likelihood via Monte-Carlo Markov Chain
            "shgo",  #: Simplicial Homology Global Optimization
            "dual_annealing",  #: Dual Annealing optimization
        ] = "ampgo",
        args=None,
        kws=None,
        iter_cb=None,
        scale_covar=True,
        nan_policy="raise",
        reduce_fcn=None,
        calc_covar=True,
        budget=None,
        **fit_kws,
    ):
        super().__init__()
        self.method = method
        self.kwargs = locals().copy()
        self.budget = budget
        del self.kwargs['self']
        del self.kwargs['__class__']
        del self.kwargs['budget']
        del self.kwargs['fit_kws']
        self.fit_kws = fit_kws if fit_kws is not None else {}
        self.best_value = float('inf')

        if not hasattr(self, 'names'): self.names = [self.method, ]

    def set_params(self, params:ParamDict):
        super().set_params(params)

        import lmfit
        self.lmfit_params = lmfit.Parameters()

        for name, param in params.items():
            for idx in range(param.data.size):
                low = param.low if param.low is not None else -np.inf
                high = param.high if param.high is not None else np.inf
                self.lmfit_params.add(f'{name}_{idx}', param.data.flat[idx], min=low, max=high,)

        self.save_param_data('best')
        return self

    def _closure(self, x:"lmfit.Parameters"):
        for name, param in self.params.items():
            for idx in range(param.data.size):
                param.data.flat[idx] = x[f'{name}_{idx}'].value

        self.last_value = float(self.closure())

        if self.last_value < self.best_value:
            self.best_value = self.last_value
            self.save_param_data('best') # save best params

        return self.last_value


    def step(self, study):
        self.closure = study.evaluate_return_scalar
        import lmfit
        res = lmfit.minimize(
            self._closure,
            self.lmfit_params,
            max_nfev = self.budget,
            **self.kwargs,
            **self.fit_kws,
        )
