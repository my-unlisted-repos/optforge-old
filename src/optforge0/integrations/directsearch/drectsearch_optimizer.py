#pylint:disable = W0707, W0621
from typing import Literal, TYPE_CHECKING, Any

import numpy as np

from ..._types import Numeric
from ..._utils import _ensure_float
from ...optim.optimizer import KwargsMinimizer
from ...study import EndStudy

__all__ = [
    "DSBase",
    "DSNoSketching",
    "DSProbabilistic",
    "DSSubspace",
    "DS_STP",
]

def _get_directsearch():
    import directsearch
    return directsearch

class DSBase(KwargsMinimizer):
    lib = 'directsearch'
    names = 'solve', 'Direct search', 'DS'
    def __init__(
        self,
        rho = None, # Forcing function
        sketch_dim = None, # Target dimension for sketching
        sketch_type = 'gaussian', # Sketching technique
        maxevals = None, # Maximum number of function evaluations
        poll_type = '2n', # Polling direction type
        alpha0 = None, # Original stepsize value
        alpha_max = 1e3, # Maximum value for the stepsize
        alpha_min = 1e-6, # Minimum value for the stepsize
        gamma_inc = 2.0, # Increasing factor for the stepsize
        gamma_dec = 0.5, # Decreasing factor for the stepsize
        verbose = False, # Display information about the method
        print_freq = None, # How frequently to display information
        use_stochastic_three_points = False, # Boolean for a specific method
        rho_uses_normd = True, # Forcing function based on direction norm
        restart: Literal["best", "last", "random"] = "last",
    ):
        super().__init__(locals().copy())
        self.directsearch = _get_directsearch()

    def minimize(self, objective):
        self.res = self.directsearch.solve(
            objective,
            x0 = self.x0,
            **self.kwargs
        )
        return self.res.f


class DSNoSketching(KwargsMinimizer):
    lib = 'directsearch'
    names = 'solve_directsearch', 'Direct search - no sketching', 'DSNoSketching'
    def __init__(
        self,
        rho = None, # Forcing function
        maxevals = None, # Maximum number of function evaluations
        poll_type = '2n', # Polling direction type
        alpha0 = None, # Original stepsize value
        alpha_max = 1e3, # Maximum value for the stepsize
        alpha_min = 1e-6, # Minimum value for the stepsize
        gamma_inc = 2.0, # Increasing factor for the stepsize
        gamma_dec = 0.5, # Decreasing factor for the stepsize
        verbose = False, # Display information about the method
        print_freq = None, # How frequently to display information
        rho_uses_normd = True, # Forcing function based on direction norm
        restart: Literal["best", "last", "random"] = "last",
    ):
        super().__init__(locals().copy())
        self.directsearch = _get_directsearch()

    def minimize(self, objective):
        self.res = self.directsearch.solve_directsearch(
            objective,
            x0 = self.x0,
            **self.kwargs
        )
        return self.res.f

class DSProbabilistic(KwargsMinimizer):
    lib = 'directsearch'
    names = 'solve_probabilistic_directsearch', 'Probabilistic descent', 'DSProbabilistic', 'dsproba'
    def __init__(
        self,
        rho = None, # Forcing function
        maxevals = None, # Maximum number of function evaluations
        alpha0 = None, # Original stepsize value
        alpha_max = 1e3, # Maximum value for the stepsize
        alpha_min = 1e-6, # Minimum value for the stepsize
        gamma_inc = 2.0, # Increasing factor for the stepsize
        gamma_dec = 0.5, # Decreasing factor for the stepsize
        verbose = False, # Display information about the method
        print_freq = None, # How frequently to display information
        rho_uses_normd = True, # Forcing function based on direction norm
        restart: Literal["best", "last", "random"] = "last",
    ):
        super().__init__(locals().copy())
        self.directsearch = _get_directsearch()

    def minimize(self, objective):
        self.res = self.directsearch.solve_probabilistic_directsearch(
            objective,
            x0 = self.x0,
            **self.kwargs
        )
        return self.res.f

class DSSubspace(KwargsMinimizer):
    lib = 'directsearch'
    names = 'solve_subspace_directsearch', 'Direct search - subspace', 'DSSubspace'
    def __init__(
        self,
        rho = None, # Forcing function
        sketch_dim = None, # Target dimension for sketching
        sketch_type = 'gaussian', # Sketching technique
        maxevals = None, # Maximum number of function evaluations
        poll_type = '2n', # Polling direction type
        alpha0 = None, # Original stepsize value
        alpha_max = 1e3, # Maximum value for the stepsize
        alpha_min = 1e-6, # Minimum value for the stepsize
        gamma_inc = 2.0, # Increasing factor for the stepsize
        gamma_dec = 0.5, # Decreasing factor for the stepsize
        verbose = False, # Display information about the method
        print_freq = None, # How frequently to display information
        rho_uses_normd = True, # Forcing function based on direction norm
        restart: Literal["best", "last", "random"] = "last",
    ):
        super().__init__(locals().copy())
        self.directsearch = _get_directsearch()

    def minimize(self, objective):
        self.res = self.directsearch.solve_subspace_directsearch(
            objective,
            x0 = self.x0,
            **self.kwargs
        )
        return self.res.f


class DS_STP(KwargsMinimizer):
    lib = 'directsearch'
    names = 'solve_stp', 'Direct search - STP', 'DS-STP'
    def __init__(
        self,
        maxevals = None, # Maximum number of function evaluations
        alpha0 = None, # Original stepsize value
        alpha_min = 1e-6, # Minimum value for the stepsize
        verbose = False, # Display information about the method
        print_freq = None, # How frequently to display information
        restart: Literal["best", "last", "random"] = "last",
    ):
        super().__init__(locals().copy())
        self.directsearch = _get_directsearch()

    def minimize(self, objective):
        self.res = self.directsearch.solve_stp(
            objective,
            x0 = self.x0,
            **self.kwargs
        )
        return self.res.f