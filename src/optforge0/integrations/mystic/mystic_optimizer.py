#pylint:disable = W0707, W0621
from typing import Literal, TYPE_CHECKING, Optional

import numpy as np

from ..._types import Numeric
from ..._utils import _ensure_float
from ...optim.optimizer import KwargsMinimizer
from ...study import EndStudy

__all__ = [
    # "OptimagicMinimize",
    # "get_all_optimagic_algorithms",
    "MysticBuckshot",
    "MysticDE",
    "MysticStornPriceDE",
    "MysticNelderMead",
    "MysticPowell",
    "MysticLattice",
    "MysticSparsity",
]

def _get_mystic():
    import mystic
    return mystic

class MysticBuckshot(KwargsMinimizer):
    names = 'Buckshot'
    def __init__(
        self,
        nopts = 8,
        #solver = None,
        disp = False,
        ftol = 0,
        #restart = 'last',
        #gtol = 1e10,
        budget: Optional[int] = None,
    ):
        super().__init__(locals().copy())
        self.mystic = _get_mystic()


    def minimize(self, objective):
        self.res = self.mystic.solvers.buckshot(
            objective,
            ndim = self.x0.size,
            bounds=self.bounds,
            maxfun = self.budget,
            **self.kwargs,
        )

class MysticDE(KwargsMinimizer):
    names = 'Differential evolution', 'DE'

    def __init__(
        self,
        npop = 4,
        ftol = 0,
        cross = 0.9,
        scale = 0.8,
        disp = False,
        restart: Literal["best", "last", "random"] = "last",
        budget: Optional[int] = None,
    ):
        super().__init__(locals().copy())
        self.mystic = _get_mystic()


    def minimize(self, objective):
        self.res = self.mystic.solvers.diffev(
            objective,
            x0 = self.x0,
            bounds=self.bounds,
            maxfun = self.budget,
            **self.kwargs,
        )

class MysticStornPriceDE(KwargsMinimizer):
    names = 'Storn & Price’s differential evolution', 'DE2', 'SPDE'

    def __init__(
        self,
        npop = 4,
        ftol = 0,
        cross = 0.9,
        scale = 0.8,
        disp = False,
        restart: Literal["best", "last", "random"] = "last",
        budget: Optional[int] = None,

    ):
        super().__init__(locals().copy())
        self.mystic = _get_mystic()


    def minimize(self, objective):
        self.res = self.mystic.solvers.diffev2(
            objective,
            x0 = self.x0,
            bounds=self.bounds,
            maxfun = self.budget,
            **self.kwargs,
        )


class MysticNelderMead(KwargsMinimizer):
    names = 'Nelder-Mead', 'Downhill Simplex', 'NM'

    def __init__(
        self,
        ftol = 0,
        xtol = 0,
        disp = False,
        restart: Literal["best", "last", "random"] = "last",
        budget: Optional[int] = None,

    ):
        super().__init__(locals().copy())
        self.mystic = _get_mystic()

    def minimize(self, objective):
        self.res = self.mystic.solvers.fmin(
            objective,
            x0 = self.x0,
            bounds=self.bounds,
            maxfun = self.budget,
            **self.kwargs,
        )


class MysticPowell(KwargsMinimizer):
    names = 'Powell'
    def __init__(
        self,
        ftol = 0,
        xtol = 0,
        #gtol = None,
        disp = False,
        restart: Literal["best", "last", "random"] = "last",
        budget: Optional[int] = None,
    ):
        super().__init__(locals().copy())
        self.mystic = _get_mystic()

    def minimize(self, objective):
        self.res = self.mystic.solvers.fmin_powell(
            objective,
            x0 = self.x0,
            bounds=self.bounds,
            maxfun = self.budget,
            **self.kwargs,
        )

class MysticLattice(KwargsMinimizer):
    names = 'Lattice'
    def __init__(
        self,
        nbins = 8,
        xtol = 0,
        #gtol = None,
        disp = False,
        budget: Optional[int] = None,
    ):
        super().__init__(locals().copy())
        self.mystic = _get_mystic()

    def minimize(self, objective):
        self.res = self.mystic.solvers.lattice(
            objective,
            ndim = self.x0.size,
            bounds=self.bounds,
            maxfun = self.budget,
            **self.kwargs,
        )

class MysticSparsity(KwargsMinimizer):
    names = 'Sparsity'
    def __init__(
        self,
        npts = 8,
        ftol = 0,
        #gtol = None,
        disp = False,
        budget: Optional[int] = None,
    ):
        super().__init__(locals().copy())
        self.mystic = _get_mystic()

    def minimize(self, objective):
        self.res = self.mystic.solvers.sparsity(
            objective,
            ndim = self.x0.size,
            bounds=self.bounds,
            maxfun = self.budget,
            **self.kwargs,
        )