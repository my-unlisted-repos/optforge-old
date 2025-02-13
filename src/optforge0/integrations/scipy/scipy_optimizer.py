#pylint:disable = W0707, W0621
from typing import Literal, Optional
from collections.abc import Callable
import numpy as np

from ..._types import Numeric
from ..._utils import _ensure_float
from ...optim.optimizer import Config, Optimizer, Minimizer, KwargsMinimizer
from ...paramdict import ParamDict
from ...python_tools import reduce_dim
from ...study import EndStudy

__all__ = [
    "ScipyMinimize",
    "ScipyBasinhopping",
    "ScipyDE",
    "ScipySHGO",
    "ScipyDualAnnealing",
    "ScipyDIRECT",
    "ScipyBrute",
    "ScipyLeastSquares",
    "ScipyLevenbergMarquardt",
    "ScipyRoot",
    "ScipyMinimizeScalar",
    "ScipyRootScalar",
]


def _get_scipy():
    import scipy.optimize
    return scipy.optimize

class ScipyMinimize(KwargsMinimizer):
    def __init__(
        self,
        method: Optional[Literal['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                    'l-bfgs-b', 'tnc', 'cobyla', 'cobyqa', 'slsqp',
                    'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact',
                    'trust-krylov']] = None,
        jac=None,
        hess=None,
        hessp=None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
        restart: Literal["best", "last", "random"] = "last",
    ):
        super().__init__(locals().copy())
        self.scipy_optimize = _get_scipy()

        if not hasattr(self, 'names'): self.names = [f'minimize.{method}', str(method)]

    def minimize(self, objective):
        self.res = self.scipy_optimize.minimize(
            objective,
            x0 = self.x0,
            bounds=self.bounds,
            **self.kwargs,
        )

class ScipyBasinhopping(KwargsMinimizer):
    names = 'basinhopping', 'BH'
    def __init__(
        self,
        niter=100,
        T=1.0,
        stepsize=0.5,
        minimizer_kwargs=None,
        take_step=None,
        accept_test=None,
        callback=None,
        interval=50,
        disp=False,
        niter_success=None,
        seed=None,
        target_accept_rate=0.5,
        stepwise_factor=0.9,
        restart: Literal["best", "last", "random"] = "best",
    ):
        super().__init__(locals().copy())
        self.scipy_optimize = _get_scipy()

    def minimize(self, objective):
        self.res = self.scipy_optimize.basinhopping(
            objective,
            x0 = self.x0,
            **self.kwargs
        )

class ScipyDE(KwargsMinimizer):
    names =  'DE-best1bin', 'Differential evolution - best1bin', 'Differential evolution', 'DE',
    def __init__(
        self,
        strategy: Literal['best1bin', 'best1exp', 'rand1bin', 'rand1exp', 'rand2bin', 'rand2exp',
            'randtobest1bin', 'randtobest1exp', 'currenttobest1bin', 'currenttobest1exp',
            'best2exp', 'best2bin'] = "best1bin",
        maxiter=1000,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        callback=None,
        disp=False,
        polish=True,
        init="latinhypercube",
        atol=0,
        updating="immediate",
        workers=1,
        constraints=(),
        integrality=None,
        vectorized=False,
        restart: Literal['last', 'best', 'random'] = 'best',
    ):
        super().__init__(locals().copy(), fallback_bounds='default')
        self.scipy_optimize = _get_scipy()

    def minimize(self, objective):
        self.res = self.scipy_optimize.differential_evolution(
            objective,
            x0 = self.x0,
            bounds = self.bounds,
            **self.kwargs
        )


class ScipySHGO(KwargsMinimizer):
    names =  'shgo', 'simplicial homology global optimization',
    def __init__(
        self,
        constraints=None,
        n=100,
        iters=1,
        callback=None,
        minimizer_kwargs=None,
        options=None,
        sampling_method: Literal['halton', 'sobol', 'simplicial']="simplicial",
        workers = 1,
        # restart: Literal['last', 'best', 'random'] = 'last',
    ):
        super().__init__(locals().copy())
        self.scipy_optimize = _get_scipy()

    def minimize(self, objective):
        self.res = self.scipy_optimize.shgo(
            objective,
            bounds = self.bounds,
            **self.kwargs
        )

class ScipyDualAnnealing(KwargsMinimizer):
    names = 'dual_annealing', "DA"
    def __init__(
        self,
        maxiter=1000,
        minimizer_kwargs=None,
        initial_temp=5230.0,
        restart_temp_ratio=2e-05,
        visit=2.62,
        accept=-5.0,
        seed=None,
        no_local_search=False,
        callback=None,
        restart: Literal['last', 'best', 'random'] = 'best',
        budget=None,
    ):
        super().__init__(locals().copy(), fallback_bounds='default')
        self.scipy_optimize = _get_scipy()

    def minimize(self, objective):
        self.res = self.scipy_optimize.dual_annealing(
            objective,
            x0 = self.x0,
            bounds = self.bounds,
            maxfun = self.budget or 100000,
            **self.kwargs
        )


class ScipyDIRECT(KwargsMinimizer):
    """This uses some kind of threading that causes errors when terminated on out of budget exception,
    so please only use this with no stop conditions. Otherwise this will raise an error!"""
    names = 'direct'
    def __init__(
        self,
        eps: float = 0.0001,
        maxiter: int = 1000,
        locally_biased: bool = True,
        f_min: float = -np.inf,
        f_min_rtol: float = 0.0001,
        vol_tol: float = 1e-16,
        len_tol: float = 0.000001,
        callback=None,
        # restart: Literal['last', 'best', 'random'] = 'last',
        budget: int | None = None,
    ):
        super().__init__(locals().copy(), fallback_bounds='default')
        self.scipy_optimize = _get_scipy()

        # try:
        #     scipy.optimize.direct(lambda x: assert_fn, bounds=[(0, 1)]) # type: ignore
        # except (SystemError, AssertionError): pass

    def minimize(self, objective):
        self.res = self.scipy_optimize.direct(
            objective,
            bounds = self.bounds,
            maxfun = self.budget or 100000,
            **self.kwargs
        )

    def step(self, study):
        super().step(study)
        raise EndStudy


class ScipyBrute(KwargsMinimizer):
    names = ('brute',)
    def __init__(
        self,
        # ranges,
        args=(),
        Ns=20,
        full_output=0,
        disp=False,
        workers=1,
        # restart: Literal['last', 'best', 'random'] = 'last',
    ):
        super().__init__(locals().copy(), fallback_bounds='default')
        self.scipy_optimize = _get_scipy()

    def minimize(self, objective):
        self.res = self.scipy_optimize.brute(
            objective,
            ranges = self.bounds,
            **self.kwargs
        )

class ScipyLeastSquares(KwargsMinimizer):
    def __init__( # pylint:disable = W0102
        self,
        # ranges,
        method: Literal['trf', 'dogbox'] = 'trf',
        jac: Literal['2-point', '3-point', 'cs'] | Callable = "2-point",
        # bounds=(-np.inf, np.inf),
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        x_scale=1.0,
        loss="linear",
        f_scale=1.0,
        diff_step=None,
        tr_solver=None,
        tr_options={},
        jac_sparsity=None,
        max_nfev=None,
        verbose=0,
        args=(),
        kwargs={},
        restart: Literal['last', 'best', 'random'] = 'last',
    ):
        super().__init__(locals().copy(), fallback_bounds = (-np.inf, np.inf))
        self.scipy_optimize = _get_scipy()

        if not hasattr(self, 'names'): self.names = [f'least_squares.{method}', str(method)]

    def minimize(self, objective):
        bounds = np.array(self.bounds)
        self.res = self.scipy_optimize.least_squares(
            objective,
            x0 = self.x0,
            bounds = (bounds[:,0], bounds[:,1]),
            **self.kwargs
        )


class ScipyLevenbergMarquardt(KwargsMinimizer):
    names = 'Levenberg-Marquardt',
    def __init__( # pylint:disable = W0102
        self,
        # ranges,
        jac: Literal['2-point', '3-point', 'cs'] | Callable = "2-point",
        # bounds=(-np.inf, np.inf),
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        x_scale=1.0,
        loss="linear",
        f_scale=1.0,
        diff_step=None,
        tr_solver=None,
        tr_options={},
        jac_sparsity=None,
        max_nfev=None,
        verbose=0,
        args=(),
        kwargs={},
        random_weights = True,
        restart: Literal['last', 'best', 'random'] = 'last',
    ):
        kwargs = locals().copy()
        self.random_weights = kwargs.pop('random_weights')
        super().__init__(kwargs, fallback_bounds = (-np.inf, np.inf))
        self.scipy_optimize = _get_scipy()
        self.SUPPORTS_MULTI_OBJECTIVE = True

    def objective(self, x:np.ndarray):
        res = np.array([super().objective(x)] * self.x0.size)
        if self.random_weights: res *= self.weights
        return res

    def minimize(self, objective):
        self.weights = np.random.uniform(0, 2, size = self.x0.shape)
        try:
            self.res = self.scipy_optimize.least_squares(
                objective,
                x0 = self.x0,
                # bounds = self.bounds,# doesnt support bounds
                method = 'lm',
                **self.kwargs
            )
        except ValueError:
            self.params.randomize()


class ScipyRoot(KwargsMinimizer):
    def __init__(
        self,
        method: Literal[
            "hybr",
            "lm",
            "broyden1",
            "broyden2",
            "anderson",
            "linearmixing",
            "diagbroyden",
            "excitingmixing",
            "krylov",
            "df-sane",
        ] = "hybr",
        jac=None,
        tol=None,
        callback=None,
        options=None,
        args=(),
        random_weights = True,
        restart: Literal['last', 'best', 'random'] = 'best',
        # ranges,
    ):
        kwargs = locals().copy()
        self.random_weights = kwargs.pop('random_weights')
        super().__init__(kwargs)
        self.scipy_optimize = _get_scipy()
        self.SUPPORTS_MULTI_OBJECTIVE = True
        if not hasattr(self, 'names'): self.names = [f'root.{method}', str(method)]

    def objective(self, x:np.ndarray):
        res = np.array([super().objective(x)] * self.x0.size)
        if self.random_weights: res *= self.weights
        return res

    def minimize(self, objective):
        self.weights = np.random.uniform(0, 2, size = self.x0.shape)
        try:
            self.res = self.scipy_optimize.root(
                objective,
                x0 = self.x0,
                **self.kwargs
            )
        except (ValueError, OverflowError):
            self.params.randomize()


class ScipyMinimizeScalar(KwargsMinimizer):
    def __init__(
        self,
        method: Literal['Brent', "Bounded", "Golden"] = 'Brent',
        bracket = None,
        args = (),
        tol = None,
        options = None,
        restart: Literal['last', 'best', 'random'] = 'best',
        mode: Literal['coord', 'mask', 'weighted'] = 'weighted',
        # ranges,
    ):
        kwargs = locals().copy()
        self.mode = kwargs.pop('mode')
        super().__init__(kwargs, fallback_bounds = 'default')

        self.scipy_optimize = _get_scipy()
        self.method = method

        if not hasattr(self, 'names'): self.names = [f'minimize_scalar.{method}-{mode}', f'{str(method)}-{mode}']

        self.cur_coord = 0

    def objective(self, scalar: float): # type:ignore # pylint:disable = W0237
        if self.mode == 'coord':
            x = self.x0.copy()
            x[self.cur_coord] = scalar
            return super().objective(x)
        elif self.mode in ('mask', 'weighted'):
            x = self.weights.copy() * scalar
            return super().objective(x)


    def minimize(self, objective):
        if self.mode == 'mask': self.weights = np.random.choice((0, 1), size = self.x0.shape)
        elif self.mode == 'weighted': self.weights = np.random.uniform(0, 2, size = self.x0.shape)

        extra_kwargs = {'bounds': self.bounds[self.cur_coord]} if self.method == 'Bounded' else {}
        self.res = self.scipy_optimize.minimize_scalar(
            objective,
            **self.kwargs,
            **extra_kwargs,
        )

        self.cur_coord += 1
        if self.cur_coord == len(self.x0): self.cur_coord = 0


class ScipyRootScalar(KwargsMinimizer):
    def __init__(
        self,
        method: Literal['bisect', 'brentq', 'brenth', 'ridder', 'toms748', 'newton', 'secant', 'halley'] = 'bisect',
        args = (),
        bracket = None,
        fprime = None,
        fprime2 = None,
        x1 = None,
        xtol = None,
        rtol = None,
        maxiter = None,
        options = None,
        restart: Literal['last', 'best', 'random'] = 'best',
        mode: Literal['coord', 'mask', 'weighted'] = 'weighted',
        # ranges,
    ):
        kwargs = locals().copy()
        self.mode = kwargs.pop('mode')
        super().__init__(kwargs, fallback_bounds = 'default')

        self.scipy_optimize = _get_scipy()
        self.method = method

        if not hasattr(self, 'names'): self.names = [f'root_scalar.{method}-{mode}', f'{str(method)}-{mode}']

        self.cur_coord = 0

    def objective(self, scalar: float): # type:ignore # pylint:disable = W0237
        if self.mode == 'coord':
            x = self.x0.copy()
            x[self.cur_coord] = scalar
            return super().objective(x)
        elif self.mode in ('mask', 'weighted'):
            x = self.weights.copy() * scalar
            return super().objective(x)


    def minimize(self, objective):
        if self.mode == 'mask':
            self.weights = np.random.choice((0, 1), size = self.x0.shape)
            x0 = 1
        elif self.mode == 'weighted':
            self.weights = np.random.uniform(0, 2, size = self.x0.shape)
            x0 = 1
        else:
            x0 = self.x0[self.cur_coord]

        self.res = self.scipy_optimize.root_scalar(
            objective,
            x0 = x0,
            **self.kwargs,
        )

        self.cur_coord += 1
        if self.cur_coord == len(self.x0): self.cur_coord = 0

