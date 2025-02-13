# pylint:disable=W0707
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, Literal

import numpy as np

from .optimizer import Config, ConfiguredOptimizer, Optimizer
from ..scheduler import SchedulableFloat, SchedulableInt
from ..python_tools import reduce_dim

__all__ = [
    "Surrogate",
    "GaussianProcesses",
    "BayesianOptimization",
]
class Surrogate(Optimizer):
    CONFIG = Config(
        supports_ask = True,
        supports_multiple_asks = False,
        requires_batch_mode = True,
        store_paramdicts='all'
    )
    def __init__(
        self,
        surrogate,
        solver_cls: type[Optimizer] | ConfiguredOptimizer | Callable | str,
        bounds: Optional[tuple[float | None, float | None] | Literal['auto']],
        max_evals:Optional[SchedulableInt],
        timeout:Optional[SchedulableFloat],
        vectorized: Literal[False] | SchedulableInt | Literal['max_evals'],
        num_best: SchedulableInt,
        init_points: int,
        use_sample_y: bool | Literal['auto'] = 'auto',
        seed: Optional[int] = None,
    ):
        """Fit a cheaper surrogate model to the known objective points and minimize it to find the next points to evaluate.

        :param surrogate: The surrogate model to fit to model parameters and objective values. It must be either a scikit-learn regressor or anything that has the same API. The two methods that are called are `fit(X, y)` and `predict(X)`. `X` is a 2D array of points of shape `(n, d)`, where `n` is the number of points and `d` is the coordinates. `y` is a 1D array of objective values with the same length as `n`.
        :param solver_cls: Constructor for the solver that will minimize the surrogate model on each step. Either type of optimizer or a string from the registry. Consider using something that supports multiple asks, since it enables vectorized minimization of the surrogate model which is much faster.
        :param max_evals: Maximum evals for minimizing the surrogate model on each step.
        :param timeout: Timeout for minimizing the surrogate model on each step.
        :param vectorized: Whether surrogate mimization should be vectorized. True is significantly more efficient, but requires an optimizer that supports multiple asks.
        :param num_asks: Number of points to evaluate the surrogate model at in one shot if `vectorized` is True. For max performance best to be the same as max evals, but that only really works for one shot optimizers like random search.
        :param num_best: Number of best points found by minimizing the surrogate model to evaluate the objective at on each step.
        :param init_points: Samples this many initial random points to fit the initial surrogate model to.
        :param use_samply_y: Whether to use `surrogate.sample_y(X)` method instead of `surrogate.predict_y(X)`. If `auto`, sets to True if surrogate has a `sample_y` method.
        :param bounds: Bounds for minimizing the surrogate model. If parameters have bounds, this should be set to the same bounds as the parameters. Defaults to None.

        Surrogate model must be non-linear, otherwise it will just predict the edges of the parameter space. Generally gaussian processes are often used as a surrogate model, for example `sklearn.gaussian_process.GaussianProcessRegressor`. But this accepts any regressor that has the same API. Maybe AutoML would work even better?
        """
        super().__init__(seed=seed)
        self.surrogate = surrogate

        if use_sample_y == 'auto': use_sample_y = hasattr(surrogate, 'sample_y')
        self.use_sample_y: bool = use_sample_y

        if isinstance(solver_cls, str):
            from ..registry.optimizers import OPTIMIZERS
            solver_cls = OPTIMIZERS[solver_cls]

        self.solver_constructor = solver_cls
        self.init_points = init_points
        self.bounds = bounds
        self.max_evals = self.schedule(max_evals)
        self.vectorized = self.schedule(vectorized)
        self.timeout = self.schedule(timeout)
        self.num_best = self.schedule(num_best)

        self.first_step = True

    def ask(self, study):
        # on first evaluation, get values at some initial points, unless init_points is 0
        if self.init_points < 1: self.first_step = False
        if self.first_step:
            self.first_step = False
            for i in range(self.init_points):
                new_params = self.params.copy()
                for p in new_params.values():
                    p.data = p.sample_random()
                yield new_params

        # otherwise, use the surrogate model to suggest the next point
        else:
            # get vector representation of all points evaluated so far
            points_slices = [trial.paramdict.params_to_vec() for trial in study.trials] # type:ignore
            points = [i[0] for i in points_slices]
            slices = points_slices[0][1]
            values = [trial.objective_value.opt_scalar_value for trial in study.trials] # type:ignore

            # fit the surrogate model to the points and values
            self.surrogate.fit(np.array(points, copy=False), np.array(values, copy=False))

            if self.use_sample_y: predict_fn = self.surrogate.sample_y
            else: predict_fn = self.surrogate.predict

            # determine bounds (roughly)
            if self.bounds == 'auto':
                high_bounds = [i.high for i in self.params.values()]
                if not any(i is None for i in high_bounds): high = float(np.max(high_bounds)) # type:ignore
                else: high = None

                low_bounds = [i.low for i in self.params.values()]
                if not any(i is None for i in low_bounds): low = float(np.min(low_bounds)) # type:ignore
                else: low = None

                self.bounds = (low, high)

            vectorized = self.vectorized()
            if vectorized == 'max_evals': vectorized = self.max_evals()

            # minimize the surrogate model to find the next points
            # import there to avoid circular import
            from ..interfaces.minimize import minimize
            study = minimize(
                # initial point is `points[0], which has (d, ) shape, so it needs to be reshaped to (1, d)`
                # then, if evaluation is vectorized, points will have (n, d) shape, otherwise also (d, )
                lambda x: predict_fn(x if x.ndim == 2 else x.reshape(1, -1)),
                x0=points[0],
                optimizer=self.solver_constructor(),
                max_evals=self.max_evals(),
                timeout=self.timeout(),
                bounds=self.bounds, # type:ignore
                vectorized=vectorized,
                log_params = True, # needs to be true because params are the prediction
                progress = False,
                catch_kb_interrupt=False,
                seed = self.rng.seed,
            )

            # sort trials to get the best points
            num_best = self.num_best()
            if num_best == 1: sorted_trials = [study.best_trial]
            else:
                sorted_trials = sorted(
                    study.trials,
                    key=lambda t: t.scalar_value,
                    )

            # yield the best points to evaluate at and refit the surrogate on next step
            for i, t in zip(range(num_best), sorted_trials):
                new_params = self.params.copy()
                new_params.vec_to_params_(np.array(reduce_dim([i.flat for i in t.params.values()])), slices=slices) # type:ignore
                yield new_params

    def plot_decision_boundary(self, custom_bounds: Optional[tuple[float,float]] = None, num_points = 10000, eps = 0., class_of_interest = None):
        bounds = custom_bounds if custom_bounds is not None else self.bounds
        from sklearn.inspection import DecisionBoundaryDisplay
        return DecisionBoundaryDisplay.from_estimator(
            self.surrogate, np.random.uniform(*bounds, (num_points, 2)),  # type:ignore
            response_method="auto",
            alpha=1,
            plot_method='pcolormesh', eps=eps,
            class_of_interest = class_of_interest,
        )


class GaussianProcesses(Surrogate):
    names = 'Gaussian Processes', 'GP'

    def __init__(
        self,
        solver_cls: type[Optimizer] | ConfiguredOptimizer | Callable | str = 'AcceleratedRandomSearch',
        max_evals:Optional[SchedulableInt] = 1000,
        timeout:Optional[SchedulableFloat] = None,
        vectorized: Literal[False] | SchedulableInt | Literal['max_evals'] = 'max_evals',
        num_best: SchedulableInt = 1,
        init_points: int = 0,
        bounds: Optional[tuple[float | None, float | None] | Literal['auto']] = 'auto',
        use_sample_y: bool | Literal['auto'] = False,
        seed: Optional[int] = None,
    ):
        from sklearn.gaussian_process import GaussianProcessRegressor
        super().__init__(
            surrogate=GaussianProcessRegressor(random_state=seed),
            solver_cls=solver_cls,
            bounds=bounds,
            max_evals=max_evals,
            timeout=timeout,
            vectorized=vectorized,
            num_best=num_best,
            init_points=init_points,
            use_sample_y=use_sample_y,
            seed = seed,
        )
GaussianProcesses.register()
BayesianOptimization = GaussianProcesses.configured(max_evals=2000, use_sample_y=True).set_name(('Bayesian optimization', 'Bayesian', 'BO'), register=True)
ImpreciseGP = GaussianProcesses.configured(
    max_evals = 1000, solver_cls='RandomSearch').set_name('Imprecise GP', register=True)
VeryImpreciseGP = GaussianProcesses.configured(
    max_evals = 100, solver_cls='RandomSearch').set_name('Very imprecise GP', register=True)
Top10GP = GaussianProcesses.configured(
    max_evals = 1000, solver_cls='RandomSearch', num_best=10).set_name('Top-10 GP', register=True)
Top100GP = GaussianProcesses.configured(
    max_evals = 1000, solver_cls='RandomSearch', num_best=100).set_name('Top-100 GP', register=True)


class RandomForestSurrogate(Surrogate):
    names = 'Random Forest Surrogate', 'RFS'

    def __init__(
        self,
        solver_cls: type[Optimizer] | ConfiguredOptimizer | Callable | str = 'AcceleratedRandomSearch',
        max_evals:Optional[SchedulableInt] = 1000,
        timeout:Optional[SchedulableFloat] = None,
        vectorized: Literal[False] | SchedulableInt | Literal['max_evals'] = 'max_evals',
        num_best: SchedulableInt = 1,
        init_points: int = 10,
        bounds: Optional[tuple[float | None, float | None] | Literal['auto']] = 'auto',
        #use_sample_y: bool | Literal['auto'] = 'auto',
        seed: Optional[int] = None,
    ):
        from sklearn.ensemble import RandomForestRegressor
        super().__init__(
            surrogate=RandomForestRegressor(random_state=seed),
            solver_cls=solver_cls,
            bounds=bounds,
            max_evals=max_evals,
            timeout=timeout,
            vectorized=vectorized,
            num_best=num_best,
            init_points=init_points,
            use_sample_y=False,
            seed = seed,
        )
RandomForestSurrogate.register()
ImpreciseRFS = RandomForestSurrogate.configured(
    max_evals = 1000, solver_cls='RandomSearch').set_name('Imprecise RFS', register=True)
VeryImpreciseRFS = RandomForestSurrogate.configured(
    max_evals = 100, solver_cls='RandomSearch').set_name('Very imprecise RFS', register=True)
Top10RFS = RandomForestSurrogate.configured(
    max_evals = 1000, solver_cls='RandomSearch', num_best=10).set_name('Top-10 RFS', register=True)
Top100RFS = RandomForestSurrogate.configured(
    max_evals = 1000, solver_cls='RandomSearch', num_best=100).set_name('Top-100 RFS', register=True)


class GradientBoostingSurrogate(Surrogate):
    names = 'Gradient Boosting Surrogate', 'GBS'

    def __init__(
        self,
        solver_cls: type[Optimizer] | ConfiguredOptimizer | Callable | str = 'AcceleratedRandomSearch',
        max_evals:Optional[SchedulableInt] = 1000,
        timeout:Optional[SchedulableFloat] = None,
        vectorized: Literal[False] | SchedulableInt | Literal['max_evals'] = 'max_evals',
        num_best: SchedulableInt = 1,
        init_points: int = 10,
        bounds: Optional[tuple[float | None, float | None] | Literal['auto']] = 'auto',
        #use_sample_y: bool | Literal['auto'] = True,
        seed: Optional[int] = None,
    ):
        try: from sklearn.ensemble import GradientBoostingRegressor
        except ModuleNotFoundError: raise ModuleNotFoundError("GradientBoostingOptimizer requires scikit-learn")
        super().__init__(
            surrogate=GradientBoostingRegressor(random_state=seed),
            solver_cls=solver_cls,
            bounds=bounds,
            max_evals=max_evals,
            timeout=timeout,
            vectorized=vectorized,
            num_best=num_best,
            init_points=init_points,
            use_sample_y=False,
            seed = seed,
        )
GradientBoostingSurrogate.register()
ImpreciseGBS = GradientBoostingSurrogate.configured(
    max_evals = 1000, solver_cls='RandomSearch').set_name('Imprecise GBS', register=True)
VeryImpreciseGBS = GradientBoostingSurrogate.configured(
    max_evals = 100, solver_cls='RandomSearch').set_name('Very imprecise GBS', register=True)
Top10GBS = GradientBoostingSurrogate.configured(
    max_evals = 1000, solver_cls='RandomSearch', num_best=10).set_name('Top-10 GBS', register=True)
Top100GBS = GradientBoostingSurrogate.configured(
    max_evals = 1000, solver_cls='RandomSearch', num_best=100).set_name('Top-100 GBS', register=True)



class ExtraTreesSurrogate(Surrogate):
    names = 'Extra Trees Surrogate', 'ETS'

    def __init__(
        self,
        solver_cls: type[Optimizer] | ConfiguredOptimizer | Callable | str = 'AcceleratedRandomSearch',
        max_evals:Optional[SchedulableInt] = 1000,
        timeout:Optional[SchedulableFloat] = None,
        vectorized: Literal[False] | SchedulableInt | Literal['max_evals'] = 'max_evals',
        num_best: SchedulableInt = 1,
        init_points: int = 10,
        bounds: Optional[tuple[float | None, float | None] | Literal['auto']] = 'auto',
        #use_sample_y: bool | Literal['auto'] = True,
        seed: Optional[int] = None,
    ):
        try: from sklearn.ensemble import RandomForestRegressor
        except ModuleNotFoundError: raise ModuleNotFoundError("ExtraTreesOptimizer requires scikit-learn")
        super().__init__(
            surrogate=RandomForestRegressor(random_state=seed),
            solver_cls=solver_cls,
            bounds=bounds,
            max_evals=max_evals,
            timeout=timeout,
            vectorized=vectorized,
            num_best=num_best,
            init_points=init_points,
            use_sample_y=False,
            seed = seed,
        )
ExtraTreesSurrogate.register()
ImpreciseETS = ExtraTreesSurrogate.configured(
    max_evals = 1000, solver_cls='RandomSearch').set_name('Imprecise ETS', register=True)
VeryImpreciseETS = ExtraTreesSurrogate.configured(
    max_evals = 100, solver_cls='RandomSearch').set_name('Very imprecise ETS', register=True)
Top10ETS = ExtraTreesSurrogate.configured(
    max_evals = 1000, solver_cls='RandomSearch', num_best=10).set_name('Top-10 ETS', register=True)
Top100ETS = ExtraTreesSurrogate.configured(
    max_evals = 1000, solver_cls='RandomSearch', num_best=100).set_name('Top-100 ETS', register=True)


class MLPSurrogate(Surrogate):
    names = 'MLP Surrogate', 'MLPS'

    def __init__(
        self,
        solver_cls: type[Optimizer] | ConfiguredOptimizer | Callable | str = 'AcceleratedRandomSearch',
        max_evals:Optional[SchedulableInt] = 1000,
        timeout:Optional[SchedulableFloat] = None,
        vectorized: Literal[False] | SchedulableInt | Literal['max_evals'] = 'max_evals',
        num_best: SchedulableInt = 1,
        init_points: int = 0,
        bounds: Optional[tuple[float | None, float | None] | Literal['auto']] = 'auto',
        #use_sample_y: bool | Literal['auto'] = True,
        seed: Optional[int] = None,
    ):
        try: from sklearn.neural_network import MLPRegressor
        except ModuleNotFoundError: raise ModuleNotFoundError("ExtraTreesOptimizer requires scikit-learn")
        super().__init__(
            surrogate=MLPRegressor((100, 100, 100), random_state=seed),
            solver_cls=solver_cls,
            bounds=bounds,
            max_evals=max_evals,
            timeout=timeout,
            vectorized=vectorized,
            num_best=num_best,
            init_points=init_points,
            use_sample_y=False,
            seed = seed,
        )
MLPSurrogate.register()
ImpreciseMLPS = MLPSurrogate.configured(
    max_evals = 1000, solver_cls='RandomSearch').set_name('Imprecise MLPS', register=True)
VeryImpreciseMLPS = MLPSurrogate.configured(
    max_evals = 100, solver_cls='RandomSearch').set_name('Very imprecise MLPS', register=True)
Top10MLPS = MLPSurrogate.configured(
    max_evals = 1000, solver_cls='RandomSearch', num_best=10).set_name('Top-10 MLPS', register=True)
Top100MLPS = MLPSurrogate.configured(
    max_evals = 1000, solver_cls='RandomSearch', num_best=100).set_name('Top-100 MLPS', register=True)
