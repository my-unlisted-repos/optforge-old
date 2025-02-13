from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Literal, cast, Any

import numpy as np

from ..._types import Numeric
from ...optim.optimizer import Config, Optimizer
from ...python_tools import reduce_dim, get__name__
from ...study import Study, EndStudy

if TYPE_CHECKING:
    import pygmo, pygmo.core

__all__ = [
    'PygmoOptimizer',
    'get_all_pygmo_algos',
]




class PygmoOptimizer(Optimizer):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode=True,
    )
    lib = 'pygmo'
    def __init__(
        self,
        opt_cls: Callable,
        popsize = 20,
        restart: Literal['last', 'best', 'random'] = 'last',
        hard_constraints: bool = True,
    ):
        super().__init__()
        self.opt = opt_cls()
        self.popsize = popsize
        self.restart = restart

        self.slices = cast(dict[str, slice], None)
        self.prob = cast(Any, None)
        self.pop = cast(Any, None)
        self.algo = cast(Any, None)

        self.hard_constraints = hard_constraints

        self.nic = cast(int, None)
        """Number of inequality constraints"""
        self.nobj = cast(int, None)
        """Number of objectives"""
        # self.nix = cast(int, None)
        # """Number of integer variables"""

        if not hasattr(self, 'names'): self.names = [get__name__(self.opt)]

    def _raise_end_study(self):
        raise EndStudy()
    def _make_problem(self, study: "Study"):
        import pygmo as pg, scipy.optimize
        if self.nobj is None:
            study.evaluate()
            trial = study.evaluated_trial
            self.nic = len(trial._hard_violations) if self.hard_constraints else 0
            self.nobj = len(trial.value) if isinstance(trial.value, np.ndarray) else 1
            self.bounds = list(zip(*self.params.get_bounds(fallback=(-np.inf, np.inf))))
            self.slices = self.params.params_to_vec()[1]

        class Problem:

            def fitness(pself, x: np.ndarray): # type:ignore # pylint:disable=E0213
                try:
                    # get objective value
                    obj = study.evaluate_vec(x, self.slices)
                    if self.nic > 0:
                        # get value without hard violations applied
                        obj = study.evaluated_trial.get_value(soft_penalty=True, hard_penalty=False)
                        if isinstance(obj, (int, float)): obj = [obj]
                        else: obj = obj.tolist()
                        # add hard violations to the objective
                        obj.extend(study.evaluated_trial.hard_violations)
                    else:
                        # make sure it is a list
                        if isinstance(obj, (int, float)): obj = [obj]
                        else: obj = obj.tolist()
                    return obj

                # pygmo problem definition with a Study is a little weird, it seems to deepcopy everything
                # and catches EndStudy, so we reraise it on the level of optimizer
                # maybe this is an issue with creating classes at runtime.
                except EndStudy:
                    self._raise_end_study()
                    return []

            def _fitness0(self, x:np.ndarray):
                return self.fitness(x)[0]

            def gradient(pself, x):# type:ignore # pylint:disable=E0213
                try: return scipy.optimize.approx_fprime(x, pself._fitness0, )
                except EndStudy: self._raise_end_study()
            def get_bounds(pself): # type:ignore # pylint:disable=E0213
                return self.bounds

            def get_nobj(pself): # type:ignore # pylint:disable=E0213
                return self.nobj

            def get_nic(pself): # type:ignore # pylint:disable=E0213
                return self.nic

            # def get_nix(pself): # type:ignore # pylint:disable=E0213
            #     return self.nix

        self.prob = Problem()


    def step(self, study: "Study"):
        if self.prob is None:
            import pygmo as pg
            self._make_problem(study)
            self.pop = pg.population(self.prob, size = self.popsize) # type:ignore # pylint:disable=E1101
            self.algo = pg.algorithm(self.opt) # type:ignore # pylint:disable=E1101
        self.pop = self.algo.evolve(self.pop)


def get_all_pygmo_algos():
    import pygmo as pg
    return [
        pg.gaco, # type:ignore # pylint:disable=E1101
        pg.maco, # type:ignore # pylint:disable=E1101
        pg.gwo, # type:ignore # pylint:disable=E1101
        pg.bee_colony, # type:ignore # pylint:disable=E1101
        pg.de, # type:ignore # pylint:disable=E1101
        pg.sea, # type:ignore # pylint:disable=E1101
        pg.sga, # type:ignore # pylint:disable=E1101
        pg.sade, # type:ignore # pylint:disable=E1101
        pg.de1220, # type:ignore # pylint:disable=E1101
        pg.cmaes, # type:ignore # pylint:disable=E1101
        pg.moead, # type:ignore # pylint:disable=E1101
        pg.moead_gen, # type:ignore # pylint:disable=E1101
        pg.compass_search, # type:ignore # pylint:disable=E1101
        pg.simulated_annealing, # type:ignore # pylint:disable=E1101
        pg.pso, # type:ignore # pylint:disable=E1101
        pg.pso_gen, # type:ignore # pylint:disable=E1101
        pg.nsga2, # type:ignore # pylint:disable=E1101
        pg.nspso, # type:ignore # pylint:disable=E1101
        pg.mbh, # type:ignore # pylint:disable=E1101
        pg.cstrs_self_adaptive, # type:ignore # pylint:disable=E1101
        pg.ipopt, # type:ignore # pylint:disable=E1101
        pg.ihs, # type:ignore # pylint:disable=E1101
        pg.xnes, # type:ignore # pylint:disable=E1101
            ]