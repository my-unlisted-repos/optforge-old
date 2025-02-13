#pylint:disable = W0641
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

from ..._types import Numeric
from ..._utils import _ensure_float
from ...optim.optimizer import Config, Minimizer, Optimizer
from ...paramdict import ParamDict
from ...python_tools import reduce_dim
from ...study import EndStudy, Study

if TYPE_CHECKING:
    import pymoo, pymoo.core.algorithm
    from pymoo.core.problem import ElementwiseProblem
__all__ = [
    "PymooOptimizer",
    "get_all_pymoo_algos",
    # TODO: PymooMixedVaraibleMinimize
]

class PymooOptimizer(Optimizer):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode=False,
    )
    lib = 'pymoo'
    def __init__(
        self,
        algorithm: 'pymoo.core.algorithm.Algorithm | Callable[..., pymoo.core.algorithm.Algorithm]',
        hard_constraints: bool = True,
        seed = None,
    ):
        super().__init__(seed = seed)
        import pymoo.core.algorithm
        if not isinstance(algorithm, pymoo.core.algorithm.Algorithm): algorithm = algorithm()
        self.algorithm = algorithm

        if not hasattr(self, 'names'): self.names = [self.algorithm.__class__.__name__]

        self.problem = None
        self.hard_constraints = hard_constraints

    def _make_problem(self, study: Study):
        self.x0, self.slices = self.params.params_to_vec()
        study.evaluate()
        trial = study.evaluated_trial
        self.num_constraints = len(trial._hard_violations) if self.hard_constraints else 0
        self.num_objectives = len(trial.value) if isinstance(trial.value, np.ndarray) else 1
        self.bounds = list(zip(*self.params.get_bounds(fallback=(-np.inf, np.inf))))

        from pymoo.core.problem import ElementwiseProblem
        class StudyProblem(ElementwiseProblem):

            def __init__(pself): # type:ignore # pylint:disable=E0213
                super().__init__(
                    n_var=len(self.x0),
                    n_obj=self.num_objectives,
                    n_ieq_constr=self.num_constraints,
                    xl=self.params.get_lower_bounds(fallback=-np.inf),
                    xu=self.params.get_upper_bounds(fallback=np.inf)
                    )

            def _evaluate(pself, x, out, *args, **kwargs): # type:ignore # pylint:disable=E0213
                obj_value = study.evaluate_vec(x, self.slices)
                if isinstance(obj_value, np.ndarray): obj_value = obj_value.tolist()
                out["F"] = obj_value
                if self.num_constraints > 0: out["G"] = study.evaluated_trial.hard_violations.tolist()

        return StudyProblem()

    def _setup(self, study: Study):
        if self.problem is None:
            self.problem = self._make_problem(study)
            self.algorithm.setup(self.problem, seed = self.rng.seed)

    def step(self, study: "Study"):
        self._setup(study)
        self.algorithm.next()

def get_all_pymoo_algos():
    from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES, SimpleCMAES, BIPOPCMAES
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.algorithms.soo.nonconvex.direct import DIRECT
    from pymoo.algorithms.soo.nonconvex.es import ES
    from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
    from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
    from pymoo.algorithms.soo.nonconvex.ga import GA, BGA
    from pymoo.algorithms.soo.nonconvex.isres import ISRES
    from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
    from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
    from pymoo.algorithms.soo.nonconvex.pso_ep import EPPSO
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
    from pymoo.algorithms.soo.nonconvex.sres import SRES
    from pymoo.algorithms.moo.age import AGEMOEA
    from pymoo.algorithms.moo.age2 import AGEMOEA2
    from pymoo.algorithms.moo.ctaea import CTAEA
    from pymoo.algorithms.moo.dnsga2 import DNSGA2
    from pymoo.algorithms.moo.kgb import KGB
    from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.rnsga2 import RNSGA2
    from pymoo.algorithms.moo.rnsga3 import RNSGA3
    from pymoo.algorithms.moo.rvea import RVEA
    from pymoo.algorithms.moo.sms import SMSEMOA
    from pymoo.algorithms.moo.spea2 import SPEA2
    from pymoo.algorithms.moo.unsga3 import UNSGA3
    return list(locals().copy().values())