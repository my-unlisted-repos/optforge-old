from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from ...optim.optimizer import Config, Optimizer
from ...param import ParamTypes

if TYPE_CHECKING:
    import zoopt

__all__ = [
    "ZOOptimizer"
]

def _mealpy_names(cls) -> list[str]:
    # we get full name like `rs.GS':
    if not isinstance(cls, type): cls = cls.__class__
    namespace = str(cls).split("'")[1].split('.')
    return [
        f"{namespace[-3]}.{namespace[-2]}.{namespace[-1]}",
        f"{namespace[-3]}.{namespace[-1]}",
        f"{namespace[-2]}.{namespace[-1]}",
        namespace[-1],
    ]


class ZOOptimizer(Optimizer):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode=True,

    )
    names = 'ZOOpt'
    wrapped_optimizer: "zoopt.Parameter"
    def __init__(
        self,
        algorithm: Literal['racos', 'poss'] | None = None,
        budget: Optional[int] = None,
        exploration_rate: float = 0.01,
        sequential: bool = True,
        precision = None,
        uncertain_bits = None,
        intermediate_result: bool = False,
        intermediate_freq: int = 100,
        autoset: bool = True,
        noise_handling: bool = False,
        resampling: bool = False,
        suppression: bool = False,
        ponss: bool = False,
        ponss_theta = None,
        ponss_b = None,
        non_update_allowed: int = 500,
        resample_times: int = 100,
        balance_rate: float = 0.5,
        high_dim_handling: bool = False,
        reducedim: bool = False,
        num_sre: int = 5,
        low_dimension = None,
        variance_A = None,
        parallel: bool = False,
        server_num: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.algorithm = algorithm
        self.budget = budget

        self.kwargs = locals().copy()
        del self.kwargs['self']
        del self.kwargs['__class__']
        del self.kwargs['algorithm']
        del self.kwargs['budget']
        del self.kwargs['seed']


    def set_params(self, params):
        super().set_params(params)

        import zoopt
        regs = self.params.get_bounds(fallback='default')
        tys = []
        order = []
        for p in self.params.values():
            if (p.TYPE == ParamTypes.DISCRETE or p.TYPE == ParamTypes.TRANSIT) and p.step == 1:
                tys.extend([False] * p.data.size)
                if p.TYPE == ParamTypes.TRANSIT: order.extend([False] * p.data.size)
                else: order.extend([True] * p.data.size)
            else:
                tys.extend([True] * p.data.size)
                if p.TYPE == ParamTypes.TRANSIT: order.extend([False] * p.data.size)
                else: order.extend([True] * p.data.size)

        self.dim = zoopt.Dimension(self.params.numel(), regs = regs, tys = tys, order = order)
        vec, self.slices = self.params.params_to_vec()
        self.sol = zoopt.Solution(vec)
        return self

    def _closure(self, solution: "zoopt.Solution"):
        x = solution.get_x()
        self.last_value = self.objective(np.array(x, copy=False), slices = self.slices)
        return self.last_value

    def step(self, study):
        import zoopt
        self.objective = partial(study.evaluate_vec, return_scalar = True)
        obj = zoopt.Objective(self._closure, dim = self.dim)
        self.parameter = zoopt.Parameter(
            algorithm=self.algorithm,
            budget = (self.budget or 0), init_samples=[self.sol],
            seed = self.rng.seed,
            **self.kwargs,
            )
        self.solution = zoopt.Opt.min(obj, self.parameter)
