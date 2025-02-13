#pylint:disable = W0707
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Optional, Any
import warnings
import numpy as np

from ..._utils import _ensure_float, _ensure_float_or_ndarray
from ...optim.optimizer import Config, Optimizer
from ...trial import Trial
from ...param import DomainLiteral, ParamTypes

if TYPE_CHECKING:
    import mealpy

__all__ = [
    "MealpyOptimizer"
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


class MealpyOptimizer(Optimizer):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode=True,
        
    )
    wrapped_optimizer: "mealpy.Optimizer"
    def __init__(
        self,
        opt_cls: "type[mealpy.Optimizer] | Callable[..., mealpy.Optimizer] | mealpy.Optimizer",
        pop_size: Optional[int] = None,
        opt_kwargs: Optional[dict[str, Any]] = None,
        param_mode: Literal["mixed", "numeric"] = "mixed",
        budget: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        import mealpy

        self.param_mode: Literal["mixed", "numeric"] = param_mode

        self.budget = budget
        self.pop_size = pop_size
        if isinstance(opt_cls, mealpy.Optimizer): self.wrapped_optimizer = opt_cls
        else:
            self.wrapped_constructor: "type[mealpy.Optimizer] | Callable" = opt_cls # type:ignore
            self.wrapped_kwargs = opt_kwargs

        if not hasattr(self, 'names'):
            cls = self.wrapped_optimizer.__class__ if self.wrapped_optimizer is not None else self.wrapped_constructor
            self.names = _mealpy_names(cls)

    def set_params(self, params):
        super().set_params(params)

        import mealpy, mealpy.utils.space
        bounds: list[mealpy.utils.space.BaseVar] = []

        for name, param in params.yield_names_params():

            low = param.low; high = param.high
            if low is None: low = param.fallback_low
            if high is None: high = param.fallback_high

            shape = param.data.shape
            size = param.data.size
            if shape != 1 and isinstance(shape, int): shape = (shape,)

            # ----------------------------- OPTFORGE SCALING ----------------------------- #
            if self.param_mode == 'numeric':
                bounds.append(mealpy.FloatVar(lb = (low,) * size, ub = (high, ) * size, name = name)) # type:ignore

            # ------------------------------ MEALPY SCALING ------------------------------ #
            else:
                # numeric
                if param.TYPE in (ParamTypes.CONTINUOUS, ParamTypes.DISCRETE, ParamTypes.ONEHOT, ParamTypes.ARGSORT):
                    # integer
                    if param.TYPE == ParamTypes.DISCRETE and param.step_original == 1:
                        bounds.append(mealpy.IntegerVar(
                            lb = (low,) * size, ub = (high, ) * size, name = name) # type:ignore
                        )
                    # float
                    else:
                        bounds.append(mealpy.FloatVar(
                            lb = (low,) * size, ub = (high, ) * size, name = name) # type:ignore
                        )

                # choice
                elif param.TYPE == ParamTypes.TRANSIT:
                    bounds.append(mealpy.MixedSetVar((param.sampler.choices_numeric.tolist(),) * size, name=name ))
                elif param.TYPE == ParamTypes.UNORDERED:
                    bounds.append(mealpy.MixedSetVar((np.linspace(low, high, 1000).tolist(), ) * size, name = name))

                else: raise ValueError(f'Unsupported param.TYPE = {param.TYPE}')

        self.bounds: list[mealpy.utils.space.BaseVar] = bounds
        return self

    def _closure(self, x:dict[str, Any] | np.ndarray):
        if isinstance(x, np.ndarray): x = self.problem.decode_solution(x)
        for name, p in x.items():
            self.params[name].data = np.array(p, copy=False).reshape(self.params[name].data.shape)
        return self.closure()


    def step(self, study):
        import mealpy
        self.closure = study.__call__ # supports multi-objective

        class Problem(mealpy.Problem):
            def __init__(prob_self,): # pylint:disable = E0213, W0237 # type:ignore
                super().__init__(bounds = self.bounds, log_to = None, )

            def obj_func(prob_self, x): # pylint:disable = E0213, W0237 # type:ignore
                x_decoded = prob_self.decode_solution(x)
                return self._closure(x_decoded)

        self.problem = Problem()

        if self.wrapped_optimizer is None:

            if self.budget is not None:
                if self.pop_size is not None: kwargs = {
                    'epochs': max(1, int(self.budget / self.pop_size)),
                    "pop_size": self.pop_size,
                    }
                else:
                    # pop_size = 100 appears to be the default for all optimizers (but I might have missed some)
                    kwargs = {'epochs': max(1, int(self.budget / 100)), }

            else:
                warnings.warn(f'No budget specified, {self.__class__.__name__} might function incorrectly.')
                kwargs = {}

            if self.wrapped_kwargs is None: self.wrapped_kwargs = {}
            self.wrapped_optimizer = self.wrapped_constructor(**kwargs, **self.wrapped_kwargs)

        self.g_best = self.wrapped_optimizer.solve(self.problem, seed = self.rng.seed) # type:ignore

