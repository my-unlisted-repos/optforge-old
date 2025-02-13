#pylint:disable = W0707
from typing import TYPE_CHECKING, Optional

import numpy as np

from ...optim.optimizer import Config, Optimizer
from ...paramdict import ParamDict
from ...python_tools import reduce_dim

if TYPE_CHECKING:
    import pypop7.optimizers.core.optimizer

    from ..._types import Numeric

__all__ = [
    "PyPop7Optimizer",
    "get_all_pypop7_optimizers",
]


def _pypop7_names(c) -> list[str]:
    # we get full name like `rs.GS':
    namespace = str(c).split("'")[1].split('.')
    return [f'{namespace[-3]}.{namespace[-1]}', namespace[-1]]

class PyPop7Optimizer(Optimizer):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode=False,
        # I couldn't find if it supports multi-objective anywhere, but maybe it does, need to look at source code

    )
    wrapped_optimizer: "pypop7.optimizers.core.optimizer.Optimizer"
    def __init__(
        self,
        optimizer_cls: "type[pypop7.optimizers.core.optimizer.Optimizer]",
        sigma:Optional[float] = 0.5,
        temperature: Optional[float] = 1.5,
        verbose:Optional[int | bool] = False, # to make it not print any stuff
        options: Optional[dict] = None,
        use_initial_boundary = False,
        init_width = 1e-3,
        seed: Optional[int] = None,
    ):
        """Wrapper for a PyPop7 optimizer. Note that this will do unlimited amount of evaluations per step, so use `max_evals` or `timeout` in your study to make sure it stops at some point, don't use `max_steps`.

        :param params: Parameters to optimize.
        :param optimizer_cls: Any optimizer class from pypop7
        :param sigma: Mutation rate for optimizers that have it (causes errors when not specified). defaults to 0.5.
        :param temperature: Temperature for optimizers that have it (causes errors when not specified). defaults to 1.5
        :param verbose: Verbose parameter, if integer, pypop7 will print progress report every this many evaluations, defaults to False
        :param options: Additional options to pass to pypop7 optimizer, this takes priority over args above like sigma. defaults to None
        :param use_initial_boundary: If True, passes `initial_lower_boundary` and `initial_upper_boundary` to be close to current parameter values (which would be `init` if you specified it for your params). This basically causes initial population to use your init, instead of being generated randomly within entire search space. Defaults to False
        :param init_width: Width of initial boundary around init, defaults to 1e-3
        """
        try: import pypop7
        except ModuleNotFoundError: raise ModuleNotFoundError("PyPop7 is not installed")

        super().__init__(seed = seed)
        self.optimizer_cls = optimizer_cls
        if options is None: options = {}
        self.options = options
        if sigma is not None and 'sigma' not in self.options: self.options['sigma'] = sigma
        if verbose is not None and 'verbose' not in self.options: self.options['verbose'] = verbose
        if seed is not None and 'seed_rng' not in self.options: self.options['seed_rng'] = seed
        if temperature is not None and 'temperature' not in self.options: self.options['temperature'] = temperature
        self.use_initial_boundary = use_initial_boundary
        self.init_width = init_width
        self.last_value = None

        if not hasattr(self, 'names'): self.names = _pypop7_names(optimizer_cls)

    def set_seed(self, seed: Optional[int]):
        self.options['seed_rng'] = seed
        return super().set_seed(seed)

    def set_params(self, params:ParamDict):
        super().set_params(params)
        lower_bound = []
        upper_bound = []
        init = []
        for param in self.params.values():
            #if param.low is None or param.high is None: raise ValueError("PyOp7 requires low and high bounds for all parameters")
            if param.low is None: low = param.fallback_low
            else: low = param.low
            if param.high is None: high = param.fallback_high
            else: high = param.high

            size = param.data.size
            lower_bound.extend([low] * size)
            upper_bound.extend([high] * size)
            if self.use_initial_boundary: init.append(param.data.ravel())

        if self.use_initial_boundary: init = reduce_dim(init)

        problem = {
            'fitness_function': self._closure,  # cost function to be minimized
            'ndim_problem': len(lower_bound),  # dimension of cost function
            'lower_boundary': np.asanyarray(lower_bound, dtype=np.float64),  # lower search boundary
            'upper_boundary': np.asanyarray(upper_bound, dtype=np.float64),
            }
        if self.use_initial_boundary:
            problem['initial_lower_boundary'] = np.asanyarray(init, dtype=np.float64) - self.init_width
            problem['initial_upper_boundary'] = np.asanyarray(init, dtype=np.float64) + self.init_width

        self.wrapped_optimizer = self.optimizer_cls(problem=problem, options=self.options)
        return self


    def _closure(self, variables: np.ndarray) -> "Numeric":
        cur = 0
        for param in self.params.values():
            size = param.data.size
            param.data = variables[cur:cur+size].reshape(param.data.shape)
            cur += size
        self.last_value = self.closure()
        return self.last_value

    def step(self, study):
        self.closure = study.evaluate_return_scalar
        self.res = self.wrapped_optimizer.optimize()
        if self.last_value is None: raise ValueError(f'{self.wrapped_optimizer.__class__.__name__} is an abstract optimizer!')


def get_all_pypop7_optimizers() -> "list[type[pypop7.optimizers.core.optimizer.Optimizer]]":
    try: import pypop7
    except ModuleNotFoundError: raise ModuleNotFoundError("PyPop7 is not installed")
    import pypop7.optimizers.core.optimizer
    from pypop7.optimizers import (bo, cc, cem, core, de, ds, eda, ep, es, ga,
                                   nes, pso, rs, sa)

    from ...python_tools import subclasses_recursive
    return list(sorted(list(subclasses_recursive(pypop7.optimizers.core.optimizer.Optimizer)), key=lambda x: x.__name__))