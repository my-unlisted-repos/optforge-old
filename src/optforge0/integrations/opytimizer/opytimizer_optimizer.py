#pylint:disable = W0707
from functools import partialmethod, partial
from typing import TYPE_CHECKING, Literal, Optional
import warnings
# disable logging
try:
    import opytimizer.opytimizer
    import opytimizer.utils.logging
    opytimizer.utils.logging.LOG_LEVEL = opytimizer.utils.logging.logging.WARNING
except (ModuleNotFoundError, ImportError): pass
import numpy as np

from ...optim.optimizer import Config, Optimizer

if TYPE_CHECKING:
    import opytimizer.core
    from opytimizer.core import Space
    from ..._types import Numeric

__all__ = [
    "OpytimizerWrapper",
    "get_all_opytimizer_optimizers"
]

def silence_opytimizer():
    import logging

    try: import opytimizer.core
    except ModuleNotFoundError: raise ModuleNotFoundError("Opytimizer is not installed")
    import opytimizer.optimizers
    import opytimizer.opytimizer
    import opytimizer.spaces
    import opytimizer.utils.logging
    logging.getLogger("opytimizer").setLevel(logging.WARNING)
    logging.getLogger("opytimizer.opytimizer").setLevel(logging.WARNING)
    logging.getLogger("opytimizer.optimizers").setLevel(logging.WARNING)
    logging.getLogger("opytimizer.spaces").setLevel(logging.WARNING)
    logging.getLogger("opytimizer.spaces.search").setLevel(logging.WARNING)
    logging.getLogger("opytimizer.core").setLevel(logging.WARNING)
    logging.getLogger("opytimizer.core.space").setLevel(logging.WARNING)
    logging.getLogger("opytimizer.core.function").setLevel(logging.WARNING)
    logging.getLogger("opytimizer.core.optimizer").setLevel(logging.WARNING)
    opytimizer.opytimizer.tqdm.__init__ = partialmethod(opytimizer.opytimizer.tqdm.__init__, disable=True) # type:ignore

    # code from https://github.com/gugarosa/opytimizer/issues/20
    # Gathers all instantiated loggers, even the children
    loggers = {name: logging.getLogger(name) for name in logging.root.manager.loggerDict} # type:ignore pylint:disable = E1101

    # Iterates through all loggers and set their level to INFO
    for name, logger in loggers.items():
        if 'opytimizer' in name: logger.setLevel(logging.WARNING)


def _opytimizer_names(cls) -> list[str]:
    # we get full name like `evolutionary.DE':
    if not isinstance(cls, type): cls = cls.__class__
    namespace = str(cls).split("'")[1].split('.')
    return [f'{namespace[-3]}.{namespace[-1]}', namespace[-1]]

class OpytimizerWrapper(Optimizer):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode=False,
    )
    wrapped_optimizer: "opytimizer.core.Optimizer"
    def __init__(
        self,
        optimizer: "opytimizer.core.Optimizer | type[opytimizer.core.Optimizer]",
        n_agents: int = 20,
        space_cls: "Optional[type[Space]]" = None,
        save_agents=False,
        silence=True,
        mode: Literal['step', 'start'] = "step",
        budget: Optional[int] = None,
    ):
        """Wrapper for any optimizer from Opytimizer library. This does `n_agents` evaluations per step.

        :param params: _description_
        :param optimizer: _description_
        :param budget: _description_, defaults to 1000
        :param n_agents: _description_, defaults to 20
        :param space_cls: _description_, defaults to SearchSpace
        :param save_agents: _description_, defaults to False
        :param silence: _description_, defaults to True
        :param mode: _description_, defaults to "step"
        """
        try: import opytimizer.spaces
        except ModuleNotFoundError: raise ModuleNotFoundError("Opytimizer is not installed")
        super().__init__()

        if silence: silence_opytimizer()
        if isinstance(optimizer, type): optimizer = optimizer()
        self.wrapped_optimizer = optimizer
        if space_cls is None: space_cls = opytimizer.spaces.SearchSpace
        self.space_cls = space_cls
        self.n_agents = n_agents
        self.save_agents = save_agents
        self.last_value = float('nan')
        self.current_step = 0
        self.budget = budget
        self.mode = mode
        self.opytimizer = None

        if not hasattr(self, 'names'): self.names = _opytimizer_names(self.wrapped_optimizer)

    def set_params(self, params):
        super().set_params(params)
        _, self.slices = self.params.params_to_vec(only_used=False)
        self.lb = self.params.get_lower_bounds(normalized=True, fallback='default')
        self.ub = self.params.get_upper_bounds(normalized=True, fallback='default')
        return self

    def step(self, study):

        if self.opytimizer is None:
            from opytimizer.core.function import Function
            from opytimizer.opytimizer import Opytimizer
            self.opytimizer = Opytimizer(
                space = self.space_cls(n_agents=self.n_agents, n_variables=len(self.lb), lower_bound=self.lb, upper_bound=self.ub),
                optimizer = self.wrapped_optimizer,
                function = Function(partial(study.evaluate_vec, slices = self.slices)),
                save_agents = self.save_agents
            )

        if self.budget is None:
            warnings.warn("Opytimizer optimizers might work incorrectly without specifying budget")
            self.budget = 1
            self.mode = 'step'

        self.opytimizer.n_iterations = max(int(self.budget / self.n_agents), 1)

        if self.mode == 'start': self.opytimizer.start(self.opytimizer.n_iterations)

        elif self.mode == 'step':
            if self.current_step == 0: self.opytimizer.optimizer.evaluate(*self.opytimizer.evaluate_args)
            self.opytimizer.optimizer.update(*self.opytimizer.update_args)
            self.opytimizer.optimizer.evaluate(*self.opytimizer.evaluate_args)

        self.current_step += 1


def get_all_opytimizer_optimizers() -> "list[type[opytimizer.core.Optimizer]]":
    try: from opytimizer.optimizers import (boolean, evolutionary, misc, population, science, social, swarm)
    except ModuleNotFoundError: raise ModuleNotFoundError("Opytimizer is not installed")
    from opytimizer import core
    from ...python_tools import subclasses_recursive
    return list(sorted(list(subclasses_recursive(core.Optimizer)), key=lambda x: x.__name__))