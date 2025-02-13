from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional
from collections.abc import Callable

from ..rng import RNG
from .._types import Numeric

if TYPE_CHECKING:
    from ..optim.optimizer import Optimizer
    from ..study import Study


class _MockOptimizer:
    def __init__(self, budget: Optional[int] = None):
        self.budget = budget
        self.current_eval = 0
        self.current_step = 0

class Scheduler(ABC):
    def __init__(self):
        self.budget = None
        self._schedulers: set["Scheduler"] = set()
        self.init_eval = 0.
        self.rng = RNG(None)

    def __str__(self): return f'scheduler:{self.__class__.__name__}'
    def __repr__(self): return str(self)

    @abstractmethod
    def step(self, study: "Study"):
        """Update the state of this scheduler.
        This is called once after each `optimizer.tell` or after each `optimizer.step`.
        Therefore `study` has already been updated with the results of all newly requested trials."""

    @abstractmethod
    def __call__(self) -> Any:
        """Get the value of this scheduler."""

    def _internal_step(self, study: "Study"):
        if self.budget is None: self.budget = study.optimizer.budget

        self.current_eval = study.current_eval - self.init_eval
        self.step(study)

        for s in self._schedulers: s._internal_step(study)

    def schedule(self, value:"Any | Scheduler") -> "_SchedulerCaller":
        if isinstance(value, _ConfiguredScheduler): value = value()
        if isinstance(value, Scheduler):
            value.rng = self.rng
            self._schedulers.add(value)
        return _SchedulerCaller(value)

    def plot(self, budget = None):
        budget = self.budget if budget is None else budget
        if budget is None: raise ValueError("No budget set")
        from copy import deepcopy

        import matplotlib.pyplot as plt
        plotter = deepcopy(self)
        optimizer = _MockOptimizer(budget)  #type:ignore
        ax = plt.gca()
        from ..plt_tools import ax_plot_
        values = []
        for i in range(budget): # type:ignore
            values.append(plotter())
            plotter._internal_step(optimizer) # type:ignore
            optimizer.current_eval += 1
            optimizer.current_step += 1
        ax_plot_(ax, list(range(values)), values, xlabel = 'eval', ylabel = 'value') # type:ignore

    @classmethod # type:ignore
    def configured[**P](cls: "Callable[P, Scheduler]", /, *args: P.args, **kwargs: P.kwargs): # pylint:disable = E0602
        return _ConfiguredScheduler(cls, *args, **kwargs)

class _ConfiguredScheduler:
    def __init__[**P](self, cls: Callable[P, Scheduler], /, *args: P.args, **kwargs: P.kwargs): # pylint:disable = E0602
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.cls(*self.args, **self.kwargs)

class _SchedulerCaller:
    def __init__(self, value: Any | Scheduler):
        self.value = value
        self.is_scheduler = isinstance(value, Scheduler)
    def __call__(self) -> Any:
        if self.is_scheduler: return self.value()
        return self.value

SchedulableInt = Scheduler | _ConfiguredScheduler | int
SchedulableFloat = Scheduler | _ConfiguredScheduler | float
SchedulableNumeric = Scheduler | _ConfiguredScheduler | Numeric
Schedulable = Scheduler | _ConfiguredScheduler | Any