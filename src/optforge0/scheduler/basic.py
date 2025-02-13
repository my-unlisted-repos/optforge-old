import math
import random
from typing import TYPE_CHECKING, Any, Optional

from .scheduler import SchedulableFloat, Scheduler

if TYPE_CHECKING:
    from ..optim.optimizer import Optimizer
    from ..study import Study

__all__ = [
    "Linear",
    "Random",
    "Multiplicative",
    "AcceleratedAnnealing",
    "Sine",
]
class Linear(Scheduler):
    """Linearly changes from `start` to `stop`."""
    def __init__(self, start: float, stop:float, budget = None):
        super().__init__()
        self.start = start
        self.stop = stop
        self.budget = budget
        self.value = start

    def step(self, study: "Study"):
        budget = study.optimizer.budget if self.budget is None else self.budget
        if budget is None: raise ValueError('Linear scheduler requires a budget')
        self.value = self.start + (self.stop - self.start) * ((self.current_eval) / budget)

    def __call__(self):
        return self.value

class Multiplicative(Scheduler):
    """Linearly changes from `start` to `stop`."""
    def __init__(self, value:SchedulableFloat, mul: SchedulableFloat):
        super().__init__()
        self.value = self.schedule(value)
        self.current = 1
        self.multiplier = self.schedule(mul)

    def step(self, study: "Study"):
        self.current *= self.multiplier()

    def __call__(self):
        return self.value() * self.current


class AcceleratedAnnealing(Scheduler):
    """Multiplies `high` by `mul` on each step. If objective improves, or if no improvement for `restart_steps`, resets to `high`."""
    def __init__(self, high: SchedulableFloat = 2, mul: SchedulableFloat = 0.9, restart_steps: SchedulableFloat = 100):
        super().__init__()
        self.high = self.schedule(high)
        self.mul = self.schedule(mul)
        self.restart_steps = self.schedule(restart_steps)

        self.no_improvement_steps = 0
        self.current_mul = 1
        self.best_value = float('inf')

    def step(self, study: "Study"):
        improved = study.best_value < self.best_value
        if improved or self.no_improvement_steps > self.restart_steps():
            self.no_improvement_steps = 0
            self.current_mul = self.mul()
            if improved: self.best_value = study.best_value
        else:
            self.current_mul *= self.mul()
            self.no_improvement_steps += 1

    def __call__(self):
        return self.high() * self.current_mul


class Random(Scheduler):
    def __init__(self, low: SchedulableFloat, high: SchedulableFloat):
        super().__init__()
        self.low = self.schedule(low)
        self.high = self.schedule(high)

    def step(self, study): pass

    def __call__(self):
        return self.rng.random.uniform(self.low(), self.high())


class Sine(Scheduler):
    """Linearly changes from `start` to `stop`."""
    def __init__(self, low: SchedulableFloat, high: SchedulableFloat, period: SchedulableFloat, budget=None):
        super().__init__()
        self.low = self.schedule(low)
        self.high = self.schedule(high)
        self.period = self.schedule(period)
        self.value = self.low()
        self.budget = budget


    def step(self, study: "Study"):
        budget = study.optimizer.budget if self.budget is None else self.budget
        if budget is None: raise ValueError('Sine scheduler requires a budget')
        self.value = math.sin(self.current_eval / ((self.period() * budget)/math.pi)) * (self.high() - self.low()) + self.low()

    def __call__(self):
        return self.value