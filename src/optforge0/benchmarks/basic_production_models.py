from typing import TYPE_CHECKING

from .benchmark import Benchmark

if TYPE_CHECKING:
    from ..trial import Trial


class TwoProductProductionPlan(Benchmark):
    def objective(self, trial: "Trial"):
        x = trial.suggest_float('x', 0, 40)
        y = trial.suggest_float('y', 0, scale = 1/40)
        y = trial.param_constr_bounds('y', high = 80 - x) # labor A
        trial.constr_bounds(2*x + y, high = 100) # labor B
        return -(40*x + 30*y)
