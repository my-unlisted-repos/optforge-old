from typing import TYPE_CHECKING, Literal

import numpy as np

from ..._types import Numeric, NumericArrayLike
from ..benchmark import Benchmark

if TYPE_CHECKING:
    from ...trial import Trial

def _suggest_transition(t: "Trial", init):
    return t.suggest_categorical_array(
            "transition",
            shape=len(init),
            choices=[True, False],
            init=init,
        )

def _suggest_pacing(t: "Trial", init):
    return t.suggest_array('pacing', shape=len(init), low=200, high=1200, init=init)


class NevergradCycling(Benchmark):
    def __init__(
        self,
        strategy_index: Literal[30, 31, 61, 22, 23, 45] = 30,
        log_params=True,
        note=None,
    ):
        """
        Team Pursuit Track Cycling Simulator from Nevergrad.

    strategy: int
        Refers to Transition strategy or Pacing strategy (or both) of the cyclists; this depends on the strategy length.
        Strategy length can only be 30, 31, 61, 22, 23, 45.
        30: mens transition strategy.
        31: mens pacing strategy.
        61: mens transition and pacing strategy combined.
        22: womens transition strategy.
        23: womens pacing strategy.
        45: womens transition and pacing strategy combined.
        """
        super().__init__(log_params = log_params, note = note)
        from nevergrad.functions.cycling import Cycling

        self.bench = Cycling(strategy_index=strategy_index)

        self.x0 = self.bench.parametrization.value

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        if isinstance(self.x0, tuple) and isinstance(self.x0[0], bool):
            return self.bench(_suggest_transition(trial, self.bench.parametrization.value))

        elif isinstance(self.x0, np.ndarray):
            return self.bench(_suggest_pacing(trial, self.bench.parametrization.value))

        d = self.x0[1].copy()
        if 'transition' in d:
            d['transition'] = _suggest_transition(trial, d['transition'])
        if 'pacing' in d:
            d['pacing'] = _suggest_pacing(trial, d['pacing'])

        return self.bench(((), d))

