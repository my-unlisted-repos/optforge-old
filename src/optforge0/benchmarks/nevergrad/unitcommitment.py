from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..._types import Numeric, NumericArrayLike
from ..benchmark import Benchmark

if TYPE_CHECKING:
    from ...trial import Trial


class NevergradUnitCommitmentProblem(Benchmark):
    def __init__(
        self,
        problem_name: Literal['semi-continuous'] = "semi-continuous",
        num_timepoints: int = 13,
        num_generators: int = 3,
        penalty_weight: float = 10000.,
        log_params=True,
        note=None,
    ):
        super().__init__(log_params = log_params, note = note)
        from nevergrad.functions.unitcommitment import UnitCommitmentProblem
        self.bench = UnitCommitmentProblem(
            problem_name=problem_name,
            num_timepoints=num_timepoints,
            num_generators=num_generators,
            penalty_weight=penalty_weight
        )
        self.x0 = self.bench.parametrization.value[1]

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        operational_output = trial.suggest_array(
            'operational_output', init = self.x0['operational_output'], low = 0, high = self.bench.p_max)
        operational_states = trial.suggest_categorical_array(
            'operational_states', shape = len(self.x0['operational_states']), init = self.x0['operational_states'], choices = [0, 1])
        return self.bench(
            operational_output=operational_output,
            operational_states=operational_states.tolist(),
        )