from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..._types import Numeric, NumericArrayLike
from ..benchmark import Benchmark

if TYPE_CHECKING:
    from ...trial import Trial


class NevergradPowerSystem(Benchmark):
    def __init__(
        self,
        num_dams: int = 13,
        depth: int = 3,
        width: int = 3,
        year_to_day_ratio: float = 2.0,
        constant_to_year_ratio: float = 1.0,
        back_to_normal: float = 0.5,
        consumption_noise: float = 0.1,
        num_thermal_plants: int = 7,
        num_years: float = 1.0,
        failure_cost: float = 500.0,
        log_params=True,
        note=None,
    ):
        """"Very simple" model of a power system from Nevergrad.

        num_dams: int
            number of dams to be managed
        depth: int
            number of layers in the neural networks
        width: int
            number of neurons per hidden layer
        year_to_day_ratio: float = 2.
            Ratio between std of consumption in the year and std of consumption in the day.
        constant_to_year_ratio: float
            Ratio between constant baseline consumption and std of consumption in the year.
        back_to_normal: float
            Part of the variability which is forgotten at each time step.
        consumption_noise: float
            Instantaneous variability.
        num_thermal_plants: int
            Number of thermal plants.
        num_years: float
            Number of years.
        failure_cost: float
            Cost of not satisfying the demand. Equivalent to an expensive infinite capacity thermal plant.
        """
        super().__init__(log_params = log_params, note = note)
        from nevergrad.functions.powersystems import PowerSystem
        self.bench = PowerSystem(
            num_dams = num_dams,
            depth = depth,
            width = width,
            year_to_day_ratio = year_to_day_ratio,
            constant_to_year_ratio = constant_to_year_ratio,
            back_to_normal = back_to_normal,
            consumption_noise = consumption_noise,
            num_thermal_plants = num_thermal_plants,
            num_years = num_years,
            failure_cost = failure_cost,
        )

        self.x0 = self.bench.parametrization.value[0]

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        values = []
        for i, x in enumerate(self.x0):
            values.append(trial.suggest_array(f'X_{i}', init = x, fallback_low=-10, fallback_high=10))
        return self.bench(*values)

    def make_plots(self, fname='ps.png'):
        self.bench.make_plots(fname)