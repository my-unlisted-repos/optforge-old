from typing import TYPE_CHECKING, Literal
from collections.abc import Sequence
import numpy as np

from ..._types import Numeric, NumericArrayLike
from ..benchmark import Benchmark

if TYPE_CHECKING:
    from ...trial import Trial

__all__ = [
    "NevergradPhotonics",
]

def _nevergrad_get_bounds(name, dimension):
    if name == "bragg":
        shape = (2, dimension // 2)
        bounds = [(2, 3), (30, 180)]
    elif name == "cf_photosic_realistic":
        shape = (2, dimension // 2)
        bounds = [(1, 9), (30, 180)]
    elif name == "cf_photosic_reference":
        shape = (1, dimension)
        bounds = [(30, 180)]
    elif name == "chirped":
        shape = (1, dimension)
        bounds = [(30, 180)]
    elif name == "morpho":
        shape = (4, dimension // 4)
        bounds = [(0, 300), (0, 600), (30, 600), (0, 300)]
    else:
        raise NotImplementedError(f"Transform for {name} is not implemented")
    return bounds

class NevergradPhotonics(Benchmark):
    def __init__(
        self,
        problem: Literal["bragg", 'morpho', "chirped", "cf_photosic_realistic", "cf_photosic_reference"],
        dimension: Literal[16, 40, 60, 80],
        log_params=True,
        note=None,
    ):
        super().__init__(log_params = log_params, note = note)
        from nevergrad.functions.photonics import Photonics
        self.bench = Photonics(problem, dimension, )

        self.x0 = self.bench.parametrization.value
        self.bounds = _nevergrad_get_bounds(problem, dimension)

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        value = []
        for i,b in enumerate(self.bounds):
            value.append(trial.suggest_array(f'X_{i}', init=self.x0[i], low = b[0], high = b[1]))
        return self.bench(np.array(value))