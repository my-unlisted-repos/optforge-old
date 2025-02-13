from typing import TYPE_CHECKING, Literal
from collections.abc import Sequence
import numpy as np

from ..._types import Numeric, NumericArrayLike
from ..benchmark import Benchmark

if TYPE_CHECKING:
    from ...trial import Trial

__all__ = [
    "NevergradClustering",
    "NevergradPerceptron",
    "NevergradSammonMapping",
]

class NevergradClustering(Benchmark):
    def __init__(self, dataset: Literal["Ruspini", "German towns"] | np.ndarray | Sequence, num_clusters, log_params = True, note = None):
        super().__init__(log_params = log_params, note = note)
        from nevergrad.functions.mlda import Clustering
        if isinstance(dataset, str):
            self.bench = Clustering.from_mlda(dataset, num_clusters)
        else:
            dataset = np.array(dataset, copy=False)
            self.bench = Clustering(dataset, num_clusters)
        self.x0 = self.bench.parametrization.value

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        X = trial.suggest_array('X', init = self.x0, fallback_low=-10, fallback_high=10)
        return self.bench(X)

class NevergradPerceptron(Benchmark):
    def __init__(
        self,
        dataset: Literal["quadratic", "sine", "abs", "heaviside"] | np.ndarray | Sequence,
        log_params=True,
        note=None,
    ):
        super().__init__(log_params = log_params, note = note)
        from nevergrad.functions.mlda import Perceptron
        if isinstance(dataset, str):
            self.bench = Perceptron.from_mlda(dataset)
        else:
            dataset = np.array(dataset, copy=False)
            if dataset.shape[0] > 2: dataset = dataset.T
            x, y = dataset
            self.bench = Perceptron(x, y)
        self.x0 = self.bench.parametrization.value

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        X = trial.suggest_array('X', init = self.x0, fallback_low=-10, fallback_high=10)
        return self.bench(X)

class NevergradSammonMapping(Benchmark):
    def __init__(
        self,
        proximity_array: Literal["Virus"] | np.ndarray | Sequence,
        log_params=True,
        note=None,
    ):
        super().__init__(log_params = log_params, note = note)
        from nevergrad.functions.mlda import SammonMapping
        if isinstance(proximity_array, str):
            self.bench = SammonMapping.from_mlda(proximity_array)
        else:
            proximity_array = np.array(proximity_array, copy=False)
            self.bench = SammonMapping(proximity_array)
        self.x0 = self.bench.parametrization.value

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        X = trial.suggest_array('X', init = self.x0, fallback_low=-10, fallback_high=10)
        return self.bench(X)