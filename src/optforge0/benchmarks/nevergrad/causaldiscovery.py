from typing import TYPE_CHECKING, Literal

from ..._types import NumericArrayLike, Numeric
from ..benchmark import Benchmark

if TYPE_CHECKING:
    from ...trial import Trial


class NevergradCausalDiscovery(Benchmark):
    def __init__(
        self,
        generator: Literal["tuebingen", "sachs", "dream4", "acylicgraph"] = "sachs",
        causal_mechanism: Literal[
            "linear",
            "polynomial",
            "sigmoid_add",
            "sigmoid_mix",
            "gp_add",
            "gp_mix",
            "nn",
        ] = "linear",
        noise: str = "gaussian",
        noise_coeff: float = 0.4,
        npoints: int = 500,
        nodes: int = 20,
        parents_max: int = 5,
        expected_degree: int = 3,
        log_params=True,
        note=None,
    ):
        """A categorical problem."""
        super().__init__(log_params = log_params, note = note)
        from nevergrad.functions.causaldiscovery import CausalDiscovery

        self.bench = CausalDiscovery(
            generator=generator,
            causal_mechanism=causal_mechanism,
            noise=noise,
            noise_coeff=noise_coeff,
            npoints=npoints,
            nodes=nodes,
            parents_max=parents_max,
            expected_degree=expected_degree,
        )
        self.x0 = self.bench.parametrization.value[1]['network_links']

    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        X = trial.suggest_categorical_array(
            "X",
            shape=len(self.x0),
            choices=(-1, 0, 1),
            init=self.x0,
        )
        return self.bench(X)