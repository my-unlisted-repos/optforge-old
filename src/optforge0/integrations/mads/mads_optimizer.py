import numpy as np

from ...optim.optimizer import KwargsMinimizer

__all__ = [
    "Orthomads",
]


class Orthomads(KwargsMinimizer):
    names = ['OrthoMADS', ]
    lib = 'mads'
    def __init__(
        self,
        dp = 0.1,
        dm = 0.01,
        dp_tol = -float('inf'),
        nitermax = float('inf'),
        displog = False,
        savelog = False,
    ):
        super().__init__(locals().copy(), fallback_bounds='default')

    def minimize(self, objective):
        from mads.mads import orthomads
        self.res = orthomads(
            design_variables = self.x0,
            bounds_upper = np.array(self.bounds)[:, 1],
            bounds_lower = np.array(self.bounds)[:, 0],
            objective_function = objective,
            **self.kwargs,
        )
