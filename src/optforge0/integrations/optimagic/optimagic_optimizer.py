#pylint:disable = W0707, W0621
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from ..._types import Numeric
from ..._utils import _ensure_float
from ...optim.optimizer import Config, Optimizer
from ...paramdict import ParamDict
from ...python_tools import reduce_dim
from ...study import EndStudy
from ...optim.optimizer import KwargsMinimizer

if TYPE_CHECKING:
    import optimagic
    import optimagic.algorithms
    import optimagic.optimizers
    import optimagic.parameters
    import optimagic.parameters.bounds

__all__ = [
    "OptimagicMinimize",
    "get_all_optimagic_algorithms",
]

def _optimagic_names(kwargs: dict[str, Any]) -> list[str]:
    algo = kwargs['algorithm']
    if not isinstance(algo, str):
        if hasattr(algo, '__name__'): name = algo.__name__ # type:ignore
        else: name = algo.__class__.__name__
    else: name = algo
    return [name, ]



class OptimagicMinimize(KwargsMinimizer):
    def __init__(
        self,
        algorithm: "str | optimagic.algorithms.Algorithm | type[optimagic.algorithms.Algorithm] | None",
        *,
        # bounds: "optimagic.parameters.bounds.Bounds | scipy.optimize._constraints.Bounds | Sequence[tuple[float, float]] | None" = None,
        constraints = None,
        fun_kwargs: dict[str, Any] | None = None,
        algo_options: dict[str, Any] | None = None,
        jac: Any | list[Any] | None = None,
        jac_kwargs: dict[str, Any] | None = None,
        fun_and_jac: Any | Any | None = None,
        fun_and_jac_kwargs: dict[str, Any] | None = None,
        numdiff_options: Any | Any | None = None,
        logging: bool | str | Any | dict[str, Any] | None = None,
        # error_handling: Any | str = ErrorHandling.RAISE,
        error_penalty: dict[str, float] | None = None,
        scaling: bool | Any = False,
        multistart: bool | Any = False,
        collect_history: bool = True,
        skip_checks: bool = True,
        # x0: Any | None = None,
        method: str | None = None,
        args: tuple[Any] | None = None,
        hess: Any | None = None,
        hessp: Any | None = None,
        callback: Any | None = None,
        options: dict[str, Any] | None = None,
        tol: Any | None = None,
        criterion: Any | None = None,
        criterion_kwargs: dict[str, Any] | None = None,
        derivative: Any | None = None,
        derivative_kwargs: dict[str, Any] | None = None,
        criterion_and_derivative: Any | None = None,
        criterion_and_derivative_kwargs: dict[str, Any] | None = None,
        log_options: dict[str, Any] | None = None,
        lower_bounds: Any | None = None,
        upper_bounds: Any | None = None,
        soft_lower_bounds: Any | None = None,
        soft_upper_bounds: Any | None = None,
        scaling_options: dict[str, Any] | None = None,
        multistart_options: dict[str, Any] | None = None,
        restart: Literal["best", "last", "random"] = "last",
        budget: Optional[int] = None,
        fake_ls: bool | Literal['auto'] = 'auto',

    ):
        super().__init__(locals().copy(), ignore=['fake_ls'])

        import optimagic
        self.optimagic = optimagic
        from optimagic import exceptions
        self.optimagic_exceptions = exceptions

        name = _optimagic_names(self.kwargs)[0].lower()
        if fake_ls == 'auto':
            if name in {'nagdfols', 'bhhh', 'pounders', 'taopounders', 'tranquilols', 'scipylsdogbox', 'scipylslm', 'scipylstrf'}:
                self.fake_ls = True
            else: self.fake_ls = False
        else: self.fake_ls: bool = fake_ls

        if not hasattr(self, 'names'): self.names = _optimagic_names(self.kwargs)

    def objective(self, x:np.ndarray):
        if self.fake_ls:
            res = np.array([super().objective(x)]) #* self.x0.size)
            return res
        return super().objective(x)

    def minimize(self, objective):
        if self.fake_ls: self.weights = np.random.uniform(0, 2, size = self.x0.shape)
        try:
            self.res = self.optimagic.minimize(
                self.optimagic.mark.least_squares(objective) if self.fake_ls else objective,
                x0 = self.x0,
                bounds=self.bounds,
                **self.kwargs,
            )
        # continue from last point on error
        except (RuntimeError, np.linalg.LinAlgError, ValueError): pass
        except self.optimagic_exceptions.InvalidFunctionError as e:
            warnings.warn(f"InvalidFunctionError: {self.kwargs['algorithm']} raised an `InvalidFunctionError` error: {e!r}.")
            raise EndStudy
        except self.optimagic_exceptions.UserFunctionRuntimeError: 
            self.params.randomize()
            self._create_vec(self.params)



def get_all_optimagic_algorithms() -> "list[type[optimagic.algorithms.Algorithm]]":
    import optimagic
    import optimagic.algorithms
    from optimagic import optimizers

    from ...python_tools import subclasses_recursive
    return list(sorted(list(subclasses_recursive(optimagic.algorithms.Algorithm)), key=lambda x: x.__name__))