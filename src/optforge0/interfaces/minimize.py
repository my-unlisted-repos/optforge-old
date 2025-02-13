from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from ..pruners import Pruner
from ..study import Study

if TYPE_CHECKING:
    from .._types import Numeric, NumericArrayLike
    from ..optim.optimizer import Optimizer
    from ..trial import Trial

__all__ = [
    "minimize",
    "minimize_trial_func"
]

def minimize(
    fun: "Callable[..., Numeric | NumericArrayLike]",
    x0: np.ndarray | float | Any,
    optimizer: "Optimizer | str",
    max_evals: Optional[int] = None,
    max_steps: Optional[int] = None,
    tol: Optional[float] = None,
    timeout: Optional[float] = None,
    args=(),
    bounds: Optional[tuple[float | None, float | None]]=None,
    log_params = True,
    n_jobs: "Optional[int]" = 1,
    executor: Optional[type[Executor]] = ThreadPoolExecutor,
    catch: tuple[type[Exception], ...] = (),
    fallback_bounds = (-1, 1),
    progress: bool = True,
    force_batch_mode:bool = False,
    vectorized: Literal[False] | int = False,
    catch_kb_interrupt: bool = True,
    seed: Optional[int] = None,
    pruner: Optional[Pruner] = None,
) -> "Study":
    """Minimize a scalar function that takes a scalar or an array as input.

    :param fun: The objective function to be minimized. ``fun(x, *args) -> float``, where ``x`` is a numpy.ndarray or a numpy scalar with same shape as `x0`.
    :param x0: Initial guess, a numpy.ndarray, a scalar, or anything that can be converted to a numpy array.
    :param optimizer: The optimizer.
    :param max_evals: Terminate after this many function evaluations, defaults to None
    :param max_steps: Terminate after this many steps. Some optimizers do multiple evaluations per step. defaults to None
    :param timeout: Terminate after this many seconds, defaults to None
    :param args: Additional arguments to pass to ``fun``. defaults to ()
    :param bounds: A ``(low, high)`` tuple. `low` and `high` can be `None`, defaults to None
    :param log_params: Whether to log ``x`` on each evaluation. Set to False if ``x`` are too big. defaults to True
    :return: A Study object. Important attributes are `best_value`, `best_params`.

    example:
    ```py
    def func(x):
        return x[0] ** 2 + x[1] ** 2

    study = minimize(
        func,
        x0 = np.array([8, 8],
        optimizer = op.optim.RandomSearch(),
        max_evals = 1000,
        bounds = (-10, 10),
    )
    study.best_params # >> {'x': array([2.77084931e-05, 8.54822059e-05])}

    ```

    """
    x0 = np.asanyarray(x0)
    if bounds is None: bounds = (None, None)
    def objective(trial: "Trial"):
        return fun(trial.suggest_array(
                    'x',
                    shape =
                    x0.shape,
                    init = x0,
                    low = bounds[0],
                    high = bounds[1],
                    fallback_low = fallback_bounds[0],
                    fallback_high = fallback_bounds[1]),
                *args)

    study = Study(log_params=log_params)
    study.set_pruner(pruner)
    study.optimize(
        objective,
        optimizer=optimizer,
        max_evals=max_evals,
        max_steps=max_steps,
        tol=tol,
        timeout=timeout,
        executor=executor,
        n_jobs=n_jobs,
        catch = catch,
        progress=progress,
        force_batch_mode=force_batch_mode,
        vectorized=vectorized,
        catch_kb_interrupt=catch_kb_interrupt,
        seed = seed,
    )
    return study

def minimize_trial_func(
    fun:"Callable[[Trial], Numeric | NumericArrayLike]",
    optimizer: "Optimizer | str",
    max_evals: Optional[int] = None,
    max_steps: Optional[int] = None,
    tol: Optional[float] = None,
    timeout: Optional[float] = None,
    log_params = True,
    n_jobs: "Optional[int]" = 1,
    executor: Optional[type[Executor]] = ThreadPoolExecutor,
    catch: tuple[type[Exception], ...] = (),
    progress = True,
    force_batch_mode:bool = False,
    vectorized: Literal[False] | int = False,
    catch_kb_interrupt = True,
    seed: Optional[int] = None,
    pruner: Optional[Pruner] = None,

) -> "Study":

    study = Study(log_params=log_params)
    study.set_pruner(pruner)
    study.optimize(
        fun,
        optimizer=optimizer,
        max_evals=max_evals,
        max_steps=max_steps,
        tol=tol,
        timeout=timeout,
        executor=executor,
        n_jobs=n_jobs,
        catch = catch,
        progress=progress,
        force_batch_mode=force_batch_mode,
        vectorized=vectorized,
        catch_kb_interrupt=catch_kb_interrupt,
        seed = seed,
    )
    return study