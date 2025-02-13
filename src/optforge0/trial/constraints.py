from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .trial import Trial


def _constr_bounds(
    self: "Trial",
    value: float | np.ndarray,
    low: float | np.ndarray | None = None,
    high: float | np.ndarray | None = None,
    inclusive=True,
    hard = True,
    reduce = True,
    eps = 0.,
    weight=1.0,
    constr_name: str | None = None,
) -> float | np.ndarray:

    if low is None: low = -np.inf
    if high is None: high = np.inf
    violations_dict = self._hard_violations if hard else self._soft_violations
    if constr_name is None: constr_name = f'bounds{len(violations_dict)}'

    if inclusive: lfunc = np.less_equal; gfunc = np.greater_equal
    else: lfunc = np.less; gfunc = np.greater

    # constraints not violated
    if lfunc(value, high).all() and gfunc(value, low).all():
        violations_dict[constr_name] = 0
        return value

    # constraints are violated
    else:
        low = low + eps
        high = high - eps

        # calculate violation value
        violation = value - high
        if violation < 0: violation = low - value
        if isinstance(violation, np.ndarray):
            if reduce: violation = float(violation.sum())
            else: violation = violation.flatten()

        # record the constraint violation
        if constr_name in violations_dict: violations_dict[constr_name] += violation * weight # type:ignore
        else: violations_dict[constr_name] = violation * weight # type:ignore

        # set param to satisfy constraint and return it
        if isinstance(value, np.ndarray): return np.clip(value, low, high)
        return max(min(value, high), low) # type:ignore