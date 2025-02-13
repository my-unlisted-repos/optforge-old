from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._types import NumericArrayLike, Numeric
from .._utils import _ensure_float_or_ndarray, _ensure_float

if TYPE_CHECKING:
    from .trial import Trial

def _param_constr_bounds(
    self: "Trial",
    param_name: str,
    low: Numeric | NumericArrayLike | None,
    high: Numeric | NumericArrayLike | None,
    inclusive,
    eps,
    weight,
    constr_name: str | None,
) -> float | np.ndarray:
    if low is None: low = -np.inf
    else: low = _ensure_float_or_ndarray(low)
    if high is None: high = np.inf
    else: high = _ensure_float_or_ndarray(high)
    param_value = self.original_params[param_name]

    if inclusive: lfunc = np.less_equal; gfunc = np.greater_equal
    else: lfunc = np.less; gfunc = np.greater
    if lfunc(param_value, high).all() and gfunc(param_value, low).all():
        # constraints not violated
        return param_value
    else:
        low = low + eps
        high = high - eps

        # calculate violation value
        violation = param_value - high
        if violation < 0: violation = low - param_value
        if isinstance(violation, np.ndarray): violation = float(violation.sum())

        # record the constraint violation
        if constr_name is None: constr_name = f'{param_name}_bounds{len(self._param_violations)}'
        if constr_name in self._param_violations: self._param_violations[constr_name] += violation * weight
        else: self._param_violations[constr_name] = violation * weight

        # set param to satisfy constraint and return it
        if isinstance(param_value, np.ndarray): sat_value = self.original_params[param_name] = np.clip(param_value, low, high) # type:ignore
        else: sat_value = self.original_params[param_name] = max(min(param_value, high), low)
        return sat_value

def _single_param_l1(
    self: "Trial",
    param_names: str,
    eq: Numeric | NumericArrayLike | None,
    axis: int | Sequence[int] | None,
    weight: float,
    constr_name: str | None,
    low: float | None,
) -> float | np.ndarray:
    param_value: float | np.ndarray = self.original_params[param_names]
    psum = np.sum(np.abs(param_value), axis=axis) # type:ignore

    if eq is None: return param_value
    else: eq = _ensure_float_or_ndarray(eq)

    if np.equal(psum, eq).all(): return param_value
    else:
        violation = float(np.sum(np.abs(psum - eq)))

        # record the constraint violation
        if constr_name is None: constr_name = f'{param_names}_eq{len(self._param_violations)}'
        if constr_name in self._param_violations: self._param_violations[constr_name] += violation * weight
        else: self._param_violations[constr_name] = violation * weight

        # set param to satisfy constraint and return it
        if low is not None:
            psum = psum - low * (param_value.size / psum.size if isinstance(param_value,np.ndarray) else 1)
            eq = eq - low
            param_value = param_value - low

        div = psum / eq
        constrained = param_value / div
        if low is not None: constrained += low

        self.original_params[param_names] = constrained
        return constrained

def _multi_param_l1(
    self: "Trial",
    param_names: list[str] | tuple[str],
    eq: Numeric | NumericArrayLike | None,
    axis: int | Sequence[int] | None,
    weight: float,
    constr_name: str | None,
    low: float | None,
) -> list[float | np.ndarray]:
    param_values: list[float | np.ndarray] = [self.original_params[i] for i in param_names]
    psum = np.add.reduce(np.abs(param_values), axis = axis) # type:ignore

    if eq is None: return param_values
    else: eq = _ensure_float_or_ndarray(eq)

    if np.equal(psum, eq).all(): return param_values
    else:
        violation = float(np.sum(np.abs(psum - eq)))

        # record the constraint violation
        if constr_name is None: constr_name = f'eq{len(self._param_violations)}'
        if constr_name in self._param_violations: self._param_violations[constr_name] += violation * weight
        else: self._param_violations[constr_name] = violation * weight

        # set param to satisfy constraint and return it
        if low is not None:
            psum = psum - low * (param_values[0].size / psum.size if isinstance(param_values[0],np.ndarray) else 1) * len(param_names)
            eq = eq - low
            param_values = [v - low for v in param_values]

        div = psum / eq

        if low is None: constrained = [i / div for i in param_values]
        else: constrained = [(i/div) + low for i in param_values]

        for c, name in zip(constrained, param_names):
            self.original_params[name] = c
        return constrained

def _param_constr_set(self: "Trial", param_name:str, eq: Numeric | NumericArrayLike | None, weight = 0, constr_name = None):
    param_value = self.original_params[param_name]
    if eq is None: return param_value
    eq = _ensure_float_or_ndarray(eq)
    if param_value == eq: return param_value
    else:
        violation = np.sum(np.abs(eq - param_value))

        # record the constraint violation
        if constr_name is None: constr_name = f'eq{len(self._param_violations)}'
        if constr_name in self._param_violations: self._param_violations[constr_name] += violation * weight
        else: self._param_violations[constr_name] = violation * weight

        return eq