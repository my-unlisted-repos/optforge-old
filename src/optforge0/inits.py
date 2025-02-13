from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Sequence
from numbers import Number
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

import numpy as np

from ._types import Numeric

if TYPE_CHECKING:
    from .param import Param

def _init_uniform_or_randn(param: "Param", shape: int | Sequence[int]) -> np.ndarray:
    if param.low is None:
        if param.high is None:
            # low None high None
            return np.random.normal(size = shape)
        # low None high defined
        return param.high - np.abs(np.random.normal(size = shape))
    if param.high is None:
        # low defined high None
        return param.low + np.abs(np.random.normal(size = shape))
    # both domains defined
    param.data = np.zeros(shape, dtype=np.float64)
    return param.sample_random()

def _init_uniform(param: "Param", shape: int | Sequence[int]) -> np.ndarray:
    if param.low is None or param.high is None: raise ValueError("Both low and high domains must be provided")
    param.data = np.zeros(shape, dtype=np.float64)
    return param.sample_random()

def _init_normal(param: "Param", shape: int | Sequence[int]) -> np.ndarray:
    return np.random.normal(size = shape)

def _init_zeros(param: "Param", shape: int | Sequence[int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)

def _init_ones(param: "Param", shape: int | Sequence[int]) -> np.ndarray:
    return np.ones(shape, dtype=np.float64)

def _init_low(param: "Param", shape: int | Sequence[int]) -> np.ndarray:
    if param.low is None: raise ValueError("Low bound must be provided")
    return np.full(shape, param.low, dtype=np.float64)

def _init_high(param: "Param", shape: int | Sequence[int]) -> np.ndarray:
    if param.high is None: raise ValueError("High bound must be provided")
    return np.full(shape, param.high, dtype=np.float64)

def _init_mean(param: "Param", shape: int | Sequence[int]) -> np.ndarray:
    if param.low is None:
        if param.high is None:
            # low None high None
            return np.full(shape, 0., dtype=np.float64)
        # low None high defined
        return np.full(shape, ((param.high / 2) if param.high > 0 else (-param.high * 2)) - 1, dtype=np.float64)
    if param.high is None:
        # low defined high None
        return np.full(shape, ((param.low * 2) if param.low > 0 else (param.low / 2)) + 1, dtype=np.float64)
    # both domains defined
    return np.full(shape, (param.low + param.high) / 2, dtype=np.float64)


_INITS = {
    "uniform_or_normal": _init_uniform_or_randn,
    "uniform": _init_uniform,
    "randn": _init_normal,
    "normal": _init_normal,
    "zeros": _init_zeros,
    "ones": _init_ones,
    "low": _init_low,
    "high": _init_high,
    "mean": _init_mean,
}

InitLiteral = Literal["uniform_or_normal", "uniform", "randn", "normal", "zeros", "ones", "low", "high", "mean"]

def _set_init_value(
    param: "Param",
    init: "InitLiteral | Callable | np.ndarray | Numeric | Sequence[Numeric]",
    shape: int | Sequence[int],
    # shape: int | Sequence[int],
    # init: "str | Callable | np.ndarray | torch.Tensor | Sequence | Number",
    # low: Optional[float],
    # high: Optional[float],
    # scaler,
    # dtype: Optional[np.dtype],
    ) -> None:
    if isinstance(init, str): param.data = _INITS[init](param, shape)
    elif callable(init):
        param.data = np.empty(shape, dtype=np.float64)
        param.set_value(init(shape))
    elif isinstance(init, (np.ndarray, Sequence)):
        param.data = np.empty(shape, dtype=np.float64)
        param.set_value(init)
    elif isinstance(init, (int,float,Number)): param.data = np.full(shape, param._forward(float(init)), dtype=np.float64)
    else: raise TypeError(type(init))