import time
from collections.abc import Sequence
from numbers import Number

import numpy as np
from ._types import Numeric, NumericArrayLike


def _ensure_float(x) -> float:
    """A function in case this logic gets more complicated, and it will once I add pytorch support"""
    return float(x)

def _maybe_ensure_float(x):
    """A function in case this logic gets more complicated, and it will once I add pytorch support"""
    if x is None: return x
    return _ensure_float(x)

def _ensure_float_or_ndarray(x) -> float | np.ndarray:
    """Converts any single scalar to a float, including size 1 numpy arrays. 
    Everything else converted to a numpy array.
    This is used for managing single- and multi-objective optimization returns."""
    # numbers are converted to floats
    if isinstance(x, Number): return float(x) # type:ignore
    # everything else is converted to a numpy array
    # it can still be a size 1 numpy array
    x = np.array(x, copy=False)
    if x.size == 1: return float(x)
    return x

def _ensure_float_or_1darray(x) -> float | np.ndarray:
    """Converts any single scalar to a float, including size 1 numpy arrays. 
    Everything else converted to a numpy array and flattened.
    This is used for managing single- and multi-objective optimization returns."""
    # numbers are converted to floats
    if isinstance(x, Number): return float(x) # type:ignore
    # everything else is converted to a numpy array
    # it can still be a size 1 numpy array
    x = np.array(x, copy=False).flatten()
    if x.size == 1: return float(x)
    return x

def time_deco(func):
    def inner(*args, **kwargs):
        time_start = time.time()
        res = func(*args, **kwargs)
        print(f"{func.__name__} {time.time() - time_start} seconds")
        return res
    return inner
