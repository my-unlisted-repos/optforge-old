import warnings
from collections.abc import Sequence

import numpy as np

try: import fast_pareto
except (ModuleNotFoundError, ImportError): fast_pareto = None

inf = float('inf')

def _slow_is_pareto_front(costs: np.ndarray | Sequence[np.ndarray]):
    """Returns list of bool values per each cost, True if it is in pareto front"""
    front = []
    for cost in costs:
        is_front = True
        for other in costs:
            if np.all(cost > other):
                is_front = False
                break
        front.append(is_front)
    return front

#@time_deco
def is_pareto_front(costs: np.ndarray | Sequence[np.ndarray]):
    if fast_pareto is not None: return fast_pareto.is_pareto_front(np.array(costs, copy=False))
    else: 
        warnings.warn('`fast_pareto` is not installed, pareto front computation may be very slow. https://github.com/nabenabe0928/fast-pareto')
        return _slow_is_pareto_front(costs)

def _compare_pareto(new: float | np.ndarray, current: "float | np.ndarray") -> "tuple[float | np.ndarray, bool]":
    """Return best value / values, and  whether `new` is better / in pareto front. True if it is, False if it isn't."""
    # value is a scalar
    if isinstance(new, float):
        if new > current: return current, False
        return new, True
    # multiple values, use pareto front
    else:
        # initial value
        current = np.array(current)
        if current.size == 1: return np.expand_dims(new, 0), True
        candidates = np.concatenate((np.expand_dims(new, 0), current)) # num x candidates
        isfront = is_pareto_front(candidates)
        if isfront[0]: return candidates[isfront], True
        return current, False

