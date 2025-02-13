from typing import Literal, TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np
if TYPE_CHECKING:
    from .paramdict import ParamDict

def _default_constraint_reduce(penalties: np.ndarray) -> float:
    return float(penalties.sum())

def _default_constraint_apply(objective_value: float | np.ndarray, penalty_value:float) -> float | np.ndarray:
    return (objective_value + penalty_value) * (1 + penalty_value)
