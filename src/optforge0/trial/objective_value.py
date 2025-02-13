from collections.abc import Callable

import numpy as np

from .._types import Numeric

__all__ = [
    "ObjectiveValue",
]

class ObjectiveValue:
    """This holds the pure objective value of the function, as well as all constraint violations"""
    # having to manage all different combinations of violations and using indicator function was very messy
    # so this is now done in a separate class
    def __init__(
        self,
        original_value: float | np.ndarray,
        param_violation: float,
        soft_violation: float,
        hard_violation: float,
        mo_indicator: Callable[[np.ndarray], Numeric],
        applier: Callable[[float | np.ndarray, float], float | np.ndarray],
    ):
        self.original_value = original_value
        """Real space value with constrained parameters and no penalties."""
        self.param_violation = param_violation
        self.soft_violation = soft_violation
        self.hard_violation = hard_violation
        self.mo_indicator = mo_indicator
        self.applier = applier

        # precomputing
        self._original_scalar_value = None

        self.opt_value = self.get()
        """Optimization space value with uncontrained parameters and all penalties applied."""
        self.opt_scalar_value: float = self.get(scalar = True) # type:ignore
        """Optimization space scalar value with uncontrained parameters and all penalties applied"""
        self.real_value = self.get(param_penalty=False)
        """Real space value with constrained parameters and all penalties except parameter penalties."""
        self.real_scalar_value: float = self.get(param_penalty=False, scalar=True) # type:ignore
        """Real space scalar value with constrained parameters and all penalties except parameter penalties."""
        self.is_viable = self.hard_violation == 0
        """Whether this value is viable, i.e. no hard constraints violated."""

    @property
    def original_scalar_value(self) -> float:
        if self._original_scalar_value is None:
            if isinstance(self.original_value, np.ndarray): self._original_scalar_value = float(self.mo_indicator(self.original_value))
            else: self._original_scalar_value = self.original_value
        return self._original_scalar_value


    def get(self, soft_penalty:bool = True, hard_penalty:bool = True, param_penalty:bool = True, scalar:bool = False):
        total_penalty = 0
        if soft_penalty: total_penalty += self.soft_violation
        if hard_penalty: total_penalty += self.hard_violation
        if param_penalty: total_penalty += self.param_violation

        if scalar: value = self.original_scalar_value
        else: value = self.original_value
        if total_penalty > 0: return self.applier(value, total_penalty, )
        return value

    def __repr__(self):
        return f'ObjectiveValue(original={self.original_value}, final={self.opt_value})'
