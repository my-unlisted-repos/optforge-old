import time
from typing import Any, Optional

import numpy as np

from ..constraint_handler import (_default_constraint_apply,
                                  _default_constraint_reduce)
from ..paramdict import ParamDict
from .objective_value import ObjectiveValue
__all__ = [
    "FinishedTrial",
]

class FinishedTrial:
    current_eval: int
    best_value: float | np.ndarray
    time_passed: float
    improved:bool
    def __init__(
        self,
        params: dict[str, Any],
        paramdict: Optional[ParamDict],
        objective_value: ObjectiveValue,
        param_violations: dict[str, float],
        soft_violations: dict[str, float | np.ndarray],
        hard_violations: dict[str, float | np.ndarray],
        logs: dict[Any, Any],
        #current_eval: int,
        current_step: int,
    ):
        self.params: dict[str, Any] = params
        self.paramdict = paramdict
        self.objective_value = objective_value
        self.param_violations = param_violations
        self.soft_violations = soft_violations
        self.hard_violations = hard_violations
        self.total_param_violation = objective_value.param_violation
        self.total_soft_violation = objective_value.soft_violation
        self.total_hard_violation = objective_value.hard_violation
        self.logs: dict[Any, Any] = logs
        self.current_step: int = current_step

        self.value = objective_value.real_value
        """Real space value with constrained parameters and all penalties except parameter penalties."""
        self.original_value = objective_value.original_value
        """Real space value with constrained parameters and no penalties."""
        self.scalar_value = objective_value.real_scalar_value
        self.end_time = time.time()

        #self.current_eval: int = current_eval
        self.is_viable = self.objective_value.is_viable

    def shallow_copy(self):
        # used in ModifiedStudy
        f = FinishedTrial(
            params = self.params,
            paramdict = self.paramdict,
            objective_value = self.objective_value,
            param_violations = self.param_violations,
            soft_violations = self.soft_violations,
            hard_violations = self.hard_violations,
            logs = self.logs,
            #current_eval = self.current_eval,
            current_step = self.current_step,
        )
        f.end_time = self.end_time
        if hasattr(self, 'current_eval'): f.current_eval = self.current_eval # pylint:disable = E1101
        if hasattr(self, 'best_value'): f.best_value = self.best_value # pylint:disable = E1101
        if hasattr(self, 'time_passed'): f.time_passed = self.time_passed # pylint:disable = E1101
        if hasattr(self, 'improved'): f.improved = self.improved # pylint:disable = E1101
        return f

# initial trial for study.best_trial
_init_trial = FinishedTrial(
    params = {},
    logs = {},
    objective_value = ObjectiveValue(np.inf, 0, 0, 0, _default_constraint_reduce, _default_constraint_apply),
    paramdict = None,
    param_violations = {},
    soft_violations = {},
    hard_violations = {},
    current_step = 0,
)
_init_trial.current_eval = 0
_init_trial.best_value = float('inf')
_init_trial.time_passed = 0
_init_trial.improved = False