from typing import TYPE_CHECKING

import numpy as np

from .._utils import _ensure_float_or_ndarray, _ensure_float_or_1darray
from ..python_tools import reduce_dim
from .finished_trial import FinishedTrial
from .objective_value import ObjectiveValue

if TYPE_CHECKING:
    from .trial import Trial
    
class EndTrial(Exception): pass

def _evaluate_trial(self: "Trial") -> "Trial":
    """Evaluates this trial, creates a `finished_trial` attribute and returns itself."""
    if self.evaluated: raise ValueError("Trial has already been evaluated.")
    # set bounds violations to 0
    for p in self.params.values(): p.oob_violation = 0.

    # evaluate the objective
    try: self.original_value = _ensure_float_or_1darray(self.objective(self))
    # EndTrial may be raised with pruning, in which case there should already be an intermediate `original_value` attribute
    except EndTrial: pass

    # sum out of bounds violations and add to param bounds
    self._param_violations['oob'] = np.sum([p.oob_violation for p in self.params.values()])

    # param violations
    self.total_param_violation = float(np.sum(list(self._param_violations.values())))

    # soft violations
    if len(self._soft_violations) > 0:
        self.soft_violations: np.ndarray = np.array(reduce_dim([np.array(i, copy=False).flatten() for i in self._soft_violations.values()])) # type:ignore
        self.total_soft_violation = self.soft_handler(self.soft_violations)
        # self.original_value = self.applier(self.original_unconstrained_value, self.total_soft_violation)
    else:
        self.soft_violations = np.array([0])
        self.total_soft_violation = 0
        # self.original_value = self.original_unconstrained_value

    # hard violations
    if len(self._hard_violations) > 0:
        self.hard_violations: np.ndarray = np.array(reduce_dim([np.array(i, copy=False).flatten() for i in self._hard_violations.values()])) # type:ignore
        self.total_hard_violation = self.hard_handler(self.hard_violations)
    else:
        self.hard_violations = np.array([0])
        self.total_hard_violation = 0


    self.objective_value = ObjectiveValue(
        original_value = self.original_value,
        param_violation = self.total_param_violation,
        soft_violation = self.total_soft_violation,
        hard_violation = self.total_hard_violation,
        mo_indicator = self.mo_indicator,
        applier = self.applier,
    )

    self.value = self.objective_value.opt_value
    self.scalar_value = self.objective_value.opt_scalar_value
    self.is_viable = self.objective_value.is_viable

    # construct finished trial
    self.finished_trial = FinishedTrial(
        params = self.original_params if self.log_params else {},
        paramdict = self.params.copy() if self.log_paramdict else None,
        objective_value = self.objective_value,
        # don't store violations if not log_params because there may be a lot of them (like 100,000)
        param_violations = self._param_violations if self.log_params else {},
        soft_violations = self._soft_violations if self.log_params else {},
        hard_violations = self._hard_violations if self.log_params else {},
        logs = self.logs,
        #current_eval = trial.current_eval,
        current_step = self.current_step,
    )

    self.evaluated = True
    return self