from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Optional, overload

import numpy as np

from .._types import ArrayLike, Numeric, NumericArrayLike
from .._utils import _ensure_float_or_1darray
from ..inits import InitLiteral
from ..param.base_param import DomainLiteral, Param
from ..param.params import (_categorical_update, _permutation_update,
                            categorical, numeric, permutation)
from ..paramdict import ParamDict
from ..python_tools import Compose
from .constraints import _constr_bounds
from .evaluate import _evaluate_trial, EndTrial
from .finished_trial import FinishedTrial
from .objective_value import ObjectiveValue
from .param_constraints import (_multi_param_l1, _param_constr_bounds,
                                _param_constr_set, _single_param_l1)
from ..pruners.base_pruner import Pruner

if TYPE_CHECKING:
    from ..study import Study

nan = float('nan')

__all__ = [
    "Trial",
    "FixedTrial",
    "EndTrial",
]


class Trial:
    value: float | np.ndarray
    """This trial's optimization space objective value with uncontrained parameters and all penalties applied."""
    scalar_value: float
    """This trial's optimization space scalar objective value with uncontrained parameters and all penalties applied."""
    best_value: float | np.ndarray
    """Best optimization space objective value so far."""
    best_scalar_value: float | np.ndarray
    """Best optimization space scalar objective value so far."""
    objective_value: ObjectiveValue
    original_value: float | np.ndarray
    total_param_violation: float
    total_soft_violation: float
    total_hard_violation: float
    soft_violations: np.ndarray
    hard_violations: np.ndarray
    is_viable: bool
    finished_trial: FinishedTrial
    def __init__(self, paramdict:"ParamDict", study:"Study", asked: bool, transform: Optional[Callable[[ParamDict], ParamDict]] = None,): # pylint:disable = W0231
        self.objective = study.objective

        self.original_params:dict[str, Any] = {}
        """Original parameters."""
        self.params: ParamDict = paramdict if transform is None else transform(paramdict)
        """Optimization space paramdict."""
        self.logs: dict[Any, Any] = {}

        self.asked = asked
        """True if this trial was asked by optimizer, False if it is suggested by the user."""

        self.log_params = study.log_params
        self.log_paramdict = study.optimizer.STORE_PARAMDICTS == 'all'

        self.storage = {}

        self.current_step = study.current_step
        self.mo_indicator = study.mo_indicator
        self.soft_handler = study.soft_handler
        self.hard_handler = study.hard_handler
        self.applier = study.applier

        self.pruner: Pruner | None = study.pruner

        self.evaluated = False
        self.improved = False

        self._param_violations: dict[str, float] = {}
        self._soft_violations: dict[str, float | np.ndarray] = {}
        self._hard_violations: dict[str, float | np.ndarray] = {}

    def __call__(self) -> "Trial":
        """Evaluates this trial, creates a `finished_trial` attribute and returns itself."""
        return _evaluate_trial(self)

    def get_value(self, soft_penalty:bool = True, hard_penalty:bool = True, param_penalty:bool = True, scalar:bool = False):
        return self.objective_value.get(
            soft_penalty=soft_penalty,
            hard_penalty=hard_penalty,
            param_penalty=param_penalty,
            scalar=scalar,
        )

    def log(self, key, value):
        self.logs[key] = value

    def report_intermediate(self, step: int, value:Numeric | NumericArrayLike):
        value = _ensure_float_or_1darray(value)
        self.original_value = value
        if isinstance(value, np.ndarray): value = self.mo_indicator(value)
        if self.pruner is not None:
            self.pruner._internal_report(step, value)

    def should_prune(self) -> bool:
        if self.pruner is None:
            return False
        return self.pruner.should_prune()

    def stop(self): raise EndTrial()

    def suggest_array(
        self,
        name: str,
        shape: Optional[int | Sequence[int]] = None,

        low: Optional[float] = None,
        high: Optional[float] = None,
        step: Optional[float] = None,
        domain: DomainLiteral = 'linear',
        scale: Optional[float] = None,
        normalize: bool = True,

        discrete_step: bool = True,
        ordered: bool = True,

        oob_penalty: Optional[float] = 1.,
        init: "InitLiteral | Callable | np.ndarray | Numeric | Sequence[Numeric]" = "mean",

        fallback_low: float = -1,
        fallback_high: float = 1,

        options: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        # first we construct attributes that define the param
        # making it a dict instead of using keyword arguments
        # makes passing and copying it much easier and faster

        # if param already exists, update its attributes
        if name in self.params:
            param: Param = self.params[name] # type:ignore
            param._update(
                low = low,
                high = high,
                step = step,
                domain = domain,
                normalize = normalize,
                scale = scale,
                oob_penalty = oob_penalty,
                fallback_low=fallback_low,
                fallback_high=fallback_high,
            )
        # else if it doesn't exist, create it
        else:
            param = self.params[name] = numeric(
                shape = shape,
                low = low,
                high = high,
                step = step,
                domain = domain,
                init = init,
                normalize = normalize,
                discrete_step=discrete_step,
                ordered = ordered,
                scale = scale,
                oob_penalty = oob_penalty,
                fallback_low=fallback_low,
                fallback_high=fallback_high,
                options = options,
            )

        # calculate the value of the parameter and save to `self.params`.
        # some methods like `suggest_float` call this method,
        # but convert the value to a float or something else before saving it,
        # so they skip this line with `_set_value = False` and do it themselves.
        self.original_params[name] = value = param()

        # Add parameter to used parameters since it was accessed
        # unused parameters won't be optimized if optimizer supports that
        self.params.used_params.add(name)

        return value

    def suggest_float(
        self,
        name: str,

        low: Optional[float] = None,
        high: Optional[float] = None,
        step: Optional[float] = None,
        domain: DomainLiteral = 'linear',
        scale: Optional[float] = None,
        normalize: bool = True,

        discrete_step: bool = True,
        ordered: bool = True,

        oob_penalty: Optional[float] = 1.,
        init: "InitLiteral | Callable | np.ndarray | Numeric | Sequence[Numeric]" = "mean",

        fallback_low: float = -1,
        fallback_high: float = 1,

        options: Optional[dict[str, Any]] = None,
    ) -> float:
        value = float(self.suggest_array(
            name = name,
            shape = 1,
            low = low,
            high = high,
            domain = domain,
            step = step,
            scale = scale,
            init = init,
            normalize=normalize,
            discrete_step = discrete_step,
            ordered=ordered,
            oob_penalty = oob_penalty,
            fallback_low = fallback_low,
            fallback_high = fallback_high,
            options = options,
        ))
        self.original_params[name] = value
        return value

    def suggest_int(
        self,
        name: str,

        low: Optional[float] = None,
        high: Optional[float] = None,
        step: Optional[float] = None,
        domain: DomainLiteral = 'linear',
        scale: Optional[float] = None,
        normalize: bool = True,

        discrete_step: bool = True,
        ordered: bool = True,

        oob_penalty: Optional[float] = 1.,
        init: "InitLiteral | Callable | np.ndarray | Numeric | Sequence[Numeric]" = "mean",

        fallback_low: float = -1,
        fallback_high: float = 1,

        options: Optional[dict[str, Any]] = None,
    ) -> int:
        value = int(self.suggest_array(
            name = name,
            shape = 1,
            low = low,
            high = high,
            domain = domain,
            step = step,
            scale = scale,
            init = init,
            normalize = normalize,
            discrete_step = discrete_step,
            ordered=ordered,
            oob_penalty = oob_penalty,
            fallback_low = fallback_low,
            fallback_high = fallback_high,
            options = options,
        ))
        self.original_params[name] = value
        return value

    def suggest_categorical_array(
        self,
        name: str,
        shape: Optional[int | Sequence[int]],
        choices: int | ArrayLike,
        scale: "Optional[float]" = None,
        # normalization is False for categorical params. When it is True, since search space is normalized to (-1,1) range,
        # the more choices there are, the smaller is the search space magnitude per parameter, which doesn't really make sense.
        normalize: bool = False,

        one_hot = False,

        oob_penalty: Optional[float] = 1,
        init: "InitLiteral | Callable | np.ndarray | Numeric | Sequence[Numeric]" = "mean",
        options: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        # first we construct attributes that define the param
        # making it a dict instead of using keyword arguments
        # makes passing and copying it much easier and faster

        # if param already exists, update its attributes
        if name in self.params:
            param: Param = self.params[name]  # type:ignore
            _categorical_update(
                param,
                choices=choices,
                scale=scale,
                oob_penalty=oob_penalty,
                normalize=normalize,
                one_hot=one_hot,
            )
        # else if it doesn't exist, create it
        else:
            param = self.params[name] = categorical(
                shape = shape,
                choices = choices,
                scale = scale,
                normalize = normalize,
                one_hot=one_hot,
                oob_penalty = oob_penalty,
                init = init,
                options = options,
            )

        # calculate the value of the parameter and save to `self.params`.
        # some methods like `suggest_float` call this method,
        # but convert the value to a float or something else before saving it,
        # so they skip this line with `_set_value = False` and do it themselves.
        self.original_params[name] = value = param()

        # Add parameter to used parameters since it was accessed
        # unused parameters won't be optimized if optimizer supports that
        self.params.used_params.add(name)

        return value

    def suggest_categorical(
        self,
        name: str,
        choices: int | ArrayLike,
        scale: "Optional[float]" = None,
        # normalization is False for categorical params. When it is True, since search space is normalized to (-1,1) range,
        # the more choices there are, the smaller is the search space magnitude per parameter, which doesn't really make sense.
        normalize: bool = False,

        one_hot = False,

        oob_penalty: Optional[float] = 1,
        init: "InitLiteral | Callable | np.ndarray | Numeric | Sequence[Numeric]" = "mean",
        options: Optional[dict[str, Any]] = None,
    ) -> Any:
        value = self.suggest_categorical_array(
            name = name,
            choices = choices,
            shape = 1,
            one_hot = one_hot,
            scale = scale,
            init = init,
            normalize = normalize,
            oob_penalty = oob_penalty,
            options = options,
        )[0]
        self.original_params[name] = value
        return value

    def suggest_permutation_array(
        self,
        name: str,
        shape: Optional[int | Sequence[int]],
        choices: int | ArrayLike,
        scale: "Optional[float]" = None,
        normalize: bool = False,

        type: Literal['argsort'] = 'argsort',

        oob_penalty: Optional[float] = 1,
        init: "InitLiteral | Callable | np.ndarray | Numeric | Sequence[Numeric]" = 'mean',
        options: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        # first we construct attributes that define the param
        # making it a dict instead of using keyword arguments
        # makes passing and copying it much easier and faster

        # if param already exists, update its attributes
        if name in self.params:
            param: Param = self.params[name]  # type:ignore
            _permutation_update(
                param,
                choices=choices,
                scale=scale,
                oob_penalty=oob_penalty,
                normalize=normalize,
                type=type,
            )
        # else if it doesn't exist, create it
        else:
            param = self.params[name] = permutation(
                shape = shape,
                choices = choices,
                scale = scale,
                normalize = normalize,
                type=type,
                oob_penalty = oob_penalty,
                init = init,
                options = options,
            )

        # calculate the value of the parameter and save to `self.params`.
        # some methods like `suggest_float` call this method,
        # but convert the value to a float or something else before saving it,
        # so they skip this line with `_set_value = False` and do it themselves.
        self.original_params[name] = value = param()

        # Add parameter to used parameters since it was accessed
        # unused parameters won't be optimized if optimizer supports that
        self.params.used_params.add(name)

        return value

    def suggest_permutation(
        self,
        name: str,
        choices: int | ArrayLike,
        scale: "Optional[float]" = None,
        normalize: bool = False,

        type: Literal['argsort'] = 'argsort',

        oob_penalty: Optional[float] = 1,
        init: "InitLiteral | Callable | np.ndarray | Numeric | Sequence[Numeric]" = 'mean',
        options: Optional[dict[str, Any]] = None,
    ) -> Any:
        value = self.suggest_permutation_array(
            name = name,
            choices = choices,
            shape = 1,
            type = type,
            scale = scale,
            init = init,
            normalize = normalize,
            oob_penalty = oob_penalty,
            options = options,
        )[0]
        self.original_params[name] = value
        return value

    def _suggest_param(self, name: str, param:Param):
        if name in self.params: param = self.params[name]
        else: self.params[name] = param
        self.original_params[name] = value = param()
        self.params.used_params.add(name)
        return value

    def suggest_constant(self, name, init, *args, **kwargs):
        self.original_params[name] = init
        return init

    def _create_original_params(self):
        for name, p in self.params.items(): self.original_params[name] = p()
        self.used_params = set(self.params.keys())

    @classmethod
    def _transformed(cls, transform: Callable[[ParamDict], ParamDict]):
        return _TransformedTrialPartial(cls, transform)

    def param_constr_bounds(
        self,
        param_name: str,
        low: Numeric | NumericArrayLike | None = None,
        high: Numeric | NumericArrayLike | None = None,
        inclusive=True,
        eps = 0.,
        weight=1.0,
        constr_name: str | None = None,
    ) -> float | np.ndarray:
        return _param_constr_bounds(
            self,
            param_name=param_name,
            low=low,
            high=high,
            inclusive=inclusive,
            eps=eps,
            weight=weight,
            constr_name=constr_name,
        )

    @overload
    def param_constr_l1(
        self,
        param_names: str,
        eq: Numeric | NumericArrayLike | None,
        axis: int | Sequence[int] | None = None,
        low: float | None = None,
        weight: float = 0.0,
        constr_name: str | None = None,
    ) -> float | np.ndarray: ...
    @overload
    def param_constr_l1(
        self,
        param_names: list[str] | tuple[str],
        eq: Numeric | NumericArrayLike | None,
        axis: int | Sequence[int] | None = None,
        low: float | None = None,
        weight: float = 0.0,
        constr_name: str | None = None,
    ) -> list[float | np.ndarray]: ...
    def param_constr_l1(
        self,
        param_names: str | list[str] | tuple[str],
        eq: Numeric | NumericArrayLike | None,
        axis: int | Sequence[int] | None = None,
        low: float | None = None,
        weight: float = 0.0,
        constr_name: str | None = None,
    ) -> float | np.ndarray | list[float | np.ndarray]:
        if isinstance(param_names, str): fn = _single_param_l1
        else: fn = _multi_param_l1
        return fn(
            self,
            param_names = param_names, # type:ignore
            eq = eq,
            axis = axis,
            low = low,
            weight=weight,
            constr_name=constr_name,
        )

    def param_constr_set(self, param_name:str, eq: Numeric | NumericArrayLike | None, weight = 0, constr_name = None):
        return _param_constr_set(self, param_name=param_name, eq=eq, weight=weight, constr_name=constr_name)

    def soft_penalty(
        self,
        value: Numeric | NumericArrayLike,
        constr_name: str | None = None
    ):
        if constr_name is None: constr_name = f'unnamed{len(self._soft_violations)}'
        if constr_name in self._soft_violations: self._soft_violations[constr_name] += _ensure_float_or_1darray(value)
        else: self._soft_violations[constr_name] = _ensure_float_or_1darray(value)

    def hard_penalty(
        self,
        value: Numeric | NumericArrayLike,
        constr_name: str | None = None
    ):
        if constr_name is None: constr_name = f'unnamed{len(self._hard_violations)}'
        if constr_name in self._hard_violations: self._hard_violations[constr_name] += _ensure_float_or_1darray(value)
        else: self._hard_violations[constr_name] = _ensure_float_or_1darray(value)


    def constr_bounds(
        self,
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
        return _constr_bounds(
            self = self,
            value = value,
            low = low,
            high = high,
            inclusive = inclusive,
            hard = hard,
            reduce = reduce,
            eps = eps,
            weight = weight,
            constr_name = constr_name,
        )

    @classmethod
    def _fixed(cls, value: float | np.ndarray, paramdict: ParamDict, study: "Study", asked:bool) -> "_TellNotAskedTrial":
        return _TellNotAskedTrial(value = value, paramdict = paramdict, study = study, asked = asked)


class _TransformedTrialPartial:
    def __init__(self, cls: "type[Trial] | _TransformedTrialPartial", transform: Callable[[ParamDict], ParamDict]):
        self.cls = cls
        self.transform = transform

    def __call__(
        self,
        paramdict: "ParamDict",
        study: "Study",
        asked: bool,
        transform: Optional[Callable[[ParamDict], ParamDict]] = None,
    ):  # pylint:disable = W0231
        if transform is None: transform = self.transform
        else: transform = Compose(self.transform, transform)
        return self.cls(paramdict=paramdict, study=study, asked = asked, transform=transform)

    def _transformed(self, transform: Callable[[ParamDict], ParamDict]):
        return _TransformedTrialPartial(self, transform)

    def _fixed(self, value: float | np.ndarray, paramdict: ParamDict, study: "Study", asked: bool) -> "_TellNotAskedTrial":
        return _TellNotAskedTrial(value = value, paramdict = paramdict, study = study, asked = asked, transform=self.transform,)
class _AlwaysReturnX:
    def __init__(self, x): self.x = x
    def __call__(self, trial: Trial): return self.x

class _TellNotAskedTrial(Trial):
    def __init__(
        self,
        value: float | np.ndarray,
        paramdict: "ParamDict",
        study: "Study",
        asked: bool,
        transform: Optional[Callable[[ParamDict], ParamDict]] = None,
    ):  # pylint:disable = W0231
        super().__init__(paramdict = paramdict, study = study, asked=asked, transform = transform)
        self._value = value
        self.objective = _AlwaysReturnX(self._value)
        self._create_original_params()


class FixedTrial(Trial):
    def __init__(
        self,
        params: dict[str, Any],
    ):  # pylint:disable = W0231
        from ..study import Study
        study = Study()
        study.objective = self._objective
        super().__init__(ParamDict(), study, asked=False)
        self.original_params = params

    def _objective(self, trial: "Trial"): return 0
    def _sg(self, name, *args, **kwargs):
        return self.original_params[name]

    def __getattribute__(self, item):
        if 'suggest' in item: return self._sg
        return super(FixedTrial, self).__getattribute__(item)



class VectorizedObjectiveTrialPack:
    def __init__(self, trials: list[Trial]):
        self.trials = trials
        self.objective = trials[0].objective

    def suggest_array(self, name, *args, **kwargs):
        return np.array([t.suggest_array(name, *args, **kwargs) for t in self.trials], copy=False)

    # def __getattr__(self, name):
    #     if name.startswith('suggest_'): raise ValueError("Only `suggest_array` is allowed for vectorized trials.")
    #     return getattr(self.trials[0], name)

    def __call__(self) -> "list[Trial]":
        for t in self.trials:
            for p in t.params.values(): p.oob_penalty = 0.
        values: float | np.ndarray = np.array(self.objective(self), copy = False) # type:ignore

        for trial, value in zip(self.trials, values):
            trial.objective = lambda trial: value # pylint:disable = W0640
            trial()

        return self.trials
