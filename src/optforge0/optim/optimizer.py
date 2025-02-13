from abc import ABC, abstractmethod
from collections.abc import (Callable, Generator, Mapping, MutableMapping,
                             Sequence)
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from ..exceptions import AskNotSupportedError
from ..param import Param
from ..paramdict import ParamDict, Store
from ..registry.groups import GROUPS
from ..rng import RNG
from ..scheduler.scheduler import (Scheduler, _ConfiguredScheduler,
                                   _SchedulerCaller)
from ..trial.finished_trial import FinishedTrial
from ..trial.trial import Trial, _TransformedTrialPartial
from ._utils import _register, _set_name
from .config import Config, ConfigLiterals

if TYPE_CHECKING:
    from ..study import Study


__all__ = [
    "Optimizer",
    "Minimizer",
    "Store",
    "Config",
]

inf = float('inf')



class Optimizer(ABC):
    trial_cls: type[Trial] | _TransformedTrialPartial = Trial
    CONFIG: MutableMapping[ConfigLiterals, Any] = {
        "supports_ask": True,
        "supports_multiple_asks": True,
        "requires_batch_mode": True,
        "store_paramdicts": 'none',
    }

    # default values
    wrapped_constructor: type | Callable | Any | None = None
    wrapped_optimizer: Any | None = None
    wrapped_kwargs: dict[str, Any] | None = None
    budget = None
    callbacks: "Optional[list[Callable[[Study, Trial, FinishedTrial], None]]]" = None
    names: Sequence[str]
    lib: str | None = None

    def __init__(
        self,
        defaults: Optional[dict[str, Any]] = None,
        seed = None,
    ):
        self.defaults:dict[str, Any] = defaults if defaults is not None else {}
        """Default optimizer settings, can be overwritten by per-param settings."""

        for k,v in self.defaults.items():
            if isinstance(v, _ConfiguredScheduler): self.defaults[k] = v()

        self.saved_params = {}
        """Stores copies of params."""

        self.params = ParamDict()

        self.CONFIG = dict(self.CONFIG).copy()

        self._schedulers: set[Scheduler] = set()
        for v in self.defaults.values():
            if isinstance(v, Scheduler): self._schedulers.add(v)

        self.children: list[Optimizer] = []

        if hasattr(self, 'names'):
            if isinstance(self.names, str): self.names = [self.names, ]
            else: self.names = list(self.names)

        self.rng = RNG(seed)

    # ------------------------------- main methods ------------------------------- #
    def ask(self, study:"Study") -> "Generator[ParamDict]":
        if self.SUPPORTS_ASK:
            raise ValueError(f'{self.__class__.__name__} has to either have `ask` method or SUPPORTS_ASK = False')
            yield # to silence pylint on def #pylint:disable = W0101
        else: raise AskNotSupportedError(f"{self.__class__.__name__} doesn't support ask and tell interface")

    def tell(self, trials: list[Trial], study: "Study") -> None:
        # default tell sets params if they are better
        for t in trials:
            if t.is_viable and t.improved: self.params.update(t.params)

    def tell_not_asked(self, trial: Trial, study: "Study"):
        self.tell([trial], study)

    def step(self, study: "Study"):
        # by default step just uses ask and tell.
        # but optimizers can overwrite it.

        # if step is used, no concurrency is used, so evals are just converted into a list
        trials = list(self._internal_ask(study))
        # evaluate each trial, which sets `params`, `value` and `finished_trial`
        # and submit it to the study
        for t in trials:
            study._submit_evaluated_trial(t())

        self._internal_tell(trials, study)

    # ------------------------------- param methods ------------------------------ #
    def set_params(self, params: ParamDict):
        """Sets this optimizers params to `params` object, overwriting existing object.
        If this method is overwritten it should usually call `super().set_params(params)`."""
        self.params = params
        self.params._set_rng(self.rng)
        if self.wrapped_constructor is not None:
            if self.wrapped_kwargs is None: self.wrapped_kwargs = {}
            self.wrap_constructor(self.wrapped_constructor, **self.wrapped_kwargs)
        # for c in self.children: c.set_params(params)
        return self

    def make_wrap_args(self) -> tuple[Sequence, Mapping]:
        """Must return (args, kwargs) that can be passed to a wrapped optimizer class."""
        return ((), {})

    def wrap(self, optimizer:Any):
        self.wrapped_optimizer = optimizer

    def set_budget(self, budget: Any) -> "Optimizer":
        self.budget = budget
        for s in self._schedulers: s.budget = budget
        # for c in self.children: c.set_budget(budget)
        return self

    def set_seed(self, seed: Optional[int]):
        self.rng = RNG(seed)
        self.params._set_rng(self.rng)
        for s in self._schedulers: s.rng = self.rng
        # for c in self.children: c.set_seed(seed)
        return self

    # --------------------------------- Modifiers -------------------------------- #
    def rescaled(self, scale: float):
        from .mods import RescaledOptimizer
        return RescaledOptimizer(self, scale = scale)

    @classmethod # type:ignore
    def _create_and_rescale[**P](cls: Callable[P, "Optimizer"], /, scale, *args: P.args, **kwargs: P.kwargs,): # pylint:disable = E0602
        return cls(*args, **kwargs).rescaled(scale)

    @classmethod # type:ignore
    def conf_rescaled[**P](cls: Callable[P, "Optimizer"], /, scale, *args: P.args, **kwargs: P.kwargs,): # pylint:disable = E0602
        return ConfiguredOptimizer(cls._create_and_rescale, scale=scale, *args, **kwargs) # type:ignore

    def unordered(self):
        from .mods import UnorderedOptimizer
        return UnorderedOptimizer(self)

    @classmethod # type:ignore
    def _create_unordered[**P](cls: Callable[P, "Optimizer"], /, *args: P.args, **kwargs: P.kwargs,): # pylint:disable = E0602
        return cls(*args, **kwargs).unordered()

    @classmethod # type:ignore
    def conf_unordered[**P](cls: Callable[P, "Optimizer"], /, *args: P.args, **kwargs: P.kwargs,): # pylint:disable = E0602
        return ConfiguredOptimizer(cls._create_unordered, *args, **kwargs) # type:ignore
    # --------------------------------- overloads -------------------------------- #

    @property
    def name(self) -> str:
        """Base name without lib prefix"""
        return self.names[0] if hasattr(self, 'names') else self.__class__.__name__

    @property
    def lname(self) -> str:
        """lib.name"""
        lib = 'optforge' if self.lib is None else self.lib
        return f'{lib}.{self.name}'

    def __str__(self):
        """lib.name"""
        return self.lname


    # ----------------- methods that actually get called by study ---------------- #
    #                   also some optimizer wrappers change them                   #
    def _internal_ask(self, study:"Study") -> Generator[Trial]:
        if self.budget is None: self.set_budget(study.max_evals)

        params = self.ask(study)
        if params is not None:
            for p in params: yield self.trial_cls(p, study, asked=True)

    def _internal_tell(self, trials: list[Trial], study: "Study"):
        self.tell(trials, study)

        # -- moved to Study --
        # if len(trials) > 0:
        #     self._scheduler_step(study)
        #     self._done_scheduler_step = True

    def _internal_step(self, study: "Study"):
        # this can be overwritten which is useful for meta optimizers
        if self.budget is None: self.set_budget(study.max_evals)

        # both `step` and `tell` methods perform a scheduler step.
        # since `step` by default calls `tell`, we make sure it doesn't step twice.

        self.step(study)

        # if not self._done_scheduler_step:
        #     self._scheduler_step(study)
        #     self._done_scheduler_step = False

        return study.last_trial.value

    def _tell_not_asked_paramdict(self, paramdict: ParamDict, value, study: "Study"):
        evaluated_trial = self.trial_cls._fixed(value = value, paramdict=paramdict, study=study, asked=False)()
        study._submit_evaluated_trial(evaluated_trial)
        self.tell_not_asked(trial = evaluated_trial, study = study)

    def _tell_not_asked_orig_params(self, params: dict[str, Any], value, study: "Study"):
        new_params = self.params.copy()
        new_params.value_dict_to_params_(params)
        self._tell_not_asked_paramdict(paramdict=new_params, value=value, study=study)

    # --------------- methods that generally shouldn't be overwritten -------------- #
    def yield_params(self, custom_params:"Optional[ParamDict]" = None, only_used:bool = False) -> "Generator[Param]":
        params = self.params if custom_params is None else custom_params
        yield from params.yield_params(only_used)

    def yield_stores(self, custom_params:"Optional[ParamDict]" = None, only_used:bool = False) -> "Generator[Store]":
        params = self.params if custom_params is None else custom_params
        yield from params.yield_stores(self.defaults, only_used)

    def yield_params_stores(self, custom_params:"Optional[ParamDict]" = None, only_used:bool = False) -> "Generator[tuple[Param, Store]]":
        params = self.params if custom_params is None else custom_params
        yield from params.yield_params_stores(self.defaults, only_used)

    def yield_names_params(self, custom_params:"Optional[ParamDict]" = None, only_used:bool = False) -> "Generator[tuple[str, Param]]":
        params = self.params if custom_params is None else custom_params
        yield from params.yield_names_params(only_used)

    def yield_names_stores(self, custom_params:"Optional[ParamDict]" = None, only_used:bool = False) -> "Generator[tuple[str, Store]]":
        params = self.params if custom_params is None else custom_params
        yield from params.yield_names_stores(self.defaults, only_used)

    def yield_names_params_stores(self, custom_params:"Optional[ParamDict]" = None, only_used:bool = False) -> "Generator[tuple[str, Param, Store]]":
        params = self.params if custom_params is None else custom_params
        yield from params.yield_names_params_stores(self.defaults, only_used)

    def save_param_data(self, key = None):
        self.saved_params[key] = self.params.copy_data()

    def load_param_data(self, key = None, delete = False):
        self.params.set_data_(self.saved_params[key])
        if delete: del self.saved_params[key]

    def wrap_constructor(self, constructor:Callable, **extra_kwargs):
        """Combines `make_wrap_args` and `wrap`"""
        args, kwargs = self.make_wrap_args()
        self.wrap(constructor(*args, **kwargs, **extra_kwargs))

    def create_config(self) -> "Config": return Config.from_dict(self.CONFIG) # type:ignore

    def schedule(self, value:Any | Scheduler) -> _SchedulerCaller:
        if isinstance(value, _ConfiguredScheduler): value = value()
        if isinstance(value, Scheduler):
            value.rng = self.rng
            self._schedulers.add(value)
        return _SchedulerCaller(value)

    def _scheduler_step(self, study):
        for p in self._schedulers: p._internal_step(study)
        # for c in self.children: c._scheduler_step(study)

    @classmethod
    def register(
        cls,
        lib: Optional[str] = None,
        groups: str | Sequence[str] = (GROUPS.MAIN, ),
        maxdims: Optional[int] = None,
    ):
        """Register this optimizer to the optimizers registry.
        You can iterate over every single optimizer with `registry.keys()` or `registry.values()`.
        Registry holds classes or constructors, not instances. For consistency all registered constructors
        require no arguments, so that they can be iterated over effortlesly."""
        return _register(
            cls = cls,
            lib = lib,
            groups = groups,
            maxdims = maxdims,
        )

    @classmethod
    def set_name(
        cls,
        names: str | Sequence[str],
        register: bool = False,
        lib: Optional[str] = None,
        groups: str | Sequence[str] = (GROUPS.MAIN, ),
        maxdims: Optional[int] = None,
    ):
        return _set_name(
            cls = cls,
            names = names,
            register = register,
            lib = lib,
            groups = groups,
            maxdims = maxdims,
        )

    @classmethod # type:ignore
    def configured[**P](cls: Callable[P, "Optimizer"], /, *args: P.args, **kwargs: P.kwargs,): # pylint:disable = E0602
        return ConfiguredOptimizer(cls, *args, **kwargs)

    @property
    def SUPPORTS_ASK(self) -> bool: return self.CONFIG["supports_ask"]
    @SUPPORTS_ASK.setter
    def SUPPORTS_ASK(self, value:bool): self.CONFIG['supports_ask'] = value

    @property
    def SUPPORTS_MULTIPLE_ASKS(self) -> bool: return self.CONFIG["supports_multiple_asks"]
    @SUPPORTS_MULTIPLE_ASKS.setter
    def SUPPORTS_MULTIPLE_ASKS(self, value:bool): self.CONFIG['supports_multiple_asks'] = value

    @property
    def REQUIRES_BATCH_MODE(self) -> bool: return self.CONFIG["requires_batch_mode"]
    @REQUIRES_BATCH_MODE.setter
    def REQUIRES_BATCH_MODE(self, value:bool): self.CONFIG['requires_batch_mode'] = value

    @property
    def STORE_PARAMDICTS(self) -> Literal['none', 'best', 'all']: return self.CONFIG["store_paramdicts"]
    @STORE_PARAMDICTS.setter
    def STORE_PARAMDICTS(self, value: Literal['none', 'best', 'all']): self.CONFIG['store_paramdicts'] = value


class OptimizerWithChildren(Optimizer):
    def __init__(
        self,
        set_params=True,
        set_budget=True,
        set_seed=True,
        scheduler_step=True,
        defaults=None,
        seed=None,
    ):
        super().__init__(defaults, seed)
        self._set_params = set_params
        self._set_budget = set_budget
        self._set_seed = set_seed
        self.__scheduler_step = scheduler_step

    def set_params(self, params):
        super().set_params(params)
        if self._set_params:
            for c in self.children: c.set_params(params)
        return self

    def set_budget(self, budget):
        super().set_budget(budget)
        if self._set_budget:
            for c in self.children: c.set_budget(budget)
        return self

    def set_seed(self, seed):
        super().set_seed(seed)
        if self._set_seed:
            for c in self.children: c.set_seed(seed)
        return self

    def _scheduler_step(self, study):
        super()._scheduler_step(study)
        if self.__scheduler_step:
            for c in self.children: c._scheduler_step(study)

class WrappedOptimizer(OptimizerWithChildren):
    def __init__(
        self,
        optimizer: Optimizer,
        set_params=True,
        set_budget=True,
        set_seed=True,
        scheduler_step=True,
        defaults=None,
        seed=None,
    ):
        super().__init__(set_params=set_params, set_budget=set_budget, set_seed=set_seed, scheduler_step=scheduler_step, defaults=defaults, seed=seed)
        self.children = [optimizer]
        self.CONFIG = self.children[0].CONFIG

    @property
    def optimizer(self):
        return self.children[0]

    def wrap(self, optimizer:Any):
        self.optimizer.wrapped_optimizer = self.wrapped_optimizer = optimizer

class Minimizer(Optimizer, ABC):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode = True,
    )

    def __init__(
        self,
        defaults: Optional[dict[str, Any]] = None,
        fallback_bounds=None,
        only_used=False,
        restart: Literal["best", "last", "random"] = "last",
        multi_objective = False,
    ):
        super().__init__(defaults)
        self.fallback_bounds = fallback_bounds
        self.only_used = only_used
        self.multi_objective = multi_objective
        self.restart = restart
        if restart == 'best': self.STORE_PARAMDICTS = 'best'
        self.last_value = float('inf')

    def _create_vec(self, params: ParamDict):
        self.x0, self.slices = self.params.params_to_vec(only_used=self.only_used)
        self.bounds = self.params.get_bounds(normalized=True, fallback = self.fallback_bounds)

    def set_params(self, params: ParamDict):
        super().set_params(params)
        self._create_vec(params)
        return self

    @abstractmethod
    def minimize(self, objective: Callable):
        ...

    def objective(self, x:np.ndarray | Any) -> float | np.ndarray:
        if not isinstance(x, np.ndarray): x = np.array(x, copy=False)
        self.last_value = self.closure(x)
        return self.last_value

    def step(self, study: "Study"):
        if self.multi_objective: self.closure = partial(study.evaluate_vec, slices=self.slices)
        else: self.closure = partial(study.evaluate_vec, slices=self.slices, return_scalar = True)

        self.minimize(self.objective)

        if self.restart == 'last': self._create_vec(self.params)
        elif self.restart == 'best': self._create_vec(study.best_trial.paramdict) #type:ignore
        elif self.restart == 'random':
            self.params.randomize()
            self._create_vec(self.params)

class KwargsMinimizer(Minimizer):
    def __init__(
        self,
        locals_copy: dict[str, Any],
        defaults: Optional[dict[str, Any]] = None,
        fallback_bounds=None,
        only_used=False,
        ignore: Optional[Sequence[str]] = None,
    ):
        del locals_copy['self']
        del locals_copy['__class__']
        self.budget = locals_copy.pop('budget', None)
        self.restart = locals_copy.pop('restart', None)

        locals_copy.pop('defaults', None)
        locals_copy.pop('fallback_bounds', None)
        locals_copy.pop('only_used', None)

        if ignore is not None:
            for i in ignore:
                del locals_copy[i]

        self.kwargs = locals_copy


        super().__init__(
            defaults=defaults,
            fallback_bounds=fallback_bounds,
            only_used=only_used,
            restart=self.restart,
        )


class ConfiguredOptimizer:
    """Basically a partial with a `names` attribute."""
    lib: str | None = None
    def __init__[**P](self, cls: Callable[P, Optimizer], /, *args: P.args, **kwargs: P.kwargs,): # pylint:disable = E0602
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        if not hasattr(self, "names"): self.names = [self.__class__.__name__, ]

    @property
    def name(self): return self.names[0]

    def __call__(self, **kwargs):
        obj: Optimizer = self.cls(*self.args, **self.kwargs, **kwargs)
        obj.names = self.names
        obj.lib = self.lib
        return obj

    def register(
        self,
        lib: Optional[str] = None,
        groups: str | Sequence[str] = (GROUPS.MAIN, ),
        maxdims: Optional[int] = None,
    ):
        """Register this optimizer to the optimizers registry.
        You can iterate over every single optimizer with `registry.keys()` or `registry.values()`.
        Registry holds classes or constructors, not instances. For consistency all registered constructors
        require no arguments, so that they can be iterated over effortlesly."""
        return _register(
            cls = self,
            lib = lib,
            groups = groups,
            maxdims = maxdims,
        )

    def set_name(
        self,
        names: str | Sequence[str],
        register: bool = False,
        lib: Optional[str] = None,
        groups: str | Sequence[str] = (GROUPS.MAIN, ),
        maxdims: Optional[int] = None,
    ):
        return _set_name(
            cls = self,
            names = names,
            register = register,
            lib = lib,
            groups = groups,
            maxdims = maxdims,
        )

    def configured[**P](self: Callable[P, Optimizer], /, *args: P.args, **kwargs: P.kwargs,): # pylint:disable = E0602
        return ConfiguredOptimizer(self, *args, **kwargs)
