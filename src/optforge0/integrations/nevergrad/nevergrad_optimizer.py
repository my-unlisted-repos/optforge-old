#pylint:disable = W0707
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Optional, Any

import numpy as np

from ..._utils import _ensure_float
from ...optim.optimizer import Config, Optimizer
from ...trial import Trial
from ...param import ParamTypes

if TYPE_CHECKING:
    import nevergrad as ng

__all__ = [
    "NevergradOptimizer"
]

def _nevergrad_names(cls) -> list[str]:
    if isinstance(cls, type): return [cls.__name__,]
    subcls = cls.__class__
    namespace = str(subcls).split("'")[1].split('.')
    if len(namespace) >= 3: return [f'{namespace[-1]}.{cls.name}', cls.name]
    return ['uninitialized', ]

class NevergradOptimizer(Optimizer):
    CONFIG = Config(
        supports_ask=True,
        supports_multiple_asks=True,
        requires_batch_mode=False,
    )
    wrapped_optimizer: "ng.optimizers.base.Optimizer"
    def __init__(
        self,
        opt_cls:"Optional[type[ng.optimizers.base.Optimizer] | Callable[..., ng.optimizers.base.Optimizer]]" = None,
        param_mode: Literal["mixed", 'numeric'] = "mixed",
        mutable_sigma=False,
        budget=None,
        deterministic=False,
        use_init = False,
        soft_constraints: bool = False,
        hard_constraints: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(
            defaults={"param_mode": param_mode, "mutable_sigma": mutable_sigma, "deterministic": deterministic, "use_init": use_init},
            seed = seed,
        )

        self.budget = budget
        self.wrapped_constructor: "type[ng.optimizers.base.Optimizer] | Callable" = opt_cls # type:ignore

        self.soft_constraints = soft_constraints
        self.hard_constraints = hard_constraints
        
        if not hasattr(self, 'names'):
            cls = self.wrapped_optimizer.__class__ if self.wrapped_optimizer is not None else self.wrapped_constructor
            self.names = _nevergrad_names(cls)

    def make_wrap_args(self) -> tuple[Sequence, Mapping]:
        import nevergrad as ng

        parametrization = ng.p.Dict()
        for name, param, store in self.params.yield_names_params_stores(self.defaults):
            param_mode:Literal["mixed", 'numeric'] = store['param_mode']
            mutable_sigma = store['mutable_sigma']
            low = param.low; high = param.high
            # doesn't seem to support only one bound
            if low is None and high is not None: low = param.get_required_low()
            if high is None and low is not None: high = param.get_required_high()
            #print(f'{low = }, {high = }')

            shape = param.data.shape
            size = param.data.size
            if shape != 1 and isinstance(shape, int): shape = (shape,)
            # ----------------------------- OPTFORGE SCALING ----------------------------- #
            if param_mode == 'numeric':
                if size != 1:
                    if store['use_init']: parametrization[name] = ng.p.Array(init = param.data, lower=low, upper=high, mutable_sigma=mutable_sigma)
                    else: parametrization[name] = ng.p.Array(shape = shape, lower=low, upper=high, mutable_sigma=mutable_sigma)
                else:
                    if store['use_init']: parametrization[name] = ng.p.Scalar(init = _ensure_float(param.data), lower=low, upper=high, mutable_sigma=mutable_sigma)
                    else: parametrization[name] = ng.p.Scalar(lower=low, upper=high, mutable_sigma=mutable_sigma)

            # ----------------------------- NEVERGRAD SCALING ---------------------------- #
            else:
                if param.TYPE in (ParamTypes.CONTINUOUS, ParamTypes.DISCRETE, ParamTypes.ARGSORT):
                    # ARRAYS
                    if size != 1:
                        # -------------------------------- ARRAY PARAM ------------------------------- #
                        if store['use_init']: parametrization[name] = ng.p.Array(init = param.data, lower=low, upper=high, mutable_sigma=mutable_sigma)
                        else: parametrization[name] = ng.p.Array(shape = shape, lower=low, upper=high, mutable_sigma=mutable_sigma)
                    # SCALARS
                    else:
                        # log is already handled by optforge
                        # if param.domain in ('log10', 'log2', 'ln'):
                        #     if param.domain == 'log10': exponent = 10
                        #     elif param.domain == 'log2': exponent = 2
                        #     else: exponent = math.e
                        #     # -------------------------------- LOG SCALAR -------------------------------- #
                        #     if store['use_init']: parametrization[name] = ng.p.Log(init = _ensure_float(param.data),lower=low, upper=high, exponent=exponent, mutable_sigma=mutable_sigma)
                        #     else: parametrization[name] = ng.p.Log(lower=low, upper=high, exponent=exponent, mutable_sigma=mutable_sigma)
                        # else:
                        # ---------------------------------- SCALAR ---------------------------------- #
                        if store['use_init']: parametrization[name] = ng.p.Scalar(init = _ensure_float(param.data), lower=low, upper=high, mutable_sigma=mutable_sigma)
                        else: parametrization[name] = ng.p.Scalar(lower=low, upper=high, mutable_sigma=mutable_sigma)
                    # -------------------------------- TRANSIT ------------------------------- #
                elif param.TYPE == ParamTypes.TRANSIT:
                    choices = param.sampler.choices_numeric
                    parametrization[name] = ng.p.TransitionChoice(choices = choices, repetitions=param.data.size, ordered=False)
                    # -------------------------------- UNORDERED ------------------------------- #
                elif param.TYPE == ParamTypes.UNORDERED:
                    if low is None: low = param.fallback_low
                    if high is None: high = param.fallback_high
                    parametrization[name] = ng.p.TransitionChoice(choices = np.linspace(low, high, 1000), repetitions=param.data.size, ordered=False)
                    # -------------------------------- ONEHOT ------------------------------- #
                elif param.TYPE == ParamTypes.ONEHOT:
                    choices = param.sampler.choices_numeric
                    parametrization[name] = ng.p.Choice(choices = choices, repetitions=int(np.prod(param.data.shape[:-1])), deterministic=store['deterministic'])

                else: raise NotImplementedError(f"Unsupported parameter type: {param.TYPE}")

        if self.rng.seed is not None: parametrization.random_state = np.random.RandomState(self.rng.seed) # pylint:disable=E1101
        self.parametrization = parametrization
        if self.budget is not None: return ((parametrization, ), {"budget": self.budget})
        return ((parametrization,), {})

    def set_seed(self, seed:Optional[int]):
        if hasattr(self, "parametrization"): self.parametrization.random_state = np.random.RandomState(self.rng.seed) # pylint:disable=E1101
        return super().set_seed(seed)

    def _set_nevergrad_params_(self, x:"ng.p.Dict"):
        """Set self.params to nevergrad parametrization `x`"""
        params = {name: (p, s) for name, p, s in self.yield_names_params_stores()}
        for name, ngp in x.items():
            p, s = params[name]
            param_mode = s['param_mode']
            if p.TYPE == ParamTypes.ONEHOT and param_mode == 'mixed':
                p.sampler.set_index_array(p, np.array(ngp.value, copy = False)) # type:ignore
            else: p.data = np.array(ngp.value, copy=False).reshape(p.data.shape)

        self.params.storage['ng'] = x

    def ask(self, study):
        x:"ng.p.Dict" = self.wrapped_optimizer.ask() # type:ignore
        self._set_nevergrad_params_(x) # this makes sure self.params is synchronised with nevergrad params
        yield self.params.copy()

    def tell(self, trials:list[Trial], study):
        for t in trials:
            if 'ng' in t.params.storage:
                constraint_violation = []
                if self.soft_constraints: constraint_violation.extend(t.soft_violations.tolist())
                if self.hard_constraints: constraint_violation.extend(t.hard_violations.tolist())
                self.wrapped_optimizer.tell(
                    candidate = t.params.storage['ng'],
                    loss = t.get_value(soft_penalty = not self.soft_constraints, hard_penalty = not self.hard_constraints),
                    constraint_violation=constraint_violation,
                )
            else:
                #raise ValueError('No original params found in trial params storage')
                #print(f'{self}: No original params found in trial params storage')
                # this happens once to tell the optimizer loss with the initial point
                # which I haven't implemented...
                pass
                # telling existing points could be done here

