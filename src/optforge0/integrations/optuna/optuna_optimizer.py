"""
Optuna references:

https://github.com/optuna/optuna

https://github.com/optuna/optuna-integration

https://github.com/optuna/optunahub

@inproceedings{akiba2019optuna,
  title={{O}ptuna: A Next-Generation Hyperparameter Optimization Framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={The 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
  pages={2623--2631},
  year={2019}
}
"""
#pylint:disable = W0707
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from ...optim.optimizer import Config, Optimizer
from ...param import ParamTypes

if TYPE_CHECKING:
    from ..._types import Numeric
    from ...trial import Trial
    import optuna


__all__ = [
    "silence_optuna",
    "OptunaSampler",
    "get_all_optuna_samplers",
    "constraints",
]

def silence_optuna():
    try: import optuna
    except ModuleNotFoundError: raise ImportError("Optuna is not installed.")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

inf = float('inf')


def _optuna_names(sampler) -> list[str]:
    names: list[str] = [sampler.__name__ if hasattr(sampler, '__name__') else sampler.__class__.__name__]
    if 'Sampler' in names[0]: names.append(names[0].replace('Sampler', ''))
    return names

def constraints(trial):
    return trial.user_attrs["constraint"]

class OptunaSampler(Optimizer):
    CONFIG = Config(
        supports_ask=True,
        supports_multiple_asks=True,
        requires_batch_mode=False,
    )
    def __init__(
        self,
        sampler: "Optional[optuna.samplers.BaseSampler | type[optuna.samplers.BaseSampler]]" = None,
        param_mode: Literal["mixed", "numeric"] = "mixed",
        directions: Optional[list[Literal["minimize", "maximize"]]] = None,
        soft_constraints = False,
        hard_constraints = False,
        silence = True,
    ):
        import optuna

        super().__init__(defaults = dict(param_mode = param_mode))
        if silence: silence_optuna()
        if isinstance(sampler, type): sampler = sampler()
        self.wrapped_optimizer = sampler
        if directions is None: self.optuna_study = optuna.create_study(sampler = self.wrapped_optimizer)
        else: self.optuna_study = optuna.create_study(sampler = self.wrapped_optimizer, directions = directions)
        self.soft_constraints = soft_constraints
        self.hard_constraints = hard_constraints

        if not hasattr(self, 'names'): self.names = _optuna_names(self.wrapped_optimizer)

    def ask(self, study):
        optuna_trial = self.optuna_study.ask()

        # sample new params with optuna sampler
        for name, param, store in self.yield_names_params_stores(only_used=True):

            low = param.low
            high = param.high
            if low is None: low = param.fallback_low
            if high is None: high = param.fallback_high

            step = param.step
            data = np.empty(param.data.shape)

            param_mode = store['param_mode'].lower()

            # suggest stuff based on type of param
            if param_mode == 'mixed':
                if param.TYPE == ParamTypes.TRANSIT:
                    suggest_fn = partial(optuna_trial.suggest_categorical,  choices = param.sampler.choices_numeric.tolist())
                elif param.TYPE == ParamTypes.UNORDERED:
                    suggest_fn = partial(optuna_trial.suggest_categorical,  choices = np.linspace(low, high, 100).tolist(),)
                elif step is not None and step % 1 == 0 and param.TYPE == ParamTypes.DISCRETE:
                    suggest_fn = partial(optuna_trial.suggest_int, low = int(low), high = int(high), step = int(step))
                else:
                    suggest_fn = partial(optuna_trial.suggest_float, low = low, high = high, step = step)

            elif param_mode == 'numeric':
                suggest_fn = partial(optuna_trial.suggest_float, low = low, high = high, step = step)

            else: raise ValueError(f'invalid param_mode {store["param_mode"]!r}')

            # iterate over the flattened array of this param and suggest all values
            for idx in range(data.size): data.flat[idx] = suggest_fn(f'p_{name}_{idx}')
            param.data = data.reshape(param.data.shape)

        self.params.storage['trial'] = optuna_trial
        yield self.params.copy()

    def tell(self, trials: "list[Trial]", study) -> "None":
        for trial in trials:
            value = trial.get_value(soft_penalty=not self.soft_constraints, hard_penalty=not self.hard_constraints)
            if isinstance(value, np.ndarray): value = value.tolist()
            if 'trial' in trial.params.storage:
                optuna_trial: "optuna.Trial" = trial.params.storage['trial']
                all_constraints = []
                if self.soft_constraints: all_constraints.extend(trial.soft_violations.tolist())
                if self.hard_constraints: all_constraints.extend(trial.hard_violations.tolist())
                if self.soft_constraints or self.hard_constraints: optuna_trial.set_user_attr('constraint', all_constraints)
                self.optuna_study.tell(optuna_trial, value)
            else:
                #raise ValueError('No optuna trial found in trial params storage')
                # this happens once to tell the optimizer loss with the initial point
                # which I haven't implemented...
                pass
                # telling existing points could be done here


def get_all_optuna_samplers() -> "list[type[optuna.samplers.BaseSampler]]":
    try:
        import optuna
        # optuna uses lazy imports, this makes sure to import all samplers
        #for i in optuna.integration.__all__: getattr(optuna.integration, i)
    except ModuleNotFoundError: raise ImportError("Optuna is not installed.")
    try:
        import optuna_integration
        for i in optuna_integration.__all__:
            try: getattr(optuna.integration, i)
            except AttributeError: pass
    except ModuleNotFoundError: pass

    from ...python_tools import subclasses_recursive
    return list(sorted(list(subclasses_recursive(optuna.samplers.BaseSampler)), key=lambda x: x.__name__))


# TODO: Optforge2Optuna