import copy
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence, Generator
from typing import TYPE_CHECKING, Any, Optional
from functools import partial
import numpy as np

from .._types import Numeric
from ..param import numeric, ScaleTransform
from ..param.unordered_sampler import UnorderedSampler
from ..paramdict import ParamDict
from ..trial import FinishedTrial, Trial
from .optimizer import Optimizer, WrappedOptimizer

if TYPE_CHECKING:
    from ..study import Study

class _ModifiedStudy:
    """A wrapper around study that encodes trial paramdicts whenever they are accessed,
    and decodes when the study is called. Encoding might make params behave like in coordinate descent,
    perform PCA dimensionality reduction on params, etc..."""
    def __init__(
        self,
        study: "Study",
        encode_fn: "Callable[[ParamDict], ParamDict]",
    ):
        self.__study = study
        self.__encode_fn = encode_fn

    def __getattr__(self, name):
        return getattr(self.__study, name)

    def __call__(self, *args, **kwargs):
        return self.__study(*args, **kwargs)

    def __encode(self, trial:FinishedTrial):
        if trial.paramdict is not None:
            trial = trial.shallow_copy()
            trial.paramdict = self.__encode_fn(trial.paramdict) # type:ignore
        return trial

    @property
    def trials(self):
        return [self.__encode(trial) for trial in self.__study.trials]

    @property
    def best_trial(self):
        return self.__encode(self.__study.best_trial)

    @property
    def last_trial(self):
        return self.__encode(self.__study.last_trial)

    @property
    def best_indicator_trial(self):
        return self.__encode(self.__study.best_indicator_trial)

    @property
    def pareto_front_trials(self):
        return [self.__encode(trial) for trial in self.__study.pareto_front_trials]


class ModifiedOptimizer(WrappedOptimizer, ABC):
    """An optimizer that operates on a modified paramdict. Must set `encode` and `decode` methods."""
    def __init__(
        self,
        optimizer: Optimizer,
        defaults = None,
    ):
        super().__init__(defaults=defaults, optimizer=optimizer)
        self.trial_cls = self.optimizer.trial_cls = self.optimizer.trial_cls._transformed(self.decode)

    @abstractmethod
    def encode(self, paramdict:ParamDict) -> ParamDict:
        """Encode the trial into whatever space this optimizer works with.
        This shouldn't make any changes to `paramdict`, i.e. copy it if needed.
        That is because changes would be kept on existing trials that hold existing decoded params.

        :param params: Original space trial to encode.
        :return: Encoded trial.
        """

    def cheap_encode(self, paramdict:ParamDict) -> ParamDict:
        """Optional cheap encoding.
        A good way to do it is to store encoded params in trial's storage on `decode`, and simply read them in this method.
        This shouldn't make any changes to `paramdict`, i.e. copy it if needed.

        :param params: Original space trial to encode.
        :return: Encoded trial.
        """
        return self.encode(paramdict)

    @abstractmethod
    def decode(self, paramdict:ParamDict) -> ParamDict:
        """Decode encoded trial into original space. This must decode both paramdict and original space params.
        This shouldn't make any changes to `paramdict`, i.e. copy it if needed.
        That is because calling a study would affect current paramdict.

        :param params: Encoded trial.
        :return: Original space trial.
        """

    # So first encoded params are set to this optimizer.
    # Then when `ask` happens, self.optimizer will yield encoded params.
    # Those params will be passed to the trial constructor, which is self.optimizer.trial_cls._transformed(self.decode).
    # So it yields trials with decoded params.
    # And when those decoded trials are passed to `tell`, they are encoded again.
    # Additionally, study respects the `trial_cls`,
    # so all evaluations done directly by calling study will be decoded.

    def set_params(self, params: ParamDict):
        return super().set_params(self.encode(params))

    def _make_modified_study(self, study) -> "Study":
        return _ModifiedStudy(study, encode_fn=self.cheap_encode) # type:ignore

    def ask(self, study:"Study") -> "Generator[ParamDict]":
        yield from self.optimizer.ask(self._make_modified_study(study)) # type:ignore

    def tell(self, trials: list[Trial], study: "Study"):
        for trial in trials: trial.params = self.cheap_encode(trial.params)
        return self.optimizer.tell(trials, study = self._make_modified_study(study))

    def _modified_tell(self, trials: list[Trial], study: "Study"):
        for trial in trials: trial.params = self.cheap_encode(trial.params)
        return self.opt_tell(trials, study = self._make_modified_study(study))

    def step(self, study:"Study"):
        self.opt_tell = self.optimizer.tell
        self.optimizer.tell = self._modified_tell
        value = self.optimizer.step(self._make_modified_study(study))
        self.optimizer.tell = self.opt_tell
        return value



class VectorizedOptimizer(ModifiedOptimizer):
    def __init__(self,optimizer: Optimizer,):
        super().__init__(optimizer=optimizer)

    def set_params(self, params: ParamDict):
        self.original_params = params
        self.vec, self.slices = params.params_to_vec()
        self.lb = np.min(params.get_lower_bounds(fallback= - np.inf))
        if self.lb == - np.inf: self.lb = None
        self.ub = np.max(params.get_upper_bounds(fallback= np.inf))
        if self.ub == - np.inf: self.ub = None
        return super().set_params(params)

    def encode(self, paramdict:ParamDict) -> ParamDict:
        self.vec, self.slices = paramdict.params_to_vec()
        p = ParamDict({'vec': numeric(self.vec.shape, low = self.lb, high = self.ub, init = self.vec, normalize=False)})
        p.used_params = {'vec'}
        return p

    def cheap_encode(self, paramdict:ParamDict) -> ParamDict:
        if 'encoded' in paramdict.storage: return paramdict.storage['encoded']
        return self.encode(paramdict)

    def decode(self, paramdict:ParamDict) -> ParamDict:
        decoded = self.original_params.copy()
        decoded.storage['encoded'] = paramdict
        decoded.vec_to_params_(paramdict['vec'].data, slices = self.slices)

        return decoded

class RescaledOptimizer(ModifiedOptimizer):
    def __init__(self,optimizer: Optimizer, scale:float):
        super().__init__(optimizer=optimizer)
        self.tfm = ScaleTransform(scale)
        self.scale = scale

    def encode(self, paramdict:ParamDict) -> ParamDict:
        paramdict = paramdict.copy()
        for param in paramdict.values():
            param.transforms.append(self.tfm)
            param.data *= self.scale
        return paramdict

    def decode(self, paramdict:ParamDict) -> ParamDict:
        paramdict = paramdict.copy()
        for param in paramdict.values():
            param.transforms.remove(self.tfm)
            param.data /= self.scale
        return paramdict


class UnorderedOptimizer(ModifiedOptimizer):
    def __init__(self,optimizer: Optimizer,):
        super().__init__(optimizer=optimizer)

    def encode(self, paramdict:ParamDict) -> ParamDict:
        paramdict = paramdict.copy()
        for param in paramdict.values():
            if param.TYPE == 'continuous' or param.TYPE == 'discrete':
                param.store['orig_sampler'] = param.sampler
                param.store['orig_type'] = param.TYPE
                param.sampler = UnorderedSampler(discrete_step = param.sampler.discrete if hasattr(param.sampler, 'discrete') else True) # type:ignore
                param.TYPE = 'unordered'
        return paramdict

    def decode(self, paramdict:ParamDict) -> ParamDict:
        paramdict = paramdict.copy()
        for param in paramdict.values():
            if 'orig_sampler' in param.store:
                param.sampler = param.store['orig_sampler']
                param.TYPE = param.store['orig_type']
        return paramdict
