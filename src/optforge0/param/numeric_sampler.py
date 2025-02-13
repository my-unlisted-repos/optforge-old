from typing import Any, Optional

import numpy as np

from .._types import SamplerType
from .base_param import Param
from .base_sampler import Sampler


def _discrete_sample_set(param: "Param", x: np.ndarray) -> np.ndarray:
    step = param.step
    if step is None or step == 0 or param.domain != 'linear': return x

    # make petrubation discrete and use continous remainder as probabilities for additional discrete step
    x_discrete = np.floor(x / step) * step
    # so x = [-4.9, 1.7], step = 1, -> x_discrete = [-5, 1]

    # relative remainder, e.g. remainder = 0.9 means it is 0.9 * step
    rel_remainder = (x - x_discrete) / step
    # this is always positive because floor
    # dividing by step makes values always be in [0, 1] range.
    # [-4.9, 1.7] - [-5, 1] = [0.1, 0.7]

    # use remainder as chances to petrub x_discrete by 1 step a positive direction.
    # so -5 has 0.1 probability to turn into -4, 1 has 0.7 probability to turn into 2
    return x_discrete + param.rng.numpy.binomial(
        n = 1,
        p = rel_remainder
        ) * step

class NumericSampler(Sampler):
    def __init__(self, discrete_step = True):
        self.discrete_step = discrete_step

    def __call__(self, param: "Param",) -> Any:
        return param.get_unscaled_array()

    def copy(self):
        return self.__class__(discrete_step = self.discrete_step)

    def sample_random(
        self,
        param: "Param",
        sampler: Optional[SamplerType] = None,
        custom_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if sampler is None: sampler = param.rng.numpy.uniform
        data = param.data if custom_data is None else custom_data

        res = sampler(low = param.get_required_low(), high = param.get_required_high(), size = data.shape)
        return res

    def sample_petrub(
        self,
        param: "Param",
        sigma: float,
        sampler: Optional[SamplerType] = None,
        custom_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = param.data if custom_data is None else custom_data
        if sampler is None: sampler = param.rng.numpy.uniform

        low = param.low; high = param.high
        if high is not None and low is not None and high - low < sigma:
            return param.sample_random(sampler = sampler, custom_data=data) - data
        if sigma < 0: return data

        min_overflow = np.clip((low - (data - sigma)), 0, None) if low is not None else 0
        max_overflow = np.clip((high - (data + sigma)), None, 0) if high is not None else 0

        res = sampler(low = -sigma, high = sigma, size = data.shape) + min_overflow + max_overflow
        if self.discrete_step: return _discrete_sample_set(param, data + res)
        return data + res

    def sample_set(self, param: "Param", x: np.ndarray, custom_data:Optional[np.ndarray] = None) -> np.ndarray:
        if self.discrete_step: return _discrete_sample_set(param, x)
        return x

    def sample_mutate(self, param: "Param", sigma: float, mutation: ..., custom_data = None) -> None:
        raise NotImplementedError

    def sample_crossover(self, param: "Param", other:np.ndarray, sigma: float, crossover: ..., custom_data = None) -> None:
        raise NotImplementedError

    def set_value(self, param: "Param", x: Any) -> None:
        return param.set_unscaled_array(np.array(x, copy=False))

