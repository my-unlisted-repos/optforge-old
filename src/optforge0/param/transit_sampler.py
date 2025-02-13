from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from .._types import SamplerType
from .base_param import Param
from .base_sampler import Sampler


class TransitSampler(Sampler):
    def __init__(self, choices: Sequence[Any] | np.ndarray):
        self.choices = np.array(choices)

    def _update(self, param: "Param",):
        self.choices_numeric = np.linspace(
            param.get_required_low(),
            param.get_required_high(),
            num = len(self.choices),
            dtype = np.float64
        )

    def copy(self):
        new = self.__class__(choices = self.choices.copy())
        new.choices_numeric = self.choices_numeric.copy()
        return new

    def __call__(self, param: "Param") -> Any:
        index_array = param.get_unscaled_array().astype(int, copy=False)

        # convert numpy scalar to shape 1 array to avoid inconsistency with using it as index array
        if index_array.ndim == 0: index_array = np.expand_dims(index_array, axis=0)
        return self.choices[index_array]

    def sample_random(
        self,
        param: "Param",
        sampler: Optional[SamplerType] = None,
        custom_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = param.data if custom_data is None else custom_data
        return param.rng.numpy.choice(self.choices_numeric, data.shape)


    def sample_petrub(
        self,
        param: "Param",
        sigma: float,
        sampler: Optional[SamplerType] = None,
        custom_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = param.data if custom_data is None else custom_data
        low: float = param.low; step: float = param.step # type:ignore

        sigma = sigma / step
        if sigma > 1: return param.sample_random(sampler = sampler, custom_data = data)
        if sigma < 0: return param.data

        change_mask = param.rng.numpy.binomial(1, np.full(data.shape, sigma)).astype(bool, copy=False) # type:ignore

        # make sure at least 1 variable is changed
        if change_mask.size > 1 and change_mask.sum() == 0:
            change_mask = np.full(data.shape, False, dtype=bool)
            change_mask.flat[param.rng.random.randrange(0, change_mask.size)] = True

        return np.where(
            change_mask,
            param.sample_random(sampler = sampler, custom_data = data),
            (
                np.round((data - low) / step) * step
            ) + low
        )


    def sample_set(self, param: "Param", x: np.ndarray, custom_data:Optional[np.ndarray] = None) -> np.ndarray:
        data = param.data if custom_data is None else custom_data
        step: float = param.step # type:ignore

        data_discrete: np.ndarray = np.round(data / step) * step

        x_discrete = np.round(x / step) * step
        remainder = (x - np.round(x / step) * step) / step
        x_discrete += param.rng.numpy.binomial(
            n = 1,
            # discrete petrubation remainder normalized by scaled step
            p = np.abs(remainder)
            ) * step * np.sign(remainder)

        return data_discrete + x_discrete


    def sample_mutate(self, param: "Param", sigma: float, mutation: ..., custom_data = None) -> None:
        raise NotImplementedError

    def sample_crossover(self, param: "Param", other:np.ndarray, sigma: float, crossover: ..., custom_data = None) -> None:
        raise NotImplementedError

    def set_numeric_array(self, param: "Param", x:np.ndarray) -> None:
        """Set this param to array of choices from `choices_numeric`."""
        param.data = x.reshape(param.data.shape)

    def set_value(self, param: "Param", x: Any) -> None:
        """Set this param from actual value that this param represents.
        Therefore ``x`` must be array of elements of ``self.choices``."""
        x = np.array(x, copy=False).reshape(param.data.shape)
        arr = np.zeros(param.data.shape, dtype=int)
        for i, ch in enumerate(self.choices):
            arr += np.where(x == ch, i, 0)
        param.set_unscaled_array(arr.astype(np.float64, copy=False))

    def get_numeric_array(self, param: "Param") -> np.ndarray:
        return param.data
