from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from .._types import SamplerType
from .base_param import Param
from .transit_sampler import TransitSampler

def _normalize(x:np.ndarray, min=0., max=1.) -> np.ndarray:
    """Normalize to `[min, max]`"""
    x -= x.min()
    if x.max() != 0: x /= x.max()
    else: return x
    return x * (max - min) + min

class ArgsortPermutationSampler(TransitSampler):

    def __call__(self, param: "Param") -> Any:
        index_array = param.get_unscaled_array().argsort(axis = -1).astype(int, copy=False)
        return self.choices[index_array]

    def sample_random(
        self,
        param: "Param",
        sampler: Optional[SamplerType] = None,
        custom_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = param.data if custom_data is None else custom_data
        data = _normalize(data.argsort(-1).astype(np.float64), param.get_required_low(), param.get_required_high())

        mat = data.reshape((np.prod(data.shape[:-1]), data.shape[-1]))
        for row in mat:
            param.rng.numpy.shuffle(row)
        return mat.reshape(data.shape)

    def sample_petrub(
        self,
        param: "Param",
        sigma: float,
        sampler: Optional[SamplerType] = None,
        custom_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # copy because we work with data directly
        data = param.data.copy() if custom_data is None else custom_data.copy()

        if sigma > 1: return param.sample_random(sampler = sampler, custom_data = data)
        if sigma < 0: return param.data

        change_mask = param.rng.numpy.binomial(1, np.full(data.shape, sigma)).astype(bool, copy=False) # type:ignore

        # make sure at least 1 variable is changed
        if change_mask.size > 1 and change_mask.sum() == 0:
            change_mask = np.full(data.shape, False, dtype=bool)
            change_mask.flat[param.rng.random.randrange(0, change_mask.size)] = True

        for idxs in np.argwhere(change_mask): # pylint:disable=C0121 # noqa: E712
            # make sure there are no same values
            data[*idxs[:-1]] = self.choices_numeric[np.argsort(data[*idxs[:-1]])]

            # swap index with a random different one
            other_idx = param.rng.random.randrange(data.shape[-1])

            data[*idxs], data[*idxs[:-1], other_idx] = data[*idxs[:-1], other_idx], data[*idxs]

        return data.astype(np.float64, copy=False)


    def sample_mutate(self, param: "Param", sigma: float, mutation: ..., custom_data = None) -> None:
        raise NotImplementedError

    def sample_crossover(self, param: "Param", other:np.ndarray, sigma: float, crossover: ..., custom_data = None) -> None:
        raise NotImplementedError
