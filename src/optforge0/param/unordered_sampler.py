import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Self

import numpy as np

from .._types import SamplerType
from .base_param import Param
from .numeric_sampler import NumericSampler


class UnorderedSampler(NumericSampler):

    def sample_petrub(
        self,
        param: "Param",
        sigma: float,
        sampler: Optional[SamplerType] = None,
        custom_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = param.data if custom_data is None else custom_data

        if param.rng.random.random() < sigma: return param.sample_random(sampler = sampler, custom_data = data)
        if sigma <= 0: return data

        change_mask = np.random.binomial(1, np.full(data.shape, sigma)).astype(bool, copy=False)

        # make sure at least 1 variable is changed
        # but if 1 variable, we don't want it to behave like random search
        if change_mask.size > 1 and change_mask.sum() == 0:
            change_mask = np.full(data.shape, False, dtype=bool)
            change_mask.flat[param.rng.random.randrange(0, change_mask.size)] = True

        return np.where(
            change_mask,
            param.sample_random(sampler = sampler, custom_data = data),
            data,
        )

    def sample_random(
        self,
        param: "Param",
        sampler: Optional[SamplerType] = None,
        custom_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = param.data if custom_data is None else custom_data

        low = param.low; high = param.high
        if low is None: low = -1
        if high is None: high = 1
        # always uniform due to unorderedness
        res = param.rng.numpy.uniform(low = low, high = high, size = data.shape)
        return res