from abc import ABC, abstractmethod
from collections.abc import (Callable, Iterable, Mapping, MutableSequence,
                             Sequence)
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Optional, overload

import numpy as np

from .._types import Numeric, SamplerType
from .._utils import _maybe_ensure_float
from ..inits import InitLiteral, _set_init_value
from .transforms import (LogTransform, NormalizeTransform, ScaleTransform,
                         Transform)
from ..rng import RNG

if TYPE_CHECKING:
    from .base_sampler import Sampler

class ParamTypes(StrEnum):
    CONTINUOUS = 'continuous'
    DISCRETE = 'discrete'
    TRANSIT = 'transit'
    ONEHOT = 'onehot'
    UNORDERED = 'unordered'
    ARGSORT = 'argsort'

DomainLiteral = Literal["log10", "log2", "ln", "linear"]

class Param:

    def __init__(
        self,
        sampler: "Sampler",
        TYPE: Literal['continuous', 'discrete', 'transit', 'onehot', 'unordered', 'argsort'],

        shape: Optional[int | Sequence[int]],

        low: Optional[float],
        high: Optional[float],
        step: Optional[float],
        domain: DomainLiteral,
        scale: Optional[float],
        normalize: bool,
        fallback_low: float,
        fallback_high: float,

        oob_penalty: Optional[float],
        init: "InitLiteral | Callable | np.ndarray | Numeric | Sequence[Numeric]",

        options: Optional[dict[str, Any]],
        *,
        _new = True
    ):
        if _new:
            if TYPE not in ParamTypes: raise ValueError(f'TYPE {TYPE} is not a valid param type. Valid types are {list(ParamTypes._value2member_map_.keys())}')
            self.TYPE = TYPE

            self.store: dict[str, Any] = options if options is not None else {}
            self.sampler: "Sampler" = sampler

            self.rng = RNG(None)

            self._update(
                low = low,
                high = high,
                step = step,
                scale = scale,
                domain = domain,
                oob_penalty = oob_penalty,
                normalize = normalize,
                fallback_low = fallback_low,
                fallback_high = fallback_high,
            )

            if shape is None:
                init = np.array(init, copy=False)
                shape = init.shape


            _set_init_value(param = self, init = init, shape = shape) # type:ignore
            self.oob_violation = 0.


    def _update(
        self,
        low: Optional[float],
        high: Optional[float],
        step: Optional[float],
        scale: Optional[float],
        domain: DomainLiteral,
        oob_penalty: Optional[float],
        normalize: bool,
        fallback_low: float,
        fallback_high: float,
    ):
        self.transforms: list[Transform] = []

        self.low_original = _maybe_ensure_float(low)
        self.high_original = _maybe_ensure_float(high)
        self.step_original = _maybe_ensure_float(step)
        self.domain: Literal["log10", "log2", "ln", "linear"] = domain
        self.oob_penalty = _maybe_ensure_float(oob_penalty)

        if domain == 'log10': self.transforms.append(LogTransform(10))
        elif domain == 'log2': self.transforms.append(LogTransform(2))
        elif domain == 'ln': self.transforms.append(LogTransform(np.e))


        if normalize and (low is not None or high is not None):
            self.transforms.append(NormalizeTransform(self.low, self.high))
        if scale is not None: self.transforms.append(ScaleTransform(_maybe_ensure_float(scale)))

        self.scale: float = (scale or 1)

        self.sampler._update(self)

        self.fallback_low_original = fallback_low
        self.fallback_high_original = fallback_high


    def __repr__(self):
        return f'P{self.data}'
    @property
    def low(self): return self._forward(self.low_original)
    @property
    def high(self): return self._forward(self.high_original)
    @property
    def fallback_low(self): return self._forward(self.fallback_low_original)
    @property
    def fallback_high(self): return self._forward(self.fallback_high_original)
    @property
    def step(self):
        if self.step_original is None: return None
        if self.domain != 'linear': raise ValueError(f'scaled step is only possible with domain = "linear", however {self.domain = }')

        if self.low_original is None or self.high_original is None: return self.step_original * self.scale

        num_steps = (self.high_original - self.low_original) / self.step_original
        return (self.high - self.low) / num_steps # type:ignore

    def get_required_low(self) -> float:
        low = self.low
        if low is None: return self.fallback_low
        return low

    def get_required_high(self) -> float:
        high = self.high
        if high is None: return self.fallback_high
        return high

    def get_required_low_original(self) -> float:
        if self.low_original is None: return self.fallback_low_original
        return self.low_original

    def get_required_high_original(self) -> float:
        if self.high_original is None: return self.fallback_high_original
        return self.high_original

    def copy(self):
        new = self.__class__.__new__(self.__class__)

        new.TYPE = self.TYPE
        new.data = self.data.copy()
        new.store = self.store.copy()
        new.sampler = self.sampler.copy()
        new.transforms = self.transforms.copy()
        new.rng = self.rng.copy()

        new.low_original = self.low_original
        new.high_original = self.high_original
        new.step_original = self.step_original
        new.fallback_low_original = self.fallback_low_original
        new.fallback_high_original = self.fallback_high_original
        new.domain = self.domain
        new.oob_penalty = self.oob_penalty

        return new


    @overload
    def _forward(self, x:float) -> float: ...
    @overload
    def _forward(self, x:np.ndarray) -> np.ndarray: ...
    @overload
    def _forward(self, x:None) -> None: ...
    def _forward(self, x):
        for i in self.transforms: x = i.forward(x)
        return x

    @overload
    def _backward(self, x:float) -> float: ...
    @overload
    def _backward(self, x:np.ndarray) -> np.ndarray: ...
    @overload
    def _backward(self, x:None) -> None: ...
    def _backward(self, x):
        for i in reversed(self.transforms): x = i.backward(x)
        return x

    @overload
    def _quantize(self, x:float) -> float: ...
    @overload
    def _quantize(self, x:np.ndarray) -> np.ndarray: ...
    @overload
    def _quantize(self, x:None) -> None: ...
    def _quantize(self, x):
        """Aplies `step`."""
        if x is None: return None
        if self.step_original is None: return x
        # rounding is done because floating point imprecision messes up floor on small (~<0.1) scales.
        return np.floor(np.round(x / self.step_original, 12)) * self.step_original

    def _clip_and_penalize(self,):
        if self.low is None and self.high is None: return self.data
        clipped_data = np.clip(self.data, self.low, self.high)
        if self.oob_penalty is not None:
            self.oob_violation = self.oob_penalty * np.sum(np.abs(self.data - clipped_data)) / clipped_data.size

        return clipped_data

    def __call__(self) -> Any:
        return self.sampler(self)

    def sample_random(
        self,
        custom_data: Optional[np.ndarray] = None,
        sampler: Optional[SamplerType] = None,
    ) -> np.ndarray:
        """_summary_

        :param param: _description_
        :param sampler: _description_, defaults to np.random.uniform
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        """
        return self.sampler.sample_random(self, sampler = sampler, custom_data = custom_data,)

    def sample_petrub(
        self,
        sigma: float,
        custom_data: Optional[np.ndarray] = None,
        sampler: Optional[SamplerType] = None,
    ) -> np.ndarray:
        """_summary_

        :param sigma: _description_
        :param custom_data: _description_, defaults to None
        :param sampler: _description_, defaults to np.random.uniform
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        """
        return self.sampler.sample_petrub(self, sigma = sigma, custom_data = custom_data, sampler = sampler)


    def sample_generate_petrubation(
        self,
        sigma: float,
        custom_data: Optional[np.ndarray] = None,
        sampler: Optional[SamplerType] = None,
    ) -> np.ndarray:
        """_summary_

        :param sigma: _description_
        :param custom_data: _description_, defaults to None
        :param sampler: _description_, defaults to np.random.uniform
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        """
        return self.sampler.sample_generate_petrubation(self, sigma = sigma, custom_data = custom_data, sampler = sampler)

    def sample_set(self, x: np.ndarray, custom_data:Optional[np.ndarray] = None) -> np.ndarray:
        """_summary_

        :param x: _description_
        :param custom_data: _description_, defaults to None
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        """
        return self.sampler.sample_set(self, x = x, custom_data = custom_data)


    def sample_mutate(self, sigma: float, mutation: ..., custom_data = None) -> None:
        """_summary_

        :param sigma: _description_
        :param mutation: _description_
        :param custom_data: _description_, defaults to None
        :raises NotImplementedError: _description_
        """
        return self.sampler.sample_mutate(self, sigma = sigma, mutation = mutation, custom_data = custom_data)


    def sample_crossover(self, other:np.ndarray, sigma: float, crossover: ..., custom_data = None) -> None:
        """_summary_

        :param other: _description_
        :param sigma: _description_
        :param crossover: _description_
        :param custom_data: _description_, defaults to None
        :raises ValueError: _description_
        :raises ValueError: _description_
        :return: _description_
        """
        return self.sampler.sample_crossover(self, other = other, sigma = sigma, crossover = crossover, custom_data = custom_data)


    def set_value(self, x: Any) -> None:
        """Set this param to actual value that this param represents. For example, array of choices for a categorical param.

        :param x: Actual value of the param to set this to.
        """
        return self.sampler.set_value(self, x)

    def set_unscaled_array(self, x:np.ndarray) -> None:
        """Convert an unscaled array into optimization space array and set this param to it.
        For one-hot encoded, `x` should be a one hot encoded array, but without transforms

        :param x: Unscaled numerical value of the param to set this to.
        """
        self.data = self._forward(x.reshape(self.data.shape).astype(np.float64, copy=False))

    def get_unscaled_array(self) -> np.ndarray:
        return self._quantize(self._backward(self._clip_and_penalize()))