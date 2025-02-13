from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING, Self

import numpy as np

from .._types import SamplerType

if TYPE_CHECKING:
    from .base_param import Param


class Sampler(ABC):
    choices: np.ndarray
    choices_numeric: np.ndarray
    @abstractmethod
    def __call__(self, param: "Param") -> Any:
        ...

    def _update(self, param: "Param") -> None: pass

    @abstractmethod
    def copy(self) -> Self:
        ...

    @abstractmethod
    def sample_random(
        self,
        param: "Param",
        sampler: Optional[SamplerType] = None,
        custom_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate a uniformly distributed random value in the search space defined by `self.low` and `self.high`.

        Sampler must accept `(low: float, high:float, size: int | tuple[int])` arguments,
        and generate values roughly within (low, high) range. Range is not strictly enforced because values are clipped
        and larger values are penalized.

        If `low` and `high` are not defined, this raises an error.

        :param param: _description_
        :param sampler: _description_, defaults to np.random.uniform
        :return: _description_
        """

    def sample_generate_petrubation(
        self,
        param: "Param",
        sigma: float,
        sampler: Optional[SamplerType] = None,
        custom_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate random array within the neighborhood around `self.data`.
        The generated array will always be within the global search space defined by `self.low` and `self.high`.
        Therefore if `self.data` is on the edge of global seach space, the petrubation will be biased
        towards the center as to not get outside of the global search space.

        Distribution of petrubation is usually uniform but depends on the type of parameter.

        :param sigma: Magnitude of the petrubation, or maximum distance from `self.data` or `custom_data` to petrubation
        :param custom_data: If specified, generate petrubation around a custom data instead of `self.data`. \
            Custom data must be scaled (i.e. have same scaling as self.data). Defaults to None.
        """
        if sampler is None: sampler = param.rng.numpy.uniform
        data = param.data if custom_data is None else custom_data
        return param.sample_petrub(sigma=sigma, custom_data=data, sampler=sampler) - data

    def sample_petrub(
        self,
        param: "Param",
        sigma: float,
        sampler: Optional[SamplerType] = None,
        custom_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Unlike `sample_petrub`, this generates a petrubation around 0 to be added to `data`, not around `data`."""
        if sampler is None: sampler = param.rng.numpy.uniform
        data = param.data if custom_data is None else custom_data
        return data + param.sample_generate_petrubation(sigma=sigma, custom_data=data, sampler=sampler)

    @abstractmethod
    def sample_set(self, param: "Param", x: np.ndarray, custom_data:Optional[np.ndarray] = None) -> np.ndarray:
        """Sample `data` probabilistically. This does not mutate `self.data` but returns a new array.
        For example, for an integer param, `data = 4.1`, it has 90% chance to return to 4 and 10% to return  5.
        For continouos param, this just returns `data`. `custom_data` usually doesn't affect the sampling, but might in specific cases.
        """
    @abstractmethod
    def sample_mutate(self, param: "Param", sigma: float, mutation: ..., custom_data = None) -> None:
        """_summary_

        :param sigma: _description_
        :param mutation: _description_
        :param custom_data: _description_, defaults to None
        :raises NotImplementedError: _description_
        """

    @abstractmethod
    def sample_crossover(self, param: "Param", other:np.ndarray, sigma: float, crossover: ..., custom_data = None) -> None:
        """_summary_

        :param other: _description_
        :param sigma: _description_
        :param crossover: _description_
        :param custom_data: _description_, defaults to None
        :raises ValueError: _description_
        :raises ValueError: _description_
        :return: _description_
        """

    @abstractmethod
    def set_value(self, param: "Param", x: Any) -> None:
        """Set this param from actual value that this param represents. For example, category for a categorical param.

        :param x: Actual value of the param to set this to.
        """