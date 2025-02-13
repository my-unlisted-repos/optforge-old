from typing import Optional, Any
from collections.abc import Sequence
import numpy as np

from .base_param import Param
from .numeric_sampler import NumericSampler

def _one_hot(idxs:np.ndarray, num_choices):
    return (idxs[..., np.newaxis] == np.arange(num_choices)).astype(idxs.dtype, copy=False)

class OnehotSampler(NumericSampler):
    def __init__(self, choices: Sequence[Any] | np.ndarray):
        super().__init__(discrete_step = False)
        self.choices = np.array(choices)

    def _update(self, param: "Param",):
        self.choices_numeric = np.arange(len(self.choices)) # type:ignore

    def copy(self):
        new = self.__class__(choices = self.choices.copy())
        new.choices_numeric = self.choices_numeric.copy()
        return new

    def __call__(self, param: "Param") -> Any:
        index_array = param.get_unscaled_array().argmax(-1).astype(int)

        # convert numpy scalar to shape 1 array to avoid inconsistency with using it as index array
        if index_array.ndim == 0: index_array = np.expand_dims(index_array, axis=0)
        return self.choices[index_array]


    def sample_mutate(self, param: "Param", sigma: float, mutation: ..., custom_data = None) -> None:
        raise NotImplementedError

    def sample_crossover(self, param: "Param", other:np.ndarray, sigma: float, crossover: ..., custom_data = None) -> None:
        raise NotImplementedError

    def set_value(self, param: "Param", x: Any) -> None:
        """Set this param from actual value that this param represents.
        Therefore ``x`` must be array of elements of ``self.choices``."""
        x = np.array(x, copy=False).reshape(param.data.shape[:-1])
        arr = np.zeros(param.data.shape, dtype=int)
        for i, ch in enumerate(self.choices):
            arr += np.where(x == ch, i, 0)
        self.set_index_array(param, arr.astype(np.float64, copy=False))


    # ----------------------------------- extra ---------------------------------- #
    def get_index_array(self, param: "Param") -> np.ndarray:
        return param.get_unscaled_array().argmax(-1)

    def set_index_array(self, param: "Param", x:np.ndarray) -> None:
        """Set this param to array of choices from `choices_numeric`."""
        x = np.array(x, copy=False).reshape(param.data.shape[:-1])
        param.set_unscaled_array(
            _one_hot(
                np.array(x, dtype=np.float64, copy=False),
                num_choices = len(self.choices),
            ) * (param.high or 1.),
        )
