from collections import UserDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from .param import Param
from .python_tools import reduce_dim
from .scheduler.scheduler import Scheduler
from ._types import Numeric
from .rng import RNG
__all__ = [
    "ParamDict",
]


class Store:
    """Wrapper around `store` and `defaults`."""
    def __init__(self, store: dict[str, Any], defaults: dict[str, Any]):
        self.store = store
        self.defaults = defaults

    def __getitem__(self, key: str):
        """Returns value of the key from `store`. If key isn't in `store`, returns it from `defaults`."""
        if key in self.store: v = self.store[key]
        else: v = self.defaults[key]
        if isinstance(v, Scheduler): return v()
        return v

    def __setitem__(self, key: str, value):
        """Sets {key: value} to `store`."""
        self.store[key] = value

    def __delitem__(self, key: str):
        """Deletes key from `store`."""
        del self.store[key]

    def __str__(self):
        text = self.defaults.copy()
        text.update(self.store)
        return str(text)

    def __repr__(self): return str(self)

    def __contains__(self, item):
        if item in self.store: return True
        if item in self.defaults: return True
        return False

class ParamDict(UserDict[str, Param]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.used_params: set[str] = set()
        self.storage = {}

    def copy_data(self) -> "dict[str, np.ndarray]":
        """Returns a dictionary with copies of all param's ndarrays.

        example:
        ```py
        # use `clone` to back up params
        backup = self.params.clone()

        # generate new random params params
        for p in self.params:
            p.data = p.uniform()

        # evaluate new loss
        loss = closure()

        # if loss didn't improve, restore the backup
        if loss > prev_loss:
            self.params.set_(backup)

        ```
        """
        return {k: v.data.copy() for k, v in self.data.items()}

    def copy(self) -> "ParamDict":
        """Returns a full copy of this ParamDict. This fully copies each param, so if you only need ndarray data, you can use `copy_data` instead."""
        c = ParamDict({k: v.copy() for k, v in self.data.items()})
        c.used_params = self.used_params.copy()
        c.storage = self.storage.copy()
        return c

    def set_data_(self, x:"dict[str, np.ndarray] | ParamDict"):
        """Takes a dictionary of ndarrays and sets this ParamDict's params to those ndarrays.

        example:
        ```py
        # use `clone` to back up params
        backup = self.params.clone()

        # generate new random params params
        for p in self.params:
            p.data = p.uniform()

        # evaluate new loss
        loss = closure()

        # if loss didn't improve, restore the backup
        if loss > prev_loss:
            self.params.set_(backup)

        ```
        """
        for k, v in x.items():
            if k in self.data:
                if isinstance(v, Param): self.data[k].data = v.data
                else: self.data[k].data = v

    def update(self, m): # type:ignore
        if isinstance(m, ParamDict): self.used_params = m.used_params
        super().update(m)

    def yield_params(self, only_used: bool = False):
        if only_used: yield from [p for name, p in self.items() if name in self.used_params]
        else: yield from self.values()

    def yield_names_params(self, only_used: bool = False):
        if only_used: yield from [(name, p) for name, p in self.items() if name in self.used_params]
        else: yield from self.items()

    def yield_stores(self, defaults:dict[str, Any], only_used: bool = False, ):
        for p in self.yield_params(only_used):
            yield Store(p.store, defaults)

    def yield_names_stores(self, defaults:dict[str, Any], only_used: bool = False, ):
        for name, p in self.yield_names_params(only_used):
            yield name, Store(p.store, defaults)

    def yield_params_stores(self, defaults:dict[str, Any], only_used: bool = False, ):
        for p in self.yield_params(only_used):
            yield p, Store(p.store, defaults)

    def yield_names_params_stores(self, defaults:dict[str, Any], only_used: bool = False, ):
        for name, p in self.yield_names_params(only_used):
            yield name, p, Store(p.store, defaults)

    def params_to_vec(self, only_used:bool = False) -> tuple[np.ndarray, dict[str, slice]]:
        if only_used: params = [(name, p) for name, p in self.items() if name in self.used_params]
        else: params = list(self.items())
        vec = np.array(reduce_dim([p.data.flat for name, p in params]), copy=False)
        slices = {}
        cur = 0
        for name, p in params:
            slices[name] = slice(cur, cur + p.data.size)
            cur += p.data.size
        return vec, slices

    def vec_to_params_(self, vec:np.ndarray, slices:dict[str, slice]) -> None:
        # cur = 0
        # for p in self.values():
        #     if (not only_used) or (p in self.used_params):
        #         size = p.data.size
        #         p.data.flat[:] = vec[cur:cur+size]
        #         cur += size

        for name, slice_ in slices.items():
            # if name in self:
                param_data = self[name].data
                vec_data = vec[slice_]
                if param_data.size != vec_data.size: raise ValueError(f'vec to params size mismatch {slice_ = }, {param_data.shape = }, {param_data.size = }')
                param_data.flat[:] = vec_data


    def array_dict_to_params_(self, d: dict[str, np.ndarray], normalized = True):
        for k,v in d.items():
            if normalized: self[k].data = v
            else: self[k].set_unscaled_array(v)

    def value_dict_to_params_(self, d: dict[str, Any]):
        """Takes in actual values. For example, FinishedTrial.params"""
        for k,v in d.items(): self[k].set_value(v)

    def scalar_dict_to_params_(
        self,
        d: dict[str, Numeric],
        normalized = True,
    ):
        if normalized:
            for k,v in d.items():
                key, idx = k.rsplit('_', maxsplit=1)
                self[key].data.flat[idx] = v

        else:
            arrs: dict[str, np.ndarray] = {}
            for k,v in d.items():
                key, idx = k.rsplit('_', maxsplit=1)
                if key not in arrs: arrs[key] = np.empty_like(self[key].data.flat)
                arrs[key][idx] = v

            for k,v in arrs.items():
                self[k].set_unscaled_array(v)

    def get_bounds(self, normalized=True, fallback: Optional[tuple[Any, Any] | Sequence[Any] | Literal['default']] = None):
        bounds = []
        for param in self.values():
            if normalized: low = param.low
            else: low = param.low_original
            if low is None and fallback is not None:
                if fallback == 'default': low = param.fallback_low if normalized else param.fallback_low_original
                else: low = fallback[0]

            if normalized: high = param.high
            else: high = param.high_original
            if high is None and fallback is not None:
                if fallback == 'default': high = param.fallback_high if normalized else param.fallback_high_original
                else: high = fallback[1]

            bounds.append([(low, high)] * param.data.size)
        return reduce_dim(bounds)

    def get_lower_bounds(self, normalized = True, fallback: Optional[Any | Literal['default']] = None):
        bounds = []
        for param in self.values():
            if normalized: low = param.low
            else: low = param.low_original
            if low is None and fallback is not None:
                if fallback == 'default': low = param.fallback_low if normalized else param.fallback_low_original
                else: low = fallback

            bounds.append([low] * param.data.size)
        return reduce_dim(bounds)

    def get_upper_bounds(self, normalized = True, fallback: Optional[Any | Literal['default']] = None):
        bounds = []
        for param in self.values():
            if normalized: high = param.high
            else: high = param.high_original
            if high is None and fallback is not None:
                if fallback == 'default': high = param.fallback_high if normalized else param.fallback_high_original
                else: high = fallback

            bounds.append([high] * param.data.size)
        return reduce_dim(bounds)

    def get_steps(self, normalized = True, fallback: Optional[Any] = 0):
        steps = []
        for param in self.values():
            if normalized: step = param.step
            else: step = param.step_original
            if step is None and fallback is not None: step = fallback

            steps.append(step)
        return steps

    def randomize(self):
        for p in self.values():
            p.data = p.sample_random()

    def numel(self):
        return sum(i.data.size for i in self.values())

    def _set_rng(self, rng: RNG):
        for p in self.values(): p.rng = rng
