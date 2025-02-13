from collections.abc import Sequence, Iterable, Callable
from typing import TYPE_CHECKING, Any, Optional, Literal

import numpy as np

from .base_param import Param, DomainLiteral
from .._types import Numeric, ArrayLike
from .numeric_sampler import NumericSampler
from .transit_sampler import TransitSampler
from .onehot_sampler import OnehotSampler
from .unordered_sampler import UnorderedSampler
from .argsort_permutation_sampler import ArgsortPermutationSampler
from ..inits import InitLiteral

__all__ = [
    "numeric",
    "categorical",
    "permutation",
]
def numeric(
    shape: Optional[int | Sequence[int]],

    low: Optional[float] = None,
    high: Optional[float] = None,
    step: Optional[float] = None,
    domain: DomainLiteral = "linear",
    scale: Optional[float] = None,
    normalize: bool = True,

    discrete_step: bool = True,
    ordered = True,

    oob_penalty: Optional[float] = 1,
    init: "InitLiteral | Callable | np.ndarray | Numeric | Sequence[Numeric]" = 'mean',

    fallback_low: float = -1,
    fallback_high: float = 1,
    options: Optional[dict[str, Any]] = None,
):
    return Param(
        TYPE = ('discrete' if discrete_step else 'continuous') if ordered else 'unordered',
        sampler = NumericSampler(discrete_step = discrete_step) if ordered else UnorderedSampler(discrete_step=discrete_step),
        shape = shape,
        low = low,
        high = high,
        step = step,
        scale = scale,
        domain = domain,
        oob_penalty = oob_penalty,
        normalize = normalize,
        fallback_low = fallback_low,
        fallback_high = fallback_high,
        init = init,
        options = options,
    )

def categorical(
    shape: Optional[int | Sequence[int]],
    choices: int | ArrayLike,

    scale: Optional[float] = None,
    normalize: bool = False,

    one_hot: bool = False,

    oob_penalty: Optional[float] = 1,
    init: "InitLiteral | Callable | np.ndarray | Numeric | Sequence[Numeric]" = 'mean',
    options: Optional[dict[str, Any]] = None,
):
    if isinstance(choices, int): choices = range(choices)
    choices = list(choices)
    if one_hot:

        # we need shape for one hot encoding
        if isinstance(shape, int): shape = (shape, )
        if shape is None:
            init = np.array(init, copy=False)
            shape = init.shape

        return Param(
            TYPE = 'onehot',
            sampler = OnehotSampler(choices = choices),
            shape = (*shape, len(choices)),
            low = -1,
            high = 1,
            step = None,
            scale = scale,
            domain = 'linear',
            oob_penalty = oob_penalty,
            normalize = normalize,
            fallback_low = -1,
            fallback_high = 1,
            init = init,
            options = options,
        )
    else:
        return Param(
            TYPE = 'transit',
            sampler = TransitSampler(choices = choices),
            shape = shape,
            low = 0,
            high = len(choices) - 1,
            step = 1,
            scale = scale,
            domain = 'linear',
            oob_penalty = oob_penalty,
            normalize = normalize,
            fallback_low = -1,
            fallback_high = 1,
            init = init,
            options = options,
        )

def _categorical_update(
    param: Param,
    choices: int | ArrayLike,
    scale: Optional[float],
    normalize: bool,

    one_hot: bool,
    oob_penalty: Optional[float],
):
    if isinstance(choices, int): choices = range(choices)
    if one_hot:
        param._update(
            low = -1,
            high = 1,
            step = None,
            scale = scale,
            domain = 'linear',
            oob_penalty = oob_penalty,
            normalize = normalize,
            fallback_low = -1,
            fallback_high = 1,
        )
    else:
        choices = list(choices)
        param._update(
            low = 0,
            high = len(choices) - 1,
            step = 1,
            scale = scale,
            domain = 'linear',
            oob_penalty = oob_penalty,
            normalize = normalize,
            fallback_low = -1,
            fallback_high = 1,
        )


def permutation(
    shape: Optional[int | Sequence[int]],
    choices: int | ArrayLike,

    scale: Optional[float] = None,
    normalize: bool = False,

    type: Literal['argsort'] = 'argsort',

    oob_penalty: Optional[float] = 1,
    init: "InitLiteral | Callable | np.ndarray | Numeric | Sequence[Numeric]" = 'mean',
    options: Optional[dict[str, Any]] = None,
):
    if isinstance(choices, int): choices = range(choices)
    choices = list(choices)
    if type == 'argsort':

        if isinstance(shape, int): shape = (shape, )
        if shape is None:
            init = np.array(init, copy=False)
            shape = init.shape

        return Param(
            TYPE = 'argsort',
            sampler = ArgsortPermutationSampler(choices = choices),
            shape = (*shape, len(choices)),
            low = -1,
            high = 1,
            step = None,
            scale = scale,
            domain = 'linear',
            oob_penalty = oob_penalty,
            normalize = normalize,
            fallback_low = -1,
            fallback_high = 1,
            init = init,
            options = options,
        )
    else: raise NotImplementedError(f'type {type} is not valid.')

def _permutation_update(
    param: Param,
    choices: int | ArrayLike,
    scale: Optional[float],
    normalize: bool,

    type: Literal['argsort'],
    oob_penalty: Optional[float],
):
    if isinstance(choices, int): choices = range(choices)
    if type == 'argsort':
        param._update(
            low = -1,
            high = 1,
            step = None,
            scale = scale,
            domain = 'linear',
            oob_penalty = oob_penalty,
            normalize = normalize,
            fallback_low = -1,
            fallback_high = 1,
        )
    else: raise NotImplementedError(f'type {type} is not valid.')
