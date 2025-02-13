from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Generic, ParamSpec, Protocol, TypeVar

import numpy as np

Numeric = int | float | np.ndarray
ArrayLike = Sequence[Any] | np.ndarray
NumericArrayLike = Sequence[Numeric] | np.ndarray
Index = int | slice | np.ndarray | Sequence[int] | Sequence[bool]

class SamplerType(Protocol):
    # Define types here, as if __call__ were a function (ignore self).
    def __call__(self, low: float, high: float, size: int | tuple[int, ...]) -> np.ndarray: ...


# ---------------------------------------------------------------------------- #
#                           IDE type hints converters                          #
# ---------------------------------------------------------------------------- #
# the following can copy IDE type hints from one function to another, and add or remove arguments
# decorate a function with f2f_removex_addy to remove x leftmost arguments and add y new arguments to the left
# if you decorate method with another method, you'd typically use at least f2f_remove1_add1, to remove old self and add new self

_P = ParamSpec("_P")
_T = TypeVar("_T")
_S = TypeVar("_S")
_R_co = TypeVar("_R_co", covariant=True)

class FuncWithArgs1(Protocol, Generic[_P, _R_co]):
    def __get__(self, instance: Any, owner: type | None = None) -> Callable[_P, _R_co]:...
    def __call__(self, __: Any, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ... # pylint:disable=E1101,E0213 #type:ignore
class FuncWithArgs2(Protocol, Generic[_P, _R_co]):
    def __get__(self, instance: Any, owner: type | None = None) -> Callable[_P, _R_co]:...
    def __call__(self, __1: Any, __2:Any, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ... # pylint:disable=E1101,E0213 #type:ignore
class FuncWithArgs3(Protocol, Generic[_P, _R_co]):
    def __get__(self, instance: Any, owner: type | None = None) -> Callable[_P, _R_co]:...
    def __call__(self, __1: Any, __2:Any,__3:Any, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ... # pylint:disable=E1101,E0213 #type:ignore
class FuncWithArgs4(Protocol, Generic[_P, _R_co]):
    def __get__(self, instance: Any, owner: type | None = None) -> Callable[_P, _R_co]:...
    def __call__(self, __1: Any, __2:Any,__3:Any,__4:Any, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ... # pylint:disable=E1101,E0213 #type:ignore



def f2f(_: Callable[_P, _T]) -> Callable[[Callable[_P, _S]], Callable[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> Callable[_P, _S]:
        return fnc
    return _fnc

def f2f_add1(_: Callable[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs1[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs1[_P, _S]:
        return fnc # type:ignore
    return _fnc

def f2f_remove1_add1(_: FuncWithArgs1[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs1[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs1[_P, _S]:
        return fnc # type:ignore
    return _fnc

def f2f_remove1(_: FuncWithArgs1[_P, _T]) -> Callable[[Callable[_P, _S]], Callable[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> Callable[_P, _S]:
        return fnc # type:ignore
    return _fnc

def f2f_remove2_add1(_: FuncWithArgs2[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs1[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs1[_P, _S]:
        return fnc # type:ignore
    return _fnc

def f2f_remove3_add1(_: FuncWithArgs3[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs1[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs1[_P, _S]:
        return fnc # type:ignore
    return _fnc

def f2f_remove3_add2(_: FuncWithArgs3[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs2[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs2[_P, _S]:
        return fnc # type:ignore
    return _fnc

def f2f_remove4_add1(_: FuncWithArgs4[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs1[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs1[_P, _S]:
        return fnc # type:ignore
    return _fnc

def f2f_remove4_add2(_: FuncWithArgs4[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs2[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs2[_P, _S]:
        return fnc # type:ignore
    return _fnc