from abc import ABC, abstractmethod
from collections.abc import Sequence, MutableSequence
from typing import overload, Optional
from functools import partial
import numpy as np


class Transform(ABC):
    @overload
    def forward(self, x:float) -> float: ...
    @overload
    def forward(self, x:np.ndarray) -> np.ndarray: ...
    @overload
    def forward(self, x:None) -> None: ...
    @abstractmethod
    def forward(self, x: float | np.ndarray | None) -> float | np.ndarray | None: ...

    @overload
    def backward(self, x:float) -> float: ...
    @overload
    def backward(self, x:np.ndarray) -> np.ndarray: ...
    @overload
    def backward(self, x:None) -> None: ...
    @abstractmethod
    def backward(self, x: float | np.ndarray | None) -> float | np.ndarray | None: ...

    def seq_forward(self, seq: Sequence[float | np.ndarray | None]) -> MutableSequence[float | np.ndarray | None]:
        return [self.forward(i) for i in seq]

    def seq_backward(self, seq: Sequence[float | np.ndarray | None]) -> MutableSequence[float | np.ndarray | None]:
        return [self.backward(i) for i in seq]

class ScaleTransform(Transform):
    def __init__(self, scale: float): self.scale = scale
    @overload
    def forward(self, x:float) -> float: ...
    @overload
    def forward(self, x:np.ndarray) -> np.ndarray: ...
    @overload
    def forward(self, x:None) -> None: ...
    def forward(self, x: float | np.ndarray | None) -> float | np.ndarray | None:
        return x * self.scale if x is not None else None

    @overload
    def backward(self, x:float) -> float: ...
    @overload
    def backward(self, x:np.ndarray) -> np.ndarray: ...
    @overload
    def backward(self, x:None) -> None: ...
    def backward(self, x: float | np.ndarray | None) -> float | np.ndarray | None:
        return x / self.scale if x is not None else None

class NormalizeTransform(Transform):
    def __init__(self, low: Optional[float], high: Optional[float]):
        self.low = low
        self.high = high

    @overload
    def forward(self, x:float) -> float: ...
    @overload
    def forward(self, x:np.ndarray) -> np.ndarray: ...
    @overload
    def forward(self, x:None) -> None: ...
    def forward(self, x):
        if x is None: return x
        if self.low is None or self.high is None: return x
        return (2*x - self.high - self.low) / (self.high - self.low)

    @overload
    def backward(self, x:float) -> float: ...
    @overload
    def backward(self, x:np.ndarray) -> np.ndarray: ...
    @overload
    def backward(self, x:None) -> None: ...
    def backward(self, x):
        if x is None: return None
        if self.low is None or self.high is None: return x
        return (x * (self.high - self.low) + (self.high + self.low)) / 2


class _BaseNLog:
    def __init__(self, base: float): self.base = base
    def __call__(self, x): return np.emath.logn(self.base, x)

class LogTransform(Transform):
    def __init__(self, base: float):
        self.base = base
        if base == 10: self.fn = np.log10
        elif base == 2: self.fn = np.log2
        elif base == np.e: self.fn = np.log
        else: self.fn = _BaseNLog(base)

    @overload
    def forward(self, x:float) -> float: ...
    @overload
    def forward(self, x:np.ndarray) -> np.ndarray: ...
    @overload
    def forward(self, x:None) -> None: ...
    def forward(self, x):
        if x is None: return None
        return self.fn(x)

    @overload
    def backward(self, x:float) -> float: ...
    @overload
    def backward(self, x:np.ndarray) -> np.ndarray: ...
    @overload
    def backward(self, x:None) -> None: ...
    def backward(self, x):
        return self.base ** x

