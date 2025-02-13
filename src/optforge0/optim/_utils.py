from collections.abc import Sequence, Callable
from typing import TYPE_CHECKING, Optional, Any

from ..registry.groups import GROUPS
from ..registry.optimizers import OPTIMIZERS

if TYPE_CHECKING:
    from .optimizer import ConfiguredOptimizer, Optimizer


def _register(
    cls: "type[Optimizer] | ConfiguredOptimizer",
    lib: Optional[str],
    groups: str | Sequence[str],
    maxdims: Optional[int],
):
    """Register this optimizer to the optimizers registry.
    You can iterate over every single optimizer with `registry.keys()` or `registry.values()`.
    Registry holds classes or constructors, not instances. For consistency all registered constructors
    require no arguments, so that they can be iterated over effortlesly."""
    if lib is None: lib = 'optforge'
    if lib.lower() in ('optforge', 'of'): cls.lib = None
    else: cls.lib = lib
    OPTIMIZERS.register(cls, lib = lib, groups=groups, maxdims=maxdims)
    return cls

def _set_name(
    cls: "type[Optimizer] | ConfiguredOptimizer",
    names: str | Sequence[str],
    register: bool, # pylint:disable=W0621
    lib: Optional[str],
    groups: str | Sequence[str],
    maxdims: Optional[int],
):
    if isinstance(names, str): names = (names, )
    cls.names = list(names)
    if register: cls.register(lib=lib, groups=groups, maxdims=maxdims)
    return cls