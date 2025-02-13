import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

from .groups import GROUPS
from ..python_tools import NicePartial, RelaxedMultikeyDict


def _check_installed(name: str):
    try:
        importlib.import_module(name)
        return True
    except ModuleNotFoundError: return False


class LibObject:
    def __init__(self, obj: Callable, groups: str | Sequence[str], maxdims: Optional[int]):
        self.obj = obj
        if isinstance(groups, str): groups = (groups, )
        self.groups = set(groups)

        self.maxdims = maxdims

    @property
    def name(self):
        return self.obj.name # type:ignore

    def __call__(self):
        return self.obj()

    def __repr__(self):
        return repr(self.obj)

class Lib:
    def __init__(self, names: str | Sequence[str], requires: Optional[str | Sequence[str]],):
        if isinstance(names, str): names = (names, )
        self.names: Sequence[str] = names
        if isinstance(requires, str): requires = (requires, )
        self.requires: Sequence[str] | None = requires
        self.objects:RelaxedMultikeyDict[LibObject] = RelaxedMultikeyDict()
        self.initialized: bool = False
        self.installed: bool = False

    def initialize(self):
        """This should ADD to the optimizers (because some may already be registered)"""

    def _internal_initialize(self):
        if self.initialized: raise ValueError(f'{self.__class__.__name__} already initialized')
        if self.requires is not None and len(self.requires) > 0:
            self.installed = all(_check_installed(name) for name in self.requires)
        if self.installed: self.initialize()
        self.initialized = True

    def register(
        self,
        obj: Any,
        groups: str | Sequence[str] = (GROUPS.MAIN, ),
        maxdims = None,
    ):
        # RelaxedMultikeyDict will raise error if any of those names are already registered
        # (which is good)
        self.objects[obj.names] = LibObject(obj, groups, maxdims=maxdims) # type:ignore
        obj.lib = self.names[0]

    def __getitem__(self, name:str): return self.objects[name]
    def __setitem__(self, name:str | Sequence[str], obj:LibObject): self.objects[name] = obj
    def __delitem__(self, name:str): del self.objects[name]
    def __contains__(self, name:str | Sequence[str]): return name in self.objects
    def __iter__(self): return iter(self.objects)
    def __len__(self): return len(self.objects)
    def keys(self): return self.objects.keys()
    def values(self): return self.objects.values()
    def items(self): return self.objects.items()
