"""Registry of optimizers, similar to nevergrad.
The point of this is that you can iterate over this and test every single optimization algorithm on your problem.
This is also used to quickly access optimizers by their string name without imports."""

import difflib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

from ..python_tools import RelaxedMultikeyDict, str_norm
from .groups import GROUPS
from .lib import Lib


class Registry:
    def __init__(self):
        self.data: RelaxedMultikeyDict[Lib] = RelaxedMultikeyDict()

    def register_lib(self, lib: Lib):
        self.data[lib.names] = lib

    def __getitem__(self, item: str) -> Callable:
        # get lib and item
        # for example, ng.Portfolio
        if '.' in item: lib_name, object_name = item.split('.', 1)
        else: lib_name, object_name = 'optforge', item

        # lib is not recognized
        if lib_name not in self.data:
            matches = difflib.get_close_matches(str_norm(lib_name), self.data.all_relaxed_keys(), n=3, cutoff=0.)
            matches = [self.data.relaxed_to_orig(key) for key in matches]
            raise KeyError(f'Library "{lib_name}" not found. Did you mean {matches}?')

        # get lib, initialize if necessary and check if it is installed
        lib:Lib = self.data[lib_name]
        if not lib.initialized: lib._internal_initialize()
        if not lib.installed: raise ModuleNotFoundError(f'{lib} requires {lib.requires}, which are not installed.')

        # item is not recognized
        if object_name not in lib.objects:
            #matches = [f'{lib_name}.{i}' for i in difflib.get_close_matches(str_norm(object_name), lib.objects.all_relaxed_keys(), n=3, cutoff=0.)]
            matches = difflib.get_close_matches(str_norm(object_name), lib.objects.all_relaxed_keys(), n=3, cutoff=0.)
            matches = '"' + '", "'.join([f'{lib_name}.{lib.objects.relaxed_to_orig(key)}' for key in matches]) + '"'
            raise KeyError(f'Object "{item}" not found. Did you mean {matches}?')

        return lib.objects[object_name]

    def __contains__(self, item: str) -> bool:
        return item in self.data

    def keys(self, groups: Optional[str | Sequence[str]] = None):
        if isinstance(groups, str): groups = [groups]
        for lib_name, lib in self.data.items():
            if not lib.initialized: lib._internal_initialize()
            if lib.installed:
                for obj_name, obj in lib.objects.items():
                    if groups is None or any(i in obj.groups for i in groups): yield f'{lib_name}.{obj_name}'

    def values(self, groups: Optional[str | Sequence[str]] = None):
        if isinstance(groups, str): groups = [groups]
        for lib in self.data.values():
            if not lib.initialized: lib._internal_initialize()
            if lib.installed:
                for obj in lib.objects.values():
                    if groups is None or any(i in obj.groups for i in groups): yield obj

    def items(self, groups: Optional[str | Sequence[str]] = None):
        if isinstance(groups, str): groups = [groups]
        for lib_name, lib in self.data.items():
            if not lib.initialized: lib._internal_initialize()
            if lib.installed:
                for obj_name, obj in lib.objects.items():
                    if groups is None or any(i in obj.groups for i in groups): yield (f'{lib_name}.{obj_name}', obj)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(len(lib.objects) for lib in self.data.values())

    def sorted_keys(self, groups: Optional[str | Sequence[str]] = None):
        return sorted(self.keys(groups))

    def sorted_items(self, groups: Optional[str | Sequence[str]] = None):
        return sorted(self.items(groups), key = lambda x: x[0])

    def sorted_values(self, groups: Optional[str | Sequence[str]] = None):
        return [i[1] for i in self.sorted_items(groups)]

    def sorted_dict(self, groups: Optional[str | Sequence[str]] = None):
        return dict(self.sorted_items(groups))

    def register(
        self,
        obj: Callable,
        lib: str,
        groups: str | Sequence[str] = (GROUPS.MAIN, ),
        maxdims: Optional[int] = None,
    ):
        lib_obj: Lib = self.data[lib]
        lib_obj.register(obj, groups = groups, maxdims=maxdims)