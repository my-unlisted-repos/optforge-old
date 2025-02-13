from collections.abc import Sequence

from ..groups import GROUPS
from ..lib import Lib
from .optimizers import OPTIMIZERS


class LibOpytimizer(Lib):
    def __init__(self):
        super().__init__(
            names = ('opytimizer', 'opy'),
            requires = 'opytimizer',
        )

    def initialize(self):
        from ...integrations.opytimizer.opytimizer_optimizer import (
            OpytimizerWrapper, _opytimizer_names,
            get_all_opytimizer_optimizers)
        opts = get_all_opytimizer_optimizers()
        for opt in opts:

            for i in (1, 2, 5, 10, 20, 50, 100, 200, 500):
                if i == 10: groups = GROUPS.MAIN
                else: groups = GROUPS.EXTRA
                names = [f'{n}-{i}' for n in _opytimizer_names(opt)]
                self.register(
                    OpytimizerWrapper.configured(opt, n_agents=i).set_name(names),
                    groups=groups,
                )
lib_opytimizer = LibOpytimizer()
OPTIMIZERS.register_lib(lib_opytimizer)

# TODO
# find how to use boolean optimizers
# Opytimizer.evolutionary.GP-1 failed with AttributeError("'SearchSpace' object has no attribute 'trees'")
# Opytimizer.science.HGSO-100 failed with IndexError('too many indices for array: array is 1-dimensional, but 2 were indexed')
# (it works but fails after a while)
#  Opytimizer.science.MOA-10 failed with SizeError: `n_agents` should have a perfect square.

