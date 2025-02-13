from collections.abc import Sequence

from ..groups import GROUPS
from ..lib import Lib
from .optimizers import OPTIMIZERS

class LibPyPop7(Lib):
    def __init__(self):
        super().__init__(
            names = ('pypop7', 'pp7'),
            requires = 'pypop7',
        )

    def initialize(self):
        from ...integrations.pypop7.pypop7_optimizer import (
            PyPop7Optimizer, _pypop7_names, get_all_pypop7_optimizers)
        opts = get_all_pypop7_optimizers()

        # those are abstract
        BLACKLIST = {
            "bo.BO",
            "cc.CC",
            "cem.CEM",
            "de.DE",
            "ds.DS",
            "eda.EDA",
            "ep.EP",
            "es.ES",
            "ga.GA",
            "nes.NES",
            "pso.PSO",
            "rs.RS",
            "sa.SA",
        }

        for opt in opts:
            names = _pypop7_names(opt)
            if names[0] not in BLACKLIST: self.register(
                PyPop7Optimizer.configured(opt).set_name(names)
            )

lib_pypop7 = LibPyPop7()
OPTIMIZERS.register_lib(lib_pypop7)

# PyPop7.ga.G3PCX failed with ValueError('cannot convert float NaN to integer')
# PyPop7.pso.CCPSO2 failed with AssertionError()
# PyPop7.pso.CPSO failed with ValueError('Maximum allowed size exceeded')