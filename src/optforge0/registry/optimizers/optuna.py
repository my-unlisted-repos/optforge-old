from collections.abc import Sequence

from ..groups import GROUPS
from ..lib import Lib
from .optimizers import OPTIMIZERS


class LibOptuna(Lib):
    def __init__(self):
        super().__init__(
            names = ('optuna',),
            requires = 'optuna',
        )

    def initialize(self):
        from ...integrations.optuna.optuna_optimizer import (
            OptunaSampler, _optuna_names, get_all_optuna_samplers)
        opts = get_all_optuna_samplers()

        # optuna reexports same optimizers to samplers and integrations
        already_set = set()

        BLACKLIST = {'GridSampler', "PartialFixedSampler", "MOTPESampler"} # MOTPE is deprecated
        for opt in opts:
            if (
                opt.__name__ not in BLACKLIST
                and (not opt.__name__.startswith("_"))
                and opt.__name__ not in already_set
            ):
                already_set.add(opt.__name__)
                names = _optuna_names(opt)
                self.register(
                    OptunaSampler.configured(opt).set_name(names),
                    maxdims=1000,
                )

        # try:
        #     import optuna_integration
        # TODO: add optuna hub????
lib_optuna = LibOptuna()
OPTIMIZERS.register_lib(lib_optuna)

