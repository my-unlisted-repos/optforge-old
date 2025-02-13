from collections.abc import Sequence

from ..groups import GROUPS
from ..lib import Lib
from .optimizers import OPTIMIZERS


def _chain_recursive_check(ng, cls, blacklist):
    """Check for problematic optimizers recursively."""
    import nevergrad as ng
    if any(cls == i for i in blacklist): return True

    # chaining
    for metacls, key in (
    (ng.optimizers.Chaining, 'optimizers'),
    (ng.optimizers.ConfPortfolio, 'optimizers'),
    (ng.optimizers.ParametrizedMetaModel, 'multivariate_optimizer'),
    (ng.optimizers.Rescaled, 'base_optimizer')):
        if isinstance(cls, metacls):
            optimizers = cls._config[key]
            if not isinstance(optimizers, Sequence): optimizers = (optimizers,)
            if any(i in optimizers for i in blacklist): return True
            chains = [i for i in optimizers if isinstance(i, ng.optimizers.Chaining)]
            for i in chains:
                if _chain_recursive_check(ng, i, blacklist): return True

    return False



class LibNevergrad(Lib):
    def __init__(self):
        super().__init__(
            names = ('nevergrad', 'ng'),
            requires = 'nevergrad',
        )

    def initialize(self):
        import nevergrad as ng

        from ...integrations.nevergrad.nevergrad_optimizer import (
            NevergradOptimizer, _nevergrad_names)

        # cobyla seems to randomly freeze, sometimes after working fine
        # TODO: test at lower dims
        FREEZELIST = {'Cobyla', 'NGOpt16', 'NGOpt36', 'NGOptSingle25'}

        for name,cls in ng.optimizers.registry.items():
            repr_name = _nevergrad_names(cls)[0]

            groups = (GROUPS.MAIN, )

            if name == 'pCarola6':
                name = 'PCarola6-2' # there is also PCarola6
                repr_name = 'Rescaled.PCarola6-2'

            #if _chain_recursive_check(ng, cls, FREEZELIST): groups = (GROUPS.FREEZES,)
            if name in FREEZELIST: groups = (GROUPS.FREEZES,)

            self.register(
                NevergradOptimizer.configured(cls).set_name([repr_name, name]),
                groups = groups
            )

        #self.objects['AX'].groups = (GROUPS.NO_FORCE_STOP, GROUPS.FREEZES,) # threads keep raising errors way past force stop

lib_nevergrad = LibNevergrad()
OPTIMIZERS.register_lib(lib_nevergrad)
