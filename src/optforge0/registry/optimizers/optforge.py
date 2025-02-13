from ..groups import GROUPS
from ..lib import Lib
from .optimizers import OPTIMIZERS

# ---------------------------------------------------------------------------- #
#                                   OPTFORGE                                   #
# ---------------------------------------------------------------------------- #
class LibOptforge(Lib):
    def __init__(self):
        super().__init__(
            names = ('optforge', 'of'),
            requires = None,
        )
        self.initialized = True
        self.installed = True

    # does nothing
    def initialize(self): pass

lib_optforge = LibOptforge()
OPTIMIZERS.register_lib(lib_optforge)

