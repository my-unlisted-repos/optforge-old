r'''
:any:`optforge.study` - Study interface.

:any:`study` - Study interface.

:any:`Study` - Study interface.

`optforge.interfaces.minimize` - Minimize interface.

`optforge.optim` - Optimizers

`optforge.benchmark` - Bptimizers

`optforge.param` - Parameter spaces
'''

#pylint:disable=C0413
import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

#from . import interfaces
from . import benchmarks, optim, param as p, study, scheduler, pruners
from .interfaces import *
from .study import Study
from .trial import Trial, FixedTrial
from . import scheduler as s
from .registry.optimizers import OPTIMIZERS
from . import integrations