from typing import TYPE_CHECKING, Optional

import numpy as np

from .optimizer import Config, Optimizer
from ..scheduler import SchedulableFloat, SchedulableInt

if TYPE_CHECKING:
    from ..paramdict import ParamDict
    from ..trial import Trial

__all__ = [
    "FDSA",
]

class FDSA(Optimizer):
    CONFIG = Config(
        supports_ask=True,
        supports_multiple_asks=False,
        requires_batch_mode=True,
    )
    names = 'FDSA'
    def __init__(self, lr:SchedulableFloat = 1e-2, c:SchedulableFloat = 1e-3):
        defaults = dict(lr = lr, c = c)
        super().__init__(defaults)

    def ask(self, study):

        for i, (p, store) in enumerate(self.yield_params_stores()):
            for idx in range(p.data.size):
                pos_params = self.params.copy()
                pos_params_list = list(pos_params.values())
                pos_params_list[i].data.flat[idx] += store['c']
                pos_params.storage['i'] = i
                pos_params.storage['idx'] = idx
                pos_params.storage['mode'] = 'pos'
                yield pos_params

                neg_params = self.params.copy()
                neg_params_list = list(neg_params.values())
                neg_params_list[i].data.flat[idx] -= store['c']
                neg_params.storage['i'] = i
                neg_params.storage['idx'] = idx
                neg_params.storage['mode'] = 'neg'
                yield neg_params


        for p, store in self.yield_params_stores():
            store['pos'] = np.zeros_like(p.data)
            store['neg'] = np.zeros_like(p.data)

    def tell(self, trials: "list[Trial]", study, ):
        params = list(self.params.values())
        for trial in trials:
            p = params[trial.params.storage['i']]
            p.store[trial.params.storage['mode']].flat[trial.params.storage['idx']] = trial.value

        for p, store in self.yield_params_stores():
            p.data -= store['lr'] * ((store['pos'] - store['neg']) / (store['c'] * 2))


    def tell_not_asked(self, trial: "Trial", study):
        if trial.improved:
            self.params.update(trial.params)

FDSA.register()
FSDABig = FDSA.configured(1e-1, 1e-2).set_name('FDSA-big', register=True)
FSDASmall = FDSA.configured(1e-3, 1e-4).set_name('FDSA-small', register=True)
FSDATiny = FDSA.configured(1e-4, 1e-5).set_name('FDSA-tiny', register=True)