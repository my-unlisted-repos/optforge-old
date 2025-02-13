from typing import TYPE_CHECKING, Optional

import numpy as np

from ..rng import RNG
from ..scheduler import SchedulableFloat, SchedulableInt
from ..trial import Trial
from .optimizer import Config, ConfiguredOptimizer, Optimizer

__all__ = [
    "SPSA",
    "RDSA"
]

def rademacher(rng: RNG, shape):
    return rng.numpy.choice([-1, 1], size=shape)

class SPSA(Optimizer):
    CONFIG = Config(
        supports_ask = True,
        supports_multiple_asks = False,
        requires_batch_mode = True,
    )
    names = 'SPSA'
    def __init__(
        self,
        lr:SchedulableFloat = 3e-2,
        c:SchedulableFloat = 1e-2,
        avg_steps:SchedulableInt = 1,
        max_diff: Optional[SchedulableFloat] = 1e-2,
        min_diff: Optional[SchedulableFloat] = 1e-8,
        sampler = rademacher,
        seed: Optional[int] = None
    ):
        defaults = dict(lr=lr, c=c, sampler = sampler, max_diff=max_diff, min_diff=min_diff)
        super().__init__(defaults, seed = seed)

        self.avg_steps = self.schedule(avg_steps)

    def ask(self, study):
        for step in range(int(self.avg_steps())):
            # add petrubation
            pos_params = self.params.copy()
            for param, store in self.yield_params_stores(pos_params):
                store['petrubation'] = store['sampler'](self.rng, param.data.shape)
                param.data += store['petrubation'] * store['c']
            yield pos_params

            # sub petrubation
            neg_params = pos_params.copy()
            for param, store in self.yield_params_stores(neg_params):
                param.data -= store['petrubation'] * (2 * store['c'])
            yield neg_params

    def tell(self, trials: list[Trial], study):
        # calculate grads
        pos, neg = trials[::2], trials[1::2]
        for step, (pos_eval, neg_eval) in enumerate(zip(pos, neg)):
            delta_loss = pos_eval.scalar_value - neg_eval.scalar_value
            for store, pos_store in zip(self.yield_stores(), self.yield_stores(pos_eval.params)):
                p_delta_loss = float(delta_loss)
                if abs(p_delta_loss) < store['min_diff']: p_delta_loss = np.sign(p_delta_loss) * store['min_diff']
                if abs(p_delta_loss) > store['max_diff']: p_delta_loss = np.sign(p_delta_loss) * store['max_diff']
                grad = p_delta_loss / (pos_store['petrubation'] * 2 * store['c'])
                if step == 0: store['grad'] = grad
                else: store['grad'] += grad

        # average grads
        if len(pos) > 1:
            for param, store in self.yield_params_stores():
                store['grad'] /= len(pos)

        # sub grads
        for param, store in self.yield_params_stores():
            param.data -= store['grad'] * store['lr']
            

    def tell_not_asked(self, trial: "Trial", study):
        if trial.improved:
            self.params.update(trial.params)

SPSA.register()
SPSABig = SPSA.configured(lr = 1e-1, c = 1e-2).set_name('SPSA-big', register=True)
SPSASmall = SPSA.configured(lr = 1e-3, c = 1e-4).set_name('SPSA-small', register=True)


class RDSA(Optimizer):
    CONFIG = Config(
        supports_ask = True,
        supports_multiple_asks = False,
        requires_batch_mode = True,
    )
    names = 'RDSA'
    def __init__(
        self,
        lr: SchedulableFloat = 1e-1,
        c: SchedulableFloat = 1e-2,
        avg_steps: SchedulableInt = 1,
        max_diff: Optional[SchedulableFloat] = 1e-2,
        min_diff: Optional[SchedulableFloat] = 1e-8,
        sampler = None,
        seed: Optional[int] = None
    ):
        defaults = dict(lr=lr, c=c, sampler = sampler, max_diff=max_diff, min_diff=min_diff)
        super().__init__(defaults, seed = seed)

        self.avg_steps = self.schedule(avg_steps)

    def ask(self, study):
        for step in range(int(self.avg_steps())):
            # add petrubation
            pos_params = self.params.copy()
            for param, store in self.yield_params_stores(pos_params):
                sampler = store['sampler']
                if sampler is None: store['petrubation'] = param.sample_generate_petrubation(store['c'])
                else:  store['petrubation'] = store['sampler'](self.rng, param.data.shape) * store['c']
                param.data += store['petrubation']
            yield pos_params

            # sub petrubation
            neg_params = pos_params.copy()
            for param, store in self.yield_params_stores(neg_params):
                param.data -= store['petrubation'] * 2
            yield neg_params

    def tell(self, trials: list[Trial], study):
        # calculate grads
        pos, neg = trials[::2], trials[1::2]
        for step, (pos_eval, neg_eval) in enumerate(zip(pos, neg)):
            delta_loss = pos_eval.scalar_value - neg_eval.scalar_value
            for store, pos_store in zip(self.yield_stores(), self.yield_stores(pos_eval.params)):
                p_delta_loss = float(delta_loss)
                if abs(p_delta_loss) < store['min_diff']: p_delta_loss = np.sign(p_delta_loss) * store['min_diff']
                if abs(p_delta_loss) > store['max_diff']: p_delta_loss = np.sign(p_delta_loss) * store['max_diff']
                grad = (pos_store['petrubation'] * p_delta_loss) / (2 * store['c'])
                if step == 0: store['grad'] = grad
                else: store['grad'] += grad

        # average grads
        if len(pos) > 1:
            for param, store in self.yield_params_stores():
                store['grad'] /= len(pos)

        # sub grads
        for param, store in self.yield_params_stores():
            param.data -= store['grad'] * store['lr']

    def tell_not_asked(self, trial: "Trial", study):
        if trial.improved:
            self.params.update(trial.params)

RDSA.register()
RDSABig = RDSA.configured(lr = 5e-1, c = 1e-1).set_name('RDSA-big', register=True)
RDSASmall = RDSA.configured(lr = 1e-2, c = 1e-3).set_name('RDSA-small', register=True)
