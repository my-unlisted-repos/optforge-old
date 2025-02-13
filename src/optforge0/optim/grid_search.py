from typing import TYPE_CHECKING

import numpy as np

from ..scheduler import SchedulableInt
from .optimizer import Config, Optimizer

if TYPE_CHECKING:
    from ..study import Study


__all__ = [
    "GridSearch",
    "CoordinateGridSearch",
    "CoordinateGridSearchAnnealing",
]



class GridSearch(Optimizer):
    CONFIG = Config(
        supports_ask = True,
        supports_multiple_asks = True,
        requires_batch_mode = False,
    )
    names = 'Grid Search', 'GS'

    def __init__(self):
        super().__init__()
        self._ask_iterator = None

    def _ask_generator(self, study: "Study"):
        new_params = self.params.copy()
        paramlist = list(new_params.values())

        # initialize to lowest value
        for p in paramlist:
            store = p.store
            if 'init' not in store:
                low = p.low; high = p.high; step = p.step
                if low is None: low = -1
                if high is None: high = 1
                # set to lowest value
                p.data = np.full_like(p.data, low)

                # set step
                store['step'] = (high - low) / 2
                if step is not None: store['step'] = max(store['step'] - (store['step'] % step), step)

                store['add_term'] = 0

            yield new_params

        # yield next grid search point


        param_idx = 0
        value_idx = 0
        while True:
            # reached end, halve step size
            if param_idx == len(paramlist):
                param_idx = 0
                for p in paramlist:
                    store = p.store
                    store['step'] /= 2
                    pstep = p.step
                    if pstep is not None: store['step'] = store['step'] - (store['step'] % pstep)
                    if store['add_term'] == 0: store['add_term'] = store['step'] / 2
                    else: store['add_term'] = store['add_term'] / 2

                    if pstep is not None and store['step'] < pstep:
                        store['step'] = pstep
                        store['add_term'] = 0

            p = paramlist[param_idx]
            store = p.store
            low = p.low; high = p.high; step = p.step
            if low is None: low = -1
            if high is None: high = 1
            step = store['step']


            # reached end of param, move to next param
            if value_idx == p.data.size:
                value_idx = 0
                param_idx += 1
                continue

            current_value = p.data.flat[value_idx]
            # print(f'{param_idx = }')
            # print(f'{value_idx = }')
            # print(f'{low = }')
            # print(f'{high = }')
            # print(f'{step = }')
            # print(f'{current_value = }')
            # print(f'{step = }')
            # print(f'{store['add_term'] = }')
            # print(f'{new_params = }')
            # print()
            # time.sleep(0.1)
            if current_value + step > high:
                p.data.flat[value_idx] = min(low + store['add_term'], high)
                value_idx += 1
            else:
                p.data.flat[value_idx] += step
                value_idx = 0
                param_idx = 0
                yield new_params

    def ask(self, study: "Study"):
        if self._ask_iterator is None: self._ask_iterator = iter(self._ask_generator(study))
        yield next(self._ask_iterator).copy()
GridSearch.register()

class CoordinateGridSearch(Optimizer):
    CONFIG = Config(
        supports_ask = True,
        supports_multiple_asks = False,
        requires_batch_mode = True,
    )
    names = 'Coordinate Grid Search', 'CGS'

    def __init__(self, num: SchedulableInt = 100):
        super().__init__()
        self.cur_param = 0
        self.cur_coord = 0

        self.num = self.schedule(num)

    def ask(self, study: "Study"):
        plist = list(self.params.values())

        if self.cur_param >= len(plist):
            self.cur_param = 0
            self.cur_coord = 0

        param = plist[self.cur_param]
        if self.cur_coord >= param.data.size:
            self.cur_coord = 0
            self.cur_param += 1
            if self.cur_param >= len(plist): self.cur_param = 0
            param = plist[self.cur_param]

        low = param.low; high = param.high; step = param.step
        if low is None: low = -1
        if high is None: high = 1
        num = min(self.num(), 100000)
        selfstep = (high - low) / num
        if step is not None: selfstep = max(selfstep - (selfstep % step), step)

        for i in np.arange(low, high, selfstep):
            new_params = self.params.copy()
            new_plist = list(new_params.values())
            new_p = new_plist[self.cur_param]
            new_p.data.flat[self.cur_coord] = i
            yield new_params

        self.cur_coord += 1

CoordinateGridSearch.register()
SGS10 = CoordinateGridSearch.configured(num=10).set_name('SGS-10', register=True)

class CoordinateGridSearchAnnealing(CoordinateGridSearch):
    names = 'CGS-annealing', 'CGSA'

    def __init__(self):
        super().__init__(num=1)
        self.cur_step = 0

    def ask(self, study: "Study"):
        yield from super().ask(study)
        self.cur_step += 1
        if self.cur_step % self.params.numel() == 0: self.num.value *= 2 # type:ignore
CoordinateGridSearchAnnealing.register()

def _check_prime(x):
    for i in range(2, (x // 2) + 1):
        if x % i == 0: return False
    return True

def _next_prime(x):
    while True:
        x += 1
        if _check_prime(x): return x
class CoordinateGridSearchSlowAnnealing(CoordinateGridSearch):
    names = 'CGS-slow-annealing', 'CGSSA'

    def __init__(self):
        super().__init__(num=1)
        self.cur_step = 0

    def ask(self, study: "Study"):
        yield from super().ask(study)
        self.cur_step += 1
        if self.cur_step % self.params.numel() == 0: 
            if self.num.value <= 3: self.num.value += 1 # type:ignore
            else: self.num.value = _next_prime(self.num.value)
CoordinateGridSearchSlowAnnealing.register()
