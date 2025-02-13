from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from .. import scheduler
from ..pareto import _compare_pareto
from ..scheduler import SchedulableFloat
from ..trial import Trial
from .optimizer import Config, Optimizer

if TYPE_CHECKING:
    from ..study import Study

__all__ = [
    "RandomSearch",
    "StochasticHillClimbing",
    "AcceleratedRandomSearch",
    "RandomStepSize",
    "RandomAnnealing",
    "CyclicRandomSearch",
]



class RandomSearch(Optimizer):
    CONFIG = Config(
        supports_ask = True,
        supports_multiple_asks = True,
        requires_batch_mode = False,
    )
    names = 'Random search', 'RS'

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)

    def ask(self, study: "Study"):
        new_params = self.params.copy()
        for param in self.yield_params(new_params):
            param.data = param.sample_random()
        yield new_params
RandomSearch.register()

def _stochastic_tell(opt: "StochasticHillClimbing | Any", trials:list[Trial], study):
    for t in trials:
        if opt.only_viable and opt.cur_viable and not t.is_viable: continue
        front, is_better = _compare_pareto(t.value, opt.best_value + opt.threshold())
        if is_better:
            opt.best_value = front
            opt.params.update(t.params)
            opt.cur_viable = t.is_viable
        if opt.compare == 'last':
            opt.best_value = t.value
            opt.cur_viable = t.is_viable

class StochasticHillClimbing(Optimizer):
    CONFIG = Config(
        supports_ask = True,
        supports_multiple_asks = True,
        requires_batch_mode = False,
    )
    names = 'Stochastic hill climbing', 'SHC'

    def __init__(
        self,
        sigma: SchedulableFloat = 0.1,
        threshold: SchedulableFloat = 0.,
        compare: Literal["lowest", "last"] = "lowest",
        only_viable = True,
        seed: Optional[int] = None,
    ):
        defaults = dict(sigma = sigma)
        super().__init__(defaults, seed = seed)

        self.compare = compare
        self.only_viable = only_viable
        self.threshold = self.schedule(threshold)
        self.cur_viable = False

        self.best_value: float | np.ndarray = float('inf')

    def ask(self, study):
        new_params = self.params.copy()
        for param, store in self.yield_params_stores(new_params, only_used=True):
            param.data = param.sample_petrub(store['sigma'])
        yield new_params

    def tell(self, trials:list[Trial], study): _stochastic_tell(self, trials, study)


StochasticHillClimbing.register()
SHCSmall = StochasticHillClimbing.configured(sigma=0.01).set_name('SHC-small', register=True)
SHCTiny = StochasticHillClimbing.configured(sigma=0.001).set_name('SHC-tiny', register=True)
SHCLast = StochasticHillClimbing.configured(compare='last').set_name('SHC-last', register=True)
SHCSmallLast = (StochasticHillClimbing
                .configured(sigma=0.01, compare='last')
                .set_name('SHC-small-last', register=True))
ThresholdAccepting = (StochasticHillClimbing
                      .configured(sigma = 1e-2, threshold = 1e-1)
                      .set_name(('Threshold accepting', 'TA'), register=True))
SHCUnordered = StochasticHillClimbing.conf_unordered().set_name('SHC-unordered', register=True)
SHCSmallUnordered = StochasticHillClimbing.conf_unordered(sigma=0.01).set_name('SHC-small-unordered', register=True)

class RandomAnnealing(StochasticHillClimbing):
    names = 'Random annealing', 'RA'

    def __init__(
        self,
        start: float = 2,
        stop: float = 1e-10,
        threshold: SchedulableFloat = 0.0,
        only_viable = True,
        budget=None,
        seed: Optional[int] = None,
    ):
        super().__init__(sigma=scheduler.Linear(start, stop), threshold = threshold, only_viable=only_viable, seed = seed)
        self.budget = budget
RandomAnnealing.register()
RASmall = RandomAnnealing.configured(start=0.5).set_name('RA-small', register=True)
RATiny = RandomAnnealing.configured(start=0.1).set_name('RA-tiny', register=True)
RASmallUnordered = RandomAnnealing.conf_unordered(start=0.5).set_name('RA-small-unordered', register=True)

class AcceleratedRandomSearch(StochasticHillClimbing):
    names = 'Accelerated Random Search', 'ARS'
    def __init__(
        self,
        mul: SchedulableFloat = 0.9,
        high: SchedulableFloat = 2,
        restart_steps: SchedulableFloat = 100,
        threshold: SchedulableFloat = 0.,
        only_viable = True,
        seed: Optional[int] = None,
    ):
        super().__init__(
            sigma=scheduler.AcceleratedAnnealing(high=high, mul=mul, restart_steps=restart_steps,),
            threshold = threshold,
            only_viable = only_viable,
            seed = seed,
        )
AcceleratedRandomSearch.register()
ARSAnnealing = (AcceleratedRandomSearch
                .configured(high = scheduler.Linear.configured(2, 0))
                .set_name(('ARS-annealing', 'ARSA'), register=True)
                )
ARSSmall = AcceleratedRandomSearch.configured(high = 0.1).set_name('ARS-small', register = True)
ARSUnordered = AcceleratedRandomSearch.conf_unordered(high=1).set_name('ARS-unordered', register=True)

class RandomStepSize(StochasticHillClimbing):
    names = 'Random step size'

    def __init__(
        self,
        low: SchedulableFloat = 1e-10,
        high: SchedulableFloat = 2,
        threshold: SchedulableFloat = 0.0,
        only_viable = True,
        seed: Optional[int] = None,
    ):
        super().__init__(sigma=scheduler.Random(low, high), threshold = threshold, only_viable=only_viable, seed = seed)
RandomStepSize.register()

class Pulsating(StochasticHillClimbing):
    names = 'Pulsating'

    def __init__(
        self,
        low: SchedulableFloat = 0,
        high: SchedulableFloat = 2,
        period: SchedulableFloat = 0.1,
        threshold: SchedulableFloat = 0.,
        only_viable = True,
        budget = None,
        seed: Optional[int] = None,
    ):
        super().__init__(sigma=scheduler.Sine(low, high, period), threshold = threshold, only_viable=only_viable, seed = seed)
        self.budget = budget
Pulsating.register()

class PulsatingAnnealing(Pulsating):
    names = 'Pulsating annealing', 'PA'

    def __init__(self, threshold: SchedulableFloat = 0., budget = None, only_viable = True, seed: Optional[int] = None):
        super().__init__(
            high = scheduler.Linear(2, 1e-10,),
            period = scheduler.Linear(0.3, 1e-10),
            threshold = threshold,
            only_viable=only_viable,
            budget = budget,
            seed = seed,
            )
PulsatingAnnealing.register()

class CyclicRandomSearch(Optimizer):
    CONFIG = Config(
        supports_ask = True,
        supports_multiple_asks = True,
        requires_batch_mode = False,
    )
    names = 'Cyclic random search', 'CRS'

    def __init__(
        self,
        threshold: SchedulableFloat = 0.,
        compare: Literal["lowest", "last"] = "lowest",
        only_viable = True,
        seed: Optional[int] = None,
        ):
        super().__init__(seed = seed)
        self.threshold = self.schedule(threshold)
        self.compare = compare
        self.only_viable = only_viable
        self.best_value: float | np.ndarray = float('inf')
        self.cur_viable = False

        self.cur_param = 0
        self.cur_coord = 0

    def ask(self, study: "Study"):
        new_params = self.params.copy()
        plist = list(new_params.values())

        if self.cur_param == len(plist):
            self.cur_param = 0
            self.cur_coord = 0

        param = plist[self.cur_param]
        if self.cur_coord == param.data.size:
            self.cur_coord = 0
            self.cur_param += 1
            if self.cur_param >= len(plist): self.cur_param = 0
            param = plist[self.cur_param]

        param.data.flat[self.cur_coord] = param.sample_random().flat[self.cur_coord]
        self.cur_coord += 1
        yield new_params

    def tell(self, trials:list[Trial], study): _stochastic_tell(self, trials, study)
CyclicRandomSearch.register()
CRSLast = CyclicRandomSearch.configured(compare='last').set_name('CRS-last', register = True)

class BestCoordinateRandomSearch(Optimizer):
    CONFIG = Config(
        supports_ask = True,
        supports_multiple_asks = True,
        requires_batch_mode = False,
    )

    names = 'Best-coordinate random search', 'BCRS'

    def __init__(
        self,
        threshold: SchedulableFloat = 0.,
        only_viable = True,
        seed: Optional[int] = None,
        ):
        super().__init__(seed = seed)
        self.threshold = self.schedule(threshold)
        self.only_viable = only_viable
        self.best_value: float | np.ndarray = float('inf')

    def ask(self, study: "Study"):
        cur_param = 0; cur_coord = 0
        while True:
            new_params = self.params.copy()
            plist = list(new_params.values())

            if cur_param >= len(plist): break

            param = plist[cur_param]
            if cur_coord >= param.data.size:
                cur_coord = 0
                cur_param += 1
                if cur_param >= len(plist): break
                param = plist[cur_param]

            param.data.flat[cur_coord] = param.sample_random().flat[cur_coord]
            cur_coord += 1
            yield new_params

    def tell(self, trials:list[Trial], study): _stochastic_tell(self, trials, study)
BestCoordinateRandomSearch.register()