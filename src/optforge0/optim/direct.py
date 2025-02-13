from typing import TYPE_CHECKING, Optional, Literal
import bisect
import numpy as np

from .optimizer import Config, Optimizer

if TYPE_CHECKING:
    from ..study import Study
    from ..trial import Trial
class _Hypercube:
    value: float
    def __init__(self, x1:np.ndarray, x2:np.ndarray):
        self.x1:np.ndarray = x1
        self.x2:np.ndarray = x2

    def set_value(self, value:float):
        self.value:float = value
        return self

    def central_point(self):
        return (self.x1 + self.x2) / 2

    def __repr__(self):
        return f'Hypercube({self.value = }, {self.x1 = }, {self.x2 = })'

class DIRECT(Optimizer):
    CONFIG = Config(
        supports_ask=True,
        supports_multiple_asks=True,
        requires_batch_mode=False,
    )
    names = ('DIRECT', )
    def __init__(
        self,
        mode: Literal['best', 'weighted', 'random', 'sequential'] = 'best',
        num: int = 10,
        coord_mode: Literal['random', 'sequential', 'batched'] = 'batched',
        shuffle = False,
        seed: Optional[int] = None
    ):
        super().__init__(seed = seed)
        self.hypercubes: list[_Hypercube] = []
        self.mode = mode
        self.coord_mode = coord_mode
        self.num = num
        self.shuffle = shuffle

        self.cur = 0

    def _weights(self):
        weights = np.array([i.value for i in self.hypercubes])
        weights -= weights.min()
        max = weights.max()
        if max == 0: return np.full_like(weights, 1/weights.size)
        weights = 1 - (weights / max)
        return weights / weights.sum()

    def ask(self, study: "Study"):
        # on first iteration initialize to entire search space, evaluate middle point
        if len(self.hypercubes) == 0:
            self.lb = np.array(self.params.get_lower_bounds(fallback='default'))
            self.ub = np.array(self.params.get_upper_bounds(fallback='default'))
            _, self.slices = self.params.params_to_vec()
            new_params = self.params.copy()
            new_params.vec_to_params_((self.lb + self.ub) / 2, self.slices)
            new_params.storage['hypercube'] = _Hypercube(self.lb, self.ub)
            yield new_params

        else:
            # pick hypercubes
            if self.mode == 'best':
                hypercubes = self.hypercubes[:self.num]
            elif self.mode in ('random', 'weighted'):
                hypercubes = self.rng.numpy.choice(
                    np.array(self.hypercubes),
                    p = self._weights() if self.mode == 'weighted' else None,
                    size = min(len(self.hypercubes), self.num)
                )
            elif self.mode == 'sequential':
                hypercubes = self.hypercubes
            else: raise ValueError(f'Invalid mode: {self.mode}')

            if self.shuffle:
                hypercubes = list(hypercubes)
                self.rng.random.shuffle(hypercubes)

            for hypercube in hypercubes:
                # randomly pick a dimension to split in
                if self.coord_mode == 'random': split_dim = self.rng.random.randrange(0, self.lb.size)
                elif self.coord_mode in ('sequential', 'batched'):
                    split_dim = self.cur % self.lb.size
                    if self.coord_mode == 'sequential': self.cur += 1
                else: raise ValueError(f'Invalid coord_mode {self.coord_mode}')

                # we split the hypercube into 3. Central inherit existing value, and lower and higher ones are evaluated
                # -- create the lower hypercube --
                # first corner (x1) will be of original hypercube
                # second corner will be 1/3 in `split_dim`
                lower_params = self.params.copy()
                lower_x2 = hypercube.x2.copy()
                lower_x2[split_dim] = hypercube.x1[split_dim] * (2/3) + hypercube.x2[split_dim] * (1/3)
                lower_params.storage['hypercube'] = lh = _Hypercube(hypercube.x1, lower_x2)
                lower_params.vec_to_params_(lh.central_point(), self.slices)
                yield lower_params

                # -- create the higher hypercube --
                # first corner (x1) will be 2/3 in `split_dim`
                # second corner will be of original hypercube
                higher_params = self.params.copy()
                higher_x1 = hypercube.x1.copy()
                higher_x1[split_dim] = hypercube.x1[split_dim] * (1/3) + hypercube.x2[split_dim] * (2/3)
                higher_params.storage['hypercube'] = hh = _Hypercube(higher_x1, hypercube.x2)
                higher_params.vec_to_params_(hh.central_point(), self.slices)
                yield higher_params

                # set best hypercube to central
                hypercube.x1 = lower_x2
                hypercube.x2 = higher_x1

            if self.coord_mode == 'batched': self.cur += 1


    def tell(self, trials: "list[Trial]", study: "Study"):
        for t in trials:
            # insort the new hypercube into self.hypercubes
            # meaning self.hypercubes will always be sorted by value in asecnding order
            # first hypercube will always be the best
            new_hypercube: _Hypercube = t.params.storage['hypercube'].set_value(t.scalar_value)
            if self.mode in ('best', 'weighted'):
                bisect.insort(self.hypercubes, new_hypercube, key = lambda x: x.value)
                if self.mode == 'best': self.hypercubes = self.hypercubes[:self.num]
            else:
                self.hypercubes.append(new_hypercube)


DIRECTBest1 = DIRECT.configured('best', num = 1).set_name(('DIRECT-best1'), register=True)