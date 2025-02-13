#pylint:disable = W0641
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ...optim.optimizer import Config, Optimizer
from ...python_tools import get__name__, reduce_dim

if TYPE_CHECKING:
  from PyXAB.algos.Algo import Algorithm
  from PyXAB.partition.Partition import Partition


__all__ = [
    "PyXABOptimizer",
    "get_all_pyxab_algos",
]

class PyXABOptimizer(Optimizer):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode=False,
    )
    wrapped_optimizer: "Algorithm | None"
    lib = 'PyXAB'
    def __init__(
        self,
        opt_cls: "Callable[..., Algorithm]",
        partition: "Optional[type[Partition]]" = None,
        budget: Optional[int] = None
    ):
        super().__init__()
        self.budget = budget

        self.opt_cls: "Callable[..., Algorithm]" = opt_cls
        self.partition = partition
        self.algo = None
        self.T = 1

        if not hasattr(self, 'names'): self.names = [get__name__(self.opt_cls), ]

    def step(self, study):
        if self.algo is None:
            if self.partition is None:
                from PyXAB.partition.BinaryPartition import BinaryPartition
                self.partition = BinaryPartition

            # pass budget which can be under different argument
            params = inspect.signature(self.opt_cls).parameters
            if 'rounds' in params: kwargs = {"rounds": self.budget}
            elif 'n' in params: kwargs = {"n": self.budget}
            else: kwargs = {}

            self.algo = self.opt_cls(**kwargs, domain = self.params.get_bounds(), partition=self.partition)
            self.slices = self.params.params_to_vec()[1]

        point = self.algo.pull(self.T)
        reward = study.evaluate_vec(np.array(point), slices=self.slices, return_scalar=True)
        self.algo.receive_reward(time = self.T, reward = -reward)
        
        
def get_all_pyxab_algos():
    from PyXAB.algos.DOO import DOO
    from PyXAB.algos.GPO import GPO
    from PyXAB.algos.HCT import HCT
    from PyXAB.algos.HOO import T_HOO
    from PyXAB.algos.PCT import PCT
    from PyXAB.algos.POO import POO
    from PyXAB.algos.SequOOL import SequOOL
    from PyXAB.algos.SOO import SOO
    from PyXAB.algos.StoSOO import StoSOO
    from PyXAB.algos.StroquOOL import StroquOOL
    from PyXAB.algos.VHCT import VHCT
    from PyXAB.algos.VPCT import VPCT
    from PyXAB.algos.VROOM import VROOM
    from PyXAB.algos.Zooming import Zooming
    return list(locals().copy().values())