#pylint:disable = W0707, W0621

from typing import Literal, Optional

import numpy as np

from ..._types import Numeric
from ...optim.optimizer import Config, Optimizer
from ...paramdict import ParamDict
from ...python_tools import reduce_dim

__all__ = [
    "NLoptWrapper",
    "ALL_ALGOS"
]

_ALGOS_LITERAL = Literal[
    "GN_DIRECT",  # = _nlopt.GN_DIRECT
    "GN_DIRECT_L",  # = _nlopt.GN_DIRECT_L
    "GN_DIRECT_L_RAND",  # = _nlopt.GN_DIRECT_L_RAND
    "GN_DIRECT_NOSCAL",  # = _nlopt.GN_DIRECT_NOSCAL
    "GN_DIRECT_L_NOSCAL",  # = _nlopt.GN_DIRECT_L_NOSCAL
    "GN_DIRECT_L_RAND_NOSCAL",  # = _nlopt.GN_DIRECT_L_RAND_NOSCAL
    "GN_ORIG_DIRECT",  # = _nlopt.GN_ORIG_DIRECT
    "GN_ORIG_DIRECT_L",  # = _nlopt.GN_ORIG_DIRECT_L
    "GD_STOGO",  # = _nlopt.GD_STOGO
    "GD_STOGO_RAND",  # = _nlopt.GD_STOGO_RAND
    "LD_LBFGS_NOCEDAL",  # = _nlopt.LD_LBFGS_NOCEDAL
    "LD_LBFGS",  # = _nlopt.LD_LBFGS
    "LN_PRAXIS",  # = _nlopt.LN_PRAXIS
    "LD_VAR1",  # = _nlopt.LD_VAR1
    "LD_VAR2",  # = _nlopt.LD_VAR2
    "LD_TNEWTON",  # = _nlopt.LD_TNEWTON
    "LD_TNEWTON_RESTART",  # = _nlopt.LD_TNEWTON_RESTART
    "LD_TNEWTON_PRECOND",  # = _nlopt.LD_TNEWTON_PRECOND
    "LD_TNEWTON_PRECOND_RESTART",  # = _nlopt.LD_TNEWTON_PRECOND_RESTART
    "GN_CRS2_LM",  # = _nlopt.GN_CRS2_LM
    "GN_MLSL",  # = _nlopt.GN_MLSL
    "GD_MLSL",  # = _nlopt.GD_MLSL
    "GN_MLSL_LDS",  # = _nlopt.GN_MLSL_LDS
    "GD_MLSL_LDS",  # = _nlopt.GD_MLSL_LDS
    "LD_MMA",  # = _nlopt.LD_MMA
    "LN_COBYLA",  # = _nlopt.LN_COBYLA
    "LN_NEWUOA",  # = _nlopt.LN_NEWUOA
    "LN_NEWUOA_BOUND",  # = _nlopt.LN_NEWUOA_BOUND
    "LN_NELDERMEAD",  # = _nlopt.LN_NELDERMEAD
    "LN_SBPLX",  # = _nlopt.LN_SBPLX
    "LN_AUGLAG",  # = _nlopt.LN_AUGLAG
    "LD_AUGLAG",  # = _nlopt.LD_AUGLAG
    "LN_AUGLAG_EQ",  # = _nlopt.LN_AUGLAG_EQ
    "LD_AUGLAG_EQ",  # = _nlopt.LD_AUGLAG_EQ
    "LN_BOBYQA",  # = _nlopt.LN_BOBYQA
    "GN_ISRES",  # = _nlopt.GN_ISRES
    "AUGLAG",  # = _nlopt.AUGLAG
    "AUGLAG_EQ",  # = _nlopt.AUGLAG_EQ
    "G_MLSL",  # = _nlopt.G_MLSL
    "G_MLSL_LDS",  # = _nlopt.G_MLSL_LDS
    "LD_SLSQP",  # = _nlopt.LD_SLSQP
    "LD_CCSAQ",  # = _nlopt.LD_CCSAQ
    "GN_ESCH",  # = _nlopt.GN_ESCH
    "GN_AGS",  # = _nlopt.GN_AGS
]

def _nlopt_str(algo: int, local_algo: int) -> list[str]:
    import nlopt

    name = "unknown"
    for attr in dir(nlopt):
        if getattr(nlopt, attr) == algo:
            name = attr
            break

    if "MLSL" in name or "AUGLAG" in name:
        local_name = "unknown"
        for attr in dir(nlopt):
            if getattr(nlopt, attr) == local_algo:
                local_name = attr
                break
        return [
            f"{name}.{local_name}",
        ]

    return [
        name,
    ]


class NLoptWrapper(Optimizer):
    CONFIG = Config(
        supports_ask=False,
        supports_multiple_asks=False,
        requires_batch_mode=False,

    )
    def __init__(
        self,
        algorithm: int | _ALGOS_LITERAL,
        approx_grad = True,
        grad_eps = 1e-5,
        local_algo: int | _ALGOS_LITERAL = "LN_NELDERMEAD", # nelder mead
        seed = None,
        ):
        super().__init__(seed = seed)
        import nlopt
        self._best_value = float('inf')

        if isinstance(algorithm, str): algorithm = getattr(nlopt, algorithm.upper())
        self.algorithm: int = algorithm # type:ignore

        if isinstance(local_algo, str): local_algo = getattr(nlopt, local_algo.upper())
        self.local_algo: int = local_algo # type:ignore

        self.approx_grad = approx_grad
        self.grad_eps = grad_eps
        if approx_grad:
            import scipy.optimize
            self.so = scipy.optimize
        else: self.so = None

        if not hasattr(self, 'names'): self.names = _nlopt_str(self.algorithm, self.local_algo)

    def set_params(self, params:ParamDict):
        super().set_params(params)
        self._init_params()
        return self

    def set_seed(self, seed: Optional[int]):
        import nlopt
        nlopt.srand(seed)
        return super().set_seed(seed)

    def _init_params(self):
        import nlopt
        self.nlopt = nlopt
        if self.rng.seed is not None: nlopt.srand(self.rng.seed)

        self.param_lengths = []
        self.lb = []
        self.ub = []
        param_data = []

        name = _nlopt_str(self.algorithm, self.local_algo)[0]
        for param in self.yield_params():
            self.param_lengths.append(param.data.size)
            low = param.low if param.low is not None else (np.inf if not name.startswith('G') else param.get_required_low())
            high = param.high if param.high is not None else (np.inf if not name.startswith('G') else param.get_required_high())

            self.lb.extend([low] * param.data.size)
            self.ub.extend([high] * param.data.size)

            param_data.append(param.data.ravel())

        self.x0 = np.asanyarray(reduce_dim(param_data))
        self.save_param_data('best')

        self._init_optimizer()


    def _init_optimizer(self):
        self.opt = self.nlopt.opt(self.algorithm, self.x0.size)
        self.opt.set_lower_bounds(self.lb)
        self.opt.set_upper_bounds(self.ub)

        self.lopt = self.nlopt.opt(self.local_algo, self.x0.size)
        self.lopt.set_lower_bounds(self.lb)
        self.lopt.set_upper_bounds(self.ub)
        # this does nothing in most cases but works for optimizers that use other optimizers
        self.opt.set_local_optimizer(self.lopt)

        if self.budget is not None:
            self.opt.set_maxeval(self.budget)
            self.lopt.set_maxeval(self.budget)

    def _closure(self, x:np.ndarray):
        cur = 0
        for param, plen in zip(self.params.values(), self.param_lengths):
            param.data = x[cur:cur+plen].reshape(param.data.shape)
            cur += plen
        self.last_value = float(self.closure())
        if self.last_value < self._best_value:
            self._best_value = self.last_value
            self.save_param_data('best') # save best params
        return self.last_value

    def _grad_closure(self, x:np.ndarray, grad:np.ndarray):
        if self.approx_grad and grad.size > 0:
            g = self.so.approx_fprime(x, self._closure, epsilon = self.grad_eps) # type:ignore
            grad[:] = g
        return self._closure(x)

    def step(self, study):
        try:
            self.closure = study.evaluate_return_scalar
            self.opt.set_min_objective(self._grad_closure)
            self.xopt = self.opt.optimize(self.x0)
            return self.opt.last_optimum_value()
        except (RuntimeError, self.nlopt.RoundoffLimited, SystemError, ) as e: #raise EndStudy
            self.load_param_data('best')
            self._init_params()
        except ValueError as e:
            # this seems to happen when gradients are zero, so we randomize params
            self.params.randomize()
            self._init_params()

ALL_ALGOS: list[_ALGOS_LITERAL] = [
    "GN_DIRECT",  # = _nlopt.GN_DIRECT
    "GN_DIRECT_L",  # = _nlopt.GN_DIRECT_L
    "GN_DIRECT_L_RAND",  # = _nlopt.GN_DIRECT_L_RAND
    "GN_DIRECT_NOSCAL",  # = _nlopt.GN_DIRECT_NOSCAL
    "GN_DIRECT_L_NOSCAL",  # = _nlopt.GN_DIRECT_L_NOSCAL
    "GN_DIRECT_L_RAND_NOSCAL",  # = _nlopt.GN_DIRECT_L_RAND_NOSCAL
    "GN_ORIG_DIRECT",  # = _nlopt.GN_ORIG_DIRECT
    "GN_ORIG_DIRECT_L",  # = _nlopt.GN_ORIG_DIRECT_L
    "GD_STOGO",  # = _nlopt.GD_STOGO
    "GD_STOGO_RAND",  # = _nlopt.GD_STOGO_RAND
    "LD_LBFGS_NOCEDAL",  # = _nlopt.LD_LBFGS_NOCEDAL
    "LD_LBFGS",  # = _nlopt.LD_LBFGS
    "LN_PRAXIS",  # = _nlopt.LN_PRAXIS
    "LD_VAR1",  # = _nlopt.LD_VAR1
    "LD_VAR2",  # = _nlopt.LD_VAR2
    "LD_TNEWTON",  # = _nlopt.LD_TNEWTON
    "LD_TNEWTON_RESTART",  # = _nlopt.LD_TNEWTON_RESTART
    "LD_TNEWTON_PRECOND",  # = _nlopt.LD_TNEWTON_PRECOND
    "LD_TNEWTON_PRECOND_RESTART",  # = _nlopt.LD_TNEWTON_PRECOND_RESTART
    "GN_CRS2_LM",  # = _nlopt.GN_CRS2_LM
    "GN_MLSL",  # = _nlopt.GN_MLSL  # !has local
    "GD_MLSL",  # = _nlopt.GD_MLSL  # !has local
    "GN_MLSL_LDS",  # = _nlopt.GN_MLSL_LDS  # !has local
    "GD_MLSL_LDS",  # = _nlopt.GD_MLSL_LDS  # !has local
    "LD_MMA",  # = _nlopt.LD_MMA
    "LN_COBYLA",  # = _nlopt.LN_COBYLA
    "LN_NEWUOA",  # = _nlopt.LN_NEWUOA
    "LN_NEWUOA_BOUND",  # = _nlopt.LN_NEWUOA_BOUND
    "LN_NELDERMEAD",  # = _nlopt.LN_NELDERMEAD
    "LN_SBPLX",  # = _nlopt.LN_SBPLX
    "LN_AUGLAG",  # = _nlopt.LN_AUGLAG
    "LD_AUGLAG",  # = _nlopt.LD_AUGLAG
    "LN_AUGLAG_EQ",  # = _nlopt.LN_AUGLAG_EQ
    "LD_AUGLAG_EQ",  # = _nlopt.LD_AUGLAG_EQ
    "LN_BOBYQA",  # = _nlopt.LN_BOBYQA
    "GN_ISRES",  # = _nlopt.GN_ISRES
    "AUGLAG",  # = _nlopt.AUGLAG
    "AUGLAG_EQ",  # = _nlopt.AUGLAG_EQ
    "G_MLSL",  # = _nlopt.G_MLSL
    "G_MLSL_LDS",  # = _nlopt.G_MLSL_LDS
    "LD_SLSQP",  # = _nlopt.LD_SLSQP
    "LD_CCSAQ",  # = _nlopt.LD_CCSAQ
    "GN_ESCH",  # = _nlopt.GN_ESCH
    "GN_AGS",  # = _nlopt.GN_AGS
]