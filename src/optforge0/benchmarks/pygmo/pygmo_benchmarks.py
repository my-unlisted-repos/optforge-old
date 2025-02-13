# pylint: disable = E1101
from typing import TYPE_CHECKING, Literal
import numpy as np
from ..benchmark import Benchmark
from ..._utils import _ensure_float_or_1darray
if TYPE_CHECKING:
    from ...trial import Trial

__all__ = [
    "PygmoProblem",
    "PygmoCEC2013",
    "PygmoCEC2006",
    "PygmoGolombRuler",
    "PygmoZDT",
    "PygmoInventory",
    "PygmoLennardJonesCluster",

]
class PygmoProblem(Benchmark):
    def __init__(self, problem, log_params = True, note = None):
        super().__init__(log_params = log_params, note = note)
        import pygmo as pg
        self.prob = pg.problem(problem) # type:ignore
        lb, ub = self.prob.get_bounds()
        if np.unique(lb).size > 1: raise ValueError(f'{self.__class__.__name__} has different lower bounds for different dimensions')
        if np.unique(ub).size > 1: raise ValueError(f'{self.__class__.__name__} has different upper bounds for different dimensions')
        self.lb = lb[0]; self.ub = ub[0]
        self.shape = len(self.lb)

        self.nobj = self.prob.get_nobj()
        self.nec = self.prob.get_nec()
        self.nic = self.prob.get_nic()

    def objective(self, trial: "Trial"):
        x = trial.suggest_array('X', self.shape, low = self.lb, high = self.ub)
        res = self.prob.fitness(x)
        if res.size == self.nobj: return float(res)

        obj = res[:self.nobj]
        if self.nec > 0:
            nec = res[self.nobj:self.nobj+self.nec]
            trial.soft_penalty(np.abs(nec), constr_name='nec')
        if self.nic > 0:
            nic = res[self.nobj+self.nec:]
            trial.constr_bounds(nic, high = 0, constr_name='nic')
        return obj

    def get_minima_params(self): return {'X': self.prob.best_known()}
    def get_minima(self): return _ensure_float_or_1darray(self.evaluate_params(self.get_minima_params()))


class PygmoCEC2013(PygmoProblem):
    def __init__(
        self,
        prob_id: Literal[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28] = 1,
        dim: Literal[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] = 2,
        log_params = True,
        note = None
    ):
        """The CEC 2013 problems: Real-Parameter Single Objective Optimization Competition.
        The 28 problems of the competition on real-parameter single objective optimization problems that was organized for the 2013 IEEE Congress on Evolutionary Computation.
        All problems are box-bounded, continuous, single objective problems.
        """
        import pygmo as pg
        prob = pg.cec2013(prob_id, dim) # type:ignore
        super().__init__(prob, log_params = log_params, note = note)

class PygmoCEC2006(PygmoProblem):
    def __init__(
        self,
        prob_id: Literal[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,28] = 1,
        log_params = True,
        note = None
    ):
        """The CEC 2006 problems: Constrained Real-Parameter Optimization.
        All problems are constrained, continuous, single objective problems.
        """
        import pygmo as pg
        prob = pg.cec2006(prob_id,) # type:ignore
        super().__init__(prob, log_params = log_params, note = note)

class PygmoCEC2009(PygmoProblem):
    def __init__(
        self,
        prob_id: Literal[1,2,3,4,5,6,7,8,9,10] = 1,
        is_constrained = False,
        dim: int = 30,
        log_params = True,
        note = None
    ):
        """The CEC 2009 problems: Competition on “Performance Assessment of Constrained / Bound
        Constrained Multi-Objective Optimization Algorithms”.
        This class instantiates any of the problems from CEC2009’s competition on multi-objective optimization algorithms,
        commonly referred to by the literature as UF1-UF10 (unconstrained) and CF1-CF10 (constrained).
        All problems are continuous, multi objective problems.
        """
        import pygmo as pg
        prob = pg.cec2009(prob_id, is_constrained, dim) # type:ignore
        super().__init__(prob, log_params = log_params, note = note)

class PygmoGolombRuler(PygmoProblem):
    def __init__(self, order, upper_bound, log_params = True, note = None):
        import pygmo as pg
        prob = pg.golomb_ruler(order, upper_bound) # type:ignore
        super().__init__(prob, log_params = log_params, note = note)

class PygmoZDT(PygmoProblem):
    def __init__(self, prob_id: Literal[1,2,3,4,5,6], param=30, log_params = True, note = None):
        """Set of two objective problems. 
        Recommended dims per problem id are [30, 30, 30, 10, 11, 10]. 
        See https://esa.github.io/pagmo2/docs/cpp/problems/zdt.html#_CPPv4N5pagmo3zdtE

        :param prob_id: problem number. Must be in [1, .., 6]
        :param param: problem parameter, representing the problem dimension except for ZDT5 where it represents the number of binary strings, defaults to 30
        :param log_params: _description_, defaults to True
        :param note: _description_, defaults to None
        """
        import pygmo as pg
        prob = pg.zdt(prob_id, param) # type:ignore
        super().__init__(prob, log_params = log_params, note = note)

class PygmoDTLZ(PygmoProblem):
    def __init__(self, prob_id: Literal[1,2,3,4,5,6,7] = 1, dim=5, fdim=3, alpha=100, log_params = True, note = None):
        """See https://esa.github.io/pagmo2/docs/cpp/problems/dtlz.html#_CPPv4N5pagmo4dtlzE

        :param prob_id: Problem ID, defaults to Literal[1,2,3,4,5,6,7]
        :param dim: Dimension of the problem, defaults to 5
        :param fdim: Number of objectives, defaults to 3
        :param alpha: controls density of solutions (used only by DTLZ4), defaults to 100
        :param log_params: _description_, defaults to True
        :param note: _description_, defaults to None
        """
        import pygmo as pg
        prob = pg.dtlz(prob_id, dim, fdim, alpha) # type:ignore
        super().__init__(prob, log_params = log_params, note = note)

class PygmoInventory(PygmoProblem):
    def __init__(self, weeks=4, sample_size=10, log_params = True, note = None):
        import pygmo as pg
        prob = pg.inventory(weeks, sample_size) # type:ignore
        super().__init__(prob, log_params = log_params, note = note)

class PygmoLennardJonesCluster(Benchmark):
    def __init__(self, atoms=150, log_params = True, note = None):
        super().__init__(log_params = log_params, note = note)
        import pygmo as pg
        self.prob = pg.problem(pg.lennard_jones(atoms)) # type:ignore
        self.lb = self.prob.get_bounds()[0] + 3 # type:ignore
        self.num = len(self.lb)

    def objective(self, trial: "Trial"):
        x = trial.suggest_array('X', self.num, low = -3, high = 3) + self.lb
        return self.lb.fitness(x)


class PygmoHockSchittkowskyNo71(PygmoProblem):
    def __init__(self, log_params = True, note = None):
        """Hock Schittkowsky No.71"""
        import pygmo as pg
        prob = pg.hock_schittkowski_71() # type:ignore
        super().__init__(prob, log_params = log_params, note = note)


class LuksanVlcek1(PygmoProblem):
    def __init__(self, dim = 3, log_params = True, note = None):
        import pygmo as pg
        prob = pg.luksan_vlcek1(dim) # type:ignore
        super().__init__(prob, log_params = log_params, note = note)


class PygmoWFG(PygmoProblem):
    def __init__(self, prob_id: Literal[1,2,3,4,5,6,7,8,9]=1, dim_dvs=5, dim_obj=3, dim_k=4, log_params = True, note = None):
        """See https://esa.github.io/pagmo2/docs/cpp/problems/wfg.html#_CPPv4N5pagmo3wfgE

        :param prob_id: problem number. Must be in [1, …, 9]., defaults to 1
        :param dim_dvs: decision vector dimension, defaults to 5
        :param dim_obj: objective function dimension, defaults to 3
        :param dim_k: position parameter. This parameter influences the shape functions of the various problems., defaults to 4
        :param log_params: _description_, defaults to True
        :param note: _description_, defaults to None
        """
        import pygmo as pg
        prob = pg.wfg(prob_id, dim_dvs, dim_obj, dim_k) # type:ignore
        super().__init__(prob, log_params = log_params, note = note)