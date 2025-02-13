from collections.abc import Callable, Sequence
from functools import partial
from typing import Mapping, Optional

from ..groups import GROUPS
from ..lib import Lib, LibObject
from ..registry import Registry, RelaxedMultikeyDict

OPTIMIZERS = Registry()



# ---------------------------------------------------------------------------- #
#                           Gradient-Free-Optimizers                           #
# ---------------------------------------------------------------------------- #
class LibGradientFreeOptimizers(Lib):
    def __init__(self):
        super().__init__(
            names = ('gradient_free_optimizers', 'GFO'),
            requires = 'gradient_free_optimizers',
        )

    def initialize(self):
        from ...integrations.gradient_free_optimizers.gfo_optimizer import (
            GFOWrapper, _gfo_names, get_all_gfo_optimizers)
        opts = get_all_gfo_optimizers()

        BLACKLIST = {'EnsembleOptimizer',}
        SLOWER = {'BayesianOptimizer', 'LipschitzOptimizer', 'TreeStructuredParzenEstimators', 'ForestOptimizer'}
        for opt in opts:
            if opt.__name__ not in BLACKLIST:
                if opt.__name__  in SLOWER: kwargs = {'continuous_space_steps': 100}
                elif opt.__name__ == 'GridSearchOptimizer': kwargs = {'continuous_space_steps': 10}
                else: kwargs = {}
                names = _gfo_names(opt)
                self.register(
                    GFOWrapper.configured(opt, **kwargs).set_name(names), # type:ignore
                    maxdims=1000,
                )

lib_gradient_free_optimizers = LibGradientFreeOptimizers()
OPTIMIZERS.register_lib(lib_gradient_free_optimizers)


# ---------------------------------------------------------------------------- #
#                                   Optimagic                                  #
# ---------------------------------------------------------------------------- #
class LibOptimagic(Lib):
    def __init__(self):
        super().__init__(
            names = ('optimagic', 'om'),
            requires = 'optimagic',
        )

    def initialize(self):
        from ...integrations.optimagic import (OptimagicMinimize,
                                               get_all_optimagic_algorithms)
        algos = get_all_optimagic_algorithms()

        BLACKLIST = { }
        DUPLICATES = { }
        for algo in algos:
            if algo.__name__ in BLACKLIST: groups = GROUPS.BROKEN
            elif algo.__name__ in DUPLICATES or 'Nlopt' in algo.__name__: groups = GROUPS.DUPLICATE
            elif algo.__name__ in ('NelderMeadParallel', 'ScipyDirect'): groups = GROUPS.NO_FORCE_STOP
            else: groups = GROUPS.MAIN
            self.register(
                OptimagicMinimize.configured(algo).set_name(algo.__name__),
                groups=(groups, )
            )

lib_optimagic = LibOptimagic()
OPTIMIZERS.register_lib(lib_optimagic)



# ---------------------------------------------------------------------------- #
#                                    mystic                                    #
# ---------------------------------------------------------------------------- #
class LibMystic(Lib):
    def __init__(self):
        super().__init__(
            names = ('mystic', ),
            requires = 'mystic',
        )

    def initialize(self):
        from ...integrations.mystic.mystic_optimizer import (  # MysticBuckshot,; MysticLattice,; MysticSparsity,
            MysticDE, MysticNelderMead, MysticPowell, MysticStornPriceDE)

        #self.register(MysticBuckshot)
        self.register(MysticDE)
        self.register(MysticStornPriceDE)
        self.register(MysticNelderMead)
        self.register(MysticPowell)
        #self.register(MysticLattice)
        #self.register(MysticSparsity)

lib_mystic = LibMystic()
OPTIMIZERS.register_lib(lib_mystic)



# ---------------------------------------------------------------------------- #
#                                    Mealpy                                    #
# ---------------------------------------------------------------------------- #
class LibMealpy(Lib):
    def __init__(self):
        super().__init__(
            names = ('mealpy', 'mp'),
            requires = 'mealpy',
        )

    def initialize(self):
        import mealpy

        from ...integrations.mealpy.mealpy_optimizer import (MealpyOptimizer,
                                                             _mealpy_names)
        for name, cls in mealpy.get_all_optimizers().items():
            # we get full name like `swarm_based.ABC.OriginalABC':
            names = _mealpy_names(cls)

            # default is 100 so we set it to None
            # because maybe I missed some optimizers that have a different default (?).
            for i in (1, 2, 5, 10, 20, 50, 100, 200, 500):
                inames = [f'{n}-{i}' if i is not None else n for n in names]
                if i is None: groups = GROUPS.MAIN
                else: groups = GROUPS.EXTRA
                self.register(
                    MealpyOptimizer.configured(cls, pop_size = i).set_name(inames),
                    groups = groups
                )

lib_mealpy = LibMealpy()
OPTIMIZERS.register_lib(lib_mealpy)

# ---------------------------------------------------------------------------- #
#                                 Directsearch                                 #
# ---------------------------------------------------------------------------- #
class LibDirectsearch(Lib):
    def __init__(self):
        super().__init__(
            names = ('directsearch', 'ds'),
            requires = 'directsearch',
        )

    def initialize(self):
        from ...integrations.directsearch import (DS_STP, DSBase,
                                                  DSNoSketching,
                                                  DSProbabilistic, DSSubspace)
        self.register(DSBase)
        self.register(DSNoSketching)
        self.register(DSProbabilistic)
        self.register(DSSubspace)
        self.register(DS_STP)

lib_directsearch = LibDirectsearch()
OPTIMIZERS.register_lib(lib_directsearch)

# ---------------------------------------------------------------------------- #
#                                     NLopt                                    #
# ---------------------------------------------------------------------------- #
class LibNLopt(Lib):
    def __init__(self):
        super().__init__(
            names = ('nlopt', 'no'),
            requires = 'nlopt',
        )

    def initialize(self):
        from ...integrations.nlopt import ALL_ALGOS, NLoptWrapper
        for algo in ALL_ALGOS:
            if algo in {
                "GD_STOGO",
                "GD_STOGO_RAND",
                # "LD_LBFGS",
                # "LD_LBFGS_NOCEDAL",
                # "GN_AGS",
                # "LD_TNEWTON",
                # "LD_TNEWTON_PRECOND",
                # "LD_TNEWTON_RESTART",
                # "LD_TNEWTON_PRECOND_RESTART",
                # "LD_VAR1",
                # "LD_VAR2",
                # "LN_NEWUOA_BOUND",
                # "GN_CRS2_LM",
            }:
                groups = (GROUPS.BROKEN,)
            elif 'AUGLAG' in algo or 'MLSL' in algo: groups = (GROUPS.UNUSED, )
            else: groups = (GROUPS.MAIN, )
            self.register(NLoptWrapper.configured(algo).set_name(algo), groups = groups)
            # if 'MLSL' not in algo and 'DIRECT' not in algo:
            #     self.register(
            #         NLoptWrapper.configured('G_MLSL_LDS', local_algo=algo).set_name(f'MLSL_LDS.{algo}'), groups = groups
            #     )


lib_nlopt = LibNLopt()
OPTIMIZERS.register_lib(lib_nlopt)


# ---------------------------------------------------------------------------- #
#                                     lmfit                                    #
# ---------------------------------------------------------------------------- #
class LibLMfit(Lib):
    def __init__(self):
        super().__init__(
            names = ('lmfit', 'lf'),
            requires = 'lmfit',
        )

    def initialize(self):
        from ...integrations.lmfit import ALL_METHODS, LMFitOptimizer
        for method in ALL_METHODS:
            if method == 'ampgo': groups = GROUPS.MAIN
            else: groups = GROUPS.DUPLICATE
            self.register(LMFitOptimizer.configured(method=method).set_name(method), groups = groups) # type:ignore

lib_lmfit = LibLMfit()
OPTIMIZERS.register_lib(lib_lmfit)


# ---------------------------------------------------------------------------- #
#                                    pysors                                    #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                     CARS                                     #
# ---------------------------------------------------------------------------- #
class LibCARS(Lib):
    def __init__(self):
        super().__init__(
            names = ('cars',),
            requires = 'cars',
        )

    def initialize(self):
        import cars

        from ...integrations.cars import CARSOptimizer
        for cls in cars.CARS, cars.CARSCR, cars.CARSNQ, cars.Nesterov:
            self.register(CARSOptimizer.configured(cls).set_name(cls.__name__)) # type:ignore
            self.register(CARSOptimizer.conf_rescaled(scale = 0.1, optimizer=cls).set_name(f'{cls.__name__}-big'))
            self.register(CARSOptimizer.conf_rescaled(scale = 0.01, optimizer=cls).set_name(f'{cls.__name__}-large'))

lib_cars = LibCARS()
OPTIMIZERS.register_lib(lib_cars)

# ---------------------------------------------------------------------------- #
#                                     ZOOpt                                    #
# ---------------------------------------------------------------------------- #
class LibZOOpt(Lib):
    def __init__(self):
        super().__init__(
            names = ('zoopt',),
            requires = 'zoopt',
        )

    def initialize(self):
        from ...integrations.zoopt import ZOOptimizer
        self.register(ZOOptimizer)

lib_zoopt = LibZOOpt()
OPTIMIZERS.register_lib(lib_zoopt)


# ---------------------------------------------------------------------------- #
#                                     pygmo                                    #
# ---------------------------------------------------------------------------- #
class LibPygmo(Lib):
    def __init__(self):
        super().__init__(
            names = ('pygmo',),
            requires = 'pygmo',
        )

    def initialize(self):
        from ...integrations.pygmo import PygmoOptimizer, get_all_pygmo_algos
        for cls in get_all_pygmo_algos():
            for i in (1, 2, 5, 10, 20, 50, 100, 200, 500):
                if i == 20: groups = GROUPS.MAIN
                else: groups = GROUPS.EXTRA
                self.register(
                    PygmoOptimizer.configured(cls, popsize=i).set_name(f'{cls.__name__}-{i}'),
                    groups=groups,
                )

lib_pygmo = LibPygmo()
OPTIMIZERS.register_lib(lib_pygmo)


# ---------------------------------------------------------------------------- #
#                                     pymoo                                    #
# ---------------------------------------------------------------------------- #
class LibPymoo(Lib):
    def __init__(self):
        super().__init__(
            names = ('pymoo',),
            requires = 'pymoo',
        )

    def initialize(self):
        from ...integrations.pymoo import PymooOptimizer, get_all_pymoo_algos
        for cls in get_all_pymoo_algos():
            self.register(PymooOptimizer.configured(cls).set_name(cls.__name__))

lib_pymoo = LibPymoo()
OPTIMIZERS.register_lib(lib_pymoo)


# ---------------------------------------------------------------------------- #
#                                     PyXAB                                    #
# ---------------------------------------------------------------------------- #
class LibPyXAB(Lib):
    def __init__(self):
        super().__init__(
            names = ('PyXAB',),
            requires = 'PyXAB',
        )

    def initialize(self):
        from PyXAB.partition.BinaryPartition import BinaryPartition
        from PyXAB.partition.DimensionBinaryPartition import \
            DimensionBinaryPartition
        from PyXAB.partition.KaryPartition import KaryPartition
        from PyXAB.partition.RandomBinaryPartition import RandomBinaryPartition
        from PyXAB.partition.RandomKaryPartition import RandomKaryPartition

        from ...integrations.pyxab import PyXABOptimizer, get_all_pyxab_algos
        for cls in get_all_pyxab_algos():
            for partition in (BinaryPartition, RandomBinaryPartition, DimensionBinaryPartition, KaryPartition, RandomKaryPartition):
                self.register(PyXABOptimizer.configured(cls, partition=partition).set_name(f'{cls.__name__}-{partition.__name__}'))

lib_pyxab = LibPyXAB()
OPTIMIZERS.register_lib(lib_pyxab)


# ---------------------------------------------------------------------------- #
#                                     mads                                     #
# ---------------------------------------------------------------------------- #
class LibMads(Lib):
    def __init__(self):
        super().__init__(
            names = ('mads',),
            requires = 'mads',
        )

    def initialize(self):
        from ...integrations.mads import Orthomads
        self.register(Orthomads)

lib_mads = LibMads()
OPTIMIZERS.register_lib(lib_mads)